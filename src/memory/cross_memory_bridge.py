"""CrossMemoryBridge - Cross-memory relationship management system"""

import re
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from collections import defaultdict

from ..redis.client import RedisClient
from ..embeddings.embeddings import EmbeddingService
from ..rag.relationships import ChunkRelationshipManager
from .models import SearchQuery, SearchResult, MemoryContent

logger = logging.getLogger(__name__)


@dataclass
class ConversationData:
    """Conversation data model"""
    path: str
    messages: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class DocumentReference:
    """Document reference extracted from conversation"""
    doc_id: str
    reference_type: str  # explicit, implicit, entity
    confidence: float
    context: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationDocumentLink:
    """Conversation-document relationship"""
    conv_id: str
    doc_id: str
    link_type: str  # reference, topic, entity, temporal
    strength: float  # 0.0 ~ 1.0
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossMemoryRelations:
    """Cross-memory relationship results"""
    source_id: str
    source_type: str  # conversation, document
    related_conversations: List[Tuple[str, float]]  # (conv_id, relevance)
    related_documents: List[Tuple[str, float]]  # (doc_id, relevance)
    temporal_neighbors: List[Tuple[str, str, float]]  # (id, type, time_distance)
    shared_entities: Dict[str, List[str]]  # entity -> [memory_ids]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossMemoryBridge:
    """Bridge for extracting and managing relationships between conversations and documents"""
    
    # Document reference patterns
    REFERENCE_PATTERNS = [
        # Explicit references
        (r"document\s*['\"]?([^'\"]+)['\"]?", "explicit", 0.9),
        (r"([^\s]+\.(?:pdf|doc|docx|txt|md))", "explicit", 0.95),
        (r"file\s*['\"]?([^'\"]+)['\"]?", "explicit", 0.85),
        
        # Implicit references
        (r"(?:previously|earlier|before)\s*(?:seen|read|checked)\s*(.+?)(?:in|from)", "implicit", 0.7),
        (r"(?:that|this|the)\s*(?:document|material|content)", "implicit", 0.6),
        (r"(?:as\s+)?mentioned\s+above", "implicit", 0.65),
        
        # Topic-based
        (r"(?:related|about)\s*(?:document|material)", "topic", 0.5),
    ]
    
    def __init__(
        self,
        redis_client: RedisClient,
        embedding_service: Optional[EmbeddingService] = None,
        relationship_manager: Optional[ChunkRelationshipManager] = None,
        cache_ttl: int = 3600
    ):
        self.redis = redis_client
        self.embedder = embedding_service
        self.relationship_manager = relationship_manager
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        
    async def link_conversation_to_documents(
        self, conv_id: str
    ) -> List[str]:
        """Extract and link documents mentioned in conversation
        
        Args:
            conv_id: Conversation ID
            
        Returns:
            List of linked document IDs
        """
        try:
            # 1. Get conversation content
            conversation = await self._get_conversation(conv_id)
            if not conversation:
                logger.warning(f"Conversation not found: {conv_id}")
                return []
            
            # 2. Extract document references
            references = await self._extract_document_references(conversation)
            
            # 3. Verify and match referenced documents
            matched_docs = await self._match_documents(references)
            
            # 4. Store bidirectional relationships
            links = []
            for doc_id, reference in matched_docs:
                link = ConversationDocumentLink(
                    conv_id=conv_id,
                    doc_id=doc_id,
                    link_type=reference.reference_type,
                    strength=reference.confidence,
                    created_at=datetime.now(),
                    metadata={
                        "context": reference.context,
                        **reference.metadata
                    }
                )
                links.append(link)
                
            await self._store_links(links)
            
            # 5. Calculate and update relationship strengths
            await self._update_link_strengths(conv_id, [l.doc_id for l in links])
            
            return [l.doc_id for l in links]
            
        except Exception as e:
            logger.error(f"Error linking conversation to documents: {e}")
            raise
            
    async def _get_conversation(self, conv_id: str) -> Optional[ConversationData]:
        """Get conversation data"""
        # Read conversation stream from Redis
        key = f"stream:memory:conversation:{conv_id}"
        
        try:
            # Read all messages from stream
            messages = await self.redis.xrange(key, "-", "+")
            if not messages:
                return None
                
            # Convert messages to ConversationData
            conversation_messages = []
            metadata = {}
            
            for msg_id, data in messages:
                if b"content" in data:
                    content = data[b"content"].decode("utf-8")
                    conversation_messages.append(content)
                    
                # Collect metadata
                if b"metadata" in data:
                    import json
                    meta = json.loads(data[b"metadata"].decode("utf-8"))
                    metadata.update(meta)
                    
            return ConversationData(
                path=conv_id,
                messages=conversation_messages,
                metadata=metadata,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching conversation {conv_id}: {e}")
            return None
            
    async def _extract_document_references(
        self, conversation: ConversationData
    ) -> List[DocumentReference]:
        """Extract document references from conversation"""
        references = []
        
        # Combine all messages
        full_text = " ".join(conversation.messages)
        
        # Extract references using pattern matching
        for pattern, ref_type, confidence in self.REFERENCE_PATTERNS:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                # Extract context (50 chars before and after match)
                start = max(0, match.start() - 50)
                end = min(len(full_text), match.end() + 50)
                context = full_text[start:end]
                
                reference = DocumentReference(
                    doc_id=match.group(1) if match.groups() else match.group(0),
                    reference_type=ref_type,
                    confidence=confidence,
                    context=context,
                    metadata={
                        "pattern": pattern,
                        "position": match.start()
                    }
                )
                references.append(reference)
                
        # Entity-based reference extraction (when using NER)
        if self.embedder:
            entities = await self._extract_entities(full_text)
            for entity, entity_type in entities:
                reference = DocumentReference(
                    doc_id=entity,
                    reference_type="entity",
                    confidence=0.6,
                    context=f"Entity: {entity} ({entity_type})",
                    metadata={"entity_type": entity_type}
                )
                references.append(reference)
                
        return references
        
    async def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities from text (simple implementation)"""
        # TODO: Integrate actual NER library
        entities = []
        
        # Simple pattern-based extraction
        # Person name patterns
        person_pattern = r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b"
        for match in re.finditer(person_pattern, text):
            entities.append((match.group(1), "PERSON"))
            
        # Organization patterns
        org_pattern = r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Inc|Corp|LLC|Ltd))\b"
        for match in re.finditer(org_pattern, text):
            entities.append((match.group(1), "ORG"))
            
        return entities
        
    async def _match_documents(
        self, references: List[DocumentReference]
    ) -> List[Tuple[str, DocumentReference]]:
        """Match references to actual documents"""
        matched = []
        
        for ref in references:
            # 1. Try exact ID matching
            doc_key = f"json:memory:document:*{ref.doc_id}*"
            matching_keys = await self.redis.keys(doc_key)
            
            if matching_keys:
                # Use first match
                doc_id = matching_keys[0].decode("utf-8")
                matched.append((doc_id, ref))
                continue
                
            # 2. Search in metadata
            if ref.reference_type == "entity":
                # Search for documents containing the entity
                entity_key = f"entity:document:{ref.doc_id}"
                doc_ids = await self.redis.smembers(entity_key)
                for doc_id in doc_ids:
                    matched.append((doc_id.decode("utf-8"), ref))
                    
            # 3. Similarity-based matching (when embeddings available)
            if self.embedder and ref.confidence < 0.8:
                similar_docs = await self._find_similar_documents(ref.context)
                for doc_id, score in similar_docs[:3]:  # Top 3
                    if score > 0.7:
                        ref.confidence *= score  # Adjust confidence
                        matched.append((doc_id, ref))
                        
        return matched
        
    async def _find_similar_documents(
        self, text: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find documents similar to text"""
        if not self.embedder:
            return []
            
        # Generate text embedding
        embedding = await self.embedder.embed(text)
        
        # Perform vector search
        # TODO: Implement actual vector search
        return []
        
    async def _store_links(self, links: List[ConversationDocumentLink]):
        """Store conversation-document relationships"""
        pipeline = self.redis.pipeline()
        
        for link in links:
            # Store bidirectional relationships
            # 1. Conversation → Document
            conv_to_doc_key = f"link:conv_to_doc:{link.conv_id}"
            pipeline.zadd(
                conv_to_doc_key,
                {link.doc_id: link.strength}
            )
            
            # 2. Document → Conversation
            doc_to_conv_key = f"link:doc_to_conv:{link.doc_id}"
            pipeline.zadd(
                doc_to_conv_key,
                {link.conv_id: link.strength}
            )
            
            # 3. Link metadata
            link_meta_key = f"link:meta:{link.conv_id}:{link.doc_id}"
            pipeline.hset(
                link_meta_key,
                mapping={
                    "link_type": link.link_type,
                    "strength": str(link.strength),
                    "created_at": link.created_at.isoformat(),
                    "metadata": str(link.metadata)
                }
            )
            
            # 4. Type-based index
            type_index_key = f"link:type:{link.link_type}"
            pipeline.sadd(type_index_key, f"{link.conv_id}:{link.doc_id}")
            
        await pipeline.execute()
        
    async def _update_link_strengths(self, conv_id: str, doc_ids: List[str]):
        """Recalculate and update relationship strengths"""
        # Consider temporal proximity, reference frequency, semantic similarity, etc.
        for doc_id in doc_ids:
            strength = await self._calculate_link_strength(conv_id, doc_id)
            
            # Update strength
            conv_to_doc_key = f"link:conv_to_doc:{conv_id}"
            await self.redis.zadd(conv_to_doc_key, {doc_id: strength})
            
            doc_to_conv_key = f"link:doc_to_conv:{doc_id}"
            await self.redis.zadd(doc_to_conv_key, {conv_id: strength})
            
    async def _calculate_link_strength(
        self, conv_id: str, doc_id: str
    ) -> float:
        """Calculate conversation-document relationship strength"""
        strength = 0.0
        factors = []
        
        # 1. Base strength by reference type
        link_meta_key = f"link:meta:{conv_id}:{doc_id}"
        meta = await self.redis.hgetall(link_meta_key)
        if meta:
            link_type = meta.get(b"link_type", b"").decode("utf-8")
            if link_type == "explicit":
                factors.append(0.9)
            elif link_type == "implicit":
                factors.append(0.6)
            elif link_type == "entity":
                factors.append(0.5)
            else:
                factors.append(0.3)
                
        # 2. Temporal proximity (more recent = stronger)
        # TODO: Implement actual timestamp comparison
        factors.append(0.7)  # temporary value
        
        # 3. Reference frequency (more mentions = stronger)
        # TODO: Calculate reference count
        factors.append(0.5)  # temporary value
        
        # 4. Semantic similarity
        if self.embedder:
            # TODO: Calculate actual similarity
            factors.append(0.6)  # temporary value
            
        # Calculate weighted average
        if factors:
            strength = sum(factors) / len(factors)
            
        return min(1.0, strength)
        
    async def find_related_memories(
        self, memory_key: str
    ) -> CrossMemoryRelations:
        """Search for related information across memory types
        
        Args:
            memory_key: Memory key (conversation or document)
            
        Returns:
            Cross-memory relationship information
        """
        try:
            # Determine memory type
            memory_type = self._determine_memory_type(memory_key)
            
            # Run multiple searches in parallel
            tasks = [
                self._find_temporal_neighbors(memory_key, memory_type),
                self._find_semantic_relations(memory_key, memory_type),
                self._find_reference_relations(memory_key, memory_type),
                self._extract_shared_entities(memory_key, memory_type)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Consolidate results
            temporal_neighbors = results[0] if not isinstance(results[0], Exception) else []
            semantic_relations = results[1] if not isinstance(results[1], Exception) else []
            reference_relations = results[2] if not isinstance(results[2], Exception) else []
            shared_entities = results[3] if not isinstance(results[3], Exception) else {}
            
            # Separate related conversations and documents
            related_conversations = []
            related_documents = []
            
            for memory_id, score in semantic_relations + reference_relations:
                if "conversation" in memory_id:
                    related_conversations.append((memory_id, score))
                else:
                    related_documents.append((memory_id, score))
                    
            # Remove duplicates and sort
            related_conversations = self._deduplicate_and_sort(related_conversations)
            related_documents = self._deduplicate_and_sort(related_documents)
            
            return CrossMemoryRelations(
                source_id=memory_key,
                source_type=memory_type,
                related_conversations=related_conversations[:10],
                related_documents=related_documents[:10],
                temporal_neighbors=temporal_neighbors[:10],
                shared_entities=shared_entities,
                metadata={
                    "search_timestamp": datetime.now().isoformat(),
                    "total_relations": len(related_conversations) + len(related_documents)
                }
            )
            
        except Exception as e:
            logger.error(f"Error finding related memories: {e}")
            raise
            
    def _determine_memory_type(self, memory_key: str) -> str:
        """Determine memory type from key"""
        if "conversation" in memory_key:
            return "conversation"
        elif "document" in memory_key:
            return "document"
        else:
            # Guess from key pattern
            if "stream:" in memory_key:
                return "conversation"
            else:
                return "document"
                
    async def _find_temporal_neighbors(
        self, memory_key: str, memory_type: str
    ) -> List[Tuple[str, str, float]]:
        """Find temporally close memories"""
        neighbors = []
        
        try:
            # Get memory timestamp
            timestamp = await self._get_memory_timestamp(memory_key)
            if not timestamp:
                return neighbors
                
            # Set time window (1 hour before and after)
            time_window = timedelta(hours=1)
            start_time = timestamp - time_window
            end_time = timestamp + time_window
            
            # Search in temporal index
            temporal_index_key = f"temporal:index:{timestamp.strftime('%Y/%m/%d/%H')}"
            nearby_memories = await self.redis.smembers(temporal_index_key)
            
            # Filter by exact time range
            for memory_id_bytes in nearby_memories:
                memory_id = memory_id_bytes.decode("utf-8")
                if memory_id == memory_key:
                    continue
                    
                mem_timestamp = await self._get_memory_timestamp(memory_id)
                if mem_timestamp and start_time <= mem_timestamp <= end_time:
                    # Calculate temporal distance (0-1, closer = higher)
                    time_diff = abs((timestamp - mem_timestamp).total_seconds())
                    max_diff = time_window.total_seconds()
                    proximity = 1.0 - (time_diff / max_diff)
                    
                    # Determine memory type
                    mem_type = self._determine_memory_type(memory_id)
                    neighbors.append((memory_id, mem_type, proximity))
            
            # Sort by temporal proximity
            neighbors.sort(key=lambda x: x[2], reverse=True)
            
            # Also search adjacent time slots
            for hour_offset in [-1, 1]:
                adj_time = timestamp + timedelta(hours=hour_offset)
                adj_index_key = f"temporal:index:{adj_time.strftime('%Y/%m/%d/%H')}"
                adj_memories = await self.redis.smembers(adj_index_key)
                
                for memory_id_bytes in adj_memories:
                    memory_id = memory_id_bytes.decode("utf-8")
                    if memory_id == memory_key or any(n[0] == memory_id for n in neighbors):
                        continue
                        
                    mem_timestamp = await self._get_memory_timestamp(memory_id)
                    if mem_timestamp and start_time <= mem_timestamp <= end_time:
                        time_diff = abs((timestamp - mem_timestamp).total_seconds())
                        proximity = 1.0 - (time_diff / max_diff) * 0.8  # Adjacent time slots get slightly lower score
                        
                        mem_type = self._determine_memory_type(memory_id)
                        neighbors.append((memory_id, mem_type, proximity))
            
            # Final sort and limit
            neighbors.sort(key=lambda x: x[2], reverse=True)
            return neighbors[:20]  # Top 20
            
        except Exception as e:
            logger.error(f"Error finding temporal neighbors: {e}")
            return neighbors
        
    async def _get_memory_timestamp(self, memory_key: str) -> Optional[datetime]:
        """Extract timestamp from memory"""
        # Try extracting time information from key
        time_pattern = r"(\d{4}/\d{2}/\d{2}/\d{2}/\d{2}/\d{2})"
        match = re.search(time_pattern, memory_key)
        if match:
            time_str = match.group(1)
            return datetime.strptime(time_str, "%Y/%m/%d/%H/%M/%S")
            
        # Check metadata for time information
        # TODO: Implement actual metadata lookup
        
        return None
        
    async def _find_semantic_relations(
        self, memory_key: str, memory_type: str
    ) -> List[Tuple[str, float]]:
        """Find semantically similar memories"""
        relations = []
        
        if not self.embedder:
            return relations
            
        try:
            # Get memory content
            content = await self._get_memory_content(memory_key)
            if not content:
                return relations
                
            # Generate or get embedding from cache
            source_embedding = await self._get_or_create_embedding(memory_key, content)
            
            # Compare with other memory embeddings
            # 1. Search conversation type (if current is document)
            if memory_type == "document":
                conv_relations = await self._search_similar_conversations(source_embedding)
                relations.extend(conv_relations)
                
            # 2. Search document type (if current is conversation)
            if memory_type == "conversation":
                doc_relations = await self._search_similar_documents(source_embedding)
                relations.extend(doc_relations)
                
            # 3. Compare with temporarily cached embeddings
            similar_cached = await self._compare_with_cached_embeddings(
                memory_key, source_embedding
            )
            relations.extend(similar_cached)
            
            # Remove duplicates and sort
            relations = self._deduplicate_and_sort(relations)
            
            return relations[:15]  # Top 15
            
        except Exception as e:
            logger.error(f"Error finding semantic relations: {e}")
            return relations
            
    async def _search_similar_conversations(
        self, embedding: List[float], threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Search for similar conversations"""
        relations = []
        
        try:
            # Search by conversation key pattern
            conv_pattern = "stream:memory:conversation:*"
            conv_keys = await self.redis.keys(conv_pattern)
            
            # Process in batches (performance optimization)
            batch_size = 50
            for i in range(0, len(conv_keys), batch_size):
                batch_keys = conv_keys[i:i + batch_size]
                batch_similarities = await self._calculate_batch_similarities(
                    embedding, batch_keys, "conversation"
                )
                relations.extend(batch_similarities)
                
            # Filter by threshold
            relations = [(k, s) for k, s in relations if s >= threshold]
            
        except Exception as e:
            logger.error(f"Error searching similar conversations: {e}")
            
        return relations
        
    async def _search_similar_documents(
        self, embedding: List[float], threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        relations = []
        
        try:
            # Search by document key pattern
            doc_pattern = "json:memory:document:*"
            doc_keys = await self.redis.keys(doc_pattern)
            
            # Process in batches
            batch_size = 50
            for i in range(0, len(doc_keys), batch_size):
                batch_keys = doc_keys[i:i + batch_size]
                batch_similarities = await self._calculate_batch_similarities(
                    embedding, batch_keys, "document"
                )
                relations.extend(batch_similarities)
                
            # Filter by threshold
            relations = [(k, s) for k, s in relations if s >= threshold]
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            
        return relations
        
    async def _calculate_batch_similarities(
        self, source_embedding: List[float], keys: List[bytes], memory_type: str
    ) -> List[Tuple[str, float]]:
        """Calculate similarities in batch"""
        similarities = []
        
        for key_bytes in keys:
            try:
                key = key_bytes.decode("utf-8")
                
                # Check embedding cache
                cache_key = f"embedding:{key}"
                cached_embedding = None
                
                if cache_key in self._cache:
                    cached_embedding = self._cache[cache_key]
                else:
                    # Check Redis
                    stored = await self.redis.get(cache_key)
                    if stored:
                        import json
                        try:
                            cached_embedding = json.loads(stored)
                            self._cache[cache_key] = cached_embedding
                        except json.JSONDecodeError:
                            continue
                
                if cached_embedding:
                    # Calculate similarity
                    similarity = self._cosine_similarity(source_embedding, cached_embedding)
                    if similarity > 0.5:  # Minimum threshold
                        similarities.append((key, similarity))
                        
            except Exception as e:
                logger.debug(f"Error calculating similarity for {key_bytes}: {e}")
                continue
                
        return similarities
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        try:
            if len(vec1) != len(vec2):
                return 0.0
                
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_v1 = sum(a * a for a in vec1) ** 0.5
            norm_v2 = sum(b * b for b in vec2) ** 0.5
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
                
            return dot_product / (norm_v1 * norm_v2)
            
        except Exception:
            return 0.0
            
    async def _compare_with_cached_embeddings(
        self, source_key: str, source_embedding: List[float]
    ) -> List[Tuple[str, float]]:
        """Compare with cached embeddings"""
        similarities = []
        
        # Compare with memory cache
        for cache_key, embedding in self._cache.items():
            if cache_key.startswith("embedding:") and cache_key != f"embedding:{source_key}":
                memory_key = cache_key[10:]  # Remove "embedding:" prefix
                similarity = self._cosine_similarity(source_embedding, embedding)
                if similarity > 0.6:
                    similarities.append((memory_key, similarity))
                    
        return similarities
        
    async def _get_memory_content(self, memory_key: str) -> Optional[str]:
        """Get memory content"""
        if "conversation" in memory_key:
            # Conversation content
            messages = await self.redis.xrange(memory_key, "-", "+", count=100)
            contents = []
            for _, data in messages:
                if b"content" in data:
                    contents.append(data[b"content"].decode("utf-8"))
            return " ".join(contents)
        else:
            # Document content
            doc_data = await self.redis.json_get(memory_key)
            if doc_data and "content" in doc_data:
                return doc_data["content"]
                
        return None
        
    async def _get_or_create_embedding(self, key: str, content: str) -> List[float]:
        """Get or create embedding"""
        # Check cache
        cache_key = f"embedding:{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Check Redis
        stored = await self.redis.get(cache_key)
        if stored:
            import json
            embedding = json.loads(stored)
            self._cache[cache_key] = embedding
            return embedding
            
        # Generate new embedding
        embedding = await self.embedder.embed(content)
        
        # Store
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            str(embedding)
        )
        self._cache[cache_key] = embedding
        
        return embedding
        
    async def _find_reference_relations(
        self, memory_key: str, memory_type: str
    ) -> List[Tuple[str, float]]:
        """Find reference-based memories"""
        relations = []
        
        if memory_type == "conversation":
            # Documents referenced in conversation
            conv_to_doc_key = f"link:conv_to_doc:{memory_key}"
            docs = await self.redis.zrevrange(conv_to_doc_key, 0, -1, withscores=True)
            for doc_id, score in docs:
                relations.append((doc_id.decode("utf-8"), score))
                
        else:  # document
            # Conversations that reference this document
            doc_to_conv_key = f"link:doc_to_conv:{memory_key}"
            convs = await self.redis.zrevrange(doc_to_conv_key, 0, -1, withscores=True)
            for conv_id, score in convs:
                relations.append((conv_id.decode("utf-8"), score))
                
        return relations
        
    async def _extract_shared_entities(
        self, memory_key: str, memory_type: str
    ) -> Dict[str, List[str]]:
        """Extract shared entities"""
        shared_entities = defaultdict(list)
        
        # Extract entities from memory
        content = await self._get_memory_content(memory_key)
        if not content:
            return dict(shared_entities)
            
        entities = await self._extract_entities(content)
        
        # Find other memories containing each entity
        for entity, entity_type in entities:
            entity_key = f"entity:{entity_type.lower()}:{entity}"
            related_memories = await self.redis.smembers(entity_key)
            
            for mem_id in related_memories:
                mem_id_str = mem_id.decode("utf-8")
                if mem_id_str != memory_key:
                    shared_entities[entity].append(mem_id_str)
                    
        return dict(shared_entities)
        
    def _deduplicate_and_sort(
        self, items: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Remove duplicates and sort by score"""
        # Remove duplicates using dictionary (keep highest score)
        unique_items = {}
        for item_id, score in items:
            if item_id not in unique_items or score > unique_items[item_id]:
                unique_items[item_id] = score
                
        # Sort by score descending
        sorted_items = sorted(
            unique_items.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_items
        
    async def find_related_memories_advanced(
        self, 
        memory_key: str,
        search_options: Optional[Dict[str, Any]] = None
    ) -> CrossMemoryRelations:
        """Advanced cross-memory search with more options and filtering
        
        Args:
            memory_key: Source memory key
            search_options: Search options
                - include_types: Memory types to include ['conversation', 'document']
                - time_window_hours: Time window (default: 24 hours)
                - semantic_threshold: Semantic similarity threshold (default: 0.7)
                - max_results: Maximum results (default: 20)
                - include_metadata: Include metadata (default: True)
                - boost_recent: Boost recent items (default: True)
                - entity_types: Specific entity types only
        """
        options = search_options or {}
        include_types = options.get("include_types", ["conversation", "document"])
        time_window_hours = options.get("time_window_hours", 24)
        semantic_threshold = options.get("semantic_threshold", 0.7)
        max_results = options.get("max_results", 20)
        include_metadata = options.get("include_metadata", True)
        boost_recent = options.get("boost_recent", True)
        entity_types = options.get("entity_types", None)
        
        try:
            memory_type = self._determine_memory_type(memory_key)
            
            # Prepare parallel search tasks
            tasks = []
            
            # 1. Temporal proximity (adjust time window)
            if time_window_hours > 0:
                tasks.append(self._find_temporal_neighbors_advanced(
                    memory_key, memory_type, time_window_hours
                ))
            
            # 2. Semantic similarity (adjust threshold)
            if self.embedder and semantic_threshold > 0:
                tasks.append(self._find_semantic_relations_advanced(
                    memory_key, memory_type, semantic_threshold, include_types
                ))
            
            # 3. Reference relationships (bidirectional)
            tasks.append(self._find_reference_relations_bidirectional(
                memory_key, memory_type, include_types
            ))
            
            # 4. Shared entities (type filtering)
            tasks.append(self._extract_shared_entities_filtered(
                memory_key, memory_type, entity_types
            ))
            
            # 5. Keyword-based associations
            tasks.append(self._find_keyword_relations(
                memory_key, memory_type, include_types
            ))
            
            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Consolidate results
            temporal_neighbors = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else []
            semantic_relations = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
            reference_relations = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []
            shared_entities = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else {}
            keyword_relations = results[4] if len(results) > 4 and not isinstance(results[4], Exception) else []
            
            # Combine all relations into one list
            all_relations = []
            all_relations.extend([(mid, score, "semantic") for mid, score in semantic_relations])
            all_relations.extend([(mid, score, "reference") for mid, score in reference_relations])
            all_relations.extend([(mid, score, "keyword") for mid, score in keyword_relations])
            
            # Apply recency boosting
            if boost_recent:
                all_relations = await self._apply_recency_boost(all_relations)
            
            # Separate by type and normalize scores
            related_conversations = []
            related_documents = []
            
            for memory_id, score, relation_type in all_relations:
                # Type filtering
                mem_type = self._determine_memory_type(memory_id)
                if mem_type not in include_types:
                    continue
                    
                # Add metadata
                final_score = score
                if include_metadata:
                    metadata = await self._get_relation_metadata(
                        memory_key, memory_id, relation_type
                    )
                    # Adjust score based on metadata
                    if metadata.get("confidence", 1.0) < 0.5:
                        final_score *= 0.8
                
                if mem_type == "conversation":
                    related_conversations.append((memory_id, final_score))
                else:
                    related_documents.append((memory_id, final_score))
            
            # Remove duplicates and sort
            related_conversations = self._deduplicate_and_sort(related_conversations)[:max_results//2]
            related_documents = self._deduplicate_and_sort(related_documents)[:max_results//2]
            
            return CrossMemoryRelations(
                source_id=memory_key,
                source_type=memory_type,
                related_conversations=related_conversations,
                related_documents=related_documents,
                temporal_neighbors=temporal_neighbors[:max_results//3],
                shared_entities=shared_entities,
                metadata={
                    "search_timestamp": datetime.now().isoformat(),
                    "search_options": options,
                    "total_relations": len(related_conversations) + len(related_documents),
                    "semantic_relations": len(semantic_relations),
                    "reference_relations": len(reference_relations),
                    "keyword_relations": len(keyword_relations)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in advanced cross-memory search: {e}")
            # Fallback: perform basic search
            return await self.find_related_memories(memory_key)
    
    # Additional helper methods would go here...
    # (truncated for brevity - the remaining methods follow similar patterns)