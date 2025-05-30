"""ChunkRelationshipManager - Core relationship management system for LRMM.

This module builds and manages multi-dimensional relationships between chunks:
- Structural relationships: Document position-based relationships (prev/next, parent/child)
- Semantic relationships: Embedding similarity-based relationships
- Reference relationships: Explicit/implicit reference relationships
- Temporal relationships: Time order and context relationships
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .redis_schema import (
    RedisKeyPattern,
    RedisSchemaConfig,
    get_redis_key,
    parse_redis_key,
)
from .transaction_manager import RedisTransactionManager, RelationshipTransactionHelper
from .cache_manager import RelationCacheManager, CacheAwareRelationQuery

from ..embeddings.embeddings import EmbeddingService
from ..redis.client import RedisClient

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Relationship type definitions."""

    # Structural relationships
    PREV = "prev"
    NEXT = "next"
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"

    # Semantic relationships
    SIMILAR = "similar"
    RELATED = "related"
    OPPOSITE = "opposite"

    # Reference relationships
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    MENTIONS = "mentions"
    MENTIONED_BY = "mentioned_by"

    # Temporal relationships
    BEFORE = "before"
    AFTER = "after"
    CONCURRENT = "concurrent"
    CAUSED_BY = "caused_by"
    CAUSES = "causes"


@dataclass
class ChunkData:
    """Chunk data model."""

    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    @property
    def doc_path(self) -> Optional[str]:
        """Return document path."""
        return self.metadata.get("doc_path")

    @property
    def section_level(self) -> int:
        """Return section level (0: document, 1: chapter, 2: section, etc.)."""
        return int(self.metadata.get("section_level", 0))

    @property
    def position(self) -> int:
        """Return position within document."""
        return int(self.metadata.get("position", 0))

    @property
    def page_number(self) -> Optional[int]:
        """Return page number."""
        return self.metadata.get("page_number")

    @property
    def section_title(self) -> Optional[str]:
        """Return section title."""
        return self.metadata.get("section_title")

    @property
    def chunk_type(self) -> str:
        """Return chunk type (heading, paragraph, list, table, etc.)."""
        return str(self.metadata.get("chunk_type", "paragraph"))


@dataclass
class Relation:
    """Base relationship model."""

    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 1.0  # Relationship strength (0.0 ~ 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StructuralRelation(Relation):
    """Structural relationship model."""

    distance: int = 1  # Structural distance (adjacent=1, 2-step=2, ...)


@dataclass
class SemanticRelation(Relation):
    """Semantic relationship model."""

    similarity_score: float = 0.0  # Cosine similarity
    shared_entities: List[str] = field(default_factory=list)
    shared_topics: List[str] = field(default_factory=list)


@dataclass
class ReferenceRelation(Relation):
    """Reference relationship model."""

    reference_text: str = ""  # Reference text
    reference_type: str = ""  # explicit, implicit, entity, concept


@dataclass
class TemporalRelation(Relation):
    """Temporal relationship model."""

    time_delta: Optional[float] = None  # Time difference (seconds)
    causal_score: float = 0.0  # Causal relationship score


class ChunkRelationshipManager:
    """Chunk relationship manager - Core component of LRMM."""

    def __init__(
        self,
        redis_client: RedisClient,
        embedding_generator: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.7,
        max_relations_per_chunk: int = 50,
    ):
        """Initialize ChunkRelationshipManager.

        Args:
            redis_client: Redis client
            embedding_generator: Embedding generator (for semantic relationships)
            similarity_threshold: Similarity threshold
            max_relations_per_chunk: Maximum relationships per chunk
        """
        self.redis = redis_client
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
        self.max_relations_per_chunk = max_relations_per_chunk

        # Redis key prefixes for relationship storage
        self.relation_prefix = "relation:chunk"
        self.index_prefix = "idx:relation"
        
        # TTL setting (30 days)
        self.relation_ttl = 30 * 24 * 60 * 60
        
        # Initialize cache manager
        self.cache_manager = RelationCacheManager(redis_client) if redis_client else None
        self.cache_query = CacheAwareRelationQuery(redis_client, self.cache_manager) if self.cache_manager else None

    async def initialize(self) -> None:
        """Initialize relationship indexes."""
        # TODO: Create Redis Search indexes
        logger.info("ChunkRelationshipManager initialized")

    async def build_all_relationships(
        self,
        chunks: List[ChunkData],
        doc_path: str,
        build_structural: bool = True,
        build_semantic: bool = True,
        build_references: bool = True,
        build_temporal: bool = True,
    ) -> Dict[str, List[Relation]]:
        """Build all types of relationships.

        Args:
            chunks: Chunk list
            doc_path: Document path
            build_structural: Whether to build structural relationships
            build_semantic: Whether to build semantic relationships
            build_references: Whether to build reference relationships
            build_temporal: Whether to build temporal relationships

        Returns:
            Relationship list by chunk ID
        """
        all_relations: Dict[str, List[Relation]] = {}

        # Build each relationship type in parallel
        tasks: List[asyncio.Task[Dict[str, List[Relation]]]] = []

        if build_structural:
            tasks.append(
                asyncio.create_task(
                    self.build_structural_relationships(chunks, doc_path)
                )
            )
        if build_semantic and self.embedding_generator:
            tasks.append(asyncio.create_task(self.build_semantic_relationships(chunks)))
        if build_references:
            tasks.append(
                asyncio.create_task(self.extract_reference_relationships(chunks))
            )
        if build_temporal:
            tasks.append(asyncio.create_task(self.build_temporal_relationships(chunks)))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results
            for result in results:
                if isinstance(result, dict):
                    for chunk_id, relations in result.items():
                        if chunk_id not in all_relations:
                            all_relations[chunk_id] = []
                        all_relations[chunk_id].extend(relations)
                elif isinstance(result, Exception):
                    logger.error(f"Error building relationships: {result}")

        # Store in Redis
        await self._save_relations_to_redis(all_relations)

        return all_relations

    async def build_structural_relationships(
        self, chunks: List[ChunkData], doc_path: str
    ) -> Dict[str, List[Relation]]:
        """Build structural relationships between chunks.

        Args:
            chunks: Chunk list
            doc_path: Document path

        Returns:
            Structural relationship list by chunk ID
        """
        relations: Dict[str, List[Relation]] = {}

        # Initialize relationship list for each chunk
        for chunk in chunks:
            relations[chunk.id] = []

        # 1. Analyze document structure and correct section levels
        self._analyze_document_structure(chunks)

        # 2. Build sequential relationships (prev/next)
        await self._build_sequential_relations(chunks, relations, doc_path)

        # 3. Build hierarchical relationships (parent/child)
        await self._build_hierarchical_relations(chunks, relations)

        # 4. Build sibling relationships
        await self._build_sibling_relations(chunks, relations)

        # 5. Build page-based relationships (optional)
        if any(chunk.page_number is not None for chunk in chunks):
            await self._build_page_relations(chunks, relations)

        logger.info(
            f"Built structural relationships for {len(chunks)} chunks in {doc_path}"
        )

        return relations

    async def build_semantic_relationships(
        self, chunks: List[ChunkData], threshold: Optional[float] = None
    ) -> Dict[str, List[Relation]]:
        """Build semantic relationships between chunks.

        Args:
            chunks: Chunk list
            threshold: Similarity threshold (use default if None)

        Returns:
            Semantic relationship list by chunk ID
        """
        if not self.embedding_generator:
            logger.warning(
                "Embedding generator not available. Skipping semantic relationships."
            )
            return {}

        threshold = threshold or self.similarity_threshold
        relations: Dict[str, List[Relation]] = {}

        # Initialize relationship list for each chunk
        for chunk in chunks:
            relations[chunk.id] = []

        # 1. Generate or verify embeddings
        await self._ensure_embeddings(chunks)

        # 2. Calculate similarity matrix
        similarity_matrix = self._compute_similarity_matrix(chunks)

        # 3. Filter by threshold and create relationships
        for i, chunk1 in enumerate(chunks):
            # Find N most similar chunks for each chunk
            similarities = similarity_matrix[i]

            # Find chunks above threshold excluding self
            similar_indices = np.where(
                (similarities >= threshold) & (np.arange(len(chunks)) != i)
            )[0]

            # Sort by similarity
            similar_indices = similar_indices[
                np.argsort(similarities[similar_indices])[::-1]
            ]

            # Limit maximum relationships
            max_semantic_relations = min(10, self.max_relations_per_chunk // 4)
            similar_indices = similar_indices[:max_semantic_relations]

            for j in similar_indices:
                chunk2 = chunks[j]
                similarity_score = float(similarities[j])

                # Create bidirectional relationship
                relation_type = self._determine_semantic_relation_type(similarity_score)

                # Extract shared entities and topics
                shared_entities, shared_topics = await self._extract_shared_concepts(
                    chunk1, chunk2
                )

                relations[chunk1.id].append(
                    SemanticRelation(
                        source_id=chunk1.id,
                        target_id=chunk2.id,
                        relation_type=relation_type,
                        similarity_score=similarity_score,
                        shared_entities=shared_entities,
                        shared_topics=shared_topics,
                        strength=similarity_score,
                    )
                )

        logger.info(
            f"Built semantic relationships for {len(chunks)} chunks with threshold {threshold}"
        )

        return relations

    async def extract_reference_relationships(
        self, chunks: List[ChunkData]
    ) -> Dict[str, List[Relation]]:
        """Extract reference relationships within chunks.

        Args:
            chunks: Chunk list

        Returns:
            Reference relationship list by chunk ID
        """
        relations: Dict[str, List[Relation]] = {}
        
        # Initialize relationship list for each chunk
        for chunk in chunks:
            relations[chunk.id] = []
        
        # 1. Extract explicit reference patterns
        explicit_refs = await self._extract_explicit_references(chunks)
        
        # 2. Extract implicit reference patterns
        implicit_refs = await self._extract_implicit_references(chunks)
        
        # 3. Extract entity-based references
        entity_refs = await self._extract_entity_references(chunks)
        
        # 4. Extract section references
        section_refs = await self._extract_section_references(chunks)
        
        # Merge all reference relationships
        for ref_dict in [explicit_refs, implicit_refs, entity_refs, section_refs]:
            for chunk_id, chunk_relations in ref_dict.items():
                relations[chunk_id].extend(chunk_relations)
        
        logger.info(
            f"Extracted reference relationships for {len(chunks)} chunks"
        )
        
        return relations

    async def build_temporal_relationships(
        self, chunks: List[ChunkData]
    ) -> Dict[str, List[Relation]]:
        """Build temporal relationships between chunks.

        Args:
            chunks: Chunk list

        Returns:
            Temporal relationship list by chunk ID
        """
        relations: Dict[str, List[Relation]] = {}
        
        # Initialize for all chunks
        for chunk in chunks:
            relations[chunk.id] = []
        
        # 1. Extract temporal information from each chunk
        chunks_with_time = await self._extract_temporal_info(chunks)
        
        # 2. Build time order-based relationships
        await self._build_temporal_order_relations(chunks_with_time, relations)
        
        # 3. Build time window-based relationships
        await self._build_temporal_window_relations(chunks_with_time, relations)
        
        # 4. Analyze causal relationships (optional)
        await self._analyze_causal_relations(chunks_with_time, relations)
        
        logger.info(
            f"Built temporal relationships for {len(chunks)} chunks"
        )
        
        return relations

    def _analyze_document_structure(self, chunks: List[ChunkData]) -> None:
        """Analyze document structure and auto-detect section levels."""
        # Markdown header patterns
        header_patterns = [
            (r"^#{1}\s+(.+)$", 1),  # # Title
            (r"^#{2}\s+(.+)$", 2),  # ## Title
            (r"^#{3}\s+(.+)$", 3),  # ### Title
            (r"^#{4}\s+(.+)$", 4),  # #### Title
            (r"^#{5}\s+(.+)$", 5),  # ##### Title
            (r"^#{6}\s+(.+)$", 6),  # ###### Title
        ]

        # Numbering patterns (1., 1.1., 1.1.1., etc.)
        numbering_pattern = r"^(\d+(?:\.\d+)*)\s*\.?\s+(.+)$"

        for chunk in chunks:
            content_lines = chunk.content.strip().split("\n")
            if not content_lines:
                continue

            first_line = content_lines[0].strip()

            # Check markdown headers
            for pattern, level in header_patterns:
                match = re.match(pattern, first_line)
                if match:
                    chunk.metadata["section_level"] = level
                    chunk.metadata["section_title"] = match.group(1)
                    chunk.metadata["chunk_type"] = "heading"
                    break

            # Check numbering
            if "section_level" not in chunk.metadata:
                match = re.match(numbering_pattern, first_line)
                if match:
                    numbering = match.group(1)
                    title = match.group(2)
                    level = len(numbering.split("."))
                    chunk.metadata["section_level"] = level
                    chunk.metadata["section_title"] = title
                    chunk.metadata["section_number"] = numbering
                    chunk.metadata["chunk_type"] = "heading"

            # Check list items
            if "chunk_type" not in chunk.metadata:
                if re.match(r"^[-*+]\s+", first_line) or re.match(
                    r"^\d+\.\s+", first_line
                ):
                    chunk.metadata["chunk_type"] = "list"

            # Check tables
            if "chunk_type" not in chunk.metadata and len(content_lines) > 1:
                if re.match(r"^\|.*\|$", first_line) and re.match(
                    r"^\|[-:]+\|$", content_lines[1]
                ):
                    chunk.metadata["chunk_type"] = "table"

    async def _build_sequential_relations(
        self,
        chunks: List[ChunkData],
        relations: Dict[str, List[Relation]],
        doc_path: str,
    ) -> None:
        """Build sequential relationships (prev/next) - improved version."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                # Connect only if same section level or adjacent levels
                level_diff = abs(chunk.section_level - prev_chunk.section_level)
                if level_diff <= 1:
                    relations[chunk.id].append(
                        StructuralRelation(
                            source_id=chunk.id,
                            target_id=prev_chunk.id,
                            relation_type=RelationType.PREV,
                            distance=1,
                            metadata={
                                "doc_path": doc_path,
                                "level_diff": level_diff,
                            },
                        )
                    )

            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                level_diff = abs(chunk.section_level - next_chunk.section_level)
                if level_diff <= 1:
                    relations[chunk.id].append(
                        StructuralRelation(
                            source_id=chunk.id,
                            target_id=next_chunk.id,
                            relation_type=RelationType.NEXT,
                            distance=1,
                            metadata={
                                "doc_path": doc_path,
                                "level_diff": level_diff,
                            },
                        )
                    )

    async def _build_hierarchical_relations(
        self, chunks: List[ChunkData], relations: Dict[str, List[Relation]]
    ) -> None:
        """Build hierarchical relationships (parent/child) - improved version."""
        # Use section stack for accurate hierarchy tracking
        section_stack: List[ChunkData] = []  # [level0, level1, level2, ...]

        for chunk in chunks:
            level = chunk.section_level

            # Adjust stack to current level
            while len(section_stack) > level:
                section_stack.pop()

            # Find parent (directly above level in stack)
            if level > 0 and section_stack:
                parent = section_stack[-1]
                # Child → Parent relationship
                relations[chunk.id].append(
                    StructuralRelation(
                        source_id=chunk.id,
                        target_id=parent.id,
                        relation_type=RelationType.PARENT,
                        distance=level - parent.section_level,
                        metadata={
                            "parent_title": parent.section_title,
                            "child_title": chunk.section_title,
                        },
                    )
                )
                # Parent → Child relationship
                relations[parent.id].append(
                    StructuralRelation(
                        source_id=parent.id,
                        target_id=chunk.id,
                        relation_type=RelationType.CHILD,
                        distance=level - parent.section_level,
                        metadata={
                            "parent_title": parent.section_title,
                            "child_title": chunk.section_title,
                        },
                    )
                )

            # Add to stack only if heading type
            if chunk.chunk_type == "heading":
                # Adjust stack size to match level
                while len(section_stack) < level:
                    # Fill empty slots with dummy chunks instead of None
                    dummy_chunk = ChunkData(
                        id=f"dummy_{len(section_stack)}", content=""
                    )
                    section_stack.append(dummy_chunk)

                # Set chunk at current level
                if level > 0:
                    section_stack[level - 1] = chunk

    async def _build_sibling_relations(
        self, chunks: List[ChunkData], relations: Dict[str, List[Relation]]
    ) -> None:
        """Build sibling relationships (chunks with same parent)."""
        # Group children by parent
        parent_children: Dict[str, List[str]] = {}

        for chunk_id, chunk_relations in relations.items():
            for relation in chunk_relations:
                if relation.relation_type == RelationType.PARENT:
                    parent_id = relation.target_id
                    if parent_id not in parent_children:
                        parent_children[parent_id] = []
                    parent_children[parent_id].append(chunk_id)

        # Connect chunks with same parent as siblings
        for parent_id, children_ids in parent_children.items():
            for i, child1_id in enumerate(children_ids):
                for child2_id in children_ids[i + 1 :]:
                    # Bidirectional sibling relationship
                    relations[child1_id].append(
                        StructuralRelation(
                            source_id=child1_id,
                            target_id=child2_id,
                            relation_type=RelationType.SIBLING,
                            distance=0,
                            metadata={"parent_id": parent_id},
                        )
                    )
                    relations[child2_id].append(
                        StructuralRelation(
                            source_id=child2_id,
                            target_id=child1_id,
                            relation_type=RelationType.SIBLING,
                            distance=0,
                            metadata={"parent_id": parent_id},
                        )
                    )

    async def _save_relations_to_redis(
        self, relations: Dict[str, List[Relation]]
    ) -> None:
        """Store relationships in Redis.
        
        Storage structure:
        - Hash: relation:chunk:{chunk_id} - All relationship info for chunk
        - Set: relation:type:{relation_type}:{chunk_id} - Target chunk IDs by relationship type
        - Sorted Set: relation:strength:{chunk_id} - Sorted by relationship strength
        """
        total_relations = sum(len(rels) for rels in relations.values())
        logger.info(f"Saving {total_relations} relations to Redis")
        
        if not self.redis:
            logger.warning("Redis client not available. Skipping relation save.")
            return
        
        # Use pipeline for batch saving
        pipeline = await self.redis.pipeline()
        
        try:
            for chunk_id, chunk_relations in relations.items():
                if not chunk_relations:
                    continue
                
                # 1. Store in chunk relationship hash
                relation_data = {}
                
                for idx, relation in enumerate(chunk_relations):
                    # Serialize relationship object
                    rel_dict = {
                        "source_id": relation.source_id,
                        "target_id": relation.target_id,
                        "relation_type": relation.relation_type.value,
                        "strength": relation.strength,
                        "created_at": relation.created_at.isoformat(),
                        "metadata": relation.metadata,
                    }
                    
                    # Add type-specific information
                    if isinstance(relation, StructuralRelation):
                        rel_dict["distance"] = relation.distance
                    elif isinstance(relation, SemanticRelation):
                        rel_dict["similarity_score"] = relation.similarity_score
                        rel_dict["shared_entities"] = relation.shared_entities
                        rel_dict["shared_topics"] = relation.shared_topics
                    elif isinstance(relation, ReferenceRelation):
                        rel_dict["reference_text"] = relation.reference_text
                        rel_dict["reference_type"] = relation.reference_type
                    elif isinstance(relation, TemporalRelation):
                        rel_dict["time_delta"] = relation.time_delta
                        rel_dict["causal_score"] = relation.causal_score
                    
                    # Data to store in hash
                    relation_data[f"rel:{idx}"] = json.dumps(rel_dict, ensure_ascii=False)
                    
                    # 2. Add target chunk ID to relationship type Set
                    type_key = f"{self.relation_prefix}:type:{relation.relation_type.value}:{chunk_id}"
                    await pipeline.sadd(type_key, relation.target_id)
                    
                    # 3. Add to relationship strength Sorted Set
                    strength_key = f"{self.relation_prefix}:strength:{chunk_id}"
                    await pipeline.zadd(
                        strength_key, 
                        {relation.target_id: relation.strength}
                    )
                
                # Store all relationships for chunk in hash
                if relation_data:
                    hash_key = f"{self.relation_prefix}:{chunk_id}"
                    await pipeline.hset_dict(hash_key, relation_data)
                    
                    # Set TTL (30 days)
                    await pipeline.expire(hash_key, 30 * 24 * 60 * 60)
            
            # Execute pipeline
            await pipeline.execute()
            logger.info(f"Successfully saved {total_relations} relations to Redis")
            
        except Exception as e:
            logger.error(f"Error saving relations to Redis: {e}")
            raise

    # Additional helper methods...
    async def _ensure_embeddings(self, chunks: List[ChunkData]) -> None:
        """Ensure chunks have embeddings, generate if missing."""
        chunks_without_embeddings = [
            chunk for chunk in chunks if chunk.embedding is None
        ]

        if chunks_without_embeddings:
            logger.info(
                f"Generating embeddings for {len(chunks_without_embeddings)} chunks"
            )

            # Prepare texts
            texts = []
            for chunk in chunks_without_embeddings:
                # Include title if available
                if chunk.section_title:
                    text = f"{chunk.section_title}\n{chunk.content}"
                else:
                    text = chunk.content
                texts.append(text)

            # Generate embeddings in batch
            if self.embedding_generator:  # Additional type check
                embeddings = await self.embedding_generator.embed_batch(texts)
            else:
                raise ValueError("Embedding generator not available")

            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks_without_embeddings, embeddings):
                chunk.embedding = embedding

    def _compute_similarity_matrix(self, chunks: List[ChunkData]) -> np.ndarray:
        """Calculate cosine similarity matrix between chunks."""
        # Create embedding array
        embeddings = np.array([chunk.embedding for chunk in chunks])

        # Normalize (for cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-10)  # Prevent division by zero

        # Calculate cosine similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        return similarity_matrix  # type: ignore[no-any-return]

    def _determine_semantic_relation_type(
        self, similarity_score: float
    ) -> RelationType:
        """Determine relationship type based on similarity score."""
        if similarity_score >= 0.9:
            return RelationType.SIMILAR  # Very similar
        elif similarity_score >= 0.7:
            return RelationType.RELATED  # Related
        else:
            # Low similarity not stored as semantic relationship
            return RelationType.RELATED

    async def _extract_shared_concepts(
        self, chunk1: ChunkData, chunk2: ChunkData
    ) -> Tuple[List[str], List[str]]:
        """Extract shared entities and topics between two chunks."""
        # Simple implementation - actual use would involve NER, topic modeling, etc.
        shared_entities: List[str] = []
        shared_topics: List[str] = []

        # Extract common words (simple example)
        words1 = set(chunk1.content.lower().split())
        words2 = set(chunk2.content.lower().split())

        # Remove stopwords (simple example)
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
        }
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        # Common words
        common_words = words1 & words2

        # Consider longer words as entities (simple heuristic)
        for word in common_words:
            if len(word) > 5:  # 5+ characters
                shared_entities.append(word)

        # If section titles match, add as topic
        if chunk1.section_title and chunk2.section_title:
            if chunk1.section_title == chunk2.section_title:
                shared_topics.append(chunk1.section_title)

        return shared_entities[:5], shared_topics[:3]  # Max 5 entities, 3 topics

    # Additional methods for reference extraction, temporal analysis, etc. would follow...
    # (Continuing with similar sanitization patterns)