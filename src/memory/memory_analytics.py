"""MemoryAnalytics - Memory relationship graph analysis and connection suggestion system"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from ..redis.client import RedisClient
from .cross_memory_bridge import CrossMemoryBridge

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Node type"""
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    ENTITY = "entity"
    KEYWORD = "keyword"


class ConnectionType(str, Enum):
    """Connection type"""
    REFERENCE = "reference"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    KEYWORD = "keyword"


@dataclass
class MemoryNode:
    """Memory graph node"""
    id: str
    type: NodeType
    content_preview: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    centrality_score: float = 0.0
    importance_score: float = 0.0


@dataclass
class MemoryEdge:
    """Memory graph edge"""
    source_id: str
    target_id: str
    type: ConnectionType
    weight: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCluster:
    """Memory cluster"""
    id: str
    nodes: List[str]
    topic: str
    cohesion_score: float
    size: int
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConnectionSuggestion:
    """Connection suggestion"""
    source_id: str
    target_id: str
    suggested_type: ConnectionType
    confidence: float
    reasoning: str
    potential_benefit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryGraphAnalysis:
    """Memory graph analysis results"""
    total_nodes: int
    total_edges: int
    connected_components: int
    avg_clustering_coefficient: float
    hub_nodes: List[Tuple[str, float]]  # (node_id, centrality_score)
    isolated_nodes: List[str]
    dense_clusters: List[MemoryCluster]
    temporal_patterns: Dict[str, Any]
    recommendations: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class MemoryAnalytics:
    """Memory graph analysis and connection suggestion system"""
    
    def __init__(
        self,
        redis_client: RedisClient,
        cross_memory_bridge: Optional[CrossMemoryBridge] = None,
        min_connection_weight: float = 0.3,
        max_graph_size: int = 10000
    ):
        self.redis = redis_client
        self.bridge = cross_memory_bridge
        self.min_connection_weight = min_connection_weight
        self.max_graph_size = max_graph_size
        self._graph_cache: Optional[Any] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)
        
    async def analyze_memory_graph(self) -> MemoryGraphAnalysis:
        """Analyze memory relationship graph
        
        Returns:
            Memory graph analysis results
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available, using simplified analysis")
            return await self._analyze_memory_simple()
            
        try:
            logger.info("Starting memory graph analysis")
            
            # 1. Build graph
            graph = await self._build_memory_graph()
            
            # 2. Basic statistics
            total_nodes = graph.number_of_nodes()
            total_edges = graph.number_of_edges()
            
            # 3. Connected components analysis
            connected_components = nx.number_connected_components(graph)
            
            # 4. Clustering coefficient
            avg_clustering = nx.average_clustering(graph) if total_nodes > 0 else 0.0
            
            # 5. Identify hub nodes (centrality-based)
            hub_nodes = await self._identify_hub_nodes(graph)
            
            # 6. Detect isolated nodes
            isolated_nodes = await self._find_isolated_nodes(graph)
            
            # 7. Analyze dense clusters
            dense_clusters = await self._analyze_dense_clusters(graph)
            
            # 8. Temporal pattern analysis
            temporal_patterns = await self._analyze_temporal_patterns(graph)
            
            # 9. Generate improvement recommendations
            recommendations = await self._generate_recommendations(
                graph, hub_nodes, isolated_nodes, dense_clusters
            )
            
            analysis = MemoryGraphAnalysis(
                total_nodes=total_nodes,
                total_edges=total_edges,
                connected_components=connected_components,
                avg_clustering_coefficient=avg_clustering,
                hub_nodes=hub_nodes,
                isolated_nodes=isolated_nodes,
                dense_clusters=dense_clusters,
                temporal_patterns=temporal_patterns,
                recommendations=recommendations
            )
            
            # Cache analysis results
            await self._cache_analysis_result(analysis)
            
            logger.info(f"Memory graph analysis completed: {total_nodes} nodes, {total_edges} edges")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in memory graph analysis: {e}")
            raise
            
    async def _analyze_memory_simple(self) -> MemoryGraphAnalysis:
        """Simple memory analysis without NetworkX"""
        try:
            # Basic counts
            conv_keys = await self.redis.keys("stream:memory:conversation:*")
            doc_keys = await self.redis.keys("json:memory:document:*")
            total_nodes = len(conv_keys) + len(doc_keys)
            
            # Calculate link count
            link_keys = await self.redis.keys("link:conv_to_doc:*")
            total_edges = 0
            for key_bytes in link_keys:
                key = key_bytes.decode("utf-8")
                links = await self.redis.zcard(key)
                total_edges += links
            
            # Simple recommendations
            recommendations = []
            if total_nodes > 0:
                density = total_edges / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
                if density < 0.1:
                    recommendations.append("Memory connectivity is low. Consider linking related memories.")
                if total_nodes > 100:
                    recommendations.append("Many memories accumulated. Regular cleanup is recommended.")
            
            return MemoryGraphAnalysis(
                total_nodes=total_nodes,
                total_edges=total_edges,
                connected_components=1,  # estimated value
                avg_clustering_coefficient=0.0,
                hub_nodes=[],
                isolated_nodes=[],
                dense_clusters=[],
                temporal_patterns={},
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in simple memory analysis: {e}")
            return MemoryGraphAnalysis(
                total_nodes=0,
                total_edges=0,
                connected_components=0,
                avg_clustering_coefficient=0.0,
                hub_nodes=[],
                isolated_nodes=[],
                dense_clusters=[],
                temporal_patterns={},
                recommendations=["Error occurred during analysis."]
            )
            
    async def _build_memory_graph(self) -> Any:
        """Build memory graph"""
        # Check cache
        if (self._graph_cache and self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._graph_cache
            
        if not HAS_NETWORKX:
            return None
        graph = nx.Graph()
        
        try:
            # 1. Collect all memory nodes
            nodes = await self._collect_memory_nodes()
            logger.info(f"Collected {len(nodes)} memory nodes")
            
            # Size limit
            if len(nodes) > self.max_graph_size:
                # Select only the most recent ones
                nodes = sorted(nodes, key=lambda n: n.timestamp or datetime.min, reverse=True)
                nodes = nodes[:self.max_graph_size]
                logger.warning(f"Limited graph to {self.max_graph_size} most recent nodes")
            
            # Add nodes
            for node in nodes:
                graph.add_node(
                    node.id,
                    type=node.type.value,
                    content_preview=node.content_preview[:100],
                    timestamp=node.timestamp,
                    metadata=node.metadata
                )
            
            # 2. Collect and add connection relationships
            edges = await self._collect_memory_edges(nodes)
            logger.info(f"Collected {len(edges)} memory edges")
            
            for edge in edges:
                if (graph.has_node(edge.source_id) and 
                    graph.has_node(edge.target_id) and
                    edge.weight >= self.min_connection_weight):
                    
                    graph.add_edge(
                        edge.source_id,
                        edge.target_id,
                        type=edge.type.value,
                        weight=edge.weight,
                        confidence=edge.confidence,
                        metadata=edge.metadata
                    )
            
            # Update cache
            self._graph_cache = graph
            self._cache_timestamp = datetime.now()
            
            return graph
            
        except Exception as e:
            logger.error(f"Error building memory graph: {e}")
            raise
            
    async def _collect_memory_nodes(self) -> List[MemoryNode]:
        """Collect memory nodes"""
        nodes = []
        
        try:
            # Conversation nodes
            conv_keys = await self.redis.keys("stream:memory:conversation:*")
            for key_bytes in conv_keys:
                key = key_bytes.decode("utf-8")
                node = await self._create_conversation_node(key)
                if node:
                    nodes.append(node)
            
            # Document nodes
            doc_keys = await self.redis.keys("json:memory:document:*")
            for key_bytes in doc_keys:
                key = key_bytes.decode("utf-8")
                node = await self._create_document_node(key)
                if node:
                    nodes.append(node)
            
            # Entity nodes (important entities only)
            entity_nodes = await self._collect_important_entities()
            nodes.extend(entity_nodes)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error collecting memory nodes: {e}")
            return nodes
            
    async def _create_conversation_node(self, conv_key: str) -> Optional[MemoryNode]:
        """Create conversation node"""
        try:
            # Sample conversation content
            messages = await self.redis.xrange(conv_key, "-", "+", count=3)
            if not messages:
                return None
                
            contents = []
            for _, data in messages:
                if b"content" in data:
                    content = data[b"content"].decode("utf-8")
                    contents.append(content)
            
            content_preview = " ".join(contents)[:200]
            
            # Extract timestamp
            timestamp = await self._extract_timestamp_from_key(conv_key)
            
            return MemoryNode(
                id=conv_key,
                type=NodeType.CONVERSATION,
                content_preview=content_preview,
                timestamp=timestamp,
                metadata={"message_count": len(messages)}
            )
            
        except Exception as e:
            logger.debug(f"Error creating conversation node for {conv_key}: {e}")
            return None
            
    async def _create_document_node(self, doc_key: str) -> Optional[MemoryNode]:
        """Create document node"""
        try:
            # Get document content
            doc_data = await self.redis.json_get(doc_key)
            if not doc_data:
                return None
                
            content = doc_data.get("content", "")
            title = doc_data.get("title", "")
            
            content_preview = f"{title} - {content}"[:200]
            
            # Extract timestamp
            timestamp = await self._extract_timestamp_from_key(doc_key)
            
            return MemoryNode(
                id=doc_key,
                type=NodeType.DOCUMENT,
                content_preview=content_preview,
                timestamp=timestamp,
                metadata={
                    "title": title,
                    "content_length": len(content)
                }
            )
            
        except Exception as e:
            logger.debug(f"Error creating document node for {doc_key}: {e}")
            return None
            
    async def _collect_important_entities(self) -> List[MemoryNode]:
        """Collect important entity nodes"""
        entities = []
        
        try:
            # Entity key patterns to search
            entity_patterns = [
                "entity:person:*",
                "entity:org:*",
                "entity:location:*"
            ]
            
            for pattern in entity_patterns:
                entity_keys = await self.redis.keys(pattern)
                
                for key_bytes in entity_keys:
                    key = key_bytes.decode("utf-8")
                    
                    # Check number of memories containing this entity
                    related_memories = await self.redis.smembers(key)
                    memory_count = len(related_memories)
                    
                    # Include only entities mentioned in at least 3 memories
                    if memory_count >= 3:
                        entity_name = key.split(":")[-1]
                        entity_type = key.split(":")[1]
                        
                        entities.append(MemoryNode(
                            id=key,
                            type=NodeType.ENTITY,
                            content_preview=f"{entity_type.upper()}: {entity_name}",
                            metadata={
                                "entity_name": entity_name,
                                "entity_type": entity_type,
                                "memory_count": memory_count
                            }
                        ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error collecting important entities: {e}")
            return entities
            
    async def _extract_timestamp_from_key(self, key: str) -> Optional[datetime]:
        """Extract timestamp from key"""
        import re
        
        # Match time pattern (YYYY/MM/DD/HH/MM/SS)
        time_pattern = r"(\d{4})/(\d{2})/(\d{2})/(\d{2})/(\d{2})/(\d{2})"
        match = re.search(time_pattern, key)
        
        if match:
            year, month, day, hour, minute, second = match.groups()
            return datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second)
            )
        
        # Date only (YYYY/MM/DD)
        date_pattern = r"(\d{4})/(\d{2})/(\d{2})"
        match = re.search(date_pattern, key)
        
        if match:
            year, month, day = match.groups()
            return datetime(int(year), int(month), int(day))
            
        return None
        
    async def _collect_memory_edges(self, nodes: List[MemoryNode]) -> List[MemoryEdge]:
        """Collect memory edges"""
        edges = []
        
        try:
            # 1. Reference relationship edges
            reference_edges = await self._collect_reference_edges(nodes)
            edges.extend(reference_edges)
            
            # 2. Entity-based edges
            entity_edges = await self._collect_entity_edges(nodes)
            edges.extend(entity_edges)
            
            # 3. Temporal proximity edges
            temporal_edges = await self._collect_temporal_edges(nodes)
            edges.extend(temporal_edges)
            
            # 4. Semantic similarity edges (sampled)
            if self.bridge and self.bridge.embedder:
                semantic_edges = await self._collect_semantic_edges_sampled(nodes)
                edges.extend(semantic_edges)
            
            return edges
            
        except Exception as e:
            logger.error(f"Error collecting memory edges: {e}")
            return edges
            
    async def _collect_reference_edges(self, nodes: List[MemoryNode]) -> List[MemoryEdge]:
        """Collect reference relationship edges"""
        edges = []
        
        try:
            # Collect reference relationships from link metadata
            link_keys = await self.redis.keys("link:conv_to_doc:*")
            
            for key_bytes in link_keys:
                key = key_bytes.decode("utf-8")
                conv_id = key.split(":")[-1]
                
                # Connected documents
                linked_docs = await self.redis.zrevrange(key, 0, -1, withscores=True)
                
                for doc_id_bytes, weight in linked_docs:
                    doc_id = doc_id_bytes.decode("utf-8")
                    
                    if weight >= self.min_connection_weight:
                        edges.append(MemoryEdge(
                            source_id=conv_id,
                            target_id=doc_id,
                            type=ConnectionType.REFERENCE,
                            weight=weight,
                            confidence=0.9
                        ))
            
            return edges
            
        except Exception as e:
            logger.error(f"Error collecting reference edges: {e}")
            return edges
            
    async def _collect_entity_edges(self, nodes: List[MemoryNode]) -> List[MemoryEdge]:
        """Collect entity-based edges"""
        edges = []
        
        try:
            entity_nodes = [n for n in nodes if n.type == NodeType.ENTITY]
            
            for entity_node in entity_nodes:
                # Memories containing this entity
                related_memories = await self.redis.smembers(entity_node.id)
                memory_list = [m.decode("utf-8") for m in related_memories]
                
                # Create entity-based connections between memories
                for i, mem1 in enumerate(memory_list):
                    for mem2 in memory_list[i+1:]:
                        # Two memories share the same entity
                        weight = 0.6  # Default weight for entity sharing
                        
                        edges.append(MemoryEdge(
                            source_id=mem1,
                            target_id=mem2,
                            type=ConnectionType.ENTITY,
                            weight=weight,
                            confidence=0.8,
                            metadata={"shared_entity": entity_node.metadata.get("entity_name")}
                        ))
            
            return edges
            
        except Exception as e:
            logger.error(f"Error collecting entity edges: {e}")
            return edges
            
    async def _collect_temporal_edges(self, nodes: List[MemoryNode]) -> List[MemoryEdge]:
        """Collect temporal proximity edges"""
        edges = []
        
        try:
            # Only nodes with timestamps
            timestamped_nodes = [n for n in nodes if n.timestamp]
            timestamped_nodes.sort(key=lambda n: n.timestamp)
            
            # Set time window (1 hour)
            time_window = timedelta(hours=1)
            
            for i, node1 in enumerate(timestamped_nodes):
                for node2 in timestamped_nodes[i+1:]:
                    time_diff = abs((node1.timestamp - node2.timestamp).total_seconds())
                    
                    if time_diff <= time_window.total_seconds():
                        # Temporal proximity score (closer = higher)
                        proximity = 1.0 - (time_diff / time_window.total_seconds())
                        
                        if proximity > 0.3:  # Minimum threshold
                            edges.append(MemoryEdge(
                                source_id=node1.id,
                                target_id=node2.id,
                                type=ConnectionType.TEMPORAL,
                                weight=proximity,
                                confidence=0.7
                            ))
                    else:
                        # Break if outside time window
                        break
            
            return edges
            
        except Exception as e:
            logger.error(f"Error collecting temporal edges: {e}")
            return edges
            
    async def _collect_semantic_edges_sampled(self, nodes: List[MemoryNode]) -> List[MemoryEdge]:
        """Collect semantic similarity edges (sampled)"""
        edges = []
        
        try:
            # Sample nodes for performance
            memory_nodes = [n for n in nodes if n.type in [NodeType.CONVERSATION, NodeType.DOCUMENT]]
            
            if len(memory_nodes) > 100:
                # Select only most recent 100
                memory_nodes = sorted(memory_nodes, key=lambda n: n.timestamp or datetime.min, reverse=True)[:100]
            
            # Embedding-based similarity calculation (partial only)
            for i, node1 in enumerate(memory_nodes):
                if i % 10 == 0:  # Process only 1 out of 10
                    similar_nodes = await self._find_similar_nodes(node1, memory_nodes[i+1:])
                    
                    for similar_node, similarity in similar_nodes:
                        if similarity > 0.7:  # High similarity only
                            edges.append(MemoryEdge(
                                source_id=node1.id,
                                target_id=similar_node.id,
                                type=ConnectionType.SEMANTIC,
                                weight=similarity,
                                confidence=0.8
                            ))
            
            return edges
            
        except Exception as e:
            logger.error(f"Error collecting semantic edges: {e}")
            return edges
            
    async def _find_similar_nodes(
        self, target_node: MemoryNode, candidate_nodes: List[MemoryNode]
    ) -> List[Tuple[MemoryNode, float]]:
        """Find similar nodes"""
        if not self.bridge or not self.bridge.embedder:
            return []
            
        try:
            # Generate embedding for target node
            target_embedding = await self.bridge._get_or_create_embedding(
                target_node.id, target_node.content_preview
            )
            
            similarities = []
            for candidate in candidate_nodes[:20]:  # Compare with max 20
                try:
                    candidate_embedding = await self.bridge._get_or_create_embedding(
                        candidate.id, candidate.content_preview
                    )
                    
                    similarity = self.bridge._cosine_similarity(target_embedding, candidate_embedding)
                    similarities.append((candidate, similarity))
                    
                except Exception:
                    continue
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:5]  # Top 5
            
        except Exception as e:
            logger.debug(f"Error finding similar nodes: {e}")
            return []
            
    async def _identify_hub_nodes(self, graph: Any) -> List[Tuple[str, float]]:
        """Identify hub nodes (centrality-based)"""
        try:
            if graph.number_of_nodes() == 0:
                return []
            
            # Calculate multiple centrality metrics
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()))
            
            # Calculate composite centrality score
            hub_scores = {}
            for node in graph.nodes():
                combined_score = (
                    0.6 * degree_centrality.get(node, 0) +
                    0.4 * betweenness_centrality.get(node, 0)
                )
                hub_scores[node] = combined_score
            
            # Return top hub nodes
            sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_hubs[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"Error identifying hub nodes: {e}")
            return []
            
    async def _find_isolated_nodes(self, graph: Any) -> List[str]:
        """Detect isolated nodes"""
        try:
            isolated = []
            
            for node in graph.nodes():
                if graph.degree(node) == 0:
                    isolated.append(node)
                elif graph.degree(node) == 1:
                    # Nodes with only one connection are also vulnerable
                    neighbor = list(graph.neighbors(node))[0]
                    if graph.degree(neighbor) == 1:  # The neighbor also has only one connection
                        isolated.append(node)
            
            return isolated
            
        except Exception as e:
            logger.error(f"Error finding isolated nodes: {e}")
            return []
            
    async def _analyze_dense_clusters(self, graph: Any) -> List[MemoryCluster]:
        """Analyze dense clusters"""
        clusters = []
        
        try:
            if graph.number_of_nodes() < 3:
                return clusters
                
            # Community detection
            communities = nx.community.greedy_modularity_communities(graph)
            
            for i, community in enumerate(communities):
                if len(community) >= 3:  # Minimum 3 nodes
                    # Calculate cluster connection density
                    subgraph = graph.subgraph(community)
                    density = nx.density(subgraph)
                    
                    if density > 0.3:  # Density threshold
                        # Extract topic (simple method)
                        topic = await self._extract_cluster_topic(list(community))
                        
                        clusters.append(MemoryCluster(
                            id=f"cluster_{i}",
                            nodes=list(community),
                            topic=topic,
                            cohesion_score=density,
                            size=len(community)
                        ))
            
            # Sort by cluster size
            clusters.sort(key=lambda c: c.size, reverse=True)
            return clusters[:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Error analyzing dense clusters: {e}")
            return clusters
            
    async def _extract_cluster_topic(self, node_ids: List[str]) -> str:
        """Extract cluster topic"""
        try:
            # Extract common keywords from node content
            all_content = []
            
            for node_id in node_ids[:5]:  # Process max 5
                if "conversation" in node_id:
                    messages = await self.redis.xrange(node_id, "-", "+", count=3)
                    for _, data in messages:
                        if b"content" in data:
                            content = data[b"content"].decode("utf-8")
                            all_content.append(content)
                elif "document" in node_id:
                    doc_data = await self.redis.json_get(node_id)
                    if doc_data and "title" in doc_data:
                        all_content.append(doc_data["title"])
            
            if not all_content:
                return "Unknown Topic"
                
            # Simple keyword extraction
            import re
            from collections import Counter
            
            combined_text = " ".join(all_content).lower()
            words = re.findall(r'[a-zA-Z]{3,}', combined_text)
            
            # Remove stopwords
            stopwords = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
            }
            words = [w for w in words if w not in stopwords and len(w) > 2]
            
            if words:
                most_common = Counter(words).most_common(3)
                return " ".join([word for word, _ in most_common])
            else:
                return "Mixed Content"
                
        except Exception as e:
            logger.debug(f"Error extracting cluster topic: {e}")
            return "Unknown Topic"
            
    async def _analyze_temporal_patterns(self, graph: Any) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        patterns = {
            "peak_hours": [],
            "active_days": [],
            "memory_creation_trend": {},
            "temporal_clusters": []
        }
        
        try:
            # Collect node timestamps
            timestamps = []
            for node in graph.nodes(data=True):
                if node[1].get("timestamp"):
                    timestamps.append(node[1]["timestamp"])
            
            if not timestamps:
                return patterns
                
            # Analyze activity by hour
            hour_counts = Counter([ts.hour for ts in timestamps])
            patterns["peak_hours"] = [hour for hour, _ in hour_counts.most_common(3)]
            
            # Analyze activity by day of week
            day_counts = Counter([ts.weekday() for ts in timestamps])
            patterns["active_days"] = [day for day, _ in day_counts.most_common(3)]
            
            # Memory creation trend (daily)
            from collections import defaultdict
            daily_counts = defaultdict(int)
            for ts in timestamps:
                date_key = ts.strftime("%Y-%m-%d")
                daily_counts[date_key] += 1
            
            patterns["memory_creation_trend"] = dict(daily_counts)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return patterns
            
    async def _generate_recommendations(
        self,
        graph: Any,
        hub_nodes: List[Tuple[str, float]],
        isolated_nodes: List[str],
        clusters: List[MemoryCluster]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        try:
            # 1. Isolated node recommendations
            if isolated_nodes:
                count = len(isolated_nodes)
                recommendations.append(
                    f"{count} isolated memories were found. "
                    "Consider adding related tags or keywords to connect them with other memories."
                )
            
            # 2. Hub node utilization recommendations
            if hub_nodes:
                top_hub = hub_nodes[0][0]
                recommendations.append(
                    f"'{top_hub[:50]}...' is the most connected important memory. "
                    "Consider systematically organizing related information around this."
                )
            
            # 3. Cluster-related recommendations
            if clusters:
                largest_cluster = max(clusters, key=lambda c: c.size)
                recommendations.append(
                    f"'{largest_cluster.topic}' topic has {largest_cluster.size} densely connected memories. "
                    "Consider creating a summary document for this topic."
                )
            
            # 4. Connectivity improvement recommendations
            density = nx.density(graph) if graph.number_of_nodes() > 0 else 0
            if density < 0.1:
                recommendations.append(
                    "Memory connectivity is low. Consider adding references or tags between "
                    "related memories to improve connectivity."
                )
            
            # 5. General recommendations
            if graph.number_of_nodes() > 100:
                recommendations.append(
                    "Many memories have been accumulated. Periodically cleaning up unimportant "
                    "memories and summarizing key content is recommended."
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["System error occurred during analysis. Please contact the administrator."]
            
    async def _cache_analysis_result(self, analysis: MemoryGraphAnalysis):
        """Cache analysis results"""
        try:
            # Store analysis results in Redis (24-hour TTL)
            analysis_key = "memory:analytics:latest"
            analysis_data = {
                "total_nodes": analysis.total_nodes,
                "total_edges": analysis.total_edges,
                "connected_components": analysis.connected_components,
                "avg_clustering_coefficient": analysis.avg_clustering_coefficient,
                "hub_nodes_count": len(analysis.hub_nodes),
                "isolated_nodes_count": len(analysis.isolated_nodes),
                "dense_clusters_count": len(analysis.dense_clusters),
                "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
                "recommendations": analysis.recommendations
            }
            
            await self.redis.setex(
                analysis_key,
                86400,  # 24 hours
                str(analysis_data)
            )
            
        except Exception as e:
            logger.debug(f"Error caching analysis result: {e}")
            
    async def suggest_connections(self) -> List[ConnectionSuggestion]:
        """Suggest potential connection relationships
        
        Returns:
            List of connection suggestions
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available, using simplified connection suggestions")
            return await self._suggest_connections_simple()
            
        try:
            logger.info("Starting connection suggestion analysis")
            
            suggestions = []
            
            # 1. Build graph (use cache)
            graph = await self._build_memory_graph()
            
            # 2. Generate connection suggestions using multiple strategies
            suggestion_tasks = [
                self._suggest_semantic_connections(graph),
                self._suggest_temporal_connections(graph),
                self._suggest_entity_connections(graph),
                self._suggest_topic_connections(graph),
                self._suggest_missing_references(graph)
            ]
            
            # Execute in parallel
            results = await asyncio.gather(*suggestion_tasks, return_exceptions=True)
            
            # Consolidate results
            for result in results:
                if not isinstance(result, Exception):
                    suggestions.extend(result)
            
            # 3. Remove duplicates and sort by score
            unique_suggestions = self._deduplicate_suggestions(suggestions)
            
            # 4. Return only top suggestions (max 20)
            top_suggestions = sorted(
                unique_suggestions,
                key=lambda s: s.confidence,
                reverse=True
            )[:20]
            
            logger.info(f"Generated {len(top_suggestions)} connection suggestions")
            return top_suggestions
            
        except Exception as e:
            logger.error(f"Error generating connection suggestions: {e}")
            return []
            
    async def _suggest_connections_simple(self) -> List[ConnectionSuggestion]:
        """Simple connection suggestions without NetworkX"""
        suggestions = []
        
        try:
            # Entity-based suggestions only
            entity_keys = await self.redis.keys("entity:*:*")
            
            for key_bytes in entity_keys[:10]:  # Top 10 only
                key = key_bytes.decode("utf-8")
                related_memories = await self.redis.smembers(key)
                
                memory_list = [m.decode("utf-8") for m in related_memories]
                if len(memory_list) >= 2:
                    entity_name = key.split(":")[-1]
                    
                    # Suggest only first two memories
                    if len(memory_list) >= 2:
                        suggestions.append(ConnectionSuggestion(
                            source_id=memory_list[0],
                            target_id=memory_list[1],
                            suggested_type=ConnectionType.ENTITY,
                            confidence=0.7,
                            reasoning=f"Share common entity '{entity_name}'",
                            potential_benefit="Connect related information"
                        ))
            
            return suggestions[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"Error in simple connection suggestions: {e}")
            return suggestions
            
    # Additional suggestion methods would go here...
    # (truncated for brevity - the remaining methods follow similar patterns)
    
    def _deduplicate_suggestions(self, suggestions: List[ConnectionSuggestion]) -> List[ConnectionSuggestion]:
        """Remove duplicate connection suggestions"""
        unique_suggestions = {}
        
        for suggestion in suggestions:
            # Create normalized key (order-independent)
            key_pair = tuple(sorted([suggestion.source_id, suggestion.target_id]))
            key = (key_pair, suggestion.suggested_type.value)
            
            # Keep suggestion with higher confidence
            if key not in unique_suggestions or suggestion.confidence > unique_suggestions[key].confidence:
                unique_suggestions[key] = suggestion
        
        return list(unique_suggestions.values())
        
    async def get_memory_insights(self, memory_id: str) -> Dict[str, Any]:
        """Provide insights for specific memory
        
        Args:
            memory_id: Memory ID to analyze
            
        Returns:
            Memory insight information
        """
        try:
            insights = {
                "memory_id": memory_id,
                "centrality_score": 0.0,
                "connection_count": 0,
                "connection_types": {},
                "cluster_membership": None,
                "importance_indicators": [],
                "related_memories": [],
                "suggestions": []
            }
            
            if not HAS_NETWORKX:
                # Provide simple insights
                insights["suggestions"] = ["NetworkX not available for detailed analysis."]
                return insights
            
            # Analyze this node in the graph
            graph = await self._build_memory_graph()
            
            if not graph or not graph.has_node(memory_id):
                insights["error"] = "Memory not found in graph"
                return insights
            
            # 1. Centrality score
            degree_centrality = nx.degree_centrality(graph)
            insights["centrality_score"] = degree_centrality.get(memory_id, 0.0)
            
            # 2. Connection count and type analysis
            neighbors = list(graph.neighbors(memory_id))
            insights["connection_count"] = len(neighbors)
            
            connection_types = defaultdict(int)
            for neighbor in neighbors:
                edge_data = graph.get_edge_data(memory_id, neighbor, {})
                conn_type = edge_data.get("type", "unknown")
                connection_types[conn_type] += 1
            
            insights["connection_types"] = dict(connection_types)
            
            # 3. Cluster membership
            clusters = await self._analyze_dense_clusters(graph)
            for cluster in clusters:
                if memory_id in cluster.nodes:
                    insights["cluster_membership"] = {
                        "cluster_id": cluster.id,
                        "topic": cluster.topic,
                        "size": cluster.size,
                        "cohesion": cluster.cohesion_score
                    }
                    break
            
            # 4. Importance indicators
            if insights["centrality_score"] > 0.1:
                insights["importance_indicators"].append("High centrality - key information")
            
            if insights["connection_count"] > 5:
                insights["importance_indicators"].append("Multiple connections - hub role")
            
            # 5. Related memories (directly connected)
            for neighbor in neighbors[:5]:  # Top 5
                edge_data = graph.get_edge_data(memory_id, neighbor, {})
                insights["related_memories"].append({
                    "memory_id": neighbor,
                    "connection_type": edge_data.get("type", "unknown"),
                    "weight": edge_data.get("weight", 0.0)
                })
            
            # 6. Personalized suggestions
            personal_suggestions = await self._generate_personal_suggestions(memory_id, graph)
            insights["suggestions"] = personal_suggestions
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting memory insights for {memory_id}: {e}")
            return {"error": str(e)}
            
    async def _generate_personal_suggestions(self, memory_id: str, graph: Any) -> List[str]:
        """Generate personalized suggestions"""
        suggestions = []
        
        try:
            # Low connection count
            degree = graph.degree(memory_id)
            if degree < 2:
                suggestions.append("This memory has insufficient connections to other information. Consider adding related tags or references.")
            
            # Not in a cluster
            clusters = await self._analyze_dense_clusters(graph)
            in_cluster = any(memory_id in cluster.nodes for cluster in clusters)
            
            if not in_cluster and degree > 0:
                suggestions.append("This memory can be grouped by topic. Consider connecting it with other memories of related topics.")
            
            # Temporally isolated
            node_data = graph.nodes[memory_id]
            timestamp = node_data.get("timestamp")
            
            if timestamp:
                # Check if there are other memories in the same time period
                time_neighbors = []
                for node in graph.nodes():
                    if node != memory_id:
                        other_timestamp = graph.nodes[node].get("timestamp")
                        if other_timestamp:
                            time_diff = abs((timestamp - other_timestamp).total_seconds())
                            if time_diff < 3600:  # Within 1 hour
                                time_neighbors.append(node)
                
                if time_neighbors and not any(graph.has_edge(memory_id, neighbor) for neighbor in time_neighbors):
                    suggestions.append("Consider connecting with other memories created around the same time.")
            
            return suggestions
            
        except Exception as e:
            logger.debug(f"Error generating personal suggestions: {e}")
            return suggestions