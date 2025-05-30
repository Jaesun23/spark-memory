"""Memory Engine - Modular action-based architecture."""

import logging
from typing import Any, Dict, List, Optional, Union

from src.memory.models import SearchResult
from src.redis.client import RedisClient
from src.memory.actions import (
    BasicActions,
    SearchActions,
    ConsolidateActions,
    LifecycleActions,
    HelpActions,
)
from src.rag.relationships import ChunkRelationshipManager
from src.memory.cross_memory_bridge import CrossMemoryBridge
from src.memory.memory_analytics import MemoryAnalytics

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Memory Engine - Entry point for all memory operations.
    
    Action-based modular architecture where each feature is implemented independently.
    """
    
    def __init__(
        self,
        redis_client: RedisClient,
        enable_security: bool = True,
        enable_events: bool = True,
        default_timezone: str = "UTC",
        enable_relationships: bool = True,
    ):
        """Initialize memory engine.

        Args:
            redis_client: Redis client instance
            enable_security: Whether to enable security features
            enable_events: Whether to enable event features
            default_timezone: Default timezone
            enable_relationships: Whether to enable chunk relationship features
        """
        self.redis = redis_client
        self.enable_security = enable_security
        self.enable_events = enable_events
        self.enable_relationships = enable_relationships
        
        # Initialize relationship managers (optional)
        self.relationship_manager = None
        self.cross_memory_bridge = None
        self.memory_analytics = None
        if enable_relationships:
            self.relationship_manager = ChunkRelationshipManager(
                redis_client=redis_client,
                similarity_threshold=0.7,
                max_relations_per_chunk=50
            )
            # Initialize CrossMemoryBridge
            self.cross_memory_bridge = CrossMemoryBridge(
                redis_client=redis_client,
                relationship_manager=self.relationship_manager
            )
            # Initialize MemoryAnalytics
            self.memory_analytics = MemoryAnalytics(
                redis_client=redis_client,
                cross_memory_bridge=self.cross_memory_bridge
            )
        
        # Initialize action handlers
        self.basic_actions = BasicActions(redis_client, default_timezone)
        self.search_actions = SearchActions(
            redis_client, 
            relationship_manager=self.relationship_manager
        )
        self.consolidate_actions = ConsolidateActions(redis_client)
        self.lifecycle_actions = LifecycleActions(redis_client)
        self.help_actions = HelpActions()
        
        # Security related (maintain existing compatibility)
        self.access_control = None
        self.audit_logger = None
        self.field_encryption = None
        self.key_manager = None
        
        # Vector store (optional)
        self.vector_store = None
        
        # Additional managers (existing compatibility)
        self.consolidator = None
        self.lifecycle_manager = None
        
        logger.info("MemoryEngine initialized with modular actions")
        
    async def execute(
        self,
        action: str,
        paths: List[str],
        content: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        principal: Optional[Any] = None,  # Principal type from security module
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]], List[SearchResult]]:
        """Execute memory command - handles routing only.

        Args:
            action: Action to execute
            paths: Memory paths
            content: Content to save/modify
            options: Additional options
            principal: Execution principal (for security)

        Returns:
            Action execution result

        Raises:
            ValueError: Invalid action or parameters
            RuntimeError: Execution error
        """
        options = options or {}

        # Security check (maintain existing logic - when security module is enabled)
        if self.enable_security and self.access_control and principal:
            # Security checks are performed only when access_control is configured
            # TODO: Activate after importing security module
            pass

        # Action routing
        try:
            logger.info(f"Executing memory action: {action} with paths: {paths}")
            
            # Basic actions
            if action in ["save", "get", "update", "delete"]:
                result = await self.basic_actions.execute(action, paths, content, options)
            
            # Search action
            elif action == "search":
                result = await self.search_actions.execute(paths, content, options)
            
            # Consolidate action
            elif action == "consolidate":
                result = await self.consolidate_actions.execute(paths, options)
            
            # Lifecycle action
            elif action == "lifecycle":
                result = await self.lifecycle_actions.execute(paths, content, options)
            
            # Help action
            elif action == "help":
                result = await self.help_actions.execute(paths, content, options)
            
            # Cross-memory actions (LRMM advanced features)
            elif action == "link_conversation":
                if not self.cross_memory_bridge:
                    raise RuntimeError("CrossMemoryBridge not initialized. Enable relationships to use this feature.")
                conv_id = paths[0] if paths else None
                if not conv_id:
                    raise ValueError("Conversation ID required for link_conversation action")
                result = await self.cross_memory_bridge.link_conversation_to_documents(conv_id)
            
            elif action == "find_cross_memory":
                if not self.cross_memory_bridge:
                    raise RuntimeError("CrossMemoryBridge not initialized. Enable relationships to use this feature.")
                memory_key = paths[0] if paths else None
                if not memory_key:
                    raise ValueError("Memory key required for find_cross_memory action")
                
                # Check advanced search options
                if options.get("advanced", False):
                    cross_relations = await self.cross_memory_bridge.find_related_memories_advanced(
                        memory_key, options.get("search_options", {})
                    )
                else:
                    cross_relations = await self.cross_memory_bridge.find_related_memories(memory_key)
                
                # Convert to serializable format
                result = {
                    "source_id": cross_relations.source_id,
                    "source_type": cross_relations.source_type,
                    "related_conversations": cross_relations.related_conversations,
                    "related_documents": cross_relations.related_documents,
                    "temporal_neighbors": cross_relations.temporal_neighbors,
                    "shared_entities": cross_relations.shared_entities,
                    "metadata": cross_relations.metadata
                }
            
            elif action == "analyze_memory_graph":
                if not self.memory_analytics:
                    raise RuntimeError("MemoryAnalytics not initialized. Enable relationships to use this feature.")
                analysis = await self.memory_analytics.analyze_memory_graph()
                # Convert to serializable format
                result = {
                    "total_nodes": analysis.total_nodes,
                    "total_edges": analysis.total_edges,
                    "connected_components": analysis.connected_components,
                    "avg_clustering_coefficient": analysis.avg_clustering_coefficient,
                    "hub_nodes": analysis.hub_nodes,
                    "isolated_nodes": analysis.isolated_nodes,
                    "dense_clusters": [
                        {
                            "id": cluster.id,
                            "topic": cluster.topic,
                            "size": cluster.size,
                            "cohesion_score": cluster.cohesion_score,
                            "nodes": cluster.nodes[:5]  # First 5 only
                        }
                        for cluster in analysis.dense_clusters
                    ],
                    "temporal_patterns": analysis.temporal_patterns,
                    "recommendations": analysis.recommendations,
                    "analysis_timestamp": analysis.analysis_timestamp.isoformat()
                }
            
            elif action == "suggest_connections":
                if not self.memory_analytics:
                    raise RuntimeError("MemoryAnalytics not initialized. Enable relationships to use this feature.")
                suggestions = await self.memory_analytics.suggest_connections()
                # Convert to serializable format
                result = [
                    {
                        "source_id": s.source_id,
                        "target_id": s.target_id,
                        "suggested_type": s.suggested_type.value,
                        "confidence": s.confidence,
                        "reasoning": s.reasoning,
                        "potential_benefit": s.potential_benefit,
                        "metadata": s.metadata
                    }
                    for s in suggestions
                ]
            
            elif action == "get_memory_insights":
                if not self.memory_analytics:
                    raise RuntimeError("MemoryAnalytics not initialized. Enable relationships to use this feature.")
                memory_key = paths[0] if paths else None
                if not memory_key:
                    raise ValueError("Memory key required for get_memory_insights action")
                result = await self.memory_analytics.get_memory_insights(memory_key)
            
            else:
                raise ValueError(
                    f"Unknown action: {action}. Valid actions: save, get, search, update, delete, consolidate, lifecycle, help, link_conversation, find_cross_memory, analyze_memory_graph, suggest_connections, get_memory_insights"
                )

            # Success audit log (when security module is enabled)
            if self.enable_security and self.audit_logger and principal:
                # TODO: Record audit log after activating security module
                pass

            return result

        except ValueError as e:
            # Parameter validation error
            logger.error(f"Invalid parameters for {action}: {e}")
            
            # Suggest help for the error
            error_msg = str(e)
            suggestion = self._suggest_fix(error_msg, action)
            
            if self.enable_security and self.audit_logger and principal:
                # TODO: Record audit log after activating security module
                pass
            
            raise RuntimeError(f"Memory operation failed: {error_msg}\n\n{suggestion}") from e

        except Exception as e:
            logger.error(f"Error executing memory action {action}: {e}")
            
            if self.enable_security and self.audit_logger and principal:
                # TODO: Record audit log after activating security module
                pass
            
            raise

    # Security related methods - activate when integrating security module
    # async def _log_audit(...) -> None:
    #     """Record audit log."""
    #     pass
    #
    # def _get_audit_event_type(self, action: str):
    #     """Return audit event type corresponding to action."""
    #     pass

    def _suggest_fix(self, error_msg: str, action: str) -> str:
        """Analyze error message and suggest fix."""
        suggestions = {
            "Content is required": """
Content parameter is required.

m_memory("save", ["category"], "content to save")
""",
            "Paths are required": """
Paths parameter is required.

m_memory("get", ["path", "to", "memory"])
""",
            "Query string is required": """
Keyword search requires a search term.

Enter search term in content parameter:
m_memory("search", [], "search term")
""",
            "Time range search requires filters": """
Time range search requires time filters.

Add time filter to options:
m_memory("search", [], None, {
    "type": "time_range",
    "filters": {"date": "2025-05-28"}
})
""",
            "CrossMemoryBridge not initialized": """
CrossMemoryBridge is not initialized.

Enable relationships when initializing memory engine to use LRMM advanced features:
- link_conversation (automatic conversation-document linking)
- find_cross_memory (cross-memory search)
""",
            "MemoryAnalytics not initialized": """
MemoryAnalytics is not initialized.

Enable relationships when initializing memory engine to use LRMM advanced features:
- analyze_memory_graph (memory graph analysis)
- suggest_connections (AI connection suggestions)
- get_memory_insights (individual memory insights)
""",
            "Conversation ID required": """
Conversation ID is required.

link_conversation action requires a conversation key:
m_memory("link_conversation", ["stream:memory:conversation:2024/01/15/10/30/00"])
""",
            "Memory key required": """
Memory key is required.

Provide memory key in one of these formats:
- Conversation: "stream:memory:conversation:2024/01/15/10/30/00"
- Document: "json:memory:document:2024/01/15/report.pdf"

Example:
m_memory("find_cross_memory", ["json:memory:document:2024/01/15/report.pdf"])
m_memory("get_memory_insights", ["stream:memory:conversation:2024/01/15/10/30/00"])
""",
        }

        for key, suggestion in suggestions.items():
            if key in error_msg:
                return suggestion

        # LRMM feature related general help
        if any(action in error_msg for action in ["link_conversation", "find_cross_memory", "analyze_memory_graph", "suggest_connections", "get_memory_insights"]):
            return """
LRMM Advanced Features Usage:

🔗 Conversation-Document Linking: m_memory("link_conversation", [conversation_ID])
🔍 Cross-Memory Search: m_memory("find_cross_memory", [memory_key])
📊 Graph Analysis: m_memory("analyze_memory_graph", [])
🤖 Connection Suggestions: m_memory("suggest_connections", [])
🔍 Memory Insights: m_memory("get_memory_insights", [memory_key])

Detailed help: m_memory("help", ["feature_name"])
"""

        return f"For '{action}' action usage, refer to help:\nm_memory('help', ['{action}'])"
    
    def set_vector_store(self, vector_store: Any) -> None:
        """Set vector store.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        # Also set to SearchActions
        self.search_actions.vector_store = vector_store
        
    def set_embedding_service(self, embedding_service: Any) -> None:
        """Set embedding service.
        
        Args:
            embedding_service: Embedding service instance
        """
        if self.relationship_manager:
            self.relationship_manager.embedding_generator = embedding_service