"""Intelligent search action implementation.

Provides advanced search capabilities including relationship-based expansion,
temporal context, and reranking.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque

from ...memory.models import (
    SearchResult as BaseSearchResult,
    SearchType,
    MemoryContent,
    MemoryType,
)
from ...memory.state_manager import StateManager
from ...rag.pipeline import RAGPipeline
from ...rag.relationships import RelationType
from ...redis.client import RedisClient

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result wrapper for simpler interface.

    Internally uses BaseSearchResult but provides
    a simpler interface.
    """

    path: str  # Actually stored as key
    score: float
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def to_base_result(self) -> BaseSearchResult:
        """Convert to BaseSearchResult."""
        memory_content = MemoryContent(type=MemoryType.DOCUMENT, data=self.content)
        return BaseSearchResult(
            key=self.path,
            content=memory_content,
            score=self.score,
            metadata=self.metadata,
        )

    @classmethod
    def from_base_result(cls, base_result: BaseSearchResult) -> "SearchResult":
        """Convert from BaseSearchResult."""
        content = (
            base_result.content.data
            if isinstance(base_result.content.data, str)
            else str(base_result.content.data)
        )
        return cls(
            path=base_result.key,
            score=base_result.score,
            content=content,
            metadata=base_result.metadata,
        )


class ExpansionStrategy(str, Enum):
    """Search expansion strategy."""

    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search
    WEIGHTED = "weighted"  # Weight-based
    HYBRID = "hybrid"  # BFS + weight


class TemporalDirection(str, Enum):
    """Temporal search direction."""

    PAST = "past"
    FUTURE = "future"
    BOTH = "both"


@dataclass
class SearchStrategy:
    """Search strategy configuration."""

    expansion_strategy: ExpansionStrategy = ExpansionStrategy.WEIGHTED
    relation_weights: Optional[Dict[RelationType, float]] = None
    max_expansion_depth: int = 2
    max_results: int = 50
    min_relevance_score: float = 0.5
    temporal_window: int = 3600  # Seconds
    temporal_direction: TemporalDirection = TemporalDirection.BOTH
    enable_reranking: bool = True

    def __post_init__(self) -> None:
        """Set default relationship weights."""
        if self.relation_weights is None:
            self.relation_weights = {
                RelationType.SIMILAR: 0.9,
                RelationType.RELATED: 0.8,
                RelationType.REFERENCES: 0.7,
                RelationType.MENTIONED_BY: 0.6,
                RelationType.PARENT: 0.8,
                RelationType.CHILD: 0.8,
                RelationType.SIBLING: 0.6,
                RelationType.NEXT: 0.5,
                RelationType.PREV: 0.5,
                RelationType.BEFORE: 0.4,
                RelationType.AFTER: 0.4,
                RelationType.CONCURRENT: 0.3,
            }


class IntelligentSearchEngine:
    """Intelligent search engine with relationship expansion capabilities."""

    def __init__(
        self,
        redis_client: RedisClient,
        rag_pipeline: Optional[RAGPipeline] = None,
        state_manager: Optional[StateManager] = None,
    ):
        """Initialize.

        Args:
            redis_client: Redis client
            rag_pipeline: RAG pipeline for semantic search
            state_manager: State manager for search history
        """
        self.redis = redis_client
        self.rag_pipeline = rag_pipeline
        self.state_manager = state_manager

    async def intelligent_search(
        self,
        query: str,
        strategy: Optional[SearchStrategy] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute intelligent search with relationship expansion.

        Args:
            query: Search query
            strategy: Search strategy configuration
            filters: Additional filters

        Returns:
            List of search results
        """
        strategy = strategy or SearchStrategy()
        filters = filters or {}

        try:
            # 1. Execute base search
            base_results = await self._execute_base_search(query, filters)
            if not base_results:
                logger.info("No base search results found")
                return []

            # 2. Expand search using relationships
            expanded_results = await self._expand_search_results(
                base_results, strategy
            )

            # 3. Apply temporal context enhancement
            temporal_results = await self._enhance_temporal_context(
                expanded_results, strategy
            )

            # 4. Apply reranking if enabled
            if strategy.enable_reranking:
                final_results = await self._rerank_results(
                    temporal_results, query, strategy
                )
            else:
                final_results = temporal_results

            # 5. Apply final filtering and limits
            filtered_results = self._apply_final_filters(
                final_results, strategy, filters
            )

            # 6. Log search for analytics
            await self._log_search_analytics(query, len(filtered_results), strategy)

            logger.info(
                f"Intelligent search completed: {len(filtered_results)} results for '{query}'"
            )
            return filtered_results

        except Exception as e:
            logger.error(f"Error in intelligent search: {e}")
            # Fallback to basic search
            return await self._fallback_search(query, filters)

    async def _execute_base_search(
        self, query: str, filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Execute base search."""
        results = []

        try:
            # Use RAG pipeline if available
            if self.rag_pipeline:
                # Semantic search using embeddings
                pipeline_results = await self.rag_pipeline.search(
                    query=query,
                    limit=filters.get("limit", 20),
                    similarity_threshold=filters.get("similarity_threshold", 0.7),
                )

                for result in pipeline_results:
                    search_result = SearchResult(
                        path=result.get("id", ""),
                        score=result.get("score", 0.0),
                        content=result.get("content", ""),
                        metadata=result.get("metadata", {}),
                    )
                    results.append(search_result)

            # Fallback: keyword search in Redis
            if not results:
                results = await self._keyword_search_redis(query, filters)

            return results

        except Exception as e:
            logger.error(f"Error in base search: {e}")
            return []

    async def _keyword_search_redis(
        self, query: str, filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Keyword search in Redis."""
        results = []

        try:
            # Simple pattern search
            search_patterns = [
                "json:memory:document:*",
                "stream:memory:conversation:*",
            ]

            for pattern in search_patterns:
                keys = await self.redis.keys(pattern)

                for key_bytes in keys[:50]:  # Limit for performance
                    key = key_bytes.decode("utf-8")

                    # Get content
                    content = await self._get_content_from_key(key)
                    if not content:
                        continue

                    # Simple keyword matching
                    if self._matches_query(content, query):
                        score = self._calculate_keyword_score(content, query)
                        if score > filters.get("min_score", 0.3):
                            results.append(
                                SearchResult(
                                    path=key,
                                    score=score,
                                    content=content[:500],  # Preview
                                    metadata={"search_type": "keyword"},
                                )
                            )

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results[: filters.get("limit", 20)]

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    async def _get_content_from_key(self, key: str) -> Optional[str]:
        """Get content from Redis key."""
        try:
            if "document" in key:
                # JSON document
                doc_data = await self.redis.json_get(key)
                if doc_data:
                    title = doc_data.get("title", "")
                    content = doc_data.get("content", "")
                    return f"{title}\n{content}" if title else content

            elif "conversation" in key:
                # Stream conversation
                messages = await self.redis.xrange(key, "-", "+", count=10)
                contents = []
                for _, data in messages:
                    if b"content" in data:
                        content = data[b"content"].decode("utf-8")
                        contents.append(content)
                return " ".join(contents)

            return None

        except Exception as e:
            logger.debug(f"Error getting content from {key}: {e}")
            return None

    def _matches_query(self, content: str, query: str) -> bool:
        """Check if content matches query."""
        content_lower = content.lower()
        query_lower = query.lower()

        # Split query into words
        query_words = re.findall(r"\b\w+\b", query_lower)
        if not query_words:
            return False

        # Check if all words are present
        for word in query_words:
            if word not in content_lower:
                return False

        return True

    def _calculate_keyword_score(self, content: str, query: str) -> float:
        """Calculate keyword matching score."""
        content_lower = content.lower()
        query_lower = query.lower()

        # Split into words
        content_words = re.findall(r"\b\w+\b", content_lower)
        query_words = re.findall(r"\b\w+\b", query_lower)

        if not query_words or not content_words:
            return 0.0

        # Calculate match ratio
        matches = 0
        for word in query_words:
            matches += content_words.count(word)

        # Basic TF score
        score = matches / len(content_words)

        # Boost for exact phrase matches
        if query_lower in content_lower:
            score *= 1.5

        return min(score, 1.0)

    async def _expand_search_results(
        self, base_results: List[SearchResult], strategy: SearchStrategy
    ) -> List[SearchResult]:
        """Expand search results using relationships."""
        if not base_results:
            return []

        expanded_results = list(base_results)
        visited_ids = {result.path for result in base_results}

        try:
            for base_result in base_results:
                # Get relationships for this result
                related_results = await self._find_related_chunks(
                    base_result.path, strategy, visited_ids
                )

                for related_result in related_results:
                    if related_result.path not in visited_ids:
                        expanded_results.append(related_result)
                        visited_ids.add(related_result.path)

            logger.info(
                f"Expanded {len(base_results)} to {len(expanded_results)} results"
            )
            return expanded_results

        except Exception as e:
            logger.error(f"Error expanding search results: {e}")
            return base_results

    async def _find_related_chunks(
        self,
        chunk_id: str,
        strategy: SearchStrategy,
        visited_ids: Set[str],
        current_depth: int = 0,
    ) -> List[SearchResult]:
        """Find chunks related to given chunk."""
        if current_depth >= strategy.max_expansion_depth:
            return []

        related_results = []

        try:
            # Get relationships from Redis
            relation_key = f"relation:chunk:{chunk_id}"
            relation_data = await self.redis.hgetall(relation_key)

            if not relation_data:
                return []

            # Process each relationship
            for field, value in relation_data.items():
                if not field.startswith("rel:"):
                    continue

                try:
                    import json

                    rel_dict = json.loads(value)
                    target_id = rel_dict["target_id"]

                    if target_id in visited_ids:
                        continue

                    # Get relationship type and weight
                    rel_type = RelationType(rel_dict["relation_type"])
                    base_weight = strategy.relation_weights.get(rel_type, 0.3)

                    # Apply depth penalty
                    depth_penalty = 0.8 ** current_depth
                    final_weight = base_weight * depth_penalty

                    if final_weight < strategy.min_relevance_score:
                        continue

                    # Get target content
                    target_content = await self._get_content_from_key(target_id)
                    if target_content:
                        related_result = SearchResult(
                            path=target_id,
                            score=final_weight,
                            content=target_content[:500],
                            metadata={
                                "relation_type": rel_type.value,
                                "expansion_depth": current_depth + 1,
                                "source_chunk": chunk_id,
                            },
                        )
                        related_results.append(related_result)

                        # Recursive expansion based on strategy
                        if strategy.expansion_strategy in [
                            ExpansionStrategy.DFS,
                            ExpansionStrategy.HYBRID,
                        ]:
                            recursive_results = await self._find_related_chunks(
                                target_id,
                                strategy,
                                visited_ids | {r.path for r in related_results},
                                current_depth + 1,
                            )
                            related_results.extend(recursive_results)

                except Exception as e:
                    logger.debug(f"Error processing relationship: {e}")
                    continue

            return related_results

        except Exception as e:
            logger.error(f"Error finding related chunks: {e}")
            return []

    async def _enhance_temporal_context(
        self, results: List[SearchResult], strategy: SearchStrategy
    ) -> List[SearchResult]:
        """Enhance results with temporal context."""
        if not results:
            return []

        enhanced_results = []

        try:
            for result in results:
                # Extract timestamp from path or metadata
                timestamp = self._extract_timestamp(result)

                if timestamp:
                    # Find temporally related content
                    temporal_boost = await self._calculate_temporal_boost(
                        timestamp, strategy
                    )

                    # Apply temporal boost to score
                    boosted_score = result.score * (1 + temporal_boost)
                    result.score = min(boosted_score, 1.0)

                    # Add temporal metadata
                    if not result.metadata:
                        result.metadata = {}
                    result.metadata["temporal_boost"] = temporal_boost
                    result.metadata["timestamp"] = timestamp.isoformat()

                enhanced_results.append(result)

            return enhanced_results

        except Exception as e:
            logger.error(f"Error enhancing temporal context: {e}")
            return results

    def _extract_timestamp(self, result: SearchResult) -> Optional[datetime]:
        """Extract timestamp from result."""
        # Try metadata first
        if result.metadata and "timestamp" in result.metadata:
            try:
                return datetime.fromisoformat(result.metadata["timestamp"])
            except ValueError:
                pass

        # Try extracting from path
        time_pattern = r"(\d{4}/\d{2}/\d{2}/\d{2}/\d{2}/\d{2})"
        match = re.search(time_pattern, result.path)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y/%m/%d/%H/%M/%S")
            except ValueError:
                pass

        return None

    async def _calculate_temporal_boost(
        self, timestamp: datetime, strategy: SearchStrategy
    ) -> float:
        """Calculate temporal relevance boost."""
        current_time = datetime.now()
        time_diff = abs((current_time - timestamp).total_seconds())

        # Define time windows with different boost values
        if time_diff <= 3600:  # 1 hour
            return 0.3
        elif time_diff <= 86400:  # 1 day
            return 0.2
        elif time_diff <= 604800:  # 1 week
            return 0.1
        else:
            return 0.0

    async def _rerank_results(
        self,
        results: List[SearchResult],
        original_query: str,
        strategy: SearchStrategy,
    ) -> List[SearchResult]:
        """Rerank results using multiple signals."""
        if not results:
            return []

        try:
            reranked_results = []

            for result in results:
                # Calculate composite score
                base_score = result.score

                # Query relevance (keyword matching)
                query_relevance = self._calculate_keyword_score(
                    result.content, original_query
                )

                # Relationship strength (from metadata)
                relationship_strength = 1.0  # Default
                if result.metadata and "relation_type" in result.metadata:
                    rel_type = RelationType(result.metadata["relation_type"])
                    relationship_strength = strategy.relation_weights.get(rel_type, 0.5)

                # Temporal relevance
                temporal_relevance = result.metadata.get("temporal_boost", 0.0)

                # Composite score
                composite_score = (
                    0.4 * base_score
                    + 0.3 * query_relevance
                    + 0.2 * relationship_strength
                    + 0.1 * temporal_relevance
                )

                result.score = min(composite_score, 1.0)
                reranked_results.append(result)

            # Sort by composite score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            return reranked_results

        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results

    def _apply_final_filters(
        self,
        results: List[SearchResult],
        strategy: SearchStrategy,
        filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """Apply final filters and limits."""
        # Filter by minimum relevance score
        filtered_results = [
            r for r in results if r.score >= strategy.min_relevance_score
        ]

        # Apply custom filters
        if "memory_type" in filters:
            memory_type = filters["memory_type"]
            filtered_results = [
                r
                for r in filtered_results
                if memory_type in r.path or memory_type == "all"
            ]

        # Apply result limit
        limit = min(strategy.max_results, filters.get("limit", strategy.max_results))
        return filtered_results[:limit]

    async def _log_search_analytics(
        self, query: str, result_count: int, strategy: SearchStrategy
    ) -> None:
        """Log search analytics."""
        try:
            analytics_key = f"analytics:search:{datetime.now().strftime('%Y%m%d')}"
            analytics_data = {
                "query": query,
                "result_count": result_count,
                "strategy": strategy.expansion_strategy.value,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis with TTL
            await self.redis.setex(
                analytics_key + f":{int(datetime.now().timestamp())}",
                86400,  # 1 day
                str(analytics_data),
            )

        except Exception as e:
            logger.debug(f"Error logging search analytics: {e}")

    async def _fallback_search(
        self, query: str, filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Fallback search when intelligent search fails."""
        try:
            logger.info("Using fallback search")
            return await self._keyword_search_redis(query, filters)
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []


# Main search action function
async def intelligent_search_action(
    redis_client: RedisClient,
    query: str,
    strategy: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    rag_pipeline: Optional[RAGPipeline] = None,
    state_manager: Optional[StateManager] = None,
) -> Dict[str, Any]:
    """Execute intelligent search action.

    Args:
        redis_client: Redis client
        query: Search query
        strategy: Search strategy options
        filters: Search filters
        rag_pipeline: RAG pipeline for semantic search
        state_manager: State manager

    Returns:
        Search results and metadata
    """
    try:
        # Create search engine
        engine = IntelligentSearchEngine(redis_client, rag_pipeline, state_manager)

        # Parse strategy
        search_strategy = SearchStrategy()
        if strategy:
            if "expansion_strategy" in strategy:
                search_strategy.expansion_strategy = ExpansionStrategy(
                    strategy["expansion_strategy"]
                )
            if "max_results" in strategy:
                search_strategy.max_results = strategy["max_results"]
            if "min_relevance_score" in strategy:
                search_strategy.min_relevance_score = strategy["min_relevance_score"]

        # Execute search
        results = await engine.intelligent_search(query, search_strategy, filters)

        # Convert to response format
        result_data = []
        for result in results:
            result_data.append({
                "path": result.path,
                "score": result.score,
                "content": result.content,
                "metadata": result.metadata or {},
            })

        return {
            "query": query,
            "results": result_data,
            "total_results": len(result_data),
            "strategy_used": search_strategy.expansion_strategy.value,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in intelligent search action: {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }