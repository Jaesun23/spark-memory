"""Main memory consolidation orchestrator."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..embeddings import VectorStore
from ..memory.models import MemoryContent, SearchQuery, SearchType
from .analyzer import SimilarityAnalyzer, SimilarityScore
from .merger import MemoryMerger, MergeResult, MergeStrategy

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine


logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""

    similarity_threshold: float = 0.7
    duplicate_threshold: float = 0.95
    min_group_size: int = 2
    max_group_size: int = 10
    time_window_hours: int = 24
    auto_merge: bool = True
    preserve_originals: bool = True
    merge_strategy: Optional[MergeStrategy] = None


@dataclass
class ConsolidationResult:
    """Result of consolidation process."""

    groups_found: int
    memories_processed: int
    merges_completed: int
    merge_results: List[MergeResult]
    errors: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "groups_found": self.groups_found,
            "memories_processed": self.memories_processed,
            "merges_completed": self.merges_completed,
            "merge_results": [r.to_dict() for r in self.merge_results],
            "errors": self.errors,
            "summary": {
                "success_rate": (
                    self.merges_completed / self.groups_found
                    if self.groups_found > 0
                    else 0
                ),
                "consolidation_ratio": (
                    1 - (self.merges_completed / self.memories_processed)
                    if self.memories_processed > 0
                    else 0
                ),
            },
        }


class MemoryConsolidator:
    """Orchestrate memory consolidation process."""

    def __init__(
        self,
        memory_engine: "MemoryEngine",
        vector_store: Optional[VectorStore] = None,
        config: Optional[ConsolidationConfig] = None,
    ):
        self.engine = memory_engine
        self.vector_store = vector_store
        self.config = config or ConsolidationConfig()
        self.analyzer = SimilarityAnalyzer()
        self.merger = MemoryMerger()

    async def consolidate_by_path(
        self, path_prefix: str, time_window: Optional[timedelta] = None
    ) -> ConsolidationResult:
        """Consolidate memories under a specific path.

        Args:
            path_prefix: Path prefix to search under
            time_window: Optional time window for consolidation

        Returns:
            Consolidation result
        """
        # Search for memories in the path
        search_query = SearchQuery(
            query="",
            search_type=SearchType.KEYWORD,
            filters={"paths": [path_prefix]},
            options={"limit": 1000},
        )

        if time_window:
            end_time = datetime.now()
            start_time = end_time - time_window
            search_query.filters["start_time"] = start_time
            search_query.filters["end_time"] = end_time

        # Get memories
        search_results = await self.engine.execute(
            action="search",
            paths=[path_prefix],
            content="",
            options=search_query.to_dict(),
        )

        if not search_results:
            return ConsolidationResult(
                groups_found=0,
                memories_processed=0,
                merges_completed=0,
                merge_results=[],
                errors=[],
            )

        # Convert to list of (key, content) tuples
        memories = [(result.key, result.content) for result in search_results]

        return await self._consolidate_memories(memories)

    async def consolidate_duplicates(
        self, content: Any, path_prefix: Optional[str] = None
    ) -> ConsolidationResult:
        """Find and consolidate duplicate memories.

        Args:
            content: Content to check for duplicates
            path_prefix: Optional path prefix to limit search

        Returns:
            Consolidation result
        """
        if not self.vector_store:
            logger.warning("Vector store not available for duplicate detection")
            return ConsolidationResult(
                groups_found=0,
                memories_processed=0,
                merges_completed=0,
                merge_results=[],
                errors=[{"error": "Vector store not available"}],
            )

        # Find duplicates using vector similarity
        duplicates = await self.vector_store.find_duplicates(
            content=content,
            threshold=self.config.duplicate_threshold,
            limit=self.config.max_group_size,
        )

        if not duplicates:
            return ConsolidationResult(
                groups_found=0,
                memories_processed=0,
                merges_completed=0,
                merge_results=[],
                errors=[],
            )

        # Get full memory content for duplicates
        memories = []
        for path, similarity in duplicates:
            try:
                result = await self.engine.execute(
                    action="get", paths=[path], content=None, options={}
                )
                if result:
                    memories.append((path, MemoryContent.from_dict(result)))
            except Exception as e:
                logger.error(f"Failed to get memory {path}: {e}")

        # Consolidate if we have enough duplicates
        if len(memories) >= self.config.min_group_size:
            return await self._consolidate_memories(
                memories, strategy=MergeStrategy.KEEP_NEWEST
            )

        return ConsolidationResult(
            groups_found=0,
            memories_processed=len(memories),
            merges_completed=0,
            merge_results=[],
            errors=[],
        )

    async def consolidate_temporal(
        self, path_prefix: str, time_buckets: List[timedelta]
    ) -> ConsolidationResult:
        """Consolidate memories by temporal buckets.

        Args:
            path_prefix: Path prefix to search under
            time_buckets: List of time bucket sizes

        Returns:
            Consolidation result
        """
        all_results = []
        total_processed = 0

        for bucket_size in time_buckets:
            # Process each time bucket
            result = await self.consolidate_by_path(
                path_prefix=path_prefix, time_window=bucket_size
            )

            all_results.extend(result.merge_results)
            total_processed += result.memories_processed

        return ConsolidationResult(
            groups_found=len(all_results),
            memories_processed=total_processed,
            merges_completed=len(all_results),
            merge_results=all_results,
            errors=[],
        )

    async def _consolidate_memories(
        self,
        memories: List[Tuple[str, MemoryContent]],
        strategy: Optional[MergeStrategy] = None,
    ) -> ConsolidationResult:
        """Internal method to consolidate a list of memories."""
        if len(memories) < self.config.min_group_size:
            return ConsolidationResult(
                groups_found=0,
                memories_processed=len(memories),
                merges_completed=0,
                merge_results=[],
                errors=[],
            )

        # Calculate similarity scores if vector store available
        similarity_scores = {}
        if self.vector_store:
            for i, (key1, content1) in enumerate(memories):
                for j, (key2, content2) in enumerate(memories[i + 1 :], i + 1):
                    try:
                        # Get vector similarity
                        text1 = self.vector_store.embeddings.prepare_text_for_embedding(
                            content1.data
                        )
                        text2 = self.vector_store.embeddings.prepare_text_for_embedding(
                            content2.data
                        )

                        # Calculate embeddings
                        emb1, emb2 = await self.vector_store.embeddings.embed_batch(
                            [text1, text2]
                        )

                        # Calculate cosine similarity
                        import numpy as np

                        similarity = float(
                            np.dot(emb1, emb2)
                            / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        )

                        similarity_scores[(key1, key2)] = similarity
                    except Exception as e:
                        logger.warning(f"Failed to calculate vector similarity: {e}")

        # Find similar groups
        groups = self.analyzer.find_similar_groups(
            memories=memories,
            similarity_scores=similarity_scores,
            threshold=self.config.similarity_threshold,
        )

        # Filter groups by size
        valid_groups = [
            group
            for group in groups
            if self.config.min_group_size <= len(group) <= self.config.max_group_size
        ]

        # Merge each group
        merge_results = []
        errors = []

        for group in valid_groups:
            try:
                # Get memories for this group
                group_memories = [
                    (key, content) for key, content in memories if key in group
                ]

                # Calculate similarity scores for this group
                group_scores = []
                for i in range(len(group_memories) - 1):
                    key1 = group_memories[i][0]
                    key2 = group_memories[i + 1][0]

                    if (key1, key2) in similarity_scores:
                        vector_sim = similarity_scores[(key1, key2)]
                    else:
                        vector_sim = None

                    score = self.analyzer.analyze(
                        group_memories[i][1], group_memories[i + 1][1], vector_sim
                    )
                    group_scores.append(score)

                # Merge the group
                merge_result = self.merger.merge(
                    memories=group_memories,
                    similarity_scores=group_scores,
                    strategy=strategy or self.config.merge_strategy,
                )

                # Save merged memory if auto_merge is enabled
                if self.config.auto_merge:
                    # Generate new path for merged memory
                    merged_path = self._generate_merged_path([key for key in group])

                    # Save merged memory
                    await self.engine.execute(
                        action="save",
                        paths=merged_path.split("/"),
                        content=merge_result.merged_content.data,
                        options={
                            "tags": ["consolidated", "merged"],
                            "importance": merge_result.merged_content.metadata.importance.value,
                        },
                    )

                    # Delete originals if not preserving
                    if not self.config.preserve_originals:
                        for key in group:
                            try:
                                await self.engine.execute(
                                    action="delete",
                                    paths=key.split(":"),
                                    content=None,
                                    options={},
                                )
                            except Exception as e:
                                logger.error(f"Failed to delete original {key}: {e}")

                merge_results.append(merge_result)

            except Exception as e:
                logger.error(f"Failed to merge group {group}: {e}")
                errors.append({"group": group, "error": str(e)})

        return ConsolidationResult(
            groups_found=len(valid_groups),
            memories_processed=len(memories),
            merges_completed=len(merge_results),
            merge_results=merge_results,
            errors=errors,
        )

    def _generate_merged_path(self, original_keys: List[str]) -> str:
        """Generate path for merged memory."""
        # Extract common prefix from keys
        if not original_keys:
            return "consolidated/unknown"

        # Find common prefix
        first_parts = original_keys[0].split(":")
        common_parts = []

        for i, part in enumerate(first_parts):
            if all(
                key.split(":")[i] == part
                for key in original_keys[1:]
                if i < len(key.split(":"))
            ):
                common_parts.append(part)
            else:
                break

        # Create consolidated path
        if common_parts:
            base_path = "/".join(common_parts[2:])  # Skip prefix and type
            return (
                f"{base_path}/consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            return f"consolidated/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
