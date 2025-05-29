"""Memory merging strategies for consolidation."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import Importance, MemoryContent, MemoryMetadata, MemoryType
from .analyzer import SimilarityScore, SimilarityType


class MergeStrategy(str, Enum):
    """Strategies for merging memories."""

    KEEP_NEWEST = "keep_newest"  # Keep most recent, add references
    KEEP_OLDEST = "keep_oldest"  # Keep oldest, update with new info
    COMBINE = "combine"  # Combine all information
    HIERARCHICAL = "hierarchical"  # Create parent-child relationship
    IMPORTANCE_BASED = "importance"  # Keep most important


@dataclass
class MergeResult:
    """Result of a memory merge operation."""

    merged_content: MemoryContent
    merge_strategy: MergeStrategy
    original_keys: List[str]
    merge_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "merged_content": self.merged_content.to_dict(),
            "merge_strategy": self.merge_strategy,
            "original_keys": self.original_keys,
            "merge_metadata": self.merge_metadata,
        }


class MemoryMerger:
    """Merge similar or duplicate memories."""

    def merge(
        self,
        memories: List[Tuple[str, MemoryContent]],
        similarity_scores: List[SimilarityScore],
        strategy: Optional[MergeStrategy] = None,
    ) -> MergeResult:
        """Merge multiple memories into one.

        Args:
            memories: List of (key, content) tuples to merge
            similarity_scores: Similarity scores between memories
            strategy: Merge strategy to use (auto-detected if None)

        Returns:
            Merge result with consolidated memory
        """
        if not memories:
            raise ValueError("No memories to merge")

        if len(memories) == 1:
            return MergeResult(
                merged_content=memories[0][1],
                merge_strategy=MergeStrategy.KEEP_NEWEST,
                original_keys=[memories[0][0]],
                merge_metadata={"single_memory": True},
            )

        # Auto-detect strategy if not provided
        if strategy is None:
            strategy = self._determine_strategy(memories, similarity_scores)

        # Apply merge strategy
        if strategy == MergeStrategy.KEEP_NEWEST:
            return self._merge_keep_newest(memories)
        elif strategy == MergeStrategy.KEEP_OLDEST:
            return self._merge_keep_oldest(memories)
        elif strategy == MergeStrategy.COMBINE:
            return self._merge_combine(memories, similarity_scores)
        elif strategy == MergeStrategy.HIERARCHICAL:
            return self._merge_hierarchical(memories, similarity_scores)
        elif strategy == MergeStrategy.IMPORTANCE_BASED:
            return self._merge_by_importance(memories)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

    def _determine_strategy(
        self,
        memories: List[Tuple[str, MemoryContent]],
        similarity_scores: List[SimilarityScore],
    ) -> MergeStrategy:
        """Automatically determine best merge strategy."""
        # Check if all are duplicates
        all_duplicates = all(score.is_duplicate() for score in similarity_scores)
        if all_duplicates:
            return MergeStrategy.KEEP_NEWEST

        # Check similarity types
        similarity_types = {score.similarity_type for score in similarity_scores}

        # If mostly temporal, use hierarchical
        if SimilarityType.TEMPORAL in similarity_types:
            return MergeStrategy.HIERARCHICAL

        # If complement types, combine
        if SimilarityType.COMPLEMENT in similarity_types:
            return MergeStrategy.COMBINE

        # Default to importance-based
        return MergeStrategy.IMPORTANCE_BASED

    def _merge_keep_newest(
        self, memories: List[Tuple[str, MemoryContent]]
    ) -> MergeResult:
        """Keep the newest memory and add references to others."""
        # Sort by creation time
        sorted_memories = sorted(
            memories, key=lambda x: x[1].metadata.created_at, reverse=True
        )

        newest_key, newest_content = sorted_memories[0]
        other_keys = [key for key, _ in sorted_memories[1:]]

        # Create merged content
        merged_content = MemoryContent(
            type=newest_content.type, data=newest_content.data
        )

        # Update metadata
        merged_content.metadata = newest_content.metadata
        merged_content.metadata.references = other_keys
        merged_content.metadata.updated_at = datetime.now(timezone.utc)

        # Add merge info to metadata
        if not hasattr(merged_content.metadata, "merge_info"):
            merged_content.metadata.merge_info = {}

        merged_content.metadata.merge_info.update(
            {
                "strategy": MergeStrategy.KEEP_NEWEST,
                "merged_from": other_keys,
                "merge_count": len(memories),
                "merge_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return MergeResult(
            merged_content=merged_content,
            merge_strategy=MergeStrategy.KEEP_NEWEST,
            original_keys=[key for key, _ in memories],
            merge_metadata={"kept_key": newest_key, "discarded_keys": other_keys},
        )

    def _merge_keep_oldest(
        self, memories: List[Tuple[str, MemoryContent]]
    ) -> MergeResult:
        """Keep the oldest memory and update with new information."""
        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda x: x[1].metadata.created_at)

        oldest_key, oldest_content = sorted_memories[0]
        newer_memories = sorted_memories[1:]

        # Create merged content starting with oldest
        merged_content = MemoryContent(
            type=oldest_content.type,
            data=self._merge_data(
                oldest_content.data, [content.data for _, content in newer_memories]
            ),
        )

        # Update metadata
        merged_content.metadata = oldest_content.metadata
        merged_content.metadata.updated_at = datetime.now(timezone.utc)

        # Collect all tags
        all_tags = set(oldest_content.metadata.tags or [])
        for _, content in newer_memories:
            if content.metadata.tags:
                all_tags.update(content.metadata.tags)
        merged_content.metadata.tags = list(all_tags)

        # Update importance to highest
        max_importance = max(content.metadata.importance for _, content in memories)
        merged_content.metadata.importance = max_importance

        return MergeResult(
            merged_content=merged_content,
            merge_strategy=MergeStrategy.KEEP_OLDEST,
            original_keys=[key for key, _ in memories],
            merge_metadata={
                "base_key": oldest_key,
                "updated_from": [key for key, _ in newer_memories],
            },
        )

    def _merge_combine(
        self,
        memories: List[Tuple[str, MemoryContent]],
        similarity_scores: List[SimilarityScore],
    ) -> MergeResult:
        """Combine all information from memories."""
        # Determine primary memory (highest importance or newest)
        primary_idx = self._find_primary_memory(memories)
        primary_key, primary_content = memories[primary_idx]

        # Combine data from all memories
        all_data = [content.data for _, content in memories]
        combined_data = self._combine_data(all_data, primary_content.type)

        # Create merged content
        merged_content = MemoryContent(type=primary_content.type, data=combined_data)

        # Combine metadata
        merged_content.metadata = self._combine_metadata(
            [content.metadata for _, content in memories]
        )

        # Add combination info
        merged_content.metadata.combination_info = {
            "primary_source": primary_key,
            "combined_sources": [key for key, _ in memories],
            "similarity_scores": [score.overall for score in similarity_scores],
        }

        return MergeResult(
            merged_content=merged_content,
            merge_strategy=MergeStrategy.COMBINE,
            original_keys=[key for key, _ in memories],
            merge_metadata={
                "combination_method": "full_merge",
                "primary_key": primary_key,
            },
        )

    def _merge_hierarchical(
        self,
        memories: List[Tuple[str, MemoryContent]],
        similarity_scores: List[SimilarityScore],
    ) -> MergeResult:
        """Create hierarchical relationship between memories."""
        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda x: x[1].metadata.created_at)

        # Create summary/parent content
        summary_data = self._create_summary(sorted_memories)

        merged_content = MemoryContent(type=MemoryType.DOCUMENT, data=summary_data)

        # Set metadata
        merged_content.metadata = MemoryMetadata(
            memory_type=MemoryType.DOCUMENT,
            importance=Importance.HIGH,
            tags=["consolidated", "summary"],
        )

        # Add child references
        merged_content.metadata.children = [key for key, _ in sorted_memories]
        merged_content.metadata.hierarchy_info = {
            "level": "parent",
            "child_count": len(memories),
            "time_range": {
                "start": sorted_memories[0][1].metadata.created_at.isoformat(),
                "end": sorted_memories[-1][1].metadata.created_at.isoformat(),
            },
        }

        return MergeResult(
            merged_content=merged_content,
            merge_strategy=MergeStrategy.HIERARCHICAL,
            original_keys=[key for key, _ in memories],
            merge_metadata={
                "hierarchy_type": "temporal_summary",
                "child_keys": [key for key, _ in sorted_memories],
            },
        )

    def _merge_by_importance(
        self, memories: List[Tuple[str, MemoryContent]]
    ) -> MergeResult:
        """Merge based on importance scores."""
        # Sort by importance
        sorted_memories = sorted(
            memories,
            key=lambda x: (
                x[1].metadata.importance.value
                if hasattr(x[1].metadata.importance, "value")
                else 0.5
            ),
            reverse=True,
        )

        most_important_key, most_important_content = sorted_memories[0]

        # Enhance most important with data from others
        enhanced_data = self._enhance_data(
            most_important_content.data,
            [content.data for _, content in sorted_memories[1:]],
        )

        merged_content = MemoryContent(
            type=most_important_content.type, data=enhanced_data
        )

        # Keep metadata from most important
        merged_content.metadata = most_important_content.metadata
        merged_content.metadata.enhanced_from = [key for key, _ in sorted_memories[1:]]

        return MergeResult(
            merged_content=merged_content,
            merge_strategy=MergeStrategy.IMPORTANCE_BASED,
            original_keys=[key for key, _ in memories],
            merge_metadata={
                "primary_key": most_important_key,
                "importance_scores": {
                    key: (
                        content.metadata.importance.value
                        if hasattr(content.metadata.importance, "value")
                        else 0.5
                    )
                    for key, content in memories
                },
            },
        )

    def _find_primary_memory(self, memories: List[Tuple[str, MemoryContent]]) -> int:
        """Find the primary memory index based on importance and recency."""
        scores = []

        for i, (_, content) in enumerate(memories):
            # Importance score (0-1)
            imp_score = (
                content.metadata.importance.value
                if hasattr(content.metadata.importance, "value")
                else 0.5
            )

            # Recency score (0-1)
            created_at = content.metadata.created_at
            now = datetime.now(timezone.utc)
            age_days = (now - created_at).days
            recency_score = max(0, 1 - (age_days / 365))  # Linear decay over a year

            # Combined score (70% importance, 30% recency)
            combined_score = 0.7 * imp_score + 0.3 * recency_score
            scores.append((i, combined_score))

        # Return index with highest score
        return max(scores, key=lambda x: x[1])[0]

    def _merge_data(self, base_data: Any, other_data: List[Any]) -> Any:
        """Merge data objects together."""
        if isinstance(base_data, dict):
            merged = base_data.copy()

            for data in other_data:
                if isinstance(data, dict):
                    # Add new keys, update existing with newer values
                    for key, value in data.items():
                        if key not in merged or value is not None:
                            merged[key] = value

            return merged

        elif isinstance(base_data, list):
            # Combine lists, removing duplicates while preserving order
            seen = set()
            merged = []

            for item in base_data + sum(other_data, []):
                # For hashable items
                if isinstance(item, (str, int, float, tuple)):
                    if item not in seen:
                        seen.add(item)
                        merged.append(item)
                else:
                    # For unhashable items, just append
                    merged.append(item)

            return merged

        else:
            # For simple types, just return the base
            return base_data

    def _combine_data(self, all_data: List[Any], memory_type: MemoryType) -> Any:
        """Combine data from multiple sources."""
        if memory_type == MemoryType.CONVERSATION:
            # For conversations, create a combined thread
            messages = []
            for data in all_data:
                if isinstance(data, list):
                    messages.extend(data)
                elif isinstance(data, dict) and "messages" in data:
                    messages.extend(data["messages"])
                else:
                    messages.append(data)

            return {
                "type": "combined_conversation",
                "messages": messages,
                "source_count": len(all_data),
            }

        elif memory_type == MemoryType.DOCUMENT:
            # For documents, create sections
            if all(isinstance(d, dict) for d in all_data):
                # Merge dicts
                combined = {}
                for data in all_data:
                    for key, value in data.items():
                        if key in combined:
                            # Create list if multiple values
                            if not isinstance(combined[key], list):
                                combined[key] = [combined[key]]
                            combined[key].append(value)
                        else:
                            combined[key] = value
                return combined
            else:
                # Create sections
                return {"type": "combined_document", "sections": all_data}

        else:
            # Default: create array
            return all_data if len(all_data) > 1 else all_data[0]

    def _combine_metadata(self, all_metadata: List[MemoryMetadata]) -> MemoryMetadata:
        """Combine metadata from multiple memories."""
        # Start with first metadata
        combined = all_metadata[0]

        # Combine tags
        all_tags = set()
        for meta in all_metadata:
            if meta.tags:
                all_tags.update(meta.tags)
        combined.tags = list(all_tags)

        # Use highest importance
        combined.importance = max(meta.importance for meta in all_metadata)

        # Use earliest creation, latest update
        combined.created_at = min(meta.created_at for meta in all_metadata)
        combined.updated_at = datetime.now(timezone.utc)

        # Combine sources
        all_sources = []
        for meta in all_metadata:
            if meta.source:
                all_sources.append(meta.source)
        if all_sources:
            combined.source = "; ".join(set(all_sources))

        return combined

    def _create_summary(
        self, memories: List[Tuple[str, MemoryContent]]
    ) -> Dict[str, Any]:
        """Create a summary of multiple memories."""
        summary = {
            "type": "memory_summary",
            "memory_count": len(memories),
            "time_range": {
                "start": memories[0][1].metadata.created_at.isoformat(),
                "end": memories[-1][1].metadata.created_at.isoformat(),
            },
            "contents": [],
        }

        for key, content in memories:
            summary["contents"].append(
                {
                    "key": key,
                    "created_at": content.metadata.created_at.isoformat(),
                    "type": content.type.value,
                    "preview": self._get_preview(content.data),
                }
            )

        return summary

    def _enhance_data(self, base_data: Any, enhancement_data: List[Any]) -> Any:
        """Enhance base data with additional information."""
        if isinstance(base_data, dict):
            enhanced = base_data.copy()

            # Add enhancement section
            enhanced["_enhancements"] = []

            for data in enhancement_data:
                if isinstance(data, dict):
                    # Extract unique information
                    unique_info = {
                        k: v
                        for k, v in data.items()
                        if k not in enhanced or enhanced[k] != v
                    }
                    if unique_info:
                        enhanced["_enhancements"].append(unique_info)
                else:
                    enhanced["_enhancements"].append(data)

            return enhanced

        else:
            # For non-dict data, create a wrapper
            return {"primary": base_data, "enhancements": enhancement_data}

    def _get_preview(self, data: Any, max_length: int = 100) -> str:
        """Get a preview of data content."""
        if isinstance(data, str):
            preview = data[:max_length]
            if len(data) > max_length:
                preview += "..."
            return preview
        elif isinstance(data, dict):
            # Try to get content from common fields
            for field in ["content", "text", "message", "title", "summary"]:
                if field in data:
                    return self._get_preview(data[field], max_length)
            # Fallback to JSON preview
            json_str = json.dumps(data, ensure_ascii=False)
            return self._get_preview(json_str, max_length)
        else:
            return self._get_preview(str(data), max_length)
