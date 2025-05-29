"""Similarity analysis for memory consolidation."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import MemoryContent, MemoryType


class SimilarityType(str, Enum):
    """Types of similarity between memories."""

    DUPLICATE = "duplicate"  # Exact or near-exact match
    SIMILAR = "similar"  # Similar content with variations
    COMPLEMENT = "complement"  # Related content that complements
    TEMPORAL = "temporal"  # Time-based proximity
    STRUCTURAL = "structural"  # Similar structure/format


@dataclass
class SimilarityScore:
    """Detailed similarity scoring between two memories."""

    overall: float
    text_similarity: float
    structural_similarity: float
    temporal_proximity: float
    metadata_match: float
    similarity_type: SimilarityType

    def is_duplicate(self, threshold: float = 0.95) -> bool:
        """Check if similarity indicates a duplicate."""
        return self.overall >= threshold

    def is_similar(self, threshold: float = 0.7) -> bool:
        """Check if similarity indicates similar content."""
        return self.overall >= threshold

    def should_merge(self, min_threshold: float = 0.7) -> bool:
        """Determine if memories should be merged."""
        return self.overall >= min_threshold


class SimilarityAnalyzer:
    """Analyze similarity between memories for consolidation."""

    def __init__(self):
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
        }

    def analyze(
        self,
        memory1: MemoryContent,
        memory2: MemoryContent,
        vector_similarity: Optional[float] = None,
    ) -> SimilarityScore:
        """Analyze similarity between two memories.

        Args:
            memory1: First memory
            memory2: Second memory
            vector_similarity: Pre-computed vector similarity if available

        Returns:
            Detailed similarity score
        """
        # Extract text content
        text1 = self._extract_text(memory1.data)
        text2 = self._extract_text(memory2.data)

        # Calculate different similarity components
        text_sim = self._text_similarity(text1, text2)
        struct_sim = self._structural_similarity(memory1.data, memory2.data)
        temporal_sim = self._temporal_proximity(
            memory1.metadata.created_at, memory2.metadata.created_at
        )
        metadata_sim = self._metadata_similarity(memory1.metadata, memory2.metadata)

        # Use vector similarity if provided, otherwise use text similarity
        content_similarity = (
            vector_similarity if vector_similarity is not None else text_sim
        )

        # Calculate weighted overall score
        weights = {"content": 0.5, "structure": 0.2, "temporal": 0.1, "metadata": 0.2}

        overall = (
            weights["content"] * content_similarity
            + weights["structure"] * struct_sim
            + weights["temporal"] * temporal_sim
            + weights["metadata"] * metadata_sim
        )

        # Determine similarity type
        similarity_type = self._determine_type(
            overall, text_sim, struct_sim, temporal_sim
        )

        return SimilarityScore(
            overall=overall,
            text_similarity=text_sim,
            structural_similarity=struct_sim,
            temporal_proximity=temporal_sim,
            metadata_match=metadata_sim,
            similarity_type=similarity_type,
        )

    def _extract_text(self, data: Any) -> str:
        """Extract text content from memory data."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # Extract from common fields
            text_parts = []
            for field in [
                "content",
                "text",
                "message",
                "description",
                "title",
                "summary",
            ]:
                if field in data:
                    text_parts.append(str(data[field]))

            # If no common fields, convert entire dict
            if not text_parts:
                text_parts.append(json.dumps(data, ensure_ascii=False))

            return " ".join(text_parts)
        else:
            return str(data)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard similarity of word sets."""
        # Tokenize and normalize
        words1 = self._tokenize(text1.lower())
        words2 = self._tokenize(text2.lower())

        # Remove stop words
        words1 = {w for w in words1 if w not in self.stop_words}
        words2 = {w for w in words2 if w not in self.stop_words}

        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _tokenize(self, text: str) -> set:
        """Tokenize text into words."""
        # Simple word tokenization
        words = re.findall(r"\b\w+\b", text)
        return set(words)

    def _structural_similarity(self, data1: Any, data2: Any) -> float:
        """Calculate structural similarity for dict/list data."""
        # Same type check
        if type(data1) != type(data2):
            return 0.0

        if isinstance(data1, dict) and isinstance(data2, dict):
            # Compare keys
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())

            if not keys1 and not keys2:
                return 1.0

            key_similarity = len(keys1 & keys2) / len(keys1 | keys2)

            # Compare value types for common keys
            common_keys = keys1 & keys2
            if common_keys:
                type_matches = sum(
                    1 for k in common_keys if type(data1[k]) == type(data2[k])
                )
                type_similarity = type_matches / len(common_keys)

                return (key_similarity + type_similarity) / 2

            return key_similarity

        elif isinstance(data1, list) and isinstance(data2, list):
            # Compare list lengths and types
            if not data1 and not data2:
                return 1.0

            len_ratio = min(len(data1), len(data2)) / max(len(data1), len(data2))

            # Sample type similarity
            sample_size = min(5, len(data1), len(data2))
            if sample_size > 0:
                type_matches = sum(
                    1 for i in range(sample_size) if type(data1[i]) == type(data2[i])
                )
                type_similarity = type_matches / sample_size

                return (len_ratio + type_similarity) / 2

            return len_ratio

        else:
            # For other types, just check equality
            return 1.0 if data1 == data2 else 0.0

    def _temporal_proximity(self, time1: datetime, time2: datetime) -> float:
        """Calculate temporal proximity score."""
        # Calculate time difference
        time_diff = abs((time1 - time2).total_seconds())

        # Define proximity thresholds (in seconds)
        thresholds = [
            (60, 1.0),  # Within 1 minute
            (300, 0.9),  # Within 5 minutes
            (3600, 0.7),  # Within 1 hour
            (86400, 0.5),  # Within 1 day
            (604800, 0.3),  # Within 1 week
            (2592000, 0.1),  # Within 30 days
        ]

        for threshold, score in thresholds:
            if time_diff <= threshold:
                return score

        return 0.0

    def _metadata_similarity(self, meta1: Any, meta2: Any) -> float:
        """Calculate metadata similarity."""
        scores = []

        # Memory type
        if hasattr(meta1, "memory_type") and hasattr(meta2, "memory_type"):
            scores.append(1.0 if meta1.memory_type == meta2.memory_type else 0.0)

        # Tags
        if hasattr(meta1, "tags") and hasattr(meta2, "tags"):
            if meta1.tags and meta2.tags:
                tags1 = set(meta1.tags)
                tags2 = set(meta2.tags)
                tag_sim = (
                    len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 1.0
                )
                scores.append(tag_sim)

        # Importance
        if hasattr(meta1, "importance") and hasattr(meta2, "importance"):
            # Convert importance to numeric if needed
            imp1 = meta1.importance.value if hasattr(meta1.importance, "value") else 0.5
            imp2 = meta2.importance.value if hasattr(meta2.importance, "value") else 0.5
            imp_diff = abs(imp1 - imp2)
            scores.append(1.0 - imp_diff)

        # Source
        if hasattr(meta1, "source") and hasattr(meta2, "source"):
            if meta1.source and meta2.source:
                scores.append(1.0 if meta1.source == meta2.source else 0.0)

        return sum(scores) / len(scores) if scores else 0.5

    def _determine_type(
        self, overall: float, text_sim: float, struct_sim: float, temporal_sim: float
    ) -> SimilarityType:
        """Determine the type of similarity."""
        # Duplicate: Very high overall similarity
        if overall >= 0.95:
            return SimilarityType.DUPLICATE

        # Temporal: High temporal proximity but lower content similarity
        if temporal_sim >= 0.7 and text_sim < 0.5:
            return SimilarityType.TEMPORAL

        # Structural: High structural similarity
        if struct_sim >= 0.8 and text_sim < 0.5:
            return SimilarityType.STRUCTURAL

        # Complement: Moderate similarity with high temporal proximity
        if 0.4 <= overall < 0.7 and temporal_sim >= 0.5:
            return SimilarityType.COMPLEMENT

        # Similar: Default for moderate to high similarity
        return SimilarityType.SIMILAR

    def find_similar_groups(
        self,
        memories: List[Tuple[str, MemoryContent]],
        similarity_scores: Optional[Dict[Tuple[str, str], float]] = None,
        threshold: float = 0.7,
    ) -> List[List[str]]:
        """Group memories by similarity.

        Args:
            memories: List of (key, content) tuples
            similarity_scores: Pre-computed similarity scores
            threshold: Minimum similarity for grouping

        Returns:
            List of groups, each containing memory keys
        """
        if len(memories) <= 1:
            return [[memories[0][0]]] if memories else []

        # Build similarity matrix
        keys = [key for key, _ in memories]
        memory_map = {key: content for key, content in memories}

        # Initialize groups
        groups = []
        processed = set()

        for i, key1 in enumerate(keys):
            if key1 in processed:
                continue

            group = [key1]
            processed.add(key1)

            for j, key2 in enumerate(keys[i + 1 :], i + 1):
                if key2 in processed:
                    continue

                # Get similarity score
                if similarity_scores and (key1, key2) in similarity_scores:
                    score = similarity_scores[(key1, key2)]
                else:
                    # Calculate similarity
                    sim_result = self.analyze(memory_map[key1], memory_map[key2])
                    score = sim_result.overall

                if score >= threshold:
                    group.append(key2)
                    processed.add(key2)

            groups.append(group)

        return groups
