"""Memory consolidation module for detecting and merging similar/duplicate memories."""

from .analyzer import SimilarityAnalyzer, SimilarityType
from .consolidator import ConsolidationConfig, MemoryConsolidator
from .merger import MemoryMerger, MergeStrategy

__all__ = [
    "SimilarityAnalyzer",
    "SimilarityType",
    "MemoryMerger",
    "MergeStrategy",
    "MemoryConsolidator",
    "ConsolidationConfig",
]
