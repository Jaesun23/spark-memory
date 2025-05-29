"""Importance evaluation system for memory lifecycle management."""

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from redis.asyncio import Redis

from ..memory.models import MemoryContent

logger = logging.getLogger(__name__)


@dataclass
class ImportanceScore:
    """Calculated importance score with detailed metrics."""

    overall_score: float  # 0.0 to 1.0
    access_frequency: float  # Normalized access frequency
    reference_count: float  # Normalized reference count
    user_rating: Optional[float] = None  # User-provided rating
    ai_score: Optional[float] = None  # AI-based evaluation
    recency_score: float = 0.0  # How recently accessed
    content_quality: float = 0.0  # Content-based quality metric

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "overall_score": self.overall_score,
            "access_frequency": self.access_frequency,
            "reference_count": self.reference_count,
            "user_rating": self.user_rating,
            "ai_score": self.ai_score,
            "recency_score": self.recency_score,
            "content_quality": self.content_quality,
        }


class ImportanceEvaluator:
    """Evaluates memory importance based on multiple factors."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.access_key_prefix = "memory:access"
        self.reference_key_prefix = "memory:refs"
        self.rating_key_prefix = "memory:rating"

        # Weights for importance calculation
        self.weights = {
            "access_frequency": 0.25,
            "reference_count": 0.20,
            "user_rating": 0.30,
            "ai_score": 0.15,
            "recency": 0.10,
        }

    async def evaluate(
        self,
        memory: MemoryContent,
        path: str,
        ai_evaluator: Optional[Any] = None,
    ) -> ImportanceScore:
        """
        Evaluate memory importance based on multiple factors.

        Args:
            memory: Memory content to evaluate
            path: Memory path
            ai_evaluator: Optional AI evaluator for content analysis

        Returns:
            ImportanceScore with detailed metrics
        """
        # Get access statistics
        access_freq = await self._get_access_frequency(path)
        ref_count = await self._get_reference_count(path)
        user_rating = await self._get_user_rating(path)
        recency = await self._calculate_recency_score(path)

        # AI-based evaluation if available
        ai_score = None
        if ai_evaluator:
            try:
                ai_score = await self._get_ai_score(memory, ai_evaluator)
            except Exception as e:
                logger.warning(f"AI evaluation failed: {e}")

        # Content quality assessment
        content_quality = self._assess_content_quality(memory)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            access_freq,
            ref_count,
            user_rating,
            ai_score,
            recency,
        )

        return ImportanceScore(
            overall_score=overall_score,
            access_frequency=access_freq,
            reference_count=ref_count,
            user_rating=user_rating,
            ai_score=ai_score,
            recency_score=recency,
            content_quality=content_quality,
        )

    async def track_access(self, path: str) -> None:
        """Track memory access for importance calculation."""
        access_key = f"{self.access_key_prefix}:{path}"

        # Increment access count
        await self.redis.hincrby(access_key, "count", 1)

        # Update last access timestamp
        await self.redis.hset(
            access_key, "last_access", datetime.now(timezone.utc).isoformat()
        )

        # Add to access time series (for frequency calculation)
        ts_key = f"{access_key}:ts"
        timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        await self.redis.zadd(ts_key, {str(timestamp): timestamp})

        # Keep only last 30 days of access data
        cutoff = timestamp - (30 * 24 * 60 * 60 * 1000)
        await self.redis.zremrangebyscore(ts_key, 0, cutoff)

    async def add_reference(self, from_path: str, to_path: str) -> None:
        """Track reference between memories."""
        ref_key = f"{self.reference_key_prefix}:{to_path}"
        await self.redis.sadd(ref_key, from_path)

    async def set_user_rating(self, path: str, rating: float) -> None:
        """Set user-provided importance rating (0.0 to 1.0)."""
        if not 0 <= rating <= 1:
            raise ValueError("Rating must be between 0.0 and 1.0")

        rating_key = f"{self.rating_key_prefix}:{path}"
        await self.redis.set(rating_key, rating)

    async def _get_access_frequency(self, path: str) -> float:
        """Calculate normalized access frequency."""
        access_key = f"{self.access_key_prefix}:{path}"
        ts_key = f"{access_key}:ts"

        # Get access count in last 30 days
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        month_ago = now - (30 * 24 * 60 * 60 * 1000)

        access_count = await self.redis.zcount(ts_key, month_ago, now)

        # Normalize using logarithmic scale (1-100 accesses → 0-1)
        if access_count == 0:
            return 0.0

        # log(1) = 0, log(100) ≈ 2, so we scale by 0.5
        normalized = min(1.0, math.log10(access_count + 1) * 0.5)
        return normalized

    async def _get_reference_count(self, path: str) -> float:
        """Calculate normalized reference count."""
        ref_key = f"{self.reference_key_prefix}:{path}"
        ref_count = await self.redis.scard(ref_key) or 0

        # Normalize (0-20 references → 0-1)
        normalized = min(1.0, ref_count / 20.0)
        return normalized

    async def _get_user_rating(self, path: str) -> Optional[float]:
        """Get user-provided rating if available."""
        rating_key = f"{self.rating_key_prefix}:{path}"
        rating = await self.redis.get(rating_key)

        if rating:
            return float(rating)
        return None

    async def _calculate_recency_score(self, path: str) -> float:
        """Calculate recency score based on last access."""
        access_key = f"{self.access_key_prefix}:{path}"
        last_access = await self.redis.hget(access_key, "last_access")

        if not last_access:
            # Never accessed, check creation time
            memory_key = f"memory:doc:{path}"
            doc = await self.redis.json().get(memory_key)
            if doc and "timestamp" in doc:
                last_access = doc["timestamp"]
            else:
                return 0.0

        # Calculate days since last access
        last_dt = datetime.fromisoformat(last_access.replace("Z", "+00:00"))
        days_ago = (datetime.now(timezone.utc) - last_dt).days

        # Exponential decay: 1.0 for today, ~0.5 for 7 days, ~0.1 for 30 days
        recency = math.exp(-days_ago / 10.0)
        return recency

    async def _get_ai_score(self, memory: MemoryContent, ai_evaluator: Any) -> float:
        """Use AI to evaluate content importance."""
        # This is a placeholder for AI evaluation
        # In real implementation, this would use LLM or other AI models
        # to analyze content quality, relevance, etc.

        # For now, return a simple heuristic based on content length
        content_str = str(memory.data)

        # Longer, more detailed content tends to be more important
        length_score = min(1.0, len(content_str) / 1000.0)

        # Check for structured data (indicates organized information)
        structure_score = 0.0
        if isinstance(memory.data, dict):
            structure_score = min(1.0, len(memory.data) / 10.0)

        return (length_score + structure_score) / 2.0

    def _assess_content_quality(self, memory: MemoryContent) -> float:
        """Assess content quality based on various metrics."""
        content = memory.data
        quality_score = 0.0

        # Check content completeness
        if isinstance(content, dict) and content:
            # Structured data - check for key fields
            fields = len(content)
            empty_fields = sum(1 for v in content.values() if not v)
            completeness = (fields - empty_fields) / fields if fields > 0 else 0
            quality_score += completeness * 0.4

        # Check content length (not too short, not too long)
        content_str = str(content)
        length = len(content_str)

        if 50 <= length <= 5000:  # Optimal length range
            quality_score += 0.3
        elif 10 <= length < 50 or 5000 < length <= 10000:
            quality_score += 0.15

        # Check for metadata completeness
        if memory.metadata:
            if hasattr(memory.metadata, "tags") and memory.metadata.tags:
                quality_score += 0.15
            if hasattr(memory.metadata, "source") and memory.metadata.source:
                quality_score += 0.15

        return min(1.0, quality_score)

    def _calculate_overall_score(
        self,
        access_freq: float,
        ref_count: float,
        user_rating: Optional[float],
        ai_score: Optional[float],
        recency: float,
    ) -> float:
        """Calculate weighted overall importance score."""
        scores = {
            "access_frequency": access_freq,
            "reference_count": ref_count,
            "recency": recency,
        }

        # Add optional scores if available
        if user_rating is not None:
            scores["user_rating"] = user_rating
        if ai_score is not None:
            scores["ai_score"] = ai_score

        # Adjust weights if some components are missing
        total_weight = sum(self.weights[k] for k in scores.keys())

        # Calculate weighted average
        overall = sum(scores[k] * self.weights[k] / total_weight for k in scores.keys())

        return min(1.0, overall)

    async def bulk_evaluate(
        self,
        memories: List[tuple[str, MemoryContent]],
        ai_evaluator: Optional[Any] = None,
    ) -> Dict[str, ImportanceScore]:
        """Evaluate multiple memories efficiently."""
        tasks = [self.evaluate(memory, path, ai_evaluator) for path, memory in memories]

        scores = await asyncio.gather(*tasks)

        return {path: score for (path, _), score in zip(memories, scores)}
