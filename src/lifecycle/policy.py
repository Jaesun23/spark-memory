"""TTL and lifecycle policies for memory management."""

import logging
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StorageTier(str, Enum):
    """Storage tiers for hierarchical memory storage."""

    HOT = "hot"  # Frequently accessed, high importance
    WARM = "warm"  # Moderate access, medium importance
    COLD = "cold"  # Rarely accessed, low importance
    ARCHIVE = "archive"  # Very old, very low importance


@dataclass
class TTLPolicy:
    """TTL policy configuration for different importance levels."""

    # Base TTL values by tier
    hot_ttl: timedelta = timedelta(days=365)  # 1 year
    warm_ttl: timedelta = timedelta(days=90)  # 3 months
    cold_ttl: timedelta = timedelta(days=30)  # 1 month
    archive_ttl: timedelta = timedelta(days=7)  # 1 week

    # Importance thresholds for tier assignment
    hot_threshold: float = 0.7
    warm_threshold: float = 0.4
    cold_threshold: float = 0.2

    # Dynamic adjustment factors
    access_boost_factor: float = 1.5  # Multiplier for frequently accessed
    reference_boost_factor: float = 1.3  # Multiplier for highly referenced
    user_rating_boost_factor: float = 2.0  # Multiplier for user-rated important

    def get_tier(self, importance_score: float) -> StorageTier:
        """Determine storage tier based on importance score."""
        if importance_score >= self.hot_threshold:
            return StorageTier.HOT
        elif importance_score >= self.warm_threshold:
            return StorageTier.WARM
        elif importance_score >= self.cold_threshold:
            return StorageTier.COLD
        else:
            return StorageTier.ARCHIVE

    def get_base_ttl(self, tier: StorageTier) -> timedelta:
        """Get base TTL for a storage tier."""
        ttl_map = {
            StorageTier.HOT: self.hot_ttl,
            StorageTier.WARM: self.warm_ttl,
            StorageTier.COLD: self.cold_ttl,
            StorageTier.ARCHIVE: self.archive_ttl,
        }
        return ttl_map[tier]

    def calculate_ttl(
        self,
        importance_score: float,
        access_frequency: float = 0.0,
        reference_count: float = 0.0,
        user_rating: Optional[float] = None,
    ) -> timedelta:
        """
        Calculate dynamic TTL based on importance and other factors.

        Args:
            importance_score: Overall importance score (0.0 to 1.0)
            access_frequency: Normalized access frequency
            reference_count: Normalized reference count
            user_rating: User-provided rating if available

        Returns:
            Calculated TTL duration
        """
        # Get base TTL from tier
        tier = self.get_tier(importance_score)
        base_ttl = self.get_base_ttl(tier)

        # Calculate boost multiplier
        multiplier = 1.0

        # Access frequency boost
        if access_frequency > 0.5:
            multiplier *= self.access_boost_factor

        # Reference count boost
        if reference_count > 0.3:
            multiplier *= self.reference_boost_factor

        # User rating boost (highest priority)
        if user_rating is not None and user_rating > 0.7:
            multiplier *= self.user_rating_boost_factor

        # Apply multiplier with reasonable bounds
        multiplier = min(5.0, max(0.5, multiplier))

        # Calculate final TTL
        final_ttl = base_ttl * multiplier

        # Cap at maximum 5 years
        max_ttl = timedelta(days=365 * 5)
        final_ttl = min(final_ttl, max_ttl)

        return final_ttl


@dataclass
class LifecyclePolicy:
    """Complete lifecycle management policy."""

    ttl_policy: TTLPolicy

    # Archival settings
    enable_archival: bool = True
    archive_after_days: int = 180  # Archive after 6 months of inactivity
    compress_archives: bool = True

    # Auto-cleanup settings
    enable_auto_cleanup: bool = True
    cleanup_check_interval: timedelta = timedelta(hours=24)
    batch_size: int = 100

    # Migration settings
    enable_tier_migration: bool = True
    migration_check_interval: timedelta = timedelta(hours=6)

    # Importance re-evaluation
    reevaluate_importance: bool = True
    reevaluation_interval: timedelta = timedelta(days=7)

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for storage."""
        return {
            "ttl_policy": {
                "hot_ttl": self.ttl_policy.hot_ttl.total_seconds(),
                "warm_ttl": self.ttl_policy.warm_ttl.total_seconds(),
                "cold_ttl": self.ttl_policy.cold_ttl.total_seconds(),
                "archive_ttl": self.ttl_policy.archive_ttl.total_seconds(),
                "hot_threshold": self.ttl_policy.hot_threshold,
                "warm_threshold": self.ttl_policy.warm_threshold,
                "cold_threshold": self.ttl_policy.cold_threshold,
            },
            "enable_archival": self.enable_archival,
            "archive_after_days": self.archive_after_days,
            "compress_archives": self.compress_archives,
            "enable_auto_cleanup": self.enable_auto_cleanup,
            "cleanup_check_interval": self.cleanup_check_interval.total_seconds(),
            "batch_size": self.batch_size,
            "enable_tier_migration": self.enable_tier_migration,
            "migration_check_interval": self.migration_check_interval.total_seconds(),
            "reevaluate_importance": self.reevaluate_importance,
            "reevaluation_interval": self.reevaluation_interval.total_seconds(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LifecyclePolicy":
        """Create policy from dictionary."""
        ttl_data = data["ttl_policy"]
        ttl_policy = TTLPolicy(
            hot_ttl=timedelta(seconds=ttl_data["hot_ttl"]),
            warm_ttl=timedelta(seconds=ttl_data["warm_ttl"]),
            cold_ttl=timedelta(seconds=ttl_data["cold_ttl"]),
            archive_ttl=timedelta(seconds=ttl_data["archive_ttl"]),
            hot_threshold=ttl_data["hot_threshold"],
            warm_threshold=ttl_data["warm_threshold"],
            cold_threshold=ttl_data["cold_threshold"],
        )

        return cls(
            ttl_policy=ttl_policy,
            enable_archival=data["enable_archival"],
            archive_after_days=data["archive_after_days"],
            compress_archives=data["compress_archives"],
            enable_auto_cleanup=data["enable_auto_cleanup"],
            cleanup_check_interval=timedelta(seconds=data["cleanup_check_interval"]),
            batch_size=data["batch_size"],
            enable_tier_migration=data["enable_tier_migration"],
            migration_check_interval=timedelta(
                seconds=data["migration_check_interval"]
            ),
            reevaluate_importance=data["reevaluate_importance"],
            reevaluation_interval=timedelta(seconds=data["reevaluation_interval"]),
        )

    @classmethod
    def default(cls) -> "LifecyclePolicy":
        """Create default lifecycle policy."""
        return cls(ttl_policy=TTLPolicy())

    @classmethod
    def aggressive(cls) -> "LifecyclePolicy":
        """Create aggressive cleanup policy for limited storage."""
        ttl_policy = TTLPolicy(
            hot_ttl=timedelta(days=90),
            warm_ttl=timedelta(days=30),
            cold_ttl=timedelta(days=7),
            archive_ttl=timedelta(days=1),
            hot_threshold=0.8,
            warm_threshold=0.5,
            cold_threshold=0.3,
        )

        return cls(
            ttl_policy=ttl_policy,
            archive_after_days=30,
            cleanup_check_interval=timedelta(hours=6),
            migration_check_interval=timedelta(hours=1),
            reevaluation_interval=timedelta(days=1),
        )

    @classmethod
    def conservative(cls) -> "LifecyclePolicy":
        """Create conservative policy that keeps data longer."""
        ttl_policy = TTLPolicy(
            hot_ttl=timedelta(days=365 * 5),  # 5 years
            warm_ttl=timedelta(days=365),  # 1 year
            cold_ttl=timedelta(days=180),  # 6 months
            archive_ttl=timedelta(days=90),  # 3 months
            hot_threshold=0.5,
            warm_threshold=0.2,
            cold_threshold=0.1,
        )

        return cls(
            ttl_policy=ttl_policy,
            archive_after_days=365,
            enable_auto_cleanup=False,  # Manual cleanup only
            migration_check_interval=timedelta(days=1),
            reevaluation_interval=timedelta(days=30),
        )
