"""Lifecycle manager for automated memory management."""

import asyncio
import gzip
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from redis.asyncio import Redis

from ..memory.models import Importance, MemoryContent, MemoryMetadata, MemoryType
from .evaluator import ImportanceEvaluator, ImportanceScore
from .policy import LifecyclePolicy, StorageTier, TTLPolicy

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manages memory lifecycle with importance-based TTL and tiered storage."""

    def __init__(
        self,
        redis_client: Redis,
        policy: Optional[LifecyclePolicy] = None,
        ai_evaluator: Optional[Any] = None,
    ):
        self.redis = redis_client
        self.policy = policy or LifecyclePolicy.default()
        self.evaluator = ImportanceEvaluator(redis_client)
        self.ai_evaluator = ai_evaluator

        # Key prefixes
        self.lifecycle_prefix = "memory:lifecycle"
        self.tier_prefix = "memory:tier"
        self.archive_prefix = "memory:archive"

        # Background task handles
        self._tasks: List[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start background lifecycle management tasks."""
        if self._running:
            return

        self._running = True

        # Start background tasks based on policy
        if self.policy.enable_auto_cleanup:
            task = asyncio.create_task(self._cleanup_loop())
            self._tasks.append(task)

        if self.policy.enable_tier_migration:
            task = asyncio.create_task(self._migration_loop())
            self._tasks.append(task)

        if self.policy.reevaluate_importance:
            task = asyncio.create_task(self._reevaluation_loop())
            self._tasks.append(task)

        logger.info("Lifecycle manager started")

    async def stop(self) -> None:
        """Stop all background tasks."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        logger.info("Lifecycle manager stopped")

    async def evaluate_and_set_ttl(
        self,
        path: str,
        memory: MemoryContent,
    ) -> Tuple[ImportanceScore, timedelta]:
        """
        Evaluate memory importance and set appropriate TTL.

        Returns:
            Tuple of (ImportanceScore, calculated TTL)
        """
        # Evaluate importance
        score = await self.evaluator.evaluate(memory, path, self.ai_evaluator)

        # Calculate TTL
        ttl = self.policy.ttl_policy.calculate_ttl(
            score.overall_score,
            score.access_frequency,
            score.reference_count,
            score.user_rating,
        )

        # Store lifecycle metadata
        await self._store_lifecycle_metadata(path, score, ttl)

        # Set Redis TTL
        memory_key = f"memory:doc:{path}"
        ttl_seconds = int(ttl.total_seconds())
        await self.redis.expire(memory_key, ttl_seconds)

        # Assign to storage tier
        tier = self.policy.ttl_policy.get_tier(score.overall_score)
        await self._assign_tier(path, tier)

        logger.debug(
            f"Set TTL for {path}: {ttl} (importance: {score.overall_score:.2f}, tier: {tier})"
        )

        return score, ttl

    async def update_importance(
        self,
        path: str,
        user_rating: Optional[float] = None,
    ) -> ImportanceScore:
        """Update memory importance with new information."""
        # Set user rating if provided
        if user_rating is not None:
            await self.evaluator.set_user_rating(path, user_rating)

        # Get memory content
        memory_key = f"memory:doc:{path}"
        doc = await self.redis.json().get(memory_key)

        if not doc:
            raise ValueError(f"Memory not found: {path}")

        # Reconstruct MemoryContent from stored data
        memory_data = doc.get("data") or doc.get("content")
        metadata_dict = doc.get("metadata", {})

        # Create MemoryMetadata from dict
        metadata = MemoryMetadata(
            importance=Importance(metadata_dict.get("importance", "normal")),
            ttl_seconds=metadata_dict.get("ttl_seconds"),
            tags=metadata_dict.get("tags", []),
            source=metadata_dict.get("source"),
        )

        # Get memory type from metadata or default
        memory_type = MemoryType(
            metadata_dict.get("memory_type", metadata_dict.get("type", "document"))
        )

        memory = MemoryContent(
            type=memory_type,
            data=memory_data,
            metadata=metadata,
        )

        # Re-evaluate and update TTL
        score, ttl = await self.evaluate_and_set_ttl(path, memory)

        return score

    async def migrate_to_archive(self, path: str) -> bool:
        """
        Archive a memory for long-term storage.

        Returns:
            True if successfully archived
        """
        try:
            # Get memory data
            memory_key = f"memory:doc:{path}"
            doc = await self.redis.json().get(memory_key)

            if not doc:
                return False

            # Prepare archive data
            archive_data = {
                "path": path,
                "content": doc,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "lifecycle_metadata": await self._get_lifecycle_metadata(path),
            }

            # Compress if policy enabled
            if self.policy.compress_archives:
                archive_json = json.dumps(archive_data)
                archive_bytes = gzip.compress(archive_json.encode())
                archive_value = archive_bytes
            else:
                archive_value = json.dumps(archive_data)

            # Store in archive
            archive_key = f"{self.archive_prefix}:{path}"
            await self.redis.set(archive_key, archive_value)

            # Remove from main storage
            await self.redis.delete(memory_key)

            # Update tier
            await self._assign_tier(path, StorageTier.ARCHIVE)

            logger.info(f"Archived memory: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to archive {path}: {e}")
            return False

    async def restore_from_archive(self, path: str) -> bool:
        """
        Restore a memory from archive.

        Returns:
            True if successfully restored
        """
        try:
            archive_key = f"{self.archive_prefix}:{path}"
            archive_data = await self.redis.get(archive_key)

            if not archive_data:
                return False

            # Decompress if needed
            if isinstance(archive_data, bytes):
                try:
                    # Try decompressing
                    archive_json = gzip.decompress(archive_data).decode()
                    data = json.loads(archive_json)
                except:
                    # Not compressed
                    data = json.loads(archive_data)
            else:
                data = json.loads(archive_data)

            # Restore to main storage
            memory_key = f"memory:doc:{path}"
            await self.redis.json().set(memory_key, "$", data["content"])

            # Re-evaluate importance
            # Reconstruct MemoryContent from archived data
            memory_data = data["content"].get("data") or data["content"].get("content")
            metadata_dict = data["content"].get("metadata", {})

            # Create MemoryMetadata from dict
            metadata = MemoryMetadata(
                importance=Importance(metadata_dict.get("importance", "normal")),
                ttl_seconds=metadata_dict.get("ttl_seconds"),
                tags=metadata_dict.get("tags", []),
                source=metadata_dict.get("source"),
            )

            # Get memory type from metadata or default
            memory_type = MemoryType(
                metadata_dict.get("memory_type", metadata_dict.get("type", "document"))
            )

            memory = MemoryContent(
                type=memory_type,
                data=memory_data,
                metadata=metadata,
            )

            await self.evaluate_and_set_ttl(path, memory)

            # Remove from archive
            await self.redis.delete(archive_key)

            logger.info(f"Restored memory from archive: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore {path}: {e}")
            return False

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics by tier."""
        stats = {
            "total_memories": 0,
            "by_tier": {},
            "average_importance": 0.0,
            "storage_efficiency": 0.0,
        }

        # Count memories by tier
        for tier in StorageTier:
            tier_key = f"{self.tier_prefix}:{tier.value}"
            count = await self.redis.scard(tier_key) or 0
            stats["by_tier"][tier.value] = count
            stats["total_memories"] += count

        # Calculate average importance
        if stats["total_memories"] > 0:
            # Sample some memories for stats
            sample_size = min(100, stats["total_memories"])
            sample_paths = []

            for tier in [StorageTier.HOT, StorageTier.WARM]:
                tier_key = f"{self.tier_prefix}:{tier.value}"
                members = await self.redis.srandmember(tier_key, sample_size // 2)
                if members:
                    sample_paths.extend(
                        members if isinstance(members, list) else [members]
                    )

            # Get importance scores
            total_importance = 0.0
            for path in sample_paths[:sample_size]:
                metadata = await self._get_lifecycle_metadata(path)
                if metadata and "importance_score" in metadata:
                    total_importance += metadata["importance_score"]["overall_score"]

            if sample_paths:
                stats["average_importance"] = total_importance / len(sample_paths)

        # Calculate storage efficiency
        hot_count = stats["by_tier"].get(StorageTier.HOT.value, 0)
        total = stats["total_memories"]

        if total > 0:
            # Efficiency = % of memories in appropriate tiers
            stats["storage_efficiency"] = 1.0 - (hot_count / total)

        return stats

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up expired memories."""
        while self._running:
            try:
                await self._cleanup_expired_memories()
                await asyncio.sleep(self.policy.cleanup_check_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _migration_loop(self) -> None:
        """Background task for migrating memories between tiers."""
        while self._running:
            try:
                await self._migrate_memories()
                await asyncio.sleep(
                    self.policy.migration_check_interval.total_seconds()
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Migration loop error: {e}")
                await asyncio.sleep(60)

    async def _reevaluation_loop(self) -> None:
        """Background task for re-evaluating memory importance."""
        while self._running:
            try:
                await self._reevaluate_importance()
                await asyncio.sleep(self.policy.reevaluation_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Re-evaluation loop error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_memories(self) -> None:
        """Clean up memories that have expired."""
        # This is handled by Redis TTL, but we clean up metadata

        # Get all lifecycle metadata keys
        pattern = f"{self.lifecycle_prefix}:*"
        cursor = 0
        cleaned = 0

        while True:
            cursor, keys = await self.redis.scan(
                cursor, match=pattern, count=self.policy.batch_size
            )

            for lifecycle_key in keys:
                path = lifecycle_key.split(":", 2)[2]
                memory_key = f"memory:doc:{path}"

                # Check if memory still exists
                exists = await self.redis.exists(memory_key)
                if not exists:
                    # Clean up orphaned metadata
                    await self.redis.delete(lifecycle_key)

                    # Remove from tier sets
                    for tier in StorageTier:
                        tier_key = f"{self.tier_prefix}:{tier.value}"
                        await self.redis.srem(tier_key, path)

                    cleaned += 1

            if cursor == 0:
                break

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired memory metadata entries")

    async def _migrate_memories(self) -> None:
        """Migrate memories between storage tiers based on importance."""
        migrations = 0

        # Check each tier for migrations
        for current_tier in StorageTier:
            if current_tier == StorageTier.ARCHIVE:
                continue  # Don't migrate from archive automatically

            tier_key = f"{self.tier_prefix}:{current_tier.value}"
            members = await self.redis.smembers(tier_key)

            for path in members:
                # Get current importance
                metadata = await self._get_lifecycle_metadata(path)
                if not metadata or "importance_score" not in metadata:
                    continue

                score = metadata["importance_score"]["overall_score"]

                # Determine correct tier
                correct_tier = self.policy.ttl_policy.get_tier(score)

                # Migrate if needed
                if correct_tier != current_tier:
                    await self._assign_tier(path, correct_tier)
                    migrations += 1

                    # Archive if moved to archive tier
                    if correct_tier == StorageTier.ARCHIVE:
                        await self.migrate_to_archive(path)

        if migrations > 0:
            logger.info(f"Migrated {migrations} memories between tiers")

    async def _reevaluate_importance(self) -> None:
        """Re-evaluate importance for a sample of memories."""
        # Sample memories from non-archive tiers
        sample_paths = []
        sample_size = 50  # Re-evaluate 50 memories per cycle

        for tier in [StorageTier.HOT, StorageTier.WARM, StorageTier.COLD]:
            tier_key = f"{self.tier_prefix}:{tier.value}"
            members = await self.redis.srandmember(tier_key, sample_size // 3)
            if members:
                sample_paths.extend(members if isinstance(members, list) else [members])

        updated = 0
        for path in sample_paths[:sample_size]:
            try:
                memory_key = f"memory:doc:{path}"
                doc = await self.redis.json().get(memory_key)

                if doc:
                    memory = MemoryContent(
                        content=doc.get("content"),
                        metadata=doc.get("metadata"),
                    )

                    await self.evaluate_and_set_ttl(path, memory)
                    updated += 1

            except Exception as e:
                logger.error(f"Failed to re-evaluate {path}: {e}")

        if updated > 0:
            logger.debug(f"Re-evaluated importance for {updated} memories")

    async def _store_lifecycle_metadata(
        self,
        path: str,
        score: ImportanceScore,
        ttl: timedelta,
    ) -> None:
        """Store lifecycle metadata for a memory."""
        metadata = {
            "importance_score": score.to_dict(),
            "ttl_seconds": ttl.total_seconds(),
            "last_evaluated": datetime.now(timezone.utc).isoformat(),
            "tier": self.policy.ttl_policy.get_tier(score.overall_score).value,
        }

        lifecycle_key = f"{self.lifecycle_prefix}:{path}"
        await self.redis.json().set(lifecycle_key, "$", metadata)

        # Set TTL on metadata too
        await self.redis.expire(lifecycle_key, int(ttl.total_seconds()))

    async def _get_lifecycle_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """Get lifecycle metadata for a memory."""
        lifecycle_key = f"{self.lifecycle_prefix}:{path}"
        return await self.redis.json().get(lifecycle_key)

    async def _assign_tier(self, path: str, tier: StorageTier) -> None:
        """Assign memory to a storage tier."""
        # Remove from all tiers
        for t in StorageTier:
            tier_key = f"{self.tier_prefix}:{t.value}"
            await self.redis.srem(tier_key, path)

        # Add to new tier
        tier_key = f"{self.tier_prefix}:{tier.value}"
        await self.redis.sadd(tier_key, path)
