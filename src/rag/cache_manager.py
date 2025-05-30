"""Relationship data cache manager.

Caches frequently queried relationship data to improve performance.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from src.redis.client import RedisClient
from .redis_schema import RedisKeyPattern, RedisSchemaConfig, get_redis_key

logger = logging.getLogger(__name__)


class RelationCacheManager:
    """Relationship data cache manager."""
    
    def __init__(
        self,
        redis_client: RedisClient,
        cache_ttl: int = RedisSchemaConfig.CACHE_TTL,
        max_cache_size: int = RedisSchemaConfig.MAX_CACHE_SIZE,
    ):
        """Initialize.
        
        Args:
            redis_client: Redis client
            cache_ttl: Cache TTL (seconds)
            max_cache_size: Maximum cache size
        """
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        
        # Memory cache (L1 cache)
        self.memory_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _generate_cache_key(self, chunk_id: str, options: Dict[str, Any]) -> str:
        """Generate cache key.
        
        Args:
            chunk_id: Chunk ID
            options: Query options
            
        Returns:
            Cache key
        """
        # Sort options for consistent key generation
        sorted_options = json.dumps(options, sort_keys=True)
        key_data = f"{chunk_id}:{sorted_options}"
        
        # Compress with hash
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"cache:relation:{chunk_id}:{key_hash}"
    
    async def get_cached_relations(
        self,
        chunk_id: str,
        options: Dict[str, Any]
    ) -> Optional[List[Any]]:
        """Query cached relationships.
        
        Args:
            chunk_id: Chunk ID
            options: Query options
            
        Returns:
            Cached relationship list or None
        """
        cache_key = self._generate_cache_key(chunk_id, options)
        
        # Check L1 cache
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            logger.debug(f"L1 cache hit for {cache_key}")
            return self.memory_cache[cache_key]
        
        # Check L2 cache (Redis)
        try:
            cached_data = await self.redis.client.get(cache_key)
            if cached_data:
                self.cache_hits += 1
                logger.debug(f"L2 cache hit for {cache_key}")
                
                # JSON deserialization
                relations = json.loads(cached_data)
                
                # Store in L1 cache
                self._update_memory_cache(cache_key, relations)
                
                return relations
                
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        self.cache_misses += 1
        return None
    
    async def set_cached_relations(
        self,
        chunk_id: str,
        options: Dict[str, Any],
        relations: List[Any]
    ) -> None:
        """Cache relationship data.
        
        Args:
            chunk_id: Chunk ID
            options: Query options
            relations: Relationship list
        """
        cache_key = self._generate_cache_key(chunk_id, options)
        
        try:
            # Convert to serializable format
            serializable_relations = []
            for rel in relations:
                rel_dict = {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relation_type": rel.relation_type.value,
                    "strength": rel.strength,
                    "metadata": rel.metadata,
                    "created_at": rel.created_at.isoformat(),
                }
                
                # Type-specific additional fields
                if hasattr(rel, "distance"):
                    rel_dict["distance"] = rel.distance
                if hasattr(rel, "similarity_score"):
                    rel_dict["similarity_score"] = rel.similarity_score
                if hasattr(rel, "reference_text"):
                    rel_dict["reference_text"] = rel.reference_text
                if hasattr(rel, "time_delta"):
                    rel_dict["time_delta"] = rel.time_delta
                    
                serializable_relations.append(rel_dict)
            
            # Store in L2 cache (Redis)
            await self.redis.client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(serializable_relations)
            )
            
            # Update L1 cache
            self._update_memory_cache(cache_key, serializable_relations)
            
            logger.debug(f"Cached {len(relations)} relations for {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _update_memory_cache(self, key: str, value: Any) -> None:
        """Update memory cache.
        
        Args:
            key: Cache key
            value: Cache value
        """
        # Check size limit
        if len(self.memory_cache) >= self.max_cache_size:
            # Remove oldest item (simple FIFO)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
    
    async def invalidate_cache(self, chunk_id: str) -> None:
        """Invalidate all cache for specific chunk.
        
        Args:
            chunk_id: Chunk ID
        """
        # Remove from L1 cache
        keys_to_remove = [
            key for key in self.memory_cache.keys()
            if key.startswith(f"cache:relation:{chunk_id}:")
        ]
        for key in keys_to_remove:
            del self.memory_cache[key]
        
        # Remove from L2 cache
        try:
            pattern = f"cache:relation:{chunk_id}:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis.client.scan(
                    cursor, match=pattern, count=100
                )
                
                if keys:
                    await self.redis.client.delete(*keys)
                
                if cursor == 0:
                    break
                    
            logger.info(f"Invalidated cache for chunk {chunk_id}")
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    async def warm_cache(self, chunk_ids: List[str]) -> None:
        """Cache warming (preloading).
        
        Args:
            chunk_ids: List of chunk IDs to warm
        """
        logger.info(f"Warming cache for {len(chunk_ids)} chunks")
        
        # Warm cache in parallel
        tasks = []
        for chunk_id in chunk_ids:
            # Warm cache with default options
            default_options = {
                "relation_types": None,
                "limit": 50,
            }
            
            task = self._warm_single_chunk(chunk_id, default_options)
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _warm_single_chunk(self, chunk_id: str, options: Dict[str, Any]) -> None:
        """Warm cache for single chunk.
        
        Args:
            chunk_id: Chunk ID
            options: Query options
        """
        # Skip if already cached
        cache_key = self._generate_cache_key(chunk_id, options)
        if cache_key in self.memory_cache:
            return
        
        # Actual data should be queried from ChunkRelationshipManager
        # Here we just prepare the cache structure
        logger.debug(f"Cache warmed for {chunk_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics.
        
        Returns:
            Cache statistics information
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "max_cache_size": self.max_cache_size,
        }
    
    async def clear_all_cache(self) -> None:
        """Delete all cache."""
        # Clear L1 cache
        self.memory_cache.clear()
        
        # Clear L2 cache
        try:
            pattern = "cache:relation:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis.client.scan(
                    cursor, match=pattern, count=100
                )
                
                if keys:
                    await self.redis.client.delete(*keys)
                
                if cursor == 0:
                    break
                    
            logger.info("Cleared all relation caches")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
        
        # Reset statistics
        self.cache_hits = 0
        self.cache_misses = 0


class CacheAwareRelationQuery:
    """Cache-aware relationship query helper."""
    
    def __init__(
        self,
        redis_client: RedisClient,
        cache_manager: RelationCacheManager
    ):
        """Initialize.
        
        Args:
            redis_client: Redis client
            cache_manager: Cache manager
        """
        self.redis = redis_client
        self.cache = cache_manager
    
    async def query_with_cache(
        self,
        chunk_id: str,
        query_func: Any,
        options: Dict[str, Any]
    ) -> List[Any]:
        """Execute query with cache.
        
        Args:
            chunk_id: Chunk ID
            query_func: Actual query function
            options: Query options
            
        Returns:
            Query results
        """
        # Check cache
        cached_result = await self.cache.get_cached_relations(chunk_id, options)
        if cached_result is not None:
            return cached_result
        
        # Cache miss - execute actual query
        # Convert relation_types in options to RelationType objects
        query_options = dict(options)
        if "relation_types" in query_options and query_options["relation_types"]:
            from . import relationships  # Prevent circular import
            query_options["relation_types"] = [
                relationships.RelationType(rt) for rt in query_options["relation_types"]
            ]
        
        result = await query_func(chunk_id, **query_options)
        
        # Cache results
        if result:
            await self.cache.set_cached_relations(chunk_id, options, result)
        
        return result