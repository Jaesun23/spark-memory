"""Redis relationship storage schema definitions and constants.

This module defines the Redis schema used by ChunkRelationshipManager.
Provides optimized key naming conventions and data structures.
"""

from enum import Enum
from typing import Dict, Any


class RedisKeyPattern(str, Enum):
    """Redis key pattern definitions."""
    
    # Metadata
    RELATION_METADATA = "hash:relation:metadata:{chunk_id}"
    CHUNK_INFO = "hash:chunk:info:{chunk_id}"
    
    # Type-based relationship storage
    RELATION_SET = "set:relation:{type}:{chunk_id}"
    REVERSE_RELATION_SET = "set:relation:reverse:{type}:{chunk_id}"
    
    # Relationship details
    RELATION_DETAIL = "json:relation:detail:{chunk_id}:{target_id}"
    
    # Score-based sorting
    RELATION_SCORE = "zset:relation:score:{type}:{chunk_id}"
    GLOBAL_RELATION_SCORE = "zset:relation:global:{type}"
    
    # Indexes
    ENTITY_INDEX = "set:index:entity:{entity}"
    TOPIC_INDEX = "set:index:topic:{topic}"
    TIME_INDEX = "zset:index:time:chunks"
    
    # Cache
    RELATION_CACHE = "cache:relation:{chunk_id}:{type}"
    SEARCH_CACHE = "cache:search:{query_hash}"
    
    # Statistics
    RELATION_STATS = "hash:stats:relation:{date}"
    CHUNK_STATS = "hash:stats:chunk:{date}"


class RedisDataStructure(str, Enum):
    """Redis data structure types."""
    
    HASH = "hash"  # Metadata, statistics
    SET = "set"  # Relationship lists, indexes
    SORTED_SET = "zset"  # Score-based sorting
    JSON = "json"  # Complex structured data
    LIST = "list"  # Ordered data
    STREAM = "stream"  # Time series data


class RedisSchemaConfig:
    """Redis schema configuration."""
    
    # TTL settings (seconds)
    DEFAULT_TTL = 86400 * 30  # 30 days
    CACHE_TTL = 3600  # 1 hour
    STATS_TTL = 86400 * 90  # 90 days
    
    # Batch sizes
    PIPELINE_BATCH_SIZE = 100
    SCAN_COUNT = 1000
    
    # Limits
    MAX_RELATIONS_PER_CHUNK = 1000
    MAX_RELATION_DEPTH = 5
    MAX_CACHE_SIZE = 10000
    
    # Index settings
    INDEX_PREFIXES = {
        "entity": "entity:",
        "topic": "topic:",
        "time": "time:",
        "type": "type:",
    }


def get_redis_key(pattern: RedisKeyPattern, **kwargs) -> str:
    """Redis key generation helper function.
    
    Args:
        pattern: Key pattern
        **kwargs: Variables needed for pattern
        
    Returns:
        Generated Redis key
    """
    return pattern.value.format(**kwargs)


def parse_redis_key(key: str) -> Dict[str, str]:
    """Redis key parsing helper function.
    
    Args:
        key: Redis key
        
    Returns:
        Parsed key components
    """
    parts = key.split(":")
    result = {"full_key": key}
    
    if len(parts) >= 3:
        result["type"] = parts[0]  # hash, set, zset, json etc
        result["namespace"] = parts[1]  # relation, index, cache etc
        
        if parts[1] == "relation":
            if len(parts) >= 4:
                result["subtype"] = parts[2]  # metadata, detail, score etc
                result["chunk_id"] = parts[3]
                if len(parts) >= 5:
                    result["target_id"] = parts[4]
        elif parts[1] == "index":
            if len(parts) >= 4:
                result["index_type"] = parts[2]  # entity, topic, time
                result["index_value"] = ":".join(parts[3:])
    
    return result


class RedisSchema:
    """Redis schema definition and validation."""
    
    @staticmethod
    def validate_key(key: str) -> bool:
        """Validate key format."""
        parts = key.split(":")
        if len(parts) < 3:
            return False
        
        # Validate data structure type
        if parts[0] not in [ds.value for ds in RedisDataStructure]:
            return False
        
        return True
    
    @staticmethod
    def get_schema_info() -> Dict[str, Any]:
        """Return schema information."""
        return {
            "version": "1.0",
            "patterns": {
                pattern.name: pattern.value 
                for pattern in RedisKeyPattern
            },
            "data_structures": {
                ds.name: ds.value 
                for ds in RedisDataStructure
            },
            "config": {
                "default_ttl": RedisSchemaConfig.DEFAULT_TTL,
                "cache_ttl": RedisSchemaConfig.CACHE_TTL,
                "max_relations_per_chunk": RedisSchemaConfig.MAX_RELATIONS_PER_CHUNK,
                "max_relation_depth": RedisSchemaConfig.MAX_RELATION_DEPTH,
            }
        }


# Schema migration support
class SchemaMigration:
    """Schema version management and migration."""
    
    CURRENT_VERSION = "1.0"
    
    @staticmethod
    async def migrate_to_latest(redis_client, from_version: str) -> bool:
        """Migrate to latest schema.
        
        Args:
            redis_client: Redis client
            from_version: Current schema version
            
        Returns:
            Migration success status
        """
        # TODO: Implement version-specific migration logic
        if from_version == SchemaMigration.CURRENT_VERSION:
            return True
        
        # Add migration logic...
        return True