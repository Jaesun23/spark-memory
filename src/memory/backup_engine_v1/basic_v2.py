"""Basic memory actions module.

This module contains the core CRUD operations for memory management,
extracted from the main engine for modularity and reusability.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from src.memory.models import (
    Importance,
    MemoryContent,
    MemoryKey,
    MemoryMetadata,
    MemoryType,
)
from src.redis.client import RedisClient
from src.utils.time_path import TimePathGenerator

logger = logging.getLogger(__name__)


class BasicActions:
    """Basic memory actions handler.
    
    Provides core CRUD operations: save, get, update, delete.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        default_timezone: str = "Asia/Seoul",
    ) -> None:
        """Initialize BasicActions.

        Args:
            redis_client: Redis client instance
            default_timezone: Default timezone for time-based paths
        """
        self.redis = redis_client
        self.time_gen = TimePathGenerator(default_timezone)
        self.default_timezone = default_timezone

    async def execute(
        self,
        action: str,
        paths: List[str],
        content: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Execute a basic memory action.

        Args:
            action: Action to execute (save, get, update, delete)
            paths: Memory paths
            content: Content for save/update actions
            options: Additional options

        Returns:
            Action execution result

        Raises:
            ValueError: Invalid action or parameters
            RuntimeError: Execution error
        """
        options = options or {}

        # Action routing
        action_map = {
            "save": self._save,
            "get": self._get,
            "update": self._update,
            "delete": self._delete,
        }

        if action not in action_map:
            raise ValueError(
                f"Unknown action: {action}. Valid actions: {list(action_map.keys())}"
            )

        try:
            logger.info(f"Executing basic action: {action} with paths: {paths}")
            return await action_map[action](paths, content, options)
        except Exception as e:
            logger.error(f"Error executing basic action {action}: {e}")
            raise RuntimeError(f"Basic action failed: {e}") from e

    async def _save(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> str:
        """Save memory.

        Args:
            paths: Save paths
            content: Content to save
            options: Save options

        Returns:
            Saved memory key
        """
        if content is None:
            raise ValueError("Content is required for save action")

        # Determine memory type
        memory_type = MemoryType(
            options.get("type", MemoryType.from_content(content).value)
        )

        # Generate time path if not provided
        if not paths or paths == [""]:
            category = options.get("category", memory_type.value)
            time_path = self.time_gen.generate_path(category)
            paths = time_path.split("/")

        # Create memory content
        memory_content = MemoryContent(
            type=memory_type,
            data=content,
        )

        # Set metadata
        if "tags" in options:
            memory_content.metadata.tags = options["tags"]
        if "importance" in options:
            memory_content.metadata.importance = Importance(options["importance"])
        if "ttl" in options:
            memory_content.metadata.ttl_seconds = options["ttl"]
        if "source" in options:
            memory_content.metadata.source = options["source"]

        # Generate key
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        # Save based on data type
        if memory_type == MemoryType.CONVERSATION:
            await self._save_conversation(key, memory_content, options)
        else:
            await self._save_json(key, memory_content, options, memory_key)

        logger.info(f"Memory saved with key: {key}")
        return key

    async def _save_conversation(
        self,
        key: str,
        content: MemoryContent,
        options: Dict[str, Any],
    ) -> None:
        """Save conversation memory (Redis Streams).

        Args:
            key: Redis key
            content: Memory content
            options: Save options
        """
        # Generate stream key
        stream_key = f"stream:{key}"

        # Prepare message data
        if isinstance(content.data, dict):
            message_data = content.data
        else:
            message_data = {"content": str(content.data)}

        # Add metadata
        message_data["_metadata"] = json.dumps(content.metadata.to_dict())

        # Add to Redis Stream
        client = self.redis.client
        message_id = await client.xadd(stream_key, message_data)

        # Set TTL
        if content.metadata.ttl_seconds:
            await client.expire(stream_key, content.metadata.ttl_seconds)

        # Save metadata as separate Hash
        await self._save_metadata_hash(key, content.metadata)

        logger.debug(f"Conversation saved to stream {stream_key} with ID {message_id}")

    async def _save_json(
        self,
        key: str,
        content: MemoryContent,
        options: Dict[str, Any],
        memory_key: Optional[MemoryKey] = None,
    ) -> None:
        """Save JSON memory (RedisJSON).

        Args:
            key: Redis key
            content: Memory content
            options: Save options
            memory_key: Memory key object (optional)
        """
        # Generate JSON key
        json_key = f"json:{key}"

        # Data to save
        data_to_save = content.to_dict()

        # Save to RedisJSON
        client = self.redis.client
        await client.json().set(json_key, "$", data_to_save)

        # Set TTL
        if content.metadata.ttl_seconds:
            await client.expire(json_key, content.metadata.ttl_seconds)

        # Save metadata as separate Hash
        await self._save_metadata_hash(key, content.metadata)

        logger.debug(f"JSON saved to {json_key}")

    async def _save_metadata_hash(
        self,
        key: str,
        metadata: MemoryMetadata,
    ) -> None:
        """Save metadata to Redis Hash.

        For fast field access and atomic updates.

        Args:
            key: Original memory key
            metadata: Metadata object
        """
        # Generate hash key
        hash_key = f"meta:{key}"

        # Prepare hash fields (convert all values to strings)
        hash_fields: Dict[str, str] = {
            "created_at": metadata.created_at.isoformat(),
            "importance": metadata.importance.value,
            "access_count": str(metadata.access_count),
        }

        # Optional fields
        if metadata.updated_at:
            hash_fields["updated_at"] = metadata.updated_at.isoformat()
        if metadata.tags:
            hash_fields["tags"] = ",".join(metadata.tags)
        if metadata.source:
            hash_fields["source"] = metadata.source
        if metadata.ttl_seconds:
            hash_fields["ttl_seconds"] = str(metadata.ttl_seconds)
        if metadata.accessed_at:
            hash_fields["accessed_at"] = metadata.accessed_at.isoformat()

        # Save to Redis Hash
        client = self.redis.client
        await client.hset(hash_key, mapping=hash_fields)  # type: ignore[arg-type]

        # Set TTL (same TTL for metadata)
        if metadata.ttl_seconds:
            await client.expire(hash_key, metadata.ttl_seconds)

        logger.debug(f"Metadata saved to hash {hash_key}")

    async def _get(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get memory.

        Args:
            paths: Paths to retrieve
            content: Not used
            options: Get options

        Returns:
            Retrieved memory content
        """
        if not paths or paths == [""]:
            raise ValueError("Paths are required for get action")

        # Infer memory type
        memory_type = MemoryType(options.get("type", "document"))

        # Generate key
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        client = self.redis.client

        # Get based on data type
        if memory_type == MemoryType.CONVERSATION:
            stream_key = f"stream:{key}"
            # Read messages from stream
            messages = await client.xrange(stream_key, "-", "+")

            if not messages:
                return []

            # Format messages
            result = []
            for msg_id, msg_data in messages:
                # Separate _metadata field
                metadata = msg_data.pop("_metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                result.append(
                    {
                        "id": msg_id,
                        "data": msg_data,
                        "metadata": metadata,
                    }
                )

            return result
        else:
            json_key = f"json:{key}"
            # Get JSON
            data = await client.json().get(json_key, "$")

            if not data:
                return {}

            # Return first result (RedisJSON returns array)
            result = data[0] if isinstance(data, list) else data

            return result if isinstance(result, dict) else {"data": result}

    async def _update(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> str:
        """Update memory.

        Args:
            paths: Memory paths to update
            content: New content
            options: Update options

        Returns:
            Updated memory key
        """
        if not paths or paths == [""]:
            raise ValueError("Paths are required for update action")
        if content is None:
            raise ValueError("Content is required for update action")

        # Infer memory type
        memory_type = MemoryType(options.get("type", "document"))

        # Generate key
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()
        json_key = f"json:{key}"

        client = self.redis.client

        # Get existing data
        existing_data = await client.json().get(json_key, "$")
        if not existing_data:
            raise ValueError(f"Memory not found: {key}")

        # Convert to MemoryContent object
        existing_content = MemoryContent.from_dict(existing_data[0])

        # Update content
        existing_content.data = content
        existing_content.metadata.update()

        # Update options
        if "tags" in options:
            existing_content.metadata.tags = options["tags"]
        if "importance" in options:
            existing_content.metadata.importance = Importance(options["importance"])

        # Save
        await client.json().set(json_key, "$", existing_content.to_dict())

        logger.info(f"Memory updated: {key}")
        return key

    async def _delete(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Delete memory.

        Args:
            paths: Memory paths to delete
            content: Not used
            options: Delete options

        Returns:
            Delete result
        """
        if not paths or paths == [""]:
            raise ValueError("Paths are required for delete action")

        # Infer memory type
        memory_type = MemoryType(options.get("type", "document"))

        # Generate key
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        client = self.redis.client
        deleted_count = 0

        # Delete based on data type
        if memory_type == MemoryType.CONVERSATION:
            stream_key = f"stream:{key}"
            if await client.exists(stream_key):
                await client.delete(stream_key)
                deleted_count += 1
        else:
            json_key = f"json:{key}"
            if await client.exists(json_key):
                await client.delete(json_key)
                deleted_count += 1

        # Pattern delete option
        if options.get("pattern", False):
            pattern = f"json:{key}:*"
            cursor = 0

            while True:
                cursor, keys = await client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100,
                )

                if keys:
                    deleted_count += await client.delete(*keys)

                if cursor == 0:
                    break

        logger.info(f"Deleted {deleted_count} memory items")

        return {
            "deleted": deleted_count,
            "key": key,
        }