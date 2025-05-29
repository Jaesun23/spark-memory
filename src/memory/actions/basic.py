"""Basic memory actions (save, get, update, delete)."""

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
    """Basic memory operations handler."""

    def __init__(self, redis_client: RedisClient, default_timezone: str = "UTC"):
        """Initialize basic actions.

        Args:
            redis_client: Redis client instance
            default_timezone: Default timezone for time paths
        """
        self.redis = redis_client
        self.time_gen = TimePathGenerator(default_timezone)

    async def execute(
        self,
        action: str,
        paths: List[str],
        content: Optional[Any],
        options: Dict[str, Any],
    ) -> Union[str, Dict[str, Any], List[str]]:
        """Execute basic action.

        Args:
            action: Action to execute (save, get, update, delete)
            paths: Memory paths
            content: Content for save/update
            options: Additional options

        Returns:
            Action result
        """
        if action == "save":
            return await self._save(paths, content, options)
        elif action == "get":
            return await self._get(paths, content, options)
        elif action == "update":
            return await self._update(paths, content, options)
        elif action == "delete":
            return await self._delete(paths, content, options)
        else:
            raise ValueError(f"Unknown basic action: {action}")

    async def _save(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> str:
        """메모리 저장.

        시간 기반 경로를 자동 생성하고 메타데이터를 설정하여
        메모리를 저장합니다.

        Args:
            paths: 저장 경로
            content: 저장할 내용
            options: 저장 옵션

        Returns:
            저장된 메모리 키
        """
        if content is None:
            raise ValueError("Content is required for save action")

        # 메모리 타입 결정
        memory_type = MemoryType(
            options.get("type", MemoryType.from_content(content).value)
        )

        # 시간 경로 생성
        if not paths or paths == [""]:
            category = options.get("category", memory_type.value)
            time_path = self.time_gen.generate_path(category)
            paths = time_path.split("/")

        # 메모리 콘텐츠 생성
        memory_content = MemoryContent(
            type=memory_type,
            data=content,
        )

        # 메타데이터 설정
        if "tags" in options:
            memory_content.metadata.tags = options["tags"]
        if "importance" in options:
            memory_content.metadata.importance = Importance(options["importance"])
        if "ttl" in options:
            memory_content.metadata.ttl_seconds = options["ttl"]
        if "source" in options:
            memory_content.metadata.source = options["source"]

        # 키 생성
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        # 데이터 타입별 저장
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
        """대화형 메모리 저장 (Redis Streams).

        Args:
            key: Redis 키
            content: 메모리 콘텐츠
            options: 저장 옵션
        """
        # 스트림 키 생성 (시간 기반)
        stream_key = f"stream:{key}"

        # 메시지 데이터 준비
        if isinstance(content.data, dict):
            message_data = content.data
        else:
            message_data = {"content": str(content.data)}

        # 메타데이터 추가
        message_data["_metadata"] = json.dumps(content.metadata.to_dict())

        # Redis Stream에 추가
        client = self.redis.client
        message_id = await client.xadd(stream_key, message_data)

        # TTL 설정
        if content.metadata.ttl_seconds:
            await client.expire(stream_key, content.metadata.ttl_seconds)

        # 메타데이터를 별도 Hash로 저장
        await self._save_metadata_hash(key, content.metadata)

        logger.debug(f"Conversation saved to stream {stream_key} with ID {message_id}")

    async def _save_json(
        self,
        key: str,
        content: MemoryContent,
        options: Dict[str, Any],
        memory_key: Optional[MemoryKey] = None,
    ) -> None:
        """JSON 형태 메모리 저장 (RedisJSON).

        Args:
            key: Redis 키
            content: 메모리 콘텐츠
            options: 저장 옵션
            memory_key: 메모리 키 객체
        """
        # JSON 키 생성
        json_key = f"json:{key}"

        # 저장할 데이터
        data_to_save = content.to_dict()

        # RedisJSON에 저장
        client = self.redis.client
        await client.json().set(json_key, "$", data_to_save)

        # TTL 설정
        if content.metadata.ttl_seconds:
            await client.expire(json_key, content.metadata.ttl_seconds)

        # 메타데이터를 별도 Hash로 저장
        await self._save_metadata_hash(key, content.metadata)

        logger.debug(f"Document saved to {json_key}")

    async def _save_metadata_hash(
        self,
        key: str,
        metadata: MemoryMetadata,
    ) -> None:
        """메타데이터를 Hash로 저장.

        빠른 필터링과 검색을 위해 메타데이터를 별도 Hash로 저장합니다.

        Args:
            key: 메모리 키
            metadata: 메타데이터
        """
        hash_key = f"meta:{key}"
        client = self.redis.client

        # 메타데이터를 평탄화
        meta_dict = {
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "importance": metadata.importance.value,
            "access_count": str(metadata.access_count),
        }

        if metadata.ttl_seconds:
            meta_dict["ttl_seconds"] = str(metadata.ttl_seconds)
        if metadata.source:
            meta_dict["source"] = metadata.source

        # 태그 처리
        for i, tag in enumerate(metadata.tags):
            meta_dict[f"tag:{i}"] = tag

        # Hash에 저장
        await client.hset(hash_key, mapping=meta_dict)

        # TTL 설정
        if metadata.ttl_seconds:
            await client.expire(hash_key, metadata.ttl_seconds)

    async def _get(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """메모리 조회.

        경로를 기반으로 메모리를 조회하고 접근 통계를 업데이트합니다.

        Args:
            paths: 조회 경로
            content: 미사용
            options: 조회 옵션

        Returns:
            메모리 데이터 또는 None
        """
        if not paths:
            raise ValueError("Paths are required for get action")

        # 메모리 타입 추론
        memory_type = MemoryType(options.get("type", MemoryType.DOCUMENT.value))

        # 키 생성
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        client = self.redis.client

        # 타입별 조회
        if memory_type == MemoryType.CONVERSATION:
            # Stream 조회
            stream_key = f"stream:{key}"
            messages = await client.xrange(stream_key)

            if not messages:
                return None

            # 메시지 파싱
            conversation_data = []
            for msg_id, data in messages:
                # 메타데이터 추출
                metadata_str = data.pop("_metadata", "{}")
                msg_metadata = json.loads(metadata_str)

                conversation_data.append(
                    {
                        "id": msg_id,
                        "data": data,
                        "metadata": msg_metadata,
                    }
                )

            result = {
                "type": memory_type.value,
                "path": "/".join(paths),
                "conversation": conversation_data,
            }

        else:
            # JSON 조회
            json_key = f"json:{key}"
            data = await client.json().get(json_key)

            if not data:
                return None

            result = {
                "type": memory_type.value,
                "path": "/".join(paths),
                "content": data,
            }

        # 접근 카운트 증가
        if options.get("update_access", True):
            await self._update_access_stats(key)

        return result

    async def _update_access_stats(self, key: str) -> None:
        """접근 통계 업데이트.

        Args:
            key: 메모리 키
        """
        hash_key = f"meta:{key}"
        client = self.redis.client

        # 접근 카운트 증가
        await client.hincrby(hash_key, "access_count", 1)

        # 마지막 접근 시간 업데이트
        await client.hset(
            hash_key,
            "last_accessed_at",
            datetime.now(timezone.utc).isoformat(),
        )

    async def _update(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> str:
        """메모리 수정.

        Args:
            paths: 수정할 경로
            content: 수정할 내용
            options: 수정 옵션

        Returns:
            수정된 메모리 키
        """
        if not paths:
            raise ValueError("Paths are required for update action")

        if content is None:
            raise ValueError("Content is required for update action")

        # 메모리 타입 추론
        memory_type = MemoryType(options.get("type", MemoryType.DOCUMENT.value))

        # 키 생성
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        client = self.redis.client

        # 기존 메모리 확인
        if memory_type == MemoryType.CONVERSATION:
            # 대화형은 append만 지원
            if content:
                stream_key = f"stream:{key}"
                message_data = (
                    content if isinstance(content, dict) else {"content": str(content)}
                )
                message_data["_metadata"] = json.dumps(
                    {"updated_at": datetime.now(timezone.utc).isoformat()}
                )
                await client.xadd(stream_key, message_data)

        else:
            # JSON 업데이트
            json_key = f"json:{key}"

            if content is not None:
                # 전체 교체
                memory_content = MemoryContent(
                    type=memory_type,
                    data=content,
                )

                # 기존 메타데이터 유지 옵션
                if options.get("preserve_metadata", False):
                    existing = await client.json().get(json_key)
                    if existing and "metadata" in existing:
                        memory_content.metadata = MemoryMetadata.from_dict(
                            existing["metadata"]
                        )

                # 메타데이터 업데이트
                memory_content.metadata.updated_at = datetime.now(timezone.utc)
                if "tags" in options:
                    if isinstance(memory_content.metadata.tags, dict):
                        memory_content.metadata.tags.update(options["tags"])
                    else:
                        memory_content.metadata.tags = options["tags"]
                if "importance" in options:
                    memory_content.metadata.importance = Importance(
                        options["importance"]
                    )

                await client.json().set(json_key, "$", memory_content.to_dict())
            else:
                # 메타데이터만 업데이트
                if "tags" in options or "importance" in options:
                    existing = await client.json().get(json_key)
                    if existing:
                        if "tags" in options:
                            if isinstance(existing["metadata"].get("tags"), dict):
                                existing["metadata"]["tags"].update(options["tags"])
                            else:
                                existing["metadata"]["tags"] = options["tags"]
                        if "importance" in options:
                            existing["metadata"]["importance"] = options["importance"]
                        existing["metadata"]["updated_at"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        await client.json().set(json_key, "$", existing)

        # 메타데이터 Hash 업데이트
        await self._update_metadata_hash(key, options)

        logger.info(f"Memory updated: {key}")
        return key

    async def _update_metadata_hash(
        self,
        key: str,
        options: Dict[str, Any],
    ) -> None:
        """메타데이터 Hash 업데이트.

        Args:
            key: 메모리 키
            options: 업데이트할 메타데이터
        """
        hash_key = f"meta:{key}"
        client = self.redis.client

        updates = {"updated_at": datetime.now(timezone.utc).isoformat()}

        if "importance" in options:
            updates["importance"] = options["importance"]

        if "tags" in options:
            for tag_key, tag_value in options["tags"].items():
                updates[f"tag:{tag_key}"] = str(tag_value)

        await client.hset(hash_key, mapping=updates)

    async def _delete(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> List[str]:
        """메모리 삭제.

        지정된 경로의 메모리를 삭제합니다. 패턴 매칭을 지원합니다.

        Args:
            paths: 삭제할 경로
            content: 미사용
            options: 삭제 옵션

        Returns:
            삭제된 키 리스트
        """
        if not paths:
            raise ValueError("Paths are required for delete action")

        client = self.redis.client
        deleted_keys = []

        # 패턴 매칭 옵션
        if options.get("pattern", False):
            # 와일드카드 패턴으로 삭제
            pattern = f"*{':'.join(paths)}*"

            # 관련 키 찾기
            all_patterns = [
                f"json:{pattern}",
                f"stream:{pattern}",
                f"meta:{pattern}",
            ]

            for pat in all_patterns:
                cursor = 0
                while True:
                    cursor, keys = await client.scan(cursor, match=pat, count=100)
                    if keys:
                        await client.delete(*keys)
                        deleted_keys.extend(keys)
                    if cursor == 0:
                        break
        else:
            # 정확한 키 삭제
            memory_type = MemoryType(options.get("type", MemoryType.DOCUMENT.value))
            memory_key = MemoryKey(paths=paths, memory_type=memory_type)
            key = memory_key.generate()

            # 관련 키들
            keys_to_delete = []
            if memory_type == MemoryType.CONVERSATION:
                keys_to_delete.append(f"stream:{key}")
            else:
                keys_to_delete.append(f"json:{key}")

            keys_to_delete.append(f"meta:{key}")

            # 삭제 실행
            for k in keys_to_delete:
                if await client.exists(k):
                    await client.delete(k)
                    deleted_keys.append(k)

        logger.info(f"Deleted {len(deleted_keys)} keys")
        return deleted_keys
