"""상태 관리자 - LangGraph 통합 및 체크포인트 관리.

MCP의 양방향 통신을 활용하여 실시간 상태 업데이트를 제공합니다.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.memory.models import Importance, MemoryMetadata
from src.redis.client import RedisClient

logger = logging.getLogger(__name__)


class StateManager:
    """상태 및 체크포인트 관리자.

    LangGraph와 통합하여 작업 상태를 저장하고 복원합니다.
    Redis Streams를 사용하여 실시간 상태 업데이트를 지원합니다.
    """

    def __init__(self, redis_client: RedisClient) -> None:
        """StateManager 초기화.

        Args:
            redis_client: Redis 클라이언트
        """
        self.redis = redis_client
        self.checkpoint_prefix = "memory:checkpoint"
        self.state_prefix = "memory:state"
        self.event_stream = "memory:events"

    async def create_checkpoint(
        self,
        paths: List[str],
        state: Dict[str, Any],
        description: Optional[str] = None,
    ) -> str:
        """체크포인트 생성.

        Args:
            paths: 체크포인트 경로
            state: 저장할 상태 데이터
            description: 체크포인트 설명

        Returns:
            체크포인트 ID
        """
        checkpoint_id = str(uuid4())
        key = f"{self.checkpoint_prefix}:{':'.join(paths)}:{checkpoint_id}"

        # 메타데이터 생성
        metadata = MemoryMetadata(
            id=checkpoint_id,
            created_at=datetime.now(),
            tags=["checkpoint"] + paths,
            importance=Importance.HIGH,
            source="state_manager",
        )

        # 체크포인트 데이터
        checkpoint_data = {
            "state": state,
            "description": description or f"Checkpoint for {'/'.join(paths)}",
            "metadata": json.dumps(metadata.to_dict()),
            "paths": paths,
        }

        # Redis에 저장
        await self.redis.client.json().set(key, "$", checkpoint_data)

        # 이벤트 발행
        await self._publish_event(
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint_id,
                "paths": paths,
                "key": key,
            },
        )

        logger.info(f"Checkpoint created: {checkpoint_id} at {key}")
        return checkpoint_id

    async def restore_checkpoint(
        self,
        paths: List[str],
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """체크포인트 복원.

        Args:
            paths: 체크포인트 경로
            checkpoint_id: 특정 체크포인트 ID (없으면 최신)

        Returns:
            복원된 상태 데이터
        """
        if checkpoint_id:
            # 특정 체크포인트 복원
            key = f"{self.checkpoint_prefix}:{':'.join(paths)}:{checkpoint_id}"
            data = await self.redis.client.json().get(key)
            if data:
                await self._publish_event(
                    "checkpoint_restored",
                    {
                        "checkpoint_id": checkpoint_id,
                        "paths": paths,
                    },
                )
                return data.get("state")  # type: ignore[no-any-return]
        else:
            # 최신 체크포인트 찾기
            pattern = f"{self.checkpoint_prefix}:{':'.join(paths)}:*"
            keys = await self._scan_keys(pattern)

            if not keys:
                return None

            # 가장 최근 체크포인트 선택
            latest_key = sorted(keys)[-1]  # UUID 기반 정렬
            data = await self.redis.client.json().get(latest_key)

            if data:
                checkpoint_id = latest_key.split(":")[-1]
                await self._publish_event(
                    "checkpoint_restored",
                    {
                        "checkpoint_id": checkpoint_id,
                        "paths": paths,
                        "latest": True,
                    },
                )
                return data.get("state")  # type: ignore[no-any-return]

        return None

    async def list_checkpoints(
        self,
        paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """체크포인트 목록 조회.

        Args:
            paths: 경로 필터 (없으면 전체)

        Returns:
            체크포인트 목록
        """
        if paths:
            pattern = f"{self.checkpoint_prefix}:{':'.join(paths)}:*"
        else:
            pattern = f"{self.checkpoint_prefix}:*"

        keys = await self._scan_keys(pattern)
        checkpoints = []

        for key in keys:
            data = await self.redis.client.json().get(key)
            if data:
                checkpoint_info = {
                    "id": key.split(":")[-1],
                    "key": key,
                    "paths": data.get("paths", []),
                    "description": data.get("description", ""),
                    "created_at": json.loads(data.get("metadata", "{}")).get(
                        "created_at"
                    ),
                }
                checkpoints.append(checkpoint_info)

        # 최신순 정렬
        checkpoints.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return checkpoints

    async def get_current_state(
        self,
        paths: List[str],
    ) -> Optional[Dict[str, Any]]:
        """현재 상태 조회.

        Args:
            paths: 상태 경로

        Returns:
            현재 상태 데이터
        """
        key = f"{self.state_prefix}:{':'.join(paths)}"
        return await self.redis.client.json().get(key)  # type: ignore[no-any-return]

    async def update_state(
        self,
        paths: List[str],
        state: Dict[str, Any],
        partial: bool = False,
    ) -> None:
        """상태 업데이트.

        Args:
            paths: 상태 경로
            state: 새로운 상태 데이터
            partial: 부분 업데이트 여부
        """
        key = f"{self.state_prefix}:{':'.join(paths)}"

        if partial:
            # 부분 업데이트
            current = await self.redis.client.json().get(key) or {}
            current.update(state)
            await self.redis.client.json().set(key, "$", current)
        else:
            # 전체 교체
            await self.redis.client.json().set(key, "$", state)

        # 이벤트 발행
        await self._publish_event(
            "state_updated",
            {
                "paths": paths,
                "partial": partial,
                "key": key,
            },
        )

    async def subscribe_to_events(
        self,
        callback: Optional[Any] = None,
        event_types: Optional[List[str]] = None,
    ) -> None:
        """이벤트 구독 (양방향 통신).

        Args:
            callback: 이벤트 콜백 함수
            event_types: 구독할 이벤트 타입
        """
        # TODO: MCP 양방향 통신과 통합
        logger.info(f"Subscribing to events: {event_types or 'all'}")

    async def _publish_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """이벤트 발행.

        Args:
            event_type: 이벤트 타입
            data: 이벤트 데이터
        """
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        # Redis Streams에 이벤트 추가
        await self.redis.client.xadd(
            self.event_stream,
            {"event": json.dumps(event)},
        )

        logger.debug(f"Event published: {event_type}")

    async def _scan_keys(self, pattern: str) -> List[str]:
        """패턴에 맞는 키 스캔.

        Args:
            pattern: 검색 패턴

        Returns:
            매칭된 키 리스트
        """
        cursor = 0
        keys = []
        while True:
            cursor, partial_keys = await self.redis.client.scan(
                cursor, match=pattern, count=100
            )
            keys.extend(
                [k.decode() if isinstance(k, bytes) else k for k in partial_keys]
            )
            if cursor == 0:
                break
        return keys
