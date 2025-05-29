"""Redis 클라이언트 래퍼 모듈.

Redis Stack과의 연결을 관리하고 기본 작업을 추상화합니다.
"""

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.commands.search.commands import SearchCommands

from ..utils import get_config

if TYPE_CHECKING:
    from redis.asyncio.client import Pipeline

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 연결 및 기본 작업 관리 클래스.

    Redis Stack과의 비동기 연결을 관리하고, 연결 풀을 통해
    효율적인 리소스 관리를 제공합니다.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        max_connections: Optional[int] = None,
        decode_responses: bool = True,
    ) -> None:
        """RedisClient 초기화.

        Args:
            url: Redis 서버 URL (None이면 설정에서 가져옴)
            max_connections: 최대 연결 수 (None이면 설정에서 가져옴)
            decode_responses: 응답 자동 디코딩 여부
        """
        config = get_config()
        self.url = url or config.redis.get_connection_url()
        self.max_connections = max_connections or config.redis.max_connections
        self.decode_responses = decode_responses
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Redis 연결 설정.

        연결 풀을 생성하고 Redis 클라이언트를 초기화합니다.
        연결 성공 여부를 ping으로 확인합니다.

        Raises:
            RuntimeError: Redis 연결 실패 시
        """
        if not self._client:
            try:
                self._pool = redis.ConnectionPool.from_url(
                    self.url,
                    max_connections=self.max_connections,
                    decode_responses=self.decode_responses,
                )
                self._client = redis.Redis(connection_pool=self._pool)

                # 연결 테스트
                await self._client.ping()
                logger.info(f"Redis 연결 성공: {self.url}")
            except Exception as e:
                logger.error(f"Redis 연결 실패: {e}")
                raise RuntimeError(f"Redis 연결 실패: {e}")

    async def disconnect(self) -> None:
        """Redis 연결 종료.

        클라이언트와 연결 풀을 안전하게 종료합니다.
        """
        if self._client:
            await self._client.close()  # Redis 5.0.1부터 aclose() 권장, 하지만 타입 정의 미지원
            self._client = None

        if self._pool:
            await self._pool.disconnect()  # aclose 대신 disconnect 사용
            self._pool = None

        logger.info("Redis 연결 종료")

    async def health_check(self) -> Dict[str, Any]:
        """Redis 서버 상태 확인.

        서버 정보, 메모리 사용량, 연결된 클라이언트 수 등을
        포함한 상태 정보를 반환합니다.

        Returns:
            서버 상태 정보 딕셔너리

        Raises:
            RuntimeError: 클라이언트가 연결되지 않은 경우
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")

        info = await self._client.info()
        memory_info = await self._client.info("memory")

        return {
            "connected": True,
            "version": info.get("redis_version", "unknown"),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": memory_info.get("used_memory_human", "0B"),
            "maxmemory_human": memory_info.get("maxmemory_human", "0B"),
            "used_memory_percentage": self._calculate_memory_percentage(
                dict(memory_info)
            ),
            "modules": await self._get_loaded_modules(),
        }

    async def dbsize(self) -> int:
        """데이터베이스 내 키 개수 반환.

        Returns:
            총 키 개수
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")

        return await self._client.dbsize()

    def _calculate_memory_percentage(self, memory_info: Dict[str, Any]) -> float:
        """메모리 사용 비율 계산.

        Args:
            memory_info: Redis 메모리 정보

        Returns:
            메모리 사용 비율 (0-100)
        """
        used = float(memory_info.get("used_memory", 0))
        max_memory = float(memory_info.get("maxmemory", 0))

        if max_memory > 0:
            return round((used / max_memory) * 100, 2)
        return 0.0

    async def _get_loaded_modules(self) -> list[str]:
        """로드된 Redis 모듈 목록 조회.

        Returns:
            모듈 이름 리스트
        """
        if not self._client:
            return []

        try:
            modules = await self._client.module_list()
            return [module["name"] for module in modules]
        except Exception:
            # Redis 버전이 낮거나 모듈이 없는 경우
            return []

    @property
    def client(self) -> redis.Redis:
        """Redis 클라이언트 인스턴스 반환.

        Returns:
            Redis 클라이언트 인스턴스

        Raises:
            RuntimeError: 클라이언트가 연결되지 않은 경우
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client

    @asynccontextmanager
    async def pipeline(self, transaction: bool = True) -> AsyncIterator["Pipeline"]:
        """파이프라인 컨텍스트 매니저.

        여러 명령을 배치로 실행할 때 사용합니다.

        Args:
            transaction: 트랜잭션 사용 여부

        Yields:
            Redis 파이프라인 인스턴스
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")

        pipe = self._client.pipeline(transaction=transaction)
        try:
            yield pipe
            await pipe.execute()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    def ft(self, index_name: str) -> SearchCommands:
        """RediSearch 인덱스 명령 반환.

        Args:
            index_name: 인덱스 이름

        Returns:
            RediSearch 명령 인터페이스
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client.ft(index_name)

    def json(self) -> Any:
        """RedisJSON 명령 인터페이스 반환.

        Returns:
            RedisJSON 명령 인터페이스
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client.json()

    def xadd(self, name: str, fields: Dict[str, Any], id: str = "*", **kwargs) -> Any:
        """Redis Stream에 엔트리 추가.

        Args:
            name: 스트림 이름
            fields: 엔트리 필드들
            id: 엔트리 ID (기본값 "*"는 자동 생성)
            **kwargs: 추가 옵션

        Returns:
            생성된 엔트리 ID
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client.xadd(name, fields, id, **kwargs)

    def xread(self, streams: Dict[str, str], **kwargs) -> Any:
        """Redis Stream에서 데이터 읽기.

        Args:
            streams: 스트림 이름과 ID 매핑
            **kwargs: 추가 옵션

        Returns:
            읽은 데이터
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client.xread(streams, **kwargs)

    def pipeline(self, transaction: bool = True) -> "Pipeline":
        """파이프라인 인스턴스 반환.

        Args:
            transaction: 트랜잭션 사용 여부

        Returns:
            Redis 파이프라인 인스턴스
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client.pipeline(transaction=transaction)
