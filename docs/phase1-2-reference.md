# LRMM Phase 1-2 참고자료: 구현 예시 코드

> 이 문서는 Phase 1, 2 체크리스트의 구체적인 구현 예시를 담고 있습니다.
> 실제 구현 시 참고용으로 활용하세요.

## Phase 1 참고 코드

### 프로젝트 구조 생성

```bash
# 디렉토리 구조 생성 스크립트
cd /Users/jason/Projects/mcp-servers/memory-one-spark
mkdir -p src/{mcp_server,memory,redis,utils} tests/{unit,integration,fixtures} docker scripts docs/{api,examples}
```

### pyproject.toml 예시

```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### .flake8 설정

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude =
    .venv,
    __pycache__,
    .git,
    .mypy_cache,
    .pytest_cache,
    build,
    dist
per-file-ignores =
    __init__.py:F401
```

### docker-compose.yml 예시

```yaml
version: '3.8'

services:
  redis-stack:
    image: redis/redis-stack:7.2.0-v10
    container_name: lrmm-redis
    restart: unless-stopped
    ports:
      - "6379:6379"    # Redis
      - "8001:8001"    # RedisInsight
    environment:
      - REDIS_ARGS=--maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
      - ./redis.conf:/redis-stack.conf:ro
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - lrmm-network

volumes:
  redis_data:
    driver: local

networks:
  lrmm-network:
    driver: bridge
```

### RedisClient 구현 예시

```python
"""Redis 클라이언트 래퍼 모듈."""
from typing import Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
import logging

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 연결 및 기본 작업 관리 클래스."""

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        max_connections: int = 50
    ) -> None:
        """RedisClient 초기화."""
        self.url = url
        self.max_connections = max_connections
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Redis 연결 설정."""
        if not self._client:
            self._pool = redis.ConnectionPool.from_url(
                self.url,
                max_connections=self.max_connections,
                decode_responses=True
            )
            self._client = redis.Redis(connection_pool=self._pool)

            # 연결 테스트
            await self._client.ping()
            logger.info(f"Redis 연결 성공: {self.url}")

    async def disconnect(self) -> None:
        """Redis 연결 종료."""
        if self._client:
            await self._client.close()
            self._client = None

        if self._pool:
            await self._pool.disconnect()
            self._pool = None

        logger.info("Redis 연결 종료")

    async def health_check(self) -> Dict[str, Any]:
        """Redis 서버 상태 확인."""
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
        }
```

## Phase 2 참고 코드

### TimePathGenerator 구현 예시

```python
"""시간 관련 유틸리티 함수."""
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from zoneinfo import ZoneInfo
import re

# 시간 경로 패턴
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
TIME_PATTERN = re.compile(r'^\d{2}:\d{2}:\d{2}$')


class TimePathGenerator:
    """시간 기반 경로 생성 및 파싱 클래스."""

    def __init__(self, default_timezone: str = "Asia/Seoul") -> None:
        """TimePathGenerator 초기화."""
        self.default_timezone = default_timezone

    def generate_path(
        self,
        category: str = "",
        timestamp: Optional[datetime] = None,
        include_microseconds: bool = False
    ) -> str:
        """현재 시간 기반 경로 생성."""
        if timestamp is None:
            timestamp = datetime.now(ZoneInfo(self.default_timezone))
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo(self.default_timezone))

        date_part = timestamp.strftime("%Y-%m-%d")

        if include_microseconds:
            time_part = timestamp.strftime("%H:%M:%S.%f")[:-3]
        else:
            time_part = timestamp.strftime("%H:%M:%S")

        parts = [date_part, time_part]
        if category:
            parts.append(category)

        return "/".join(parts)
```

### 메모리 모델 예시

```python
"""메모리 시스템 데이터 모델."""
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(str, Enum):
    """메모리 타입 열거형."""

    CONVERSATION = "conversation"
    DOCUMENT = "document"
    STATE = "state"
    INSIGHT = "insight"
    SYSTEM = "system"


@dataclass
class MemoryContent:
    """메모리 콘텐츠 기본 모델."""

    type: MemoryType
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """초기화 후 처리."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "type": self.type.value,
            "data": self.data,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
```

### MemoryEngine 핵심 메서드 예시

```python
async def save(
    self,
    paths: List[str],
    content: Any,
    options: Optional[Dict[str, Any]] = None
) -> str:
    """메모리 저장."""
    options = options or {}

    # 메모리 타입 결정
    memory_type = self._determine_memory_type(content, options)

    # 시간 경로 생성
    if not paths or paths == [""]:
        category = options.get("category", memory_type.value)
        time_path = self.time_gen.generate_path(category)
        paths = [time_path]

    # 키 생성
    key = self._generate_key(paths, memory_type)

    # 저장 방식 결정
    if memory_type == MemoryType.CONVERSATION:
        return await self._save_conversation(key, content, options)
    else:
        return await self._save_json(key, content, options)
```

## 테스트 코드 예시

### 프로젝트 구조 검증 테스트

```python
import os
import sys
from pathlib import Path


def test_project_structure():
    """프로젝트 구조 검증."""
    root = Path(__file__).parent.parent

    # 필수 디렉토리
    required_dirs = [
        "src/mcp_server",
        "src/memory",
        "src/redis",
        "src/utils",
        "tests/unit",
        "tests/integration",
        "docker",
    ]

    for dir_path in required_dirs:
        assert (root / dir_path).exists(), f"{dir_path} 없음"
        assert (root / dir_path / "__init__.py").exists()
```

### Redis 통합 테스트

```python
import pytest
import redis
import json


class TestRedisStack:
    """Redis Stack 통합 테스트."""

    @pytest.fixture
    def redis_client(self):
        """Redis 클라이언트 fixture."""
        client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        yield client
        # 테스트 데이터 정리
        client.flushdb()

    def test_required_modules(self, redis_client):
        """필수 모듈 로드 확인."""
        modules = redis_client.module_list()
        module_names = {m['name'].lower() for m in modules}

        required = {'search', 'timeseries', 'json'}
        missing = required - module_names

        assert not missing, f"누락된 모듈: {missing}"
```

---

> 이 참고자료는 실제 구현 시 가이드로 활용하되, 프로젝트의 특성과 요구사항에 맞게 수정하여 사용하세요. 😊
