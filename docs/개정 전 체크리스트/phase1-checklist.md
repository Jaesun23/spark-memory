# LRMM Phase 1 체크리스트: 기초 환경 구축

> 기간: Day 1-3
> 목표: Redis Stack 환경과 기본 인프라 구축

## 사전 준비 사항

### Python 코딩 표준 설정
- [ ] Black (라인 길이 88자)
- [ ] isort (import 정렬)
- [ ] flake8 (코드 품질 검사)
- [ ] mypy (타입 체크)
- [ ] pre-commit hooks 설정

### 개발 환경
- [ ] Python 3.11+ 설치
- [ ] Docker Desktop 설치
- [ ] VS Code 또는 선호 IDE 설정

---

## 단위작업 1.1: 프로젝트 구조 생성

### 목표
기본 디렉토리 구조와 설정 파일을 생성하여 프로젝트 기반을 마련한다.

### 구현
```bash
# 실행 명령어
cd /Users/jason/Projects/mcp-servers/memory-one-spark
mkdir -p src/{mcp_server,memory,redis,utils} tests/{unit,integration,fixtures} docker scripts docs/{api,examples}
```

### 파일 생성 체크리스트
- [ ] `src/__init__.py`
- [ ] `src/mcp_server/__init__.py`
- [ ] `src/memory/__init__.py`
- [ ] `src/redis/__init__.py`
- [ ] `src/utils/__init__.py`
- [ ] `tests/__init__.py`
- [ ] `pyproject.toml`
- [ ] `.flake8`
- [ ] `.gitignore`
- [ ] `requirements.txt`
- [ ] `requirements-dev.txt`

### pyproject.toml 내용
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

### 유효성 검사
- [ ] 모든 디렉토리 생성 확인
- [ ] `python -c "import src"` 실행 가능
- [ ] 설정 파일 문법 검증

### 테스트
```python
# tests/test_project_structure.py
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


def test_python_import():
    """Python 모듈 임포트 테스트."""
    try:
        import src
        import src.mcp_server
        import src.memory
        import src.redis
        import src.utils
    except ImportError as e:
        pytest.fail(f"Import 실패: {e}")
```

### 완료 기준
- [ ] 모든 파일 생성 완료
- [ ] 유효성 검사 통과
- [ ] 테스트 통과
- [ ] Git 초기 커밋

---

## 단위작업 1.2: 개발 도구 설정

### 목표
코드 품질을 보장하기 위한 개발 도구들을 설정하고 검증한다.

### 구현

#### requirements-dev.txt
```
black==24.4.2
isort==5.13.2
flake8==7.0.0
mypy==1.10.0
pytest==8.2.1
pytest-asyncio==0.23.7
pytest-cov==5.0.0
pre-commit==3.7.1
```

#### .flake8
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

#### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
```

### 유효성 검사
- [ ] `pip install -r requirements-dev.txt` 성공
- [ ] `black --check src/` 통과
- [ ] `isort --check-only src/` 통과
- [ ] `flake8 src/` 에러 없음
- [ ] `pre-commit install` 성공

### 테스트
```bash
# 개발 도구 작동 테스트 스크립트
# scripts/check_code_quality.sh
#!/bin/bash
set -e

echo "🔍 Code Quality Check 시작..."

echo "1. Black 검사..."
black --check src/ tests/

echo "2. isort 검사..."
isort --check-only src/ tests/

echo "3. Flake8 검사..."
flake8 src/ tests/

echo "4. MyPy 검사..."
mypy src/

echo "✅ 모든 검사 통과!"
```

### 완료 기준
- [ ] 모든 개발 도구 설치 완료
- [ ] 코드 품질 검사 스크립트 작동
- [ ] pre-commit hooks 활성화
- [ ] 팀 개발 환경 문서화

---

## 단위작업 1.3: Redis Stack Docker 환경

### 목표
Redis Stack을 Docker로 실행하고 모든 필수 모듈이 활성화되었는지 확인한다.

### 구현

#### docker/docker-compose.yml
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

  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: lrmm-redis-commander
    restart: unless-stopped
    environment:
      - REDIS_HOSTS=local:redis-stack:6379
    ports:
      - "8081:8081"
    depends_on:
      redis-stack:
        condition: service_healthy
    networks:
      - lrmm-network

volumes:
  redis_data:
    driver: local

networks:
  lrmm-network:
    driver: bridge
```

#### docker/redis.conf
```conf
# Redis Stack 설정
port 6379
bind 0.0.0.0
protected-mode no

# 메모리 설정
maxmemory 2gb
maxmemory-policy allkeys-lru

# 지속성
save 900 1
save 300 10
save 60 10000
dbfilename dump.rdb
dir /data

# AOF
appendonly yes
appendfsync everysec

# 모듈 (Redis Stack은 자동 로드)
# loadmodule /opt/redis-stack/lib/redisearch.so
# loadmodule /opt/redis-stack/lib/redisjson.so
# loadmodule /opt/redis-stack/lib/redistimeseries.so
```

### 유효성 검사
- [ ] Docker Compose 문법 검증: `docker-compose -f docker/docker-compose.yml config`
- [ ] 포트 사용 가능 확인: `lsof -i :6379,8001,8081`
- [ ] Docker 이미지 다운로드 가능

### 테스트
```python
# tests/integration/test_redis_stack.py
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

    def test_connection(self, redis_client):
        """Redis 연결 테스트."""
        assert redis_client.ping() is True

    def test_required_modules(self, redis_client):
        """필수 모듈 로드 확인."""
        modules = redis_client.module_list()
        module_names = {m['name'].lower() for m in modules}

        required = {'search', 'timeseries', 'json'}
        missing = required - module_names

        assert not missing, f"누락된 모듈: {missing}"

    def test_json_operations(self, redis_client):
        """RedisJSON 작동 테스트."""
        key = "test:json:doc"
        doc = {
            "name": "test",
            "tags": ["memory", "lrmm"],
            "metadata": {"version": 1}
        }

        # JSON.SET
        result = redis_client.json().set(key, "$", doc)
        assert result is True

        # JSON.GET
        retrieved = redis_client.json().get(key)
        assert retrieved == doc

    def test_search_operations(self, redis_client):
        """RediSearch 작동 테스트."""
        # 인덱스 생성은 Phase 2에서 구현
        pass
```

### 실행 및 검증
```bash
# Docker 실행
cd docker
docker-compose up -d

# 상태 확인
docker-compose ps
docker-compose logs redis-stack

# Redis CLI 테스트
docker exec -it lrmm-redis redis-cli ping

# 모듈 확인
docker exec -it lrmm-redis redis-cli MODULE LIST
```

### 완료 기준
- [ ] Docker Compose로 Redis Stack 실행
- [ ] 모든 포트 정상 접근 가능
- [ ] RedisInsight UI 접근 가능 (http://localhost:8001)
- [ ] 모든 필수 모듈 활성화 확인
- [ ] 통합 테스트 통과

---

## 단위작업 1.4: Redis 클라이언트 래퍼

### 목표
Redis 연결을 관리하고 기본 작업을 추상화하는 클라이언트 래퍼를 구현한다.

### 구현

#### src/redis/client.py
```python
"""Redis 클라이언트 래퍼 모듈."""
from typing import Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
import logging

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 연결 및 기본 작업 관리 클래스.

    Attributes:
        url: Redis 연결 URL
        max_connections: 최대 연결 수
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        max_connections: int = 50
    ) -> None:
        """RedisClient 초기화.

        Args:
            url: Redis 연결 URL
            max_connections: 커넥션 풀 최대 연결 수
        """
        self.url = url
        self.max_connections = max_connections
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Redis 연결 설정.

        Raises:
            redis.ConnectionError: 연결 실패 시
        """
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
        """Redis 서버 상태 확인.

        Returns:
            서버 상태 정보

        Raises:
            RuntimeError: 연결되지 않은 상태에서 호출 시
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
            "used_memory_peak_human": memory_info.get(
                "used_memory_peak_human", "0B"
            ),
            "maxmemory_human": memory_info.get("maxmemory_human", "0B"),
        }

    @asynccontextmanager
    async def pipeline(self, transaction: bool = True) -> AsyncIterator:
        """Redis 파이프라인 컨텍스트 매니저.

        Args:
            transaction: 트랜잭션 모드 사용 여부

        Yields:
            Redis 파이프라인 객체

        Raises:
            RuntimeError: 연결되지 않은 상태에서 호출 시
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")

        async with self._client.pipeline(transaction=transaction) as pipe:
            yield pipe

    @property
    def client(self) -> redis.Redis:
        """원시 Redis 클라이언트 접근.

        Returns:
            Redis 클라이언트 인스턴스

        Raises:
            RuntimeError: 연결되지 않은 상태에서 접근 시
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        return self._client
```

#### src/redis/__init__.py
```python
"""Redis 관련 모듈."""
from .client import RedisClient

__all__ = ["RedisClient"]
```

### 유효성 검사
- [ ] Type hints 완전성 확인
- [ ] Docstring 형식 검증 (Google style)
- [ ] 예외 처리 적절성
- [ ] 로깅 구현
- [ ] Black/isort/flake8 통과

### 테스트
```python
# tests/unit/test_redis_client.py
"""Redis 클라이언트 단위 테스트."""
import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.redis import RedisClient


class TestRedisClient:
    """RedisClient 테스트."""

    @pytest.fixture
    def client(self):
        """테스트용 클라이언트."""
        return RedisClient("redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """정상 연결 테스트."""
        with patch("redis.asyncio.ConnectionPool.from_url"):
            with patch("redis.asyncio.Redis") as mock_redis:
                mock_instance = AsyncMock()
                mock_redis.return_value = mock_instance

                await client.connect()

                assert client._client is not None
                mock_instance.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """연결 실패 테스트."""
        with patch("redis.asyncio.ConnectionPool.from_url"):
            with patch("redis.asyncio.Redis") as mock_redis:
                mock_instance = AsyncMock()
                mock_instance.ping.side_effect = redis.ConnectionError()
                mock_redis.return_value = mock_instance

                with pytest.raises(redis.ConnectionError):
                    await client.connect()

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, client):
        """미연결 상태 health_check 테스트."""
        with pytest.raises(RuntimeError, match="not connected"):
            await client.health_check()

    @pytest.mark.asyncio
    async def test_pipeline_context_manager(self, client):
        """파이프라인 컨텍스트 매니저 테스트."""
        # Mock 설정
        mock_pipeline = AsyncMock()
        mock_client = AsyncMock()
        mock_client.pipeline.return_value.__aenter__.return_value = mock_pipeline
        client._client = mock_client

        # 파이프라인 사용
        async with client.pipeline() as pipe:
            assert pipe == mock_pipeline

        mock_client.pipeline.assert_called_once_with(transaction=True)
```

### 완료 기준
- [ ] 구현 완료
- [ ] 유효성 검사 통과
- [ ] 단위 테스트 작성 및 통과
- [ ] 통합 테스트 통과
- [ ] 코드 리뷰 및 리팩토링
- [ ] API 문서 작성

---

## Phase 1 완료 체크리스트

### 산출물
- [ ] 프로젝트 구조 완성
- [ ] 개발 환경 설정 완료
- [ ] Redis Stack 실행 중
- [ ] Redis 클라이언트 구현

### 품질 지표
- [ ] 코드 커버리지 > 80%
- [ ] 린터 에러 0개
- [ ] 모든 함수 타입 힌트 적용
- [ ] 모든 클래스/함수 문서화

### 다음 단계
- [ ] Phase 2 체크리스트 검토
- [ ] 팀 리뷰 미팅
- [ ] Phase 1 회고 문서 작성

---

작성일: 2025-05-27
작성자: 1호
검토자: Jason
