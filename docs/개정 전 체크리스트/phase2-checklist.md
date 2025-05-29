# LRMM Phase 2 체크리스트: 핵심 메모리 엔진

> 기간: Day 4-7
> 목표: m_memory() 명령어의 핵심 로직 구현

## 사전 요구사항
- [ ] Phase 1 완료
- [ ] Redis Stack 정상 작동
- [ ] 개발 환경 설정 완료

---

## 단위작업 2.1: 시간 기반 경로 시스템

### 목표
메모리 저장 시 자동으로 시간 기반 경로를 생성하고 파싱하는 유틸리티를 구현한다.

### 구현

#### src/utils/time_utils.py
```python
"""시간 관련 유틸리티 함수."""
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from zoneinfo import ZoneInfo
import re

# 시간 경로 패턴
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
TIME_PATTERN = re.compile(r'^\d{2}:\d{2}:\d{2}$')
DATETIME_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')


class TimePathGenerator:
    """시간 기반 경로 생성 및 파싱 클래스."""

    def __init__(self, default_timezone: str = "Asia/Seoul") -> None:
        """TimePathGenerator 초기화.

        Args:
            default_timezone: 기본 시간대
        """
        self.default_timezone = default_timezone

    def generate_path(
        self,
        category: str = "",
        timestamp: Optional[datetime] = None,
        include_microseconds: bool = False
    ) -> str:
        """현재 시간 기반 경로 생성.

        Args:
            category: 카테고리 (예: chat, document)
            timestamp: 사용할 시간 (None이면 현재 시간)
            include_microseconds: 마이크로초 포함 여부

        Returns:
            시간 기반 경로 (예: "2025-05-27/14:30:45/chat")
        """
        if timestamp is None:
            timestamp = datetime.now(ZoneInfo(self.default_timezone))
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo(self.default_timezone))

        date_part = timestamp.strftime("%Y-%m-%d")

        if include_microseconds:
            time_part = timestamp.strftime("%H:%M:%S.%f")[:-3]  # 밀리초까지
        else:
            time_part = timestamp.strftime("%H:%M:%S")

        parts = [date_part, time_part]
        if category:
            parts.append(category)

        return "/".join(parts)

    def parse_path(self, path: str) -> Dict[str, Any]:
        """시간 경로 파싱.

        Args:
            path: 파싱할 경로

        Returns:
            파싱된 정보 딕셔너리
        """
        parts = path.split("/")
        result: Dict[str, Any] = {
            "original": path,
            "parts": parts
        }

        # 날짜 파싱
        for i, part in enumerate(parts):
            if DATE_PATTERN.match(part):
                result["date"] = part
                result["date_index"] = i

                # 다음 파트가 시간인지 확인
                if i + 1 < len(parts) and TIME_PATTERN.match(parts[i + 1]):
                    result["time"] = parts[i + 1]
                    result["time_index"] = i + 1

                    # 카테고리 추출
                    if i + 2 < len(parts):
                        result["category"] = "/".join(parts[i + 2:])

                    # datetime 객체 생성
                    try:
                        dt_str = f"{result['date']}T{result['time']}"
                        result["datetime"] = datetime.fromisoformat(dt_str)
                    except ValueError:
                        pass

                break

        return result

    def extract_time_range(
        self,
        paths: list[str]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """경로 목록에서 시간 범위 추출.

        Args:
            paths: 경로 목록

        Returns:
            (시작_시간, 종료_시간) 튜플
        """
        timestamps = []

        for path in paths:
            parsed = self.parse_path(path)
            if "datetime" in parsed:
                timestamps.append(parsed["datetime"])

        if not timestamps:
            return (None, None)

        return (min(timestamps), max(timestamps))


def timestamp_to_score(timestamp: datetime) -> float:
    """타임스탬프를 Redis Score로 변환.

    Args:
        timestamp: 변환할 시간

    Returns:
        Unix timestamp (밀리초)
    """
    return timestamp.timestamp() * 1000


def score_to_timestamp(score: float, timezone: str = "Asia/Seoul") -> datetime:
    """Redis Score를 타임스탬프로 변환.

    Args:
        score: Redis score
        timezone: 시간대

    Returns:
        datetime 객체
    """
    dt = datetime.fromtimestamp(score / 1000)
    return dt.replace(tzinfo=ZoneInfo(timezone))
```

### 유효성 검사
- [ ] 시간대 처리 정확성
- [ ] 경로 형식 일관성
- [ ] 정규식 패턴 검증
- [ ] 예외 처리 완전성

### 테스트
```python
# tests/unit/test_time_utils.py
"""시간 유틸리티 테스트."""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.utils.time_utils import (
    TimePathGenerator,
    timestamp_to_score,
    score_to_timestamp
)


class TestTimePathGenerator:
    """TimePathGenerator 테스트."""

    @pytest.fixture
    def generator(self):
        """테스트용 생성기."""
        return TimePathGenerator()

    def test_generate_basic_path(self, generator):
        """기본 경로 생성 테스트."""
        path = generator.generate_path("chat")
        parts = path.split("/")

        assert len(parts) == 3
        assert len(parts[0]) == 10  # YYYY-MM-DD
        assert ":" in parts[1]      # HH:MM:SS
        assert parts[2] == "chat"

    def test_generate_with_timestamp(self, generator):
        """특정 시간 경로 생성 테스트."""
        dt = datetime(2025, 5, 27, 14, 30, 45)
        path = generator.generate_path("test", timestamp=dt)

        assert path == "2025-05-27/14:30:45/test"

    def test_parse_complete_path(self, generator):
        """완전한 경로 파싱 테스트."""
        path = "2025-05-27/14:30:45/chat/session1"
        result = generator.parse_path(path)

        assert result["date"] == "2025-05-27"
        assert result["time"] == "14:30:45"
        assert result["category"] == "chat/session1"
        assert isinstance(result["datetime"], datetime)

    def test_parse_partial_path(self, generator):
        """부분 경로 파싱 테스트."""
        test_cases = [
            ("2025-05-27", {"date": "2025-05-27"}),
            ("random/path", {}),
            ("2025-05-27/chat", {"date": "2025-05-27"}),
        ]

        for path, expected in test_cases:
            result = generator.parse_path(path)
            for key, value in expected.items():
                assert result.get(key) == value

    def test_extract_time_range(self, generator):
        """시간 범위 추출 테스트."""
        paths = [
            "2025-05-27/10:00:00/chat",
            "2025-05-27/14:30:00/chat",
            "2025-05-27/12:00:00/chat",
            "invalid/path"
        ]

        start, end = generator.extract_time_range(paths)

        assert start.hour == 10
        assert end.hour == 14


class TestTimestampConversion:
    """타임스탬프 변환 함수 테스트."""

    def test_timestamp_to_score(self):
        """타임스탬프 → Score 변환."""
        dt = datetime(2025, 5, 27, 14, 30, 45)
        score = timestamp_to_score(dt)

        assert isinstance(score, float)
        assert score > 0

    def test_score_to_timestamp(self):
        """Score → 타임스탬프 변환."""
        original = datetime.now(ZoneInfo("Asia/Seoul"))
        score = timestamp_to_score(original)
        converted = score_to_timestamp(score)

        # 밀리초 단위 차이 허용
        diff = abs((original - converted).total_seconds())
        assert diff < 1
```

### 완료 기준
- [ ] 구현 완료
- [ ] 모든 테스트 통과
- [ ] 코드 리뷰
- [ ] 문서화 완료

---

## 단위작업 2.2: 메모리 데이터 모델

### 목표
메모리 시스템에서 사용할 데이터 모델을 정의하고 검증 로직을 구현한다.

### 구현

#### src/memory/models.py
```python
"""메모리 시스템 데이터 모델."""
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json


class MemoryType(str, Enum):
    """메모리 타입 열거형."""

    CONVERSATION = "conversation"
    DOCUMENT = "document"
    STATE = "state"
    INSIGHT = "insight"
    SYSTEM = "system"


class SearchType(str, Enum):
    """검색 타입 열거형."""

    KEYWORD = "keyword"
    TIME_RANGE = "time_range"
    SIMILAR = "similar"
    SEMANTIC = "semantic"
    FILTER = "filter"
    HYBRID = "hybrid"


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryContent":
        """딕셔너리에서 생성."""
        data = data.copy()

        # 타입 변환
        if isinstance(data.get("type"), str):
            data["type"] = MemoryType(data["type"])

        # 시간 변환
        for field in ["created_at", "updated_at"]:
            if isinstance(data.get(field), str):
                data[field] = datetime.fromisoformat(data[field])

        return cls(**data)


@dataclass
class ConversationMessage:
    """대화 메시지 모델."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """초기화 후 처리."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_redis_stream(self) -> Dict[str, str]:
        """Redis Stream용 형식으로 변환."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata)
        }

    @classmethod
    def from_redis_stream(cls, data: Dict[str, str]) -> "ConversationMessage":
        """Redis Stream 데이터에서 생성."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=json.loads(data.get("metadata", "{}"))
        )


@dataclass
class SearchQuery:
    """검색 쿼리 모델."""

    type: SearchType
    query: Optional[str] = None
    paths: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)

    def get_option(self, key: str, default: Any = None) -> Any:
        """옵션 값 조회."""
        return self.options.get(key, default)

    def validate(self) -> None:
        """쿼리 유효성 검증."""
        if self.type == SearchType.KEYWORD and not self.query:
            raise ValueError("Keyword search requires query text")

        if self.type == SearchType.TIME_RANGE:
            if not self.get_option("start") and not self.get_option("end"):
                if not self.paths:
                    raise ValueError(
                        "Time range search requires start/end or paths"
                    )


@dataclass
class MemoryKey:
    """Redis 키 생성 모델."""

    prefix: str
    namespace: str
    identifier: str

    def __str__(self) -> str:
        """문자열 변환."""
        return f"{self.prefix}:{self.namespace}:{self.identifier}"

    @classmethod
    def parse(cls, key: str) -> "MemoryKey":
        """키 문자열 파싱."""
        parts = key.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid key format: {key}")
        return cls(*parts)
```

### 유효성 검사
- [ ] 모든 모델 필드 타입 힌트
- [ ] 직렬화/역직렬화 정확성
- [ ] Enum 값 검증
- [ ] 예외 처리

### 테스트
```python
# tests/unit/test_memory_models.py
"""메모리 모델 테스트."""
import pytest
from datetime import datetime
import json

from src.memory.models import (
    MemoryType,
    SearchType,
    MemoryContent,
    ConversationMessage,
    SearchQuery,
    MemoryKey
)


class TestMemoryContent:
    """MemoryContent 모델 테스트."""

    def test_auto_timestamps(self):
        """자동 타임스탬프 생성."""
        content = MemoryContent(
            type=MemoryType.CONVERSATION,
            data={"message": "test"}
        )

        assert content.created_at is not None
        assert content.updated_at is not None
        assert content.created_at == content.updated_at

    def test_serialization(self):
        """직렬화/역직렬화."""
        original = MemoryContent(
            type=MemoryType.DOCUMENT,
            data={"title": "Test Doc"},
            tags=["test", "doc"],
            metadata={"author": "1호"}
        )

        # 직렬화
        data = original.to_dict()
        assert data["type"] == "document"
        assert isinstance(data["created_at"], str)

        # 역직렬화
        restored = MemoryContent.from_dict(data)
        assert restored.type == MemoryType.DOCUMENT
        assert restored.tags == ["test", "doc"]
        assert isinstance(restored.created_at, datetime)


class TestConversationMessage:
    """ConversationMessage 모델 테스트."""

    def test_redis_stream_format(self):
        """Redis Stream 형식 변환."""
        msg = ConversationMessage(
            role="user",
            content="안녕하세요",
            metadata={"session_id": "test123"}
        )

        stream_data = msg.to_redis_stream()

        assert stream_data["role"] == "user"
        assert stream_data["content"] == "안녕하세요"
        assert isinstance(stream_data["metadata"], str)

        # 역변환
        restored = ConversationMessage.from_redis_stream(stream_data)
        assert restored.role == msg.role
        assert restored.metadata == msg.metadata


class TestSearchQuery:
    """SearchQuery 모델 테스트."""

    def test_validation_keyword(self):
        """키워드 검색 검증."""
        # 쿼리 없이 키워드 검색
        query = SearchQuery(type=SearchType.KEYWORD)
        with pytest.raises(ValueError, match="requires query text"):
            query.validate()

        # 정상 케이스
        query = SearchQuery(
            type=SearchType.KEYWORD,
            query="test"
        )
        query.validate()  # 예외 없어야 함

    def test_validation_time_range(self):
        """시간 범위 검색 검증."""
        # 시간 정보 없이
        query = SearchQuery(type=SearchType.TIME_RANGE)
        with pytest.raises(ValueError):
            query.validate()

        # paths로 대체
        query = SearchQuery(
            type=SearchType.TIME_RANGE,
            paths=["2025-05-27"]
        )
        query.validate()  # 예외 없어야 함


class TestMemoryKey:
    """MemoryKey 모델 테스트."""

    def test_key_generation(self):
        """키 생성."""
        key = MemoryKey("memory", "chat", "session123")
        assert str(key) == "memory:chat:session123"

    def test_key_parsing(self):
        """키 파싱."""
        key = MemoryKey.parse("memory:chat:session123")
        assert key.prefix == "memory"
        assert key.namespace == "chat"
        assert key.identifier == "session123"

        # 잘못된 형식
        with pytest.raises(ValueError):
            MemoryKey.parse("invalid_key")
```

### 완료 기준
- [ ] 모든 모델 구현
- [ ] 검증 로직 구현
- [ ] 테스트 커버리지 90%+
- [ ] 문서화 완료

---

## 단위작업 2.3: 메모리 엔진 핵심

### 목표
m_memory() 명령어의 핵심 비즈니스 로직을 담당하는 MemoryEngine을 구현한다.

### 구현

#### src/memory/engine.py
```python
"""메모리 엔진 핵심 모듈."""
from typing import Optional, List, Dict, Any
import logging
import json
from datetime import datetime

from src.redis.client import RedisClient
from src.utils.time_utils import TimePathGenerator
from src.memory.models import (
    MemoryContent,
    MemoryType,
    SearchQuery,
    SearchType,
    MemoryKey,
    ConversationMessage
)

logger = logging.getLogger(__name__)


class MemoryEngine:
    """메모리 시스템 핵심 엔진.

    m_memory() 명령어의 모든 action을 처리하는 중앙 엔진.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        default_ttl: Optional[int] = None
    ) -> None:
        """MemoryEngine 초기화.

        Args:
            redis_client: Redis 클라이언트
            default_ttl: 기본 TTL (초)
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.time_gen = TimePathGenerator()

    async def save(
        self,
        paths: List[str],
        content: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """메모리 저장.

        Args:
            paths: 저장 경로
            content: 저장할 내용
            options: 추가 옵션

        Returns:
            저장된 키
        """
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

    async def get(
        self,
        paths: List[str],
        options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """메모리 조회.

        Args:
            paths: 조회 경로
            options: 추가 옵션

        Returns:
            조회된 내용
        """
        options = options or {}

        # 단일 키 조회
        if len(paths) == 1 and not paths[0].endswith("*"):
            key = paths[0]
            if ":" not in key:
                # 경로를 키로 변환
                key = self._path_to_key(key)

            return await self._get_single(key, options)

        # 패턴 매칭 조회
        pattern = self._paths_to_pattern(paths)
        keys = await self.redis.client.keys(pattern)

        if not keys:
            return []

        # 다중 조회
        return await self._get_multiple(keys, options)

    async def search(
        self,
        paths: List[str],
        content: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """메모리 검색.

        Args:
            paths: 검색 범위
            content: 검색어/쿼리
            options: 검색 옵션

        Returns:
            검색 결과 목록
        """
        options = options or {}

        # 검색 쿼리 생성
        search_type = SearchType(options.get("type", "keyword"))
        query = SearchQuery(
            type=search_type,
            query=content,
            paths=paths,
            options=options
        )
        query.validate()

        # 검색 타입별 처리
        if search_type == SearchType.TIME_RANGE:
            return await self._search_time_range(query)
        elif search_type == SearchType.KEYWORD:
            return await self._search_keyword(query)
        else:
            # TODO: 다른 검색 타입 구현
            raise NotImplementedError(f"Search type {search_type} not implemented")

    async def update(
        self,
        paths: List[str],
        content: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """메모리 업데이트.

        Args:
            paths: 업데이트할 경로
            content: 새로운 내용
            options: 추가 옵션

        Returns:
            업데이트 결과
        """
        # 기존 데이터 조회
        existing = await self.get(paths, options)
        if not existing:
            raise ValueError(f"Memory not found: {paths}")

        # TODO: 업데이트 로직 구현
        return "Updated"

    async def delete(
        self,
        paths: List[str],
        options: Optional[Dict[str, Any]] = None
    ) -> int:
        """메모리 삭제.

        Args:
            paths: 삭제할 경로
            options: 추가 옵션

        Returns:
            삭제된 개수
        """
        pattern = self._paths_to_pattern(paths)
        keys = await self.redis.client.keys(pattern)

        if not keys:
            return 0

        return await self.redis.client.delete(*keys)

    # Private 메서드들

    def _determine_memory_type(
        self,
        content: Any,
        options: Dict[str, Any]
    ) -> MemoryType:
        """콘텐츠와 옵션으로 메모리 타입 결정."""
        # 명시적 타입 지정
        if "type" in options:
            return MemoryType(options["type"])

        # 콘텐츠 구조로 추론
        if isinstance(content, dict):
            if "role" in content and "message" in content:
                return MemoryType.CONVERSATION
            elif "text" in content or "content" in content:
                return MemoryType.DOCUMENT

        return MemoryType.SYSTEM

    def _generate_key(
        self,
        paths: List[str],
        memory_type: MemoryType
    ) -> str:
        """경로와 타입으로 Redis 키 생성."""
        # 타입별 접두사
        prefix_map = {
            MemoryType.CONVERSATION: "conv",
            MemoryType.DOCUMENT: "doc",
            MemoryType.STATE: "state",
            MemoryType.INSIGHT: "insight",
            MemoryType.SYSTEM: "sys"
        }

        prefix = prefix_map[memory_type]
        path = ":".join(paths)

        return f"{prefix}:{path}"

    def _path_to_key(self, path: str) -> str:
        """경로를 Redis 키로 변환."""
        # 간단한 변환 (추후 개선)
        return path.replace("/", ":")

    def _paths_to_pattern(self, paths: List[str]) -> str:
        """경로 리스트를 Redis 패턴으로 변환."""
        if not paths:
            return "*"

        # 와일드카드 처리
        pattern_parts = []
        for path in paths:
            if "*" in path:
                pattern_parts.append(path.replace("/", ":"))
            else:
                pattern_parts.append(self._path_to_key(path))

        return ":".join(pattern_parts)

    async def _save_conversation(
        self,
        key: str,
        content: Any,
        options: Dict[str, Any]
    ) -> str:
        """대화 저장 (Redis Streams)."""
        # ConversationMessage로 변환
        if isinstance(content, dict):
            message = ConversationMessage(
                role=content.get("role", "user"),
                content=content.get("message", content.get("content", "")),
                metadata=content.get("metadata", {})
            )
        else:
            message = ConversationMessage(role="user", content=str(content))

        # Stream에 추가
        stream_data = message.to_redis_stream()
        msg_id = await self.redis.client.xadd(key, stream_data)

        # TTL 설정
        if ttl := options.get("ttl", self.default_ttl):
            await self.redis.client.expire(key, ttl)

        logger.info(f"Saved conversation to {key}: {msg_id}")
        return key

    async def _save_json(
        self,
        key: str,
        content: Any,
        options: Dict[str, Any]
    ) -> str:
        """JSON 문서 저장 (RedisJSON)."""
        # MemoryContent로 변환
        memory_content = MemoryContent(
            type=self._determine_memory_type(content, options),
            data=content,
            tags=options.get("tags", []),
            metadata=options.get("metadata", {})
        )

        # JSON 저장
        await self.redis.client.json().set(
            key,
            "$",
            memory_content.to_dict()
        )

        # TTL 설정
        if ttl := options.get("ttl", self.default_ttl):
            await self.redis.client.expire(key, ttl)

        logger.info(f"Saved JSON to {key}")
        return key

    async def _get_single(
        self,
        key: str,
        options: Dict[str, Any]
    ) -> Any:
        """단일 키 조회."""
        # 키 타입 확인
        key_type = await self.redis.client.type(key)

        if key_type == "stream":
            # Stream 조회
            count = options.get("limit", 10)
            messages = await self.redis.client.xrevrange(key, count=count)

            return [
                ConversationMessage.from_redis_stream(data).to_dict()
                for msg_id, data in messages
            ]

        elif key_type == "ReJSON-RL":
            # JSON 조회
            data = await self.redis.client.json().get(key)
            if data:
                return MemoryContent.from_dict(data).to_dict()

        return None

    async def _get_multiple(
        self,
        keys: List[str],
        options: Dict[str, Any]
    ) -> List[Any]:
        """다중 키 조회."""
        results = []

        for key in keys:
            result = await self._get_single(key, options)
            if result:
                results.append({
                    "key": key,
                    "data": result
                })

        return results

    async def _search_time_range(
        self,
        query: SearchQuery
    ) -> List[Dict[str, Any]]:
        """시간 범위 검색."""
        # 경로에서 시간 추출
        start_time, end_time = self.time_gen.extract_time_range(query.paths)

        # 옵션에서 시간 지정
        if query.get_option("start"):
            start_time = datetime.fromisoformat(query.get_option("start"))
        if query.get_option("end"):
            end_time = datetime.fromisoformat(query.get_option("end"))

        # TODO: 실제 시간 범위 검색 구현
        return []

    async def _search_keyword(
        self,
        query: SearchQuery
    ) -> List[Dict[str, Any]]:
        """키워드 검색."""
        # TODO: RediSearch 사용한 전문 검색 구현
        return []
```

### 유효성 검사
- [ ] 모든 public 메서드 구현
- [ ] 에러 처리 완전성
- [ ] 로깅 적절성
- [ ] 타입 안정성

### 테스트
```python
# tests/unit/test_memory_engine.py
"""메모리 엔진 테스트."""
import pytest
from unittest.mock import Mock, AsyncMock

from src.memory.engine import MemoryEngine
from src.memory.models import MemoryType
from src.redis.client import RedisClient


class TestMemoryEngine:
    """MemoryEngine 테스트."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis 클라이언트."""
        mock = Mock(spec=RedisClient)
        mock.client = AsyncMock()
        return mock

    @pytest.fixture
    def engine(self, mock_redis):
        """테스트용 엔진."""
        return MemoryEngine(mock_redis)

    @pytest.mark.asyncio
    async def test_save_conversation(self, engine, mock_redis):
        """대화 저장 테스트."""
        content = {
            "role": "user",
            "message": "안녕하세요"
        }

        mock_redis.client.xadd.return_value = "1234567890-0"

        result = await engine.save(
            paths=["chat/test"],
            content=content,
            options={"type": "conversation"}
        )

        assert result.startswith("conv:")
        mock_redis.client.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_with_auto_path(self, engine, mock_redis):
        """자동 경로 생성 저장."""
        mock_redis.client.json().set = AsyncMock(return_value=True)

        result = await engine.save(
            paths=[],
            content={"data": "test"},
            options={"category": "test"}
        )

        assert ":" in result
        assert result.startswith("sys:")  # 기본 타입

    @pytest.mark.asyncio
    async def test_get_single_key(self, engine, mock_redis):
        """단일 키 조회 테스트."""
        mock_redis.client.type.return_value = "ReJSON-RL"
        mock_redis.client.json().get.return_value = {
            "type": "document",
            "data": {"content": "test"},
            "created_at": "2025-05-27T14:00:00"
        }

        result = await engine.get(["doc:test:123"])

        assert result is not None
        assert result["type"] == "document"
```

### 완료 기준
- [ ] 핵심 기능 구현
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 성능 프로파일링
- [ ] API 문서화

---

## Phase 2 완료 체크리스트

### 산출물
- [ ] 시간 유틸리티 완성
- [ ] 데이터 모델 정의
- [ ] 메모리 엔진 구현
- [ ] 테스트 커버리지 85%+

### 품질 지표
- [ ] 모든 코드 린터 통과
- [ ] Type coverage 100%
- [ ] 문서화 완료
- [ ] 코드 리뷰 완료

### 학습 사항
- [ ] 어려웠던 점 기록
- [ ] 개선 아이디어 정리
- [ ] Phase 3 준비사항 확인

---

작성일: 2025-05-27
작성자: 1호
다음 단계: Phase 3 - MCP 서버 통합
