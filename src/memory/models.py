"""메모리 시스템 데이터 모델 정의.

이 모듈은 LRMM 메모리 시스템에서 사용하는 모든 데이터 모델을 정의합니다.
각 모델은 타입 안정성, 직렬화/역직렬화, 검증 로직을 포함합니다.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4
from zoneinfo import ZoneInfo

# Type variables
T = TypeVar("T")


class MemoryError(Exception):
    """Base exception for memory-related errors."""

    pass


class MemoryType(str, Enum):
    """메모리 타입 열거형.

    메모리의 종류를 구분하여 적절한 저장 방식과 처리 로직을 선택합니다.
    """

    CONVERSATION = "conversation"
    DOCUMENT = "document"
    STATE = "state"
    INSIGHT = "insight"
    SYSTEM = "system"

    @classmethod
    def from_content(cls, content: Any) -> "MemoryType":
        """콘텐츠 내용을 분석하여 적절한 메모리 타입을 추론합니다."""
        if isinstance(content, dict):
            if any(key in content for key in ["message", "role", "conversation"]):
                return cls.CONVERSATION
            elif any(key in content for key in ["checkpoint", "state", "progress"]):
                return cls.STATE
            elif any(key in content for key in ["metric", "stats", "system"]):
                return cls.SYSTEM
        return cls.DOCUMENT


class SearchType(str, Enum):
    """검색 타입 열거형.

    검색 방식을 지정하여 적절한 검색 알고리즘을 선택합니다.
    """

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    TIME_RANGE = "time_range"
    EXACT = "exact"


class Importance(str, Enum):
    """중요도 수준 열거형."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

    def to_ttl_days(self) -> Optional[int]:
        """중요도에 따른 기본 TTL(일) 반환."""
        ttl_map = {
            self.LOW: 7,
            self.NORMAL: 30,
            self.HIGH: 365,
            self.CRITICAL: None,  # 영구 보존
        }
        return ttl_map.get(self, 30)


@dataclass
class MemoryMetadata:
    """메모리 메타데이터.

    모든 메모리 항목이 공통으로 가지는 메타데이터입니다.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(
        default_factory=lambda: datetime.now(ZoneInfo("Asia/Seoul"))
    )
    updated_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    importance: Importance = Importance.NORMAL
    ttl_seconds: Optional[int] = None
    source: Optional[str] = None
    version: int = 1

    def __post_init__(self) -> None:
        """초기화 후 처리."""
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.ttl_seconds is None and self.importance:
            ttl_days = self.importance.to_ttl_days()
            if ttl_days:
                self.ttl_seconds = ttl_days * 24 * 60 * 60

    def update(self) -> None:
        """메타데이터 업데이트 시 호출."""
        self.updated_at = datetime.now(ZoneInfo("Asia/Seoul"))
        self.version += 1

    def access(self) -> None:
        """메모리 접근 시 호출."""
        self.accessed_at = datetime.now(ZoneInfo("Asia/Seoul"))
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        data = asdict(self)
        # datetime 객체를 ISO 형식 문자열로 변환
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass
class MemoryContent(Generic[T]):
    """메모리 콘텐츠 기본 모델.

    모든 메모리 타입의 기본 구조를 정의합니다.
    """

    type: MemoryType
    data: T
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)

    def validate(self) -> bool:
        """데이터 유효성 검증."""
        if not self.type:
            raise ValueError("Memory type is required")
        if self.data is None:
            raise ValueError("Memory data cannot be None")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        self.validate()
        return {
            "type": self.type.value,
            "data": (
                self.data
                if isinstance(self.data, (dict, list, str, int, float, bool))
                else str(self.data)
            ),
            "metadata": self.metadata.to_dict(),
        }

    def to_json(self) -> str:
        """JSON 문자열로 변환."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryContent":
        """딕셔너리에서 객체 생성."""
        memory_type = MemoryType(data["type"])
        metadata_dict = data.get("metadata", {})

        # datetime 문자열을 객체로 변환
        for key in ["created_at", "updated_at", "accessed_at"]:
            if key in metadata_dict and metadata_dict[key]:
                metadata_dict[key] = datetime.fromisoformat(metadata_dict[key])

        # importance 문자열을 Enum으로 변환
        if "importance" in metadata_dict and isinstance(
            metadata_dict["importance"], str
        ):
            metadata_dict["importance"] = Importance(metadata_dict["importance"])

        metadata = MemoryMetadata(**metadata_dict)
        return cls(type=memory_type, data=data["data"], metadata=metadata)


@dataclass
class ConversationMessage:
    """대화 메시지 모델.

    대화형 메모리에 저장되는 개별 메시지입니다.
    """

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(ZoneInfo("Asia/Seoul"))
    )
    context: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """메시지 유효성 검증."""
        valid_roles = {"user", "assistant", "system"}
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role: {self.role}. Must be one of {valid_roles}")
        if not self.content:
            raise ValueError("Message content cannot be empty")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        self.validate()
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class SearchQuery:
    """검색 쿼리 모델.

    메모리 검색 요청을 표현합니다.
    """

    query: str
    search_type: SearchType = SearchType.KEYWORD
    filters: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """검색 쿼리 유효성 검증."""
        # 경로가 있으면 query 없어도 OK (경로 기반 검색)
        has_paths = self.options.get("paths", [])
        if not self.query and not has_paths and self.search_type != SearchType.TIME_RANGE:
            raise ValueError("Query string or paths required for search")

        # 검색 타입별 필수 옵션 검증
        if self.search_type == SearchType.TIME_RANGE:
            if not any(
                key in self.filters for key in ["start_time", "end_time", "date", "from", "to"]
            ):
                raise ValueError(
                    "Time range search requires time filters. "
                    "Use one of: 'date' (YYYY-MM-DD), 'from'/'to' (ISO datetime), "
                    "or 'start_time'/'end_time'"
                )

        return True

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        self.validate()
        return {
            "query": self.query,
            "search_type": self.search_type.value,
            "filters": self.filters,
            "options": self.options,
        }


@dataclass
class MemoryKey:
    """메모리 키 생성 모델.

    Redis에 저장할 때 사용할 키를 생성합니다.
    """

    paths: List[str]
    memory_type: MemoryType
    prefix: str = "memory"

    def validate(self) -> bool:
        """키 유효성 검증."""
        if not self.paths:
            raise ValueError("At least one path component is required")

        # 경로 컴포넌트 검증
        for path in self.paths:
            if not path or "/" in path:
                raise ValueError(f"Invalid path component: {path}")

        return True

    def generate(self) -> str:
        """Redis 키 생성."""
        self.validate()

        # 시간 형식의 콜론을 대시로 변환 (HH:MM:SS -> HH-MM-SS)
        safe_paths = []
        for path in self.paths:
            # 시간 패턴 감지 (HH:MM:SS 또는 HH:MM:SS.mmm)
            if ":" in path and (len(path) == 8 or (len(path) == 12 and path[8] == ".")):
                safe_path = path.replace(":", "-")
            else:
                safe_path = path
            safe_paths.append(safe_path)

        # 키 형식: prefix:type:path1:path2:...
        components = [self.prefix, self.memory_type.value] + safe_paths
        return ":".join(components)

    @classmethod
    def parse(cls, key: str) -> "MemoryKey":
        """키 문자열에서 객체 생성."""
        parts = key.split(":")
        if len(parts) < 3:
            raise ValueError(f"Invalid key format: {key}")

        prefix = parts[0]
        memory_type = MemoryType(parts[1])

        # 경로 복원 (시간 형식의 대시를 콜론으로 복원)
        paths = []
        for path in parts[2:]:
            # 시간 패턴 감지 (HH-MM-SS 또는 HH-MM-SS.mmm)
            if "-" in path and (len(path) == 8 or (len(path) == 12 and path[8] == ".")):
                # 날짜 형식이 아닌 경우만 변환 (YYYY-MM-DD는 유지)
                if len(path.split("-")) == 3 and not path.startswith("20"):
                    restored_path = path.replace("-", ":")
                else:
                    restored_path = path
            else:
                restored_path = path
            paths.append(restored_path)

        return cls(paths=paths, memory_type=memory_type, prefix=prefix)


@dataclass
class SearchResult:
    """검색 결과 모델."""

    key: str
    content: MemoryContent
    score: float = 1.0
    highlights: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "key": self.key,
            "content": self.content.to_dict(),
            "score": self.score,
            "highlights": self.highlights,
            "metadata": self.metadata,
        }
