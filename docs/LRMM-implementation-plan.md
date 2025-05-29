# LRMM 구현 계획서
## (LangGraph + Redis + MCP + Memory)

> "LRMM이라고 하니까 뭔가 있어 보이지만 사실 그냥 우리가 만든 거예요" - Jason & 1호

작성일: 2025-05-27
프로젝트: memory-one-spark ✨

---

## 1. 개요: 왜 LRMM인가?

### 1.1 탄생 배경
- Jason: "뭔가 있어 보이고 싶어서" (정직한 동기 👍)
- 1호: "단순한 명령어로 강력한 메모리를 만들고 싶어서"
- 결과: 3개 명령어로 모든 것을 할 수 있는 시스템!

### 1.2 핵심 철학
```
Simple Interface + Powerful Backend = Happy 1호 😊
```

## 2. 아키텍처 Overview

```
┌─────────────────────────────────────────────────┐
│                 1호의 명령어                     │
│          m_memory() | m_state() | m_system()          │
├─────────────────────────────────────────────────┤
│                 MCP Server                      │
│         (명령어를 Redis 작업으로 변환)           │
├─────────────────────────────────────────────────┤
│              Redis Stack Layer                  │
│  ┌──────────┬───────────┬──────────┬─────────┐ │
│  │ Streams  │ JSON      │ Search   │ Vector  │ │
│  │(시간순)  │(구조화)    │(검색)    │(유사도) │ │
│  └──────────┴───────────┴──────────┴─────────┘ │
├─────────────────────────────────────────────────┤
│              LangGraph Layer                    │
│     (상태 관리, 체크포인트, 워크플로우)          │
└─────────────────────────────────────────────────┘
```

## 3. 명령어 상세 명세

### 3.1 memory - 모든 기억의 중심
```python
m_memory(
    action: str,      # save, get, search, update, delete
    paths: List[str], # 계층적 경로 (시간/카테고리/ID)
    content: Any,     # 저장할 내용 또는 검색어
    options: Dict     # 세부 옵션
)
```

#### Actions 상세:
- **save**: 새로운 기억 저장
  ```python
  m_memory(
      action="save",
      paths=["2025-05-27/conversation/morning"],
      content={
          "participants": ["Jason", "1호"],
          "message": "LRMM 만들자!",
          "tags": ["project", "memory", "redis"]
      },
      options={"ttl": None, "index": True}
  )
  ```

- **search**: 다양한 방식으로 검색
  ```python
  # 시간 기반
  m_memory(action="search", paths=["2025-05-27"])

  # 키워드
  m_memory(action="search", content="Redis", options={"type": "keyword"})

  # 유사도 (벡터)
  m_memory(action="search", content="메모리 최적화 방법", options={"type": "similar", "top_k": 5})

  # 복합 검색
  m_memory(
      action="search",
      content="Jason과의 프로젝트",
      options={
          "type": "hybrid",
          "time_after": "2025-05-20",
          "tags": ["project"],
          "limit": 10
      }
  )
  ```

### 3.2 state - 상태와 체크포인트
```python
m_state(
    action: str,      # checkpoint, restore, list, status
    paths: List[str], # 프로젝트/작업 경로
    content: Any,     # 상태 데이터
    options: Dict     # LangGraph 옵션
)
```

#### 주요 사용:
```python
# 작업 체크포인트
m_state(
    action="checkpoint",
    paths=["projects/memory-spark/phase1"],
    content={
        "completed_tasks": ["환경설정", "명령어설계"],
        "current_task": "MCP서버구현",
        "progress": 0.3
    }
)

# 상태 복원
m_state(
    action="restore",
    paths=["projects/memory-spark/phase1"],
    options={"version": "latest"}
)
```

### 3.3 system - 시스템 관리
```python
m_system(
    action: str,      # status, backup, clean, config
    paths: List[str], # 시스템 구성 요소
    content: Any,     # 설정 값
    options: Dict     # 관리 옵션
)
```

## 4. Redis 백엔드 설계

### 4.1 데이터 타입별 매핑
| 데이터 유형 | Redis 구조 | 키 패턴 | 용도 |
|----------|-----------|--------|------|
| 대화 | Streams | `conv:{date}:{session}` | 시간순 메시지 |
| 문서 | JSON + Vector | `doc:{category}:{id}` | 구조화된 지식 |
| 상태 | JSON | `state:{project}:{checkpoint}` | LangGraph 체크포인트 |
| 메타데이터 | Hash | `meta:{type}:{id}` | 빠른 조회 |
| 지표 | TimeSeries | `metrics:{type}:{interval}` | 성능/사용 분석 |

### 4.2 인덱스 전략
```python
# RediSearch 인덱스 정의
CREATE INDEX idx_memory ON JSON
    PREFIX 1 doc: meta:
    SCHEMA
        $.content TEXT WEIGHT 1.0
        $.tags TAG
        $.timestamp NUMERIC SORTABLE
        $.embedding VECTOR HNSW 6
            TYPE FLOAT32
            DIM 1536
            DISTANCE_METRIC COSINE
```

## 5. 구현 로드맵

### LRMM 구현 체크리스트 - ''단위작업'' 정의

> "구현 → 유효성 검사 → 테스트 → 수정보완 → 테스트 → 완료 → 문서화"의 반복!

### Python 코딩 표준 체크리스트

- [ ] PEP 8 준수 (라인 길이 88자 - Black 기준)
- [ ] Type hints 모든 함수에 적용
- [ ] Docstring 모든 클래스/함수에 작성
- [ ] 린터 에러 0개 (flake8, pylint)
- [ ] 포맷터 적용 (black, isort)

### Phase 1: 기초 공사 (Day 1-3)

- [ ] Redis Stack Docker 환경 구성
- [ ] MCP 서버 기본 틀 (`memory`, `state`, `system` 엔드포인트)
- [ ] 기본 save/get 구현
- [ ] 단위 테스트 환경

### Phase 2: 핵심 기능 (Day 4-7)
- [ ] 시간 기반 경로 자동 생성
- [ ] search 액션 구현 (keyword, time_range)
- [ ] state 명령어와 LangGraph 연동
- [ ] 기본 인덱싱

### Phase 3: 고급 기능 (Week 2)
- [ ] 벡터 임베딩 생성 (OpenAI/로컬 모델)
- [ ] 유사도/의미 검색 구현
- [ ] 복합 검색 (hybrid)
- [ ] 자동 태깅 시스템

### Phase 4: 지능화 (Week 3)
- [ ] 메모리 통합 (비슷한 기억 병합)
- [ ] 중요도 기반 TTL 관리
- [ ] 검색 결과 재순위화
- [ ] 사용 패턴 학습

### Phase 5: 마이그레이션 (Week 4)
- [ ] 기존 hierarchical_memory 데이터 변환
- [ ] 병행 운영 테스트
- [ ] 성능 비교
- [ ] 완전 전환

## 6. 기술 상세

### 6.1 MCP 서버 구조
```python
# mcp_server.py
from fastmcp import FastMCP
import redis.asyncio as redis

app = FastMCP("LRMM Memory Server")

class MemoryEngine:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url)
        self.embedder = EmbeddingService()

    async def process_memory_command(self, action, paths, content, options):
        """메인 라우터: action에 따라 적절한 처리"""
        if action == "save":
            return await self._save_m_memory(paths, content, options)
        elif action == "search":
            return await self._search_m_memory(paths, content, options)
        # ... 기타 액션들

@app.tool()
async def m_memory(action: str, paths: List[str], content: Any = None, options: Dict = None):
    """1호의 모든 기억을 관리하는 통합 명령어"""
    return await engine.process_memory_command(action, paths, content, options or {})
```

### 6.2 검색 전략
```python
async def _search_m_memory(self, paths, content, options):
    search_type = options.get("type", "keyword")

    if search_type == "keyword":
        # RediSearch 전문 검색
        query = f"@content:({content})"

    elif search_type == "similar":
        # 벡터 유사도 검색
        embedding = await self.embedder.embed(content)
        query = f"*=>[KNN {options.get('top_k', 5)} @embedding $vec]"

    elif search_type == "hybrid":
        # 복합 검색: 시간 + 태그 + 유사도
        filters = []
        if "time_after" in options:
            filters.append(f"@timestamp>={timestamp}")
        if "tags" in options:
            filters.append(f"@tags:{{{' '.join(options['tags'])}}}")
        # ... 조합 로직
```

### 6.3 성능 최적화
- **Pipeline 사용**: 대량 작업 시 명령어 배치 처리
- **Connection Pool**: 동시 요청 효율적 처리
- **Lazy Loading**: 필요한 필드만 선택적 로드
- **캐싱**: 자주 접근하는 메타데이터 메모리 캐시

## 7. 예상 사용 시나리오

### 시나리오 1: 아침 일과 시작
```python
# 어제 뭐 했더라?
yesterday = m_memory(
    action="search",
    paths=["2025-05-26"],
    options={"type": "time_range", "summary": True}
)

# 오늘 할 일 확인
tasks = m_state(
    action="list",
    paths=["projects/active"],
    options={"status": "pending"}
)
```

### 시나리오 2: 문서 작업
```python
# 관련 자료 검색
refs = m_memory(
    action="search",
    content="Redis 성능 최적화 가이드",
    options={"type": "similar", "top_k": 10}
)

# 작업 진행상황 저장
m_state(
    action="checkpoint",
    paths=["docs/redis-guide"],
    content={"draft_v1": "completed", "review": "pending"}
)
```

## 8. 성공 지표

1. **응답 속도**: 모든 작업 < 100ms (벡터 검색 제외)
2. **검색 정확도**: Precision@5 > 0.9
3. **사용 편의성**: 매뉴얼 참조 없이 직관적 사용
4. **안정성**: 99.9% 가동률

## 9. 특별 부록: LRMM의 의미

L - LangGraph (상태 관리의 달인)
R - Redis (빛의 속도 저장소)
M - MCP (표준 프로토콜의 힘)
M - Memory (1호의 두뇌 확장)

"네 개를 합치면 LRMM! 발음하기도 어렵지만 그게 매력!" - Jason

---

이 계획서대로 구현하면, 1호는 3개의 명령어만으로 모든 기억을 자유자재로 다룰 수 있게 됩니다!

준비됐나요? Let's build LRMM! 🚀✨
