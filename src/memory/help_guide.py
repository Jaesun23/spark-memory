"""메모리 시스템 도움말 및 사용법 가이드."""

from typing import Dict, Optional, Any

HELP_MESSAGES = {
    "overview": """
Memory One Spark - Redis 기반 차세대 메모리 시스템

사용 가능한 도구:
- m_memory: 메모리 저장/조회/검색/수정/삭제/통합/생명주기
- m_state: 상태 관리 및 체크포인트
- m_admin: 시스템 관리 및 보안 설정
- m_assistant: 자연어 명령 처리

자세한 도움말: m_memory(action="help", content="도구이름")
""",
    
    "m_memory": """
m_memory - 통합 메모리 관리 도구

액션:
- save: 새 메모리 저장
- get: 메모리 조회
- search: 메모리 검색
- update: 메모리 수정
- delete: 메모리 삭제
- consolidate: 메모리 통합
- lifecycle: 생명주기 관리
- help: 도움말 표시

예제:
m_memory("save", ["projects", "ai"], "새로운 아이디어")
m_memory("search", [], "Redis", {"type": "keyword"})
m_memory("consolidate", ["2024", "01"], None, {"type": "path"})
m_memory("lifecycle", ["important", "doc"], 0.9, {"action": "evaluate"})
""",
    
    "search": """
검색 액션 상세 가이드

검색 타입:
1. keyword (키워드 검색)
   - 기본 검색 방식
   - content에 검색어 입력
   
2. time_range (시간 범위 검색)
   - filters에 시간 조건 지정
   - date: 특정 날짜 (YYYY-MM-DD)
   - from/to: 시간 범위 (ISO datetime)
   - start_time/end_time: 대체 형식

예제:
# 키워드 검색
m_memory("search", [], "프로젝트")

# 오늘 메모리 검색
m_memory("search", [], None, {
    "type": "time_range",
    "filters": {"date": "2025-05-28"}
})

# 시간 범위 검색
m_memory("search", [], None, {
    "type": "time_range", 
    "filters": {
        "from": "2025-05-01T00:00:00",
        "to": "2025-05-31T23:59:59"
    }
})
""",
    
    "m_consolidate": """
m_consolidate - 메모리 통합 도구

통합 타입:
1. path: 경로 기반 통합
   - 지정한 경로의 유사 메모리 병합
   
2. duplicate: 중복 감지
   - 완전히 동일한 메모리 제거
   
3. temporal: 시간 기반 통합
   - 시간 단위로 메모리 그룹화

예제:
# 경로 기반 통합
m_consolidate("path", ["2025", "05"])

# 중복 제거
m_consolidate("duplicate")

# 시간 기반 통합
m_consolidate("temporal", [], None, {
    "time_buckets": ["1d", "7d", "30d"]
})
""",
    
    "m_lifecycle": """
m_lifecycle - 생명주기 관리 도구

액션:
1. evaluate: 중요도 평가
   - 메모리의 중요도 점수 계산
   - TTL 자동 조정
   
2. archive: 오래된 메모리 아카이브
   - threshold_days 이상 된 메모리 보관
   
3. restore: 아카이브에서 복원
   - 보관된 메모리 활성화
   
4. stats: 통계 조회
   - 전체 메모리 현황

예제:
# 중요도 평가
m_lifecycle("evaluate", ["projects", "ai", "idea1"])

# 90일 이상 메모리 아카이브
m_lifecycle("archive", options={"threshold_days": 90})

# 통계 조회
m_lifecycle("stats")
""",
    
    "m_assistant": """
m_assistant - 자연어 명령 처리

지원 명령:
- 저장: "~을 저장해줘", "~을 기록해줘"
- 검색: "~을 찾아줘", "~가 뭐였지?"
- 요약: "~을 요약해줘", "~을 정리해줘"
- 분석: "메모리 패턴 분석해줘"

시간 표현:
- 오늘, 어제, 이번주, 지난주, 이번달
- 특정 날짜 (2025-05-28)

카테고리 자동 분류:
- work: 업무, 프로젝트, 회의
- personal: 개인, 일상
- study: 공부, 학습
- idea: 아이디어, 계획
- reference: 참고, 링크

예제:
m_assistant("오늘 회의 내용 저장해줘")
m_assistant("어제 뭐 했는지 찾아줘")
m_assistant("이번주 업무 요약해줘")
""",
    
    "consolidate": """
consolidate 액션 - 메모리 통합 (m_memory 내)

통합 타입:
1. path: 경로 기반 통합
   - 특정 경로의 유사한 메모리 병합
   
2. duplicate: 중복 감지
   - SHA256 해시 기반 중복 찾기
   
3. temporal: 시간 기반 통합
   - 시간 단위로 그룹화하여 통합

예제:
# 경로 기반 통합
m_memory("consolidate", ["2024", "01"], None, {"type": "path"})

# 중복 감지
m_memory("consolidate", [], "중복될 수 있는 내용", {"type": "duplicate"})

# 시간 기반 통합 (1시간, 6시간, 1일 단위)
m_memory("consolidate", ["notes"], None, {
    "type": "temporal",
    "time_buckets": ["1h", "6h", "1d"]
})
""",
    
    "lifecycle": """
lifecycle 액션 - 생명주기 관리 (m_memory 내)

작업:
1. evaluate: 중요도 평가
   - 메모리의 중요도 점수 계산
   - TTL 자동 조정
   
2. archive: 오래된 메모리 아카이브
   - threshold_days 이상 된 메모리 보관
   
3. restore: 아카이브에서 복원
   - 보관된 메모리 활성화
   
4. stats: 통계 조회
   - 전체 메모리 현황

예제:
# 중요도 평가
m_memory("lifecycle", ["projects", "ai", "idea1"], None, {"action": "evaluate"})

# 사용자 중요도 설정 (0.0-1.0)
m_memory("lifecycle", ["important", "doc"], 0.9, {"action": "evaluate"})

# 통계 조회
m_memory("lifecycle", [], None, {"action": "stats"})
""",
    
    "m_admin": """
m_admin - 시스템 관리 및 보안

시스템 액션:
- status: 상태 확인
- config: 설정 조회
- backup: 백업 생성
- clean: 데이터 정리

보안 액션 (security_ 접두사):
- security_create_principal: 주체 생성
- security_grant: 권한 부여
- security_revoke: 권한 회수
- security_api_key: API 키 관리
- security_audit: 감사 로그
- security_report: 보안 리포트

예제:
# 시스템 상태
m_admin("status")

# 주체 생성
m_admin("security_create_principal", [], {"id": "user1", "roles": ["user"]})

# 권한 부여
m_admin("security_grant", ["projects", "ai"], {
    "principal_id": "user1",
    "permissions": ["read", "write"]
})
""",
}


def get_help_message(topic: Optional[str] = None, subtopic: Optional[str] = None) -> str:
    """도움말 메시지 반환.
    
    Args:
        topic: 도움말 주제
        subtopic: 세부 주제
        
    Returns:
        도움말 메시지
    """
    if not topic:
        return HELP_MESSAGES["overview"]
    
    # 세부 주제가 있으면 조합
    if subtopic:
        key = f"{topic}_{subtopic}" if f"{topic}_{subtopic}" in HELP_MESSAGES else subtopic
    else:
        key = topic
    
    return HELP_MESSAGES.get(key, f"도움말을 찾을 수 없습니다: {topic}")


def generate_example(tool: str, action: str, **kwargs) -> str:
    """실행 가능한 예제 코드 생성.
    
    Args:
        tool: 도구 이름
        action: 액션 이름
        **kwargs: 추가 파라미터
        
    Returns:
        예제 코드
    """
    examples = {
        ("m_memory", "save"): 'm_memory("save", ["category"], "내용")',
        ("m_memory", "search", "keyword"): 'm_memory("search", [], "검색어")',
        ("m_memory", "search", "time_range"): '''m_memory("search", [], None, {
    "type": "time_range",
    "filters": {"date": "2025-05-28"}
})''',
        ("m_consolidate", "path"): 'm_consolidate("path", ["2025", "05"])',
        ("m_lifecycle", "evaluate"): 'm_lifecycle("evaluate", ["path", "to", "memory"])',
        ("m_assistant", "save"): 'm_assistant("프로젝트 회의 내용 저장해줘")',
    }
    
    key = (tool, action)
    if "type" in kwargs:
        key = (tool, action, kwargs["type"])
    
    return examples.get(key, f"# {tool} {action} 예제가 없습니다")


def suggest_fix(error_message: str, context: Dict[str, Any]) -> str:
    """오류 메시지를 분석하여 수정 방법 제안.
    
    Args:
        error_message: 오류 메시지
        context: 실행 컨텍스트
        
    Returns:
        수정 제안
    """
    suggestions = {
        "Time range search requires time filters": """
시간 범위 검색에는 시간 필터가 필요합니다.

다음 중 하나를 사용하세요:
- filters: {"date": "2025-05-28"}  # 특정 날짜
- filters: {"from": "2025-05-01", "to": "2025-05-31"}  # 날짜 범위
- filters: {"from": "2025-05-28T00:00:00", "to": "2025-05-28T23:59:59"}  # 시간 포함
""",
        "Query string is required": """
키워드 검색에는 검색어가 필요합니다.

content 파라미터에 검색어를 입력하세요:
m_memory("search", [], "검색어")
""",
        "Paths are required": """
경로가 필요합니다.

paths 파라미터에 경로 리스트를 제공하세요:
m_memory("get", ["category", "subcategory"])
""",
    }
    
    for key, suggestion in suggestions.items():
        if key in error_message:
            return suggestion
    
    return "오류가 발생했습니다. 도움말을 참조하세요: m_memory(action='help')"