"""Memory Engine - Modular action-based architecture."""

import logging
from typing import Any, Dict, List, Optional, Union

from src.memory.models import SearchResult
from src.redis.client import RedisClient
from src.memory.actions import (
    BasicActions,
    SearchActions,
    ConsolidateActions,
    LifecycleActions,
    HelpActions,
)

logger = logging.getLogger(__name__)


class MemoryEngine:
    """메모리 엔진 - 모든 메모리 작업의 진입점.
    
    액션 기반 모듈화 아키텍처로 각 기능이 독립적으로 구현됨.
    """
    
    def __init__(
        self,
        redis_client: RedisClient,
        enable_security: bool = False,
        enable_events: bool = True,
        default_timezone: str = "UTC",
    ):
        """메모리 엔진 초기화.

        Args:
            redis_client: Redis 클라이언트
            enable_security: 보안 기능 활성화 여부
            enable_events: 이벤트 기능 활성화 여부
            default_timezone: 기본 타임존
        """
        self.redis = redis_client
        self.enable_security = enable_security
        self.enable_events = enable_events
        
        # 액션 핸들러 초기화
        self.basic_actions = BasicActions(redis_client, default_timezone)
        self.search_actions = SearchActions(redis_client)
        self.consolidate_actions = ConsolidateActions(redis_client)
        self.lifecycle_actions = LifecycleActions(redis_client)
        self.help_actions = HelpActions()
        
        # 보안 관련 (기존 호환성 유지)
        self.access_control = None
        self.audit_logger = None
        self.field_encryption = None
        self.key_manager = None
        
        # 벡터 스토어 (선택적)
        self.vector_store = None
        
        # 추가 매니저들 (기존 호환성)
        self.consolidator = None
        self.lifecycle_manager = None
        
        logger.info("MemoryEngine initialized with modular actions")
        
    async def execute(
        self,
        action: str,
        paths: List[str],
        content: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        principal: Optional[Any] = None,  # Principal type from security module
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]], List[SearchResult]]:
        """메모리 명령어 실행 - 라우팅만 담당.

        Args:
            action: 실행할 액션
            paths: 메모리 경로
            content: 저장/수정할 내용
            options: 추가 옵션
            principal: 실행 주체 (보안용)

        Returns:
            액션 실행 결과

        Raises:
            ValueError: 잘못된 액션이나 파라미터
            RuntimeError: 실행 중 오류
        """
        options = options or {}

        # 보안 체크 (기존 로직 유지 - 보안 모듈 활성화 시)
        if self.enable_security and self.access_control and principal:
            # 보안 검사는 access_control이 설정된 경우에만 수행
            # TODO: security 모듈 임포트 후 활성화
            pass

        # 액션 라우팅
        try:
            logger.info(f"Executing memory action: {action} with paths: {paths}")
            
            # Basic actions
            if action in ["save", "get", "update", "delete"]:
                result = await self.basic_actions.execute(action, paths, content, options)
            
            # Search action
            elif action == "search":
                result = await self.search_actions.execute(paths, content, options)
            
            # Consolidate action
            elif action == "consolidate":
                result = await self.consolidate_actions.execute(paths, options)
            
            # Lifecycle action
            elif action == "lifecycle":
                result = await self.lifecycle_actions.execute(paths, content, options)
            
            # Help action
            elif action == "help":
                result = await self.help_actions.execute(paths, content, options)
            
            else:
                raise ValueError(
                    f"Unknown action: {action}. Valid actions: save, get, search, update, delete, consolidate, lifecycle, help"
                )

            # 성공 감사 로그 (보안 모듈 활성화 시)
            if self.enable_security and self.audit_logger and principal:
                # TODO: security 모듈 활성화 후 감사 로그 기록
                pass

            return result

        except ValueError as e:
            # 파라미터 검증 오류
            logger.error(f"Invalid parameters for {action}: {e}")
            
            # 오류에 대한 도움말 제안
            error_msg = str(e)
            suggestion = self._suggest_fix(error_msg, action)
            
            if self.enable_security and self.audit_logger and principal:
                # TODO: security 모듈 활성화 후 감사 로그 기록
                pass
            
            raise RuntimeError(f"Memory operation failed: {error_msg}\n\n{suggestion}") from e

        except Exception as e:
            logger.error(f"Error executing memory action {action}: {e}")
            
            if self.enable_security and self.audit_logger and principal:
                # TODO: security 모듈 활성화 후 감사 로그 기록
                pass
            
            raise

    # 보안 관련 메서드들 - security 모듈 통합 시 활성화
    # async def _log_audit(...) -> None:
    #     """감사 로그 기록."""
    #     pass
    #
    # def _get_audit_event_type(self, action: str):
    #     """액션에 대응하는 감사 이벤트 타입 반환."""
    #     pass

    def _suggest_fix(self, error_msg: str, action: str) -> str:
        """오류 메시지를 분석하여 수정 방법 제안."""
        suggestions = {
            "Content is required": """
content 파라미터가 필요합니다.

m_memory("save", ["category"], "저장할 내용")
""",
            "Paths are required": """
paths 파라미터가 필요합니다.

m_memory("get", ["path", "to", "memory"])
""",
            "Query string is required": """
키워드 검색에는 검색어가 필요합니다.

content 파라미터에 검색어를 입력하세요:
m_memory("search", [], "검색어")
""",
            "Time range search requires filters": """
시간 범위 검색에는 필터가 필요합니다.

options에 시간 필터를 추가하세요:
m_memory("search", [], None, {
    "type": "time_range",
    "filters": {"date": "2025-05-28"}
})
""",
        }

        for key, suggestion in suggestions.items():
            if key in error_msg:
                return suggestion

        return f"'{action}' 액션 사용법은 help를 참조하세요:\nm_memory('help', ['{action}'])"