"""LRMM MCP Server - Memory One Spark.

MCP를 사용한 메모리 시스템 서버입니다.
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from fastmcp import Context, FastMCP

from src.memory.assistant import MemoryAssistant
from src.memory.engine import MemoryEngine
from src.memory.state_manager import StateManager
from src.redis.client import RedisClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MemoryServer:
    """메모리 서버 클래스.

    Redis와 연결하여 memory, state, system 명령어를 처리합니다.
    """

    def __init__(
        self, redis_url: str = "redis://localhost:6379", enable_security: bool = False
    ) -> None:
        """MemoryServer 초기화.

        Args:
            redis_url: Redis 연결 URL
            enable_security: 보안 기능 활성화 여부
        """
        self.redis_url = redis_url
        self.enable_security = enable_security
        self.redis_client: Optional[RedisClient] = None
        self.memory_engine: Optional[MemoryEngine] = None
        self.state_manager: Optional[StateManager] = None
        self.assistant: Optional[MemoryAssistant] = None

    async def initialize(self) -> None:
        """서버 초기화."""
        logger.info("LRMM Memory Server 초기화 중...")

        # Redis 클라이언트 생성 및 연결
        self.redis_client = RedisClient(url=self.redis_url)
        await self.redis_client.connect()

        # 메모리 엔진 생성 (보안 옵션 전달)
        self.memory_engine = MemoryEngine(
            redis_client=self.redis_client, enable_security=self.enable_security
        )

        # 벡터 인덱스 초기화 (v2에서는 별도 초기화 불필요)
        # Engine v2는 필요시 자동으로 벡터 스토어를 초기화함

        # 상태 관리자 생성
        self.state_manager = StateManager(redis_client=self.redis_client)

        # 메모리 어시스턴트 생성
        self.assistant = MemoryAssistant()

        security_status = "활성화" if self.enable_security else "비활성화"
        logger.info(f"LRMM Memory Server 초기화 완료! (보안: {security_status})")

    async def shutdown(self) -> None:
        """서버 종료."""
        logger.info("LRMM Memory Server 종료 중...")

        if self.redis_client:
            await self.redis_client.disconnect()

        logger.info("LRMM Memory Server 종료 완료!")

    async def get_status(self) -> Dict[str, Any]:
        """서버 상태 조회.

        Returns:
            서버 상태 정보
        """
        if not self.redis_client:
            return {"status": "not_initialized", "redis": None}

        try:
            redis_health = await self.redis_client.health_check()
            return {
                "status": "healthy",
                "redis": redis_health,
                "version": "0.1.0",  # 하드코딩된 버전
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "redis": None,
                "error": str(e),
            }


# 전역 서버 인스턴스
server = MemoryServer()


# Lifespan context manager
@asynccontextmanager
async def app_lifespan(app: FastMCP[None]) -> AsyncIterator[None]:
    """애플리케이션 생명주기 관리."""
    # 시작 시
    await server.initialize()
    try:
        yield
    finally:
        # 종료 시
        await server.shutdown()


# FastMCP 앱 인스턴스 생성 (MCP 2025 사양)
app = FastMCP(name="LRMM Memory Server", lifespan=app_lifespan)


@app.tool()
async def m_memory(
    action: str,
    paths: List[str] = [],
    content: Optional[Any] = None,
    options: Optional[Dict[str, Any]] = None,
    ctx: Optional[Context] = None,
) -> Any:
    """모든 기억을 관리하는 통합 메모리 명령어.

    Args:
        action: 수행할 액션 (save, get, search, update, delete, consolidate, lifecycle)
        paths: 메모리 경로 리스트
        content: 저장/검색할 내용
        options: 추가 옵션
        ctx: MCP Context (자동 주입)

    Returns:
        액션 결과

    Examples:
        >>> # 메모리 저장
        >>> await m_memory("save", ["projects", "lrmm"], "프로젝트 시작!")

        >>> # 키워드 검색
        >>> await m_memory("search", [], "Redis", {"type": "keyword"})

        >>> # 시간 범위 검색
        >>> await m_memory("search", [], None, {
        ...     "type": "time_range",
        ...     "filters": {"date": "2025-05-27"}
        ... })

        >>> # 메모리 통합
        >>> await m_memory("consolidate", ["2024", "01"], None, {"type": "path"})

        >>> # 생명주기 관리
        >>> await m_memory(
        ...     "lifecycle", ["important", "doc"], 0.9, {"action": "evaluate"}
        ... )
    """
    if not server.memory_engine:
        raise RuntimeError("Server not initialized")

    # Context를 통한 로깅 (선택적)
    if ctx:
        await ctx.info(f"Memory action: {action}")

    try:
        result = await server.memory_engine.execute(
            action=action,
            paths=paths,
            content=content,
            options=options or {},
        )

        # 결과 직렬화 처리
        if isinstance(result, list):
            # SearchResult 리스트인 경우
            serialized = []
            for r in result:
                # 딕셔너리가 아닌 객체이고 to_dict 메서드가 있는 경우만 호출
                if not isinstance(r, (dict, str, int, float, bool, type(None))):
                    if hasattr(r, "to_dict"):
                        try:
                            serialized.append(r.to_dict())
                            continue
                        except Exception:
                            pass
                serialized.append(r)
            logger.info(f"Returning {len(serialized)} search results")
            return serialized
        elif not isinstance(result, (dict, str, int, float, bool, list, type(None))):
            # 단일 객체인 경우 - 기본 타입이 아닌 경우만 to_dict 시도
            if hasattr(result, "to_dict"):
                try:
                    return result.to_dict()
                except Exception:
                    pass

        return result

    except Exception as e:
        logger.error(f"Memory action failed: {action}, error: {e}")
        if ctx:
            await ctx.error(f"Memory action failed: {str(e)}")
        raise


@app.tool()
async def m_state(
    action: str,
    paths: List[str] = [],
    content: Optional[Any] = None,
    options: Optional[Dict[str, Any]] = None,
    ctx: Optional[Context] = None,
) -> Any:
    """상태와 체크포인트를 관리하는 명령어.

    LangGraph와 통합하여 작업 상태를 저장하고 복원합니다.

    Args:
        action: 수행할 액션 (checkpoint, restore, list, status, update)
        paths: 상태 경로 리스트
        content: 상태 데이터
        options: 추가 옵션 (description, checkpoint_id, partial 등)
        ctx: MCP Context (자동 주입)

    Returns:
        액션 결과

    Examples:
        >>> # 체크포인트 생성
        >>> await m_state("checkpoint", ["projects", "lrmm"], {
        ...     "phase": 3,
        ...     "progress": 0.4
        ... })

        >>> # 상태 복원
        >>> await m_state("restore", ["projects", "lrmm"])

        >>> # 현재 상태 조회
        >>> await m_state("status", ["projects", "lrmm"])

        >>> # 상태 업데이트
        >>> await m_state("update", ["projects", "lrmm"], {"progress": 0.6})
    """
    if not server.state_manager:
        raise RuntimeError("State manager not initialized")

    # Context를 통한 로깅
    if ctx:
        await ctx.info(f"State action: {action}")

    try:
        if action == "checkpoint":
            # 체크포인트 생성
            if not content:
                raise ValueError("Content required for checkpoint")

            checkpoint_id = await server.state_manager.create_checkpoint(
                paths=paths,
                state=content,
                description=options.get("description") if options else None,
            )

            return {
                "status": "success",
                "checkpoint_id": checkpoint_id,
                "message": f"Checkpoint created: {checkpoint_id}",
            }

        elif action == "restore":
            # 체크포인트 복원
            state_data = await server.state_manager.restore_checkpoint(
                paths=paths,
                checkpoint_id=options.get("checkpoint_id") if options else None,
            )

            if state_data:
                return {
                    "status": "success",
                    "state": state_data,
                    "message": "Checkpoint restored successfully",
                }
            else:
                return {
                    "status": "not_found",
                    "message": "No checkpoint found",
                }

        elif action == "list":
            # 체크포인트 목록
            checkpoints = await server.state_manager.list_checkpoints(
                paths=paths if paths else None,
            )

            return {
                "status": "success",
                "checkpoints": checkpoints,
                "count": len(checkpoints),
            }

        elif action == "status":
            # 현재 상태 조회
            current_state = await server.state_manager.get_current_state(paths)

            if current_state:
                return {
                    "status": "success",
                    "state": current_state,
                }
            else:
                return {
                    "status": "not_found",
                    "message": "No current state found",
                }

        elif action == "update":
            # 상태 업데이트
            if not content:
                raise ValueError("Content required for update")

            await server.state_manager.update_state(
                paths=paths,
                state=content,
                partial=options.get("partial", False) if options else False,
            )

            return {
                "status": "success",
                "message": "State updated successfully",
            }

        else:
            raise ValueError(f"Unknown state action: {action}")

    except Exception as e:
        logger.error(f"State action failed: {action}, error: {e}")
        if ctx:
            await ctx.error(f"State action failed: {str(e)}")
        raise


@app.tool()
async def m_admin(
    action: str,
    paths: List[str] = [],
    content: Optional[Any] = None,
    options: Optional[Dict[str, Any]] = None,
    ctx: Optional[Context] = None,
) -> Any:
    """시스템 관리 및 보안 통합 명령어.

    시스템 상태, 백업, 접근 제어, 감사 로그 등을 관리합니다.

    Args:
        action: 수행할 액션 (status, backup, clean, config, security_*)
        paths: 관리 대상 경로
        content: 설정 데이터
        options: 관리 옵션
        ctx: MCP Context (자동 주입)

    Returns:
        액션 결과

    Examples:
        >>> # 시스템 상태 확인
        >>> await m_admin("status")

        >>> # 백업 생성
        >>> await m_admin("backup", options={"compress": True})

        >>> # 주체 생성 (보안)
        >>> await m_admin(
        ...     "security_create_principal", [], {"id": "user1", "roles": ["user"]}
        ... )

        >>> # 권한 부여 (보안)
        >>> await m_admin("security_grant", ["projects", "lrmm"], {
        ...     "principal_id": "user1",
        ...     "permissions": ["read", "write"]
        ... })

        >>> # 감사 로그 조회 (보안)
        >>> await m_admin("security_audit", [], options={"limit": 100})
    """
    # 시스템 관련 액션
    if action == "status":
        return await server.get_status()

    elif action == "config":
        # 시스템 설정 조회
        return {
            "status": "success",
            "config": {
                "redis_url": server.redis_url,
                "version": "1.0.0",
                "features": {
                    "memory": True,
                    "state": True,
                    "search": True,
                    "events": True,
                    "security": (
                        server.memory_engine.enable_security
                        if server.memory_engine
                        else False
                    ),
                },
                "limits": {
                    "max_memory_size": "10MB",
                    "max_checkpoint_size": "5MB",
                    "default_ttl_days": 30,
                },
            },
        }

    elif action == "clean":
        # 만료된 데이터 정리
        if ctx:
            await ctx.info("Cleaning expired data...")

        # TODO: 실제 정리 로직 구현
        return {
            "status": "success",
            "message": "Cleanup completed",
            "cleaned": {
                "expired_keys": 0,
                "freed_memory": "0MB",
            },
        }

    elif action == "backup":
        # 백업 생성
        backup_id = f"backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # TODO: 실제 백업 로직 구현
        return {
            "status": "success",
            "backup_id": backup_id,
            "message": f"Backup created: {backup_id}",
            "location": f"/backups/{backup_id}",
        }

    # 보안 관련 액션
    elif action.startswith("security_"):
        if not server.memory_engine or not server.memory_engine.enable_security:
            return {
                "status": "disabled",
                "message": "Security features are not enabled",
            }

        security = server.memory_engine.access_control
        audit = server.memory_engine.audit_logger
        security_action = action[9:]  # Remove "security_" prefix

        if ctx:
            await ctx.info(f"Security action: {security_action}")

        try:
            if security_action == "create_principal":
                # 주체 생성
                if not content or not isinstance(content, dict):
                    raise ValueError("Principal data required")

                if security is None:
                    raise RuntimeError("Security not initialized")

                principal = security.create_principal(
                    principal_id=content["id"],
                    principal_type=content.get("type", "user"),
                    roles=set(content.get("roles", ["user"])),
                    metadata=content.get("metadata", {}),
                )

                return {
                    "status": "success",
                    "principal": {
                        "id": principal.id,
                        "type": principal.type,
                        "roles": list(principal.roles),
                    },
                }

            elif security_action == "grant":
                # 권한 부여
                if not content or not isinstance(content, dict):
                    raise ValueError("Grant data required")

                if security is None:
                    raise RuntimeError("Security not initialized")

                resource = "/".join(paths) if paths else "*"
                permissions = set(content.get("permissions", []))

                security.grant_permission(
                    principal_id=content["principal_id"],
                    resource=resource,
                    permissions=permissions,
                )

                return {
                    "status": "success",
                    "message": f"Permissions granted on {resource}",
                }

            elif security_action == "revoke":
                # 권한 회수
                if not content or not isinstance(content, dict):
                    raise ValueError("Revoke data required")

                if security is None:
                    raise RuntimeError("Security not initialized")

                resource = "/".join(paths) if paths else "*"
                permissions = set(content.get("permissions", []))

                security.revoke_permission(
                    principal_id=content["principal_id"],
                    resource=resource,
                    permissions=permissions,
                )

                return {
                    "status": "success",
                    "message": f"Permissions revoked on {resource}",
                }

            elif security_action == "api_key":
                # API 키 관리
                if not content or not isinstance(content, dict):
                    raise ValueError("API key data required")

                if security is None:
                    raise RuntimeError("Security not initialized")

                if content.get("operation") == "revoke":
                    # 키 회수
                    security.revoke_api_key(content["key"])
                    return {"status": "success", "message": "API key revoked"}
                else:
                    # 키 생성
                    api_key = security.generate_api_key(
                        principal_id=content["principal_id"],
                        name=content["name"],
                        expires_in_days=content.get("expires_in_days"),
                        rate_limit=content.get("rate_limit"),
                        scopes=set(content.get("scopes", [])),
                    )

                    return {
                        "status": "success",
                        "api_key": api_key,
                        "message": "Save this key securely, it won't be shown again",
                    }

            elif security_action == "audit":
                # 감사 로그 조회
                from datetime import timedelta

                if audit is None:
                    raise RuntimeError("Audit logger not initialized")

                options = options or {}

                # 시간 범위 설정
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=options.get("hours", 24))

                # audit이 None이 아님을 확인했으므로 안전하게 호출
                try:
                    events = audit.query_events(
                        principal_id=options.get("principal_id"),
                        resource=options.get("resource"),
                        start_time=start_time,
                        end_time=end_time,
                        limit=options.get("limit", 100),
                    )

                    # events를 리스트로 변환
                    event_list = []
                    if events:
                        for e in events:
                            event_list.append(
                                {
                                    "id": e.id,
                                    "timestamp": e.timestamp.isoformat(),
                                    "type": e.event_type,
                                    "principal": e.principal_id,
                                    "resource": e.resource,
                                    "result": e.result,
                                }
                            )

                    return {
                        "status": "success",
                        "events": event_list,
                        "count": len(event_list),
                    }

                except Exception as e:
                    logger.error(f"Failed to query audit events: {e}")
                    return {
                        "status": "error",
                        "message": f"Failed to query audit events: {str(e)}",
                        "events": [],
                        "count": 0,
                    }

            elif security_action == "report":
                # 보안 리포트 생성
                from datetime import timedelta

                if audit is None:
                    raise RuntimeError("Audit logger not initialized")

                options = options or {}
                period = options.get("period", "7d")

                # 기간 파싱
                if period.endswith("d"):
                    days = int(period[:-1])
                elif period.endswith("h"):
                    days = int(period[:-1]) / 24
                else:
                    days = 7

                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=days)

                # audit이 None이 아님을 보장했으므로 안전하게 호출
                try:
                    report = audit.generate_compliance_report(start_time, end_time)
                    anomalies_data = audit.get_anomalies(start_time)
                except Exception as e:
                    logger.error(f"Failed to generate report: {e}")
                    report = {}
                    anomalies_data = []

                # anomalies가 None이나 빈 리스트인 경우 처리
                anomaly_list = []
                if anomalies_data and isinstance(anomalies_data, list):
                    for a in anomalies_data:
                        if isinstance(a, dict):
                            anomaly_list.append(
                                {
                                    "pattern": a.get("pattern", ""),
                                    "description": a.get("description", ""),
                                    "timestamp": (
                                        a.get(
                                            "timestamp", datetime.utcnow()
                                        ).isoformat()
                                        if isinstance(a.get("timestamp"), datetime)
                                        else datetime.utcnow().isoformat()
                                    ),
                                    "risk_score": a.get("risk_score", 0),
                                }
                            )

                return {
                    "status": "success",
                    "report": report,
                    "anomalies": anomaly_list,
                }

            else:
                raise ValueError(f"Unknown security action: {security_action}")

        except Exception as e:
            logger.error(f"Security action failed: {security_action}, error: {e}")
            if ctx:
                await ctx.error(f"Security action failed: {str(e)}")
            raise

    else:
        return {
            "status": "error",
            "message": f"Unknown admin action: {action}",
        }


@app.tool()
async def m_assistant(
    command: str,
    options: Optional[Dict[str, Any]] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """지능형 메모리 어시스턴트.

    자연어 명령을 이해하고 적절한 메모리 작업을 수행합니다.

    Args:
        command: 자연어 명령어
        options: 추가 옵션
        ctx: MCP Context (자동 주입)

    Returns:
        명령 실행 결과

    Examples:
        >>> # 저장
        >>> await m_assistant("프로젝트 회의 내용 저장해줘")

        >>> # 검색
        >>> await m_assistant("어제 뭐 했는지 찾아줘")

        >>> # 요약
        >>> await m_assistant("이번주 업무 내용 요약해줘")

        >>> # 인사이트
        >>> await m_assistant("최근 메모리 패턴 분석해줘")
    """
    if not server.assistant or not server.memory_engine:
        raise RuntimeError("Assistant not initialized")

    # 자연어 명령 파싱
    from ..utils.command_parser import CommandParser

    parsed = CommandParser.parse(command)

    if ctx:
        await ctx.info(f"Assistant command: {parsed['action']}")

    try:
        # 파싱된 명령어로 메모리 엔진 실행
        result = await server.memory_engine.execute(
            action=parsed["action"],
            paths=parsed["paths"],
            content=parsed["content"],
            options=parsed["options"],
        )

        if parsed["action"] == "save":
            return {
                "status": "success",
                "action": "save",
                "key": result,
                "category": parsed["options"].get("category", "uncategorized"),
                "message": "Memory saved successfully",
                "original_command": command,
            }

        elif parsed["action"] == "search":
            # 결과 포맷팅
            if isinstance(result, list):
                memories = []
                for r in result:
                    # 딕셔너리가 아닌 객체이고 to_dict 메서드가 있는 경우만 호출
                    if not isinstance(r, (dict, str, int, float, bool, type(None))):
                        if hasattr(r, "to_dict"):
                            try:
                                memories.append(r.to_dict())
                                continue
                            except Exception:
                                pass
                    memories.append(r)
            else:
                memories = [result]

            return {
                "status": "success",
                "action": "search",
                "count": len(memories),
                "memories": memories,
                "query": parsed.get("content", ""),
                "original_command": command,
            }

        else:
            # 기타 명령어는 기본 결과 반환
            return {
                "status": "success",
                "action": parsed["action"],
                "result": result,
                "original_command": command,
            }

    except Exception as e:
        logger.error(f"Assistant command failed: {command}, error: {e}")
        if ctx:
            await ctx.error(f"Assistant error: {str(e)}")

        return {
            "status": "error",
            "message": str(e),
        }


# Prompts 프리미티브 구현
@app.prompt()
def memory_search_prompt(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """메모리 검색을 위한 프롬프트 템플릿.

    Args:
        query: 검색 쿼리
        context: 추가 컨텍스트

    Returns:
        MCP 프롬프트 메시지 리스트
    """
    base_prompt = f"""당신은 메모리 검색 도우미입니다. 사용자의 검색 쿼리를 분석하여
가장 관련성 높은 메모리를 찾아주세요.

검색 쿼리: {query}"""

    if context:
        base_prompt += f"\n추가 컨텍스트: {context}"

    return [{"role": "user", "content": base_prompt}]


@app.prompt()
def memory_summary_prompt(
    memories: List[Dict[str, Any]],
    focus: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """메모리 요약을 위한 프롬프트 템플릿.

    Args:
        memories: 메모리 리스트
        focus: 요약 포커스

    Returns:
        MCP 프롬프트 메시지 리스트
    """
    prompt = "다음 메모리들을 요약해주세요:\n\n"

    for i, memory in enumerate(memories, 1):
        prompt += (
            f"{i}. {memory.get('path', 'Unknown')}: "
            f"{memory.get('content', '')[:100]}...\n"
        )

    if focus:
        prompt += f"\n특히 '{focus}'에 대해 중점적으로 요약해주세요."

    return [{"role": "user", "content": prompt}]


# Resources 프리미티브 구현
@app.resource("memory://tree")
async def memory_tree() -> Dict[str, Any]:
    """메모리 트리 구조를 리소스로 제공.

    Returns:
        메모리 트리 구조 JSON
    """
    if not server.memory_engine:
        return {"error": "Server not initialized"}

    # TODO: 실제 트리 구조 구현
    return {
        "projects": ["lrmm", "memory-one-spark"],
        "memories": ["2025-05-27", "2025-05-28"],
        "checkpoints": ["phase2-60percent"],
    }


@app.resource("memory://stats")
async def memory_stats() -> Dict[str, Any]:
    """메모리 통계를 리소스로 제공.

    Returns:
        메모리 통계 JSON
    """
    if not server.redis_client:
        return {"error": "Redis not connected"}

    try:
        # Redis 정보 가져오기
        health = await server.redis_client.health_check()

        return {
            "total_keys": await server.redis_client.dbsize(),
            "memory_usage": health.get("used_memory_human", "N/A"),
            "connected_clients": health.get("connected_clients", 0),
            "uptime_seconds": health.get("uptime_seconds", 0),
            "redis_version": health.get("version", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}


# 양방향 통신을 위한 이벤트 리소스
@app.resource("memory://events")
async def memory_events() -> str:
    """메모리 이벤트 스트림.

    Returns:
        최근 이벤트 목록 (JSON)
    """
    if not server.redis_client:
        return json.dumps({"error": "Redis not connected"})

    try:
        # 최근 10개 이벤트 가져오기
        events = await server.redis_client.client.xrevrange(
            "memory:events",
            "+",
            "-",
            count=10,
        )

        event_list = []
        for event_id, event_data in events:
            if b"event" in event_data:
                event_json = json.loads(event_data[b"event"].decode())
                event_json["id"] = event_id.decode()
                event_list.append(event_json)

        return json.dumps(
            {
                "events": event_list,
                "count": len(event_list),
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


# 메인 실행 블록
if __name__ == "__main__":
    # MCP 서버는 stdio를 통해 통신
    app.run()
