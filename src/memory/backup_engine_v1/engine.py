"""메모리 엔진 핵심 구현.

이 모듈은 memory() 명령어의 핵심 비즈니스 로직을 담당합니다.
모든 메모리 관련 작업(저장, 조회, 검색, 수정, 삭제)을 처리합니다.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from src.consolidation import ConsolidationConfig, MemoryConsolidator
from src.embeddings import EmbeddingModel, EmbeddingService, VectorStore
from src.lifecycle import ImportanceScore, LifecycleManager, LifecyclePolicy
from src.memory.models import (
    Importance,
    MemoryContent,
    MemoryKey,
    MemoryMetadata,
    MemoryType,
    SearchQuery,
    SearchResult,
    SearchType,
)
from src.redis.client import RedisClient
from src.security import (
    AccessContext,
    AccessControlService,
    AuditEventType,
    AuditLogger,
    EncryptionService,
    FieldLevelEncryption,
    Permission,
    Principal,
)
from src.utils.time_path import TimePathGenerator

logger = logging.getLogger(__name__)


class MemoryEngine:
    """메모리 엔진 핵심 클래스.

    memory() 명령어의 모든 액션을 처리하고 적절한 Redis 작업으로 변환합니다.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        default_timezone: str = "Asia/Seoul",
        enable_vector_search: bool = True,
        embedding_model: Optional[EmbeddingModel] = None,
        lifecycle_policy: Optional[LifecyclePolicy] = None,
        enable_security: bool = True,
        encryption_key: Optional[bytes] = None,
    ) -> None:
        """MemoryEngine 초기화.

        Args:
            redis_client: Redis 클라이언트 인스턴스
            default_timezone: 기본 시간대
            enable_vector_search: 벡터 검색 활성화 여부
            embedding_model: 사용할 임베딩 모델
            lifecycle_policy: 생명주기 관리 정책
            enable_security: 보안 기능 활성화 여부
            encryption_key: 암호화 키 (None이면 자동 생성)
        """
        self.redis = redis_client
        self.time_gen = TimePathGenerator(default_timezone)
        self.default_timezone = default_timezone
        self.enable_vector_search = enable_vector_search
        self.enable_security = enable_security

        # 벡터 검색 초기화
        self.vector_store: Optional[VectorStore] = None
        if enable_vector_search:
            try:
                embedding_service = EmbeddingService(
                    primary_model=embedding_model or EmbeddingModel.LOCAL_MINILM,
                    fallback_model=EmbeddingModel.LOCAL_MINILM,
                )
                self.vector_store = VectorStore(
                    redis_client=redis_client, embedding_service=embedding_service
                )
            except Exception as e:
                logger.warning(f"Vector search initialization failed: {e}")
                self.enable_vector_search = False

        # 메모리 통합기 초기화
        self.consolidator: Optional[MemoryConsolidator] = None
        if self.vector_store:
            self.consolidator = MemoryConsolidator(
                memory_engine=self,
                vector_store=self.vector_store,
                config=ConsolidationConfig(),
            )

        # 생명주기 관리자 초기화
        self.lifecycle_manager = LifecycleManager(
            redis_client=redis_client.client,
            policy=lifecycle_policy,
        )

        # 보안 컴포넌트 초기화
        self.encryption_service: Optional[EncryptionService] = None
        self.field_encryption: Optional[FieldLevelEncryption] = None
        self.access_control: Optional[AccessControlService] = None
        self.audit_logger: Optional[AuditLogger] = None

        if enable_security:
            # 암호화 서비스
            self.encryption_service = EncryptionService(master_key=encryption_key)
            self.field_encryption = FieldLevelEncryption(self.encryption_service)

            # 접근 제어 서비스
            self.access_control = AccessControlService()

            # 감사 로거
            self.audit_logger = AuditLogger(retention_days=90)

    async def initialize(self) -> None:
        """메모리 엔진 초기화."""
        # 벡터 검색 인덱스 초기화
        if self.vector_store:
            try:
                await self.vector_store.create_index()
                logger.info("Vector search index initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vector index: {e}")

        # 생명주기 관리자 시작
        await self.lifecycle_manager.start()
        logger.info("Lifecycle manager started")

    async def shutdown(self) -> None:
        """메모리 엔진 종료."""
        await self.lifecycle_manager.stop()
        logger.info("Memory engine shutdown complete")

    async def initialize_vector_index(self) -> None:
        """벡터 검색 인덱스 초기화 (deprecated, use initialize instead)."""
        if self.vector_store:
            try:
                await self.vector_store.create_index()
                logger.info("Vector search index initialized successfully")
            except Exception as e:
                logger.error(f"Failed to create vector index: {e}")
                raise

    async def execute(
        self,
        action: str,
        paths: List[str],
        content: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        principal: Optional[Principal] = None,
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]], List[SearchResult]]:
        """메모리 명령어 실행.

        Args:
            action: 실행할 액션 (save, get, search, update, delete)
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

        # 보안 체크
        if self.enable_security and self.access_control and principal:
            permission_map = {
                "save": Permission.WRITE,
                "get": Permission.READ,
                "search": Permission.SEARCH,
                "update": Permission.WRITE,
                "delete": Permission.DELETE,
                "consolidate": Permission.CONSOLIDATE,
                "lifecycle": Permission.LIFECYCLE,
            }

            required_permission = permission_map.get(action)
            if required_permission:
                resource = "/".join(paths) if paths else "*"
                access_context = AccessContext(
                    principal=principal,
                    resource=resource,
                    action=required_permission,
                )

                if not self.access_control.check_permission(access_context):
                    # 감사 로그
                    if self.audit_logger:
                        await self._log_audit(
                            AuditEventType.ACCESS_DENIED,
                            principal,
                            resource,
                            action,
                            "failure",
                            {"reason": "insufficient_permissions"},
                        )
                    raise RuntimeError(
                        f"Access denied: insufficient permissions for {action}"
                    )

        # help 액션 특별 처리
        if action == "help":
            from src.memory.help_guide import get_help_message
            topic = paths[0] if paths else None
            subtopic = content if isinstance(content, str) else None
            return {"help": get_help_message(topic, subtopic)}
        
        # 액션 라우팅
        action_map = {
            "save": self._save,
            "get": self._get,
            "search": self._search,
            "update": self._update,
            "delete": self._delete,
            "consolidate": self._consolidate,
            "lifecycle": self._lifecycle,
        }

        if action not in action_map:
            raise ValueError(
                f"Unknown action: {action}. Valid actions: {list(action_map.keys())}"
            )

        try:
            logger.info(f"Executing memory action: {action} with paths: {paths}")
            result = await action_map[action](paths, content, options)

            # 성공 감사 로그
            if self.enable_security and self.audit_logger and principal:
                resource = "/".join(paths) if paths else "*"
                await self._log_audit(
                    self._get_audit_event_type(action),
                    principal,
                    resource,
                    action,
                    "success",
                )

            return result
        except Exception as e:
            logger.error(f"Error executing memory action {action}: {e}")

            # 실패 감사 로그
            if self.enable_security and self.audit_logger and principal:
                resource = "/".join(paths) if paths else "*"
                await self._log_audit(
                    self._get_audit_event_type(action),
                    principal,
                    resource,
                    action,
                    "failure",
                    {"error": str(e)},
                )

            # 도움말 제안 추가
            from src.memory.help_guide import suggest_fix
            error_msg = str(e)
            suggestion = suggest_fix(error_msg, {"action": action, "paths": paths})
            
            raise RuntimeError(f"Memory operation failed: {error_msg}\n\n{suggestion}") from e

    async def _save(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> str:
        """메모리 저장.

        Args:
            paths: 저장 경로
            content: 저장할 내용
            options: 저장 옵션

        Returns:
            저장된 메모리 키
        """
        if content is None:
            raise ValueError("Content is required for save action")

        # 메모리 타입 결정
        memory_type = MemoryType(
            options.get("type", MemoryType.from_content(content).value)
        )

        # 시간 경로 생성
        if not paths or paths == [""]:
            category = options.get("category", memory_type.value)
            time_path = self.time_gen.generate_path(category)
            paths = time_path.split("/")

        # 메모리 콘텐츠 생성
        memory_content = MemoryContent(
            type=memory_type,
            data=content,
        )

        # 메타데이터 설정
        if "tags" in options:
            memory_content.metadata.tags = options["tags"]
        if "importance" in options:
            memory_content.metadata.importance = Importance(options["importance"])
        if "ttl" in options:
            memory_content.metadata.ttl_seconds = options["ttl"]
        if "source" in options:
            memory_content.metadata.source = options["source"]

        # 키 생성
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        # 데이터 타입별 저장
        if memory_type == MemoryType.CONVERSATION:
            await self._save_conversation(key, memory_content, options)
        else:
            await self._save_json(key, memory_content, options, memory_key)

        logger.info(f"Memory saved with key: {key}")
        return key

    async def _save_conversation(
        self,
        key: str,
        content: MemoryContent,
        options: Dict[str, Any],
    ) -> None:
        """대화형 메모리 저장 (Redis Streams).

        Args:
            key: Redis 키
            content: 메모리 콘텐츠
            options: 저장 옵션
        """
        # 스트림 키 생성 (시간 기반)
        stream_key = f"stream:{key}"

        # 메시지 데이터 준비
        if isinstance(content.data, dict):
            message_data = content.data
        else:
            message_data = {"content": str(content.data)}

        # 암호화 적용
        if (
            self.enable_security
            and self.field_encryption
            and options.get("encrypt", True)
        ):
            message_data = self.field_encryption.encrypt_dict(message_data)

        # 메타데이터 추가
        message_data["_metadata"] = json.dumps(content.metadata.to_dict())

        # Redis Stream에 추가
        client = self.redis.client
        message_id = await client.xadd(stream_key, message_data)

        # TTL 설정
        if content.metadata.ttl_seconds:
            await client.expire(stream_key, content.metadata.ttl_seconds)

        # 메타데이터를 별도 Hash로 저장
        await self._save_metadata_hash(key, content.metadata)

        logger.debug(f"Conversation saved to stream {stream_key} with ID {message_id}")

    async def _save_json(
        self,
        key: str,
        content: MemoryContent,
        options: Dict[str, Any],
        memory_key: Optional[MemoryKey] = None,
    ) -> None:
        """JSON 형태 메모리 저장 (RedisJSON).

        Args:
            key: Redis 키
            content: 메모리 콘텐츠
            options: 저장 옵션
        """
        # JSON 키 생성
        json_key = f"json:{key}"

        # 저장할 데이터
        data_to_save = content.to_dict()

        # 암호화 적용
        if (
            self.enable_security
            and self.field_encryption
            and options.get("encrypt", True)
        ):
            data_to_save = self.field_encryption.encrypt_dict(data_to_save)

        # RedisJSON에 저장
        client = self.redis.client
        await client.json().set(json_key, "$", data_to_save)

        # TTL 설정
        if content.metadata.ttl_seconds:
            await client.expire(json_key, content.metadata.ttl_seconds)

        # 메타데이터를 별도 Hash로 저장
        await self._save_metadata_hash(key, content.metadata)

        # 벡터 인덱싱
        if self.vector_store and options.get("index", True) and memory_key:
            try:
                await self.vector_store.add_memory(
                    path="/".join(memory_key.paths),
                    content=content.data,
                    metadata={
                        "type": str(content.metadata.memory_type),
                        "timestamp": content.metadata.created_at.timestamp(),
                        "importance": content.metadata.importance.value,
                    },
                )
                logger.debug(f"Vector embedding added for {json_key}")
            except Exception as e:
                logger.warning(f"Failed to add vector embedding: {e}")

        # 생명주기 관리 - 중요도 평가 및 TTL 설정
        if memory_key:
            path = "/".join(memory_key.paths)
            # Just pass the existing MemoryContent object
            memory_content = content

            # 사용자 지정 중요도가 있으면 적용
            user_rating = None
            if content.metadata.importance != Importance.NORMAL:
                # 중요도를 0-1 범위로 변환
                importance_map = {
                    Importance.LOW: 0.2,
                    Importance.NORMAL: 0.5,
                    Importance.HIGH: 0.8,
                    Importance.CRITICAL: 1.0,
                }
                user_rating = importance_map.get(content.metadata.importance, 0.5)

            # 중요도 평가 및 TTL 설정
            score, ttl = await self.lifecycle_manager.evaluate_and_set_ttl(
                path, memory_content
            )

            # 사용자 지정 중요도가 있으면 업데이트
            if user_rating:
                await self.lifecycle_manager.update_importance(path, user_rating)

            logger.debug(
                f"Lifecycle management applied: importance={score.overall_score:.2f}, "
                f"ttl={ttl}, tier={self.lifecycle_manager.policy.ttl_policy.get_tier(score.overall_score)}"
            )

        logger.debug(f"JSON saved to {json_key}")

    async def _get(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """메모리 조회.

        Args:
            paths: 조회할 경로
            content: 사용하지 않음
            options: 조회 옵션

        Returns:
            조회된 메모리 내용
        """
        if not paths or paths == [""]:
            raise ValueError("Paths are required for get action")

        # 메모리 타입 추론
        memory_type = MemoryType(options.get("type", "document"))

        # 키 생성
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        client = self.redis.client

        # 데이터 타입별 조회
        if memory_type == MemoryType.CONVERSATION:
            stream_key = f"stream:{key}"
            # 스트림에서 메시지 읽기
            messages = await client.xrange(stream_key, "-", "+")

            if not messages:
                return []

            # 메시지 포맷팅
            result = []
            for msg_id, msg_data in messages:
                # _metadata 필드 분리
                metadata = msg_data.pop("_metadata", "{}")
                if isinstance(metadata, str):
                    import json

                    metadata = json.loads(metadata)

                # 복호화 적용
                if self.enable_security and self.field_encryption:
                    msg_data = self.field_encryption.decrypt_dict(msg_data)

                result.append(
                    {
                        "id": msg_id,
                        "data": msg_data,
                        "metadata": metadata,
                    }
                )

            return result
        else:
            json_key = f"json:{key}"
            # JSON 조회
            data = await client.json().get(json_key, "$")

            if not data:
                return {}

            # 첫 번째 결과 반환 (RedisJSON은 배열로 반환)
            result = data[0] if isinstance(data, list) else data

            # 복호화 적용
            if (
                self.enable_security
                and self.field_encryption
                and isinstance(result, dict)
            ):
                result = self.field_encryption.decrypt_dict(result)

            # 접근 추적 (생명주기 관리)
            path = "/".join(paths)
            await self.lifecycle_manager.evaluator.track_access(path)

            return result if isinstance(result, dict) else {"data": result}

    async def _search(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> List[SearchResult]:
        """메모리 검색.

        Args:
            paths: 검색 범위 (선택적)
            content: 검색어
            options: 검색 옵션

        Returns:
            검색 결과 목록
        """
        search_type = SearchType(options.get("type", "keyword"))

        # 검색 쿼리 생성
        filters = options.get("filters", {})
        if paths:
            filters["paths"] = paths

        query = SearchQuery(
            query=content or "",
            search_type=search_type,
            filters=filters,
            options=options,
        )
        query.validate()

        # 검색 타입별 처리
        if search_type == SearchType.KEYWORD:
            return await self._search_keyword(query)
        elif search_type == SearchType.TIME_RANGE:
            return await self._search_time_range(query)
        elif search_type == SearchType.SEMANTIC and self.vector_store:
            return await self._search_semantic(query)
        else:
            # 다른 검색 타입은 추후 구현
            logger.warning(f"Search type {search_type} not implemented yet")
            return []

    async def _search_keyword(self, query: SearchQuery) -> List[SearchResult]:
        """키워드 검색 구현.

        Args:
            query: 검색 쿼리

        Returns:
            검색 결과 목록
        """
        client = self.redis.client
        results: List[SearchResult] = []

        # 패턴 생성
        base_pattern = "json:memory:*"

        # 경로 필터가 있으면 적용
        if "paths" in query.filters and query.filters["paths"]:
            path_str = ":".join(query.filters["paths"])
            pattern = f"json:memory:*:{path_str}*"
        else:
            pattern = base_pattern

        # 키 스캔
        cursor = 0
        limit = query.options.get("limit", 10)

        while len(results) < limit:
            cursor, keys = await client.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )

            # 각 키에서 데이터 조회
            for key in keys:
                if len(results) >= limit:
                    break

                try:
                    data = await client.json().get(key, "$")
                    if data and isinstance(data, list) and data[0]:
                        content_data = data[0]

                        # 복호화 적용
                        if self.enable_security and self.field_encryption:
                            content_data = self.field_encryption.decrypt_dict(content_data)

                        # 키워드 매칭 확인
                        if query.query:
                            data_str = str(content_data.get("data", "")).lower()
                            if query.query.lower() not in data_str:
                                continue

                        # MemoryContent 객체로 변환
                        memory_content = MemoryContent.from_dict(content_data)

                        # SearchResult 생성
                        result = SearchResult(
                            key=key,
                            content=memory_content,
                            score=1.0,  # 기본 점수
                        )
                        results.append(result)
                        logger.debug(f"Found match: {key}")

                except Exception as e:
                    logger.warning(f"Error reading key {key}: {e}")
                    continue

            if cursor == 0:
                break

        return results

    async def _search_time_range(self, query: SearchQuery) -> List[SearchResult]:
        """시간 범위 검색 구현.

        Args:
            query: 검색 쿼리

        Returns:
            검색 결과 목록
        """
        # 시간 범위 파싱 (다양한 형식 지원)
        start_date = query.filters.get("start_time") or query.filters.get("from") or query.filters.get("date")
        end_date = query.filters.get("end_time") or query.filters.get("to") or query.filters.get("date")

        if not (start_date or end_date):
            raise ValueError("Time range search requires start_time or date filter")

        # 날짜를 datetime 객체로 변환
        from datetime import datetime
        try:
            if start_date:
                # 날짜만 있으면 시작 시간 추가
                if len(start_date) == 10:  # YYYY-MM-DD
                    start_dt = datetime.fromisoformat(f"{start_date}T00:00:00")
                else:
                    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            else:
                start_dt = None
                
            if end_date:
                # 날짜만 있으면 끝 시간 추가
                if len(end_date) == 10:  # YYYY-MM-DD
                    end_dt = datetime.fromisoformat(f"{end_date}T23:59:59")
                else:
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            else:
                end_dt = None
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")

        # 모든 메모리를 스캔 (패턴 기반 검색은 날짜 형식이 복잡해서 비효율적)
        pattern = "json:memory:*"

        # 키워드 검색과 유사한 로직 사용
        client = self.redis.client
        results: List[SearchResult] = []
        cursor = 0
        limit = query.options.get("limit", 100)

        while len(results) < limit:
            cursor, keys = await client.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )

            for key in keys:
                if len(results) >= limit:
                    break

                try:
                    data = await client.json().get(key, "$")
                    if data and isinstance(data, list) and data[0]:
                        content_data = data[0]
                        
                        # 복호화 적용
                        if self.enable_security and self.field_encryption:
                            content_data = self.field_encryption.decrypt_dict(content_data)
                        
                        memory_content = MemoryContent.from_dict(content_data)
                        
                        # 시간 범위 확인
                        created_at = memory_content.metadata.created_at
                        
                        # 시간 범위 필터링
                        if start_dt and created_at < start_dt:
                            continue
                        if end_dt and created_at > end_dt:
                            continue

                        result = SearchResult(
                            key=key,
                            content=memory_content,
                            score=1.0,
                        )
                        results.append(result)

                except Exception as e:
                    logger.warning(f"Error reading key {key}: {e}")
                    continue

            if cursor == 0:
                break

        # 시간순 정렬
        results.sort(
            key=lambda r: r.content.metadata.created_at,
            reverse=True,
        )

        return results

    async def _search_semantic(self, query: SearchQuery) -> List[SearchResult]:
        """의미 기반 검색 구현.

        Args:
            query: 검색 쿼리

        Returns:
            검색 결과 목록
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []

        # 검색 옵션 준비
        k = query.options.get("limit", 10)
        filters = {}

        # 필터 설정
        if query.filters.get("paths"):
            filters["path_prefix"] = "/".join(query.filters["paths"])
        if query.filters.get("importance"):
            filters["min_importance"] = query.filters["importance"]
        if query.filters.get("type"):
            filters["type"] = query.filters["type"]

        # 하이브리드 검색 여부
        hybrid = query.options.get("hybrid", False)

        try:
            # 벡터 검색 수행
            similar_memories = await self.vector_store.search_similar(
                query_text=query.query, k=k, filters=filters, hybrid=hybrid
            )

            # 결과 변환
            results = []
            for path, score, metadata in similar_memories:
                # Redis에서 실제 데이터 가져오기
                memory_type = MemoryType(metadata.get("type", "document"))
                memory_key = MemoryKey(paths=path.split("/"), memory_type=memory_type)
                key = memory_key.generate()
                json_key = f"json:{key}"

                client = self.redis.client
                data = await client.json().get(json_key, "$")

                if data:
                    content_dict = data[0] if isinstance(data, list) else data
                    memory_content = MemoryContent.from_dict(content_dict)

                    result = SearchResult(
                        key=key,
                        content=memory_content,
                        score=1 - score,  # Convert distance to similarity
                        metadata={"vector_score": score},
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def _update(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> str:
        """메모리 수정.

        Args:
            paths: 수정할 메모리 경로
            content: 새로운 내용
            options: 수정 옵션

        Returns:
            수정된 메모리 키
        """
        if not paths or paths == [""]:
            raise ValueError("Paths are required for update action")
        if content is None:
            raise ValueError("Content is required for update action")

        # 메모리 타입 추론
        memory_type = MemoryType(options.get("type", "document"))

        # 키 생성
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()
        json_key = f"json:{key}"

        client = self.redis.client

        # 기존 데이터 조회
        existing_data = await client.json().get(json_key, "$")
        if not existing_data:
            raise ValueError(f"Memory not found: {key}")

        # MemoryContent 객체로 변환
        existing_content = MemoryContent.from_dict(existing_data[0])

        # 내용 업데이트
        existing_content.data = content
        existing_content.metadata.update()

        # 옵션 업데이트
        if "tags" in options:
            existing_content.metadata.tags = options["tags"]
        if "importance" in options:
            existing_content.metadata.importance = Importance(options["importance"])

        # 저장
        await client.json().set(json_key, "$", existing_content.to_dict())

        logger.info(f"Memory updated: {key}")
        return key

    async def _delete(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """메모리 삭제.

        Args:
            paths: 삭제할 메모리 경로
            content: 사용하지 않음
            options: 삭제 옵션

        Returns:
            삭제 결과
        """
        if not paths or paths == [""]:
            raise ValueError("Paths are required for delete action")

        # 메모리 타입 추론
        memory_type = MemoryType(options.get("type", "document"))

        # 키 생성
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        key = memory_key.generate()

        client = self.redis.client
        deleted_count = 0

        # 데이터 타입별 삭제
        if memory_type == MemoryType.CONVERSATION:
            stream_key = f"stream:{key}"
            if await client.exists(stream_key):
                await client.delete(stream_key)
                deleted_count += 1
        else:
            json_key = f"json:{key}"
            if await client.exists(json_key):
                await client.delete(json_key)
                deleted_count += 1

        # 패턴 삭제 옵션
        if options.get("pattern", False):
            pattern = f"json:{key}:*"
            cursor = 0

            while True:
                cursor, keys = await client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100,
                )

                if keys:
                    deleted_count += await client.delete(*keys)

                if cursor == 0:
                    break

        logger.info(f"Deleted {deleted_count} memory items")

        return {
            "deleted": deleted_count,
            "key": key,
        }

    async def _save_metadata_hash(
        self,
        key: str,
        metadata: MemoryMetadata,
    ) -> None:
        """메타데이터를 Redis Hash로 저장.

        빠른 필드 접근과 원자적 업데이트를 위해 별도 Hash로 저장.

        Args:
            key: 원본 메모리 키
            metadata: 메타데이터 객체
        """
        # Hash 키 생성
        hash_key = f"meta:{key}"

        # Hash 필드 준비 (모든 값을 문자열로 변환)
        hash_fields: Dict[str, str] = {
            "created_at": metadata.created_at.isoformat(),
            "importance": metadata.importance.value,
            "access_count": str(metadata.access_count),
        }

        # 선택적 필드
        if metadata.updated_at:
            hash_fields["updated_at"] = metadata.updated_at.isoformat()
        if metadata.tags:
            hash_fields["tags"] = ",".join(metadata.tags)
        if metadata.source:
            hash_fields["source"] = metadata.source
        if metadata.ttl_seconds:
            hash_fields["ttl_seconds"] = str(metadata.ttl_seconds)
        if metadata.accessed_at:
            hash_fields["accessed_at"] = metadata.accessed_at.isoformat()

        # Redis Hash에 저장
        client = self.redis.client
        await client.hset(hash_key, mapping=hash_fields)  # type: ignore[arg-type]

        # TTL 설정 (메타데이터도 동일한 TTL 적용)
        if metadata.ttl_seconds:
            await client.expire(hash_key, metadata.ttl_seconds)

        logger.debug(f"Metadata saved to hash {hash_key}")

    async def _update_metadata_field(
        self,
        key: str,
        field: str,
        value: Any,
    ) -> None:
        """메타데이터의 특정 필드를 원자적으로 업데이트.

        Args:
            key: 원본 메모리 키
            field: 업데이트할 필드
            value: 새로운 값
        """
        hash_key = f"meta:{key}"
        client = self.redis.client

        # 특별 처리가 필요한 필드들
        if field == "access_count":
            # 접근 횟수는 증가
            await client.hincrby(hash_key, field, 1)
            # 마지막 접근 시간도 업데이트
            await client.hset(
                hash_key, "accessed_at", datetime.now(timezone.utc).isoformat()
            )
        elif field == "tags" and isinstance(value, list):
            # 태그는 리스트를 쉼표로 구분된 문자열로
            await client.hset(hash_key, field, ",".join(value))
        else:
            # 일반 필드 업데이트
            await client.hset(hash_key, field, str(value))

        logger.debug(f"Updated metadata field {field} for {hash_key}")

    async def _get_metadata_from_hash(
        self,
        key: str,
    ) -> Optional[Dict[str, str]]:
        """Hash에서 메타데이터 조회.

        Args:
            key: 원본 메모리 키

        Returns:
            메타데이터 딕셔너리 또는 None
        """
        hash_key = f"meta:{key}"
        client = self.redis.client

        metadata = await client.hgetall(hash_key)
        if not metadata:
            return None

        # 문자열을 원래 타입으로 변환
        if "tags" in metadata and metadata["tags"]:
            metadata["tags"] = metadata["tags"].split(",")
        if "access_count" in metadata:
            metadata["access_count"] = int(metadata["access_count"])
        if "ttl_seconds" in metadata:
            metadata["ttl_seconds"] = int(metadata["ttl_seconds"])

        return metadata

    async def _update_json_path(
        self,
        key: str,
        path: str,
        value: Any,
    ) -> None:
        """JSONPath를 사용한 부분 업데이트.

        RedisJSON의 JSONPath 기능을 활용하여 중첩된 JSON 구조의
        특정 부분만 효율적으로 업데이트.

        Args:
            key: Redis 키
            path: JSONPath 표현식 (예: "$.data.settings.theme")
            value: 업데이트할 값
        """
        # json: prefix가 없으면 추가
        if not key.startswith("json:"):
            json_key = f"json:{key}"
        else:
            json_key = key

        client = self.redis.client

        try:
            # JSONPath를 사용한 부분 업데이트
            await client.json().set(json_key, path, value)

            # 메타데이터의 updated_at 업데이트
            await self._update_metadata_field(
                key, "updated_at", datetime.now(timezone.utc).isoformat()
            )

            logger.debug(f"Updated JSON path {path} in {json_key}")
        except Exception as e:
            logger.error(f"Failed to update JSON path {path}: {e}")
            raise

    async def _get_stream_range(
        self,
        key: str,
        start_id: str = "-",
        end_id: str = "+",
        count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Redis Streams에서 시간 범위 기반 조회.

        Args:
            key: 스트림 키
            start_id: 시작 ID ("-"는 처음부터)
            end_id: 종료 ID ("+"는 끝까지)
            count: 최대 조회 개수

        Returns:
            메시지 리스트
        """
        stream_key = f"stream:{key}"
        client = self.redis.client

        try:
            # 범위 조회
            if count:
                messages = await client.xrange(
                    stream_key, min=start_id, max=end_id, count=count
                )
            else:
                messages = await client.xrange(stream_key, min=start_id, max=end_id)

            # 메시지 포맷팅
            result = []
            for msg_id, data in messages:
                # 메타데이터 분리
                metadata_str = data.pop("_metadata", "{}")
                metadata = json.loads(metadata_str) if metadata_str else {}
                result.append(
                    {
                        "id": msg_id,
                        "data": data,
                        "metadata": metadata,
                        "timestamp": self._parse_stream_timestamp(msg_id),
                    }
                )

            return result
        except Exception as e:
            logger.error(f"Failed to get stream range: {e}")
            return []

    def _parse_stream_timestamp(self, stream_id: str) -> datetime:
        """Stream ID에서 타임스탬프 추출.

        Args:
            stream_id: Redis Stream ID (예: "1234567890-0")

        Returns:
            datetime 객체
        """
        timestamp_ms = int(stream_id.split("-")[0])
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    def _generate_key(self, paths: List[str], memory_type: MemoryType) -> str:
        """메모리 키 생성.

        Args:
            paths: 경로 리스트
            memory_type: 메모리 타입

        Returns:
            생성된 키
        """
        memory_key = MemoryKey(paths=paths, memory_type=memory_type)
        return memory_key.generate()

    async def _consolidate(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """메모리 통합 실행.

        Args:
            paths: 통합할 메모리 경로
            content: 중복 검사할 내용 (선택적)
            options: 통합 옵션

        Returns:
            통합 결과
        """
        if not self.consolidator:
            raise RuntimeError("Consolidator not initialized (vector search disabled)")

        consolidation_type = options.get("type", "path")

        if consolidation_type == "path":
            # 경로 기반 통합
            path_prefix = "/".join(paths) if paths else ""
            result = await self.consolidator.consolidate_by_path(
                path_prefix=path_prefix, time_window=options.get("time_window")
            )

        elif consolidation_type == "duplicate":
            # 중복 기반 통합
            if content is None:
                raise ValueError("Content required for duplicate consolidation")

            result = await self.consolidator.consolidate_duplicates(
                content=content, path_prefix="/".join(paths) if paths else None
            )

        elif consolidation_type == "temporal":
            # 시간 기반 통합
            from datetime import timedelta

            path_prefix = "/".join(paths) if paths else ""
            time_buckets = options.get(
                "time_buckets",
                [timedelta(hours=1), timedelta(hours=6), timedelta(days=1)],
            )

            result = await self.consolidator.consolidate_temporal(
                path_prefix=path_prefix, time_buckets=time_buckets
            )

        else:
            raise ValueError(f"Unknown consolidation type: {consolidation_type}")

        return result.to_dict()

    async def _lifecycle(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """생명주기 관리 작업 실행.

        Args:
            paths: 대상 메모리 경로
            content: 사용자 중요도 평가 (선택적)
            options: 생명주기 옵션

        Returns:
            생명주기 작업 결과
        """
        lifecycle_action = options.get("action", "evaluate")

        if lifecycle_action == "evaluate":
            # 중요도 평가 및 TTL 업데이트
            if not paths or paths == [""]:
                raise ValueError("Paths required for importance evaluation")

            path = "/".join(paths)
            user_rating = float(content) if content is not None else None

            # 중요도 업데이트
            score = await self.lifecycle_manager.update_importance(
                path=path, user_rating=user_rating
            )

            return {
                "path": path,
                "importance_score": score.to_dict(),
                "tier": self.lifecycle_manager.policy.ttl_policy.get_tier(
                    score.overall_score
                ).value,
            }

        elif lifecycle_action == "archive":
            # 메모리 아카이빙
            if not paths or paths == [""]:
                raise ValueError("Paths required for archiving")

            archived = []
            failed = []

            for path_parts in [paths]:  # Support batch operations later
                path = "/".join(path_parts)
                success = await self.lifecycle_manager.migrate_to_archive(path)

                if success:
                    archived.append(path)
                else:
                    failed.append(path)

            return {
                "archived": archived,
                "failed": failed,
                "total": len(archived) + len(failed),
            }

        elif lifecycle_action == "restore":
            # 아카이브에서 복원
            if not paths or paths == [""]:
                raise ValueError("Paths required for restoration")

            restored = []
            failed = []

            for path_parts in [paths]:
                path = "/".join(path_parts)
                success = await self.lifecycle_manager.restore_from_archive(path)

                if success:
                    restored.append(path)
                else:
                    failed.append(path)

            return {
                "restored": restored,
                "failed": failed,
                "total": len(restored) + len(failed),
            }

        elif lifecycle_action == "stats":
            # 스토리지 통계
            stats = await self.lifecycle_manager.get_storage_stats()
            return stats

        else:
            raise ValueError(
                f"Unknown lifecycle action: {lifecycle_action}. "
                f"Valid actions: evaluate, archive, restore, stats"
            )

    async def _log_audit(
        self,
        event_type: AuditEventType,
        principal: Optional[Principal],
        resource: str,
        action: str,
        result: str = "success",
        metadata: Optional[Dict] = None,
    ) -> None:
        """감사 로그 기록.

        Args:
            event_type: 이벤트 타입
            principal: 실행 주체
            resource: 접근 리소스
            action: 실행 액션
            result: 결과 (success/failure)
            metadata: 추가 메타데이터
        """
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=event_type,
                principal_id=principal.id if principal else None,
                resource=resource,
                action=action,
                result=result,
                metadata=metadata,
            )

    def _get_audit_event_type(self, action: str) -> AuditEventType:
        """액션에 대응하는 감사 이벤트 타입 반환."""
        event_map = {
            "save": AuditEventType.WRITE,
            "get": AuditEventType.READ,
            "search": AuditEventType.SEARCH,
            "update": AuditEventType.WRITE,
            "delete": AuditEventType.DELETE,
        }
        return event_map.get(action, AuditEventType.READ)

    async def _consolidate(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """메모리 통합 수행."""
        consolidate_type = options.get("type", "path")
        
        if consolidate_type == "path":
            if not paths:
                raise ValueError("Paths are required for path-based consolidation")
            
            # 해당 경로의 모든 메모리 검색
            search_result = await self._search(
                paths=paths,
                content="*",
                options={"type": "keyword", "limit": 1000},
            )
            
            # 유사한 메모리 그룹화
            merged_count = 0
            groups = self._group_similar_memories(search_result)
            
            for group in groups:
                if len(group) > 1:
                    # 그룹 내 메모리 병합
                    await self._merge_memory_group(group)
                    merged_count += len(group) - 1
            
            return {
                "type": "path",
                "paths": paths,
                "total_memories": len(search_result),
                "merges_completed": merged_count,
                "groups_found": len(groups),
            }
        
        elif consolidate_type == "duplicate":
            # 전체 또는 특정 content 기반 중복 검사
            if content:
                # 특정 content와 유사한 메모리 검색
                search_result = await self._search(
                    paths=[],
                    content=content,
                    options={"type": "keyword", "limit": 100},
                )
            else:
                # 전체 메모리 중복 검사
                search_result = await self._search(
                    paths=[],
                    content="*",
                    options={"type": "keyword", "limit": 1000},
                )
            
            # 중복 감지 및 병합
            duplicates = self._find_duplicate_memories(search_result)
            merged_count = 0
            
            for dup_group in duplicates:
                if len(dup_group) > 1:
                    await self._merge_memory_group(dup_group)
                    merged_count += len(dup_group) - 1
            
            return {
                "type": "duplicate",
                "total_checked": len(search_result),
                "duplicates_found": sum(len(g) - 1 for g in duplicates),
                "merges_completed": merged_count,
            }
        
        elif consolidate_type == "temporal":
            # 시간 기반 통합
            time_buckets = options.get("time_buckets", ["1d", "7d", "30d"])
            
            results = {
                "type": "temporal",
                "buckets": {},
            }
            
            for bucket in time_buckets:
                # 시간 범위 계산
                days = self._parse_time_bucket(bucket)
                start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
                
                # 해당 기간 메모리 검색
                search_result = await self._search(
                    paths=paths,
                    content="",
                    options={
                        "type": "time_range",
                        "filters": {"from": start_date},
                        "limit": 1000,
                    },
                )
                
                # 시간 단위로 그룹화
                time_groups = self._group_by_time_unit(search_result, bucket)
                merged_count = 0
                
                for group in time_groups:
                    if len(group) > 1 and self._should_merge_temporal(group):
                        await self._merge_memory_group(group)
                        merged_count += len(group) - 1
                
                results["buckets"][bucket] = {
                    "memories": len(search_result),
                    "groups": len(time_groups),
                    "merged": merged_count,
                }
            
            return results
        
        else:
            raise ValueError(f"Unknown consolidation type: {consolidate_type}")

    async def _lifecycle(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """생명주기 관리 수행."""
        action = options.get("action", "stats")
        
        if action == "evaluate":
            if not paths:
                raise ValueError("Paths are required for evaluation")
            
            # 메모리 조회 시도
            try:
                memory_data = await self._get(paths, None, {})
                
                # 간단한 중요도 점수 계산
                score = 0.5  # 기본 점수
                tier = "medium"  # 기본 티어
                
                return {
                    "action": "evaluate",
                    "paths": paths,
                    "importance_score": {
                        "overall_score": score,
                        "access_frequency": 0,
                        "user_rating": content if isinstance(content, (int, float)) else None,
                    },
                    "tier": tier,
                    "new_ttl_days": 90,
                }
                
            except Exception as e:
                logger.error(f"Failed to evaluate memory: {e}")
                return {
                    "action": "evaluate",
                    "paths": paths,
                    "error": str(e),
                }
        
        elif action == "archive":
            # 간소화된 아카이브 구현
            return {
                "action": "archive",
                "threshold_days": options.get("threshold_days", 90),
                "total": 0,
                "archived": [],
                "message": "Archive functionality is not yet fully implemented",
            }
        
        elif action == "restore":
            # 간소화된 복원 구현
            return {
                "action": "restore",
                "paths": paths,
                "restored": [],
                "message": "Restore functionality is not yet fully implemented",
            }
        
        elif action == "stats":
            # 기본 통계
            client = self.redis.client
            
            # 전체 메모리 수 계산
            total_memories = 0
            cursor = 0
            pattern = "json:memory:*"
            
            while True:
                cursor, keys = await client.scan(cursor=cursor, match=pattern, count=100)
                total_memories += len(keys)
                if cursor == 0:
                    break
            
            return {
                "action": "stats",
                "total_memories": total_memories,
                "importance_distribution": {
                    "critical": 0,
                    "high": 0,
                    "medium": total_memories,  # 임시로 모두 medium으로
                    "low": 0,
                    "trivial": 0,
                },
                "storage_info": {
                    "hot": 0,
                    "warm": total_memories,
                    "cold": 0,
                },
            }
        
        else:
            raise ValueError(f"Unknown lifecycle action: {action}")
    
    # Consolidation 헬퍼 메서드들
    def _group_similar_memories(self, memories: List[SearchResult]) -> List[List[SearchResult]]:
        """유사한 메모리를 그룹화."""
        groups = []
        used = set()
        
        for i, mem1 in enumerate(memories):
            if i in used:
                continue
            
            group = [mem1]
            used.add(i)
            
            content1 = str(mem1.content.data)[:100].lower()
            
            for j, mem2 in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                content2 = str(mem2.content.data)[:100].lower()
                
                # 간단한 유사도 체크 (첫 100자의 70% 이상 일치)
                similarity = self._calculate_similarity(content1, content2)
                if similarity > 0.7:
                    group.append(mem2)
                    used.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _find_duplicate_memories(self, memories: List[SearchResult]) -> List[List[SearchResult]]:
        """중복 메모리 찾기."""
        from hashlib import sha256
        
        hash_groups = {}
        
        for memory in memories:
            # 내용의 해시 생성
            content_str = str(memory.content.data)
            content_hash = sha256(content_str.encode()).hexdigest()
            
            if content_hash not in hash_groups:
                hash_groups[content_hash] = []
            hash_groups[content_hash].append(memory)
        
        return [group for group in hash_groups.values() if len(group) > 1]
    
    async def _merge_memory_group(self, group: List[SearchResult]) -> None:
        """메모리 그룹 병합."""
        if len(group) < 2:
            return
        
        # 가장 중요하거나 최신 메모리를 기준으로 선택
        primary = max(group, key=lambda m: (
            self._importance_to_int(m.content.metadata.importance),
            m.content.metadata.created_at
        ))
        
        # 나머지 메모리들의 정보를 primary에 병합
        merged_tags = set(primary.content.metadata.tags)
        total_access_count = primary.content.metadata.access_count
        
        for memory in group:
            if memory.key == primary.key:
                continue
            
            # 태그 병합
            merged_tags.update(memory.content.metadata.tags)
            
            # 접근 횟수 합산
            total_access_count += memory.content.metadata.access_count
            
            # 중복 메모리 삭제
            key_parts = memory.key.split(":")
            if len(key_parts) >= 3:
                await self._delete(key_parts[2:], None, {})
        
        # primary 메모리 업데이트
        primary_key_parts = primary.key.split(":")
        if len(primary_key_parts) >= 3:
            # 메타데이터 업데이트
            await self._update_metadata_field(
                primary.key.replace("json:", ""),
                "tags",
                list(merged_tags)
            )
            await self._update_metadata_field(
                primary.key.replace("json:", ""),
                "access_count",
                total_access_count
            )
    
    def _parse_time_bucket(self, bucket: str) -> int:
        """시간 버킷 문자열을 일 수로 변환."""
        if bucket.endswith("h"):
            hours = int(bucket[:-1])
            return max(1, hours // 24)  # 최소 1일
        elif bucket.endswith("d"):
            return int(bucket[:-1])
        elif bucket.endswith("w"):
            return int(bucket[:-1]) * 7
        elif bucket.endswith("m"):
            return int(bucket[:-1]) * 30
        else:
            return 1
    
    def _group_by_time_unit(self, memories: List[SearchResult], unit: str) -> List[List[SearchResult]]:
        """시간 단위로 메모리 그룹화."""
        groups = {}
        
        for memory in memories:
            created = memory.content.metadata.created_at
            
            if unit.endswith("h"):
                bucket = created.strftime("%Y-%m-%d-%H")
            elif unit.endswith("d"):
                bucket = created.strftime("%Y-%m-%d")
            elif unit.endswith("w"):
                # 주 단위
                bucket = created.strftime("%Y-%W")
            else:
                bucket = created.strftime("%Y-%m")
            
            if bucket not in groups:
                groups[bucket] = []
            groups[bucket].append(memory)
        
        return list(groups.values())
    
    def _should_merge_temporal(self, group: List[SearchResult]) -> bool:
        """시간 기반 병합 여부 결정."""
        # 그룹 내 모든 메모리가 낮은 중요도인 경우에만 병합
        for memory in group:
            if memory.content.metadata.importance in [Importance.HIGH, Importance.CRITICAL]:
                return False
        return True
    
    def _importance_to_int(self, importance: Importance) -> int:
        """중요도를 정수로 변환."""
        importance_map = {
            Importance.CRITICAL: 5,
            Importance.HIGH: 4,
            Importance.MEDIUM: 3,
            Importance.LOW: 2,
            Importance.TRIVIAL: 1,
        }
        return importance_map.get(importance, 3)
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """두 문자열의 유사도 계산 (0-1)."""
        # 간단한 문자 단위 유사도
        if not str1 or not str2:
            return 0.0
        
        # 더 짧은 문자열의 길이
        min_len = min(len(str1), len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        # 일치하는 문자 수 계산
        matches = sum(1 for c1, c2 in zip(str1, str2) if c1 == c2)
        
        # 유사도 계산
        return matches / max_len
