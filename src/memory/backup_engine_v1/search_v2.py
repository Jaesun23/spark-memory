"""Search memory actions (keyword, time range, semantic)."""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from src.memory.models import (
    MemoryContent,
    MemoryKey,
    MemoryType,
    SearchQuery,
    SearchResult,
    SearchType,
)
from src.redis.client import RedisClient

logger = logging.getLogger(__name__)


class SearchActions:
    """Search memory operations handler."""
    
    def __init__(self, redis_client: RedisClient, vector_store=None, field_encryption=None):
        """Initialize search actions.
        
        Args:
            redis_client: Redis client instance
            vector_store: Vector store for semantic search (optional)
            field_encryption: Field encryption service (optional)
        """
        self.redis = redis_client
        self.vector_store = vector_store
        self.field_encryption = field_encryption
        
    async def execute(
        self,
        action: str,
        paths: List[str],
        content: Optional[Any],
        options: Dict[str, Any]
    ) -> List[SearchResult]:
        """Execute search action.
        
        Args:
            action: Should be "search"
            paths: Search scope paths (optional)
            content: Search query string
            options: Search options including type, filters, etc.
            
        Returns:
            List of search results
        """
        if action != "search":
            raise ValueError(f"Invalid action for SearchActions: {action}")
            
        return await self._search(paths, content, options)
    
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
                        if self.field_encryption:
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
                        if self.field_encryption:
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