"""Search memory actions."""

import json
import logging
from datetime import datetime, timezone
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
    """Search operations handler."""
    
    def __init__(
        self, 
        redis_client: RedisClient,
        vector_store: Optional[Any] = None,
        field_encryption: Optional[Any] = None
    ):
        """Initialize search actions.
        
        Args:
            redis_client: Redis client instance
            vector_store: Vector store for semantic search
            field_encryption: Field encryption handler
        """
        self.redis = redis_client
        self.vector_store = vector_store
        self.field_encryption = field_encryption
        
    async def execute(
        self,
        paths: List[str],
        content: Optional[Any],
        options: Dict[str, Any]
    ) -> List[SearchResult]:
        """Execute search action.
        
        Args:
            paths: Search paths
            content: Search query
            options: Search options
            
        Returns:
            List of search results
        """
        return await self._search(paths, content, options)
    
    async def _search(
        self,
        paths: List[str],
        content: Any,
        options: Dict[str, Any],
    ) -> List[SearchResult]:
        """메모리 검색.

        다양한 검색 방식을 지원하며 필터와 정렬 옵션을
        제공합니다.

        Args:
            paths: 검색 경로 (비어있으면 전체)
            content: 검색 쿼리
            options: 검색 옵션

        Returns:
            검색 결과 리스트
        """
        # 검색 쿼리 생성
        search_type = SearchType(options.get("type", SearchType.KEYWORD.value))
        
        query = SearchQuery(
            query=content if isinstance(content, str) else "",
            search_type=search_type,
            filters=options.get("filters", {}),
            options={
                "paths": paths if paths else [],
                "limit": options.get("limit", 10),
                "offset": options.get("offset", 0),
            }
        )
        
        # 쿼리 검증
        query.validate()
        
        # 검색 타입별 처리
        if query.search_type == SearchType.KEYWORD:
            return await self._search_keyword(query, paths)
        elif query.search_type == SearchType.TIME_RANGE:
            return await self._search_time_range(query, paths)
        elif query.search_type == SearchType.SEMANTIC:
            return await self._search_semantic(query, paths)
        else:
            raise ValueError(f"Unsupported search type: {query.search_type}")

    def _search_in_data(self, data: Any, keyword: str) -> bool:
        """재귀적으로 데이터에서 키워드를 검색.
        
        Args:
            data: 검색할 데이터
            keyword: 검색 키워드 (소문자)
            
        Returns:
            키워드 발견 여부
        """
        if isinstance(data, str):
            return keyword in data.lower()
        elif isinstance(data, dict):
            # 키와 값 모두 검색
            for key, value in data.items():
                if keyword in str(key).lower() or self._search_in_data(value, keyword):
                    return True
        elif isinstance(data, list):
            # 리스트의 모든 요소 검색
            for item in data:
                if self._search_in_data(item, keyword):
                    return True
        elif data is not None:
            # 기타 타입은 문자열로 변환해서 검색
            return keyword in str(data).lower()
        return False

    async def _search_keyword(self, query: SearchQuery, paths: List[str]) -> List[SearchResult]:
        """키워드 기반 검색.

        Args:
            query: 검색 쿼리

        Returns:
            검색 결과
        """
        client = self.redis.client
        results = []
        
        # 검색 패턴 생성
        if paths:
            # 경로 기반 패턴 (예: *:test:* 또는 *:projects:BlueprintAI:*)
            base_pattern = f"*{':'.join(paths)}*"
        else:
            base_pattern = "*"
        
        # 키워드 기반 검색 패턴 생성
        if query.query and query.query not in ["", "*"]:
            search_patterns = [
                f"json:{base_pattern}",
                f"meta:{base_pattern}",
            ]
            # 키워드는 나중에 내용 필터링으로 처리
        else:
            # 경로만으로 검색
            search_patterns = [
                f"json:{base_pattern}",
                f"meta:{base_pattern}",
            ]
        
        # 각 패턴으로 검색
        matched_keys = set()
        for pattern in search_patterns:
            cursor = 0
            while True:
                cursor, keys = await client.scan(
                    cursor, 
                    match=pattern, 
                    count=100
                )
                matched_keys.update(keys)
                if cursor == 0:
                    break
        
        # JSON 키만 필터링하고 키워드가 키에 포함되어 있는지도 확인
        json_keys = []
        for key in matched_keys:
            if key.startswith("json:"):
                # 키워드가 있으면 키 자체에도 키워드가 포함되어 있는지 확인
                if query.query and query.query != "*":
                    if query.query.lower() in key.lower():
                        json_keys.append(key)
                    else:
                        # 키에 없으면 일단 후보로 추가 (데이터에서 찾을 수 있음)
                        json_keys.append(key)
                else:
                    json_keys.append(key)
        
        # 결과 조회
        limit = query.options.get("limit", 10)
        for json_key in json_keys[:limit]:
            try:
                # JSON 데이터 조회
                data = await client.json().get(json_key)
                if not data:
                    continue
                
                # 키워드 필터링 (재귀적 검색)
                if query.query and query.query != "*":
                    if not self._search_in_data(data, query.query.lower()):
                        continue
                
                # 복호화
                if self.field_encryption and data.get("_encrypted"):
                    data = self.field_encryption.decrypt_dict(data)
                
                # 메타데이터 조회
                meta_key = json_key.replace("json:", "meta:")
                metadata = await client.hgetall(meta_key)
                
                # 필터 적용
                if query.filters:
                    if "tags" in query.filters:
                        # 태그 필터
                        match = True
                        for tag_key, tag_value in query.filters["tags"].items():
                            if metadata.get(f"tag:{tag_key}") != str(tag_value):
                                match = False
                                break
                        if not match:
                            continue
                    
                    if "importance" in query.filters:
                        # 중요도 필터
                        if metadata.get("importance") != query.filters["importance"]:
                            continue
                
                # SearchResult 생성
                base_key = json_key.replace("json:", "")
                
                # data가 dict인 경우 MemoryContent로 변환
                if isinstance(data, dict):
                    memory_content = MemoryContent.from_dict(data)
                else:
                    memory_content = data
                
                result = SearchResult(
                    key=base_key,
                    score=1.0,  # 키워드 검색은 관련도 점수 없음
                    content=memory_content,
                    metadata=metadata,
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing key {json_key}: {e}")
                continue
        
        return results

    async def _search_time_range(self, query: SearchQuery, paths: List[str]) -> List[SearchResult]:
        """시간 범위 기반 검색.

        Args:
            query: 검색 쿼리

        Returns:
            검색 결과
        """
        if not query.filters:
            raise ValueError("Time range search requires filters")
        
        client = self.redis.client
        results = []
        
        # 시간 범위 추출
        from_date = query.filters.get("from")
        to_date = query.filters.get("to")
        date = query.filters.get("date")
        
        # 날짜 패턴 생성
        if date:
            # 특정 날짜
            date_parts = date.split("-")
            if len(date_parts) == 3:  # YYYY-MM-DD
                pattern = f"*{date_parts[0]}-{date_parts[1]}-{date_parts[2]}*"
            elif len(date_parts) == 2:  # YYYY-MM
                pattern = f"*{date_parts[0]}-{date_parts[1]}*"
            else:  # YYYY
                pattern = f"*{date_parts[0]}*"
        else:
            # 전체 검색
            pattern = "*"
        
        # 경로 필터 추가
        if paths:
            path_pattern = ":".join(paths)
            pattern = f"*{path_pattern}*" if pattern == "*" else f"*{path_pattern}*{pattern}*"
        
        # 메타데이터 키 검색
        meta_pattern = f"meta:{pattern}"
        matched_keys = set()
        
        cursor = 0
        while True:
            cursor, keys = await client.scan(
                cursor,
                match=meta_pattern,
                count=100
            )
            matched_keys.update(keys)
            if cursor == 0:
                break
        
        # 시간 범위 필터링
        for meta_key in matched_keys:
            try:
                metadata = await client.hgetall(meta_key)
                created_at = metadata.get("created_at")
                
                if not created_at:
                    continue
                
                # ISO 형식 파싱
                created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                
                # 시간 범위 체크
                if from_date:
                    from_time = datetime.fromisoformat(f"{from_date}T00:00:00+00:00")
                    if created_time < from_time:
                        continue
                
                if to_date:
                    to_time = datetime.fromisoformat(f"{to_date}T23:59:59+00:00")
                    if created_time > to_time:
                        continue
                
                # JSON 데이터 조회
                json_key = meta_key.replace("meta:", "json:")
                data = await client.json().get(json_key)
                
                if not data:
                    continue
                
                # 복호화
                if self.field_encryption and data.get("_encrypted"):
                    data = self.field_encryption.decrypt_dict(data)
                
                # SearchResult 생성
                base_key = meta_key.replace("meta:", "")
                
                # data가 dict인 경우 MemoryContent로 변환
                if isinstance(data, dict):
                    memory_content = MemoryContent.from_dict(data)
                else:
                    memory_content = data
                    
                result = SearchResult(
                    key=base_key,
                    score=1.0,
                    content=memory_content,
                    metadata=metadata,
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing key {meta_key}: {e}")
                continue
        
        # 시간순 정렬
        results.sort(
            key=lambda r: r.metadata.get("created_at", ""),
            reverse=True
        )
        
        # 페이지네이션
        offset = query.options.get("offset", 0)
        limit = query.options.get("limit", 10)
        start = offset
        end = start + limit
        
        return results[start:end]

    async def _search_semantic(self, query: SearchQuery, paths: List[str]) -> List[SearchResult]:
        """의미 기반 검색 (벡터 검색).

        Args:
            query: 검색 쿼리

        Returns:
            검색 결과
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not configured for semantic search")
        
        if not query.query:
            raise ValueError("Query text required for semantic search")
        
        # 벡터 검색 수행
        limit = query.options.get("limit", 10)
        vector_results = await self.vector_store.search(
            query=query.query,
            limit=limit,
            filters=query.filters,
        )
        
        # SearchResult로 변환
        results = []
        client = self.redis.client
        
        for vr in vector_results:
            try:
                # 메모리 키 생성
                paths = vr["metadata"]["path"].split("/")
                memory_type = MemoryType(vr["metadata"].get("type", "document"))
                memory_key = MemoryKey(paths=paths, memory_type=memory_type)
                key = memory_key.generate()
                
                # JSON 데이터 조회
                json_key = f"json:{key}"
                data = await client.json().get(json_key)
                
                if not data:
                    continue
                
                # 복호화
                if self.field_encryption and data.get("_encrypted"):
                    data = self.field_encryption.decrypt_dict(data)
                
                # 메타데이터 조회
                meta_key = f"meta:{key}"
                metadata = await client.hgetall(meta_key)
                
                # SearchResult 생성
                # data가 dict인 경우 MemoryContent로 변환
                if isinstance(data, dict):
                    memory_content = MemoryContent.from_dict(data)
                else:
                    memory_content = data
                    
                result = SearchResult(
                    key=key,
                    score=vr["score"],
                    content=memory_content,
                    metadata=metadata,
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing vector result: {e}")
                continue
        
        return results