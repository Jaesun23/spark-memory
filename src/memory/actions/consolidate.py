"""Consolidate memory actions."""

import logging
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional

from src.memory.models import (
    Importance,
    MemoryKey,
    MemoryType,
    SearchResult,
)
from src.redis.client import RedisClient
from src.memory.actions.search import SearchActions

logger = logging.getLogger(__name__)


class ConsolidateActions:
    """Memory consolidation operations handler."""
    
    def __init__(self, redis_client: RedisClient):
        """Initialize consolidate actions.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.search_actions = SearchActions(redis_client)
        
    async def execute(
        self,
        paths: List[str],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute consolidate action.
        
        Args:
            paths: Memory paths for consolidation
            options: Consolidation options
            
        Returns:
            Consolidation results
        """
        return await self._consolidate(paths, None, options)
    
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
            search_result = await self.search_actions._search(
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
                search_result = await self.search_actions._search(
                    paths=[],
                    content=content,
                    options={"type": "keyword", "limit": 100},
                )
            else:
                # 전체 메모리 중복 검사
                search_result = await self.search_actions._search(
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
                search_result = await self.search_actions._search(
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
            
            content1 = str(mem1.content.get("data", ""))[:100].lower()
            
            for j, mem2 in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                content2 = str(mem2.content.get("data", ""))[:100].lower()
                
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
        hash_groups = {}
        
        for memory in memories:
            # 내용의 해시 생성
            content_str = str(memory.content.get("data", ""))
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
            self._importance_to_int(m.metadata.get("importance", "medium")),
            m.metadata.get("created_at", "")
        ))
        
        # 나머지 메모리들의 정보를 primary에 병합
        merged_tags = {}
        if primary.metadata:
            # 기존 태그 수집
            for key, value in primary.metadata.items():
                if key.startswith("tag:"):
                    tag_name = key[4:]  # Remove "tag:" prefix
                    merged_tags[tag_name] = value
        
        total_access_count = int(primary.metadata.get("access_count", 0))
        
        client = self.redis.client
        
        for memory in group:
            if memory.key == primary.key:
                continue
            
            # 태그 병합
            if memory.metadata:
                for key, value in memory.metadata.items():
                    if key.startswith("tag:"):
                        tag_name = key[4:]
                        merged_tags[tag_name] = value
            
            # 접근 횟수 합산
            total_access_count += int(memory.metadata.get("access_count", 0))
            
            # 중복 메모리 삭제
            await self._delete_memory(memory.key)
        
        # primary 메모리 업데이트
        await self._update_memory_metadata(primary.key, {
            "tags": merged_tags,
            "access_count": total_access_count,
        })
    
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
            created_str = memory.metadata.get("created_at", "")
            if not created_str:
                continue
                
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                
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
            except Exception as e:
                logger.warning(f"Failed to parse created_at: {created_str}, error: {e}")
                continue
        
        return list(groups.values())
    
    def _should_merge_temporal(self, group: List[SearchResult]) -> bool:
        """시간 기반 병합 여부 결정."""
        # 그룹 내 모든 메모리가 낮은 중요도인 경우에만 병합
        for memory in group:
            importance = memory.metadata.get("importance", "medium")
            if importance in ["high", "critical"]:
                return False
        return True
    
    def _importance_to_int(self, importance: str) -> int:
        """중요도를 정수로 변환."""
        importance_map = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "trivial": 1,
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
    
    async def _delete_memory(self, key: str) -> None:
        """메모리 삭제 헬퍼."""
        client = self.redis.client
        
        # 관련 키들 삭제
        keys_to_delete = [
            f"json:{key}",
            f"meta:{key}",
            f"stream:{key}",
        ]
        
        for k in keys_to_delete:
            if await client.exists(k):
                await client.delete(k)
    
    async def _update_memory_metadata(self, key: str, updates: Dict[str, Any]) -> None:
        """메모리 메타데이터 업데이트."""
        client = self.redis.client
        meta_key = f"meta:{key}"
        
        # 메타데이터 Hash 업데이트
        metadata_updates = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        if "access_count" in updates:
            metadata_updates["access_count"] = str(updates["access_count"])
        
        if "tags" in updates:
            for tag_key, tag_value in updates["tags"].items():
                metadata_updates[f"tag:{tag_key}"] = str(tag_value)
        
        await client.hset(meta_key, mapping=metadata_updates)