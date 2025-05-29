"""Lifecycle memory actions."""

import logging
from typing import Any, Dict, List, Optional

from src.redis.client import RedisClient
from src.memory.actions.basic import BasicActions

logger = logging.getLogger(__name__)


class LifecycleActions:
    """Memory lifecycle operations handler."""
    
    def __init__(self, redis_client: RedisClient):
        """Initialize lifecycle actions.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.basic_actions = BasicActions(redis_client)
        
    async def execute(
        self,
        paths: List[str],
        content: Optional[Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute lifecycle action.
        
        Args:
            paths: Memory paths
            content: User rating or importance value for evaluate
            options: Lifecycle options
            
        Returns:
            Lifecycle action results
        """
        return await self._lifecycle(paths, content, options)
    
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
                memory_data = await self.basic_actions._get(paths, None, {})
                
                # 간단한 중요도 점수 계산
                score = 0.5  # 기본 점수
                tier = "medium"  # 기본 티어
                
                # 사용자가 제공한 점수가 있으면 사용
                if isinstance(content, (int, float)):
                    user_rating = float(content)
                    score = user_rating
                    
                    # 티어 결정
                    if score >= 0.8:
                        tier = "critical"
                    elif score >= 0.6:
                        tier = "high"
                    elif score >= 0.4:
                        tier = "medium"
                    elif score >= 0.2:
                        tier = "low"
                    else:
                        tier = "trivial"
                
                # TTL 결정 (중요도에 따라)
                ttl_days_map = {
                    "critical": 365,  # 1년
                    "high": 180,      # 6개월
                    "medium": 90,     # 3개월
                    "low": 30,        # 1개월
                    "trivial": 7,     # 1주일
                }
                new_ttl_days = ttl_days_map.get(tier, 90)
                
                return {
                    "action": "evaluate",
                    "paths": paths,
                    "importance_score": {
                        "overall_score": score,
                        "access_frequency": 0,
                        "user_rating": content if isinstance(content, (int, float)) else None,
                    },
                    "tier": tier,
                    "new_ttl_days": new_ttl_days,
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
            threshold_days = options.get("threshold_days", 90)
            
            # 아카이브할 메모리 찾기
            archived = []
            # TODO: 실제 아카이브 로직 구현 필요
            
            return {
                "action": "archive",
                "threshold_days": threshold_days,
                "total": len(archived),
                "archived": archived,
                "message": "Archive functionality is simplified in v2",
            }
        
        elif action == "restore":
            # 간소화된 복원 구현
            restored = []
            # TODO: 실제 복원 로직 구현 필요
            
            return {
                "action": "restore",
                "paths": paths,
                "restored": restored,
                "failed": [],
                "total": len(restored),
                "message": "Restore functionality is simplified in v2",
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
            
            # 메타데이터 기반 통계 (샘플링)
            importance_dist = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "trivial": 0,
            }
            
            # 샘플 키들의 메타데이터 확인
            sample_size = min(100, total_memories)
            if sample_size > 0:
                cursor = 0
                sample_keys = []
                cursor, keys = await client.scan(cursor=0, match="meta:memory:*", count=sample_size)
                sample_keys.extend(keys)
                
                for meta_key in sample_keys[:sample_size]:
                    try:
                        metadata = await client.hgetall(meta_key)
                        importance = metadata.get("importance", "medium")
                        if importance in importance_dist:
                            importance_dist[importance] += 1
                    except Exception as e:
                        logger.warning(f"Failed to get metadata for {meta_key}: {e}")
                
                # 비율로 전체 추정
                if sample_keys:
                    ratio = total_memories / len(sample_keys)
                    for tier in importance_dist:
                        importance_dist[tier] = int(importance_dist[tier] * ratio)
            
            # 모든 메모리가 통계에 포함되도록 조정
            counted = sum(importance_dist.values())
            if counted < total_memories:
                importance_dist["medium"] += total_memories - counted
            
            return {
                "action": "stats",
                "total_memories": total_memories,
                "importance_distribution": importance_dist,
                "storage_info": {
                    "hot": 0,  # 자주 접근
                    "warm": total_memories,  # 보통
                    "cold": 0,  # 거의 접근 안함
                },
            }
        
        else:
            raise ValueError(
                f"Unknown lifecycle action: {action}. "
                f"Valid actions: evaluate, archive, restore, stats"
            )