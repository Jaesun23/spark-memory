"""마이그레이션 롤백 관리자."""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RollbackPoint:
    """롤백 포인트."""
    point_id: str
    phase_id: str
    timestamp: datetime
    items_processed: int
    state_snapshot: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True


@dataclass
class RollbackResult:
    """롤백 결과."""
    success: bool
    point_id: str
    items_rolled_back: int
    duration_seconds: float
    error_messages: List[str] = field(default_factory=list)


class RollbackManager:
    """롤백 관리자."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 10,
        auto_cleanup: bool = True
    ):
        """초기화.
        
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            max_checkpoints: 최대 체크포인트 수
            auto_cleanup: 자동 정리 활성화
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        
        self.rollback_points: Dict[str, RollbackPoint] = {}
        self.rollback_history: List[RollbackResult] = []
        
        # 기존 체크포인트 로드
        self._load_existing_checkpoints()
    
    async def create_checkpoint(
        self,
        phase_id: str,
        state_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> RollbackPoint:
        """체크포인트 생성.
        
        Args:
            phase_id: 단계 ID
            state_data: 상태 데이터
            metadata: 메타데이터
            
        Returns:
            생성된 롤백 포인트
        """
        point_id = f"checkpoint_{phase_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        rollback_point = RollbackPoint(
            point_id=point_id,
            phase_id=phase_id,
            timestamp=datetime.now(),
            items_processed=state_data.get("items_processed", 0),
            state_snapshot=state_data,
            metadata=metadata or {}
        )
        
        # 메모리에 저장
        self.rollback_points[point_id] = rollback_point
        
        # 파일로 저장
        await self._save_checkpoint(rollback_point)
        
        # 자동 정리
        if self.auto_cleanup:
            await self._cleanup_old_checkpoints()
        
        logger.info(f"Created rollback point: {point_id}")
        
        return rollback_point
    
    async def rollback_to_point(
        self,
        point_id: str,
        rollback_handler: Optional[Any] = None
    ) -> RollbackResult:
        """특정 포인트로 롤백.
        
        Args:
            point_id: 롤백 포인트 ID
            rollback_handler: 롤백 처리기
            
        Returns:
            롤백 결과
        """
        if point_id not in self.rollback_points:
            return RollbackResult(
                success=False,
                point_id=point_id,
                items_rolled_back=0,
                duration_seconds=0,
                error_messages=[f"Rollback point {point_id} not found"]
            )
        
        rollback_point = self.rollback_points[point_id]
        
        if not rollback_point.is_valid:
            return RollbackResult(
                success=False,
                point_id=point_id,
                items_rolled_back=0,
                duration_seconds=0,
                error_messages=[f"Rollback point {point_id} is invalid"]
            )
        
        start_time = datetime.now()
        errors = []
        items_rolled_back = 0
        
        try:
            # 롤백 핸들러가 제공된 경우 사용
            if rollback_handler:
                result = await rollback_handler.rollback(rollback_point.state_snapshot)
                items_rolled_back = result.get("items_rolled_back", 0)
            else:
                # 기본 롤백 로직
                items_rolled_back = await self._default_rollback(rollback_point)
            
            # 이후 체크포인트 무효화
            await self._invalidate_later_checkpoints(rollback_point.timestamp)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = RollbackResult(
                success=True,
                point_id=point_id,
                items_rolled_back=items_rolled_back,
                duration_seconds=duration,
                error_messages=errors
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            errors.append(str(e))
            
            result = RollbackResult(
                success=False,
                point_id=point_id,
                items_rolled_back=items_rolled_back,
                duration_seconds=duration,
                error_messages=errors
            )
        
        # 롤백 이력 저장
        self.rollback_history.append(result)
        
        return result
    
    async def _default_rollback(self, rollback_point: RollbackPoint) -> int:
        """기본 롤백 수행.
        
        Args:
            rollback_point: 롤백 포인트
            
        Returns:
            롤백된 항목 수
        """
        # 기본 구현 - 실제로는 데이터 복원 로직 필요
        logger.info(f"Performing default rollback to {rollback_point.point_id}")
        
        # 상태 복원 시뮬레이션
        items_to_rollback = rollback_point.items_processed
        
        # 실제로는 여기서 데이터베이스 트랜잭션, 파일 복원 등 수행
        await asyncio.sleep(0.1)  # 작업 시뮬레이션
        
        return items_to_rollback
    
    async def _save_checkpoint(self, rollback_point: RollbackPoint) -> None:
        """체크포인트 파일 저장.
        
        Args:
            rollback_point: 롤백 포인트
        """
        checkpoint_file = self.checkpoint_dir / f"{rollback_point.point_id}.json"
        
        checkpoint_data = {
            "point_id": rollback_point.point_id,
            "phase_id": rollback_point.phase_id,
            "timestamp": rollback_point.timestamp.isoformat(),
            "items_processed": rollback_point.items_processed,
            "state_snapshot": rollback_point.state_snapshot,
            "metadata": rollback_point.metadata,
            "is_valid": rollback_point.is_valid
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _load_existing_checkpoints(self) -> None:
        """기존 체크포인트 로드."""
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                rollback_point = RollbackPoint(
                    point_id=data["point_id"],
                    phase_id=data["phase_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    items_processed=data["items_processed"],
                    state_snapshot=data["state_snapshot"],
                    metadata=data.get("metadata", {}),
                    is_valid=data.get("is_valid", True)
                )
                
                self.rollback_points[rollback_point.point_id] = rollback_point
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
    
    async def _cleanup_old_checkpoints(self) -> None:
        """오래된 체크포인트 정리."""
        if len(self.rollback_points) <= self.max_checkpoints:
            return
        
        # 시간순 정렬
        sorted_points = sorted(
            self.rollback_points.values(),
            key=lambda p: p.timestamp
        )
        
        # 오래된 체크포인트 제거
        to_remove = len(sorted_points) - self.max_checkpoints
        for point in sorted_points[:to_remove]:
            # 파일 삭제
            checkpoint_file = self.checkpoint_dir / f"{point.point_id}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            # 메모리에서 제거
            del self.rollback_points[point.point_id]
            
            logger.info(f"Removed old checkpoint: {point.point_id}")
    
    async def _invalidate_later_checkpoints(self, timestamp: datetime) -> None:
        """특정 시점 이후 체크포인트 무효화.
        
        Args:
            timestamp: 기준 시간
        """
        for point in self.rollback_points.values():
            if point.timestamp > timestamp:
                point.is_valid = False
                await self._save_checkpoint(point)
                logger.info(f"Invalidated checkpoint: {point.point_id}")
    
    def get_available_rollback_points(
        self,
        phase_id: Optional[str] = None
    ) -> List[RollbackPoint]:
        """사용 가능한 롤백 포인트 조회.
        
        Args:
            phase_id: 단계 ID (선택)
            
        Returns:
            롤백 포인트 목록
        """
        points = list(self.rollback_points.values())
        
        # 단계별 필터링
        if phase_id:
            points = [p for p in points if p.phase_id == phase_id]
        
        # 유효한 포인트만
        points = [p for p in points if p.is_valid]
        
        # 시간순 정렬
        points.sort(key=lambda p: p.timestamp, reverse=True)
        
        return points
    
    def get_rollback_history(self) -> List[RollbackResult]:
        """롤백 이력 조회.
        
        Returns:
            롤백 결과 목록
        """
        return self.rollback_history.copy()
    
    async def validate_checkpoint(self, point_id: str) -> bool:
        """체크포인트 유효성 검증.
        
        Args:
            point_id: 체크포인트 ID
            
        Returns:
            유효성 여부
        """
        if point_id not in self.rollback_points:
            return False
        
        point = self.rollback_points[point_id]
        
        # 기본 유효성 검사
        if not point.is_valid:
            return False
        
        # 체크포인트 파일 존재 확인
        checkpoint_file = self.checkpoint_dir / f"{point_id}.json"
        if not checkpoint_file.exists():
            return False
        
        # 추가 검증 로직 (필요시)
        # ...
        
        return True