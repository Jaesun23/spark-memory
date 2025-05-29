"""단계별 마이그레이션 관리자."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..security.data_classifier import DataClassifier, DataSensitivity
from ..extractors.secure_extractor import SecureExtractor
from ..tools.access_controller import MigrationAccessController, Permission
from ..tools.execution_monitor import ExecutionMonitor

logger = logging.getLogger(__name__)


class PhaseStatus(Enum):
    """마이그레이션 단계 상태."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationStrategy(Enum):
    """마이그레이션 전략."""
    PILOT = "pilot"              # 파일럿 테스트
    INCREMENTAL = "incremental"  # 점진적 확대
    PARALLEL = "parallel"        # 병행 운영
    CUTOVER = "cutover"          # 전체 전환


@dataclass
class MigrationPhase:
    """마이그레이션 단계."""
    phase_id: str
    name: str
    strategy: MigrationStrategy
    data_patterns: List[str]
    user_groups: List[str]
    sensitivity_levels: List[DataSensitivity]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: PhaseStatus = PhaseStatus.PENDING
    progress: float = 0.0
    error_count: int = 0
    rollback_point: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseResult:
    """단계 실행 결과."""
    phase_id: str
    status: PhaseStatus
    items_processed: int
    items_failed: int
    duration_seconds: float
    rollback_available: bool
    error_messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class PhasedMigrator:
    """단계별 마이그레이션 실행기."""
    
    def __init__(
        self,
        data_classifier: DataClassifier,
        extractor: SecureExtractor,
        access_controller: MigrationAccessController,
        monitor: ExecutionMonitor,
        max_parallel_phases: int = 1,
        rollback_enabled: bool = True
    ):
        """초기화.
        
        Args:
            data_classifier: 데이터 분류기
            extractor: 보안 추출기
            access_controller: 접근 제어기
            monitor: 실행 모니터
            max_parallel_phases: 최대 병렬 단계 수
            rollback_enabled: 롤백 활성화 여부
        """
        self.classifier = data_classifier
        self.extractor = extractor
        self.access_controller = access_controller
        self.monitor = monitor
        self.max_parallel_phases = max_parallel_phases
        self.rollback_enabled = rollback_enabled
        
        self.phases: List[MigrationPhase] = []
        self.active_phases: Dict[str, asyncio.Task] = {}
        self.completed_phases: List[str] = []
        self.phase_callbacks: Dict[str, List[Callable]] = {}
    
    def create_pilot_phase(
        self,
        name: str,
        test_patterns: List[str],
        test_users: List[str]
    ) -> MigrationPhase:
        """파일럿 단계 생성.
        
        Args:
            name: 단계 이름
            test_patterns: 테스트 데이터 패턴
            test_users: 테스트 사용자 그룹
            
        Returns:
            생성된 파일럿 단계
        """
        phase = MigrationPhase(
            phase_id=f"pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            strategy=MigrationStrategy.PILOT,
            data_patterns=test_patterns,
            user_groups=test_users,
            sensitivity_levels=[DataSensitivity.PUBLIC, DataSensitivity.INTERNAL],
            metadata={
                "test_mode": True,
                "max_items": 1000,
                "validation_required": True
            }
        )
        
        self.phases.append(phase)
        logger.info(f"Created pilot phase: {phase.phase_id}")
        
        return phase
    
    def plan_incremental_phases(
        self,
        sensitivity_order: bool = True,
        batch_size: int = 10000
    ) -> List[MigrationPhase]:
        """점진적 마이그레이션 단계 계획.
        
        Args:
            sensitivity_order: 민감도 순서대로 진행
            batch_size: 배치 크기
            
        Returns:
            계획된 단계 목록
        """
        phases = []
        
        # 민감도별 단계 생성
        if sensitivity_order:
            sensitivities = [
                DataSensitivity.PUBLIC,
                DataSensitivity.INTERNAL,
                DataSensitivity.CONFIDENTIAL,
                DataSensitivity.SECRET
            ]
            
            for i, sensitivity in enumerate(sensitivities):
                phase = MigrationPhase(
                    phase_id=f"incremental_{i+1}_{sensitivity.value}",
                    name=f"Incremental Migration - {sensitivity.value}",
                    strategy=MigrationStrategy.INCREMENTAL,
                    data_patterns=["*"],
                    user_groups=["all"],
                    sensitivity_levels=[sensitivity],
                    metadata={
                        "batch_size": batch_size,
                        "priority": i + 1
                    }
                )
                phases.append(phase)
        
        self.phases.extend(phases)
        logger.info(f"Planned {len(phases)} incremental phases")
        
        return phases
    
    async def execute_phase(
        self,
        phase: MigrationPhase,
        session_id: str,
        dry_run: bool = False
    ) -> PhaseResult:
        """단계 실행.
        
        Args:
            phase: 실행할 단계
            session_id: 세션 ID
            dry_run: 테스트 실행 여부
            
        Returns:
            실행 결과
        """
        # 권한 확인
        if not self.access_controller.check_permission(
            session_id, Permission.EXECUTE
        ):
            raise PermissionError("No permission to execute migration")
        
        # 단계 시작
        phase.status = PhaseStatus.RUNNING
        phase.start_time = datetime.now()
        
        # 모니터링 시작
        await self.monitor.start_monitoring()
        
        try:
            # 단계별 실행
            if phase.strategy == MigrationStrategy.PILOT:
                result = await self._execute_pilot(phase, dry_run)
            elif phase.strategy == MigrationStrategy.INCREMENTAL:
                result = await self._execute_incremental(phase, dry_run)
            elif phase.strategy == MigrationStrategy.PARALLEL:
                result = await self._execute_parallel(phase, dry_run)
            else:
                result = await self._execute_cutover(phase, dry_run)
            
            # 결과에 따른 상태 업데이트
            phase.status = result.status
            phase.end_time = datetime.now()
            if result.status == PhaseStatus.COMPLETED:
                self.completed_phases.append(phase.phase_id)
            
        except Exception as e:
            # 실패 처리
            phase.status = PhaseStatus.FAILED
            phase.end_time = datetime.now()
            phase.error_count += 1
            
            result = PhaseResult(
                phase_id=phase.phase_id,
                status=PhaseStatus.FAILED,
                items_processed=0,
                items_failed=0,
                duration_seconds=0,
                rollback_available=self.rollback_enabled,
                error_messages=[str(e)]
            )
            
            logger.error(f"Phase {phase.phase_id} failed: {e}")
        
        finally:
            # 모니터링 중지
            await self.monitor.stop_monitoring()
        
        # 콜백 실행
        await self._execute_callbacks(phase.phase_id, result)
        
        return result
    
    async def _execute_pilot(
        self,
        phase: MigrationPhase,
        dry_run: bool
    ) -> PhaseResult:
        """파일럿 실행.
        
        Args:
            phase: 파일럿 단계
            dry_run: 테스트 실행 여부
            
        Returns:
            실행 결과
        """
        logger.info(f"Executing pilot phase: {phase.phase_id}")
        
        items_processed = 0
        items_failed = 0
        errors = []
        
        # 테스트 데이터 추출
        for pattern in phase.data_patterns:
            try:
                # 데이터 분류
                classified_data = await self.classifier.classify_pattern(pattern)
                
                # 민감도 필터링
                filtered_data = [
                    item for item in classified_data
                    if item.sensitivity in phase.sensitivity_levels
                ]
                
                # 제한된 수만 처리
                max_items = phase.metadata.get("max_items", 1000)
                for item in filtered_data[:max_items]:
                    try:
                        if not dry_run:
                            # 실제 추출 수행
                            # TODO: SecureExtractor의 실제 메소드로 교체 필요
                            pass  # await self.extractor.extract_item(item)
                        items_processed += 1
                        phase.progress = items_processed / max_items * 100
                    except Exception as e:
                        items_failed += 1
                        errors.append(f"Failed to process {item}: {e}")
                
            except Exception as e:
                errors.append(f"Pattern processing failed: {e}")
        
        duration = (datetime.now() - phase.start_time).total_seconds()
        
        return PhaseResult(
            phase_id=phase.phase_id,
            status=PhaseStatus.COMPLETED if items_failed == 0 and len(errors) == 0 else PhaseStatus.FAILED,
            items_processed=items_processed,
            items_failed=items_failed,
            duration_seconds=duration,
            rollback_available=True,
            error_messages=errors,
            metrics={
                "success_rate": (items_processed - items_failed) / items_processed * 100
                if items_processed > 0 else 0
            }
        )
    
    async def _execute_incremental(
        self,
        phase: MigrationPhase,
        dry_run: bool
    ) -> PhaseResult:
        """점진적 실행.
        
        Args:
            phase: 점진적 단계
            dry_run: 테스트 실행 여부
            
        Returns:
            실행 결과
        """
        logger.info(f"Executing incremental phase: {phase.phase_id}")
        
        batch_size = phase.metadata.get("batch_size", 10000)
        items_processed = 0
        items_failed = 0
        errors = []
        
        # 민감도별 데이터 처리
        for sensitivity in phase.sensitivity_levels:
            try:
                # 해당 민감도 데이터 추출
                data_items = await self.classifier.find_by_sensitivity(sensitivity)
                
                # 배치 처리
                for i in range(0, len(data_items), batch_size):
                    batch = data_items[i:i + batch_size]
                    
                    for item in batch:
                        try:
                            if not dry_run:
                                # TODO: SecureExtractor의 실제 메소드로 교체 필요
                                pass  # await self.extractor.extract_item(item)
                            items_processed += 1
                        except Exception as e:
                            items_failed += 1
                            errors.append(str(e))
                    
                    # 진행률 업데이트
                    phase.progress = (i + len(batch)) / len(data_items) * 100
                    
                    # 배치 간 대기
                    await asyncio.sleep(1)
                    
            except Exception as e:
                errors.append(f"Sensitivity {sensitivity} processing failed: {e}")
        
        duration = (datetime.now() - phase.start_time).total_seconds()
        
        return PhaseResult(
            phase_id=phase.phase_id,
            status=PhaseStatus.COMPLETED if items_failed < items_processed * 0.05 
                   else PhaseStatus.FAILED,
            items_processed=items_processed,
            items_failed=items_failed,
            duration_seconds=duration,
            rollback_available=True,
            error_messages=errors
        )
    
    async def _execute_parallel(
        self,
        phase: MigrationPhase,
        dry_run: bool
    ) -> PhaseResult:
        """병행 실행.
        
        Args:
            phase: 병행 단계
            dry_run: 테스트 실행 여부
            
        Returns:
            실행 결과
        """
        # 병행 실행은 별도 구현 필요
        logger.info(f"Executing parallel phase: {phase.phase_id}")
        
        # 간단한 구현
        return PhaseResult(
            phase_id=phase.phase_id,
            status=PhaseStatus.COMPLETED,
            items_processed=0,
            items_failed=0,
            duration_seconds=0,
            rollback_available=True
        )
    
    async def _execute_cutover(
        self,
        phase: MigrationPhase,
        dry_run: bool
    ) -> PhaseResult:
        """전체 전환 실행.
        
        Args:
            phase: 전환 단계
            dry_run: 테스트 실행 여부
            
        Returns:
            실행 결과
        """
        # 전체 전환은 별도 구현 필요
        logger.info(f"Executing cutover phase: {phase.phase_id}")
        
        return PhaseResult(
            phase_id=phase.phase_id,
            status=PhaseStatus.COMPLETED,
            items_processed=0,
            items_failed=0,
            duration_seconds=0,
            rollback_available=False
        )
    
    def add_phase_callback(
        self,
        phase_id: str,
        callback: Callable[[PhaseResult], None]
    ) -> None:
        """단계 콜백 추가.
        
        Args:
            phase_id: 단계 ID
            callback: 콜백 함수
        """
        if phase_id not in self.phase_callbacks:
            self.phase_callbacks[phase_id] = []
        self.phase_callbacks[phase_id].append(callback)
    
    async def _execute_callbacks(
        self,
        phase_id: str,
        result: PhaseResult
    ) -> None:
        """콜백 실행.
        
        Args:
            phase_id: 단계 ID
            result: 실행 결과
        """
        if phase_id in self.phase_callbacks:
            for callback in self.phase_callbacks[phase_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def get_phase_status(self, phase_id: str) -> Optional[MigrationPhase]:
        """단계 상태 조회.
        
        Args:
            phase_id: 단계 ID
            
        Returns:
            단계 정보
        """
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        return None
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """전체 진행 상황 요약.
        
        Returns:
            진행 상황 요약
        """
        total_phases = len(self.phases)
        completed = len(self.completed_phases)
        
        return {
            "total_phases": total_phases,
            "completed_phases": completed,
            "active_phases": list(self.active_phases.keys()),
            "progress_percentage": completed / total_phases * 100 if total_phases > 0 else 0,
            "phase_details": [
                {
                    "phase_id": phase.phase_id,
                    "name": phase.name,
                    "status": phase.status.value,
                    "progress": phase.progress
                }
                for phase in self.phases
            ]
        }