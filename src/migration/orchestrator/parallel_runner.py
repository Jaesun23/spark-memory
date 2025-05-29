"""병행 마이그레이션 실행기."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class MigrationBatch:
    """마이그레이션 배치."""
    batch_id: str
    source_system: str
    target_system: str
    items: List[Any]
    priority: int = 0
    dependencies: Set[str] = field(default_factory=set)
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_count: int = 0


@dataclass 
class SyncStatus:
    """동기화 상태."""
    last_sync: datetime
    items_synced: int
    items_pending: int
    sync_errors: int
    is_consistent: bool


class ParallelRunner:
    """병행 운영 마이그레이션 실행기."""
    
    def __init__(
        self,
        max_workers: int = 4,
        sync_interval: int = 60,
        consistency_check_interval: int = 300
    ):
        """초기화.
        
        Args:
            max_workers: 최대 워커 수
            sync_interval: 동기화 간격 (초)
            consistency_check_interval: 일관성 검사 간격 (초)
        """
        self.max_workers = max_workers
        self.sync_interval = sync_interval
        self.consistency_check_interval = consistency_check_interval
        
        self.batches: Dict[str, MigrationBatch] = {}
        self.active_batches: Set[str] = set()
        self.completed_batches: Set[str] = set()
        
        self.sync_status: Dict[str, SyncStatus] = {}
        self._sync_task: Optional[asyncio.Task] = None
        self._consistency_task: Optional[asyncio.Task] = None
        
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    
    def create_batch(
        self,
        batch_id: str,
        source: str,
        target: str,
        items: List[Any],
        priority: int = 0,
        dependencies: Optional[Set[str]] = None
    ) -> MigrationBatch:
        """배치 생성.
        
        Args:
            batch_id: 배치 ID
            source: 소스 시스템
            target: 대상 시스템
            items: 마이그레이션 항목
            priority: 우선순위
            dependencies: 의존성 배치 ID
            
        Returns:
            생성된 배치
        """
        batch = MigrationBatch(
            batch_id=batch_id,
            source_system=source,
            target_system=target,
            items=items,
            priority=priority,
            dependencies=dependencies or set()
        )
        
        self.batches[batch_id] = batch
        logger.info(f"Created batch {batch_id} with {len(items)} items")
        
        return batch
    
    async def start_parallel_migration(self) -> None:
        """병행 마이그레이션 시작."""
        logger.info("Starting parallel migration")
        
        # 동기화 작업 시작
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        # 일관성 검사 작업 시작
        self._consistency_task = asyncio.create_task(self._consistency_check_loop())
        
        # 배치 실행
        await self._execute_batches()
    
    async def stop_parallel_migration(self) -> None:
        """병행 마이그레이션 중지."""
        logger.info("Stopping parallel migration")
        
        # 작업 취소
        if self._sync_task:
            self._sync_task.cancel()
        if self._consistency_task:
            self._consistency_task.cancel()
        
        # 활성 배치 대기
        await self._wait_for_active_batches()
        
        # 스레드 풀 종료
        self.thread_pool.shutdown(wait=True)
    
    async def _execute_batches(self) -> None:
        """배치 실행."""
        while self.batches:
            # 실행 가능한 배치 찾기
            ready_batches = self._find_ready_batches()
            
            if not ready_batches:
                await asyncio.sleep(1)
                continue
            
            # 우선순위별 정렬
            ready_batches.sort(key=lambda b: b.priority, reverse=True)
            
            # 병렬 실행
            tasks = []
            for batch in ready_batches[:self.max_workers - len(self.active_batches)]:
                task = asyncio.create_task(self._execute_batch(batch))
                tasks.append(task)
                self.active_batches.add(batch.batch_id)
            
            # 완료 대기
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_batch(self, batch: MigrationBatch) -> None:
        """단일 배치 실행.
        
        Args:
            batch: 실행할 배치
        """
        batch.status = "running"
        batch.start_time = datetime.now()
        
        try:
            # 실제 마이그레이션 수행
            processed = 0
            failed = 0
            
            for item in batch.items:
                try:
                    # 스레드 풀에서 실행 (I/O 바운드 작업 가정)
                    await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        self._migrate_item,
                        item,
                        batch.source_system,
                        batch.target_system
                    )
                    processed += 1
                except Exception as e:
                    failed += 1
                    batch.error_count += 1
                    logger.error(f"Failed to migrate item in batch {batch.batch_id}: {e}")
            
            batch.status = "completed" if failed == 0 else "completed_with_errors"
            
        except Exception as e:
            batch.status = "failed"
            logger.error(f"Batch {batch.batch_id} failed: {e}")
        
        finally:
            batch.end_time = datetime.now()
            if batch.batch_id in self.active_batches:
                self.active_batches.remove(batch.batch_id)
            self.completed_batches.add(batch.batch_id)
            if batch.batch_id in self.batches:
                del self.batches[batch.batch_id]
    
    def _migrate_item(self, item: Any, source: str, target: str) -> None:
        """단일 항목 마이그레이션.
        
        Args:
            item: 마이그레이션 항목
            source: 소스 시스템
            target: 대상 시스템
        """
        # 실제 마이그레이션 로직
        # 여기서는 시뮬레이션
        import time
        time.sleep(0.01)  # I/O 작업 시뮬레이션
    
    def _find_ready_batches(self) -> List[MigrationBatch]:
        """실행 가능한 배치 찾기.
        
        Returns:
            실행 가능한 배치 목록
        """
        ready = []
        
        for batch in self.batches.values():
            if batch.status != "pending":
                continue
            
            # 의존성 확인
            if batch.dependencies:
                if not all(dep in self.completed_batches for dep in batch.dependencies):
                    continue
            
            # 워커 수 제한 확인
            if len(self.active_batches) >= self.max_workers:
                break
            
            ready.append(batch)
        
        return ready
    
    async def _sync_loop(self) -> None:
        """동기화 루프."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._perform_sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error: {e}")
    
    async def _perform_sync(self) -> None:
        """동기화 수행."""
        for system_pair in self._get_system_pairs():
            source, target = system_pair
            
            try:
                # 동기화 로직 (실제 구현 필요)
                sync_result = await self._sync_systems(source, target)
                
                self.sync_status[f"{source}-{target}"] = SyncStatus(
                    last_sync=datetime.now(),
                    items_synced=sync_result.get("synced", 0),
                    items_pending=sync_result.get("pending", 0),
                    sync_errors=sync_result.get("errors", 0),
                    is_consistent=sync_result.get("consistent", True)
                )
                
            except Exception as e:
                logger.error(f"Sync failed for {source}-{target}: {e}")
    
    async def _sync_systems(self, source: str, target: str) -> Dict[str, Any]:
        """시스템 간 동기화.
        
        Args:
            source: 소스 시스템
            target: 대상 시스템
            
        Returns:
            동기화 결과
        """
        # 실제 동기화 로직 구현 필요
        return {
            "synced": 100,
            "pending": 10,
            "errors": 0,
            "consistent": True
        }
    
    async def _consistency_check_loop(self) -> None:
        """일관성 검사 루프."""
        while True:
            try:
                await asyncio.sleep(self.consistency_check_interval)
                await self._check_consistency()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consistency check error: {e}")
    
    async def _check_consistency(self) -> None:
        """데이터 일관성 검사."""
        for system_pair in self._get_system_pairs():
            source, target = system_pair
            
            try:
                # 일관성 검사 로직 (실제 구현 필요)
                is_consistent = await self._verify_consistency(source, target)
                
                if not is_consistent:
                    logger.warning(f"Inconsistency detected between {source} and {target}")
                    # 알림 또는 자동 복구 로직
                
            except Exception as e:
                logger.error(f"Consistency check failed for {source}-{target}: {e}")
    
    async def _verify_consistency(self, source: str, target: str) -> bool:
        """일관성 검증.
        
        Args:
            source: 소스 시스템
            target: 대상 시스템
            
        Returns:
            일관성 여부
        """
        # 실제 검증 로직 구현 필요
        return True
    
    def _get_system_pairs(self) -> Set[tuple]:
        """시스템 쌍 목록 생성.
        
        Returns:
            (소스, 대상) 튜플 집합
        """
        pairs = set()
        
        for batch in list(self.batches.values()) + list(self.completed_batches):
            if isinstance(batch, str):
                continue
            pairs.add((batch.source_system, batch.target_system))
        
        return pairs
    
    async def _wait_for_active_batches(self) -> None:
        """활성 배치 완료 대기."""
        while self.active_batches:
            await asyncio.sleep(1)
    
    def get_migration_status(self) -> Dict[str, Any]:
        """마이그레이션 상태 조회.
        
        Returns:
            상태 정보
        """
        return {
            "total_batches": len(self.batches) + len(self.completed_batches),
            "active_batches": len(self.active_batches),
            "completed_batches": len(self.completed_batches),
            "pending_batches": len([b for b in self.batches.values() if b.status == "pending"]),
            "sync_status": {
                pair: {
                    "last_sync": status.last_sync.isoformat(),
                    "items_synced": status.items_synced,
                    "items_pending": status.items_pending,
                    "is_consistent": status.is_consistent
                }
                for pair, status in self.sync_status.items()
            }
        }
    
    def get_batch_details(self, batch_id: str) -> Optional[MigrationBatch]:
        """배치 상세 정보 조회.
        
        Args:
            batch_id: 배치 ID
            
        Returns:
            배치 정보
        """
        return self.batches.get(batch_id)