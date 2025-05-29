"""Secure migration runner with isolation."""

import asyncio
import logging
import os
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
import resource

from .access_controller import MigrationAccessController, Session, Permission
from .execution_monitor import ExecutionMonitor, Alert, AlertLevel

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """실행 오류."""
    pass


@dataclass
class MigrationTask:
    """마이그레이션 작업."""
    task_id: str
    name: str
    source_pattern: str
    target_path: str
    options: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    items_processed: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class ExecutionEnvironment:
    """실행 환경 설정."""
    working_dir: str
    temp_dir: str
    max_memory_mb: int = 2048
    max_cpu_seconds: int = 3600
    max_file_size_mb: int = 1024
    allowed_paths: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)


class SecureMigrationRunner:
    """보안 마이그레이션 실행기."""
    
    def __init__(
        self,
        access_controller: MigrationAccessController,
        monitor: ExecutionMonitor
    ):
        """초기화.
        
        Args:
            access_controller: 접근 제어기
            monitor: 실행 모니터
        """
        self.access_controller = access_controller
        self.monitor = monitor
        self.tasks: Dict[str, MigrationTask] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # 알림 콜백 설정
        self.monitor.add_alert_callback(self._handle_alert)
    
    async def create_task(
        self,
        session_id: str,
        name: str,
        source_pattern: str,
        target_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> MigrationTask:
        """마이그레이션 작업 생성.
        
        Args:
            session_id: 세션 ID
            name: 작업 이름
            source_pattern: 소스 패턴
            target_path: 대상 경로
            options: 추가 옵션
            
        Returns:
            생성된 작업
            
        Raises:
            PermissionError: 권한 없음
        """
        # 권한 확인
        if not self.access_controller.check_permission(session_id, Permission.EXECUTE):
            raise PermissionError("No permission to execute migration")
        
        task_id = f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(self.tasks)}"
        
        task = MigrationTask(
            task_id=task_id,
            name=name,
            source_pattern=source_pattern,
            target_path=target_path,
            options=options or {},
            session_id=session_id
        )
        
        self.tasks[task_id] = task
        logger.info(f"Migration task created: {task_id}")
        
        return task
    
    async def execute_task(
        self,
        task_id: str,
        extractor: Any,  # SecureExtractor
        validator: Any   # DataValidator
    ) -> None:
        """작업 실행.
        
        Args:
            task_id: 작업 ID
            extractor: 데이터 추출기
            validator: 데이터 검증기
            
        Raises:
            ValueError: 잘못된 작업
            PermissionError: 권한 없음
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # 권한 재확인
        if not self.access_controller.check_permission(
            task.session_id,
            Permission.EXECUTE
        ):
            raise PermissionError("Session expired or no permission")
        
        # 중복 실행 방지
        if task_id in self._running_tasks:
            raise ValueError(f"Task {task_id} is already running")
        
        # 비동기 실행
        self._running_tasks[task_id] = asyncio.create_task(
            self._execute_isolated(task, extractor, validator)
        )
    
    async def _execute_isolated(
        self,
        task: MigrationTask,
        extractor: Any,
        validator: Any
    ) -> None:
        """격리된 환경에서 실행.
        
        Args:
            task: 작업
            extractor: 추출기
            validator: 검증기
        """
        # 실행 환경 생성
        env = await self._create_environment(task)
        
        try:
            task.status = "running"
            task.started_at = datetime.now()
            
            # 모니터링 시작
            await self.monitor.start_monitoring()
            
            # 리소스 제한 설정
            self._set_resource_limits(env)
            
            # 추출 작업 생성
            job = await extractor.create_extraction_job(
                task.source_pattern,
                env.temp_dir,
                **task.options
            )
            
            # 데이터 추출 및 검증
            async for extracted_data in extractor.extract_data(job):
                try:
                    # 검증
                    validation_result = validator.validate(
                        extracted_data.path,
                        extracted_data.content,
                        extracted_data.metadata
                    )
                    
                    if validation_result.is_valid:
                        # 대상 경로로 이동
                        await self._move_to_target(
                            extracted_data,
                            env.temp_dir,
                            task.target_path
                        )
                        task.items_processed += 1
                        self.monitor.record_item_processed(True)
                    else:
                        error_msg = f"Validation failed for {extracted_data.path}"
                        task.errors.append(error_msg)
                        self.monitor.record_item_processed(False)
                        logger.error(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error processing {extracted_data.path}: {str(e)}"
                    task.errors.append(error_msg)
                    self.monitor.record_item_processed(False)
                    logger.error(error_msg)
            
            task.status = "completed"
            
        except Exception as e:
            task.status = "failed"
            task.errors.append(str(e))
            logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            task.completed_at = datetime.now()
            
            # 모니터링 중지
            await self.monitor.stop_monitoring()
            
            # 환경 정리
            await self._cleanup_environment(env)
            
            # 실행 중 작업에서 제거
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
            
            logger.info(f"Task {task.task_id} finished with status: {task.status}")
    
    async def _create_environment(self, task: MigrationTask) -> ExecutionEnvironment:
        """실행 환경 생성.
        
        Args:
            task: 작업
            
        Returns:
            실행 환경
        """
        # 임시 디렉토리 생성
        temp_base = tempfile.mkdtemp(prefix=f"migration_{task.task_id}_")
        working_dir = os.path.join(temp_base, "work")
        temp_dir = os.path.join(temp_base, "temp")
        
        os.makedirs(working_dir)
        os.makedirs(temp_dir)
        
        env = ExecutionEnvironment(
            working_dir=working_dir,
            temp_dir=temp_dir,
            max_memory_mb=task.options.get("max_memory_mb", 2048),
            max_cpu_seconds=task.options.get("max_cpu_seconds", 3600),
            allowed_paths=[working_dir, temp_dir, task.target_path]
        )
        
        logger.info(f"Created isolated environment for task {task.task_id}")
        
        return env
    
    async def _cleanup_environment(self, env: ExecutionEnvironment) -> None:
        """실행 환경 정리.
        
        Args:
            env: 실행 환경
        """
        try:
            # 임시 디렉토리 삭제
            base_dir = os.path.dirname(env.working_dir)
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            
            logger.info(f"Cleaned up environment: {base_dir}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup environment: {e}")
    
    def _set_resource_limits(self, env: ExecutionEnvironment) -> None:
        """리소스 제한 설정.
        
        Args:
            env: 실행 환경
        """
        try:
            # 메모리 제한 (바이트)
            memory_limit = env.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # CPU 시간 제한 (초)
            resource.setrlimit(resource.RLIMIT_CPU, (env.max_cpu_seconds, env.max_cpu_seconds))
            
            # 파일 크기 제한 (바이트)
            file_limit = env.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
            
            logger.info(f"Resource limits set: Memory={env.max_memory_mb}MB, CPU={env.max_cpu_seconds}s")
            
        except Exception as e:
            logger.warning(f"Failed to set resource limits: {e}")
    
    async def _move_to_target(
        self,
        extracted_data: Any,
        temp_dir: str,
        target_path: str
    ) -> None:
        """데이터를 대상 경로로 이동.
        
        Args:
            extracted_data: 추출된 데이터
            temp_dir: 임시 디렉토리
            target_path: 대상 경로
        """
        # 안전한 경로 생성
        safe_filename = Path(extracted_data.path).name
        target_file = os.path.join(target_path, safe_filename)
        
        # 대상 디렉토리 생성
        os.makedirs(target_path, exist_ok=True)
        
        # 파일 이동 (임시 디렉토리에서)
        temp_file = os.path.join(temp_dir, f"{safe_filename}.tmp")
        
        # 데이터 저장
        import json
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump({
                "path": extracted_data.path,
                "content": extracted_data.content,
                "metadata": extracted_data.metadata,
                "checksum": extracted_data.checksum
            }, f, indent=2)
        
        # 원자적 이동
        shutil.move(temp_file, target_file)
        
        logger.debug(f"Moved {extracted_data.path} to {target_file}")
    
    def _handle_alert(self, alert: Alert) -> None:
        """알림 처리.
        
        Args:
            alert: 알림
        """
        # 심각한 알림 처리
        if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            logger.error(f"Critical alert: {alert.message}")
            
            # 실행 중인 작업 중지 고려
            if alert.metric_type.value in ["error_rate", "memory_usage"]:
                # TODO: 자동 중지 로직
                pass
    
    async def cancel_task(self, task_id: str) -> None:
        """작업 취소.
        
        Args:
            task_id: 작업 ID
        """
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            self.tasks[task_id].status = "cancelled"
            logger.info(f"Task {task_id} cancelled")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회.
        
        Args:
            task_id: 작업 ID
            
        Returns:
            작업 상태
        """
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "items_processed": task.items_processed,
            "errors_count": len(task.errors),
            "recent_errors": task.errors[-10:],  # 최근 10개
            "monitoring": self.monitor.get_summary() if task.status == "running" else None
        }
    
    def list_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        """작업 목록 조회.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            작업 목록
        """
        # 권한 확인
        if not self.access_controller.check_permission(session_id, Permission.VIEW_LOGS):
            return []
        
        tasks = []
        for task in self.tasks.values():
            # 세션 소유자 또는 관리자만 조회
            session = self.access_controller.validate_session(session_id)
            if session and (task.session_id == session_id or session.role.value == "admin"):
                tasks.append({
                    "task_id": task.task_id,
                    "name": task.name,
                    "status": task.status,
                    "created_by": task.session_id,
                    "items_processed": task.items_processed
                })
        
        return tasks