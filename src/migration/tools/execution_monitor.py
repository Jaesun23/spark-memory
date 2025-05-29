"""Execution monitoring for migration tools."""

import asyncio
import logging
import psutil
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """경고 레벨."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """측정 지표 타입."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    ITEMS_PROCESSED = "items_processed"
    ERROR_RATE = "error_rate"
    EXECUTION_TIME = "execution_time"


@dataclass
class Alert:
    """경고 알림."""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    message: str
    timestamp: datetime
    value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """성능 지표."""
    type: MetricType
    value: float
    timestamp: datetime
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """모니터링 설정."""
    cpu_threshold: float = 80.0              # CPU 사용률 임계값 (%)
    memory_threshold: float = 80.0           # 메모리 사용률 임계값 (%)
    disk_io_threshold: float = 100.0         # 디스크 I/O 임계값 (MB/s)
    network_io_threshold: float = 100.0      # 네트워크 I/O 임계값 (MB/s)
    error_rate_threshold: float = 5.0        # 오류율 임계값 (%)
    check_interval: int = 5                  # 체크 간격 (초)
    alert_cooldown: int = 300                # 알림 쿨다운 (초)


class ExecutionMonitor:
    """마이그레이션 실행 모니터링."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """초기화.
        
        Args:
            config: 모니터링 설정
        """
        self.config = config or MonitoringConfig()
        self.metrics: List[Metric] = []
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_alerts: Dict[str, datetime] = {}
        
        # 성능 카운터
        self.items_processed = 0
        self.errors_count = 0
        self.start_time: Optional[datetime] = None
        
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """알림 콜백 추가.
        
        Args:
            callback: 알림 콜백 함수
        """
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """모니터링 시작."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self.start_time = datetime.now()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Execution monitoring started")
    
    async def stop_monitoring(self) -> None:
        """모니터링 중지."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Execution monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """모니터링 루프."""
        while self._monitoring:
            try:
                # 시스템 메트릭 수집
                await self._collect_system_metrics()
                
                # 애플리케이션 메트릭 수집
                await self._collect_app_metrics()
                
                # 임계값 체크
                self._check_thresholds()
                
                # 대기
                await asyncio.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _collect_system_metrics(self) -> None:
        """시스템 메트릭 수집."""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric(
                MetricType.CPU_USAGE,
                cpu_percent,
                "%"
            )
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            self._record_metric(
                MetricType.MEMORY_USAGE,
                memory.percent,
                "%",
                {"available_gb": memory.available / (1024**3)}
            )
            
            # 디스크 I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # MB/s 계산 (간단한 예시)
                disk_speed = (disk_io.read_bytes + disk_io.write_bytes) / (1024**2) / self.config.check_interval
                self._record_metric(
                    MetricType.DISK_IO,
                    disk_speed,
                    "MB/s"
                )
            
            # 네트워크 I/O
            net_io = psutil.net_io_counters()
            if net_io:
                # MB/s 계산
                net_speed = (net_io.bytes_sent + net_io.bytes_recv) / (1024**2) / self.config.check_interval
                self._record_metric(
                    MetricType.NETWORK_IO,
                    net_speed,
                    "MB/s"
                )
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _collect_app_metrics(self) -> None:
        """애플리케이션 메트릭 수집."""
        if not self.start_time:
            return
        
        # 처리 항목 수
        self._record_metric(
            MetricType.ITEMS_PROCESSED,
            self.items_processed,
            "items"
        )
        
        # 오류율
        if self.items_processed > 0:
            error_rate = (self.errors_count / self.items_processed) * 100
            self._record_metric(
                MetricType.ERROR_RATE,
                error_rate,
                "%"
            )
        
        # 실행 시간
        execution_time = (datetime.now() - self.start_time).total_seconds()
        self._record_metric(
            MetricType.EXECUTION_TIME,
            execution_time,
            "seconds"
        )
    
    def _record_metric(
        self,
        metric_type: MetricType,
        value: float,
        unit: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """메트릭 기록.
        
        Args:
            metric_type: 메트릭 타입
            value: 값
            unit: 단위
            metadata: 추가 메타데이터
        """
        metric = Metric(
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # 메모리 관리 (최근 1000개만 유지)
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def _check_thresholds(self) -> None:
        """임계값 체크 및 알림."""
        # 최근 메트릭 가져오기
        recent_metrics = {}
        for metric in reversed(self.metrics):
            if metric.type not in recent_metrics:
                recent_metrics[metric.type] = metric
        
        # CPU 임계값
        if MetricType.CPU_USAGE in recent_metrics:
            cpu = recent_metrics[MetricType.CPU_USAGE]
            if cpu.value > self.config.cpu_threshold:
                self._create_alert(
                    AlertLevel.WARNING,
                    MetricType.CPU_USAGE,
                    f"CPU usage high: {cpu.value:.1f}%",
                    cpu.value,
                    self.config.cpu_threshold
                )
        
        # 메모리 임계값
        if MetricType.MEMORY_USAGE in recent_metrics:
            memory = recent_metrics[MetricType.MEMORY_USAGE]
            if memory.value > self.config.memory_threshold:
                self._create_alert(
                    AlertLevel.WARNING,
                    MetricType.MEMORY_USAGE,
                    f"Memory usage high: {memory.value:.1f}%",
                    memory.value,
                    self.config.memory_threshold
                )
        
        # 오류율 임계값
        if MetricType.ERROR_RATE in recent_metrics:
            error_rate = recent_metrics[MetricType.ERROR_RATE]
            if error_rate.value > self.config.error_rate_threshold:
                self._create_alert(
                    AlertLevel.ERROR,
                    MetricType.ERROR_RATE,
                    f"Error rate high: {error_rate.value:.1f}%",
                    error_rate.value,
                    self.config.error_rate_threshold
                )
    
    def _create_alert(
        self,
        level: AlertLevel,
        metric_type: MetricType,
        message: str,
        value: float,
        threshold: float
    ) -> None:
        """알림 생성.
        
        Args:
            level: 경고 레벨
            metric_type: 메트릭 타입
            message: 메시지
            value: 현재 값
            threshold: 임계값
        """
        # 쿨다운 체크
        alert_key = f"{metric_type.value}:{level.value}"
        if alert_key in self._last_alerts:
            last_alert_time = self._last_alerts[alert_key]
            if (datetime.now() - last_alert_time).total_seconds() < self.config.alert_cooldown:
                return
        
        alert = Alert(
            alert_id=f"alert-{datetime.now().timestamp()}",
            level=level,
            metric_type=metric_type,
            message=message,
            timestamp=datetime.now(),
            value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        self._last_alerts[alert_key] = datetime.now()
        
        # 콜백 호출
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"Alert: {message}")
    
    def record_item_processed(self, success: bool = True) -> None:
        """처리 항목 기록.
        
        Args:
            success: 성공 여부
        """
        self.items_processed += 1
        if not success:
            self.errors_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """모니터링 요약.
        
        Returns:
            요약 정보
        """
        if not self.start_time:
            return {"status": "not_started"}
        
        execution_time = (datetime.now() - self.start_time).total_seconds()
        
        # 최근 메트릭
        recent_metrics = {}
        for metric in reversed(self.metrics[-100:]):  # 최근 100개
            if metric.type not in recent_metrics:
                recent_metrics[metric.type] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat()
                }
        
        return {
            "status": "monitoring" if self._monitoring else "stopped",
            "start_time": self.start_time.isoformat(),
            "execution_time_seconds": execution_time,
            "items_processed": self.items_processed,
            "errors_count": self.errors_count,
            "error_rate": (self.errors_count / self.items_processed * 100) if self.items_processed > 0 else 0,
            "recent_metrics": recent_metrics,
            "alerts_count": len(self.alerts),
            "recent_alerts": [
                {
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alerts[-10:]  # 최근 10개
            ]
        }
    
    async def export_metrics(self, filepath: str) -> None:
        """메트릭 내보내기.
        
        Args:
            filepath: 저장할 파일 경로
        """
        export_data = {
            "summary": self.get_summary(),
            "metrics": [
                {
                    "type": metric.type.value,
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "metadata": metric.metadata
                }
                for metric in self.metrics
            ],
            "alerts": [
                {
                    "id": alert.alert_id,
                    "level": alert.level.value,
                    "metric_type": alert.metric_type.value,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alerts
            ]
        }
        
        import aiofiles
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(export_data, indent=2))
        
        logger.info(f"Metrics exported to {filepath}")
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """이상 패턴 감지.
        
        Returns:
            감지된 이상 패턴
        """
        anomalies = []
        
        # CPU 스파이크 감지
        cpu_metrics = [m for m in self.metrics if m.type == MetricType.CPU_USAGE]
        if len(cpu_metrics) >= 5:
            recent_cpu = [m.value for m in cpu_metrics[-5:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            if avg_cpu > 90:
                anomalies.append({
                    "type": "cpu_spike",
                    "severity": "high",
                    "description": f"Sustained high CPU usage: {avg_cpu:.1f}%"
                })
        
        # 메모리 누수 감지
        memory_metrics = [m for m in self.metrics if m.type == MetricType.MEMORY_USAGE]
        if len(memory_metrics) >= 10:
            # 메모리 사용량이 계속 증가하는지 확인
            recent_memory = [m.value for m in memory_metrics[-10:]]
            if all(recent_memory[i] <= recent_memory[i+1] for i in range(9)):
                anomalies.append({
                    "type": "memory_leak",
                    "severity": "medium",
                    "description": "Possible memory leak detected"
                })
        
        # 갑작스러운 오류율 증가
        error_metrics = [m for m in self.metrics if m.type == MetricType.ERROR_RATE]
        if len(error_metrics) >= 2:
            if error_metrics[-1].value > error_metrics[-2].value * 2:
                anomalies.append({
                    "type": "error_spike",
                    "severity": "high",
                    "description": f"Error rate doubled: {error_metrics[-1].value:.1f}%"
                })
        
        return anomalies