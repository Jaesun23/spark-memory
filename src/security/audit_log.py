"""
Audit logging module for Memory One Spark.

This module provides comprehensive audit logging with anomaly detection.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from ..memory.models import MemoryError


class AuditError(MemoryError):
    """Raised when audit operations fail."""

    pass


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    API_KEY_USED = "api_key_used"

    # Data access
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SEARCH = "search"

    # Administrative
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"

    # Security
    ENCRYPTION_KEY_ROTATED = "encryption_key_rotated"
    ANOMALY_DETECTED = "anomaly_detected"
    ACCESS_DENIED = "access_denied"

    # System
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGED = "config_changed"
    BACKUP_CREATED = "backup_created"
    RESTORE_PERFORMED = "restore_performed"


class AuditEvent(BaseModel):
    """Represents an audit event."""

    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: AuditEventType
    principal_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"  # success, failure, error
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_score: float = 0.0


class AnomalyPattern(BaseModel):
    """Pattern for detecting anomalies."""

    name: str
    description: str
    event_types: Set[AuditEventType]
    threshold: int
    time_window: int  # minutes
    risk_score: float


class AuditLogger:
    """
    Audit logger with anomaly detection.

    Features:
    - Comprehensive event logging
    - Anomaly detection
    - Event correlation
    - Compliance reporting
    """

    def __init__(self, retention_days: int = 90):
        """
        Initialize audit logger.

        Args:
            retention_days: Days to retain audit logs
        """
        self.retention_days = retention_days
        self.events: List[AuditEvent] = []
        self.event_index: Dict[str, List[int]] = defaultdict(list)
        self.anomaly_patterns = self._init_anomaly_patterns()
        self.detected_anomalies: List[Dict] = []

    def _init_anomaly_patterns(self) -> List[AnomalyPattern]:
        """Initialize anomaly detection patterns."""
        return [
            AnomalyPattern(
                name="brute_force_attack",
                description="Multiple failed login attempts",
                event_types={AuditEventType.LOGIN_FAILED},
                threshold=5,
                time_window=5,
                risk_score=0.8,
            ),
            AnomalyPattern(
                name="privilege_escalation",
                description="Unusual permission changes",
                event_types={
                    AuditEventType.PERMISSION_GRANTED,
                    AuditEventType.ROLE_ASSIGNED,
                },
                threshold=3,
                time_window=10,
                risk_score=0.9,
            ),
            AnomalyPattern(
                name="data_exfiltration",
                description="Excessive data access",
                event_types={AuditEventType.READ, AuditEventType.SEARCH},
                threshold=100,
                time_window=5,
                risk_score=0.7,
            ),
            AnomalyPattern(
                name="mass_deletion",
                description="Multiple delete operations",
                event_types={AuditEventType.DELETE},
                threshold=10,
                time_window=5,
                risk_score=0.9,
            ),
            AnomalyPattern(
                name="api_abuse",
                description="Excessive API usage",
                event_types={AuditEventType.API_KEY_USED},
                threshold=1000,
                time_window=1,
                risk_score=0.6,
            ),
        ]

    def log_event(
        self,
        event_type: AuditEventType,
        principal_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        metadata: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            principal_id: ID of principal performing action
            resource: Resource being accessed
            action: Specific action taken
            result: Result of action (success/failure/error)
            metadata: Additional event data
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created audit event
        """
        event = AuditEvent(
            id=f"evt_{datetime.utcnow().timestamp()}_{len(self.events)}",
            event_type=event_type,
            principal_id=principal_id,
            resource=resource,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )

        # Add to events
        event_idx = len(self.events)
        self.events.append(event)

        # Update indices
        if principal_id:
            self.event_index[f"principal:{principal_id}"].append(event_idx)
        if resource:
            self.event_index[f"resource:{resource}"].append(event_idx)
        self.event_index[f"type:{event_type}"].append(event_idx)

        # Check for anomalies
        self._check_anomalies(event)

        # Clean old events
        self._cleanup_old_events()

        return event

    def _check_anomalies(self, event: AuditEvent) -> None:
        """Check if event triggers any anomaly patterns."""
        for pattern in self.anomaly_patterns:
            if event.event_type in pattern.event_types:
                # Count matching events in time window
                window_start = event.timestamp - timedelta(minutes=pattern.time_window)

                count = sum(
                    1
                    for e in self.events
                    if e.event_type in pattern.event_types
                    and e.timestamp >= window_start
                    and e.timestamp <= event.timestamp
                    and (not event.principal_id or e.principal_id == event.principal_id)
                )

                if count >= pattern.threshold:
                    # Anomaly detected
                    anomaly = {
                        "pattern": pattern.name,
                        "description": pattern.description,
                        "timestamp": event.timestamp,
                        "principal_id": event.principal_id,
                        "event_count": count,
                        "risk_score": pattern.risk_score,
                        "events": self._get_pattern_events(
                            pattern, window_start, event.timestamp, event.principal_id
                        ),
                    }

                    self.detected_anomalies.append(anomaly)

                    # Log anomaly as an event
                    self.log_event(
                        AuditEventType.ANOMALY_DETECTED,
                        principal_id=event.principal_id,
                        metadata={
                            "pattern": pattern.name,
                            "risk_score": pattern.risk_score,
                            "event_count": count,
                        },
                    )

    def _get_pattern_events(
        self,
        pattern: AnomalyPattern,
        start: datetime,
        end: datetime,
        principal_id: Optional[str],
    ) -> List[str]:
        """Get event IDs matching pattern in time window."""
        return [
            e.id
            for e in self.events
            if e.event_type in pattern.event_types
            and e.timestamp >= start
            and e.timestamp <= end
            and (not principal_id or e.principal_id == principal_id)
        ]

    def _cleanup_old_events(self) -> None:
        """Remove events older than retention period."""
        if not self.events:
            return

        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)

        # Find cutoff index
        cutoff_idx = 0
        for i, event in enumerate(self.events):
            if event.timestamp >= cutoff:
                cutoff_idx = i
                break

        if cutoff_idx > 0:
            # Remove old events
            self.events = self.events[cutoff_idx:]

            # Rebuild indices
            self.event_index.clear()
            for idx, event in enumerate(self.events):
                if event.principal_id:
                    self.event_index[f"principal:{event.principal_id}"].append(idx)
                if event.resource:
                    self.event_index[f"resource:{event.resource}"].append(idx)
                self.event_index[f"type:{event.event_type}"].append(idx)

    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        principal_id: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Query audit events.

        Args:
            event_type: Filter by event type
            principal_id: Filter by principal
            resource: Filter by resource
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum results

        Returns:
            List of matching events
        """
        # Start with all events or use index
        if principal_id:
            indices = self.event_index.get(f"principal:{principal_id}", [])
            candidates = [self.events[i] for i in indices]
        elif resource:
            indices = self.event_index.get(f"resource:{resource}", [])
            candidates = [self.events[i] for i in indices]
        elif event_type:
            indices = self.event_index.get(f"type:{event_type}", [])
            candidates = [self.events[i] for i in indices]
        else:
            candidates = self.events

        # Apply filters
        results = []
        for event in reversed(candidates):  # Most recent first
            if event_type and event.event_type != event_type:
                continue
            if principal_id and event.principal_id != principal_id:
                continue
            if resource and event.resource != resource:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            results.append(event)
            if len(results) >= limit:
                break

        return results

    def get_anomalies(
        self, start_time: Optional[datetime] = None, min_risk_score: float = 0.5
    ) -> List[Dict]:
        """Get detected anomalies."""
        anomalies = []

        for anomaly in self.detected_anomalies:
            if start_time and anomaly["timestamp"] < start_time:
                continue
            if anomaly["risk_score"] < min_risk_score:
                continue

            anomalies.append(anomaly)

        return anomalies

    def generate_compliance_report(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report for period.

        Args:
            start_time: Report start time
            end_time: Report end time

        Returns:
            Compliance report data
        """
        events = self.query_events(
            start_time=start_time, end_time=end_time, limit=10000
        )

        # Event statistics
        event_stats = defaultdict(int)
        for event in events:
            event_stats[event.event_type] += 1

        # Principal activity
        principal_activity = defaultdict(int)
        for event in events:
            if event.principal_id:
                principal_activity[event.principal_id] += 1

        # Failed operations
        failed_ops = [e for e in events if e.result != "success"]

        # Security events
        security_events = [
            e
            for e in events
            if e.event_type
            in {
                AuditEventType.LOGIN_FAILED,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.ANOMALY_DETECTED,
                AuditEventType.ENCRYPTION_KEY_ROTATED,
            }
        ]

        # Anomalies in period
        period_anomalies = [
            a
            for a in self.detected_anomalies
            if a["timestamp"] >= start_time and a["timestamp"] <= end_time
        ]

        return {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "total_events": len(events),
            "event_breakdown": dict(event_stats),
            "active_principals": len(principal_activity),
            "top_principals": sorted(
                principal_activity.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "failed_operations": len(failed_ops),
            "security_events": len(security_events),
            "anomalies_detected": len(period_anomalies),
            "high_risk_anomalies": len(
                [a for a in period_anomalies if a["risk_score"] >= 0.8]
            ),
            "compliance_issues": self._check_compliance_issues(events),
        }

    def _check_compliance_issues(self, events: List[AuditEvent]) -> List[Dict]:
        """Check for compliance issues in events."""
        issues = []

        # Check for unencrypted data access
        unencrypted_access = [
            e
            for e in events
            if e.event_type in {AuditEventType.READ, AuditEventType.WRITE}
            and not e.metadata.get("encrypted", True)
        ]

        if unencrypted_access:
            issues.append(
                {
                    "type": "unencrypted_access",
                    "severity": "high",
                    "count": len(unencrypted_access),
                    "description": "Data accessed without encryption",
                }
            )

        # Check for missing audit trails
        principals_without_logout = set()
        for event in events:
            if event.event_type == AuditEventType.LOGIN:
                principals_without_logout.add(event.principal_id)
            elif event.event_type == AuditEventType.LOGOUT:
                principals_without_logout.discard(event.principal_id)

        if principals_without_logout:
            issues.append(
                {
                    "type": "missing_logout",
                    "severity": "medium",
                    "count": len(principals_without_logout),
                    "description": "Sessions without proper logout",
                }
            )

        return issues
