"""
Security module for Memory One Spark.

This module provides comprehensive security features including:
- Data encryption (at-rest and in-transit)
- Key management with rotation
- Role-based access control (RBAC)
- API key management
- Audit logging with anomaly detection
"""

from .access_control import (
    ROLE_PERMISSIONS,
    AccessContext,
    AccessControlError,
    AccessControlService,
    APIKey,
    Permission,
    Principal,
    Role,
)
from .audit_log import (
    AnomalyPattern,
    AuditError,
    AuditEvent,
    AuditEventType,
    AuditLogger,
)
from .encryption import (
    EncryptionError,
    EncryptionService,
    FieldLevelEncryption,
)
from .key_manager import (
    KeyManagementError,
    KeyManager,
)

__all__ = [
    # Encryption
    "EncryptionService",
    "FieldLevelEncryption",
    "EncryptionError",
    # Key management
    "KeyManager",
    "KeyManagementError",
    # Access control
    "AccessControlService",
    "Principal",
    "APIKey",
    "Permission",
    "Role",
    "AccessContext",
    "AccessControlError",
    "ROLE_PERMISSIONS",
    # Audit logging
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AnomalyPattern",
    "AuditError",
]
