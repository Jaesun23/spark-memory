"""
Access control module for Memory One Spark.

This module provides Role-Based Access Control (RBAC) and API key management.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from ..memory.models import MemoryError


class AccessControlError(MemoryError):
    """Raised when access control operations fail."""

    pass


class Permission(str, Enum):
    """Memory operation permissions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

    # Specific permissions
    SEARCH = "search"
    CONSOLIDATE = "consolidate"
    LIFECYCLE = "lifecycle"
    SYSTEM = "system"


class Role(str, Enum):
    """Predefined roles with permission sets."""

    VIEWER = "viewer"
    USER = "user"
    EDITOR = "editor"
    ADMIN = "admin"
    SYSTEM = "system"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {Permission.READ, Permission.SEARCH},
    Role.USER: {Permission.READ, Permission.WRITE, Permission.SEARCH},
    Role.EDITOR: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.SEARCH,
        Permission.CONSOLIDATE,
    },
    Role.ADMIN: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.ADMIN,
        Permission.SEARCH,
        Permission.CONSOLIDATE,
        Permission.LIFECYCLE,
        Permission.SYSTEM,
    },
    Role.SYSTEM: set(Permission),  # All permissions
}


class Principal(BaseModel):
    """Represents a security principal (user or service)."""

    id: str
    type: str = "user"  # user, service, api_key
    roles: Set[Role] = Field(default_factory=set)
    permissions: Set[Permission] = Field(default_factory=set)
    metadata: Dict = Field(default_factory=dict)
    created: datetime = Field(default_factory=datetime.utcnow)
    last_access: Optional[datetime] = None


class APIKey(BaseModel):
    """API key for authentication."""

    key_hash: str
    principal_id: str
    name: str
    created: datetime = Field(default_factory=datetime.utcnow)
    expires: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None  # Requests per minute
    scopes: Set[str] = Field(default_factory=set)
    active: bool = True


class AccessContext(BaseModel):
    """Context for access control decisions."""

    principal: Principal
    resource: str  # Memory path or resource identifier
    action: Permission
    metadata: Dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AccessControlService:
    """
    Service for managing access control.

    Features:
    - Role-based access control
    - API key management
    - Permission checking
    - Rate limiting
    """

    def __init__(self):
        """Initialize access control service."""
        self.principals: Dict[str, Principal] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.resource_permissions: Dict[str, Dict[str, Set[Permission]]] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}

    def create_principal(
        self,
        principal_id: str,
        principal_type: str = "user",
        roles: Optional[Set[Role]] = None,
        permissions: Optional[Set[Permission]] = None,
        metadata: Optional[Dict] = None,
    ) -> Principal:
        """
        Create a new security principal.

        Args:
            principal_id: Unique identifier
            principal_type: Type of principal (user, service, etc.)
            roles: Initial roles
            permissions: Additional permissions
            metadata: Additional metadata

        Returns:
            Created principal
        """
        if principal_id in self.principals:
            raise AccessControlError(f"Principal already exists: {principal_id}")

        principal = Principal(
            id=principal_id,
            type=principal_type,
            roles=roles or {Role.USER},
            permissions=permissions or set(),
            metadata=metadata or {},
        )

        self.principals[principal_id] = principal
        return principal

    def generate_api_key(
        self,
        principal_id: str,
        name: str,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        scopes: Optional[Set[str]] = None,
    ) -> str:
        """
        Generate a new API key.

        Args:
            principal_id: Principal this key belongs to
            name: Descriptive name for the key
            expires_in_days: Days until expiration
            rate_limit: Requests per minute limit
            scopes: Allowed scopes

        Returns:
            Generated API key (only returned once!)
        """
        if principal_id not in self.principals:
            raise AccessControlError(f"Principal not found: {principal_id}")

        # Generate secure random key
        api_key = f"msk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Create key record
        expires = None
        if expires_in_days:
            expires = datetime.utcnow() + timedelta(days=expires_in_days)

        key_record = APIKey(
            key_hash=key_hash,
            principal_id=principal_id,
            name=name,
            expires=expires,
            rate_limit=rate_limit,
            scopes=scopes or set(),
        )

        self.api_keys[key_hash] = key_record
        return api_key

    def validate_api_key(self, api_key: str) -> Optional[Principal]:
        """
        Validate an API key and return associated principal.

        Args:
            api_key: API key to validate

        Returns:
            Associated principal or None if invalid
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash not in self.api_keys:
            return None

        key_record = self.api_keys[key_hash]

        # Check if active
        if not key_record.active:
            return None

        # Check expiration
        if key_record.expires and datetime.utcnow() > key_record.expires:
            key_record.active = False
            return None

        # Check rate limit
        if key_record.rate_limit and not self._check_rate_limit(
            key_hash, key_record.rate_limit
        ):
            return None

        # Update usage
        key_record.last_used = datetime.utcnow()
        key_record.usage_count += 1

        # Return principal
        return self.principals.get(key_record.principal_id)

    def _check_rate_limit(self, key_hash: str, limit: int) -> bool:
        """Check if rate limit is exceeded."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Get or create rate limit tracking
        if key_hash not in self.rate_limits:
            self.rate_limits[key_hash] = []

        # Remove old entries
        self.rate_limits[key_hash] = [
            ts for ts in self.rate_limits[key_hash] if ts > minute_ago
        ]

        # Check limit
        if len(self.rate_limits[key_hash]) >= limit:
            return False

        # Add current request
        self.rate_limits[key_hash].append(now)
        return True

    def check_permission(self, context: AccessContext) -> bool:
        """
        Check if access should be granted.

        Args:
            context: Access context with principal, resource, and action

        Returns:
            True if access granted
        """
        principal = context.principal

        # Update last access
        principal.last_access = datetime.utcnow()

        # Check direct permissions
        if context.action in principal.permissions:
            return True

        # Check role permissions
        for role in principal.roles:
            if context.action in ROLE_PERMISSIONS.get(role, set()):
                return True

        # Check resource-specific permissions
        if context.resource in self.resource_permissions:
            resource_perms = self.resource_permissions[context.resource]
            if principal.id in resource_perms:
                if context.action in resource_perms[principal.id]:
                    return True

        return False

    def grant_permission(
        self, principal_id: str, resource: str, permissions: Set[Permission]
    ) -> None:
        """Grant permissions on a specific resource."""
        if principal_id not in self.principals:
            raise AccessControlError(f"Principal not found: {principal_id}")

        if resource not in self.resource_permissions:
            self.resource_permissions[resource] = {}

        if principal_id not in self.resource_permissions[resource]:
            self.resource_permissions[resource][principal_id] = set()

        self.resource_permissions[resource][principal_id].update(permissions)

    def revoke_permission(
        self, principal_id: str, resource: str, permissions: Set[Permission]
    ) -> None:
        """Revoke permissions on a specific resource."""
        if resource in self.resource_permissions:
            if principal_id in self.resource_permissions[resource]:
                self.resource_permissions[resource][principal_id] -= permissions

    def assign_role(self, principal_id: str, role: Role) -> None:
        """Assign a role to a principal."""
        if principal_id not in self.principals:
            raise AccessControlError(f"Principal not found: {principal_id}")

        self.principals[principal_id].roles.add(role)

    def remove_role(self, principal_id: str, role: Role) -> None:
        """Remove a role from a principal."""
        if principal_id not in self.principals:
            raise AccessControlError(f"Principal not found: {principal_id}")

        self.principals[principal_id].roles.discard(role)

    def list_api_keys(self, principal_id: str) -> List[Dict]:
        """List API keys for a principal."""
        keys = []
        for key_hash, key_record in self.api_keys.items():
            if key_record.principal_id == principal_id:
                keys.append(
                    {
                        "name": key_record.name,
                        "created": key_record.created,
                        "expires": key_record.expires,
                        "last_used": key_record.last_used,
                        "usage_count": key_record.usage_count,
                        "active": key_record.active,
                    }
                )
        return keys

    def revoke_api_key(self, api_key: str) -> None:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash in self.api_keys:
            self.api_keys[key_hash].active = False
