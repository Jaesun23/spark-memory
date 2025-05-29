"""Access control management for Memory One Spark migration."""

from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel


class Permission(str, Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class Role(str, Enum):
    """User roles."""
    VIEWER = "viewer"
    USER = "user"
    OPERATOR = "operator"
    ADMIN = "admin"


class AccessControlManager:
    """Manages access control for migration operations."""
    
    def __init__(self):
        """Initialize access control manager."""
        self.role_permissions: Dict[Role, Set[Permission]] = {
            Role.VIEWER: {Permission.READ},
            Role.USER: {Permission.READ, Permission.WRITE},
            Role.OPERATOR: {Permission.READ, Permission.WRITE, Permission.DELETE},
            Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
        }
        
    async def check_permission(
        self,
        user_role: Role,
        required_permission: Permission
    ) -> bool:
        """Check if a role has the required permission."""
        permissions = self.role_permissions.get(user_role, set())
        return required_permission in permissions
    
    async def get_user_permissions(self, user_role: Role) -> Set[Permission]:
        """Get all permissions for a user role."""
        return self.role_permissions.get(user_role, set()).copy()