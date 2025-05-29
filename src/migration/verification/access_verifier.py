"""
Access rights verification for Memory One Spark migration.

This module verifies that access control and permissions are
properly configured and maintained after migration to Redis.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from pydantic import BaseModel

from ...redis.client import RedisClient
from ...utils.logging import get_logger

logger = get_logger(__name__)


class AccessLevel(str, Enum):
    """Access levels for memory resources."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class AccessRule(BaseModel):
    """Individual access rule definition."""
    
    user: str
    resource_pattern: str
    access_level: AccessLevel
    commands: Optional[List[str]] = None
    expires_at: Optional[datetime] = None


class AccessViolation(BaseModel):
    """Record of an access violation or discrepancy."""
    
    violation_id: str
    timestamp: datetime
    user: str
    resource: str
    attempted_action: str
    expected_access: AccessLevel
    actual_access: AccessLevel
    details: Optional[Dict[str, Any]] = None


class AccessVerificationReport(BaseModel):
    """Complete access verification report."""
    
    report_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_users_checked: int
    total_resources_checked: int
    access_rules_verified: int
    violations_found: List[AccessViolation] = []
    orphaned_resources: List[str] = []
    unauthorized_users: List[str] = []
    summary: Dict[str, Any] = {}


class AccessVerifier:
    """
    Verifies access control configuration and enforcement.
    
    Capabilities:
    - User permission verification
    - Resource access validation
    - ACL rule compliance
    - Access pattern analysis
    - Permission migration validation
    """
    
    def __init__(self, redis_client: RedisClient):
        """
        Initialize the access verifier.
        
        Args:
            redis_client: Redis client for access checks
        """
        self.redis_client = redis_client
        self.report: Optional[AccessVerificationReport] = None
        self.violation_counter = 0
    
    async def verify_access_configuration(
        self,
        expected_rules: List[AccessRule],
        resource_patterns: List[str] = ["*"]
    ) -> AccessVerificationReport:
        """
        Verify that access configuration matches expected rules.
        
        Args:
            expected_rules: List of expected access rules
            resource_patterns: Resource patterns to verify
            
        Returns:
            Access verification report
        """
        report_id = f"access_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.report = AccessVerificationReport(
            report_id=report_id,
            start_time=datetime.now(),
            total_users_checked=0,
            total_resources_checked=0,
            access_rules_verified=0
        )
        
        logger.info(f"Starting access verification {report_id}")
        
        # Get current ACL configuration
        current_acl = await self._get_current_acl()
        
        # Verify each expected rule
        for rule in expected_rules:
            await self._verify_access_rule(rule, current_acl)
        
        # Check for unauthorized access
        await self._check_unauthorized_access(current_acl, expected_rules)
        
        # Verify resource permissions
        await self._verify_resource_permissions(resource_patterns, expected_rules)
        
        # Check for orphaned resources
        await self._check_orphaned_resources(resource_patterns, expected_rules)
        
        # Generate summary
        self.report.end_time = datetime.now()
        self.report.summary = self._generate_summary()
        
        logger.info(f"Access verification complete: {len(self.report.violations_found)} violations found")
        return self.report
    
    async def _get_current_acl(self) -> Dict[str, Any]:
        """Get current ACL configuration from Redis."""
        try:
            acl_list = await self.redis_client.execute_command("ACL", "LIST")
            
            # Parse ACL rules into structured format
            acl_dict = {}
            for user_acl in acl_list:
                parts = user_acl.split()
                if len(parts) >= 2 and parts[0] == "user":
                    username = parts[1]
                    acl_dict[username] = {
                        "enabled": "on" in parts,
                        "passwords": [],
                        "commands": [],
                        "keys": [],
                        "raw": user_acl
                    }
                    
                    # Parse command permissions
                    for i, part in enumerate(parts):
                        if part.startswith("+"):
                            acl_dict[username]["commands"].append(part[1:])
                        elif part.startswith("~"):
                            acl_dict[username]["keys"].append(part[1:])
            
            return acl_dict
            
        except Exception as e:
            logger.error(f"Failed to get ACL configuration: {e}")
            return {}
    
    async def _verify_access_rule(
        self,
        rule: AccessRule,
        current_acl: Dict[str, Any]
    ):
        """Verify a single access rule."""
        self.report.access_rules_verified += 1
        
        if rule.user not in current_acl:
            self._add_violation(
                user=rule.user,
                resource=rule.resource_pattern,
                attempted_action="user_existence",
                expected_access=rule.access_level,
                actual_access=AccessLevel.NONE,
                details={"error": "User not found in ACL"}
            )
            return
        
        user_acl = current_acl[rule.user]
        
        # Check if user is enabled
        if not user_acl["enabled"]:
            self._add_violation(
                user=rule.user,
                resource=rule.resource_pattern,
                attempted_action="user_status",
                expected_access=rule.access_level,
                actual_access=AccessLevel.NONE,
                details={"error": "User is disabled"}
            )
            return
        
        # Verify key patterns
        if not self._pattern_matches(rule.resource_pattern, user_acl["keys"]):
            self._add_violation(
                user=rule.user,
                resource=rule.resource_pattern,
                attempted_action="key_pattern",
                expected_access=rule.access_level,
                actual_access=AccessLevel.NONE,
                details={
                    "expected_pattern": rule.resource_pattern,
                    "user_patterns": user_acl["keys"]
                }
            )
        
        # Verify command permissions based on access level
        required_commands = self._get_required_commands(rule.access_level)
        missing_commands = []
        
        for cmd in required_commands:
            if not self._has_command_permission(cmd, user_acl["commands"]):
                missing_commands.append(cmd)
        
        if missing_commands:
            self._add_violation(
                user=rule.user,
                resource=rule.resource_pattern,
                attempted_action="command_permissions",
                expected_access=rule.access_level,
                actual_access=self._determine_actual_access(user_acl["commands"]),
                details={"missing_commands": missing_commands}
            )
    
    async def _check_unauthorized_access(
        self,
        current_acl: Dict[str, Any],
        expected_rules: List[AccessRule]
    ):
        """Check for users with access that shouldn't have it."""
        expected_users = {rule.user for rule in expected_rules}
        self.report.total_users_checked = len(current_acl)
        
        for username, user_acl in current_acl.items():
            if username == "default":
                # Special handling for default user
                if user_acl["enabled"]:
                    self.report.unauthorized_users.append("default (should be disabled)")
                continue
            
            if username not in expected_users and user_acl["enabled"]:
                self.report.unauthorized_users.append(username)
                self._add_violation(
                    user=username,
                    resource="*",
                    attempted_action="unauthorized_access",
                    expected_access=AccessLevel.NONE,
                    actual_access=self._determine_actual_access(user_acl["commands"]),
                    details={"error": "User not in expected access list"}
                )
    
    async def _verify_resource_permissions(
        self,
        resource_patterns: List[str],
        expected_rules: List[AccessRule]
    ):
        """Verify permissions on actual resources."""
        all_resources = set()
        
        for pattern in resource_patterns:
            resources = await self.redis_client.scan_keys(pattern)
            all_resources.update(resources)
        
        self.report.total_resources_checked = len(all_resources)
        
        # Sample verification of actual access
        sample_size = min(len(all_resources), 100)
        sample_resources = list(all_resources)[:sample_size]
        
        for resource in sample_resources:
            # Check which users can access this resource
            for rule in expected_rules:
                if self._resource_matches_pattern(resource, rule.resource_pattern):
                    # Verify user can actually access the resource
                    await self._verify_resource_access(resource, rule)
    
    async def _verify_resource_access(
        self,
        resource: str,
        rule: AccessRule
    ):
        """Verify actual access to a specific resource."""
        # This would typically involve simulating access attempts
        # For now, we'll check if the key pattern allows access
        
        try:
            # Get user's ACL info
            user_info = await self.redis_client.execute_command("ACL", "GETUSER", rule.user)
            
            if not user_info:
                self._add_violation(
                    user=rule.user,
                    resource=resource,
                    attempted_action="resource_access",
                    expected_access=rule.access_level,
                    actual_access=AccessLevel.NONE,
                    details={"error": "User not found"}
                )
                return
            
            # Parse user info to check key patterns
            key_patterns = []
            for i, item in enumerate(user_info):
                if i > 0 and user_info[i-1] == "keys":
                    key_patterns = item if isinstance(item, list) else [item]
                    break
            
            # Check if resource matches any pattern
            resource_accessible = any(
                self._resource_matches_pattern(resource, pattern.lstrip("~"))
                for pattern in key_patterns
            )
            
            if not resource_accessible:
                self._add_violation(
                    user=rule.user,
                    resource=resource,
                    attempted_action="resource_pattern_match",
                    expected_access=rule.access_level,
                    actual_access=AccessLevel.NONE,
                    details={"user_patterns": key_patterns}
                )
                
        except Exception as e:
            logger.warning(f"Could not verify resource access for {rule.user} on {resource}: {e}")
    
    async def _check_orphaned_resources(
        self,
        resource_patterns: List[str],
        expected_rules: List[AccessRule]
    ):
        """Check for resources with no access rules."""
        all_resources = set()
        
        for pattern in resource_patterns:
            resources = await self.redis_client.scan_keys(pattern)
            all_resources.update(resources)
        
        # Build set of all covered patterns
        covered_patterns = {rule.resource_pattern for rule in expected_rules}
        
        # Check each resource
        for resource in all_resources:
            is_covered = False
            for pattern in covered_patterns:
                if self._resource_matches_pattern(resource, pattern):
                    is_covered = True
                    break
            
            if not is_covered:
                self.report.orphaned_resources.append(resource)
    
    def _pattern_matches(self, expected: str, actual_patterns: List[str]) -> bool:
        """Check if expected pattern is covered by actual patterns."""
        for pattern in actual_patterns:
            if pattern == "*" or pattern == expected:
                return True
            # Simple pattern matching (could be enhanced)
            if expected.startswith(pattern.rstrip("*")):
                return True
        return False
    
    def _resource_matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if a resource matches a pattern."""
        if pattern == "*":
            return True
        
        # Convert Redis pattern to regex-like matching
        import fnmatch
        return fnmatch.fnmatch(resource, pattern)
    
    def _get_required_commands(self, access_level: AccessLevel) -> List[str]:
        """Get required Redis commands for an access level."""
        if access_level == AccessLevel.READ:
            return ["get", "mget", "exists", "scan", "type"]
        elif access_level == AccessLevel.WRITE:
            return ["get", "set", "del", "exists", "scan", "expire"]
        elif access_level == AccessLevel.ADMIN:
            return ["get", "set", "del", "flushdb", "config", "acl"]
        return []
    
    def _has_command_permission(self, command: str, user_commands: List[str]) -> bool:
        """Check if user has permission for a command."""
        return (
            "all" in user_commands or
            command in user_commands or
            f"@{command}" in user_commands
        )
    
    def _determine_actual_access(self, commands: List[str]) -> AccessLevel:
        """Determine actual access level from commands."""
        if "all" in commands or "flushdb" in commands:
            return AccessLevel.ADMIN
        elif any(cmd in commands for cmd in ["set", "del"]):
            return AccessLevel.WRITE
        elif any(cmd in commands for cmd in ["get", "mget"]):
            return AccessLevel.READ
        return AccessLevel.NONE
    
    def _add_violation(
        self,
        user: str,
        resource: str,
        attempted_action: str,
        expected_access: AccessLevel,
        actual_access: AccessLevel,
        details: Optional[Dict[str, Any]] = None
    ):
        """Add an access violation to the report."""
        self.violation_counter += 1
        violation = AccessViolation(
            violation_id=f"AV-{self.violation_counter:04d}",
            timestamp=datetime.now(),
            user=user,
            resource=resource,
            attempted_action=attempted_action,
            expected_access=expected_access,
            actual_access=actual_access,
            details=details
        )
        self.report.violations_found.append(violation)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate verification summary."""
        return {
            "total_violations": len(self.report.violations_found),
            "unauthorized_users": len(self.report.unauthorized_users),
            "orphaned_resources": len(self.report.orphaned_resources),
            "compliance_rate": (
                (self.report.access_rules_verified - len(self.report.violations_found)) /
                self.report.access_rules_verified * 100
            ) if self.report.access_rules_verified > 0 else 0,
            "verification_status": "PASSED" if not self.report.violations_found else "FAILED"
        }
    
    async def audit_access_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Audit Redis ACL logs for access violations.
        
        Args:
            start_time: Start of audit period
            end_time: End of audit period
            
        Returns:
            List of access events
        """
        try:
            # Get ACL log entries
            acl_log = await self.redis_client.execute_command("ACL", "LOG")
            
            events = []
            for entry in acl_log:
                event = {
                    "timestamp": entry.get("timestamp"),
                    "user": entry.get("username"),
                    "reason": entry.get("reason"),
                    "command": entry.get("object"),
                    "client": entry.get("client-info", {})
                }
                
                # Filter by time if specified
                if start_time or end_time:
                    event_time = datetime.fromtimestamp(float(entry.get("timestamp", 0)))
                    if start_time and event_time < start_time:
                        continue
                    if end_time and event_time > end_time:
                        continue
                
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to audit access logs: {e}")
            return []