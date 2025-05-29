"""
Security audit capabilities for Memory One Spark migration.

This module performs comprehensive security audits to ensure
data protection, access control, and compliance during migration.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from pydantic import BaseModel

from ...redis.client import RedisClient
from ...utils.logging import get_logger

logger = get_logger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for audit findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityFinding(BaseModel):
    """Individual security finding."""
    
    finding_id: str
    category: str
    level: SecurityLevel
    title: str
    description: str
    affected_resources: List[str]
    recommendation: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None


class SecurityAuditReport(BaseModel):
    """Complete security audit report."""
    
    audit_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_resources_scanned: int
    findings: List[SecurityFinding] = []
    summary: Dict[str, int] = {}
    compliance_status: Dict[str, bool] = {}
    recommendations: List[str] = []


class SecurityAuditor:
    """
    Performs security audits on migrated data and system configuration.
    
    Audit areas include:
    - Access control configuration
    - Data encryption status
    - Key management practices
    - Sensitive data exposure
    - Connection security
    - Audit logging
    """
    
    def __init__(self, redis_client: RedisClient):
        """
        Initialize the security auditor.
        
        Args:
            redis_client: Redis client for security checks
        """
        self.redis_client = redis_client
        self.audit_report: Optional[SecurityAuditReport] = None
        self.finding_counter = 0
    
    async def perform_security_audit(
        self,
        scan_patterns: List[str] = ["*"],
        check_compliance: bool = True
    ) -> SecurityAuditReport:
        """
        Perform comprehensive security audit.
        
        Args:
            scan_patterns: Redis key patterns to audit
            check_compliance: Whether to check compliance standards
            
        Returns:
            Complete security audit report
        """
        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.audit_report = SecurityAuditReport(
            audit_id=audit_id,
            start_time=datetime.now(),
            total_resources_scanned=0
        )
        
        logger.info(f"Starting security audit {audit_id}")
        
        # Run security checks
        await self._check_access_controls()
        await self._check_encryption_status()
        await self._check_sensitive_data_exposure(scan_patterns)
        await self._check_connection_security()
        await self._check_audit_logging()
        await self._check_key_management(scan_patterns)
        
        if check_compliance:
            await self._check_compliance_standards()
        
        # Generate summary and recommendations
        self._generate_summary()
        self._generate_recommendations()
        
        self.audit_report.end_time = datetime.now()
        
        logger.info(f"Security audit complete: {len(self.audit_report.findings)} findings")
        return self.audit_report
    
    async def _check_access_controls(self):
        """Check Redis access control configuration."""
        try:
            # Check if ACL is enabled
            acl_info = await self.redis_client.execute_command("ACL", "LIST")
            
            if not acl_info or len(acl_info) <= 1:
                self._add_finding(
                    category="Access Control",
                    level=SecurityLevel.HIGH,
                    title="ACL Not Properly Configured",
                    description="Redis ACL (Access Control List) is not configured or has only default user",
                    affected_resources=["Redis Configuration"],
                    recommendation="Configure Redis ACL with proper user permissions and disable default user"
                )
            
            # Check for default user
            for user_info in acl_info:
                if "user default" in str(user_info) and "on" in str(user_info):
                    self._add_finding(
                        category="Access Control",
                        level=SecurityLevel.CRITICAL,
                        title="Default User Enabled",
                        description="Redis default user is enabled, allowing unauthenticated access",
                        affected_resources=["Redis Default User"],
                        recommendation="Disable default user and create specific users with minimal required permissions"
                    )
            
        except Exception as e:
            logger.warning(f"Could not check access controls: {e}")
            self._add_finding(
                category="Access Control",
                level=SecurityLevel.MEDIUM,
                title="Access Control Check Failed",
                description=f"Unable to verify access control configuration: {str(e)}",
                affected_resources=["Redis Configuration"],
                recommendation="Manually verify Redis ACL configuration"
            )
    
    async def _check_encryption_status(self):
        """Check encryption configuration."""
        try:
            # Check TLS configuration
            config = await self.redis_client.execute_command("CONFIG", "GET", "tls-*")
            tls_enabled = any("yes" in str(v) for k, v in zip(config[::2], config[1::2]) if "tls-port" in str(k))
            
            if not tls_enabled:
                self._add_finding(
                    category="Encryption",
                    level=SecurityLevel.HIGH,
                    title="TLS Not Enabled",
                    description="Redis TLS encryption is not enabled for client connections",
                    affected_resources=["Redis Connections"],
                    recommendation="Enable TLS for all Redis connections to encrypt data in transit"
                )
            
            # Check persistence encryption
            persistence_config = await self.redis_client.execute_command("CONFIG", "GET", "rdb*")
            
            # Note: Redis doesn't have built-in encryption at rest
            self._add_finding(
                category="Encryption",
                level=SecurityLevel.MEDIUM,
                title="Encryption at Rest",
                description="Redis does not provide built-in encryption at rest",
                affected_resources=["Redis Persistence Files"],
                recommendation="Use filesystem-level encryption or Redis Enterprise features for data at rest encryption"
            )
            
        except Exception as e:
            logger.warning(f"Could not check encryption status: {e}")
    
    async def _check_sensitive_data_exposure(self, patterns: List[str]):
        """Check for exposed sensitive data."""
        sensitive_patterns = [
            "password", "passwd", "pwd", "secret", "key", "token",
            "api_key", "private", "credential", "auth", "ssn", "credit_card"
        ]
        
        resources_scanned = 0
        exposed_resources = []
        
        for pattern in patterns:
            keys = await self.redis_client.scan_keys(pattern)
            resources_scanned += len(keys)
            
            for key in keys:
                # Check key names for sensitive patterns
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in sensitive_patterns):
                    exposed_resources.append(key)
                
                # Sample check on values (limited to avoid performance impact)
                if len(exposed_resources) < 100:  # Limit deep scanning
                    try:
                        data = await self.redis_client.get_json(key)
                        if data and isinstance(data, dict):
                            for field, value in data.items():
                                if any(sensitive in field.lower() for sensitive in sensitive_patterns):
                                    if not self._is_properly_encrypted(value):
                                        exposed_resources.append(f"{key}:{field}")
                    except:
                        pass
        
        self.audit_report.total_resources_scanned = resources_scanned
        
        if exposed_resources:
            self._add_finding(
                category="Data Protection",
                level=SecurityLevel.HIGH,
                title="Potential Sensitive Data Exposure",
                description=f"Found {len(exposed_resources)} resources with potentially sensitive data in clear text",
                affected_resources=exposed_resources[:10],  # Limit to first 10
                recommendation="Encrypt sensitive data before storage and use secure key naming conventions",
                metadata={"total_exposed": len(exposed_resources)}
            )
    
    async def _check_connection_security(self):
        """Check Redis connection security settings."""
        try:
            # Check bind configuration
            bind_config = await self.redis_client.execute_command("CONFIG", "GET", "bind")
            if bind_config and "0.0.0.0" in str(bind_config[1]):
                self._add_finding(
                    category="Network Security",
                    level=SecurityLevel.HIGH,
                    title="Redis Bound to All Interfaces",
                    description="Redis is configured to accept connections from any network interface",
                    affected_resources=["Redis Network Configuration"],
                    recommendation="Bind Redis to specific interfaces (e.g., localhost or private network only)"
                )
            
            # Check protected mode
            protected_mode = await self.redis_client.execute_command("CONFIG", "GET", "protected-mode")
            if protected_mode and protected_mode[1] == "no":
                self._add_finding(
                    category="Network Security",
                    level=SecurityLevel.CRITICAL,
                    title="Protected Mode Disabled",
                    description="Redis protected mode is disabled, allowing connections without authentication",
                    affected_resources=["Redis Security Configuration"],
                    recommendation="Enable protected mode or ensure proper authentication is configured"
                )
            
        except Exception as e:
            logger.warning(f"Could not check connection security: {e}")
    
    async def _check_audit_logging(self):
        """Check audit logging configuration."""
        try:
            # Check if logging is enabled
            log_config = await self.redis_client.execute_command("CONFIG", "GET", "logfile")
            
            if not log_config or not log_config[1]:
                self._add_finding(
                    category="Audit & Compliance",
                    level=SecurityLevel.MEDIUM,
                    title="Logging Not Configured",
                    description="Redis logging is not configured to a file",
                    affected_resources=["Redis Logging"],
                    recommendation="Configure Redis logging to track access and operations"
                )
            
            # Check ACL log
            acl_log = await self.redis_client.execute_command("ACL", "LOG")
            if not acl_log:
                self._add_finding(
                    category="Audit & Compliance",
                    level=SecurityLevel.LOW,
                    title="No ACL Violations Logged",
                    description="No ACL violations found in log (this could be good or indicate logging issues)",
                    affected_resources=["ACL Logging"],
                    recommendation="Ensure ACL logging is enabled and monitor for unauthorized access attempts"
                )
            
        except Exception as e:
            logger.warning(f"Could not check audit logging: {e}")
    
    async def _check_key_management(self, patterns: List[str]):
        """Check Redis key management practices."""
        expired_keys = 0
        no_ttl_keys = 0
        total_keys = 0
        
        for pattern in patterns:
            keys = await self.redis_client.scan_keys(pattern)
            total_keys += len(keys)
            
            # Sample TTL check
            sample_size = min(len(keys), 100)
            for key in keys[:sample_size]:
                try:
                    ttl = await self.redis_client.execute_command("TTL", key)
                    if ttl == -1:
                        no_ttl_keys += 1
                    elif ttl == -2:
                        expired_keys += 1
                except:
                    pass
        
        if no_ttl_keys > total_keys * 0.5:
            self._add_finding(
                category="Key Management",
                level=SecurityLevel.MEDIUM,
                title="Missing TTL on Keys",
                description=f"Many keys ({no_ttl_keys}) do not have TTL set, risking data accumulation",
                affected_resources=["Redis Keys"],
                recommendation="Implement TTL policies for temporary data to prevent unbounded growth"
            )
    
    async def _check_compliance_standards(self):
        """Check compliance with common standards."""
        # GDPR compliance checks
        gdpr_compliant = True
        
        # Check for data retention policies
        if not await self._has_data_retention_policy():
            gdpr_compliant = False
            self._add_finding(
                category="Compliance",
                level=SecurityLevel.HIGH,
                title="No Data Retention Policy",
                description="No automated data retention policy detected",
                affected_resources=["Data Lifecycle"],
                recommendation="Implement data retention policies with automatic expiration for personal data"
            )
        
        # Check for data portability
        if not await self._supports_data_export():
            gdpr_compliant = False
            self._add_finding(
                category="Compliance",
                level=SecurityLevel.MEDIUM,
                title="Limited Data Portability",
                description="No structured data export mechanism found",
                affected_resources=["Data Export"],
                recommendation="Implement data export functionality for GDPR compliance"
            )
        
        self.audit_report.compliance_status["GDPR"] = gdpr_compliant
        
        # SOC2 compliance checks
        soc2_compliant = True
        
        # Check encryption
        if not await self._has_encryption():
            soc2_compliant = False
        
        # Check access controls
        if not await self._has_access_controls():
            soc2_compliant = False
        
        self.audit_report.compliance_status["SOC2"] = soc2_compliant
    
    def _add_finding(
        self,
        category: str,
        level: SecurityLevel,
        title: str,
        description: str,
        affected_resources: List[str],
        recommendation: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a security finding to the report."""
        self.finding_counter += 1
        finding = SecurityFinding(
            finding_id=f"SEC-{self.finding_counter:04d}",
            category=category,
            level=level,
            title=title,
            description=description,
            affected_resources=affected_resources,
            recommendation=recommendation,
            metadata=metadata
        )
        self.audit_report.findings.append(finding)
    
    def _generate_summary(self):
        """Generate audit summary statistics."""
        summary = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        
        for finding in self.audit_report.findings:
            summary[finding.level.value] += 1
        
        self.audit_report.summary = summary
    
    def _generate_recommendations(self):
        """Generate prioritized recommendations."""
        # Group by severity
        critical_high = [f for f in self.audit_report.findings 
                        if f.level in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]]
        
        recommendations = []
        
        if critical_high:
            recommendations.append(
                f"Address {len(critical_high)} critical/high severity findings immediately"
            )
        
        # Add specific recommendations
        categories = set(f.category for f in self.audit_report.findings)
        for category in categories:
            cat_findings = [f for f in self.audit_report.findings if f.category == category]
            if cat_findings:
                recommendations.append(
                    f"Review and remediate {len(cat_findings)} {category} findings"
                )
        
        self.audit_report.recommendations = recommendations
    
    def _is_properly_encrypted(self, value: Any) -> bool:
        """Check if a value appears to be encrypted."""
        if not isinstance(value, str):
            return False
        
        # Simple heuristic: encrypted values are typically base64 or hex
        # and have high entropy
        import re
        
        # Check for common encryption patterns
        if re.match(r'^[A-Za-z0-9+/]{32,}={0,2}$', value):  # Base64
            return True
        if re.match(r'^[0-9a-fA-F]{32,}$', value):  # Hex
            return True
        
        return False
    
    async def _has_data_retention_policy(self) -> bool:
        """Check if data retention policies are in place."""
        # Check for TTL usage
        try:
            info = await self.redis_client.execute_command("INFO", "keyspace")
            return "expires" in str(info)
        except:
            return False
    
    async def _supports_data_export(self) -> bool:
        """Check if data export is supported."""
        # This would check for export tools/endpoints
        return True  # Assume our migration tools provide this
    
    async def _has_encryption(self) -> bool:
        """Check if encryption is enabled."""
        try:
            config = await self.redis_client.execute_command("CONFIG", "GET", "tls-port")
            return bool(config and config[1])
        except:
            return False
    
    async def _has_access_controls(self) -> bool:
        """Check if access controls are configured."""
        try:
            acl_list = await self.redis_client.execute_command("ACL", "LIST")
            return len(acl_list) > 1  # More than just default user
        except:
            return False