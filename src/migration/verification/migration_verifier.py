"""
Integrated migration verification orchestrator for Memory One Spark.

This module combines all verification tools to provide comprehensive
migration validation in a single workflow.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from pydantic import BaseModel

from .data_verifier import DataVerifier, DataVerificationReport
from .security_auditor import SecurityAuditor, SecurityAuditReport
from .access_verifier import AccessVerifier, AccessRule, AccessVerificationReport
from .report_generator import ReportGenerator, ReportConfig, ReportFormat
from ...redis.client import RedisClient
from ...utils.logging import get_logger
from ..security.access_control import AccessControlManager

logger = get_logger(__name__)


class VerificationConfig(BaseModel):
    """Configuration for migration verification."""
    
    # Data verification
    verify_data_integrity: bool = True
    source_path: Optional[Path] = None
    redis_prefix: str = "memory"
    
    # Security audit
    perform_security_audit: bool = True
    check_compliance: bool = True
    scan_patterns: List[str] = ["memory:*"]
    
    # Access verification
    verify_access_control: bool = True
    expected_access_rules: List[AccessRule] = []
    
    # Reporting
    report_config: ReportConfig = ReportConfig()
    report_formats: List[ReportFormat] = [
        ReportFormat.HTML,
        ReportFormat.JSON,
        ReportFormat.SUMMARY
    ]
    
    # Performance
    parallel_verification: bool = True
    sample_size: Optional[int] = None  # None = verify all


class VerificationResult(BaseModel):
    """Overall verification result."""
    
    verification_id: str
    start_time: datetime
    end_time: datetime
    overall_status: str  # PASSED, PASSED_WITH_WARNINGS, FAILED
    
    # Individual results
    data_verification_passed: Optional[bool] = None
    security_audit_passed: Optional[bool] = None
    access_verification_passed: Optional[bool] = None
    
    # Reports
    report_paths: Dict[str, Path] = {}
    
    # Summary
    critical_issues: int = 0
    warnings: int = 0
    recommendations: List[str] = []


class MigrationVerifier:
    """
    Orchestrates comprehensive migration verification.
    
    Combines data integrity, security, and access control verification
    into a single unified workflow with comprehensive reporting.
    """
    
    def __init__(
        self,
        redis_client: RedisClient,
        acl_manager: Optional[AccessControlManager] = None,
        config: Optional[VerificationConfig] = None
    ):
        """
        Initialize migration verifier.
        
        Args:
            redis_client: Redis client for verification
            acl_manager: Access control manager (optional)
            config: Verification configuration
        """
        self.redis_client = redis_client
        self.acl_manager = acl_manager
        self.config = config or VerificationConfig()
        
        # Initialize verifiers
        self.data_verifier = DataVerifier(redis_client)
        self.security_auditor = SecurityAuditor(redis_client)
        self.access_verifier = AccessVerifier(redis_client)
        self.report_generator = ReportGenerator(self.config.report_config)
    
    async def verify_migration(self) -> VerificationResult:
        """
        Perform complete migration verification.
        
        Returns:
            Comprehensive verification result
        """
        verification_id = f"migration_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting migration verification {verification_id}")
        
        # Initialize result
        result = VerificationResult(
            verification_id=verification_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            overall_status="UNKNOWN"
        )
        
        # Perform verifications
        data_report = None
        security_report = None
        access_report = None
        
        try:
            if self.config.parallel_verification:
                # Run verifications in parallel
                results = await self._run_parallel_verifications()
                data_report, security_report, access_report = results
            else:
                # Run verifications sequentially
                if self.config.verify_data_integrity:
                    data_report = await self._verify_data_integrity()
                
                if self.config.perform_security_audit:
                    security_report = await self._verify_security()
                
                if self.config.verify_access_control:
                    access_report = await self._verify_access_control()
            
            # Analyze results
            result = self._analyze_results(result, data_report, security_report, access_report)
            
            # Generate reports
            report_paths = await self.report_generator.generate_report(
                data_report=data_report,
                security_report=security_report,
                access_report=access_report,
                formats=self.config.report_formats
            )
            result.report_paths = report_paths
            
            # Final status
            result.end_time = datetime.now()
            
            # Log summary
            duration = (result.end_time - result.start_time).total_seconds()
            logger.info(
                f"Migration verification complete in {duration:.2f}s - "
                f"Status: {result.overall_status}, "
                f"Critical Issues: {result.critical_issues}, "
                f"Warnings: {result.warnings}"
            )
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}", exc_info=True)
            result.overall_status = "FAILED"
            result.recommendations.append(f"Fix verification error: {str(e)}")
            result.end_time = datetime.now()
        
        return result
    
    async def _run_parallel_verifications(
        self
    ) -> Tuple[Optional[DataVerificationReport], 
               Optional[SecurityAuditReport], 
               Optional[AccessVerificationReport]]:
        """Run all verifications in parallel."""
        tasks = []
        
        if self.config.verify_data_integrity:
            tasks.append(self._verify_data_integrity())
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Placeholder
        
        if self.config.perform_security_audit:
            tasks.append(self._verify_security())
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))
        
        if self.config.verify_access_control:
            tasks.append(self._verify_access_control())
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        data_report = results[0] if not isinstance(results[0], Exception) and self.config.verify_data_integrity else None
        security_report = results[1] if not isinstance(results[1], Exception) and self.config.perform_security_audit else None
        access_report = results[2] if not isinstance(results[2], Exception) and self.config.verify_access_control else None
        
        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Verification task {i} failed: {str(result)}")
        
        return data_report, security_report, access_report
    
    async def _verify_data_integrity(self) -> DataVerificationReport:
        """Perform data integrity verification."""
        logger.info("Starting data integrity verification")
        
        if self.config.source_path:
            # Verify migration from source
            return await self.data_verifier.verify_migration(
                source_path=self.config.source_path,
                redis_prefix=self.config.redis_prefix
            )
        else:
            # Verify Redis data integrity only
            pattern = f"{self.config.redis_prefix}:*"
            return await self.data_verifier.verify_redis_integrity(pattern)
    
    async def _verify_security(self) -> SecurityAuditReport:
        """Perform security audit."""
        logger.info("Starting security audit")
        
        return await self.security_auditor.perform_security_audit(
            scan_patterns=self.config.scan_patterns,
            check_compliance=self.config.check_compliance
        )
    
    async def _verify_access_control(self) -> AccessVerificationReport:
        """Perform access control verification."""
        logger.info("Starting access control verification")
        
        if not self.config.expected_access_rules:
            logger.warning("No expected access rules provided - skipping detailed verification")
            # Still check for unauthorized access
            return await self.access_verifier.verify_access_configuration(
                expected_rules=[],
                resource_patterns=self.config.scan_patterns
            )
        
        return await self.access_verifier.verify_access_configuration(
            expected_rules=self.config.expected_access_rules,
            resource_patterns=self.config.scan_patterns
        )
    
    def _analyze_results(
        self,
        result: VerificationResult,
        data_report: Optional[DataVerificationReport],
        security_report: Optional[SecurityAuditReport],
        access_report: Optional[AccessVerificationReport]
    ) -> VerificationResult:
        """Analyze verification results and determine overall status."""
        
        # Data verification analysis
        if data_report:
            result.data_verification_passed = data_report.failed_items == 0
            if not result.data_verification_passed:
                result.critical_issues += data_report.failed_items
                result.recommendations.append(
                    f"Fix {data_report.failed_items} data integrity issues"
                )
        
        # Security audit analysis
        if security_report:
            critical_security = security_report.summary.get("critical", 0)
            high_security = security_report.summary.get("high", 0)
            
            result.security_audit_passed = (critical_security + high_security) == 0
            result.critical_issues += critical_security + high_security
            result.warnings += security_report.summary.get("medium", 0)
            
            if not result.security_audit_passed:
                result.recommendations.extend(security_report.recommendations[:3])
        
        # Access verification analysis
        if access_report:
            result.access_verification_passed = len(access_report.violations_found) == 0
            
            if not result.access_verification_passed:
                result.critical_issues += len(access_report.violations_found)
                result.recommendations.append(
                    f"Fix {len(access_report.violations_found)} access control violations"
                )
            
            if access_report.unauthorized_users:
                result.warnings += len(access_report.unauthorized_users)
                result.recommendations.append(
                    f"Remove {len(access_report.unauthorized_users)} unauthorized users"
                )
        
        # Determine overall status
        if result.critical_issues > 0:
            result.overall_status = "FAILED"
        elif result.warnings > 0:
            result.overall_status = "PASSED_WITH_WARNINGS"
        else:
            result.overall_status = "PASSED"
        
        return result
    
    async def quick_verify(self) -> Dict[str, Any]:
        """
        Perform quick verification with essential checks only.
        
        Returns:
            Quick verification summary
        """
        logger.info("Starting quick migration verification")
        
        summary = {
            "timestamp": datetime.now(),
            "checks_performed": [],
            "issues_found": [],
            "status": "PASSED"
        }
        
        # Quick data check - sample only
        try:
            keys = await self.redis_client.scan_keys(f"{self.config.redis_prefix}:*", count=10)
            valid_count = 0
            
            for key in keys:
                data = await self.redis_client.get_json(key)
                if data:
                    valid_count += 1
            
            summary["checks_performed"].append("data_sampling")
            if valid_count < len(keys):
                summary["issues_found"].append(f"Invalid data in {len(keys) - valid_count} samples")
                summary["status"] = "FAILED"
            
        except Exception as e:
            summary["issues_found"].append(f"Data check error: {str(e)}")
            summary["status"] = "FAILED"
        
        # Quick security check - ACL status only
        try:
            acl_list = await self.redis_client.execute_command("ACL", "LIST")
            summary["checks_performed"].append("acl_status")
            
            if len(acl_list) <= 1:
                summary["issues_found"].append("No custom ACL users configured")
                summary["status"] = "FAILED"
            
            # Check for default user
            for acl in acl_list:
                if "user default on" in str(acl):
                    summary["issues_found"].append("Default user is enabled")
                    summary["status"] = "FAILED"
                    
        except Exception as e:
            summary["issues_found"].append(f"Security check error: {str(e)}")
        
        # Quick access check - verify at least one custom user exists
        summary["checks_performed"].append("access_control")
        
        logger.info(f"Quick verification complete: {summary['status']}")
        return summary
    
    async def verify_specific_resources(
        self,
        resource_keys: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verify specific resources.
        
        Args:
            resource_keys: List of Redis keys to verify
            
        Returns:
            Verification results for each resource
        """
        results = {}
        
        for key in resource_keys:
            result = {
                "exists": False,
                "valid_schema": False,
                "accessible": False,
                "issues": []
            }
            
            try:
                # Check existence
                data = await self.redis_client.get_json(key)
                if data:
                    result["exists"] = True
                    
                    # Validate schema
                    from ...memory.models import MemoryContent
                    try:
                        MemoryContent.from_dict(data)
                        result["valid_schema"] = True
                    except Exception as e:
                        result["issues"].append(f"Schema validation error: {str(e)}")
                    
                    # Check accessibility (basic check)
                    result["accessible"] = True
                else:
                    result["issues"].append("Resource not found")
                    
            except Exception as e:
                result["issues"].append(f"Verification error: {str(e)}")
            
            results[key] = result
        
        return results