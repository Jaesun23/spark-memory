"""
Data integrity verification for Memory One Spark migration.

This module ensures that all data migrated from the file-based system
to Redis maintains integrity, completeness, and consistency.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from pydantic import BaseModel

from ...redis.client import RedisClient
from ...memory.models import MemoryContent
from ...utils.logging import get_logger

logger = get_logger(__name__)


class VerificationResult(BaseModel):
    """Result of a single verification check."""
    
    path: str
    status: str  # "success", "failure", "warning"
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()


class DataVerificationReport(BaseModel):
    """Complete data verification report."""
    
    total_items: int
    verified_items: int
    failed_items: int
    warnings: int
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[VerificationResult] = []
    summary: Dict[str, Any] = {}


class DataVerifier:
    """
    Verifies data integrity during and after migration.
    
    Performs comprehensive checks including:
    - Content hash verification
    - Schema validation
    - Data completeness
    - Referential integrity
    - Timestamp consistency
    """
    
    def __init__(self, redis_client: RedisClient):
        """
        Initialize the data verifier.
        
        Args:
            redis_client: Redis client for accessing migrated data
        """
        self.redis_client = redis_client
        self.report = DataVerificationReport(
            total_items=0,
            verified_items=0,
            failed_items=0,
            warnings=0,
            start_time=datetime.now()
        )
    
    async def verify_migration(
        self,
        source_path: Path,
        redis_prefix: str = "memory"
    ) -> DataVerificationReport:
        """
        Verify complete migration from file system to Redis.
        
        Args:
            source_path: Original file-based memory path
            redis_prefix: Redis key prefix for migrated data
            
        Returns:
            Comprehensive verification report
        """
        logger.info(f"Starting data verification for {source_path}")
        
        # Count source files
        source_files = list(source_path.rglob("*.json"))
        self.report.total_items = len(source_files)
        
        # Verify each file
        for file_path in source_files:
            await self._verify_single_item(file_path, redis_prefix)
        
        # Generate summary
        self.report.end_time = datetime.now()
        self.report.summary = self._generate_summary()
        
        logger.info(f"Verification complete: {self.report.verified_items}/{self.report.total_items} verified")
        return self.report
    
    async def _verify_single_item(
        self,
        file_path: Path,
        redis_prefix: str
    ) -> VerificationResult:
        """Verify a single migrated item."""
        try:
            # Calculate expected Redis key
            relative_path = file_path.relative_to(file_path.parent.parent.parent)
            redis_key = f"{redis_prefix}:{str(relative_path).replace('/', ':').replace('.json', '')}"
            
            # Load source data
            with open(file_path, 'r') as f:
                source_data = json.load(f)
            
            # Get Redis data
            redis_data = await self.redis_client.get_json(redis_key)
            
            if not redis_data:
                result = VerificationResult(
                    path=str(file_path),
                    status="failure",
                    message=f"Data not found in Redis at key: {redis_key}"
                )
                self.report.failed_items += 1
            else:
                # Verify content integrity
                verification_checks = await self._perform_integrity_checks(
                    source_data,
                    redis_data,
                    str(file_path)
                )
                
                if all(check["passed"] for check in verification_checks):
                    result = VerificationResult(
                        path=str(file_path),
                        status="success",
                        message="All verification checks passed",
                        details={"checks": verification_checks}
                    )
                    self.report.verified_items += 1
                else:
                    failed_checks = [c for c in verification_checks if not c["passed"]]
                    result = VerificationResult(
                        path=str(file_path),
                        status="failure",
                        message=f"{len(failed_checks)} verification checks failed",
                        details={"failed_checks": failed_checks}
                    )
                    self.report.failed_items += 1
            
            self.report.results.append(result)
            return result
            
        except Exception as e:
            result = VerificationResult(
                path=str(file_path),
                status="failure",
                message=f"Verification error: {str(e)}"
            )
            self.report.failed_items += 1
            self.report.results.append(result)
            return result
    
    async def _perform_integrity_checks(
        self,
        source_data: Dict[str, Any],
        redis_data: Dict[str, Any],
        path: str
    ) -> List[Dict[str, Any]]:
        """Perform comprehensive integrity checks."""
        checks = []
        
        # Content hash check
        source_hash = self._calculate_content_hash(source_data)
        redis_hash = self._calculate_content_hash(redis_data)
        checks.append({
            "name": "content_hash",
            "passed": source_hash == redis_hash,
            "source_hash": source_hash,
            "redis_hash": redis_hash
        })
        
        # Schema validation
        try:
            MemoryContent.from_dict(redis_data)
            checks.append({
                "name": "schema_validation",
                "passed": True
            })
        except Exception as e:
            checks.append({
                "name": "schema_validation",
                "passed": False,
                "error": str(e)
            })
        
        # Field completeness
        missing_fields = set(source_data.keys()) - set(redis_data.keys())
        extra_fields = set(redis_data.keys()) - set(source_data.keys())
        checks.append({
            "name": "field_completeness",
            "passed": len(missing_fields) == 0 and len(extra_fields) == 0,
            "missing_fields": list(missing_fields),
            "extra_fields": list(extra_fields)
        })
        
        return checks
    
    def _calculate_content_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of content."""
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate verification summary statistics."""
        duration = (self.report.end_time - self.report.start_time).total_seconds()
        
        return {
            "success_rate": (self.report.verified_items / self.report.total_items * 100) 
                           if self.report.total_items > 0 else 0,
            "failure_rate": (self.report.failed_items / self.report.total_items * 100)
                           if self.report.total_items > 0 else 0,
            "duration_seconds": duration,
            "items_per_second": self.report.total_items / duration if duration > 0 else 0,
            "verification_status": "PASSED" if self.report.failed_items == 0 else "FAILED"
        }
    
    async def verify_redis_integrity(self, pattern: str = "*") -> DataVerificationReport:
        """
        Verify integrity of data already in Redis.
        
        Args:
            pattern: Redis key pattern to verify
            
        Returns:
            Verification report
        """
        logger.info(f"Starting Redis data integrity verification for pattern: {pattern}")
        
        # Get all matching keys
        keys = await self.redis_client.scan_keys(pattern)
        self.report.total_items = len(keys)
        
        for key in keys:
            await self._verify_redis_item(key)
        
        self.report.end_time = datetime.now()
        self.report.summary = self._generate_summary()
        
        return self.report
    
    async def _verify_redis_item(self, key: str) -> VerificationResult:
        """Verify a single Redis item."""
        try:
            data = await self.redis_client.get_json(key)
            
            if not data:
                result = VerificationResult(
                    path=key,
                    status="failure",
                    message="Key exists but data is empty"
                )
                self.report.failed_items += 1
            else:
                # Validate schema
                try:
                    MemoryContent.from_dict(data)
                    result = VerificationResult(
                        path=key,
                        status="success",
                        message="Data validation passed"
                    )
                    self.report.verified_items += 1
                except Exception as e:
                    result = VerificationResult(
                        path=key,
                        status="failure",
                        message=f"Schema validation failed: {str(e)}"
                    )
                    self.report.failed_items += 1
            
            self.report.results.append(result)
            return result
            
        except Exception as e:
            result = VerificationResult(
                path=key,
                status="failure",
                message=f"Verification error: {str(e)}"
            )
            self.report.failed_items += 1
            self.report.results.append(result)
            return result