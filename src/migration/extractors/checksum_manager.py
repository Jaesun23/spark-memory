"""Checksum management for data integrity."""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class ChecksumRecord:
    """체크섬 기록."""
    path: str
    algorithm: str
    checksum: str
    size_bytes: int
    created_at: datetime
    verified_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChecksumManager:
    """체크섬 기반 무결성 관리."""
    
    SUPPORTED_ALGORITHMS = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha384": hashlib.sha384,
        "sha512": hashlib.sha512,
        "blake2b": hashlib.blake2b,
        "blake2s": hashlib.blake2s,
    }
    
    def __init__(self, default_algorithm: str = "sha256"):
        """초기화.
        
        Args:
            default_algorithm: 기본 해시 알고리즘
        """
        if default_algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {default_algorithm}")
        
        self.default_algorithm = default_algorithm
        self.records: Dict[str, ChecksumRecord] = {}
        self._anomalies: List[Dict[str, Any]] = []
        
    def calculate_checksum(
        self,
        data: Any,
        algorithm: Optional[str] = None
    ) -> Tuple[str, int]:
        """체크섬 계산.
        
        Args:
            data: 데이터
            algorithm: 해시 알고리즘
            
        Returns:
            (체크섬, 데이터 크기)
        """
        algo = algorithm or self.default_algorithm
        if algo not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algo}")
        
        # 데이터를 바이트로 변환
        if isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, (dict, list)):
            # JSON으로 정렬된 문자열 생성
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
        
        # 체크섬 계산
        hasher = self.SUPPORTED_ALGORITHMS[algo]()
        hasher.update(data_bytes)
        
        return hasher.hexdigest(), len(data_bytes)
    
    def create_record(
        self,
        path: str,
        data: Any,
        algorithm: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChecksumRecord:
        """체크섬 레코드 생성.
        
        Args:
            path: 데이터 경로
            data: 데이터
            algorithm: 해시 알고리즘
            metadata: 추가 메타데이터
            
        Returns:
            생성된 레코드
        """
        algo = algorithm or self.default_algorithm
        checksum, size = self.calculate_checksum(data, algo)
        
        record = ChecksumRecord(
            path=path,
            algorithm=algo,
            checksum=checksum,
            size_bytes=size,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.records[path] = record
        logger.info(f"Created checksum record for {path}: {checksum[:16]}...")
        
        return record
    
    def verify_data(
        self,
        path: str,
        data: Any,
        expected_checksum: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """데이터 무결성 검증.
        
        Args:
            path: 데이터 경로
            data: 검증할 데이터
            expected_checksum: 예상 체크섬 (없으면 레코드에서 조회)
            
        Returns:
            (검증 성공 여부, 오류 메시지)
        """
        # 예상 체크섬 확인
        if not expected_checksum:
            if path not in self.records:
                return False, f"No checksum record found for {path}"
            expected_checksum = self.records[path].checksum
            algorithm = self.records[path].algorithm
        else:
            # 알고리즘 추측 (체크섬 길이 기반)
            algorithm = self._guess_algorithm(expected_checksum)
        
        # 체크섬 재계산
        try:
            calculated_checksum, size = self.calculate_checksum(data, algorithm)
        except Exception as e:
            return False, f"Failed to calculate checksum: {str(e)}"
        
        # 비교
        if calculated_checksum != expected_checksum:
            self._record_anomaly(path, "checksum_mismatch", {
                "expected": expected_checksum,
                "calculated": calculated_checksum
            })
            return False, f"Checksum mismatch: expected {expected_checksum[:16]}..., got {calculated_checksum[:16]}..."
        
        # 레코드 업데이트
        if path in self.records:
            self.records[path].verified_at = datetime.now()
        
        return True, None
    
    def verify_batch(
        self,
        items: List[Tuple[str, Any, str]]
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """배치 검증.
        
        Args:
            items: (경로, 데이터, 예상 체크섬) 튜플 목록
            
        Returns:
            경로별 검증 결과
        """
        results = {}
        
        for path, data, expected_checksum in items:
            results[path] = self.verify_data(path, data, expected_checksum)
        
        return results
    
    async def save_manifest(self, filepath: str) -> None:
        """체크섬 매니페스트 저장.
        
        Args:
            filepath: 저장할 파일 경로
        """
        manifest = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "default_algorithm": self.default_algorithm,
            "records": {}
        }
        
        for path, record in self.records.items():
            manifest["records"][path] = {
                "algorithm": record.algorithm,
                "checksum": record.checksum,
                "size_bytes": record.size_bytes,
                "created_at": record.created_at.isoformat(),
                "verified_at": record.verified_at.isoformat() if record.verified_at else None,
                "metadata": record.metadata
            }
        
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(manifest, indent=2))
        
        logger.info(f"Saved checksum manifest to {filepath}")
    
    async def load_manifest(self, filepath: str) -> None:
        """체크섬 매니페스트 로드.
        
        Args:
            filepath: 로드할 파일 경로
        """
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            content = await f.read()
            manifest = json.loads(content)
        
        self.default_algorithm = manifest.get("default_algorithm", "sha256")
        
        for path, record_data in manifest.get("records", {}).items():
            record = ChecksumRecord(
                path=path,
                algorithm=record_data["algorithm"],
                checksum=record_data["checksum"],
                size_bytes=record_data["size_bytes"],
                created_at=datetime.fromisoformat(record_data["created_at"]),
                verified_at=datetime.fromisoformat(record_data["verified_at"]) 
                           if record_data.get("verified_at") else None,
                metadata=record_data.get("metadata", {})
            )
            self.records[path] = record
        
        logger.info(f"Loaded {len(self.records)} checksum records from {filepath}")
    
    def compare_checksums(
        self,
        source_checksum: str,
        target_checksum: str,
        algorithm: Optional[str] = None
    ) -> bool:
        """체크섬 비교.
        
        Args:
            source_checksum: 원본 체크섬
            target_checksum: 대상 체크섬
            algorithm: 알고리즘 (검증용)
            
        Returns:
            일치 여부
        """
        # 대소문자 무시 비교
        return source_checksum.lower() == target_checksum.lower()
    
    def detect_duplicates(self) -> Dict[str, List[str]]:
        """중복 데이터 탐지.
        
        Returns:
            체크섬별 중복 경로 목록
        """
        checksum_to_paths = {}
        
        for path, record in self.records.items():
            checksum = record.checksum
            if checksum not in checksum_to_paths:
                checksum_to_paths[checksum] = []
            checksum_to_paths[checksum].append(path)
        
        # 중복만 필터링
        duplicates = {
            checksum: paths 
            for checksum, paths in checksum_to_paths.items() 
            if len(paths) > 1
        }
        
        return duplicates
    
    def _guess_algorithm(self, checksum: str) -> str:
        """체크섬 길이로 알고리즘 추측.
        
        Args:
            checksum: 체크섬 문자열
            
        Returns:
            추측된 알고리즘
        """
        length_to_algorithm = {
            32: "md5",
            40: "sha1",
            64: "sha256",
            96: "sha384",
            128: "sha512",
        }
        
        checksum_length = len(checksum)
        return length_to_algorithm.get(checksum_length, self.default_algorithm)
    
    def _record_anomaly(self, path: str, anomaly_type: str, details: Dict[str, Any]) -> None:
        """이상 항목 기록.
        
        Args:
            path: 데이터 경로
            anomaly_type: 이상 유형
            details: 상세 정보
        """
        anomaly = {
            "timestamp": datetime.now().isoformat(),
            "path": path,
            "type": anomaly_type,
            "details": details
        }
        
        self._anomalies.append(anomaly)
        logger.warning(f"Anomaly detected: {anomaly_type} for {path}")
    
    def get_integrity_report(self) -> Dict[str, Any]:
        """무결성 보고서 생성.
        
        Returns:
            무결성 보고서
        """
        verified_count = sum(1 for r in self.records.values() if r.verified_at)
        
        report = {
            "total_records": len(self.records),
            "verified_count": verified_count,
            "unverified_count": len(self.records) - verified_count,
            "algorithm_distribution": {},
            "size_statistics": {
                "total_bytes": sum(r.size_bytes for r in self.records.values()),
                "average_bytes": sum(r.size_bytes for r in self.records.values()) / len(self.records) if self.records else 0,
                "max_bytes": max((r.size_bytes for r in self.records.values()), default=0),
                "min_bytes": min((r.size_bytes for r in self.records.values()), default=0),
            },
            "duplicates": len(self.detect_duplicates()),
            "anomalies": len(self._anomalies),
            "anomaly_details": self._anomalies[-10:]  # 최근 10개
        }
        
        # 알고리즘별 분포
        for record in self.records.values():
            algo = record.algorithm
            report["algorithm_distribution"][algo] = \
                report["algorithm_distribution"].get(algo, 0) + 1
        
        return report