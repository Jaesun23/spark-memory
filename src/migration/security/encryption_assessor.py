"""Encryption assessment for migration security."""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EncryptionStatus(Enum):
    """암호화 상태."""
    NOT_ENCRYPTED = "not_encrypted"
    ENCRYPTED = "encrypted"
    PARTIALLY_ENCRYPTED = "partially_encrypted"
    UNKNOWN = "unknown"


class EncryptionStrength(Enum):
    """암호화 강도."""
    NONE = 0
    WEAK = 1
    MEDIUM = 2
    STRONG = 3
    UNKNOWN = 4


@dataclass
class EncryptionAssessment:
    """암호화 평가 결과."""
    path: str
    status: EncryptionStatus
    algorithm: Optional[str] = None
    key_size: Optional[int] = None
    strength: EncryptionStrength = EncryptionStrength.NONE
    needs_reencryption: bool = False
    recommendations: List[str] = field(default_factory=list)


@dataclass
class KeyInfo:
    """암호화 키 정보."""
    key_id: str
    algorithm: str
    size: int
    created_at: str
    expires_at: Optional[str] = None
    usage_count: int = 0
    is_rotatable: bool = True


class EncryptionAssessor:
    """암호화 상태 평가 및 관리 도구."""
    
    # 약한 암호화 알고리즘
    WEAK_ALGORITHMS = {"des", "3des", "rc4", "md5"}
    
    # 권장 알고리즘
    RECOMMENDED_ALGORITHMS = {
        "aes-256-gcm": EncryptionStrength.STRONG,
        "aes-256-cbc": EncryptionStrength.STRONG,
        "aes-128-gcm": EncryptionStrength.MEDIUM,
        "chacha20-poly1305": EncryptionStrength.STRONG,
    }
    
    def __init__(self):
        """초기화."""
        self.assessments: Dict[str, EncryptionAssessment] = {}
        self.keys: Dict[str, KeyInfo] = {}
        
    def assess_encryption(self, path: str, data: Any, metadata: Dict[str, Any]) -> EncryptionAssessment:
        """데이터 암호화 상태 평가.
        
        Args:
            path: 데이터 경로
            data: 데이터 내용
            metadata: 메타데이터
            
        Returns:
            암호화 평가 결과
        """
        assessment = EncryptionAssessment(
            path=path,
            status=EncryptionStatus.NOT_ENCRYPTED
        )
        
        # 메타데이터에서 암호화 정보 확인
        if metadata.get("encrypted"):
            assessment.status = EncryptionStatus.ENCRYPTED
            assessment.algorithm = metadata.get("encryption_algorithm", "unknown")
            assessment.key_size = metadata.get("key_size")
            
            # 암호화 강도 평가
            assessment.strength = self._evaluate_strength(
                assessment.algorithm,
                assessment.key_size
            )
            
            # 재암호화 필요성 평가
            if assessment.strength in [EncryptionStrength.NONE, EncryptionStrength.WEAK]:
                assessment.needs_reencryption = True
                assessment.recommendations.append(
                    f"Upgrade from {assessment.algorithm} to AES-256-GCM"
                )
        else:
            # 데이터 패턴으로 암호화 여부 추측
            if self._looks_encrypted(data):
                assessment.status = EncryptionStatus.UNKNOWN
                assessment.recommendations.append(
                    "Data appears encrypted but metadata is missing"
                )
        
        # 부분 암호화 검사
        if assessment.status == EncryptionStatus.ENCRYPTED:
            if self._has_unencrypted_fields(data, metadata):
                assessment.status = EncryptionStatus.PARTIALLY_ENCRYPTED
                assessment.recommendations.append(
                    "Some fields are not encrypted"
                )
        
        self.assessments[path] = assessment
        return assessment
    
    def register_key(self, key_info: KeyInfo) -> None:
        """암호화 키 등록.
        
        Args:
            key_info: 키 정보
        """
        self.keys[key_info.key_id] = key_info
        logger.info(f"Registered key: {key_info.key_id}")
    
    def evaluate_key_management(self) -> Dict[str, Any]:
        """키 관리 시스템 평가.
        
        Returns:
            키 관리 평가 결과
        """
        evaluation = {
            "total_keys": len(self.keys),
            "expired_keys": 0,
            "weak_keys": 0,
            "rotation_needed": 0,
            "recommendations": []
        }
        
        from datetime import datetime
        current_time = datetime.now()
        
        for key in self.keys.values():
            # 만료된 키
            if key.expires_at:
                expires = datetime.fromisoformat(key.expires_at)
                if expires < current_time:
                    evaluation["expired_keys"] += 1
            
            # 약한 키
            strength = self._evaluate_strength(key.algorithm, key.size)
            if strength in [EncryptionStrength.NONE, EncryptionStrength.WEAK]:
                evaluation["weak_keys"] += 1
            
            # 순환 필요
            if key.is_rotatable and key.usage_count > 1000000:  # 백만 회 사용
                evaluation["rotation_needed"] += 1
        
        # 권장사항 생성
        if evaluation["expired_keys"] > 0:
            evaluation["recommendations"].append(
                f"Remove {evaluation['expired_keys']} expired keys"
            )
        
        if evaluation["weak_keys"] > 0:
            evaluation["recommendations"].append(
                f"Replace {evaluation['weak_keys']} weak encryption keys"
            )
        
        if evaluation["rotation_needed"] > 0:
            evaluation["recommendations"].append(
                f"Rotate {evaluation['rotation_needed']} heavily used keys"
            )
        
        return evaluation
    
    def plan_reencryption(self) -> List[Tuple[str, str, str]]:
        """재암호화 계획 생성.
        
        Returns:
            (경로, 현재 알고리즘, 권장 알고리즘) 튜플 목록
        """
        reencryption_plan = []
        
        for assessment in self.assessments.values():
            if assessment.needs_reencryption:
                current_algo = assessment.algorithm or "none"
                recommended_algo = "aes-256-gcm"  # 기본 권장
                
                reencryption_plan.append((
                    assessment.path,
                    current_algo,
                    recommended_algo
                ))
        
        # 우선순위 정렬: 암호화되지 않은 것 > 약한 암호화
        reencryption_plan.sort(
            key=lambda x: (
                x[1] == "none",  # 암호화 안됨 우선
                x[1] in self.WEAK_ALGORITHMS  # 약한 알고리즘 다음
            ),
            reverse=True
        )
        
        return reencryption_plan
    
    def _evaluate_strength(self, algorithm: str, key_size: Optional[int]) -> EncryptionStrength:
        """암호화 강도 평가.
        
        Args:
            algorithm: 암호화 알고리즘
            key_size: 키 크기
            
        Returns:
            암호화 강도
        """
        if not algorithm:
            return EncryptionStrength.NONE
        
        algo_lower = algorithm.lower()
        
        # 약한 알고리즘
        if algo_lower in self.WEAK_ALGORITHMS:
            return EncryptionStrength.WEAK
        
        # 권장 알고리즘
        if algo_lower in self.RECOMMENDED_ALGORITHMS:
            return self.RECOMMENDED_ALGORITHMS[algo_lower]
        
        # 키 크기 기반 평가
        if key_size:
            if key_size >= 256:
                return EncryptionStrength.STRONG
            elif key_size >= 128:
                return EncryptionStrength.MEDIUM
            else:
                return EncryptionStrength.WEAK
        
        return EncryptionStrength.UNKNOWN
    
    def _looks_encrypted(self, data: Any) -> bool:
        """데이터가 암호화되어 보이는지 확인.
        
        Args:
            data: 검사할 데이터
            
        Returns:
            암호화 여부 추측
        """
        if isinstance(data, str):
            # Base64 인코딩 패턴
            if len(data) % 4 == 0 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in data):
                return True
            
            # 높은 엔트로피 (랜덤성)
            if len(set(data)) / len(data) > 0.7:  # 70% 이상 유니크 문자
                return True
        
        elif isinstance(data, bytes):
            # 바이너리 데이터는 암호화일 가능성 높음
            return True
        
        return False
    
    def _has_unencrypted_fields(self, data: Any, metadata: Dict[str, Any]) -> bool:
        """암호화되지 않은 필드 존재 여부 확인.
        
        Args:
            data: 데이터
            metadata: 메타데이터
            
        Returns:
            미암호화 필드 존재 여부
        """
        if isinstance(data, dict):
            encrypted_fields = set(metadata.get("encrypted_fields", []))
            all_fields = set(data.keys())
            
            # 메타데이터 필드 제외
            data_fields = all_fields - {"_id", "_metadata", "_encrypted"}
            
            if encrypted_fields and data_fields - encrypted_fields:
                return True
        
        return False
    
    def generate_security_report(self) -> Dict[str, Any]:
        """보안 평가 리포트 생성.
        
        Returns:
            보안 리포트
        """
        report = {
            "total_assessed": len(self.assessments),
            "encryption_status": {},
            "strength_distribution": {},
            "reencryption_needed": 0,
            "key_management": self.evaluate_key_management(),
            "high_risk_items": []
        }
        
        # 상태별 집계
        for status in EncryptionStatus:
            count = sum(1 for a in self.assessments.values() 
                       if a.status == status)
            report["encryption_status"][status.value] = count
        
        # 강도별 집계
        for strength in EncryptionStrength:
            count = sum(1 for a in self.assessments.values() 
                       if a.strength == strength)
            report["strength_distribution"][strength.name] = count
        
        # 재암호화 필요 항목
        report["reencryption_needed"] = sum(
            1 for a in self.assessments.values() 
            if a.needs_reencryption
        )
        
        # 고위험 항목
        for assessment in self.assessments.values():
            if (assessment.status == EncryptionStatus.NOT_ENCRYPTED or
                assessment.strength in [EncryptionStrength.NONE, EncryptionStrength.WEAK]):
                report["high_risk_items"].append({
                    "path": assessment.path,
                    "status": assessment.status.value,
                    "strength": assessment.strength.name,
                    "recommendations": assessment.recommendations
                })
        
        return report