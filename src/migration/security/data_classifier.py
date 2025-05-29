"""Data classification for security assessment."""

import re
import logging
from typing import Dict, List, Set, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class DataSensitivity(Enum):
    """데이터 민감도 레벨."""
    PUBLIC = "public"         # 공개 가능
    INTERNAL = "internal"     # 내부용
    CONFIDENTIAL = "confidential"  # 기밀
    SECRET = "secret"         # 극비
    PII = "pii"              # 개인식별정보


@dataclass
class DataClassification:
    """데이터 분류 결과."""
    path: str
    sensitivity: DataSensitivity
    pii_fields: List[str] = field(default_factory=list)
    compliance_tags: Set[str] = field(default_factory=set)
    encryption_required: bool = False
    notes: str = ""


class DataClassifier:
    """데이터 분류 및 민감도 평가 도구."""
    
    # PII 패턴 정의
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        "korean_rrn": r'\b\d{6}-[1-4]\d{6}\b',  # 주민등록번호
        "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    }
    
    # 민감 키워드
    SENSITIVE_KEYWORDS = {
        "password", "secret", "token", "key", "api_key",
        "private", "credential", "auth", "비밀번호", "인증"
    }
    
    def __init__(self):
        """초기화."""
        self.classifications: Dict[str, DataClassification] = {}
        
    def classify_data(self, path: str, content: Any) -> DataClassification:
        """데이터 분류 및 민감도 평가.
        
        Args:
            path: 데이터 경로
            content: 데이터 내용
            
        Returns:
            분류 결과
        """
        classification = DataClassification(path=path, sensitivity=DataSensitivity.PUBLIC)
        
        # 문자열로 변환
        content_str = str(content) if not isinstance(content, str) else content
        
        # PII 검사
        pii_found = self._detect_pii(content_str)
        if pii_found:
            classification.pii_fields = pii_found
            classification.sensitivity = DataSensitivity.PII
            classification.encryption_required = True
            classification.compliance_tags.add("GDPR")
            classification.compliance_tags.add("PIPA")  # 개인정보보호법
        
        # 민감 키워드 검사
        if self._contains_sensitive_keywords(content_str):
            if classification.sensitivity != DataSensitivity.PII:
                classification.sensitivity = DataSensitivity.CONFIDENTIAL
            classification.encryption_required = True
        
        # 경로 기반 분류
        path_sensitivity = self._classify_by_path(path)
        if path_sensitivity.value > classification.sensitivity.value:
            classification.sensitivity = path_sensitivity
        
        # 컴플라이언스 태그 추가
        self._add_compliance_tags(classification, content_str)
        
        # 저장
        self.classifications[path] = classification
        
        logger.info(f"Classified {path}: {classification.sensitivity.value}")
        return classification
    
    def _detect_pii(self, content: str) -> List[str]:
        """PII 데이터 검출.
        
        Args:
            content: 검사할 내용
            
        Returns:
            발견된 PII 필드 목록
        """
        found_pii = []
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, content, re.IGNORECASE):
                found_pii.append(pii_type)
                logger.warning(f"PII detected: {pii_type}")
        
        return found_pii
    
    def _contains_sensitive_keywords(self, content: str) -> bool:
        """민감 키워드 포함 여부 확인.
        
        Args:
            content: 검사할 내용
            
        Returns:
            민감 키워드 포함 여부
        """
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in self.SENSITIVE_KEYWORDS)
    
    def _classify_by_path(self, path: str) -> DataSensitivity:
        """경로 기반 민감도 분류.
        
        Args:
            path: 데이터 경로
            
        Returns:
            민감도 레벨
        """
        path_lower = path.lower()
        
        if any(term in path_lower for term in ["secret", "credential", "auth"]):
            return DataSensitivity.SECRET
        elif any(term in path_lower for term in ["private", "confidential"]):
            return DataSensitivity.CONFIDENTIAL
        elif any(term in path_lower for term in ["internal", "workspace"]):
            return DataSensitivity.INTERNAL
        
        return DataSensitivity.PUBLIC
    
    def _add_compliance_tags(self, classification: DataClassification, content: str) -> None:
        """컴플라이언스 태그 추가.
        
        Args:
            classification: 분류 객체
            content: 데이터 내용
        """
        # 의료 정보
        if any(term in content.lower() for term in ["medical", "health", "patient", "진료", "환자"]):
            classification.compliance_tags.add("HIPAA")
            classification.compliance_tags.add("의료법")
        
        # 금융 정보
        if any(term in content.lower() for term in ["financial", "payment", "account", "금융", "계좌"]):
            classification.compliance_tags.add("PCI-DSS")
            classification.compliance_tags.add("전자금융거래법")
    
    def generate_report(self) -> Dict[str, Any]:
        """분류 결과 리포트 생성.
        
        Returns:
            분류 리포트
        """
        report = {
            "total_items": len(self.classifications),
            "by_sensitivity": {},
            "pii_count": 0,
            "encryption_required": 0,
            "compliance_summary": {},
        }
        
        # 민감도별 집계
        for level in DataSensitivity:
            count = sum(1 for c in self.classifications.values() 
                       if c.sensitivity == level)
            report["by_sensitivity"][level.value] = count
        
        # PII 및 암호화 필요 데이터 집계
        for classification in self.classifications.values():
            if classification.pii_fields:
                report["pii_count"] += 1
            if classification.encryption_required:
                report["encryption_required"] += 1
            
            # 컴플라이언스 태그 집계
            for tag in classification.compliance_tags:
                report["compliance_summary"][tag] = \
                    report["compliance_summary"].get(tag, 0) + 1
        
        return report
    
    def get_high_risk_items(self) -> List[DataClassification]:
        """고위험 데이터 항목 반환.
        
        Returns:
            고위험 데이터 목록
        """
        high_risk = []
        
        for classification in self.classifications.values():
            if (classification.sensitivity in [DataSensitivity.SECRET, DataSensitivity.PII] or
                len(classification.pii_fields) > 0):
                high_risk.append(classification)
        
        return sorted(high_risk, 
                     key=lambda x: (x.sensitivity.value, len(x.pii_fields)), 
                     reverse=True)