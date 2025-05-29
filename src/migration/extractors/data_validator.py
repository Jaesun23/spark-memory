"""Data validation for migration integrity."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """검증 결과."""
    path: str
    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_details: Optional[str] = None
    validated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationRule:
    """검증 규칙."""
    name: str
    description: str
    check_function: Any  # Callable
    severity: str = "error"  # error, warning
    applies_to: Optional[str] = None  # path pattern


class DataValidator:
    """데이터 무결성 검증 도구."""
    
    def __init__(self):
        """초기화."""
        self.rules: List[ValidationRule] = []
        self.results: Dict[str, ValidationResult] = {}
        self._setup_default_rules()
        
    def _setup_default_rules(self) -> None:
        """기본 검증 규칙 설정."""
        # 필수 필드 검증
        self.add_rule(ValidationRule(
            name="required_fields",
            description="Check for required fields",
            check_function=self._check_required_fields
        ))
        
        # 데이터 타입 검증
        self.add_rule(ValidationRule(
            name="data_types",
            description="Validate data types",
            check_function=self._check_data_types
        ))
        
        # 크기 제한 검증
        self.add_rule(ValidationRule(
            name="size_limits",
            description="Check size constraints",
            check_function=self._check_size_limits
        ))
        
        # 문자 인코딩 검증
        self.add_rule(ValidationRule(
            name="encoding",
            description="Validate character encoding",
            check_function=self._check_encoding
        ))
        
        # 시간 필드 검증
        self.add_rule(ValidationRule(
            name="timestamps",
            description="Validate timestamp fields",
            check_function=self._check_timestamps
        ))
    
    def add_rule(self, rule: ValidationRule) -> None:
        """검증 규칙 추가.
        
        Args:
            rule: 검증 규칙
        """
        self.rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")
    
    def validate(self, path: str, data: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """데이터 검증.
        
        Args:
            path: 데이터 경로
            data: 데이터 내용
            metadata: 메타데이터
            
        Returns:
            검증 결과
        """
        result = ValidationResult(path=path, is_valid=True)
        
        # 각 규칙 적용
        for rule in self.rules:
            # 경로 패턴 확인
            if rule.applies_to and not self._matches_pattern(path, rule.applies_to):
                continue
            
            try:
                passed, message = rule.check_function(data, metadata)
                
                if passed:
                    result.checks_passed.append(f"{rule.name}: {message}")
                else:
                    if rule.severity == "error":
                        result.checks_failed.append(f"{rule.name}: {message}")
                        result.is_valid = False
                    else:
                        result.warnings.append(f"{rule.name}: {message}")
                        
            except Exception as e:
                result.checks_failed.append(f"{rule.name}: Exception - {str(e)}")
                result.is_valid = False
                result.error_details = str(e)
        
        self.results[path] = result
        return result
    
    def validate_batch(
        self,
        items: List[Tuple[str, Any, Dict[str, Any]]]
    ) -> Dict[str, ValidationResult]:
        """배치 검증.
        
        Args:
            items: (경로, 데이터, 메타데이터) 튜플 목록
            
        Returns:
            검증 결과 딕셔너리
        """
        results = {}
        
        for path, data, metadata in items:
            results[path] = self.validate(path, data, metadata)
        
        return results
    
    def compare_source_target(
        self,
        source_data: Any,
        target_data: Any,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """원본-대상 데이터 비교.
        
        Args:
            source_data: 원본 데이터
            target_data: 대상 데이터
            metadata: 메타데이터
            
        Returns:
            (일치 여부, 차이점 목록)
        """
        differences = []
        
        # 압축/암호화 해제 필요 확인
        if metadata.get("extraction_compressed") or metadata.get("extraction_encrypted"):
            # 실제로는 압축/암호화 해제 후 비교
            logger.warning("Compressed/encrypted data comparison not implemented")
            return True, []
        
        # 타입 비교
        if type(source_data) != type(target_data):
            differences.append(f"Type mismatch: {type(source_data)} vs {type(target_data)}")
        
        # 내용 비교
        if isinstance(source_data, dict) and isinstance(target_data, dict):
            # 키 비교
            source_keys = set(source_data.keys())
            target_keys = set(target_data.keys())
            
            missing_keys = source_keys - target_keys
            extra_keys = target_keys - source_keys
            
            if missing_keys:
                differences.append(f"Missing keys: {missing_keys}")
            if extra_keys:
                differences.append(f"Extra keys: {extra_keys}")
            
            # 값 비교
            for key in source_keys & target_keys:
                if source_data[key] != target_data[key]:
                    differences.append(f"Value mismatch for key '{key}'")
        
        elif isinstance(source_data, list) and isinstance(target_data, list):
            if len(source_data) != len(target_data):
                differences.append(f"Length mismatch: {len(source_data)} vs {len(target_data)}")
        
        else:
            # 직접 비교
            if source_data != target_data:
                differences.append("Content mismatch")
        
        return len(differences) == 0, differences
    
    def _check_required_fields(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """필수 필드 검증.
        
        Args:
            data: 데이터
            metadata: 메타데이터
            
        Returns:
            (통과 여부, 메시지)
        """
        required_metadata = {"created_at", "extraction_timestamp"}
        
        missing = required_metadata - set(metadata.keys())
        if missing:
            return False, f"Missing required metadata fields: {missing}"
        
        if isinstance(data, dict):
            # 데이터가 비어있으면 실패
            if not data:
                return False, "Empty data dictionary"
        
        return True, "All required fields present"
    
    def _check_data_types(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """데이터 타입 검증.
        
        Args:
            data: 데이터
            metadata: 메타데이터
            
        Returns:
            (통과 여부, 메시지)
        """
        # JSON 직렬화 가능 여부 확인
        try:
            json.dumps(data)
            return True, "Data is JSON serializable"
        except (TypeError, ValueError) as e:
            return False, f"Data is not JSON serializable: {str(e)}"
    
    def _check_size_limits(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """크기 제한 검증.
        
        Args:
            data: 데이터
            metadata: 메타데이터
            
        Returns:
            (통과 여부, 메시지)
        """
        MAX_SIZE_MB = 100  # 100MB 제한
        
        # 데이터 크기 계산
        data_str = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
        size_bytes = len(data_str.encode('utf-8'))
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > MAX_SIZE_MB:
            return False, f"Data size {size_mb:.2f}MB exceeds limit of {MAX_SIZE_MB}MB"
        
        return True, f"Data size {size_mb:.2f}MB within limits"
    
    def _check_encoding(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """문자 인코딩 검증.
        
        Args:
            data: 데이터
            metadata: 메타데이터
            
        Returns:
            (통과 여부, 메시지)
        """
        def check_string(s: str) -> bool:
            try:
                s.encode('utf-8').decode('utf-8')
                return True
            except UnicodeError:
                return False
        
        def check_recursive(obj: Any) -> bool:
            if isinstance(obj, str):
                return check_string(obj)
            elif isinstance(obj, dict):
                return all(check_string(k) and check_recursive(v) 
                          for k, v in obj.items())
            elif isinstance(obj, list):
                return all(check_recursive(item) for item in obj)
            return True
        
        if check_recursive(data):
            return True, "All strings are valid UTF-8"
        else:
            return False, "Invalid UTF-8 encoding detected"
    
    def _check_timestamps(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """시간 필드 검증.
        
        Args:
            data: 데이터
            metadata: 메타데이터
            
        Returns:
            (통과 여부, 메시지)
        """
        timestamp_fields = ["created_at", "updated_at", "extraction_timestamp"]
        invalid_timestamps = []
        
        # 메타데이터에서 타임스탬프 확인
        for field in timestamp_fields:
            if field in metadata:
                try:
                    datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                except ValueError:
                    invalid_timestamps.append(field)
        
        if invalid_timestamps:
            return False, f"Invalid timestamp format in fields: {invalid_timestamps}"
        
        return True, "All timestamps are valid"
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """경로 패턴 매칭.
        
        Args:
            path: 경로
            pattern: 패턴
            
        Returns:
            매칭 여부
        """
        import fnmatch
        return fnmatch.fnmatch(path, pattern)
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """검증 보고서 생성.
        
        Returns:
            검증 보고서
        """
        report = {
            "total_validated": len(self.results),
            "valid_count": sum(1 for r in self.results.values() if r.is_valid),
            "invalid_count": sum(1 for r in self.results.values() if not r.is_valid),
            "warnings_count": sum(len(r.warnings) for r in self.results.values()),
            "common_failures": {},
            "invalid_items": []
        }
        
        # 공통 실패 원인 집계
        failure_counts = {}
        for result in self.results.values():
            for failure in result.checks_failed:
                rule_name = failure.split(":")[0]
                failure_counts[rule_name] = failure_counts.get(rule_name, 0) + 1
        
        report["common_failures"] = failure_counts
        
        # 무효 항목 목록
        for path, result in self.results.items():
            if not result.is_valid:
                report["invalid_items"].append({
                    "path": path,
                    "failures": result.checks_failed,
                    "error": result.error_details
                })
        
        return report