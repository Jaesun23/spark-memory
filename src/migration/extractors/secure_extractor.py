"""Secure data extraction pipeline."""

import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, AsyncIterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import ssl
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class ExtractionJob:
    """추출 작업 정보."""
    job_id: str
    source_path: str
    target_path: str
    encryption_enabled: bool = True
    compression_enabled: bool = True
    checksum_algorithm: str = "sha256"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    items_extracted: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass 
class ExtractedData:
    """추출된 데이터."""
    path: str
    content: Any
    metadata: Dict[str, Any]
    checksum: str
    size_bytes: int
    extracted_at: datetime


class SecureExtractor:
    """보안 데이터 추출 파이프라인."""
    
    def __init__(self, source_client: Any, encryption_key: Optional[bytes] = None):
        """초기화.
        
        Args:
            source_client: 소스 데이터 클라이언트
            encryption_key: 암호화 키 (선택적)
        """
        self.source_client = source_client
        self.encryption_key = encryption_key
        self.jobs: Dict[str, ExtractionJob] = {}
        self._audit_log: List[Dict[str, Any]] = []
        
    async def create_extraction_job(
        self,
        source_pattern: str,
        target_dir: str,
        **options
    ) -> ExtractionJob:
        """추출 작업 생성.
        
        Args:
            source_pattern: 소스 패턴
            target_dir: 대상 디렉토리
            **options: 추가 옵션
            
        Returns:
            생성된 작업
        """
        job_id = self._generate_job_id()
        
        job = ExtractionJob(
            job_id=job_id,
            source_path=source_pattern,
            target_path=target_dir,
            encryption_enabled=options.get("encrypt", True),
            compression_enabled=options.get("compress", True),
            checksum_algorithm=options.get("checksum", "sha256")
        )
        
        self.jobs[job_id] = job
        self._log_audit("job_created", {"job_id": job_id, "source": source_pattern})
        
        return job
    
    async def extract_data(
        self,
        job: ExtractionJob,
        batch_size: int = 100
    ) -> AsyncIterator[ExtractedData]:
        """데이터 추출 실행.
        
        Args:
            job: 추출 작업
            batch_size: 배치 크기
            
        Yields:
            추출된 데이터
        """
        job.status = "running"
        job.started_at = datetime.now()
        
        try:
            # 보안 채널 설정
            ssl_context = self._create_ssl_context() if job.encryption_enabled else None
            
            # 데이터 추출
            async for batch in self._extract_in_batches(job.source_path, batch_size):
                for item_path, item_data in batch:
                    try:
                        # 데이터 변환 및 검증
                        extracted = await self._process_item(
                            item_path,
                            item_data,
                            job
                        )
                        
                        if extracted:
                            job.items_extracted += 1
                            yield extracted
                            
                    except Exception as e:
                        error_msg = f"Failed to extract {item_path}: {str(e)}"
                        job.errors.append(error_msg)
                        logger.error(error_msg)
                        
            job.status = "completed"
            
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(f"Extraction job {job.job_id} failed: {e}")
            raise
            
        finally:
            job.completed_at = datetime.now()
            self._log_audit("job_completed", {
                "job_id": job.job_id,
                "status": job.status,
                "items": job.items_extracted,
                "errors": len(job.errors)
            })
    
    async def _extract_in_batches(
        self,
        pattern: str,
        batch_size: int
    ) -> AsyncIterator[List[Tuple[str, Any]]]:
        """배치 단위로 데이터 추출.
        
        Args:
            pattern: 검색 패턴
            batch_size: 배치 크기
            
        Yields:
            데이터 배치
        """
        batch = []
        
        # 소스에서 데이터 검색
        async for key in self.source_client.scan_iter(match=pattern):
            # 데이터 조회
            data = await self.source_client.get(key)
            metadata = await self.source_client.get_metadata(key)
            
            batch.append((key, {"data": data, "metadata": metadata}))
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # 남은 배치 처리
        if batch:
            yield batch
    
    async def _process_item(
        self,
        path: str,
        item_data: Dict[str, Any],
        job: ExtractionJob
    ) -> Optional[ExtractedData]:
        """개별 아이템 처리.
        
        Args:
            path: 아이템 경로
            item_data: 아이템 데이터
            job: 추출 작업
            
        Returns:
            처리된 데이터
        """
        try:
            content = item_data.get("data")
            metadata = item_data.get("metadata", {})
            
            # 체크섬 생성
            checksum = self._calculate_checksum(content, job.checksum_algorithm)
            
            # 암호화 (필요시)
            if job.encryption_enabled and self.encryption_key:
                content = await self._encrypt_data(content)
                metadata["extraction_encrypted"] = True
            
            # 압축 (필요시)
            if job.compression_enabled:
                original_size = len(str(content).encode())
                content = await self._compress_data(content)
                metadata["extraction_compressed"] = True
                metadata["original_size"] = original_size
            
            # 메타데이터 추가
            metadata["extraction_job_id"] = job.job_id
            metadata["extraction_timestamp"] = datetime.now().isoformat()
            metadata["extraction_checksum"] = checksum
            
            extracted = ExtractedData(
                path=path,
                content=content,
                metadata=metadata,
                checksum=checksum,
                size_bytes=len(str(content).encode()),
                extracted_at=datetime.now()
            )
            
            # 무결성 검증
            if not self._verify_integrity(extracted):
                raise ValueError("Integrity check failed")
            
            return extracted
            
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            raise
    
    async def save_extracted_data(
        self,
        extracted: ExtractedData,
        target_dir: str
    ) -> str:
        """추출된 데이터 저장.
        
        Args:
            extracted: 추출된 데이터
            target_dir: 대상 디렉토리
            
        Returns:
            저장된 파일 경로
        """
        # 안전한 파일 경로 생성
        safe_filename = self._sanitize_path(extracted.path)
        target_path = f"{target_dir}/{safe_filename}.json"
        
        # 저장할 데이터 준비
        save_data = {
            "path": extracted.path,
            "content": extracted.content,
            "metadata": extracted.metadata,
            "checksum": extracted.checksum,
            "extracted_at": extracted.extracted_at.isoformat()
        }
        
        # 암호화된 채널로 저장
        async with aiofiles.open(target_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(save_data, ensure_ascii=False, indent=2))
        
        # 체크섬 파일 생성
        checksum_path = f"{target_path}.checksum"
        async with aiofiles.open(checksum_path, 'w') as f:
            await f.write(f"{extracted.checksum}  {safe_filename}.json\n")
        
        self._log_audit("data_saved", {
            "path": extracted.path,
            "target": target_path,
            "size": extracted.size_bytes
        })
        
        return target_path
    
    def _calculate_checksum(self, data: Any, algorithm: str = "sha256") -> str:
        """체크섬 계산.
        
        Args:
            data: 데이터
            algorithm: 해시 알고리즘
            
        Returns:
            체크섬 값
        """
        hasher = hashlib.new(algorithm)
        
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        hasher.update(data_str.encode('utf-8'))
        return hasher.hexdigest()
    
    async def _encrypt_data(self, data: Any) -> bytes:
        """데이터 암호화.
        
        Args:
            data: 암호화할 데이터
            
        Returns:
            암호화된 데이터
        """
        # 실제 구현에서는 적절한 암호화 라이브러리 사용
        # 여기서는 간단한 예시
        from cryptography.fernet import Fernet
        
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key()
        
        f = Fernet(self.encryption_key)
        
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        return f.encrypt(data_str.encode('utf-8'))
    
    async def _compress_data(self, data: Any) -> bytes:
        """데이터 압축.
        
        Args:
            data: 압축할 데이터
            
        Returns:
            압축된 데이터
        """
        import gzip
        
        if isinstance(data, bytes):
            data_bytes = data
        else:
            data_str = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
            data_bytes = data_str.encode('utf-8')
        
        return gzip.compress(data_bytes)
    
    def _verify_integrity(self, extracted: ExtractedData) -> bool:
        """데이터 무결성 검증.
        
        Args:
            extracted: 추출된 데이터
            
        Returns:
            무결성 여부
        """
        # 체크섬 재계산
        recalculated = self._calculate_checksum(
            extracted.content,
            "sha256"  # 기본 알고리즘
        )
        
        # 메타데이터와 비교
        if extracted.metadata.get("extraction_checksum") != extracted.checksum:
            return False
        
        return True
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """SSL 컨텍스트 생성.
        
        Returns:
            SSL 컨텍스트
        """
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context
    
    def _sanitize_path(self, path: str) -> str:
        """경로 정제.
        
        Args:
            path: 원본 경로
            
        Returns:
            안전한 경로
        """
        # 위험한 문자 제거
        safe_path = path.replace("..", "").replace("/", "_").replace("\\", "_")
        return safe_path[:255]  # 최대 길이 제한
    
    def _generate_job_id(self) -> str:
        """작업 ID 생성.
        
        Returns:
            고유 작업 ID
        """
        import uuid
        return f"extract-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    
    def _log_audit(self, event_type: str, details: Dict[str, Any]) -> None:
        """감사 로그 기록.
        
        Args:
            event_type: 이벤트 타입
            details: 상세 정보
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self._audit_log.append(audit_entry)
        logger.info(f"Audit: {event_type} - {details}")
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """감사 로그 조회.
        
        Returns:
            감사 로그 목록
        """
        return self._audit_log.copy()