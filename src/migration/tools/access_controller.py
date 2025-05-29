"""Access control for migration tools."""

import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """인증 오류."""
    pass


class AuthorizationError(Exception):
    """권한 오류."""
    pass


class UserRole(Enum):
    """마이그레이션 역할."""
    ADMIN = "admin"          # 모든 권한
    OPERATOR = "operator"    # 실행 권한
    AUDITOR = "auditor"      # 읽기 전용
    VIEWER = "viewer"        # 제한된 보기


class Permission(Enum):
    """마이그레이션 권한."""
    EXECUTE = "execute"              # 마이그레이션 실행
    CONFIGURE = "configure"          # 설정 변경
    VIEW_DATA = "view_data"          # 데이터 조회
    VIEW_LOGS = "view_logs"          # 로그 조회
    EXPORT_DATA = "export_data"      # 데이터 내보내기
    DELETE_DATA = "delete_data"      # 데이터 삭제
    MANAGE_USERS = "manage_users"    # 사용자 관리


@dataclass
class Session:
    """인증 세션."""
    session_id: str
    user_id: str
    role: UserRole
    created_at: datetime
    expires_at: datetime
    ip_address: str
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """마이그레이션 사용자."""
    user_id: str
    username: str
    role: UserRole
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None


class MigrationAccessController:
    """마이그레이션 도구 접근 제어."""
    
    # 역할별 권한 매핑
    ROLE_PERMISSIONS = {
        UserRole.ADMIN: {
            Permission.EXECUTE,
            Permission.CONFIGURE,
            Permission.VIEW_DATA,
            Permission.VIEW_LOGS,
            Permission.EXPORT_DATA,
            Permission.DELETE_DATA,
            Permission.MANAGE_USERS,
        },
        UserRole.OPERATOR: {
            Permission.EXECUTE,
            Permission.VIEW_DATA,
            Permission.VIEW_LOGS,
            Permission.EXPORT_DATA,
        },
        UserRole.AUDITOR: {
            Permission.VIEW_DATA,
            Permission.VIEW_LOGS,
        },
        UserRole.VIEWER: {
            Permission.VIEW_LOGS,
        },
    }
    
    def __init__(
        self,
        max_attempts: int = 5,
        lockout_duration: int = 300,  # 5분
        session_duration: int = 3600,  # 1시간
    ):
        """초기화.
        
        Args:
            max_attempts: 최대 로그인 시도 횟수
            lockout_duration: 잠금 시간 (초)
            session_duration: 세션 유효 시간 (초)
        """
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.max_attempts = max_attempts
        self.lockout_duration = lockout_duration
        self.session_duration = session_duration
        
        # 기본 관리자 생성
        self._create_default_admin()
    
    def _create_default_admin(self) -> None:
        """기본 관리자 계정 생성."""
        admin_password = secrets.token_urlsafe(16)
        admin_user = User(
            user_id="admin",
            username="admin",
            role=UserRole.ADMIN,
            password_hash=self._hash_password(admin_password),
            created_at=datetime.now()
        )
        
        self.users["admin"] = admin_user
        logger.warning(f"Default admin created with password: {admin_password}")
        logger.warning("CHANGE THIS PASSWORD IMMEDIATELY!")
    
    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole,
        executor_session: Optional[Session] = None
    ) -> User:
        """사용자 생성.
        
        Args:
            username: 사용자명
            password: 비밀번호
            role: 역할
            executor_session: 실행자 세션
            
        Returns:
            생성된 사용자
            
        Raises:
            PermissionError: 권한 없음
            ValueError: 잘못된 입력
        """
        # 권한 확인
        if executor_session and not self._has_permission(
            executor_session, Permission.MANAGE_USERS
        ):
            raise PermissionError("No permission to manage users")
        
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        user = User(
            user_id=username,
            username=username,
            role=role,
            password_hash=self._hash_password(password),
            created_at=datetime.now()
        )
        
        self.users[username] = user
        logger.info(f"User created: {username} with role {role.value}")
        
        return user
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str,
        mfa_code: Optional[str] = None
    ) -> Session:
        """사용자 인증.
        
        Args:
            username: 사용자명
            password: 비밀번호
            ip_address: IP 주소
            mfa_code: MFA 코드
            
        Returns:
            인증 세션
            
        Raises:
            ValueError: 인증 실패
        """
        if username not in self.users:
            raise ValueError("Invalid credentials")
        
        user = self.users[username]
        
        # 잠금 확인
        if user.locked_until and user.locked_until > datetime.now():
            raise ValueError(f"Account locked until {user.locked_until}")
        
        # 비밀번호 확인
        if not self._verify_password(password, user.password_hash):
            user.failed_attempts += 1
            
            if user.failed_attempts >= self.max_attempts:
                user.locked_until = datetime.now() + timedelta(
                    seconds=self.lockout_duration
                )
                logger.warning(f"User {username} locked after {user.failed_attempts} failed attempts")
            
            raise ValueError("Invalid credentials")
        
        # MFA 확인
        if user.mfa_enabled:
            if not mfa_code or not self._verify_mfa(user.mfa_secret, mfa_code):
                raise ValueError("Invalid MFA code")
        
        # 세션 생성
        session = Session(
            session_id=secrets.token_urlsafe(32),
            user_id=user.user_id,
            role=user.role,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.session_duration),
            ip_address=ip_address,
            permissions=self.ROLE_PERMISSIONS[user.role].copy()
        )
        
        self.sessions[session.session_id] = session
        
        # 사용자 정보 업데이트
        user.failed_attempts = 0
        user.last_login = datetime.now()
        
        logger.info(f"User {username} authenticated from {ip_address}")
        
        return session
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """세션 검증.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            유효한 세션 또는 None
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # 만료 확인
        if session.expires_at < datetime.now():
            del self.sessions[session_id]
            logger.info(f"Session {session_id} expired")
            return None
        
        return session
    
    def check_permission(
        self,
        session_id: str,
        permission: Permission
    ) -> bool:
        """권한 확인.
        
        Args:
            session_id: 세션 ID
            permission: 확인할 권한
            
        Returns:
            권한 보유 여부
        """
        session = self.validate_session(session_id)
        if not session:
            return False
        
        return self._has_permission(session, permission)
    
    def revoke_session(self, session_id: str) -> None:
        """세션 취소.
        
        Args:
            session_id: 세션 ID
        """
        if session_id in self.sessions:
            user_id = self.sessions[session_id].user_id
            del self.sessions[session_id]
            logger.info(f"Session revoked for user {user_id}")
    
    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str
    ) -> None:
        """비밀번호 변경.
        
        Args:
            username: 사용자명
            old_password: 기존 비밀번호
            new_password: 새 비밀번호
            
        Raises:
            ValueError: 검증 실패
        """
        if username not in self.users:
            raise ValueError("User not found")
        
        user = self.users[username]
        
        if not self._verify_password(old_password, user.password_hash):
            raise ValueError("Invalid current password")
        
        user.password_hash = self._hash_password(new_password)
        logger.info(f"Password changed for user {username}")
    
    def enable_mfa(self, username: str) -> str:
        """MFA 활성화.
        
        Args:
            username: 사용자명
            
        Returns:
            MFA 시크릿
            
        Raises:
            ValueError: 사용자 없음
        """
        if username not in self.users:
            raise ValueError("User not found")
        
        user = self.users[username]
        secret = secrets.token_urlsafe(32)
        
        user.mfa_enabled = True
        user.mfa_secret = secret
        
        logger.info(f"MFA enabled for user {username}")
        
        return secret
    
    def _has_permission(self, session: Session, permission: Permission) -> bool:
        """권한 보유 확인.
        
        Args:
            session: 세션
            permission: 권한
            
        Returns:
            권한 보유 여부
        """
        return permission in session.permissions
    
    def _hash_password(self, password: str) -> str:
        """비밀번호 해시.
        
        Args:
            password: 비밀번호
            
        Returns:
            해시값
        """
        # 실제로는 bcrypt 등 사용
        salt = secrets.token_bytes(32)
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        ).hex() + ":" + salt.hex()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """비밀번호 검증.
        
        Args:
            password: 비밀번호
            password_hash: 저장된 해시
            
        Returns:
            일치 여부
        """
        try:
            hash_value, salt = password_hash.split(":")
            test_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                bytes.fromhex(salt),
                100000
            ).hex()
            return test_hash == hash_value
        except Exception:
            return False
    
    def _verify_mfa(self, secret: str, code: str) -> bool:
        """MFA 코드 검증.
        
        Args:
            secret: MFA 시크릿
            code: 입력된 코드
            
        Returns:
            검증 성공 여부
        """
        # 실제로는 TOTP 등 사용
        # 여기서는 간단한 예시
        import hmac
        import time
        
        time_step = int(time.time() // 30)
        expected = hmac.new(
            secret.encode(),
            str(time_step).encode(),
            hashlib.sha256
        ).hexdigest()[:6]
        
        return code == expected