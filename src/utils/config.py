"""환경 설정 관리 모듈.

환경 변수를 로드하고 검증하며, 설정 값에 대한 타입 안전성을 제공합니다.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class RedisConfig:
    """Redis 연결 설정."""

    url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    password: Optional[str] = field(
        default_factory=lambda: os.getenv("REDIS_PASSWORD") or None
    )
    max_connections: int = field(
        default_factory=lambda: int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    )

    def get_connection_url(self) -> str:
        """패스워드가 포함된 전체 연결 URL 반환."""
        if self.password:
            # URL에 패스워드 삽입
            parts = self.url.split("://")
            if len(parts) == 2:
                return (
                    f"{parts[0]}://:{self.password}@{parts[1].replace('redis://', '')}"
                )
        return self.url


@dataclass
class MCPConfig:
    """MCP 서버 설정."""

    server_name: str = field(
        default_factory=lambda: os.getenv("MCP_SERVER_NAME", "LRMM Memory Server")
    )
    server_version: str = field(
        default_factory=lambda: os.getenv("MCP_SERVER_VERSION", "0.1.0")
    )


@dataclass
class SecurityConfig:
    """보안 설정."""

    api_key: Optional[str] = field(default_factory=lambda: os.getenv("API_KEY") or None)
    allowed_ips: List[str] = field(
        default_factory=lambda: os.getenv("ALLOWED_IPS", "127.0.0.1,localhost").split(
            ","
        )
    )

    def is_ip_allowed(self, ip: str) -> bool:
        """IP 주소가 허용 목록에 있는지 확인."""
        return ip in self.allowed_ips or "0.0.0.0" in self.allowed_ips


@dataclass
class LoggingConfig:
    """로깅 설정."""

    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))

    def get_log_path(self) -> Optional[Path]:
        """로그 파일 경로 반환."""
        if self.file:
            path = Path(self.file)
            # 디렉토리가 없으면 생성
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return None


@dataclass
class AppConfig:
    """전체 애플리케이션 설정."""

    env: str = field(default_factory=lambda: os.getenv("ENV", "development"))
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    redis: RedisConfig = field(default_factory=RedisConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def is_production(self) -> bool:
        """프로덕션 환경인지 확인."""
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        """개발 환경인지 확인."""
        return self.env == "development"

    def validate(self) -> List[str]:
        """설정 유효성 검사.

        Returns:
            경고 메시지 리스트
        """
        warnings = []

        # 프로덕션 환경 검사
        if self.is_production:
            if not self.redis.password:
                warnings.append(
                    "Production 환경에서 Redis 패스워드가 설정되지 않았습니다."
                )
            if not self.security.api_key:
                warnings.append("Production 환경에서 API 키가 설정되지 않았습니다.")
            if self.debug:
                warnings.append("Production 환경에서 DEBUG 모드가 활성화되어 있습니다.")

        # 개발 환경 검사
        if self.is_development:
            if self.redis.password:
                warnings.append(
                    "Development 환경에서 Redis 패스워드가 설정되어 있습니다."
                )

        return warnings


# 전역 설정 인스턴스
config = AppConfig()


def get_config() -> AppConfig:
    """전역 설정 인스턴스 반환."""
    return config
