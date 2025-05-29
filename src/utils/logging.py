"""로깅 설정 모듈.

애플리케이션 전체의 로깅을 설정합니다.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import get_config


def setup_logging(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """로깅 설정 및 로거 반환.

    Args:
        name: 로거 이름
        level: 로그 레벨
        log_file: 로그 파일 경로

    Returns:
        설정된 로거 인스턴스
    """
    config = get_config()

    # 기본값 설정
    if level is None:
        level = config.logging.level
    if log_file is None:
        log_file = config.logging.get_log_path()

    # 로거 생성
    logger = logging.getLogger(name or "memory_one")
    logger.setLevel(getattr(logging, level))

    # 기존 핸들러 제거
    logger.handlers.clear()

    # 포맷터 생성
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (설정된 경우)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """지정된 이름의 로거를 가져옵니다.
    
    Args:
        name: 로거 이름
        
    Returns:
        로거 인스턴스
    """
    return logging.getLogger(name)


# 루트 로거 설정
root_logger = setup_logging()
