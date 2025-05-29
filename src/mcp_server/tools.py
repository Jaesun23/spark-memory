"""MCP 도구 정의 및 자연어 처리."""

from typing import Dict, Any, Optional, List
from fastmcp import Context
import logging

from ..utils.command_parser import CommandParser

logger = logging.getLogger(__name__)


async def handle_natural_language_command(
    text: str,
    server: Any,
    ctx: Optional[Context] = None
) -> Any:
    """자연어 명령어 처리.
    
    Args:
        text: 자연어 명령어
        server: MemoryServer 인스턴스
        ctx: MCP Context
        
    Returns:
        명령 실행 결과
    """
    # 자연어 파싱
    parsed = CommandParser.parse(text)
    
    if ctx:
        await ctx.info(f"Parsed command: action={parsed['action']}")
    
    # 파싱 결과로 m_memory 실행
    result = await server.memory_engine.execute(
        action=parsed["action"],
        paths=parsed["paths"],
        content=parsed["content"],
        options=parsed["options"]
    )
    
    return result


def detect_command_type(text: str) -> str:
    """텍스트에서 명령어 타입 감지.
    
    Args:
        text: 입력 텍스트
        
    Returns:
        명령어 타입 (natural, structured)
    """
    # 구조화된 명령어 패턴
    structured_patterns = [
        # JSON 형태
        r'^\s*\{.*\}\s*$',
        # action 파라미터 명시
        r'action\s*[:=]',
        # 함수 호출 형태
        r'm_memory\s*\(',
    ]
    
    import re
    for pattern in structured_patterns:
        if re.search(pattern, text, re.DOTALL):
            return "structured"
    
    return "natural"