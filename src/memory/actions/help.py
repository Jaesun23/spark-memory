"""Help system actions."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HelpActions:
    """Help system operations handler."""
    
    def __init__(self):
        """Initialize help actions."""
        pass
        
    async def execute(
        self,
        paths: List[str],
        content: Optional[Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute help action.
        
        Args:
            paths: Help topic path
            content: Subtopic
            options: Help options
            
        Returns:
            Help content
        """
        from src.memory.help_guide import get_help_message
        
        topic = paths[0] if paths else None
        subtopic = content if isinstance(content, str) else None
        
        help_text = get_help_message(topic, subtopic)
        
        return {"help": help_text}