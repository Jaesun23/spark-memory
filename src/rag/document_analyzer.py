"""Document structure analyzer.

Analyzes hierarchical structure, sections, metadata, etc. of documents.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Document type."""

    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    CODE = "code"
    CONVERSATION = "conversation"
    STRUCTURED = "structured"  # JSON, YAML etc
    MIXED = "mixed"


class HeadingLevel(int, Enum):
    """Heading level."""

    TITLE = 0  # Document title
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6


@dataclass
class DocumentSection:
    """Document section."""

    title: str
    level: HeadingLevel
    start_pos: int
    end_pos: Optional[int] = None
    content: str = ""
    subsections: List["DocumentSection"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_subsection(self, subsection: "DocumentSection") -> None:
        """Add subsection."""
        self.subsections.append(subsection)

    def get_full_path(self) -> str:
        """Return full path of section."""
        if hasattr(self, "_parent") and self._parent:
            return f"{self._parent.get_full_path()} > {self.title}"
        return self.title


@dataclass
class DocumentStructure:
    """Document structure."""

    doc_type: DocumentType
    title: Optional[str] = None
    sections: List[DocumentSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    outline: List[Tuple[int, str]] = field(default_factory=list)  # (level, title)

    def add_section(self, section: DocumentSection) -> None:
        """Add section."""
        self.sections.append(section)
        self.outline.append((section.level.value, section.title))

    def get_section_at_position(self, position: int) -> Optional[DocumentSection]:
        """Return section at specific position."""

        def _find_section_recursive(
            section: DocumentSection, pos: int
        ) -> Optional[DocumentSection]:
            """Recursively find section."""
            if section.start_pos <= pos < (section.end_pos or float("inf")):
                # Check subsections
                for subsection in section.subsections:
                    result = _find_section_recursive(subsection, pos)
                    if result:
                        return result
                return section
            return None

        # Search in all top-level sections
        for section in self.sections:
            result = _find_section_recursive(section, position)
            if result:
                return result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_type": self.doc_type.value,
            "title": self.title,
            "sections": [
                {
                    "title": s.title,
                    "level": s.level.value,
                    "start_pos": s.start_pos,
                    "end_pos": s.end_pos,
                    "subsections": len(s.subsections),
                }
                for s in self.sections
            ],
            "metadata": self.metadata,
            "outline": self.outline,
        }


class DocumentAnalyzer:
    """Document structure analyzer."""

    # Markdown heading patterns
    MARKDOWN_HEADING_PATTERNS = [
        (r"^(#{1})\s+(.+)$", HeadingLevel.H1),
        (r"^(#{2})\s+(.+)$", HeadingLevel.H2),
        (r"^(#{3})\s+(.+)$", HeadingLevel.H3),
        (r"^(#{4})\s+(.+)$", HeadingLevel.H4),
        (r"^(#{5})\s+(.+)$", HeadingLevel.H5),
        (r"^(#{6})\s+(.+)$", HeadingLevel.H6),
    ]

    # Numbering patterns
    NUMBERED_HEADING_PATTERNS = [
        (r"^(\d+)\.\s+(.+)$", HeadingLevel.H1),  # 1. Title
        (r"^(\d+)\.(\d+)\.\s+(.+)$", HeadingLevel.H2),  # 1.1. Title
        (r"^(\d+)\.(\d+)\.(\d+)\.\s+(.+)$", HeadingLevel.H3),  # 1.1.1. Title
    ]

    # Other structure patterns
    STRUCTURE_PATTERNS = {
        "list_item": r"^[\*\-\+]\s+(.+)$",
        "numbered_list": r"^\d+\.\s+(.+)$",
        "code_block": r"^```(\w+)?$",
        "quote": r"^>\s*(.+)$",
        "table_separator": r"^\|[\s\-\|]+\|$",
    }

    def __init__(self) -> None:
        """Initialize."""
        self.section_stack: List[DocumentSection] = []

    async def analyze_document(self, content: Any) -> DocumentStructure:
        """Analyze document structure.

        Args:
            content: Document content to analyze

        Returns:
            Document structure
        """
        # Detect document type
        doc_type = self._detect_document_type(content)

        # Create document structure
        structure = DocumentStructure(doc_type=doc_type)

        # Type-specific analysis
        if doc_type == DocumentType.MARKDOWN:
            await self._analyze_markdown(content, structure)
        elif doc_type == DocumentType.PLAIN_TEXT:
            await self._analyze_plain_text(content, structure)
        elif doc_type == DocumentType.CONVERSATION:
            await self._analyze_conversation(content, structure)
        elif doc_type == DocumentType.STRUCTURED:
            await self._analyze_structured(content, structure)
        else:
            await self._analyze_mixed(content, structure)

        return structure

    def _detect_document_type(self, content: Any) -> DocumentType:
        """Detect document type."""
        if isinstance(content, dict):
            # Check conversation format
            if any(key in content for key in ["messages", "conversation", "chat"]):
                return DocumentType.CONVERSATION
            # Structured data
            return DocumentType.STRUCTURED

        if isinstance(content, str):
            # Check markdown features
            if self._has_markdown_features(content):
                return DocumentType.MARKDOWN
            # Check code features
            if self._has_code_features(content):
                return DocumentType.CODE
            # Plain text
            return DocumentType.PLAIN_TEXT

        return DocumentType.MIXED

    def _has_markdown_features(self, text: str) -> bool:
        """Check markdown features."""
        markdown_indicators = [
            r"^#{1,6}\s+",  # Headings
            r"\[.+\]\(.+\)",  # Links
            r"!\[.+\]\(.+\)",  # Images
            r"^```",  # Code blocks
            r"^\*{1,2}.+\*{1,2}",  # Bold
            r"^_{1,2}.+_{1,2}",  # Italic
        ]

        for pattern in markdown_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _has_code_features(self, text: str) -> bool:
        """Check code features."""
        code_indicators = [
            r"^\s*(def|class|function|const|let|var)\s+",
            r"^\s*(import|from|require)\s+",
            r"[{};]\s*$",
            r"^\s*(if|for|while|switch)\s*\(",
            r"^\s*return\s+",
            r"^\s*pass\s*$",
        ]

        matches = 0
        for pattern in code_indicators:
            if re.search(pattern, text, re.MULTILINE):
                matches += 1

        return matches >= 3  # Stricter criteria

    async def _analyze_markdown(
        self, content: str, structure: DocumentStructure
    ) -> None:
        """Analyze markdown document."""
        lines = content.split("\n")
        current_pos = 0

        # Initialize section stack
        self.section_stack = []

        # Track whether first H1 is used as title
        first_h1_is_title = False

        # Use first H1 as document title
        for i, line in enumerate(lines):
            for pattern, level in self.MARKDOWN_HEADING_PATTERNS:
                match = re.match(pattern, line.strip())
                if match and level == HeadingLevel.H1 and not structure.title:
                    structure.title = match.group(2)
                    first_h1_is_title = True
                    break

        # Section analysis
        found_first_h1 = False
        for i, line in enumerate(lines):
            current_pos += len(line) + 1  # +1 for newline

            # Check heading patterns
            for pattern, level in self.MARKDOWN_HEADING_PATTERNS:
                match = re.match(pattern, line.strip())
                if match:
                    title = match.group(2)

                    # Skip first H1 if used as title
                    if (
                        level == HeadingLevel.H1
                        and not found_first_h1
                        and first_h1_is_title
                    ):
                        found_first_h1 = True
                        break

                    section = DocumentSection(
                        title=title,
                        level=level,
                        start_pos=current_pos - len(line) - 1,
                    )

                    # Set end position for previous section
                    if self.section_stack:
                        self.section_stack[-1].end_pos = section.start_pos - 1

                    # Add section to appropriate location
                    self._add_section_to_structure(section, structure)
                    break

        # Set end position for last section
        if self.section_stack:
            self.section_stack[-1].end_pos = current_pos

        # Extract metadata
        structure.metadata["line_count"] = len(lines)
        structure.metadata["has_code_blocks"] = "```" in content
        structure.metadata["has_tables"] = "|" in content and "---|" in content

    def _add_section_to_structure(
        self, section: DocumentSection, structure: DocumentStructure
    ) -> None:
        """Add section to structure."""
        # Remove from stack sections with level >= current section level
        while (
            self.section_stack
            and self.section_stack[-1].level.value >= section.level.value
        ):
            self.section_stack.pop()

        # If parent section exists, add as subsection
        if (
            self.section_stack
            and self.section_stack[-1].level.value < section.level.value
        ):
            parent = self.section_stack[-1]
            parent.add_subsection(section)
            setattr(section, "_parent", parent)  # Use setattr for dynamic attribute
        else:
            # Top-level section
            structure.add_section(section)

        # Add to stack
        self.section_stack.append(section)

    async def _analyze_plain_text(
        self, content: str, structure: DocumentStructure
    ) -> None:
        """Analyze plain text."""
        lines = content.split("\n")

        # Initialize section stack
        self.section_stack = []

        # Check numbering patterns
        for i, line in enumerate(lines):
            for pattern, level in self.NUMBERED_HEADING_PATTERNS:
                match = re.match(pattern, line.strip())
                if match:
                    title = match.groups()[-1]  # Last group is title
                    section = DocumentSection(
                        title=title,
                        level=level,
                        start_pos=sum(len(lines[j]) + 1 for j in range(i)),
                    )

                    # Use same hierarchy logic as markdown
                    self._add_section_to_structure(section, structure)
                    break

        # Metadata
        structure.metadata["line_count"] = len(lines)
        structure.metadata["word_count"] = len(content.split())

    async def _analyze_conversation(
        self, content: Dict[str, Any], structure: DocumentStructure
    ) -> None:
        """Analyze conversation format."""
        messages = content.get("messages", content.get("conversation", []))

        if not messages:
            return

        # Group conversation into sessions
        current_session = None
        session_count = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")

            # Start new session (starts with user message)
            if role == "user" and (
                not current_session
                or current_session.metadata.get("last_role") == "assistant"
            ):
                session_count += 1
                current_session = DocumentSection(
                    title=f"Conversation Session {session_count}",
                    level=HeadingLevel.H1,
                    start_pos=i,
                    metadata={"message_count": 0},
                )
                structure.add_section(current_session)

            if current_session:
                current_session.metadata["message_count"] += 1
                current_session.metadata["last_role"] = role

        # Metadata
        structure.metadata["total_messages"] = len(messages)
        structure.metadata["session_count"] = session_count

    async def _analyze_structured(
        self, content: Dict[str, Any], structure: DocumentStructure
    ) -> None:
        """Analyze structured data."""
        # Use top-level keys as sections
        for key, value in content.items():
            section = DocumentSection(
                title=key,
                level=HeadingLevel.H1,
                start_pos=0,
                metadata={"type": type(value).__name__},
            )

            # Analyze nested structure
            if isinstance(value, dict):
                await self._analyze_nested_dict(value, section, HeadingLevel.H2)
            elif isinstance(value, list):
                section.metadata["item_count"] = len(value)

            structure.add_section(section)

        structure.metadata["key_count"] = len(content)

    async def _analyze_nested_dict(
        self, data: Dict[str, Any], parent: DocumentSection, level: HeadingLevel
    ) -> None:
        """Analyze nested dictionary."""
        if level.value > HeadingLevel.H6.value:
            return

        for key, value in data.items():
            subsection = DocumentSection(
                title=key,
                level=level,
                start_pos=0,
                metadata={"type": type(value).__name__},
            )
            parent.add_subsection(subsection)

            if isinstance(value, dict) and level.value < HeadingLevel.H6.value:
                next_level = HeadingLevel(level.value + 1)
                await self._analyze_nested_dict(value, subsection, next_level)

    async def _analyze_mixed(self, content: Any, structure: DocumentStructure) -> None:
        """Analyze mixed format."""
        # Convert to string and analyze
        if hasattr(content, "__str__"):
            text_content = str(content)
            await self._analyze_plain_text(text_content, structure)

        structure.metadata["original_type"] = type(content).__name__

    def extract_outline(self, structure: DocumentStructure) -> List[str]:
        """Extract document outline.

        Args:
            structure: Document structure

        Returns:
            List of outline strings
        """
        outline = []

        def _process_section(section: DocumentSection, indent: int = 0) -> None:
            prefix = "  " * indent + "- "
            outline.append(f"{prefix}{section.title}")

            for subsection in section.subsections:
                _process_section(subsection, indent + 1)

        for section in structure.sections:
            _process_section(section)

        return outline