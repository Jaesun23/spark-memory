"""Structured chunking system.

Creates chunks intelligently by recognizing document structure.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .document_analyzer import DocumentStructure, DocumentSection, HeadingLevel

logger = logging.getLogger(__name__)


class ChunkType(str, Enum):
    """Chunk type."""

    TITLE = "title"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    QUOTE = "quote"
    MIXED = "mixed"


@dataclass
class StructuredChunk:
    """Structured chunk."""

    id: str
    content: str
    chunk_type: ChunkType
    start_pos: int
    end_pos: int

    # Structure information
    section_path: List[str] = field(default_factory=list)  # Section hierarchy path
    section_level: Optional[int] = None
    parent_section: Optional[str] = None

    # Context information
    context_before: str = ""  # Previous context
    context_after: str = ""  # Following context

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_full_path(self) -> str:
        """Return full path string."""
        return " > ".join(self.section_path) if self.section_path else ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "section_path": self.section_path,
            "section_level": self.section_level,
            "parent_section": self.parent_section,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "metadata": self.metadata,
        }


@dataclass
class ChunkingConfig:
    """Chunking configuration."""

    # Size settings
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    target_chunk_size: int = 500

    # Overlap settings
    overlap_size: int = 50
    context_window: int = 100

    # Structure settings
    preserve_structure: bool = True
    split_by_heading: bool = True
    include_heading_in_chunk: bool = True

    # Special handling
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True


class StructuredChunker:
    """Structured chunking processor."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        self._chunk_counter = 0

    async def create_structured_chunks(
        self, text: str, structure: DocumentStructure
    ) -> List[StructuredChunk]:
        """Create chunks using structure information.

        Args:
            text: Original text
            structure: Document structure information

        Returns:
            List of structured chunks
        """
        chunks: List[StructuredChunk] = []

        # Whether structure-based chunking is enabled
        if self.config.preserve_structure and structure.sections:
            chunks = await self._chunk_by_structure(text, structure)
        else:
            # Default chunking without structure info
            chunks = await self._chunk_by_size(text)

        # Add context
        await self._add_context_to_chunks(chunks, text)

        # Post-processing
        chunks = await self._post_process_chunks(chunks)

        logger.info(f"Created {len(chunks)} structured chunks")
        return chunks

    async def _chunk_by_structure(
        self, text: str, structure: DocumentStructure
    ) -> List[StructuredChunk]:
        """Structure-based chunking.

        Args:
            text: Original text
            structure: Document structure

        Returns:
            Chunk list
        """
        chunks: List[StructuredChunk] = []

        # Process by section
        for section in structure.sections:
            section_chunks = await self._process_section(section, text, [])
            chunks.extend(section_chunks)

        # Handle remaining text outside sections
        if chunks:
            last_pos = chunks[-1].end_pos
            if last_pos < len(text):
                remaining_text = text[last_pos:]
                if remaining_text.strip():
                    remaining_chunks = await self._chunk_by_size(
                        remaining_text, start_offset=last_pos
                    )
                    chunks.extend(remaining_chunks)

        return chunks

    async def _process_section(
        self, section: DocumentSection, text: str, parent_path: List[str]
    ) -> List[StructuredChunk]:
        """Process section.

        Args:
            section: Section information
            text: Full text
            parent_path: Parent section path

        Returns:
            Section's chunk list
        """
        chunks: List[StructuredChunk] = []

        # Current section path
        current_path = parent_path + [section.title]

        # Extract section content
        section_start = section.start_pos
        section_end = section.end_pos or len(text)
        section_text = text[section_start:section_end]

        # Whether to create title as separate chunk
        if self.config.split_by_heading and not self.config.include_heading_in_chunk:
            # Create title chunk
            title_end = section_text.find("\n")
            if title_end > 0:
                title_chunk = await self._create_chunk(
                    section_text[:title_end],
                    ChunkType.TITLE,
                    section_start,
                    section_start + title_end,
                    current_path,
                    section.level.value,
                )
                chunks.append(title_chunk)

                # Process remaining text
                content_start = section_start + title_end + 1
                section_text = text[content_start:section_end]
                section_start = content_start

        # If there are subsections
        if section.subsections:
            # Process content before subsections
            for i, subsection in enumerate(section.subsections):
                # Content from current position to subsection start
                if section_start < subsection.start_pos:
                    pre_text = text[section_start : subsection.start_pos]
                    if pre_text.strip():
                        pre_chunks = await self._chunk_text_by_content(
                            pre_text, section_start, current_path, section.level.value
                        )
                        chunks.extend(pre_chunks)

                # Process subsection
                sub_chunks = await self._process_section(subsection, text, current_path)
                chunks.extend(sub_chunks)

                # Update next start position
                section_start = subsection.end_pos or subsection.start_pos

        # Process content after last subsection
        if section_start < section_end:
            remaining_text = text[section_start:section_end]
            if remaining_text.strip():
                remaining_chunks = await self._chunk_text_by_content(
                    remaining_text, section_start, current_path, section.level.value
                )
                chunks.extend(remaining_chunks)

        return chunks

    async def _chunk_text_by_content(
        self, text: str, start_offset: int, section_path: List[str], section_level: int
    ) -> List[StructuredChunk]:
        """Content-based chunking.

        Args:
            text: Text to chunk
            start_offset: Start offset
            section_path: Section path
            section_level: Section level

        Returns:
            Chunk list
        """
        chunks: List[StructuredChunk] = []

        # Handle special blocks
        if self.config.preserve_code_blocks:
            text, code_chunks = await self._extract_code_blocks(
                text, start_offset, section_path, section_level
            )
            chunks.extend(code_chunks)

        if self.config.preserve_tables:
            text, table_chunks = await self._extract_tables(
                text, start_offset, section_path, section_level
            )
            chunks.extend(table_chunks)

        # Split remaining text by paragraph
        paragraphs = text.split("\n\n")
        current_pos = start_offset

        for para in paragraphs:
            para = para.strip()
            if not para:
                current_pos += 2  # Empty line
                continue

            # If paragraph is larger than max size, split further
            if len(para) > self.config.max_chunk_size:
                para_chunks = await self._split_large_paragraph(
                    para, current_pos, section_path, section_level
                )
                chunks.extend(para_chunks)
            else:
                chunk = await self._create_chunk(
                    para,
                    ChunkType.PARAGRAPH,
                    current_pos,
                    current_pos + len(para),
                    section_path,
                    section_level,
                )
                chunks.append(chunk)

            current_pos += len(para) + 2  # Paragraph + empty line

        return chunks

    async def _extract_code_blocks(
        self, text: str, start_offset: int, section_path: List[str], section_level: int
    ) -> Tuple[str, List[StructuredChunk]]:
        """Extract code blocks.

        Args:
            text: Original text
            start_offset: Start offset
            section_path: Section path
            section_level: Section level

        Returns:
            (Text with code blocks removed, code chunk list)
        """
        chunks: List[StructuredChunk] = []

        # Code block pattern
        code_pattern = r"```[\s\S]*?```"

        for match in re.finditer(code_pattern, text):
            code_block = match.group()
            start = start_offset + match.start()
            end = start_offset + match.end()

            chunk = await self._create_chunk(
                code_block, ChunkType.CODE, start, end, section_path, section_level
            )
            chunks.append(chunk)

        # Replace code blocks with placeholder
        modified_text = re.sub(code_pattern, "\n[CODE_BLOCK]\n", text)

        return modified_text, chunks

    async def _extract_tables(
        self, text: str, start_offset: int, section_path: List[str], section_level: int
    ) -> Tuple[str, List[StructuredChunk]]:
        """Extract tables.

        Args:
            text: Original text
            start_offset: Start offset
            section_path: Section path
            section_level: Section level

        Returns:
            (Text with tables removed, table chunk list)
        """
        chunks: List[StructuredChunk] = []

        # Simple markdown table pattern
        table_pattern = r"\|[^\n]+\|(\n\|[-:\s|]+\|)?(\n\|[^\n]+\|)+"

        for match in re.finditer(table_pattern, text):
            table = match.group()
            start = start_offset + match.start()
            end = start_offset + match.end()

            chunk = await self._create_chunk(
                table, ChunkType.TABLE, start, end, section_path, section_level
            )
            chunks.append(chunk)

        # Replace tables with placeholder
        modified_text = re.sub(table_pattern, "\n[TABLE]\n", text)

        return modified_text, chunks

    async def _split_large_paragraph(
        self,
        paragraph: str,
        start_offset: int,
        section_path: List[str],
        section_level: int,
    ) -> List[StructuredChunk]:
        """Split large paragraph.

        Args:
            paragraph: Paragraph text
            start_offset: Start offset
            section_path: Section path
            section_level: Section level

        Returns:
            Chunk list
        """
        chunks: List[StructuredChunk] = []

        # Split by sentence
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        current_chunk = []
        current_size = 0
        current_pos = start_offset

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding current sentence doesn't exceed max size
            if current_size + sentence_size <= self.config.target_chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # Include space
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk = await self._create_chunk(
                        chunk_text,
                        ChunkType.PARAGRAPH,
                        current_pos,
                        current_pos + len(chunk_text),
                        section_path,
                        section_level,
                    )
                    chunks.append(chunk)
                    current_pos += len(chunk_text) + 1

                # Start new chunk
                current_chunk = [sentence]
                current_size = sentence_size

        # Handle last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = await self._create_chunk(
                chunk_text,
                ChunkType.PARAGRAPH,
                current_pos,
                current_pos + len(chunk_text),
                section_path,
                section_level,
            )
            chunks.append(chunk)

        return chunks

    async def _chunk_by_size(
        self, text: str, start_offset: int = 0
    ) -> List[StructuredChunk]:
        """Size-based default chunking.

        Args:
            text: Text to chunk
            start_offset: Start offset

        Returns:
            Chunk list
        """
        chunks: List[StructuredChunk] = []

        # Split by paragraph
        paragraphs = text.split("\n\n")
        current_pos = start_offset

        for para in paragraphs:
            para = para.strip()
            if not para:
                current_pos += 2
                continue

            # If paragraph is larger than max size, split
            if len(para) > self.config.max_chunk_size:
                # Split by sentence
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk = []
                current_size = 0

                for sentence in sentences:
                    if current_size + len(sentence) <= self.config.target_chunk_size:
                        current_chunk.append(sentence)
                        current_size += len(sentence) + 1
                    else:
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            chunk = await self._create_chunk(
                                chunk_text,
                                ChunkType.PARAGRAPH,
                                current_pos,
                                current_pos + len(chunk_text),
                                [],
                                None,
                            )
                            chunks.append(chunk)
                            current_pos += len(chunk_text) + 1

                        current_chunk = [sentence]
                        current_size = len(sentence)

                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk = await self._create_chunk(
                        chunk_text,
                        ChunkType.PARAGRAPH,
                        current_pos,
                        current_pos + len(chunk_text),
                        [],
                        None,
                    )
                    chunks.append(chunk)
                    current_pos += len(chunk_text)
            else:
                # Use entire paragraph as one chunk
                chunk = await self._create_chunk(
                    para,
                    ChunkType.PARAGRAPH,
                    current_pos,
                    current_pos + len(para),
                    [],
                    None,
                )
                chunks.append(chunk)
                current_pos += len(para)

            current_pos += 2  # Empty line between paragraphs

        return chunks

    async def _create_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        start_pos: int,
        end_pos: int,
        section_path: List[str],
        section_level: Optional[int],
    ) -> StructuredChunk:
        """Create chunk.

        Args:
            content: Chunk content
            chunk_type: Chunk type
            start_pos: Start position
            end_pos: End position
            section_path: Section path
            section_level: Section level

        Returns:
            Created chunk
        """
        self._chunk_counter += 1

        chunk = StructuredChunk(
            id=f"chunk_{self._chunk_counter:06d}",
            content=content,
            chunk_type=chunk_type,
            start_pos=start_pos,
            end_pos=end_pos,
            section_path=section_path,
            section_level=section_level,
            parent_section=section_path[-1] if section_path else None,
            metadata={
                "char_count": len(content),
                "word_count": len(content.split()),
                "line_count": content.count("\n") + 1,
            },
        )

        return chunk

    async def _add_context_to_chunks(
        self, chunks: List[StructuredChunk], text: str
    ) -> None:
        """Add context to chunks.

        Args:
            chunks: Chunk list
            text: Original text
        """
        for i, chunk in enumerate(chunks):
            # Previous context
            if chunk.start_pos > 0:
                context_start = max(0, chunk.start_pos - self.config.context_window)
                chunk.context_before = text[context_start : chunk.start_pos].strip()

            # Following context
            if chunk.end_pos < len(text):
                context_end = min(len(text), chunk.end_pos + self.config.context_window)
                chunk.context_after = text[chunk.end_pos : context_end].strip()

    async def _post_process_chunks(
        self, chunks: List[StructuredChunk]
    ) -> List[StructuredChunk]:
        """Post-process chunks.

        Args:
            chunks: Chunk list

        Returns:
            Post-processed chunk list
        """
        processed_chunks: List[StructuredChunk] = []

        for chunk in chunks:
            # Handle chunks below minimum size
            if len(chunk.content) < self.config.min_chunk_size:
                # Keep code or tables regardless of size
                if chunk.chunk_type in [ChunkType.CODE, ChunkType.TABLE]:
                    processed_chunks.append(chunk)
                else:
                    # Try merging with previous chunk
                    if (
                        processed_chunks
                        and processed_chunks[-1].chunk_type == chunk.chunk_type
                    ):
                        prev_chunk = processed_chunks[-1]
                        if (
                            len(prev_chunk.content) + len(chunk.content)
                            <= self.config.max_chunk_size
                        ):
                            prev_chunk.content += "\n\n" + chunk.content
                            prev_chunk.end_pos = chunk.end_pos
                            prev_chunk.context_after = chunk.context_after
                            prev_chunk.metadata["char_count"] = len(prev_chunk.content)
                            prev_chunk.metadata["word_count"] = len(
                                prev_chunk.content.split()
                            )
                            continue

                    # If can't merge, add as is
                    processed_chunks.append(chunk)
            else:
                processed_chunks.append(chunk)

        return processed_chunks

    def get_chunking_stats(self, chunks: List[StructuredChunk]) -> Dict[str, Any]:
        """Return chunking statistics.

        Args:
            chunks: Chunk list

        Returns:
            Statistics information
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "chunk_types": {},
            }

        sizes = [len(chunk.content) for chunk in chunks]
        type_counts: Dict[str, int] = {}

        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) // len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "chunk_types": type_counts,
            "total_characters": sum(sizes),
            "has_structure": any(chunk.section_path for chunk in chunks),
        }