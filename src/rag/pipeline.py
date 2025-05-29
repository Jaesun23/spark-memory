"""RAG pipeline implementation."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..embeddings import VectorStore
from ..memory.models import SearchResult


class ChunkingStrategy(str, Enum):
    """Text chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100


class TextChunker:
    """Text chunking utility."""

    def __init__(self, config: ChunkConfig):
        self.config = config

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks based on strategy."""
        if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text)
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text)
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text)
        else:
            # Semantic chunking would require more sophisticated NLP
            return self._chunk_fixed_size(text)

    def _chunk_fixed_size(self, text: str) -> List[Dict[str, Any]]:
        """Fixed size chunking with overlap."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.config.chunk_size

            # Find a good break point (space, newline, punctuation)
            if end < text_length:
                for i in range(
                    end, max(start + self.config.min_chunk_size, end - 50), -1
                ):
                    if text[i] in [" ", "\n", ".", "!", "?"]:
                        end = i + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "start": start,
                        "end": end,
                        "index": len(chunks),
                    }
                )

            # Move start with overlap
            start = end - self.config.chunk_overlap
            if start >= text_length:
                break

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[Dict[str, Any]]:
        """Chunk by sentences."""
        # Simple sentence splitting (could use NLTK or spaCy for better results)
        import re

        sentences = re.split(r"[.!?]+", text)

        chunks = []
        current_chunk = ""
        current_start = 0

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "start": current_start,
                            "end": current_start + len(current_chunk),
                            "index": len(chunks),
                        }
                    )
                current_chunk = sentence
                current_start = text.find(sentence, current_start)
            else:
                current_chunk += ". " + sentence if current_chunk else sentence

        # Add final chunk
        if current_chunk:
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "start": current_start,
                    "end": len(text),
                    "index": len(chunks),
                }
            )

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[Dict[str, Any]]:
        """Chunk by paragraphs."""
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""
        current_start = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "start": current_start,
                            "end": current_start + len(current_chunk),
                            "index": len(chunks),
                        }
                    )
                current_chunk = paragraph
                current_start = text.find(paragraph, current_start)
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "start": current_start,
                    "end": len(text),
                    "index": len(chunks),
                }
            )

        return chunks


class RAGPipeline:
    """RAG pipeline for memory retrieval and augmentation."""

    def __init__(
        self, vector_store: VectorStore, chunk_config: Optional[ChunkConfig] = None
    ):
        self.vector_store = vector_store
        self.chunker = TextChunker(chunk_config or ChunkConfig())

    async def process_document(
        self, path: str, content: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Process document for RAG by chunking and indexing.

        Returns list of chunk paths that were created.
        """
        # Extract text content
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            text = self._extract_text_from_dict(content)
        else:
            text = str(content)

        # Chunk the text
        chunks = self.chunker.chunk_text(text)

        # Store chunks with embeddings
        chunk_paths = []
        memories_to_add = []

        for chunk in chunks:
            chunk_path = f"{path}/chunk_{chunk['index']}"
            chunk_metadata = {
                **(metadata or {}),
                "parent_path": path,
                "chunk_index": chunk["index"],
                "chunk_start": chunk["start"],
                "chunk_end": chunk["end"],
                "type": "chunk",
            }

            memories_to_add.append((chunk_path, chunk["text"], chunk_metadata))
            chunk_paths.append(chunk_path)

        # Batch add all chunks
        if memories_to_add:
            await self.vector_store.batch_add_memories(memories_to_add)

        return chunk_paths

    async def retrieve_context(
        self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, str]]:
        """Retrieve relevant context for a query.

        Returns list of (path, score, text) tuples.
        """
        # Search for similar chunks
        results = await self.vector_store.search_similar(
            query_text=query,
            k=k,
            filters=filters,
            hybrid=True,  # Use hybrid search for better results
        )

        # Extract text content
        context_items = []
        for path, score, metadata in results:
            content = metadata.get("content", "")
            if isinstance(content, dict):
                text = self._extract_text_from_dict(content)
            else:
                text = str(content)

            context_items.append((path, score, text))

        return context_items

    def _extract_text_from_dict(self, data: Dict[str, Any]) -> str:
        """Extract text from dictionary content."""
        text_parts = []

        # Priority fields for text extraction
        priority_fields = ["content", "text", "description", "summary", "title"]

        for field in priority_fields:
            if field in data:
                text_parts.append(str(data[field]))

        # If no priority fields, convert entire dict
        if not text_parts:
            text_parts.append(json.dumps(data, ensure_ascii=False))

        return " ".join(text_parts)
