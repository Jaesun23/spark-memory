"""Embedding service for generating text embeddings."""

import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import openai
except ImportError:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingModel(str, Enum):
    """Available embedding models."""

    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    LOCAL_MINILM = "all-MiniLM-L6-v2"
    LOCAL_MPNET = "all-mpnet-base-v2"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(
        self, model: str = EmbeddingModel.OPENAI_SMALL, api_key: Optional[str] = None
    ):
        if openai is None:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.model = model
        self.client = (
            openai.AsyncOpenAI(api_key=api_key) if api_key else openai.AsyncOpenAI()
        )

        # Model dimensions
        self.dimensions = {
            EmbeddingModel.OPENAI_SMALL: 1536,
            EmbeddingModel.OPENAI_LARGE: 3072,
        }

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = await self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimensions.get(self.model, 1536)


class LocalEmbeddings(EmbeddingProvider):
    """Local embeddings using sentence-transformers."""

    def __init__(self, model: str = EmbeddingModel.LOCAL_MINILM):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model
        self.model = SentenceTransformer(model)

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        # Run in thread pool to avoid blocking
        import asyncio

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text, convert_to_tensor=False).tolist()
        )
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        import asyncio

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.encode(texts, convert_to_tensor=False).tolist()
        )
        return embeddings

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        # Standard dimensions for common models
        dimensions = {
            EmbeddingModel.LOCAL_MINILM: 384,
            EmbeddingModel.LOCAL_MPNET: 768,
        }
        return dimensions.get(
            self.model_name, self.model.get_sentence_embedding_dimension()
        )


class EmbeddingService:
    """Service for managing embeddings with fallback support."""

    def __init__(
        self,
        primary_model: EmbeddingModel = EmbeddingModel.OPENAI_SMALL,
        fallback_model: Optional[EmbeddingModel] = EmbeddingModel.LOCAL_MINILM,
        openai_api_key: Optional[str] = None,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.providers: Dict[EmbeddingModel, EmbeddingProvider] = {}

        # Initialize primary provider
        try:
            if "openai" in primary_model:
                self.providers[primary_model] = OpenAIEmbeddings(
                    primary_model, openai_api_key
                )
            else:
                self.providers[primary_model] = LocalEmbeddings(primary_model)
        except ImportError as e:
            if fallback_model:
                print(
                    f"Primary model {primary_model} not available: {e}. Using fallback."
                )
            else:
                raise

        # Initialize fallback provider if specified
        if fallback_model and fallback_model not in self.providers:
            try:
                if "openai" in fallback_model:
                    self.providers[fallback_model] = OpenAIEmbeddings(
                        fallback_model, openai_api_key
                    )
                else:
                    self.providers[fallback_model] = LocalEmbeddings(fallback_model)
            except ImportError:
                pass

    def get_provider(self) -> EmbeddingProvider:
        """Get the active embedding provider."""
        if self.primary_model in self.providers:
            return self.providers[self.primary_model]
        elif self.fallback_model and self.fallback_model in self.providers:
            return self.providers[self.fallback_model]
        else:
            raise RuntimeError("No embedding provider available")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        provider = self.get_provider()
        return await provider.embed_text(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        provider = self.get_provider()
        return await provider.embed_batch(texts)

    def get_dimension(self) -> int:
        """Get the embedding dimension of the active provider."""
        provider = self.get_provider()
        return provider.get_dimension()

    def prepare_text_for_embedding(self, content: Any) -> str:
        """Prepare content for embedding by extracting text."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Extract relevant fields from structured data
            parts = []
            for key in ["title", "name", "content", "description", "text"]:
                if key in content:
                    parts.append(str(content[key]))
            return " ".join(parts) if parts else json.dumps(content)
        else:
            return str(content)
