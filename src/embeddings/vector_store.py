"""Vector store implementation using Redis with RediSearch."""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from ..redis.client import RedisClient
from .embeddings import EmbeddingService


class VectorStore:
    """Vector store for semantic search using Redis."""

    def __init__(
        self,
        redis_client: RedisClient,
        embedding_service: EmbeddingService,
        index_name: str = "memory_vectors",
        prefix: str = "memory:",
    ):
        self.redis = redis_client
        self.embeddings = embedding_service
        self.index_name = index_name
        self.prefix = prefix
        self.vector_dim = self.embeddings.get_dimension()

    async def create_index(self, force: bool = False) -> None:
        """Create vector search index."""
        try:
            # Check if index exists
            info = await self.redis.ft(self.index_name).info()
            if not force:
                return  # Index already exists
            else:
                # Drop existing index
                await self.redis.ft(self.index_name).dropindex()
        except:
            pass  # Index doesn't exist, create it

        # Define index schema
        schema = [
            TextField("path", sortable=True),
            TextField("content", weight=1.0),
            TagField("type"),
            NumericField("timestamp", sortable=True),
            NumericField("importance", sortable=True),
            VectorField(
                "embedding",
                "HNSW",  # Using HNSW algorithm
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE",
                    "INITIAL_CAP": 10000,
                    "M": 16,  # Number of connections
                    "EF_CONSTRUCTION": 200,  # Build-time accuracy
                    "EF_RUNTIME": 10,  # Query-time accuracy
                },
            ),
        ]

        # Create index
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.JSON)

        await self.redis.ft(self.index_name).create_index(
            fields=schema, definition=definition
        )

    async def add_memory(
        self, path: str, content: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add memory with vector embedding."""
        # Prepare text for embedding
        text = self.embeddings.prepare_text_for_embedding(content)

        # Generate embedding
        embedding = await self.embeddings.embed_text(text)

        # Prepare document
        doc = {
            "path": path,
            "content": content if isinstance(content, str) else json.dumps(content),
            "type": metadata.get("type", "memory") if metadata else "memory",
            "timestamp": metadata.get("timestamp", 0) if metadata else 0,
            "importance": metadata.get("importance", 0.5) if metadata else 0.5,
            "embedding": embedding,
        }

        # Store in Redis
        key = f"{self.prefix}{path}"
        await self.redis.json().set(key, "$", doc)

    async def search_similar(
        self,
        query_text: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        hybrid: bool = False,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar memories using vector similarity.

        Returns list of (path, score, content) tuples.
        """
        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query_text)

        # Build query
        base_query = f"*=>[KNN {k} @embedding $vec AS score]"

        # Add filters if provided
        if filters:
            filter_parts = []
            if "type" in filters:
                filter_parts.append(f"@type:{{{filters['type']}}}")
            if "min_importance" in filters:
                filter_parts.append(f"@importance:[{filters['min_importance']} inf]")
            if "path_prefix" in filters:
                filter_parts.append(f"@path:{filters['path_prefix']}*")

            if filter_parts:
                base_query = f"({' '.join(filter_parts)}) {base_query}"

        # Add hybrid search if requested
        if hybrid:
            # Combine with text search
            text_query = self._prepare_text_query(query_text)
            base_query = f"({text_query}) {base_query}"

        # Create query object
        query = (
            Query(base_query)
            .sort_by("score", asc=True)  # Lower score = more similar
            .return_fields(
                "path", "content", "type", "timestamp", "importance", "score"
            )
            .dialect(2)
        )

        # Execute search
        params = {"vec": np.array(query_embedding, dtype=np.float32).tobytes()}
        results = await self.redis.ft(self.index_name).search(
            query, query_params=params
        )

        # Process results
        similar_memories = []
        for doc in results.docs:
            path = doc.path
            score = float(doc.score)
            content = (
                json.loads(doc.content) if doc.content.startswith("{") else doc.content
            )

            similar_memories.append(
                (
                    path,
                    score,
                    {
                        "content": content,
                        "type": doc.type,
                        "timestamp": float(doc.timestamp),
                        "importance": float(doc.importance),
                    },
                )
            )

        return similar_memories

    async def find_duplicates(
        self, content: Any, threshold: float = 0.95, limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Find potential duplicate memories based on similarity.

        Returns list of (path, similarity_score) tuples.
        """
        text = self.embeddings.prepare_text_for_embedding(content)
        results = await self.search_similar(text, k=limit)

        # Filter by threshold
        duplicates = [
            (path, 1 - score)  # Convert distance to similarity
            for path, score, _ in results
            if (1 - score) >= threshold
        ]

        return duplicates

    async def update_embedding(self, path: str, content: Any) -> None:
        """Update embedding for existing memory."""
        # Generate new embedding
        text = self.embeddings.prepare_text_for_embedding(content)
        embedding = await self.embeddings.embed_text(text)

        # Update in Redis
        key = f"{self.prefix}{path}"
        await self.redis.json().set(key, "$.embedding", embedding)

    async def batch_add_memories(
        self, memories: List[Tuple[str, Any, Optional[Dict[str, Any]]]]
    ) -> None:
        """Add multiple memories in batch."""
        # Prepare texts for embedding
        texts = [
            self.embeddings.prepare_text_for_embedding(content)
            for _, content, _ in memories
        ]

        # Generate embeddings in batch
        embeddings = await self.embeddings.embed_batch(texts)

        # Store documents
        pipe = self.redis.pipeline()
        for (path, content, metadata), embedding in zip(memories, embeddings):
            doc = {
                "path": path,
                "content": content if isinstance(content, str) else json.dumps(content),
                "type": metadata.get("type", "memory") if metadata else "memory",
                "timestamp": metadata.get("timestamp", 0) if metadata else 0,
                "importance": metadata.get("importance", 0.5) if metadata else 0.5,
                "embedding": embedding,
            }

            key = f"{self.prefix}{path}"
            pipe.json().set(key, "$", doc)

        await pipe.execute()

    def _prepare_text_query(self, text: str) -> str:
        """Prepare text for hybrid search query."""
        # Clean and escape special characters
        cleaned = text.replace('"', '\\"').replace("'", "\\'")

        # Split into words and create fuzzy search
        words = cleaned.split()
        if len(words) <= 3:
            # For short queries, search for exact phrase
            return f'@content:"{cleaned}"'
        else:
            # For longer queries, use fuzzy matching on individual words
            fuzzy_words = [
                f"@content:{word}~" for word in words[:5]
            ]  # Limit to 5 words
            return " ".join(fuzzy_words)
