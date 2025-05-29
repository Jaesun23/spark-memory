"""Context building for RAG responses."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ContextWindow:
    """Configuration for context window management."""

    max_tokens: int = 4096
    max_chunks: int = 10
    include_metadata: bool = True
    rerank: bool = True


class ContextBuilder:
    """Build context for LLM prompts from retrieved chunks."""

    def __init__(self, config: Optional[ContextWindow] = None):
        self.config = config or ContextWindow()

    def build_context(
        self,
        retrieved_chunks: List[Tuple[str, float, str]],
        query: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build context from retrieved chunks.

        Args:
            retrieved_chunks: List of (path, score, text) tuples
            query: Original query
            system_prompt: Optional system prompt

        Returns:
            Context dictionary with prompt and metadata
        """
        # Rerank if enabled (simple score-based for now)
        if self.config.rerank:
            chunks = sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)
        else:
            chunks = retrieved_chunks

        # Limit chunks
        chunks = chunks[: self.config.max_chunks]

        # Build context sections
        context_sections = []
        total_tokens = 0

        for i, (path, score, text) in enumerate(chunks):
            # Simple token estimation (4 chars per token)
            chunk_tokens = len(text) // 4

            if total_tokens + chunk_tokens > self.config.max_tokens:
                # Truncate if needed
                remaining_tokens = self.config.max_tokens - total_tokens
                if remaining_tokens > 100:  # Only include if meaningful
                    text = text[: remaining_tokens * 4]
                else:
                    break

            section = f"[Source {i+1}: {path} (relevance: {score:.2f})]\n{text}\n"
            context_sections.append(section)
            total_tokens += chunk_tokens

        # Build final context
        context = "\n---\n".join(context_sections)

        # Create prompt
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        prompt_parts.extend(
            ["Context from memory:", context, "", f"Query: {query}", "", "Response:"]
        )

        prompt = "\n".join(prompt_parts)

        # Build response
        response = {
            "prompt": prompt,
            "context": context,
            "query": query,
            "metadata": {
                "num_chunks": len(context_sections),
                "total_tokens": total_tokens,
                "sources": [
                    {"path": p, "score": s}
                    for p, s, _ in chunks[: len(context_sections)]
                ],
            },
        }

        if self.config.include_metadata:
            response["chunks"] = [
                {"path": p, "score": s, "text": t[:200] + "..." if len(t) > 200 else t}
                for p, s, t in chunks[: len(context_sections)]
            ]

        return response

    def format_response_with_sources(
        self, response: str, sources: List[Dict[str, Any]]
    ) -> str:
        """Format LLM response with source citations.

        Args:
            response: LLM response text
            sources: List of source metadata

        Returns:
            Formatted response with citations
        """
        if not sources:
            return response

        # Add source citations
        citations = []
        for i, source in enumerate(sources):
            citation = f"[{i+1}] {source['path']} (relevance: {source['score']:.2f})"
            citations.append(citation)

        formatted = f"{response}\n\n---\nSources:\n" + "\n".join(citations)

        return formatted

    def extract_answer_from_context(
        self, context: str, query: str, max_length: int = 500
    ) -> Optional[str]:
        """Extract direct answer from context if possible.

        This is a simple extraction - for production use,
        you'd want to use an LLM for this task.
        """
        # Simple keyword matching for now
        query_words = set(query.lower().split())

        # Find sentences containing query words
        sentences = context.split(".")
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                relevant_sentences.append(sentence.strip())

        if not relevant_sentences:
            return None

        # Join most relevant sentences up to max length
        answer = ". ".join(relevant_sentences)
        if len(answer) > max_length:
            answer = answer[:max_length] + "..."

        return answer
