"""Test memory models."""

import pytest
from datetime import datetime

from src.memory.models import MemoryType, SearchType, MemoryError


class TestMemoryType:
    """Test MemoryType enum."""

    def test_memory_type_values(self):
        """Test memory type enum values."""
        assert MemoryType.CONVERSATION.value == "conversation"
        assert MemoryType.DOCUMENT.value == "document"
        assert MemoryType.STATE.value == "state"
        assert MemoryType.INSIGHT.value == "insight"
        assert MemoryType.SYSTEM.value == "system"

    def test_from_content_conversation(self):
        """Test inferring conversation type from content."""
        content = {"message": "Hello", "role": "user"}
        assert MemoryType.from_content(content) == MemoryType.CONVERSATION
        
        content = {"conversation": "chat_id"}
        assert MemoryType.from_content(content) == MemoryType.CONVERSATION

    def test_from_content_state(self):
        """Test inferring state type from content."""
        content = {"checkpoint": "v1", "data": {}}
        assert MemoryType.from_content(content) == MemoryType.STATE
        
        content = {"state": "running", "progress": 0.5}
        assert MemoryType.from_content(content) == MemoryType.STATE

    def test_from_content_system(self):
        """Test inferring system type from content."""
        content = {"metric": "cpu_usage", "value": 0.8}
        assert MemoryType.from_content(content) == MemoryType.SYSTEM
        
        content = {"stats": {"count": 100}, "system": "monitoring"}
        assert MemoryType.from_content(content) == MemoryType.SYSTEM

    def test_from_content_default_document(self):
        """Test default document type for unknown content."""
        content = {"title": "Document", "body": "Content"}
        assert MemoryType.from_content(content) == MemoryType.DOCUMENT
        
        content = "plain string content"
        assert MemoryType.from_content(content) == MemoryType.DOCUMENT


class TestSearchType:
    """Test SearchType enum."""

    def test_search_type_values(self):
        """Test search type enum values."""
        # Check if SearchType enum exists and has expected values
        assert hasattr(SearchType, "KEYWORD")
        assert hasattr(SearchType, "TIME_RANGE")
        assert hasattr(SearchType, "SEMANTIC")
        assert hasattr(SearchType, "HYBRID")
        assert hasattr(SearchType, "EXACT")


class TestMemoryError:
    """Test MemoryError exception."""

    def test_memory_error_creation(self):
        """Test creating MemoryError."""
        error = MemoryError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)