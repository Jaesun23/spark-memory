"""Basic tests for utils module."""

import pytest
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

from src.utils.time_path import TimePathGenerator


class TestTimePathGenerator:
    """Test time path generator."""

    @pytest.fixture
    def generator(self):
        """Create TimePathGenerator instance."""
        return TimePathGenerator("Asia/Seoul")

    def test_generate_path_basic(self, generator):
        """Test basic path generation."""
        # Test with specific timestamp
        dt = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("Asia/Seoul"))
        path = generator.generate_path(timestamp=dt)
        assert path == "2024-01-15/14:30:45"

    def test_generate_path_with_category(self, generator):
        """Test path generation with category."""
        dt = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("Asia/Seoul"))
        path = generator.generate_path(category="conversation", timestamp=dt)
        assert path == "2024-01-15/14:30:45/conversation"

    def test_generate_path_with_milliseconds(self, generator):
        """Test path generation with milliseconds."""
        dt = datetime(2024, 1, 15, 14, 30, 45, 123456, tzinfo=ZoneInfo("Asia/Seoul"))
        path = generator.generate_path(timestamp=dt, include_microseconds=True)
        assert path == "2024-01-15/14:30:45.123"

    def test_parse_path_basic(self, generator):
        """Test basic path parsing."""
        result = generator.parse_path("2024-01-15/14:30:45")
        assert result is not None
        assert result["date"] == date(2024, 1, 15)
        assert result["time"] == time(14, 30, 45)
        assert result["category"] is None
        assert result["has_milliseconds"] is False

    def test_parse_path_with_category(self, generator):
        """Test path parsing with category."""
        result = generator.parse_path("2024-01-15/14:30:45/conversation")
        assert result is not None
        assert result["category"] == "conversation"

    def test_parse_path_with_milliseconds(self, generator):
        """Test path parsing with milliseconds."""
        result = generator.parse_path("2024-01-15/14:30:45.123")
        assert result is not None
        assert result["has_milliseconds"] is True
        assert result["time"].microsecond == 123000

    def test_is_date_path(self, generator):
        """Test date path validation."""
        assert generator.is_date_path("2024-01-15")
        assert not generator.is_date_path("2024/01/15")
        assert not generator.is_date_path("invalid")

    def test_is_time_path(self, generator):
        """Test time path validation."""
        assert generator.is_time_path("14:30:45")
        assert generator.is_time_path("14:30:45.123")
        assert not generator.is_time_path("14:30")
        assert not generator.is_time_path("invalid")