"""Test configuration module."""

import os
import pytest
from unittest.mock import patch

from src.utils.config import get_config, RedisConfig


class TestRedisConfig:
    """Test Redis configuration."""

    def test_default_redis_config(self):
        """Test default Redis configuration values."""
        config = get_config()
        
        assert config.redis.url == "redis://localhost:6379"
        assert config.redis.password is None
        assert config.redis.max_connections == 50

    def test_redis_connection_url_without_password(self):
        """Test Redis connection URL generation without password."""
        config = get_config()
        url = config.redis.get_connection_url()
        
        assert url == "redis://localhost:6379"

    def test_redis_url_with_password(self):
        """Test Redis URL with password."""
        from src.utils.config import RedisConfig
        
        # Create a RedisConfig with password
        redis_config = RedisConfig()
        redis_config.url = "redis://localhost:6379"
        redis_config.password = "mypass"
        
        # Test URL generation with password
        url = redis_config.get_connection_url()
        assert ":mypass@" in url
        assert url == "redis://:mypass@localhost:6379"

    def test_redis_max_connections(self):
        """Test Redis max connections configuration."""
        from src.utils.config import RedisConfig
        
        # Test default value
        redis_config = RedisConfig()
        assert redis_config.max_connections == 50

    def test_config_singleton(self):
        """Test that config is a singleton."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2