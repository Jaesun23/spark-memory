# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Project
```bash
# Install and start Redis Stack (required)
brew tap redis-stack/redis-stack
brew install redis-stack

# Option 1: Use the setup script (recommended)
./setup/redis-setup.sh

# Option 2: Manual start with default settings
redis-stack-server

# Note: redis-stack-server ignores 'dir' in config files
# Use setup script for custom data directories

# Run as MCP server (recommended)
uvx spark-memory

# Or with Python
python -m spark_memory

# With debug logging
LOG_LEVEL=DEBUG python -m spark_memory
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov

# Run specific test categories
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
```

### Code Quality
```bash
# Type checking
mypy src/

# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Development Setup
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## High-Level Architecture

### LRMM (LangGraph + Redis + MCP + Memory) System
This project implements a memory management system with three core commands:
- `m_memory`: All memory operations (save, get, search, update, delete, consolidate, lifecycle)
- `m_state`: State and checkpoint management integrated with LangGraph
- `m_admin`: System administration including security features

### Key Components

1. **MCP Server Layer** (`src/mcp_server/`)
   - FastMCP-based server exposing tools via Model Context Protocol
   - Entry point for all memory operations
   - Handles request routing and response serialization

2. **Memory Engine** (`src/memory/engine.py`)
   - Central routing hub using modular action-based architecture
   - Delegates to specialized action handlers:
     - `BasicActions`: save, get, update, delete
     - `SearchActions`: keyword, time_range, similar, hybrid search
     - `ConsolidateActions`: memory consolidation
     - `LifecycleActions`: importance-based TTL management
     - `HelpActions`: contextual help system

3. **Redis Client Layer** (`src/redis/client.py`)
   - Async Redis Stack wrapper with connection pooling
   - Provides access to Redis modules: JSON, Search, Streams, TimeSeries

4. **Data Organization**
   - Time-based paths: `YYYY-MM-DD/category/subcategory`
   - Key patterns:
     - Conversations: `conv:{date}:{session}` (Streams)
     - Documents: `doc:{category}:{id}` (JSON + Vector)
     - States: `state:{project}:{checkpoint}` (JSON)
     - Metadata: `meta:{type}:{id}` (Hash)

### Security Features (Optional)
When `enable_security=True`:
- Field-level encryption for sensitive data
- Role-based access control (RBAC)
- API key management with rate limiting
- Comprehensive audit logging
- Compliance reporting

### Important Patterns

1. **Error Handling**: The system provides helpful error messages with usage examples
2. **Type Safety**: All source code requires type hints (enforced by mypy)
3. **Async First**: All operations are async using `redis.asyncio`
4. **Modular Actions**: Each action type is self-contained in its own module

## Working with Tests

Tests are excluded from mypy checking but should still maintain good structure:
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Use pytest fixtures for common setup
- Mock Redis operations for unit tests

## Redis Stack Requirements

Ensure Redis Stack is running with these modules:
- RedisJSON
- RediSearch
- RedisTimeSeries
- RedisStream

Check module availability:
```bash
redis-cli MODULE LIST
```