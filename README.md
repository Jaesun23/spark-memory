# Spark Memory ✨

[![PyPI version](https://badge.fury.io/py/spark-memory.svg)](https://badge.fury.io/py/spark-memory)
[![Python Version](https://img.shields.io/pypi/pyversions/spark-memory.svg)](https://pypi.org/project/spark-memory/)
[![Tests](https://github.com/Jaesun23/spark-memory/actions/workflows/test.yml/badge.svg)](https://github.com/Jaesun23/spark-memory/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

High-performance MCP (Model Context Protocol) memory server powered by Redis Stack and LangGraph.

## Features

- ⚡ **Lightning Fast**: Redis Stack-based with millisecond response times
- 🧠 **Intelligent Memory**: Vector search and semantic consolidation
- 🔒 **Enterprise Security**: Field-level encryption, RBAC, audit logging
- 📈 **Scalable**: Distributed environment support
- 🎯 **Easy Deployment**: One-line execution via uvx

## Installation

### Requirements

- Python 3.11+
- Redis Stack 7.2.0+ (with JSON, Search, TimeSeries modules)

### Install via pip

```bash
pip install spark-memory
```

### Install Redis Stack

**macOS:**
```bash
brew tap redis-stack/redis-stack
brew install redis-stack
brew services start redis-stack
```

**Ubuntu/Debian:**
```bash
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-stack
```

**Docker:**
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

## Quick Start

### 1. Run as MCP Server

```bash
# Using uvx (recommended)
uvx spark-memory

# Or with Python
python -m spark_memory
```

### 2. Configure with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "spark-memory": {
      "command": "uvx",
      "args": ["spark-memory"]
    }
  }
}
```

### 3. Use in Claude

Once configured, you can use these commands in Claude:

- `m_memory`: Save, search, and manage memories
- `m_state`: Checkpoint and state management
- `m_admin`: System administration
- `m_assistant`: Natural language memory commands

## Architecture

Spark Memory uses a layered architecture:

1. **MCP Server Layer**: FastMCP server exposing tools via Model Context Protocol
2. **Memory Engine Layer**: Core business logic for memory operations
3. **Redis Client Layer**: Redis Stack wrapper with connection pooling
4. **Security Layer**: Encryption, access control, and audit logging

## Memory Operations

### Save Memory
```python
# In Claude, you can say:
# "Save this conversation about Python optimization"
# "Remember that the meeting is at 3pm tomorrow"
```

### Search Memory
```python
# "What did we discuss about Redis?"
# "Find all memories from last week"
```

### State Management
```python
# "Create a checkpoint for the current project"
# "Restore the previous state"
```

## Configuration

Environment variables:
- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_PASSWORD`: Redis password (optional)
- `LOG_LEVEL`: Logging level (default: INFO)

## Development

```bash
# Clone the repository
git clone https://github.com/Jaesun23/spark-memory
cd spark-memory

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with debug logging
LOG_LEVEL=DEBUG python -m spark_memory
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- Issues: [GitHub Issues](https://github.com/Jaesun23/spark-memory/issues)
- Discussions: [GitHub Discussions](https://github.com/Jaesun23/spark-memory/discussions)

---

Made with ❤️ by Jason and 1호