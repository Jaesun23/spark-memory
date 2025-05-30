# Spark Memory ✨

[![PyPI version](https://badge.fury.io/py/spark-memory.svg)](https://badge.fury.io/py/spark-memory)
[![Python Version](https://img.shields.io/pypi/pyversions/spark-memory.svg)](https://pypi.org/project/spark-memory/)
[![Tests](https://github.com/Jaesun23/spark-memory/actions/workflows/test.yml/badge.svg)](https://github.com/Jaesun23/spark-memory/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Next-generation AI memory system - Combining Redis speed with LangGraph intelligence to create LRMM

## 🚀 Project Overview

Spark Memory is a next-generation memory system integrating LangGraph, Redis Stack, and MCP (Model Context Protocol). It replaces file-based hierarchical memory with faster, smarter, and more scalable memory management.

### Core Features
- ⚡ **Lightning Fast**: Redis Stack-based with millisecond response times
- 🧠 **Intelligent Memory**: Vector search and semantic consolidation
- 🔗 **LRMM Advanced Features**: CrossMemoryBridge, MemoryAnalytics fully implemented
- 🔒 **Enterprise Security**: Field-level encryption, RBAC, audit logging
- 📈 **Infinite Scalability**: Distributed environment support
- 🎯 **Easy Deployment**: One-line execution via uvx

## 🛠️ Technology Stack

- **Python 3.11+** (Latest version recommended)
- **Redis Stack 7.2.0+** (JSON, Search, TimeSeries modules required)
- **FastMCP** (Model Context Protocol server)
- **LangGraph** (State management and workflow)

## 📁 Project Structure

```
spark-memory/
├── src/
│   ├── mcp_server/     # MCP server implementation
│   ├── memory/         # Memory engine core
│   ├── rag/            # RAG pipeline with LRMM features
│   ├── redis/          # Redis client wrapper
│   ├── security/       # Security features (encryption, RBAC, audit)
│   └── utils/          # Common utilities
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
└── docs/               # Project documentation
```

## 🏃‍♂️ Quick Start

### 1. Install Spark Memory

```bash
# Using uvx (recommended)
uvx spark-memory

# Or with pip
pip install spark-memory
python -m spark_memory
```

### 2. Install and Setup Redis Stack

#### macOS
```bash
# Install via Homebrew
brew tap redis-stack/redis-stack
brew install redis-stack

# Automatic setup (recommended)
./setup/redis-setup.sh
```

#### Ubuntu/Debian
```bash
# Add repository
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

# Install
sudo apt-get update
sudo apt-get install redis-stack-server

# Start
redis-stack-server
```

#### Windows
Download from [Redis Stack Official Downloads](https://redis.io/download/#redis-stack-downloads)

> ⚠️ **Important**: `redis-stack-server` ignores the `dir` directive in config files.
> Use the setup script for custom data directories.
> 
> ⚠️ **Additional constraint**: When using Redis Stack, `appenddirname` cannot be configured.
> The AOF directory name must use the default `appendonlydir`.
> Setting this causes module conflicts and prevents Redis from starting.
> 
> See the [Redis Stack Configuration Guide](docs/REDIS_STACK_CONFIGURATION.md) for details.

#### Manual execution
```bash
# General execution (uses default data directory)
redis-stack-server

# Custom data directory execution (recommended)
# The setup/redis-setup.sh script automates this
```

### 3. Configure with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "spark-memory": {
      "command": "uvx",
      "args": ["spark-memory"],
      "env": {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## 🧪 Development Setup

### Development Environment

```bash
# 1. Clone repository
git clone https://github.com/Jaesun23/spark-memory.git
cd spark-memory

# 2. Install development dependencies (using uv)
uv pip install -e ".[dev]"

# 3. Environment variable setup
cp .env.example .env
# Edit .env file to add necessary settings

# 4. Run tests
pytest tests/
```

### Environment Variables

`.env` file example:
```bash
# Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # Required for production
REDIS_DB=0
REDIS_KEY_PREFIX=spark_memory

# Security settings
ENCRYPTION_KEY=your-32-byte-encryption-key-here
ENABLE_SECURITY=true  # Recommended true for production

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # Recommended json for production
```

### Test Execution

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires Redis Stack)
pytest tests/integration/ -v -m integration

# Full tests with coverage
pytest --cov=src tests/

# LRMM feature testing
python test_cross_memory_simple.py
python test_advanced_cross_memory.py
python test_memory_analytics.py
```

## 🔐 Security Features

Spark Memory provides enterprise-grade security features:

### Encryption
- **AES-256-GCM**: For high-security data
- **Fernet**: Symmetric encryption for general data
- **Field-level encryption**: Automatic detection and encryption of sensitive fields

### Access Control
- **RBAC**: 5 predefined roles (Viewer, User, Editor, Admin, System)
- **API key management**: Includes rate limiting
- **Fine-grained permissions**: Read, write, delete, search, etc.

### Audit Logging
- Audit trail for all operations
- Anomaly detection (brute force attacks, privilege escalation, data exfiltration)
- Structured JSON log format

## 📋 MCP Tools List

### m_memory
Unified memory management tool
- **Basic operations**: Create, read, update, delete, search
- **LRMM Advanced Features**: Conversation-document auto-linking, cross-memory search, memory graph analysis
- **Memory consolidation**: Path/duplicate/time-based consolidation
- **Lifecycle management**: Importance evaluation, archive, restore

**LRMM Advanced Actions** (fully implemented in this project):
- `link_conversation`: Automatic conversation-document linking
- `find_cross_memory`: Cross-memory search (basic/advanced mode)
- `analyze_memory_graph`: Memory graph analysis
- `suggest_connections`: AI connection suggestions
- `get_memory_insights`: Individual memory insights

### m_state
State and checkpoint management
- Checkpoint creation/restoration
- State update and query
- LangGraph integration support

### m_admin
System management and security integration
- System: Status check, backup, data cleanup
- Security: Principal creation, permission management, API keys, audit logs

### m_assistant
Natural language command processing
- Save/search/summary/analysis command understanding
- Automatic category classification
- Time expression recognition

## 🤝 Contributing

1. Fork this repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m '✨ Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

### Commit Message Rules

- ✨ New feature
- 🐛 Bug fix
- 📝 Documentation update
- 🔧 Configuration change
- ✅ Add/modify tests
- ♻️ Refactoring
- 🔒 Security related

## 📄 License

MIT License - Feel free to use!

## 👥 Contributors

- **Jason** - Project conception and direction
- **Claude** - LRMM advanced feature implementation and documentation

---

> "Simple interface, powerful backend, intelligent memory" - Spark Memory Philosophy ✨