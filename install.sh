#!/bin/bash
# Complete installation script for Memory One Spark

set -e  # Exit on error

echo "🚀 Memory One Spark - Complete Installation"
echo "=========================================="

# 1. Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed. Please install it first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# 2. Check if Redis Stack is installed
if ! brew list redis-stack &> /dev/null; then
    echo "📦 Installing Redis Stack..."
    brew tap redis-stack/redis-stack
    brew install redis-stack
else
    echo "✅ Redis Stack is already installed"
fi

# 3. Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 4. Run Redis setup
echo ""
echo "🔧 Setting up Redis Stack..."
./setup/redis-setup.sh

# 5. Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
uv pip install -e ".[dev]"

# 6. Test the installation
echo ""
echo "🧪 Testing installation..."
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is responding"
else
    echo "❌ Redis is not responding"
    exit 1
fi

# 7. Quick test of the MCP server
echo ""
echo "🧪 Testing MCP server..."
python -c "from src.mcp_server.server import app; print('✅ MCP server imports successfully')" || {
    echo "❌ Failed to import MCP server"
    exit 1
}

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📖 Next steps:"
echo "1. Add to Claude Desktop config:"
echo "   - Config location: ~/Library/Application Support/Claude/claude_desktop_config.json"
echo "   - Or use the provided mcp_config.json as reference"
echo ""
echo "2. Start using Memory One Spark:"
echo "   - Development: python -m src"
echo "   - Production: uvx memory-one-spark"
echo ""
echo "3. Useful commands:"
echo "   - Check Redis: redis-cli ping"
echo "   - View logs: tail -f ~/dotfiles/config/mcp/memory/redis-stack.log"
echo "   - Run tests: pytest tests/"