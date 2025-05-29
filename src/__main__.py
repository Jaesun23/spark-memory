"""LRMM MCP Server 메인 진입점."""

import sys
from src.mcp_server.server import app


def main():
    """Main entry point for uvx execution."""
    # FastMCP는 stdio를 통해 통신
    app.run()


if __name__ == "__main__":
    main()
