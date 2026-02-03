#!/usr/bin/env python3
"""Entry point for running the MCP server.

This script starts the MCP server that exposes Hani Replica's capabilities
as tools that can be used by Claude Desktop, Cursor, and other MCP clients.

Usage:
    python scripts/run_mcp_server.py

To connect from Claude Desktop, add to your claude_desktop_config.json:
{
    "mcpServers": {
        "hani-replica": {
            "command": "python",
            "args": ["/path/to/hani_replica/scripts/run_mcp_server.py"],
            "env": {
                "OPENAI_API_KEY": "your-key",
                "ANTHROPIC_API_KEY": "your-key",
                "GITHUB_TOKEN": "your-token"
            }
        }
    }
}
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LOG_FILE, LOG_LEVEL, ensure_directories


def main():
    """Main entry point."""
    # Ensure directories exist
    ensure_directories()

    # Configure logging (to file only, stdout is for MCP protocol)
    log_level = getattr(logging, LOG_LEVEL)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Hani Replica MCP server...")

    # Import and run server
    from src.mcp import run_server

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
    except Exception as e:
        logger.error(f"MCP server crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
