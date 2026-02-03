"""MCP Client for connecting to external MCP servers.

This client allows the Hani Replica bot to connect to and use tools
from external MCP servers, extending its capabilities dynamically.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None
    description: str = ""


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


class MCPClient:
    """Client for connecting to external MCP servers.

    This client manages connections to multiple MCP servers and provides
    a unified interface for discovering and calling their tools.
    """

    def __init__(self):
        """Initialize the MCP client."""
        self._servers: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, MCPTool] = {}  # tool_name -> MCPTool

    def register_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration.

        Args:
            config: Server configuration.
        """
        self._servers[config.name] = config
        logger.info(f"Registered MCP server: {config.name}")

    def unregister_server(self, name: str) -> bool:
        """Unregister an MCP server.

        Args:
            name: Server name.

        Returns:
            True if unregistered, False if not found.
        """
        if name in self._servers:
            del self._servers[name]
            if name in self._sessions:
                del self._sessions[name]
            # Remove tools from this server
            self._tools = {
                k: v for k, v in self._tools.items()
                if v.server_name != name
            }
            logger.info(f"Unregistered MCP server: {name}")
            return True
        return False

    async def connect(self, server_name: str) -> bool:
        """Connect to an MCP server.

        Args:
            server_name: Name of the registered server.

        Returns:
            True if connected successfully, False otherwise.
        """
        if server_name not in self._servers:
            logger.error(f"Server not registered: {server_name}")
            return False

        config = self._servers[server_name]

        try:
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env=config.env,
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # Store the session
                    self._sessions[server_name] = session

                    # Discover and cache tools
                    await self._discover_tools(server_name, session)

                    logger.info(f"Connected to MCP server: {server_name}")
                    return True

        except Exception as e:
            logger.error(f"Error connecting to {server_name}: {e}")
            return False

    async def _discover_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Discover tools from a connected server.

        Args:
            server_name: Name of the server.
            session: Active client session.
        """
        try:
            result = await session.list_tools()

            for tool in result.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {},
                    server_name=server_name,
                )

                # Prefix tool name with server name to avoid conflicts
                full_name = f"{server_name}:{tool.name}"
                self._tools[full_name] = mcp_tool

                logger.debug(f"Discovered tool: {full_name}")

            logger.info(
                f"Discovered {len(result.tools)} tools from {server_name}"
            )

        except Exception as e:
            logger.error(f"Error discovering tools from {server_name}: {e}")

    def list_tools(self, server_name: str | None = None) -> list[MCPTool]:
        """List available tools.

        Args:
            server_name: Optional server name filter.

        Returns:
            List of available tools.
        """
        if server_name:
            return [
                t for t in self._tools.values()
                if t.server_name == server_name
            ]
        return list(self._tools.values())

    def get_tool(self, tool_name: str) -> MCPTool | None:
        """Get a tool by name.

        Args:
            tool_name: Full tool name (server:tool).

        Returns:
            MCPTool or None if not found.
        """
        return self._tools.get(tool_name)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on an MCP server.

        Args:
            tool_name: Full tool name (server:tool).
            arguments: Tool arguments.

        Returns:
            Tool result as dictionary.

        Raises:
            ValueError: If tool not found.
            RuntimeError: If server not connected.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        session = self._sessions.get(tool.server_name)
        if not session:
            raise RuntimeError(f"Server not connected: {tool.server_name}")

        try:
            # Extract the original tool name (without server prefix)
            original_name = tool_name.split(":", 1)[1] if ":" in tool_name else tool_name

            result = await session.call_tool(original_name, arguments)

            # Parse the result content
            if result.content:
                for content in result.content:
                    if hasattr(content, "text"):
                        try:
                            return json.loads(content.text)
                        except json.JSONDecodeError:
                            return {"text": content.text}

            return {"status": "success", "content": result.content}

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            server_name: Name of the server.
        """
        if server_name in self._sessions:
            del self._sessions[server_name]

        # Remove tools from this server
        self._tools = {
            k: v for k, v in self._tools.items()
            if v.server_name != server_name
        }

        logger.info(f"Disconnected from MCP server: {server_name}")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        self._sessions.clear()
        self._tools.clear()
        logger.info("Disconnected from all MCP servers")

    def get_server_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered servers.

        Returns:
            Dictionary mapping server names to status info.
        """
        status = {}

        for name, config in self._servers.items():
            tools_count = len([
                t for t in self._tools.values()
                if t.server_name == name
            ])

            status[name] = {
                "registered": True,
                "connected": name in self._sessions,
                "command": config.command,
                "description": config.description,
                "tools_count": tools_count,
            }

        return status


class MCPClientManager:
    """Manager for the MCP client singleton.

    Provides a global instance of the MCP client for use across the application.
    """

    _instance: MCPClient | None = None

    @classmethod
    def get_client(cls) -> MCPClient:
        """Get or create the MCP client instance.

        Returns:
            MCPClient instance.
        """
        if cls._instance is None:
            cls._instance = MCPClient()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the client instance."""
        if cls._instance:
            asyncio.get_event_loop().run_until_complete(
                cls._instance.disconnect_all()
            )
        cls._instance = None


# Predefined server configurations for common MCP servers
COMMON_SERVERS = {
    "filesystem": MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/"],
        description="Filesystem access via MCP",
    ),
    "brave-search": MCPServerConfig(
        name="brave-search",
        command="npx",
        args=["-y", "@anthropic/mcp-server-brave-search"],
        description="Web search via Brave Search API",
    ),
    "github": MCPServerConfig(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        description="GitHub API access via MCP",
    ),
    "google-drive": MCPServerConfig(
        name="google-drive",
        command="npx",
        args=["-y", "@anthropic/mcp-server-google-drive"],
        description="Google Drive access via MCP",
    ),
    "slack": MCPServerConfig(
        name="slack",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-slack"],
        description="Slack API access via MCP",
    ),
}
