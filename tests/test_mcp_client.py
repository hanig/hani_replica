"""Tests for MCP client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.mcp.client import (
    MCPClient,
    MCPClientManager,
    MCPServerConfig,
    MCPTool,
    COMMON_SERVERS,
)


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_basic_config(self):
        """Test creating basic config."""
        config = MCPServerConfig(
            name="test-server",
            command="python",
            args=["-m", "test_server"],
        )

        assert config.name == "test-server"
        assert config.command == "python"
        assert config.args == ["-m", "test_server"]
        assert config.env is None
        assert config.description == ""

    def test_full_config(self):
        """Test creating config with all fields."""
        config = MCPServerConfig(
            name="test-server",
            command="node",
            args=["server.js"],
            env={"API_KEY": "secret"},
            description="Test server",
        )

        assert config.env == {"API_KEY": "secret"}
        assert config.description == "Test server"


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_create_tool(self):
        """Test creating tool."""
        tool = MCPTool(
            name="search",
            description="Search for content",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            server_name="test-server",
        )

        assert tool.name == "search"
        assert tool.description == "Search for content"
        assert tool.server_name == "test-server"
        assert "properties" in tool.input_schema


class TestMCPClient:
    """Tests for MCPClient class."""

    @pytest.fixture
    def client(self):
        """Create MCP client instance."""
        return MCPClient()

    def test_init(self, client):
        """Test client initialization."""
        assert client._servers == {}
        assert client._sessions == {}
        assert client._tools == {}

    def test_register_server(self, client):
        """Test registering a server."""
        config = MCPServerConfig(
            name="test",
            command="python",
            args=["test.py"],
        )

        client.register_server(config)

        assert "test" in client._servers
        assert client._servers["test"] == config

    def test_unregister_server(self, client):
        """Test unregistering a server."""
        config = MCPServerConfig(name="test", command="python")
        client.register_server(config)

        result = client.unregister_server("test")

        assert result is True
        assert "test" not in client._servers

    def test_unregister_nonexistent_server(self, client):
        """Test unregistering server that doesn't exist."""
        result = client.unregister_server("nonexistent")
        assert result is False

    def test_unregister_removes_tools(self, client):
        """Test that unregistering removes server's tools."""
        config = MCPServerConfig(name="test", command="python")
        client.register_server(config)

        # Add a tool manually for this server
        tool = MCPTool(
            name="search",
            description="Search",
            input_schema={},
            server_name="test",
        )
        client._tools["test:search"] = tool

        client.unregister_server("test")

        assert "test:search" not in client._tools

    def test_list_tools_empty(self, client):
        """Test listing tools when none registered."""
        tools = client.list_tools()
        assert tools == []

    def test_list_tools(self, client):
        """Test listing all tools."""
        tool1 = MCPTool(
            name="search",
            description="Search",
            input_schema={},
            server_name="server1",
        )
        tool2 = MCPTool(
            name="fetch",
            description="Fetch",
            input_schema={},
            server_name="server2",
        )

        client._tools["server1:search"] = tool1
        client._tools["server2:fetch"] = tool2

        tools = client.list_tools()

        assert len(tools) == 2

    def test_list_tools_by_server(self, client):
        """Test listing tools filtered by server."""
        tool1 = MCPTool(name="t1", description="", input_schema={}, server_name="s1")
        tool2 = MCPTool(name="t2", description="", input_schema={}, server_name="s1")
        tool3 = MCPTool(name="t3", description="", input_schema={}, server_name="s2")

        client._tools["s1:t1"] = tool1
        client._tools["s1:t2"] = tool2
        client._tools["s2:t3"] = tool3

        tools = client.list_tools("s1")

        assert len(tools) == 2
        assert all(t.server_name == "s1" for t in tools)

    def test_get_tool(self, client):
        """Test getting a specific tool."""
        tool = MCPTool(name="search", description="", input_schema={}, server_name="s1")
        client._tools["s1:search"] = tool

        result = client.get_tool("s1:search")

        assert result == tool

    def test_get_tool_not_found(self, client):
        """Test getting nonexistent tool."""
        result = client.get_tool("unknown:tool")
        assert result is None

    def test_get_server_status(self, client):
        """Test getting server status."""
        config = MCPServerConfig(
            name="test",
            command="python",
            description="Test server",
        )
        client.register_server(config)

        # Add a tool
        tool = MCPTool(name="t1", description="", input_schema={}, server_name="test")
        client._tools["test:t1"] = tool

        status = client.get_server_status()

        assert "test" in status
        assert status["test"]["registered"] is True
        assert status["test"]["connected"] is False
        assert status["test"]["command"] == "python"
        assert status["test"]["description"] == "Test server"
        assert status["test"]["tools_count"] == 1

    @pytest.mark.asyncio
    async def test_connect_unregistered_server(self, client):
        """Test connecting to unregistered server."""
        result = await client.connect("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, client):
        """Test calling nonexistent tool raises error."""
        with pytest.raises(ValueError, match="Tool not found"):
            await client.call_tool("unknown:tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, client):
        """Test calling tool when server not connected raises error."""
        tool = MCPTool(name="t1", description="", input_schema={}, server_name="test")
        client._tools["test:t1"] = tool

        with pytest.raises(RuntimeError, match="Server not connected"):
            await client.call_tool("test:t1", {})

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnecting from a server."""
        # Add mock session and tools
        client._sessions["test"] = MagicMock()
        tool = MCPTool(name="t1", description="", input_schema={}, server_name="test")
        client._tools["test:t1"] = tool

        await client.disconnect("test")

        assert "test" not in client._sessions
        assert "test:t1" not in client._tools

    @pytest.mark.asyncio
    async def test_disconnect_all(self, client):
        """Test disconnecting from all servers."""
        client._sessions["s1"] = MagicMock()
        client._sessions["s2"] = MagicMock()
        client._tools["s1:t1"] = MagicMock()
        client._tools["s2:t2"] = MagicMock()

        await client.disconnect_all()

        assert client._sessions == {}
        assert client._tools == {}


class TestMCPClientManager:
    """Tests for MCPClientManager singleton."""

    def teardown_method(self):
        """Reset singleton after each test."""
        MCPClientManager._instance = None

    def test_get_client_creates_instance(self):
        """Test that get_client creates a new instance."""
        client = MCPClientManager.get_client()

        assert client is not None
        assert isinstance(client, MCPClient)

    def test_get_client_returns_same_instance(self):
        """Test that get_client returns singleton."""
        client1 = MCPClientManager.get_client()
        client2 = MCPClientManager.get_client()

        assert client1 is client2

    @patch("asyncio.get_event_loop")
    def test_reset_clears_instance(self, mock_get_loop):
        """Test that reset clears the instance."""
        mock_loop = MagicMock()
        mock_loop.run_until_complete = MagicMock()
        mock_get_loop.return_value = mock_loop

        client = MCPClientManager.get_client()
        MCPClientManager.reset()

        assert MCPClientManager._instance is None

        # Next call should create new instance
        client2 = MCPClientManager.get_client()
        assert client2 is not client


class TestCommonServers:
    """Tests for predefined server configurations."""

    def test_common_servers_exist(self):
        """Test that common servers are defined."""
        assert "filesystem" in COMMON_SERVERS
        assert "brave-search" in COMMON_SERVERS
        assert "github" in COMMON_SERVERS
        assert "google-drive" in COMMON_SERVERS
        assert "slack" in COMMON_SERVERS

    def test_filesystem_server_config(self):
        """Test filesystem server configuration."""
        config = COMMON_SERVERS["filesystem"]

        assert config.name == "filesystem"
        assert config.command == "npx"
        assert "@modelcontextprotocol/server-filesystem" in config.args

    def test_brave_search_server_config(self):
        """Test Brave search server configuration."""
        config = COMMON_SERVERS["brave-search"]

        assert config.name == "brave-search"
        assert config.command == "npx"

    def test_github_server_config(self):
        """Test GitHub server configuration."""
        config = COMMON_SERVERS["github"]

        assert config.name == "github"
        assert "@modelcontextprotocol/server-github" in config.args


class TestMCPClientIntegration:
    """Integration tests for MCP client (mocked)."""

    @pytest.fixture
    def client(self):
        """Create client with registered server."""
        client = MCPClient()
        config = MCPServerConfig(
            name="test-server",
            command="python",
            args=["-m", "test_server"],
        )
        client.register_server(config)
        return client

    @pytest.mark.asyncio
    @patch("src.mcp.client.stdio_client")
    @patch("src.mcp.client.ClientSession")
    async def test_connect_and_discover_tools(
        self, mock_session_class, mock_stdio_client, client
    ):
        """Test connecting to a server and discovering tools."""
        # Set up mocks
        mock_session = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "discovered_tool"
        mock_tool.description = "A discovered tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_result
        mock_session.initialize.return_value = None

        # Create async context manager mocks
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None
        mock_session_class.return_value = mock_session_cm

        mock_stdio_cm = AsyncMock()
        mock_stdio_cm.__aenter__.return_value = (AsyncMock(), AsyncMock())
        mock_stdio_cm.__aexit__.return_value = None
        mock_stdio_client.return_value = mock_stdio_cm

        # Connect
        result = await client.connect("test-server")

        assert result is True
        mock_session.initialize.assert_called_once()
        mock_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.mcp.client.stdio_client")
    async def test_connect_handles_error(self, mock_stdio_client, client):
        """Test that connect handles errors gracefully."""
        mock_stdio_client.side_effect = Exception("Connection failed")

        result = await client.connect("test-server")

        assert result is False
