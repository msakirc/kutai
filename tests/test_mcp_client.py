# tests/test_mcp_client.py
"""
Tests for MCP (Model Context Protocol) client.

Covers:
  - connect and list_tools (mock subprocess)
  - call_tool (mock subprocess communication)
  - tool registration into TOOL_REGISTRY
  - disconnect and cleanup
  - SSE transport basics
  - config loading
"""
import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Helpers to build mock MCP responses
# ---------------------------------------------------------------------------

def _jsonrpc_response(result: dict, id: int = 1) -> bytes:
    """Build a JSON-RPC 2.0 success response as newline-terminated bytes."""
    return json.dumps({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    }).encode("utf-8") + b"\n"


def _jsonrpc_error(code: int, message: str, id: int = 1) -> bytes:
    """Build a JSON-RPC 2.0 error response."""
    return json.dumps({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message},
    }).encode("utf-8") + b"\n"


MOCK_INITIALIZE_RESULT = {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "test-server", "version": "0.1.0"},
}

MOCK_TOOLS_LIST_RESULT = {
    "tools": [
        {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    ],
}

MOCK_TOOL_CALL_RESULT = {
    "content": [
        {"type": "text", "text": "Hello from the file!"},
    ],
}


def _make_mock_process(responses: list[bytes]):
    """Create a mock asyncio subprocess that returns canned responses."""
    proc = MagicMock()
    proc.returncode = None

    # Mock stdin
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()

    # Mock stdout — return responses in order
    read_iter = iter(responses)

    async def mock_readline():
        try:
            return next(read_iter)
        except StopIteration:
            return b""

    proc.stdout = MagicMock()
    proc.stdout.readline = mock_readline

    # Mock stderr
    proc.stderr = MagicMock()

    # Mock terminate/wait/kill
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    return proc


# ---------------------------------------------------------------------------
# Test: Connect and List Tools
# ---------------------------------------------------------------------------

class TestConnectAndListTools(unittest.TestCase):
    """Test connecting to a stdio MCP server and discovering tools."""

    def test_connect_and_list_tools(self):
        """Verify that connect() initializes the server and discovers tools."""
        from src.tools.mcp_client import MCPClient

        client = MCPClient()

        # Three responses: initialize, notifications/initialized, tools/list
        mock_proc = _make_mock_process([
            _jsonrpc_response(MOCK_INITIALIZE_RESULT, id=1),
            _jsonrpc_response({}, id=2),  # notifications/initialized
            _jsonrpc_response(MOCK_TOOLS_LIST_RESULT, id=3),
        ])

        async def _test():
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                await client.connect("test_fs", ["fake-mcp-server"])

            tools = await client.list_tools()
            return tools

        tools = run_async(_test())

        # Should have discovered 2 tools
        self.assertEqual(len(tools), 2)

        names = {t["name"] for t in tools}
        self.assertIn("mcp_test_fs_read_file", names)
        self.assertIn("mcp_test_fs_write_file", names)

        # Check tool metadata
        read_tool = [t for t in tools if t["original_name"] == "read_file"][0]
        self.assertEqual(read_tool["server"], "test_fs")
        self.assertIn("path", read_tool["inputSchema"]["properties"])

    def test_list_tools_filtered_by_server(self):
        """list_tools(server_name=...) filters to one server."""
        from src.tools.mcp_client import MCPClient

        client = MCPClient()

        # Manually inject tools for two servers
        client._tools = {
            "mcp_a_tool1": {"server": "a", "original_name": "tool1",
                            "description": "A1", "inputSchema": {}},
            "mcp_b_tool2": {"server": "b", "original_name": "tool2",
                            "description": "B2", "inputSchema": {}},
        }

        async def _test():
            return await client.list_tools(server_name="a")

        tools = run_async(_test())
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "mcp_a_tool1")


# ---------------------------------------------------------------------------
# Test: Call Tool
# ---------------------------------------------------------------------------

class TestCallTool(unittest.TestCase):
    """Test calling a tool on a connected MCP server."""

    def test_call_tool(self):
        """call_tool sends tools/call and returns text content."""
        from src.tools.mcp_client import MCPClient, _StdioConnection

        client = MCPClient()

        # Set up a mock connection with call_tool response
        mock_proc = _make_mock_process([
            _jsonrpc_response(MOCK_TOOL_CALL_RESULT, id=1),
        ])
        conn = _StdioConnection(mock_proc)
        client.connections["test_fs"] = conn

        async def _test():
            return await client.call_tool("test_fs", "read_file", {"path": "/tmp/test.txt"})

        result = run_async(_test())
        self.assertEqual(result, "Hello from the file!")

    def test_call_tool_not_connected(self):
        """call_tool raises MCPConnectionError if server is not connected."""
        from src.tools.mcp_client import MCPClient, MCPConnectionError

        client = MCPClient()

        async def _test():
            await client.call_tool("nonexistent", "read_file", {"path": "x"})

        with self.assertRaises(MCPConnectionError):
            run_async(_test())

    def test_call_tool_error_response(self):
        """call_tool raises MCPError on JSON-RPC error response."""
        from src.tools.mcp_client import MCPClient, MCPError, _StdioConnection

        client = MCPClient()

        mock_proc = _make_mock_process([
            _jsonrpc_error(-32602, "Invalid params", id=1),
        ])
        conn = _StdioConnection(mock_proc)
        client.connections["test_fs"] = conn

        async def _test():
            await client.call_tool("test_fs", "read_file", {"path": "/bad"})

        with self.assertRaises(MCPError) as ctx:
            run_async(_test())
        self.assertIn("Invalid params", str(ctx.exception))


# ---------------------------------------------------------------------------
# Test: Tool Registration
# ---------------------------------------------------------------------------

class TestToolRegistration(unittest.TestCase):
    """Test registering MCP tools into the project TOOL_REGISTRY."""

    def test_tool_registration(self):
        """register_all_tools() adds tools to TOOL_REGISTRY with correct shape."""
        from src.tools.mcp_client import MCPClient
        from src.tools import TOOL_REGISTRY

        client = MCPClient()

        # Inject mock tools
        client._tools = {
            "mcp_test_read_file": {
                "server": "test",
                "original_name": "read_file",
                "description": "Read a file from the filesystem",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                },
            },
        }

        # Snapshot existing keys
        original_keys = set(TOOL_REGISTRY.keys())

        try:
            client.register_all_tools()

            # Tool should now be in TOOL_REGISTRY
            self.assertIn("mcp_test_read_file", TOOL_REGISTRY)

            entry = TOOL_REGISTRY["mcp_test_read_file"]
            self.assertIn("function", entry)
            self.assertIn("description", entry)
            self.assertIn("example", entry)
            self.assertIn("[MCP/test]", entry["description"])
            self.assertTrue(callable(entry["function"]))

            # Example should be valid JSON
            example = json.loads(entry["example"])
            self.assertEqual(example["tool"], "mcp_test_read_file")
        finally:
            # Clean up — remove injected tools
            client.unregister_all_tools()
            # Verify cleanup
            self.assertNotIn("mcp_test_read_file", TOOL_REGISTRY)

    def test_unregister_all_tools(self):
        """unregister_all_tools() removes MCP tools from TOOL_REGISTRY."""
        from src.tools.mcp_client import MCPClient
        from src.tools import TOOL_REGISTRY

        client = MCPClient()
        client._tools = {
            "mcp_srv_alpha": {
                "server": "srv",
                "original_name": "alpha",
                "description": "test",
                "inputSchema": {},
            },
        }

        client.register_all_tools()
        self.assertIn("mcp_srv_alpha", TOOL_REGISTRY)

        client.unregister_all_tools()
        self.assertNotIn("mcp_srv_alpha", TOOL_REGISTRY)


# ---------------------------------------------------------------------------
# Test: Disconnect
# ---------------------------------------------------------------------------

class TestDisconnect(unittest.TestCase):
    """Test disconnecting from MCP servers."""

    def test_disconnect(self):
        """disconnect() closes the connection and removes tools."""
        from src.tools.mcp_client import MCPClient, _StdioConnection

        client = MCPClient()

        # Set up a mock connection
        mock_proc = _make_mock_process([])
        conn = _StdioConnection(mock_proc)
        client.connections["test_fs"] = conn
        client._tools = {
            "mcp_test_fs_read_file": {
                "server": "test_fs",
                "original_name": "read_file",
                "description": "Read a file",
                "inputSchema": {},
            },
            "mcp_other_tool": {
                "server": "other",
                "original_name": "tool",
                "description": "Other tool",
                "inputSchema": {},
            },
        }

        async def _test():
            await client.disconnect("test_fs")

        run_async(_test())

        # Connection should be removed
        self.assertNotIn("test_fs", client.connections)

        # Only the test_fs tool should be removed
        self.assertNotIn("mcp_test_fs_read_file", client._tools)
        self.assertIn("mcp_other_tool", client._tools)

        # Process should have been terminated
        mock_proc.terminate.assert_called_once()

    def test_disconnect_nonexistent(self):
        """disconnect() handles missing server gracefully."""
        from src.tools.mcp_client import MCPClient

        client = MCPClient()

        async def _test():
            await client.disconnect("nonexistent")

        # Should not raise
        run_async(_test())

    def test_disconnect_all(self):
        """disconnect_all() closes all connections."""
        from src.tools.mcp_client import MCPClient, _StdioConnection

        client = MCPClient()

        for name in ["a", "b"]:
            proc = _make_mock_process([])
            client.connections[name] = _StdioConnection(proc)

        async def _test():
            await client.disconnect_all()

        run_async(_test())
        self.assertEqual(len(client.connections), 0)


# ---------------------------------------------------------------------------
# Test: Config Loading
# ---------------------------------------------------------------------------

class TestConfigLoading(unittest.TestCase):
    """Test MCP configuration loading from YAML."""

    def test_load_config_from_dict(self):
        """_expand_env_vars resolves ${VAR} references."""
        from src.tools.mcp_client import _expand_env_vars

        os.environ["TEST_MCP_TOKEN"] = "secret123"
        try:
            result = _expand_env_vars("Bearer ${TEST_MCP_TOKEN}")
            self.assertEqual(result, "Bearer secret123")
        finally:
            del os.environ["TEST_MCP_TOKEN"]

    def test_expand_env_missing_var(self):
        """_expand_env_vars returns empty string for missing vars."""
        from src.tools.mcp_client import _expand_env_vars

        result = _expand_env_vars("${NONEXISTENT_VAR_12345}")
        self.assertEqual(result, "")

    def test_load_mcp_config_no_file(self):
        """load_mcp_config returns {} when no config file exists."""
        from src.tools.mcp_client import load_mcp_config

        # Use a nonexistent path
        result = load_mcp_config("/tmp/nonexistent_mcp_config_xyz.yaml")
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Test: JSON-RPC Helpers
# ---------------------------------------------------------------------------

class TestJSONRPCHelpers(unittest.TestCase):
    """Test the low-level JSON-RPC message helpers."""

    def test_make_request(self):
        """_make_request produces valid JSON-RPC 2.0."""
        from src.tools.mcp_client import _make_request

        data = _make_request("tools/list", {"cursor": None})
        msg = json.loads(data.decode("utf-8").strip())
        self.assertEqual(msg["jsonrpc"], "2.0")
        self.assertEqual(msg["method"], "tools/list")
        self.assertIn("id", msg)
        self.assertEqual(msg["params"], {"cursor": None})

    def test_parse_response_success(self):
        """_parse_response extracts result from success response."""
        from src.tools.mcp_client import _parse_response

        result = _parse_response('{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}')
        self.assertEqual(result, {"tools": []})

    def test_parse_response_error(self):
        """_parse_response raises MCPError on error response."""
        from src.tools.mcp_client import _parse_response, MCPError

        with self.assertRaises(MCPError) as ctx:
            _parse_response(
                '{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}'
            )
        self.assertIn("Method not found", str(ctx.exception))


# ---------------------------------------------------------------------------
# Test: Wrapper Callable
# ---------------------------------------------------------------------------

class TestWrapperCallable(unittest.TestCase):
    """Test that the registered wrapper correctly delegates to call_tool."""

    def test_mcp_tool_wrapper(self):
        """_mcp_tool_wrapper passes kwargs through to call_tool."""
        from src.tools.mcp_client import MCPClient

        client = MCPClient()

        # Mock call_tool
        expected_result = "file contents here"
        client.call_tool = AsyncMock(return_value=expected_result)

        async def _test():
            return await client._mcp_tool_wrapper(
                "test_fs", "read_file", path="/tmp/test.txt"
            )

        result = run_async(_test())
        self.assertEqual(result, expected_result)
        client.call_tool.assert_called_once_with(
            "test_fs", "read_file", {"path": "/tmp/test.txt"}
        )


if __name__ == "__main__":
    unittest.main()
