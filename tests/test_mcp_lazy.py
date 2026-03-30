"""Tests for MCP on-demand lazy connection."""
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestMCPLazyConnect(unittest.TestCase):

    def test_non_mcp_tool_not_affected(self):
        """Regular tools should not trigger MCP lazy connect."""
        from src.tools import execute_tool
        result = run_async(execute_tool("nonexistent_tool"))
        self.assertIn("Unknown tool", result)

    def test_mcp_tool_triggers_lazy_connect(self):
        """mcp_* tool should attempt lazy connection."""
        mock_config = {
            "fetch": {
                "command": ["npx", "-y", "@anthropic-ai/mcp-fetch@latest"],
                "auto_connect": False,
            }
        }

        mock_client = MagicMock()
        mock_client.connections = {}
        mock_client.connect_stdio = AsyncMock()
        mock_client.register_all_tools = MagicMock()

        # The __init__.py re-exports mcp_client (the singleton), shadowing the module.
        # Access the actual module via sys.modules.
        mcp_mod = sys.modules["src.tools.mcp_client"]
        original_load = mcp_mod.load_mcp_config
        original_client = mcp_mod.mcp_client

        mcp_mod.load_mcp_config = lambda *a, **kw: mock_config
        mcp_mod.mcp_client = mock_client
        try:
            from src.tools import execute_tool
            # This will try to lazy-connect via the mocked client
            # Tool won't actually be in registry after mock, so we get Unknown tool
            result = run_async(execute_tool("mcp_fetch_fetch_url", url="https://example.com"))
            # Verify lazy connect was attempted
            mock_client.connect_stdio.assert_called_once()
            mock_client.register_all_tools.assert_called_once()
        finally:
            mcp_mod.load_mcp_config = original_load
            mcp_mod.mcp_client = original_client

    def test_list_mcp_servers(self):
        from src.tools import list_mcp_servers
        servers = list_mcp_servers()
        # Should return dict (may be empty if no mcp.yaml)
        self.assertIsInstance(servers, dict)


if __name__ == "__main__":
    unittest.main()
