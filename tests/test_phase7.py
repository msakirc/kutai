# tests/test_phase7.py
"""
Tests for Phase 7: Tool Ecosystem Expansion

  7.1 Diff/patch tool (search-and-replace)
  7.2 HTTP client tool
  7.3 File download tool
  7.4 Tool registry completeness
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── 7.1 Patch File Tool ─────────────────────────────────────────────────

class TestPatchFile(unittest.TestCase):
    """Test search-and-replace file editing."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

        # Create test file
        self.test_file = os.path.join(self.tmp_dir, "test.py")
        with open(self.test_file, "w") as f:
            f.write("def hello():\n    print('hello')\n\ndef world():\n    print('world')\n")

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_simple_replace(self):
        from tools.patch_file import patch_file
        result = run_async(patch_file(
            "test.py",
            "print('hello')",
            "print('hi')",
        ))
        self.assertIn("✅", result)
        with open(self.test_file) as f:
            content = f.read()
        self.assertIn("print('hi')", content)
        self.assertNotIn("print('hello')", content)

    def test_multiline_replace(self):
        from tools.patch_file import patch_file
        result = run_async(patch_file(
            "test.py",
            "def hello():\n    print('hello')",
            "def greet(name):\n    print(f'hello {name}')",
        ))
        self.assertIn("✅", result)
        with open(self.test_file) as f:
            content = f.read()
        self.assertIn("def greet(name):", content)

    def test_search_not_found(self):
        from tools.patch_file import patch_file
        result = run_async(patch_file(
            "test.py",
            "nonexistent text",
            "replacement",
        ))
        self.assertIn("❌", result)
        self.assertIn("not found", result)

    def test_empty_search_rejected(self):
        from tools.patch_file import patch_file
        result = run_async(patch_file("test.py", "", "replacement"))
        self.assertIn("❌", result)
        self.assertIn("empty", result)

    def test_file_not_found(self):
        from tools.patch_file import patch_file
        result = run_async(patch_file("nosuch.py", "x", "y"))
        self.assertIn("❌", result)
        self.assertIn("not found", result)

    def test_path_escape_blocked(self):
        from tools.patch_file import patch_file
        result = run_async(patch_file("../../etc/passwd", "x", "y"))
        self.assertIn("❌", result)

    def test_delete_block(self):
        from tools.patch_file import patch_file
        result = run_async(patch_file(
            "test.py",
            "\ndef world():\n    print('world')\n",
            "",
        ))
        self.assertIn("✅", result)
        with open(self.test_file) as f:
            content = f.read()
        self.assertNotIn("world", content)

    def test_duplicate_match_rejected(self):
        """Multiple matches should be rejected for safety."""
        # Create file with duplicate content
        dup_file = os.path.join(self.tmp_dir, "dup.py")
        with open(dup_file, "w") as f:
            f.write("x = 1\nx = 1\n")
        from tools.patch_file import patch_file
        result = run_async(patch_file("dup.py", "x = 1", "x = 2"))
        self.assertIn("❌", result)
        self.assertIn("2 times", result)


# ─── 7.2 HTTP Client Tool ────────────────────────────────────────────────

class TestHTTPClient(unittest.TestCase):
    """Test HTTP client tool structure."""

    def test_invalid_method(self):
        from tools.http_client import http_request
        result = run_async(http_request("INVALID", "https://example.com"))
        self.assertIn("❌", result)
        self.assertIn("Unsupported", result)

    def test_invalid_url(self):
        from tools.http_client import http_request
        result = run_async(http_request("GET", "not-a-url"))
        self.assertIn("❌", result)

    def test_format_response_success(self):
        from tools.http_client import _format_response
        result = _format_response(200, {"Content-Type": "application/json"}, '{"ok": true}')
        self.assertIn("✅", result)
        self.assertIn("200", result)
        self.assertIn("application/json", result)

    def test_format_response_error(self):
        from tools.http_client import _format_response
        result = _format_response(404, {}, "Not Found")
        self.assertIn("❌", result)
        self.assertIn("404", result)

    def test_format_response_truncates(self):
        from tools.http_client import _format_response, MAX_RESPONSE_SIZE
        long_body = "x" * (MAX_RESPONSE_SIZE + 1000)
        result = _format_response(200, {}, long_body)
        self.assertIn("truncated", result)
        self.assertLess(len(result), MAX_RESPONSE_SIZE + 500)


# ─── 7.3 File Download Tool ──────────────────────────────────────────────

class TestDownload(unittest.TestCase):
    """Test download tool structure."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_invalid_url(self):
        from tools.download import download_file
        result = run_async(download_file("not-a-url", "test.txt"))
        self.assertIn("❌", result)

    def test_empty_save_as(self):
        from tools.download import download_file
        result = run_async(download_file("https://example.com/file.txt", ""))
        self.assertIn("❌", result)

    def test_path_escape_blocked(self):
        from tools.download import download_file
        result = run_async(download_file(
            "https://example.com/file.txt", "../../etc/shadow"
        ))
        self.assertIn("❌", result)

    def test_human_size(self):
        from tools.download import _human_size
        self.assertEqual(_human_size(500), "500B")
        self.assertIn("KB", _human_size(1500))
        self.assertIn("MB", _human_size(5_000_000))


# ─── 7.4 Tool Registry ───────────────────────────────────────────────────

class TestToolRegistry(unittest.TestCase):
    """Verify new tools are registered."""

    def test_patch_file_registered(self):
        from tools import TOOL_REGISTRY
        self.assertIn("patch_file", TOOL_REGISTRY)
        self.assertIn("function", TOOL_REGISTRY["patch_file"])
        self.assertIn("description", TOOL_REGISTRY["patch_file"])
        self.assertIn("example", TOOL_REGISTRY["patch_file"])

    def test_http_request_registered(self):
        from tools import TOOL_REGISTRY
        self.assertIn("http_request", TOOL_REGISTRY)

    def test_download_file_registered(self):
        from tools import TOOL_REGISTRY
        self.assertIn("download_file", TOOL_REGISTRY)

    def test_tool_schemas_include_new_tools(self):
        from tools import TOOL_SCHEMAS
        schema_names = {
            s["function"]["name"] for s in TOOL_SCHEMAS
        }
        self.assertIn("patch_file", schema_names)
        self.assertIn("http_request", schema_names)
        self.assertIn("download_file", schema_names)

    def test_patch_file_schema_has_required_params(self):
        from tools import TOOL_SCHEMAS
        for s in TOOL_SCHEMAS:
            if s["function"]["name"] == "patch_file":
                params = s["function"]["parameters"]
                self.assertIn("filepath", params["properties"])
                self.assertIn("search_block", params["properties"])
                self.assertIn("replace_block", params["properties"])
                self.assertIn("filepath", params["required"])
                break
        else:
            self.fail("patch_file not in TOOL_SCHEMAS")


# ─── 7.5 Tool Files Exist ────────────────────────────────────────────────

class TestToolFiles(unittest.TestCase):
    """Verify new tool files exist."""

    def _tool_path(self, name):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools", name,
        )

    def test_patch_file_exists(self):
        self.assertTrue(os.path.isfile(self._tool_path("patch_file.py")))

    def test_http_client_exists(self):
        self.assertTrue(os.path.isfile(self._tool_path("http_client.py")))

    def test_download_exists(self):
        self.assertTrue(os.path.isfile(self._tool_path("download.py")))


# ─── 7.6 Execute Tool Integration ────────────────────────────────────────

class TestExecuteTool(unittest.TestCase):
    """Test execute_tool with new tools."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

        # Create test file
        with open(os.path.join(self.tmp_dir, "app.py"), "w") as f:
            f.write("x = 1\ny = 2\n")

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_execute_patch_file(self):
        from tools import execute_tool
        result = run_async(execute_tool(
            "patch_file",
            filepath="app.py",
            search_block="x = 1",
            replace_block="x = 42",
        ))
        self.assertIn("✅", result)

    def test_execute_unknown_tool(self):
        from tools import execute_tool
        result = run_async(execute_tool("nonexistent_tool"))
        self.assertIn("❌", result)
        self.assertIn("Unknown tool", result)


if __name__ == "__main__":
    unittest.main()
