# tests/test_phase2.py
"""
Tests for Phase 2 Coding Pipeline tools.
"""
import sys
import os
import unittest
import tempfile
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.edit_file import edit_file
from tools.deps import _extract_imports
from tools.workspace import WORKSPACE_DIR

class TestPhase2Tools(unittest.IsolatedAsyncioTestCase):

    async def test_edit_file_basic(self):
        # Setup a temporary file in the workspace
        test_file = "test_edit_file_temp.txt"
        full_path = os.path.join(WORKSPACE_DIR, test_file)
        
        with open(full_path, "w") as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")

        try:
            # Replace line 3-4 with "new line A\nnew line B"
            res = await edit_file(test_file, start_line=3, end_line=4, new_content="new 3\nnew 4\n")
            self.assertIn("✅ Edited", res)

            with open(full_path, "r") as f:
                content = f.read()

            expected = "line 1\nline 2\nnew 3\nnew 4\nline 5\n"
            self.assertEqual(content, expected)

        finally:
            if os.path.exists(full_path):
                os.remove(full_path)

    async def test_edit_file_out_of_bounds(self):
        test_file = "test_edit_fail.txt"
        full_path = os.path.join(WORKSPACE_DIR, test_file)
        
        with open(full_path, "w") as f:
            f.write("line 1\nline 2\n")

        try:
            # Start line > total lines
            res = await edit_file(test_file, start_line=5, end_line=6, new_content="x\n")
            self.assertIn("❌ start_line", res)
            
            # Start line < 1
            res = await edit_file(test_file, start_line=0, end_line=1, new_content="x\n")
            self.assertIn("❌ start_line must be >= 1", res)

        finally:
            if os.path.exists(full_path):
                os.remove(full_path)


class TestDepsExtraction(unittest.TestCase):

    def test_extract_imports(self):
        source = """
import os
import sys
from collections import defaultdict
import datetime as dt

def my_func():
    import json
    from math import sqrt
    pass
"""
        imports = _extract_imports(source)
        expected = {"os", "sys", "collections", "datetime", "json", "math"}
        self.assertEqual(imports, expected)

    def test_extract_imports_syntax_error_fallback(self):
        # If code has a syntax error, it falls back to regex
        source = """
import valid_package
from another import thing
def unclosed_parenthesis(
import fallback_works
"""
        imports = _extract_imports(source)
        # Regex backend might not catch `from another import thing` effectively,
        # but it definitely catches `import valid_package` and `import fallback_works`.
        self.assertIn("valid_package", imports)
        self.assertIn("another", imports)   # regex: from another -> group(1) = another
        self.assertIn("fallback_works", imports)


if __name__ == "__main__":
    unittest.main()
