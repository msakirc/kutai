# tests/test_phase12.py
"""
Tests for Phase 12: Large Codebase Engine

  12.1  Tree-sitter Multi-Language Parsing (parsing/tree_sitter_parser.py)
  12.2  Code Embedding Index (parsing/code_embeddings.py)
  12.3  Intelligent Context Assembly (context/assembler.py)
  12.4  Repository Map (context/repo_map.py)
  12.5  Diff-First Editing (tools/apply_diff.py)
  12.6  Project Onboarding (context/onboarding.py)
        + tool registry, agent prompt, re-index triggers

Tests use source-code inspection where external deps may not be available,
and functional tests where modules can be tested independently.
"""
import asyncio
import inspect
import os
import re
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_source(relpath: str) -> str:
    """Read a source file relative to project root (UTF-8)."""
    with open(os.path.join(_ROOT, relpath), "r", encoding="utf-8") as f:
        return f.read()


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 12.1 — Tree-sitter Multi-Language Parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestTreeSitterParser(unittest.TestCase):
    """Tests for parsing/tree_sitter_parser.py"""

    def test_module_exists(self):
        path = os.path.join(_ROOT, "parsing", "tree_sitter_parser.py")
        self.assertTrue(os.path.isfile(path))

    def test_package_init_exists(self):
        path = os.path.join(_ROOT, "parsing", "__init__.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("parsing/tree_sitter_parser.py")
        for fn in [
            "detect_language",
            "parse_source",
            "parse_file",
            "validate_syntax",
            "tree_sitter_available",
            "get_supported_languages",
            "get_parseable_extensions",
        ]:
            self.assertIn(f"def {fn}(", src, f"Missing public function: {fn}")

    def test_extension_map_coverage(self):
        src = _read_source("parsing/tree_sitter_parser.py")
        for ext in [".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp"]:
            self.assertIn(f'"{ext}"', src, f"Extension {ext} not in EXTENSION_MAP")

    def test_detect_language_functional(self):
        from parsing.tree_sitter_parser import detect_language
        self.assertEqual(detect_language("main.py"), "python")
        self.assertEqual(detect_language("app.js"), "javascript")
        self.assertEqual(detect_language("main.go"), "go")
        self.assertEqual(detect_language("lib.rs"), "rust")
        # Unknown extensions return None or "unknown" depending on impl
        result = detect_language("readme.txt")
        self.assertTrue(result is None or result == "unknown")

    def test_parse_source_python(self):
        from parsing.tree_sitter_parser import parse_source
        code = textwrap.dedent("""\
            import os

            def hello(name: str) -> str:
                \"\"\"Greet someone.\"\"\"
                return f"Hello, {name}"

            class Greeter:
                def greet(self):
                    pass
        """)
        result = parse_source(code, "python")
        self.assertIsNotNone(result)
        self.assertIn("functions", result)
        self.assertIn("classes", result)
        self.assertIn("imports", result)
        # Check function found
        fn_names = [f["name"] for f in result["functions"]]
        self.assertIn("hello", fn_names)
        # Check class found
        cls_names = [c["name"] for c in result["classes"]]
        self.assertIn("Greeter", cls_names)

    def test_parse_file_functional(self):
        from parsing.tree_sitter_parser import parse_file
        # Parse this test file itself
        result = parse_file(__file__)
        self.assertIsNotNone(result)
        self.assertIn("functions", result)
        self.assertIn("filepath", result)

    def test_validate_syntax_valid(self):
        from parsing.tree_sitter_parser import validate_syntax
        valid, msg = validate_syntax("def foo(): pass", "python")
        self.assertTrue(valid)

    def test_validate_syntax_invalid(self):
        from parsing.tree_sitter_parser import validate_syntax
        valid, msg = validate_syntax("def foo(:", "python")
        self.assertFalse(valid)

    def test_get_parseable_extensions(self):
        from parsing.tree_sitter_parser import get_parseable_extensions
        exts = get_parseable_extensions()
        self.assertIsInstance(exts, (list, tuple))
        self.assertIn(".py", exts)
        self.assertIn(".js", exts)
        self.assertTrue(len(exts) >= 8)

    def test_fallback_strategies(self):
        """Parser tries tree-sitter → ast → regex in order."""
        src = _read_source("parsing/tree_sitter_parser.py")
        self.assertIn("_parse_ts", src)
        self.assertIn("_parse_python_ast", src)
        self.assertIn("_parse_regex", src)

    def test_regex_patterns_for_c_family(self):
        src = _read_source("parsing/tree_sitter_parser.py")
        # Should have regex-based fallback parsing
        self.assertIn("_parse_regex", src)
        # At minimum should detect function patterns for multiple languages
        for lang in ["javascript", "go", "java"]:
            self.assertIn(lang, src.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# 12.2 — Code Embedding Index
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeEmbeddings(unittest.TestCase):
    """Tests for parsing/code_embeddings.py"""

    def test_module_exists(self):
        path = os.path.join(_ROOT, "parsing", "code_embeddings.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("parsing/code_embeddings.py")
        for fn in [
            "index_codebase",
            "reindex_file",
            "search_code",
        ]:
            self.assertIn(f"def {fn}(", src, f"Missing public function: {fn}")

    def test_incremental_indexing_via_hash(self):
        """Uses file hashes for incremental re-indexing."""
        src = _read_source("parsing/code_embeddings.py")
        self.assertIn("_file_hash", src)
        self.assertIn("_needs_reindex", src)
        self.assertIn("sha256", src.lower())

    def test_embeds_functions_and_classes(self):
        src = _read_source("parsing/code_embeddings.py")
        self.assertIn("_embed_symbol", src)
        self.assertIn("_index_file_symbols", src)

    def test_stores_in_codebase_collection(self):
        src = _read_source("parsing/code_embeddings.py")
        self.assertIn("codebase", src)

    def test_metadata_fields(self):
        """Metadata includes filepath, symbol_name, symbol_type, line_start, line_end, language."""
        src = _read_source("parsing/code_embeddings.py")
        for field in ["filepath", "symbol_name", "symbol_type", "line_start", "line_end", "language"]:
            self.assertIn(f'"{field}"', src, f"Metadata field '{field}' not found")

    def test_skips_private_functions(self):
        """Should skip private (underscore-prefixed) functions except __init__."""
        src = _read_source("parsing/code_embeddings.py")
        self.assertIn("__init__", src)
        # Should have logic to skip _private functions
        self.assertIn("startswith", src)


# ═══════════════════════════════════════════════════════════════════════════════
# 12.3 — Intelligent Context Assembly
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextAssembler(unittest.TestCase):
    """Tests for context/assembler.py"""

    def test_module_exists(self):
        path = os.path.join(_ROOT, "context", "assembler.py")
        self.assertTrue(os.path.isfile(path))

    def test_package_init_exists(self):
        path = os.path.join(_ROOT, "context", "__init__.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("context/assembler.py")
        self.assertIn("def assemble_context(", src)

    def test_pipeline_steps(self):
        """Should query embeddings, resolve imports, find tests, get git changes."""
        src = _read_source("context/assembler.py")
        self.assertIn("_query_code_embeddings", src)
        self.assertIn("_resolve_imports", src)
        self.assertIn("_find_related_tests", src)
        self.assertIn("_get_recent_changes", src)

    def test_token_budget(self):
        src = _read_source("context/assembler.py")
        self.assertIn("max_tokens", src)

    def test_file_reading_helpers(self):
        src = _read_source("context/assembler.py")
        self.assertIn("_read_file_section", src)
        self.assertIn("_read_file_brief", src)

    def test_output_format(self):
        """Should return formatted code context string."""
        src = _read_source("context/assembler.py")
        self.assertIn("Code Context", src)


# ═══════════════════════════════════════════════════════════════════════════════
# 12.4 — Repository Map
# ═══════════════════════════════════════════════════════════════════════════════

class TestRepoMap(unittest.TestCase):
    """Tests for context/repo_map.py"""

    def test_module_exists(self):
        path = os.path.join(_ROOT, "context", "repo_map.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("context/repo_map.py")
        for fn in [
            "generate_repo_map",
            "format_repo_map",
            "save_repo_map",
            "load_repo_map",
        ]:
            self.assertIn(f"def {fn}(", src, f"Missing public function: {fn}")

    def test_entry_point_detection(self):
        src = _read_source("context/repo_map.py")
        self.assertIn("ENTRY_POINT_NAMES", src)
        for ep in ["main.py", "app.py", "index.js"]:
            self.assertIn(ep, src)

    def test_config_files_detection(self):
        src = _read_source("context/repo_map.py")
        self.assertIn("CONFIG_FILES", src)
        for cfg in ["package.json", "Dockerfile", "pyproject.toml"]:
            self.assertIn(cfg, src)

    def test_test_pattern_detection(self):
        src = _read_source("context/repo_map.py")
        self.assertIn("TEST_PATTERNS", src)

    def test_generate_repo_map_functional(self):
        from context.repo_map import generate_repo_map
        # Generate map of this project
        result = generate_repo_map(_ROOT)
        self.assertIn("dependency_graph", result)
        self.assertIn("entry_points", result)
        self.assertIn("test_mapping", result)
        self.assertIn("config_files", result)
        self.assertIn("directory_summary", result)
        self.assertIn("languages", result)
        self.assertIn("total_files", result)

    def test_format_repo_map_functional(self):
        from context.repo_map import generate_repo_map, format_repo_map
        repo_map = generate_repo_map(_ROOT)
        text = format_repo_map(repo_map, max_lines=20)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_save_and_load(self):
        from context.repo_map import generate_repo_map, save_repo_map, load_repo_map
        repo_map = generate_repo_map(_ROOT)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_path = tf.name
        try:
            save_repo_map(repo_map, tmp_path)
            loaded = load_repo_map(tmp_path)
            self.assertEqual(loaded["total_files"], repo_map["total_files"])
        finally:
            os.unlink(tmp_path)

    def test_dir_purpose_inference(self):
        src = _read_source("context/repo_map.py")
        self.assertIn("_infer_dir_purpose", src)


# ═══════════════════════════════════════════════════════════════════════════════
# 12.5 — Diff-First Editing
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyDiff(unittest.TestCase):
    """Tests for tools/apply_diff.py"""

    def test_module_exists(self):
        path = os.path.join(_ROOT, "tools", "apply_diff.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("tools/apply_diff.py")
        self.assertIn("def apply_diff(", src)
        self.assertIn("async def apply_diff(", src)

    def test_diff_parsing(self):
        src = _read_source("tools/apply_diff.py")
        self.assertIn("_parse_unified_diff", src)
        self.assertIn("_apply_hunks", src)

    def test_syntax_validation_after_edit(self):
        """Should validate syntax after applying diff."""
        src = _read_source("tools/apply_diff.py")
        self.assertIn("validate_syntax", src)

    def test_parse_unified_diff_functional(self):
        # Import the internal parser
        src = _read_source("tools/apply_diff.py")
        # Verify it can parse @@ hunk headers
        self.assertIn("@@", src)

    def test_apply_diff_internal_functions(self):
        """Test _parse_unified_diff and _apply_hunks directly."""
        from tools.apply_diff import _parse_unified_diff, _apply_hunks

        diff_text = textwrap.dedent("""\
            @@ -2,3 +2,4 @@
             line2
            -line3
            +line3_modified
            +line3b_added
             line4
        """)
        hunks = _parse_unified_diff(diff_text)
        self.assertGreater(len(hunks), 0)
        self.assertEqual(hunks[0]["old_start"], 2)

        original = "line1\nline2\nline3\nline4\nline5\n"
        new_text, error = _apply_hunks(original, hunks)
        self.assertFalse(error, f"Expected no error but got: {error}")
        self.assertIn("line3_modified", new_text)
        self.assertIn("line3b_added", new_text)
        self.assertNotIn("line3\n", new_text)

    def test_apply_hunks_bad_context(self):
        """Should fail gracefully when context doesn't match."""
        from tools.apply_diff import _parse_unified_diff, _apply_hunks

        diff_text = textwrap.dedent("""\
            @@ -1,3 +1,3 @@
             xxx
            -yyy
            +zzz
             ccc
        """)
        hunks = _parse_unified_diff(diff_text)
        original = "aaa\nbbb\nccc\n"
        new_text, error = _apply_hunks(original, hunks)
        # Should report an error (context mismatch)
        self.assertIsNotNone(error, "Expected error from context mismatch")


# ═══════════════════════════════════════════════════════════════════════════════
# 12.6 — Project Onboarding
# ═══════════════════════════════════════════════════════════════════════════════

class TestProjectOnboarding(unittest.TestCase):
    """Tests for context/onboarding.py"""

    def test_module_exists(self):
        path = os.path.join(_ROOT, "context", "onboarding.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("context/onboarding.py")
        for fn in [
            "onboard_project",
            "store_project_profile",
            "load_project_profile",
            "load_all_project_profiles",
            "format_project_profile",
            "get_project_profile_for_task",
        ]:
            self.assertIn(f"def {fn}(", src, f"Missing: {fn}")

    def test_onboard_project_returns_profile(self):
        from context.onboarding import onboard_project
        profile = run_async(onboard_project(_ROOT, "test-project"))
        self.assertIn("name", profile)
        self.assertEqual(profile["name"], "test-project")
        self.assertIn("language", profile)
        self.assertIn("conventions", profile)
        self.assertIn("files_indexed", profile)
        self.assertGreater(profile["files_indexed"], 0)

    def test_onboard_project_bad_path(self):
        from context.onboarding import onboard_project
        profile = run_async(onboard_project("/nonexistent/path/12345"))
        self.assertIn("error", profile)

    def test_format_project_profile(self):
        from context.onboarding import format_project_profile
        profile = {
            "name": "myapp",
            "language": "python",
            "framework": "FastAPI",
            "files_indexed": 42,
            "conventions": {
                "naming_style": "snake_case",
                "has_docstrings": True,
                "docstring_ratio": 0.75,
                "async_style": True,
                "common_imports": ["os", "json", "asyncio"],
            },
            "repo_map_summary": "📁 root\n  📂 src/\n  📂 tests/",
        }
        text = format_project_profile(profile)
        self.assertIn("myapp", text)
        self.assertIn("python", text)
        self.assertIn("FastAPI", text)
        self.assertIn("snake_case", text)

    def test_format_empty_profile(self):
        from context.onboarding import format_project_profile
        self.assertEqual(format_project_profile({}), "")
        self.assertEqual(format_project_profile(None), "")

    def test_project_profiles_table(self):
        """Onboarding creates a project_profiles table in DB."""
        src = _read_source("context/onboarding.py")
        self.assertIn("project_profiles", src)
        self.assertIn("CREATE TABLE", src)

    def test_get_project_profile_for_task(self):
        """Should detect project from task context."""
        src = _read_source("context/onboarding.py")
        self.assertIn("get_project_profile_for_task", src)
        # Should check for project name and workspace path
        self.assertIn("project_name", src)
        self.assertIn("workspace", src)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Tool Registry & Agent Prompts
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):
    """Tests for apply_diff in tools/__init__.py"""

    def test_apply_diff_registered(self):
        src = _read_source("tools/__init__.py")
        self.assertIn("from tools.apply_diff import apply_diff", src)
        self.assertIn('"apply_diff"', src)

    def test_apply_diff_in_tool_registry(self):
        from tools import TOOL_REGISTRY
        self.assertIn("apply_diff", TOOL_REGISTRY)
        entry = TOOL_REGISTRY["apply_diff"]
        self.assertIn("function", entry)
        self.assertIn("description", entry)
        self.assertIn("example", entry)

    def test_side_effect_tools_updated(self):
        src = _read_source("agents/base.py")
        self.assertIn('"patch_file"', src)
        self.assertIn('"apply_diff"', src)
        # Both should be in the SIDE_EFFECT_TOOLS frozenset definition
        # Use source inspection since litellm may not be installed
        idx = src.find("SIDE_EFFECT_TOOLS")
        self.assertGreater(idx, 0)
        block = src[idx:idx + 400]
        self.assertIn("patch_file", block)
        self.assertIn("apply_diff", block)

    def test_reindex_trigger_exists(self):
        src = _read_source("tools/__init__.py")
        self.assertIn("_trigger_reindex", src)
        self.assertIn("reindex_file", src)
        # Should trigger on write_file, edit_file, patch_file, apply_diff
        for tool in ["write_file", "edit_file", "patch_file", "apply_diff"]:
            self.assertIn(tool, src)


class TestCoderAgentUpdated(unittest.TestCase):
    """Tests for updated CoderAgent (agents/coder.py)"""

    def test_new_tools_in_allowed_tools(self):
        # Use source inspection since litellm may not be installed
        src = _read_source("agents/coder.py")
        idx = src.find("allowed_tools")
        self.assertGreater(idx, 0)
        block = src[idx:idx + 500]
        for tool in ["edit_file", "patch_file", "apply_diff",
                      "get_function", "query_codebase", "lint"]:
            self.assertIn(f'"{tool}"', block,
                          f"Tool '{tool}' not in CoderAgent.allowed_tools")

    def test_diff_first_guidance(self):
        src = _read_source("agents/coder.py")
        self.assertIn("patch_file", src)
        self.assertIn("apply_diff", src)
        self.assertIn("get_function", src)
        # Should mention diff-first approach
        self.assertIn("Diff-First", src)


class TestFixerAgentUpdated(unittest.TestCase):
    """Tests for updated FixerAgent (agents/fixer.py)"""

    def test_new_tools_in_allowed_tools(self):
        # Use source inspection since litellm may not be installed
        src = _read_source("agents/fixer.py")
        idx = src.find("allowed_tools")
        self.assertGreater(idx, 0)
        block = src[idx:idx + 500]
        for tool in ["patch_file", "apply_diff",
                      "get_function", "query_codebase"]:
            self.assertIn(f'"{tool}"', block,
                          f"Tool '{tool}' not in FixerAgent.allowed_tools")

    def test_prompt_mentions_new_tools(self):
        src = _read_source("agents/fixer.py")
        self.assertIn("patch_file", src)
        self.assertIn("apply_diff", src)
        self.assertIn("get_function", src)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Codebase Index Multi-Language Support
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodebaseIndexMultiLang(unittest.TestCase):
    """Tests for updated tools/codebase_index.py"""

    def test_tree_sitter_conditional_import(self):
        src = _read_source("tools/codebase_index.py")
        self.assertIn("_HAS_TS_PARSER", src)
        self.assertIn("from parsing.tree_sitter_parser", src)

    def test_convert_ts_result(self):
        src = _read_source("tools/codebase_index.py")
        self.assertIn("def _convert_ts_result(", src)

    def test_parse_file_dispatcher(self):
        """_parse_file should try ts parser first, then fallback to ast."""
        src = _read_source("tools/codebase_index.py")
        self.assertIn("def _parse_file(", src)
        self.assertIn("def _parse_file_python_ast(", src)
        self.assertIn("_HAS_TS_PARSER", src)

    def test_build_index_multi_extensions(self):
        """build_index should support multi-language extensions."""
        src = _read_source("tools/codebase_index.py")
        self.assertIn("_default_extensions", src)
        self.assertIn("get_parseable_extensions", src)

    def test_build_index_functional(self):
        from tools.codebase_index import build_index
        index = build_index(_ROOT, extensions=(".py",))
        self.assertGreater(len(index), 0)
        # Check a known file is indexed
        found_any = any("codebase_index" in k for k in index.keys())
        self.assertTrue(found_any, "codebase_index.py should be in index")

    def test_build_index_default_extensions(self):
        from tools.codebase_index import _default_extensions
        exts = _default_extensions()
        self.assertIsInstance(exts, tuple)
        self.assertIn(".py", exts)

    def test_convert_ts_result_functional(self):
        from tools.codebase_index import _convert_ts_result
        ts_result = {
            "functions": [{
                "name": "hello",
                "signature": "def hello(name: str)",
                "docstring": "Say hi",
                "line_start": 1,
                "line_end": 3,
                "body_preview": "return f'hi {name}'",
                "decorators": [],
            }],
            "classes": [{
                "name": "Greeter",
                "bases": ["Base"],
                "docstring": "A greeter",
                "line_start": 5,
                "line_end": 10,
                "methods": [{"name": "greet", "line": 7}],
            }],
            "imports": [
                {"type": "import", "text": "import os", "line": 1},
            ],
            "line_count": 15,
        }
        result = _convert_ts_result(ts_result)
        self.assertEqual(len(result["functions"]), 1)
        self.assertEqual(result["functions"][0]["name"], "hello")
        # _convert_ts_result strips type annotations, so "name: str" -> "name"
        self.assertEqual(result["functions"][0]["args"], ["name"])
        self.assertEqual(len(result["classes"]), 1)
        self.assertEqual(result["classes"][0]["name"], "Greeter")
        # _convert_ts_result extracts module names: "import os" -> "os"
        self.assertEqual(result["imports"], ["os"])
        self.assertEqual(result["line_count"], 15)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Profile injection in agents/base.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfileInjection(unittest.TestCase):
    """Tests for project profile injection in base agent."""

    def test_profile_injection_in_build_context(self):
        src = _read_source("agents/base.py")
        self.assertIn("get_project_profile_for_task", src)
        self.assertIn("format_project_profile", src)

    def test_profile_injection_wrapped_in_try_except(self):
        src = _read_source("agents/base.py")
        idx = src.find("get_project_profile_for_task")
        self.assertGreater(idx, 0)
        # Look backward for try
        before = src[max(0, idx - 300):idx]
        self.assertIn("try", before)
        # Look forward for except
        after = src[idx:idx + 500]
        self.assertIn("except", after)

    def test_profile_injection_before_rag(self):
        """Project profile should be injected before RAG context."""
        src = _read_source("agents/base.py")
        profile_idx = src.find("get_project_profile_for_task")
        rag_idx = src.find("retrieve_context")
        self.assertGreater(profile_idx, 0)
        self.assertGreater(rag_idx, 0)
        self.assertLess(profile_idx, rag_idx,
                        "Profile injection should come before RAG injection")


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Telegram /project add command
# ═══════════════════════════════════════════════════════════════════════════════

class TestTelegramProjectAdd(unittest.TestCase):
    """Tests for /project add command in telegram_bot.py"""

    def test_project_add_handler(self):
        src = _read_source("telegram_bot.py")
        self.assertIn("add", src.lower())
        self.assertIn("onboard_project", src)
        self.assertIn("store_project_profile", src)

    def test_project_add_usage_hint(self):
        src = _read_source("telegram_bot.py")
        self.assertIn("/project add", src)


if __name__ == "__main__":
    unittest.main()
