# tests/test_phase8.py
"""
Tests for Phase 8: Coding Pipeline Maturity

  8.1  Adaptive pipeline stages (complexity classification)
  8.2  Stage selection per complexity
  8.3  AST-aware code tools
  8.4  Codebase indexing
  8.5  Convention detection
  8.6  Codebase map generation
  8.7  PR summary
  8.8  Incremental implementation tracking
  8.9  Tool-callable wrappers
  8.10 Tool registry completeness
  8.11 execute_tool integration
  8.12 Pipeline module structure
  8.13 Phase 8 file existence
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


# ─── 8.1 Complexity Classification ───────────────────────────────────────────

class TestComplexityClassification(unittest.TestCase):
    """Test task complexity classification (via pipeline_utils, no litellm)."""

    def test_oneliner_typo(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Fix typo in README", ""), "oneliner")

    def test_oneliner_rename(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Rename function", "rename foo to bar"), "oneliner")

    def test_oneliner_add_comment(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Add comment to function", ""), "oneliner")

    def test_bugfix_bug(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Fix bug in login", "Users get error 500"), "bugfix")

    def test_bugfix_hotfix(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Hotfix for crash", "app crashes on start"), "bugfix")

    def test_bugfix_fix_error(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Fix error in parser", ""), "bugfix")

    def test_refactor(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Refactor auth module", ""), "refactor")

    def test_refactor_restructure(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Restructure database layer", ""), "refactor")

    def test_feature_default(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Add user profile page", ""), "feature")

    def test_tdd_explicit_keyword(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Add test-driven login", ""), "tdd")

    def test_tdd_in_description(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Build API", "use TDD approach"), "tdd")

    def test_tdd_tests_first(self):
        from pipeline_utils import classify_complexity
        self.assertEqual(classify_complexity("Auth service", "write tests first"), "tdd")


# ─── 8.2 Stage Selection ─────────────────────────────────────────────────────

class TestStageSelection(unittest.TestCase):
    """Test stage selection based on complexity."""

    def test_oneliner_stages(self):
        from pipeline_utils import get_stages_for_complexity
        stages = get_stages_for_complexity("oneliner")
        self.assertNotIn("architect", stages)
        self.assertNotIn("test", stages)
        self.assertIn("implement", stages)
        self.assertIn("review", stages)
        self.assertIn("commit", stages)

    def test_bugfix_stages(self):
        from pipeline_utils import get_stages_for_complexity
        stages = get_stages_for_complexity("bugfix")
        self.assertNotIn("architect", stages)
        self.assertIn("fix", stages)
        self.assertIn("review", stages)
        self.assertIn("commit", stages)

    def test_refactor_no_tests(self):
        from pipeline_utils import get_stages_for_complexity
        stages = get_stages_for_complexity("refactor")
        self.assertNotIn("test", stages)
        self.assertIn("architect", stages)
        self.assertIn("implement", stages)

    def test_feature_full_pipeline(self):
        from pipeline_utils import get_stages_for_complexity
        stages = get_stages_for_complexity("feature")
        self.assertIn("architect", stages)
        self.assertIn("implement", stages)
        self.assertIn("deps", stages)
        self.assertIn("test", stages)
        self.assertIn("review_fix", stages)
        self.assertIn("commit", stages)

    def test_tdd_test_before_implement(self):
        from pipeline_utils import get_stages_for_complexity
        stages = get_stages_for_complexity("tdd")
        self.assertIn("test", stages)
        self.assertIn("implement", stages)
        test_idx = stages.index("test")
        impl_idx = stages.index("implement")
        self.assertLess(test_idx, impl_idx)

    def test_unknown_defaults_to_feature(self):
        from pipeline_utils import get_stages_for_complexity
        stages = get_stages_for_complexity("unknown_xyz")
        feature_stages = get_stages_for_complexity("feature")
        self.assertEqual(stages, feature_stages)

    def test_returns_list(self):
        from pipeline_utils import get_stages_for_complexity
        for complexity in ("oneliner", "bugfix", "refactor", "feature", "tdd"):
            self.assertIsInstance(get_stages_for_complexity(complexity), list)


# ─── 8.3 AST-Aware Code Tools ────────────────────────────────────────────────

class TestASTTools(unittest.TestCase):
    """Test AST-based code analysis tools."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

        self.test_file = os.path.join(self.tmp_dir, "sample.py")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write(
                '"""Module docstring."""\n'
                'import os\n'
                'from typing import Optional\n'
                '\n'
                'def greet(name: str) -> str:\n'
                '    """Say hello."""\n'
                '    return f"Hello {name}"\n'
                '\n'
                'async def async_fetch(url: str) -> str:\n'
                '    """Fetch a URL."""\n'
                '    return "data"\n'
                '\n'
                'class MyClass(object):\n'
                '    """A sample class."""\n'
                '    x = 10\n'
                '\n'
                '    def method_a(self, arg1):\n'
                '        return arg1\n'
                '\n'
                '    async def method_b(self, arg2, arg3):\n'
                '        return arg2 + arg3\n'
            )

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_function_toplevel(self):
        from tools.ast_tools import get_function
        result = run_async(get_function("sample.py", "greet"))
        self.assertIn("✅", result)
        self.assertIn("def greet", result)
        self.assertIn("Hello", result)

    def test_get_function_async(self):
        from tools.ast_tools import get_function
        result = run_async(get_function("sample.py", "async_fetch"))
        self.assertIn("✅", result)
        self.assertIn("async def async_fetch", result)

    def test_get_function_method_dotted(self):
        from tools.ast_tools import get_function
        result = run_async(get_function("sample.py", "MyClass.method_a"))
        self.assertIn("✅", result)
        self.assertIn("def method_a", result)

    def test_get_function_finds_method_by_name(self):
        """get_function should find a method even without the class prefix."""
        from tools.ast_tools import get_function
        result = run_async(get_function("sample.py", "method_a"))
        self.assertIn("✅", result)
        self.assertIn("MyClass.method_a", result)

    def test_get_function_not_found(self):
        from tools.ast_tools import get_function
        result = run_async(get_function("sample.py", "nonexistent_func"))
        self.assertIn("❌", result)

    def test_get_function_file_missing(self):
        from tools.ast_tools import get_function
        result = run_async(get_function("nosuch.py", "greet"))
        self.assertIn("❌", result)

    def test_get_function_path_escape(self):
        from tools.ast_tools import get_function
        result = run_async(get_function("../../etc/passwd", "greet"))
        self.assertIn("❌", result)

    def test_replace_function(self):
        from tools.ast_tools import replace_function
        new_code = 'def greet(name: str) -> str:\n    return f"Hi {name}!"\n'
        result = run_async(replace_function("sample.py", "greet", new_code))
        self.assertIn("✅", result)
        with open(self.test_file, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Hi {name}!", content)
        self.assertNotIn("Hello {name}", content)

    def test_replace_function_not_found(self):
        from tools.ast_tools import replace_function
        result = run_async(replace_function("sample.py", "nosuch", "def nosuch(): pass\n"))
        self.assertIn("❌", result)

    def test_replace_function_invalid_syntax(self):
        from tools.ast_tools import replace_function
        result = run_async(replace_function("sample.py", "greet", "def greet(:\n"))
        self.assertIn("❌", result)
        self.assertIn("syntax", result.lower())

    def test_list_classes(self):
        from tools.ast_tools import list_classes
        result = run_async(list_classes("sample.py"))
        self.assertIn("MyClass", result)
        self.assertIn("method_a", result)
        self.assertIn("method_b", result)

    def test_list_classes_no_classes(self):
        no_cls = os.path.join(self.tmp_dir, "nocls.py")
        with open(no_cls, "w") as f:
            f.write("x = 1\n")
        from tools.ast_tools import list_classes
        result = run_async(list_classes("nocls.py"))
        self.assertIn("No classes", result)

    def test_list_functions(self):
        from tools.ast_tools import list_functions
        result = run_async(list_functions("sample.py"))
        self.assertIn("greet", result)
        self.assertIn("async_fetch", result)
        # Class methods should NOT appear in top-level list
        self.assertNotIn("method_a", result)

    def test_get_imports(self):
        from tools.ast_tools import get_imports
        result = run_async(get_imports("sample.py"))
        self.assertIn("import os", result)
        self.assertIn("from typing import Optional", result)

    def test_get_imports_no_imports(self):
        no_imp = os.path.join(self.tmp_dir, "noimp.py")
        with open(no_imp, "w") as f:
            f.write("x = 1\n")
        from tools.ast_tools import get_imports
        result = run_async(get_imports("noimp.py"))
        self.assertIn("No imports", result)


# ─── 8.4 Codebase Indexing ───────────────────────────────────────────────────

class TestCodebaseIndex(unittest.TestCase):
    """Test codebase indexing and querying."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp_dir, "src"))
        with open(os.path.join(self.tmp_dir, "src", "app.py"), "w", encoding="utf-8") as f:
            f.write(
                '"""Main application."""\n'
                'import os\n'
                'from datetime import datetime\n'
                '\n'
                'def start_app():\n'
                '    """Start the application."""\n'
                '    pass\n'
                '\n'
                'class AppServer:\n'
                '    def handle_request(self, req):\n'
                '        pass\n'
            )
        with open(os.path.join(self.tmp_dir, "src", "utils.py"), "w", encoding="utf-8") as f:
            f.write(
                'import json\n'
                'import os\n'
                '\n'
                'def parse_config(path):\n'
                '    pass\n'
                '\n'
                'def validate_input(data):\n'
                '    pass\n'
            )
        with open(os.path.join(self.tmp_dir, "tests.py"), "w", encoding="utf-8") as f:
            f.write(
                'import unittest\n'
                'class TestApp(unittest.TestCase):\n'
                '    def test_start(self):\n'
                '        pass\n'
            )

    def tearDown(self):
        from tools.codebase_index import clear_index
        clear_index(self.tmp_dir)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_index_finds_all_files(self):
        from tools.codebase_index import build_index
        index = build_index(self.tmp_dir)
        self.assertIn("src/app.py", index)
        self.assertIn("src/utils.py", index)
        self.assertIn("tests.py", index)

    def test_index_functions(self):
        from tools.codebase_index import build_index
        index = build_index(self.tmp_dir)
        app_funcs = [f["name"] for f in index["src/app.py"]["functions"]]
        self.assertIn("start_app", app_funcs)

    def test_index_classes(self):
        from tools.codebase_index import build_index
        index = build_index(self.tmp_dir)
        app_classes = [c["name"] for c in index["src/app.py"]["classes"]]
        self.assertIn("AppServer", app_classes)

    def test_index_imports(self):
        from tools.codebase_index import build_index
        index = build_index(self.tmp_dir)
        self.assertIn("os", index["src/app.py"]["imports"])

    def test_index_docstring(self):
        from tools.codebase_index import build_index
        index = build_index(self.tmp_dir)
        self.assertIn("Main application", index["src/app.py"]["docstring"])

    def test_query_function(self):
        from tools.codebase_index import build_index, query_index
        index = build_index(self.tmp_dir)
        results = query_index(index, "start_app", "function")
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["type"], "function")
        self.assertEqual(results[0]["name"], "start_app")

    def test_query_class(self):
        from tools.codebase_index import build_index, query_index
        index = build_index(self.tmp_dir)
        results = query_index(index, "AppServer", "class")
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["type"], "class")

    def test_query_import(self):
        from tools.codebase_index import build_index, query_index
        index = build_index(self.tmp_dir)
        results = query_index(index, "json", "import")
        self.assertTrue(len(results) >= 1)

    def test_query_file(self):
        from tools.codebase_index import build_index, query_index
        index = build_index(self.tmp_dir)
        results = query_index(index, "utils", "file")
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["type"], "file")

    def test_query_no_results(self):
        from tools.codebase_index import build_index, query_index
        index = build_index(self.tmp_dir)
        results = query_index(index, "zzzznonexistent", "all")
        self.assertEqual(len(results), 0)

    def test_cache(self):
        from tools.codebase_index import build_index, get_cached_index, clear_index
        build_index(self.tmp_dir)
        self.assertIsNotNone(get_cached_index(self.tmp_dir))
        clear_index(self.tmp_dir)
        self.assertIsNone(get_cached_index(self.tmp_dir))

    def test_skips_pycache(self):
        pycache = os.path.join(self.tmp_dir, "__pycache__")
        os.makedirs(pycache, exist_ok=True)
        with open(os.path.join(pycache, "cached.cpython-311.pyc"), "w") as f:
            f.write("x = 1")
        from tools.codebase_index import build_index
        index = build_index(self.tmp_dir)
        for path in index:
            self.assertNotIn("__pycache__", path)

    def test_method_search(self):
        from tools.codebase_index import build_index, query_index
        index = build_index(self.tmp_dir)
        results = query_index(index, "handle_request", "function")
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["type"], "method")

    def test_class_methods_listed(self):
        from tools.codebase_index import build_index
        index = build_index(self.tmp_dir)
        cls_list = index["src/app.py"]["classes"]
        self.assertEqual(len(cls_list), 1)
        method_names = [m["name"] for m in cls_list[0]["methods"]]
        self.assertIn("handle_request", method_names)


# ─── 8.5 Convention Detection ────────────────────────────────────────────────

class TestConventionDetection(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        for i in range(3):
            with open(os.path.join(self.tmp_dir, f"mod{i}.py"), "w", encoding="utf-8") as f:
                f.write(
                    f'"""Module {i}."""\n'
                    f'import os\n'
                    f'import json\n'
                    f'\n'
                    f'async def do_thing_{i}(arg_one, arg_two):\n'
                    f'    """Do thing {i}."""\n'
                    f'    return arg_one + arg_two\n'
                    f'\n'
                    f'async def process_item_{i}(item):\n'
                    f'    pass\n'
                )

    def tearDown(self):
        from tools.codebase_index import clear_index
        clear_index(self.tmp_dir)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_detect_snake_case(self):
        from tools.codebase_index import build_index, detect_conventions
        index = build_index(self.tmp_dir)
        conv = detect_conventions(index)
        self.assertEqual(conv["naming_style"], "snake_case")

    def test_detect_docstrings(self):
        from tools.codebase_index import build_index, detect_conventions
        index = build_index(self.tmp_dir)
        conv = detect_conventions(index)
        self.assertTrue(conv["has_docstrings"])

    def test_detect_async(self):
        from tools.codebase_index import build_index, detect_conventions
        index = build_index(self.tmp_dir)
        conv = detect_conventions(index)
        self.assertTrue(conv["async_style"])

    def test_detect_common_imports(self):
        from tools.codebase_index import build_index, detect_conventions
        index = build_index(self.tmp_dir)
        conv = detect_conventions(index)
        import_names = [imp.split(" (")[0] for imp in conv["common_imports"]]
        self.assertIn("os", import_names)
        self.assertIn("json", import_names)

    def test_detect_total_functions(self):
        from tools.codebase_index import build_index, detect_conventions
        index = build_index(self.tmp_dir)
        conv = detect_conventions(index)
        self.assertEqual(conv["total_functions"], 6)  # 2 per file × 3 files
        self.assertEqual(conv["total_files"], 3)

    def test_empty_index(self):
        from tools.codebase_index import detect_conventions
        conv = detect_conventions({})
        self.assertIn("error", conv)


# ─── 8.6 Codebase Map ────────────────────────────────────────────────────────

class TestCodebaseMap(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp_dir, "src"))
        with open(os.path.join(self.tmp_dir, "src", "main.py"), "w", encoding="utf-8") as f:
            f.write(
                '"""Main entry point."""\n'
                'def main():\n'
                '    pass\n'
                'class App:\n'
                '    def run(self):\n'
                '        pass\n'
            )
        with open(os.path.join(self.tmp_dir, "setup.py"), "w", encoding="utf-8") as f:
            f.write('from setuptools import setup\n')

    def tearDown(self):
        from tools.codebase_index import clear_index
        clear_index(self.tmp_dir)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_map_contains_header(self):
        from tools.codebase_index import build_index, get_codebase_map
        index = build_index(self.tmp_dir)
        cmap = get_codebase_map(index)
        self.assertIn("Codebase Map", cmap)

    def test_map_lists_files(self):
        from tools.codebase_index import build_index, get_codebase_map
        index = build_index(self.tmp_dir)
        cmap = get_codebase_map(index)
        self.assertIn("main.py", cmap)

    def test_map_shows_classes(self):
        from tools.codebase_index import build_index, get_codebase_map
        index = build_index(self.tmp_dir)
        cmap = get_codebase_map(index)
        self.assertIn("App", cmap)

    def test_map_shows_functions(self):
        from tools.codebase_index import build_index, get_codebase_map
        index = build_index(self.tmp_dir)
        cmap = get_codebase_map(index)
        self.assertIn("main", cmap)

    def test_empty_map(self):
        from tools.codebase_index import get_codebase_map
        result = get_codebase_map({})
        self.assertIn("Empty index", result)


# ─── 8.7 PR Summary ─────────────────────────────────────────────────────────

class TestPRSummary(unittest.TestCase):

    def test_summary_contains_title(self):
        from pipeline_utils import generate_pr_summary
        result = run_async(generate_pr_summary(
            title="Add login feature",
            files_changed=["auth.py", "login.py"],
            stages_run=["architect", "implement", "test", "review", "commit"],
            review_iterations=2,
            complexity="feature",
        ))
        self.assertIn("Add login feature", result)

    def test_summary_contains_files(self):
        from pipeline_utils import generate_pr_summary
        result = run_async(generate_pr_summary(
            title="Test",
            files_changed=["auth.py", "login.py"],
            stages_run=["architect"],
            review_iterations=0,
            complexity="feature",
        ))
        self.assertIn("auth.py", result)

    def test_summary_contains_complexity(self):
        from pipeline_utils import generate_pr_summary
        result = run_async(generate_pr_summary(
            title="Test",
            files_changed=[],
            stages_run=["fix", "commit"],
            review_iterations=1,
            complexity="bugfix",
        ))
        self.assertIn("bugfix", result)

    def test_summary_no_files(self):
        from pipeline_utils import generate_pr_summary
        result = run_async(generate_pr_summary(
            title="Quick fix",
            files_changed=[],
            stages_run=["fix", "commit"],
            review_iterations=0,
            complexity="bugfix",
        ))
        self.assertIn("no specific files", result)

    def test_summary_review_iterations(self):
        from pipeline_utils import generate_pr_summary
        result = run_async(generate_pr_summary(
            title="T",
            files_changed=[],
            stages_run=["review"],
            review_iterations=3,
            complexity="feature",
        ))
        self.assertIn("3", result)


# ─── 8.8 Incremental Progress ────────────────────────────────────────────────

class TestIncrementalProgress(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load(self):
        from pipeline_utils import _save_progress, _load_progress
        _save_progress(42, {"completed_files": ["a.py", "b.py"], "stage": "implement"})
        progress = _load_progress(42)
        self.assertEqual(progress["completed_files"], ["a.py", "b.py"])
        self.assertEqual(progress["stage"], "implement")

    def test_load_missing_returns_empty(self):
        from pipeline_utils import _load_progress
        self.assertEqual(_load_progress(999), {})

    def test_cleanup_removes_file(self):
        from pipeline_utils import _save_progress, _cleanup_progress, _load_progress
        _save_progress(42, {"completed_files": ["a.py"]})
        _cleanup_progress(42)
        self.assertEqual(_load_progress(42), {})

    def test_none_goal_id_is_noop(self):
        from pipeline_utils import _save_progress, _load_progress
        _save_progress(None, {"x": 1})
        self.assertEqual(_load_progress(None), {})

    def test_overwrite_progress(self):
        from pipeline_utils import _save_progress, _load_progress
        _save_progress(10, {"completed_files": ["a.py"]})
        _save_progress(10, {"completed_files": ["a.py", "b.py"]})
        progress = _load_progress(10)
        self.assertEqual(len(progress["completed_files"]), 2)


# ─── 8.9 Tool-Callable Wrappers ─────────────────────────────────────────────

class TestToolWrappers(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

        with open(os.path.join(self.tmp_dir, "example.py"), "w", encoding="utf-8") as f:
            f.write("def hello():\n    pass\n")

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        from tools.codebase_index import clear_index
        clear_index(self.tmp_dir)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_index_workspace(self):
        from tools.codebase_index import index_workspace
        result = run_async(index_workspace("."))
        self.assertIn("✅", result)
        self.assertIn("1 file", result)

    def test_query_codebase_found(self):
        from tools.codebase_index import query_codebase
        result = run_async(query_codebase("hello"))
        self.assertIn("hello", result)

    def test_query_codebase_not_found(self):
        from tools.codebase_index import query_codebase
        result = run_async(query_codebase("zzznonexistent"))
        self.assertIn("❌", result)

    def test_codebase_map_returns_string(self):
        from tools.codebase_index import codebase_map
        result = run_async(codebase_map("."))
        self.assertIsInstance(result, str)


# ─── 8.10 Tool Registry ─────────────────────────────────────────────────────

class TestPhase8ToolRegistry(unittest.TestCase):

    def test_ast_tools_in_registry(self):
        from tools import TOOL_REGISTRY
        for name in ("get_function", "replace_function", "list_classes",
                     "list_functions", "get_imports"):
            self.assertIn(name, TOOL_REGISTRY, f"Missing: {name}")

    def test_index_tools_in_registry(self):
        from tools import TOOL_REGISTRY
        for name in ("index_workspace", "query_codebase", "codebase_map"):
            self.assertIn(name, TOOL_REGISTRY, f"Missing: {name}")

    def test_ast_tools_in_schemas(self):
        from tools import TOOL_SCHEMAS
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        for name in ("get_function", "replace_function", "list_classes",
                     "list_functions", "get_imports"):
            self.assertIn(name, schema_names, f"Missing from schemas: {name}")

    def test_index_tools_in_schemas(self):
        from tools import TOOL_SCHEMAS
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        for name in ("index_workspace", "query_codebase", "codebase_map"):
            self.assertIn(name, schema_names, f"Missing from schemas: {name}")

    def test_get_function_schema_has_params(self):
        from tools import TOOL_SCHEMAS
        for s in TOOL_SCHEMAS:
            if s["function"]["name"] == "get_function":
                props = s["function"]["parameters"]["properties"]
                self.assertIn("filepath", props)
                self.assertIn("function_name", props)
                return
        self.fail("get_function schema not found")

    def test_replace_function_schema_has_params(self):
        from tools import TOOL_SCHEMAS
        for s in TOOL_SCHEMAS:
            if s["function"]["name"] == "replace_function":
                props = s["function"]["parameters"]["properties"]
                self.assertIn("filepath", props)
                self.assertIn("function_name", props)
                self.assertIn("new_code", props)
                return
        self.fail("replace_function schema not found")


# ─── 8.11 execute_tool Integration ──────────────────────────────────────────

class TestExecutePhase8Tools(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

        with open(os.path.join(self.tmp_dir, "target.py"), "w", encoding="utf-8") as f:
            f.write(
                "def compute(x, y):\n"
                "    return x + y\n"
            )

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_execute_get_function(self):
        from tools import execute_tool
        result = run_async(execute_tool(
            "get_function", filepath="target.py", function_name="compute"
        ))
        self.assertIn("✅", result)
        self.assertIn("def compute", result)

    def test_execute_list_functions(self):
        from tools import execute_tool
        result = run_async(execute_tool("list_functions", filepath="target.py"))
        self.assertIn("compute", result)

    def test_execute_replace_function(self):
        from tools import execute_tool
        result = run_async(execute_tool(
            "replace_function",
            filepath="target.py",
            function_name="compute",
            new_code="def compute(x, y):\n    return x * y\n",
        ))
        self.assertIn("✅", result)

    def test_execute_list_classes(self):
        from tools import execute_tool
        result = run_async(execute_tool("list_classes", filepath="target.py"))
        self.assertIn("No classes", result)

    def test_execute_get_imports(self):
        from tools import execute_tool
        result = run_async(execute_tool("get_imports", filepath="target.py"))
        self.assertIn("No imports", result)

    def test_execute_index_workspace(self):
        from tools import execute_tool
        result = run_async(execute_tool("index_workspace"))
        self.assertIn("✅", result)


# ─── 8.12 Pipeline Module Structure ─────────────────────────────────────────

class TestPipelineStructure(unittest.TestCase):
    """Verify pipeline_utils has all Phase 8 pure functions (no litellm needed)."""

    def test_classify_complexity_callable(self):
        from pipeline_utils import classify_complexity
        self.assertTrue(callable(classify_complexity))

    def test_get_stages_callable(self):
        from pipeline_utils import get_stages_for_complexity
        self.assertTrue(callable(get_stages_for_complexity))

    def test_generate_pr_summary_callable(self):
        from pipeline_utils import generate_pr_summary
        self.assertTrue(callable(generate_pr_summary))

    def test_progress_functions_callable(self):
        from pipeline_utils import _save_progress, _load_progress, _cleanup_progress
        for fn in (_save_progress, _load_progress, _cleanup_progress):
            self.assertTrue(callable(fn))

    def test_convention_context_callable(self):
        from pipeline_utils import _get_convention_context
        self.assertTrue(callable(_get_convention_context))

    def test_codebase_map_context_callable(self):
        from pipeline_utils import _get_codebase_map_context
        self.assertTrue(callable(_get_codebase_map_context))

    def test_pipeline_utils_importable_without_litellm(self):
        """pipeline_utils must not transitively import litellm."""
        import importlib
        import sys
        # Temporarily block litellm
        orig = sys.modules.get("litellm", None)
        sys.modules["litellm"] = None  # type: ignore
        try:
            if "pipeline_utils" in sys.modules:
                del sys.modules["pipeline_utils"]
            import pipeline_utils  # noqa: F401 — just confirm it loads
        except ImportError as e:
            if "litellm" in str(e):
                self.fail(f"pipeline_utils transitively imports litellm: {e}")
        finally:
            if orig is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = orig


# ─── 8.13 File Existence ────────────────────────────────────────────────────

class TestPhase8Files(unittest.TestCase):

    def _src_path(self, *parts):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, *parts)

    def test_ast_tools_exists(self):
        self.assertTrue(os.path.isfile(self._src_path("tools", "ast_tools.py")))

    def test_codebase_index_exists(self):
        self.assertTrue(os.path.isfile(self._src_path("tools", "codebase_index.py")))

    def test_pipeline_utils_exists(self):
        self.assertTrue(os.path.isfile(self._src_path("pipeline_utils.py")))

    def test_pipeline_exists(self):
        self.assertTrue(os.path.isfile(self._src_path("pipeline.py")))

    def test_pipeline_source_uses_pipeline_utils(self):
        """pipeline.py should import from pipeline_utils."""
        src = self._src_path("pipeline.py")
        with open(src, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("from pipeline_utils import", content)


if __name__ == "__main__":
    unittest.main()
