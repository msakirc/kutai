"""Tests for pipeline artifact extraction (Fix #4)."""

import asyncio
import json
import unittest

from src.workflows.engine.pipeline_artifacts import (
    extract_pipeline_artifacts,
    _extract_files_changed,
    _extract_test_results,
    _build_implementation_summary,
    _parse_context,
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestExtractFilesFromResultText(unittest.TestCase):

    def test_extract_created_file(self):
        result = {"result": "Created src/foo.py and ran tests"}
        files = run_async(_extract_files_changed(result, None))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["path"], "src/foo.py")
        self.assertEqual(files[0]["action"], "modified")

    def test_extract_multiple_actions(self):
        result = {
            "result": (
                "Created src/main.py then Modified utils/helper.js "
                "and Updated tests/test_app.py"
            )
        }
        files = run_async(_extract_files_changed(result, None))
        self.assertEqual(len(files), 3)
        paths = [f["path"] for f in files]
        self.assertIn("src/main.py", paths)
        self.assertIn("utils/helper.js", paths)
        self.assertIn("tests/test_app.py", paths)

    def test_extract_backtick_quoted(self):
        result = {"result": "Added `src/models/user.py` to project"}
        files = run_async(_extract_files_changed(result, None))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["path"], "src/models/user.py")

    def test_no_files_in_text(self):
        result = {"result": "Everything looks good, no changes needed."}
        files = run_async(_extract_files_changed(result, None))
        self.assertEqual(files, [])


class TestExtractTestResults(unittest.TestCase):

    def test_passed_and_failed(self):
        result = {"result": "Ran suite: 10 passed, 2 failed"}
        tr = _extract_test_results(result)
        self.assertIsNotNone(tr)
        self.assertEqual(tr["passed"], 10)
        self.assertEqual(tr["failed"], 2)
        self.assertEqual(tr["total"], 12)

    def test_only_passed(self):
        result = {"result": "All 15 tests passed successfully"}
        tr = _extract_test_results(result)
        self.assertIsNotNone(tr)
        self.assertEqual(tr["passed"], 15)
        self.assertEqual(tr["failed"], 0)
        self.assertEqual(tr["total"], 15)

    def test_no_match(self):
        result = {"result": "Deployed to staging environment"}
        tr = _extract_test_results(result)
        self.assertIsNone(tr)

    def test_empty_result(self):
        result = {"result": ""}
        tr = _extract_test_results(result)
        self.assertIsNone(tr)


class TestBuildImplementationSummary(unittest.TestCase):

    def test_includes_title_and_files(self):
        task = {"title": "Implement auth module", "context": "{}"}
        result = {"result": "Done implementing auth"}
        files = [
            {"path": "src/auth.py", "action": "added"},
            {"path": "tests/test_auth.py", "action": "added"},
        ]
        summary = _build_implementation_summary(task, result, files)
        self.assertIn("## Implementation: Implement auth module", summary)
        self.assertIn("**Files touched:** 2", summary)
        self.assertIn("src/auth.py", summary)
        self.assertIn("**Result excerpt:**", summary)

    def test_includes_feature_name(self):
        ctx = {"workflow_context": {"feature_name": "User Authentication"}}
        task = {"title": "Build login", "context": json.dumps(ctx)}
        result = {"result": "Login built"}
        summary = _build_implementation_summary(task, result, [])
        self.assertIn("**Feature:** User Authentication", summary)

    def test_truncates_long_result(self):
        task = {"title": "Big task", "context": "{}"}
        result = {"result": "x" * 500}
        summary = _build_implementation_summary(task, result, [])
        self.assertIn("...", summary)


class TestExtractPipelineArtifactsFull(unittest.TestCase):

    def test_full_integration(self):
        ctx = {
            "is_workflow_step": True,
            "step_id": "8.1.code",
            "workflow_context": {"feature_name": "Payments"},
        }
        task = {
            "title": "Implement payment processing",
            "context": json.dumps(ctx),
        }
        result = {
            "result": "Created src/payments.py and Added tests/test_payments.py. 5 tests passed, 0 failed."
        }

        artifacts = run_async(extract_pipeline_artifacts(task, result, None))

        self.assertIn("8.1.code_implementation_summary", artifacts)
        self.assertIn("8.1.code_files_changed", artifacts)
        self.assertIn("8.1.code_test_results", artifacts)

        files = json.loads(artifacts["8.1.code_files_changed"])
        self.assertEqual(len(files), 2)

        tests = json.loads(artifacts["8.1.code_test_results"])
        self.assertEqual(tests["passed"], 5)
        self.assertEqual(tests["failed"], 0)

    def test_fallback_step_id(self):
        task = {"title": "Task", "context": "{}"}
        result = {"result": "done"}
        artifacts = run_async(extract_pipeline_artifacts(task, result, None))
        self.assertIn("unknown_implementation_summary", artifacts)


class TestDeduplicateFiles(unittest.TestCase):

    def test_same_file_mentioned_twice(self):
        result = {
            "result": "Created src/app.py then Modified src/app.py to add routes"
        }
        files = run_async(_extract_files_changed(result, None))
        paths = [f["path"] for f in files]
        self.assertEqual(paths.count("src/app.py"), 1)


class TestNoFilesNoTests(unittest.TestCase):

    def test_minimal_result_still_produces_summary(self):
        task = {"title": "Review step", "context": "{}"}
        result = {"result": "Reviewed the code, looks fine."}
        artifacts = run_async(extract_pipeline_artifacts(task, result, None))
        # Should still have implementation summary
        self.assertIn("unknown_implementation_summary", artifacts)
        # No files or tests
        self.assertNotIn("unknown_files_changed", artifacts)
        self.assertNotIn("unknown_test_results", artifacts)


class TestParseContext(unittest.TestCase):

    def test_dict_context(self):
        self.assertEqual(_parse_context({"context": {"a": 1}}), {"a": 1})

    def test_json_string_context(self):
        self.assertEqual(_parse_context({"context": '{"b": 2}'}), {"b": 2})

    def test_invalid_json(self):
        self.assertEqual(_parse_context({"context": "not json"}), {})

    def test_missing_context(self):
        self.assertEqual(_parse_context({}), {})


if __name__ == "__main__":
    unittest.main()
