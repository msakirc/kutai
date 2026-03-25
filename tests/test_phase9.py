# tests/test_phase9.py
"""
Tests for Phase 9: Reliability Foundation

  9.1  Task State Machine (states, transitions, validation)
  9.2  Structured Output Enforcement (output validators)
  9.3  Error Taxonomy & Retry Policies
  9.4  Atomic DB Operations (write_file atomic, WAL checkpoint)
  9.5  Per-Task Token & Cost Budgets
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


# ─── 9.1 Task State Machine ──────────────────────────────────────────────────

class TestTaskStates(unittest.TestCase):
    """Test TaskState enum and completeness."""

    def test_all_states_defined(self):
        from state_machine import TaskState
        expected = {
            "pending", "processing", "completed", "failed", "cancelled",
            "needs_clarification", "needs_review", "needs_subtasks",
            "waiting_subtasks", "paused", "rejected",
        }
        actual = {s.value for s in TaskState}
        self.assertEqual(expected, actual)

    def test_state_enum_is_string(self):
        from state_machine import TaskState
        self.assertEqual(str(TaskState.PENDING), "TaskState.PENDING")
        self.assertEqual(TaskState.PENDING.value, "pending")

    def test_error_categories_defined(self):
        from state_machine import ErrorCategory
        expected = {
            "model_error", "tool_error", "timeout",
            "budget_exceeded", "invalid_output",
            "dependency_failed", "cancelled", "unknown",
        }
        actual = {e.value for e in ErrorCategory}
        self.assertEqual(expected, actual)


class TestTransitions(unittest.TestCase):
    """Test state transition validation."""

    def test_pending_to_processing(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("pending", "processing"))

    def test_pending_to_cancelled(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("pending", "cancelled"))

    def test_processing_to_completed(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("processing", "completed"))

    def test_processing_to_failed(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("processing", "failed"))

    def test_processing_to_needs_clarification(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("processing", "needs_clarification"))

    def test_processing_to_waiting_subtasks(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("processing", "waiting_subtasks"))

    def test_failed_to_pending_retry(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("failed", "pending"))

    def test_completed_is_terminal(self):
        from state_machine import validate_transition, TaskState
        for state in TaskState:
            if state.value != "completed":
                self.assertFalse(
                    validate_transition("completed", state.value),
                    f"completed -> {state.value} should be invalid",
                )

    def test_cancelled_is_terminal(self):
        from state_machine import validate_transition, TaskState
        for state in TaskState:
            if state.value != "cancelled":
                self.assertFalse(
                    validate_transition("cancelled", state.value),
                    f"cancelled -> {state.value} should be invalid",
                )

    def test_invalid_transition_pending_to_completed(self):
        from state_machine import validate_transition
        self.assertFalse(validate_transition("pending", "completed"))

    def test_invalid_transition_pending_to_failed(self):
        from state_machine import validate_transition
        self.assertFalse(validate_transition("pending", "failed"))

    def test_paused_to_pending(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("paused", "pending"))

    def test_pending_to_paused(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("pending", "paused"))

    def test_waiting_subtasks_to_completed(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("waiting_subtasks", "completed"))

    def test_needs_clarification_to_pending(self):
        from state_machine import validate_transition
        self.assertTrue(validate_transition("needs_clarification", "pending"))

    def test_transitions_dict_covers_all_states(self):
        from state_machine import TRANSITIONS, TaskState
        for state in TaskState:
            self.assertIn(
                state.value, TRANSITIONS,
                f"State '{state.value}' missing from TRANSITIONS dict",
            )


class TestClassifyError(unittest.TestCase):
    """Test error classification from exceptions."""

    def test_timeout_error(self):
        from state_machine import classify_error, ErrorCategory
        import asyncio
        result = classify_error(asyncio.TimeoutError())
        self.assertEqual(result, ErrorCategory.TIMEOUT)

    def test_timeout_in_message(self):
        from state_machine import classify_error, ErrorCategory
        result = classify_error(Exception("request timed out: timeout reached"))
        self.assertEqual(result, ErrorCategory.TIMEOUT)

    def test_model_error_rate_limit(self):
        from state_machine import classify_error, ErrorCategory
        result = classify_error(Exception("rate_limit_exceeded: 429 Too Many Requests"))
        self.assertEqual(result, ErrorCategory.MODEL_ERROR)

    def test_model_error_api(self):
        from state_machine import classify_error, ErrorCategory
        result = classify_error(Exception("litellm.APIError: connection failed"))
        self.assertEqual(result, ErrorCategory.MODEL_ERROR)

    def test_tool_error(self):
        from state_machine import classify_error, ErrorCategory
        result = classify_error(Exception("Tool error: command failed with exit code 1"))
        self.assertEqual(result, ErrorCategory.TOOL_ERROR)

    def test_budget_error(self):
        from state_machine import classify_error, ErrorCategory
        result = classify_error(Exception("Budget exceeded for daily scope"))
        self.assertEqual(result, ErrorCategory.BUDGET_EXCEEDED)

    def test_invalid_output(self):
        from state_machine import classify_error, ErrorCategory
        result = classify_error(Exception("JSON parse error on response"))
        self.assertEqual(result, ErrorCategory.INVALID_OUTPUT)

    def test_unknown_error(self):
        from state_machine import classify_error, ErrorCategory
        result = classify_error(Exception("something completely random happened"))
        self.assertEqual(result, ErrorCategory.UNKNOWN)

    def test_cancellation(self):
        from state_machine import classify_error, ErrorCategory
        import asyncio
        result = classify_error(asyncio.CancelledError())
        self.assertEqual(result, ErrorCategory.CANCELLED)


class TestInvalidTransitionException(unittest.TestCase):
    """Test the InvalidTransition exception."""

    def test_message_format(self):
        from state_machine import InvalidTransition
        exc = InvalidTransition(42, "pending", "completed")
        self.assertIn("42", str(exc))
        self.assertIn("pending", str(exc))
        self.assertIn("completed", str(exc))

    def test_attributes(self):
        from state_machine import InvalidTransition
        exc = InvalidTransition(99, "failed", "processing")
        self.assertEqual(exc.task_id, 99)
        self.assertEqual(exc.from_state, "failed")
        self.assertEqual(exc.to_state, "processing")


# ─── 9.2 Structured Output Enforcement ───────────────────────────────────────

class TestOutputValidators(unittest.TestCase):
    """Test per-task-type output validators (Pydantic action models)."""

    def test_models_py_has_json_schema(self):
        from models import get_action_json_schema
        schema = get_action_json_schema()
        self.assertIn("json_schema", schema)
        self.assertIn("action", schema["json_schema"]["schema"]["properties"])

    def test_validate_action_tool_call(self):
        from models import validate_action
        result = validate_action({
            "action": "tool_call",
            "tool": "read_file",
            "args": {"filepath": "test.py"},
        })
        self.assertEqual(result["action"], "tool_call")
        self.assertEqual(result["tool"], "read_file")

    def test_validate_action_final_answer(self):
        from models import validate_action
        result = validate_action({
            "action": "final_answer",
            "result": "The answer is 42",
        })
        self.assertEqual(result["action"], "final_answer")
        self.assertEqual(result["result"], "The answer is 42")

    def test_validate_action_invalid_raises(self):
        from models import validate_action
        # tool_call without tool name should raise
        with self.assertRaises(ValueError):
            validate_action({"action": "tool_call"})


class TestTaskTypeValidators(unittest.TestCase):
    """Test per-task-type output validators (Phase 9.2)."""

    def test_code_task_valid_with_filepath(self):
        from models import validate_task_output
        errors = validate_task_output("coder", "Created src/main.py with logic")
        self.assertEqual(errors, [])

    def test_code_task_valid_with_code_block(self):
        from models import validate_task_output
        errors = validate_task_output("coder", "```python\nprint('hello')\n```")
        self.assertEqual(errors, [])

    def test_code_task_valid_with_keywords(self):
        from models import validate_task_output
        errors = validate_task_output("coder", "Added def process_data that uses import json")
        self.assertEqual(errors, [])

    def test_code_task_invalid_no_code(self):
        from models import validate_task_output
        errors = validate_task_output("coder", "Everything looks good!")
        self.assertTrue(len(errors) > 0)
        self.assertIn("code", errors[0].lower())

    def test_implementer_counts_as_code(self):
        from models import validate_task_output
        errors = validate_task_output("implementer", "Created utils/helper.py")
        self.assertEqual(errors, [])

    def test_research_task_valid_with_url(self):
        from models import validate_task_output
        errors = validate_task_output("researcher", "Found info at https://example.com/docs")
        self.assertEqual(errors, [])

    def test_research_task_valid_with_source_ref(self):
        from models import validate_task_output
        errors = validate_task_output("researcher", "According to the documentation, X is true")
        self.assertEqual(errors, [])

    def test_research_task_invalid_no_source(self):
        from models import validate_task_output
        errors = validate_task_output("researcher", "The answer is 42.")
        self.assertTrue(len(errors) > 0)
        self.assertIn("URL", errors[0])

    def test_planner_task_valid_with_steps(self):
        from models import validate_task_output
        errors = validate_task_output("planner", "step 1. Do X\nstep 2. Do Y")
        self.assertEqual(errors, [])

    def test_planner_task_valid_with_list(self):
        from models import validate_task_output
        errors = validate_task_output("planner", "Plan:\n- First thing\n- Second thing")
        self.assertEqual(errors, [])

    def test_planner_task_valid_with_subtasks(self):
        from models import validate_task_output
        errors = validate_task_output("planner", "Created 3 subtasks for this mission")
        self.assertEqual(errors, [])

    def test_planner_task_invalid_no_structure(self):
        from models import validate_task_output
        errors = validate_task_output("planner", "Done thinking about it")
        self.assertTrue(len(errors) > 0)
        self.assertIn("subtask", errors[0].lower())

    def test_unknown_agent_type_no_validation(self):
        from models import validate_task_output
        errors = validate_task_output("custom_agent", "Literally anything")
        self.assertEqual(errors, [])

    def test_fixer_counts_as_code(self):
        from models import validate_task_output
        errors = validate_task_output("fixer", "Fixed bug in src/app.py")
        self.assertEqual(errors, [])

    def test_architect_counts_as_planner(self):
        from models import validate_task_output
        errors = validate_task_output("architect", "Architecture:\n- Module A\n- Module B")
        self.assertEqual(errors, [])


class TestParseAgentResponseRefactored(unittest.TestCase):
    """Test the refactored _parse_agent_response (Phase 9.2).

    Cannot import base.py directly (litellm dependency), so we
    extract and test the parsing logic inline.
    """

    def _make_parser(self):
        """Build a lightweight parser matching the refactored base.py logic."""
        import re

        def _try_parse_json(text):
            try:
                stripped = text
                if stripped.startswith("```"):
                    stripped = (
                        stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
                    )
                    stripped = stripped.rsplit("```", 1)[0]
                obj = json.loads(stripped.strip())
                return obj if isinstance(obj, dict) else None
            except (json.JSONDecodeError, IndexError):
                return None

        def _normalize_action(parsed):
            action = parsed.get("action")
            if not action:
                if "tool" in parsed:
                    parsed["action"] = "tool_call"
                elif "result" in parsed:
                    parsed["action"] = "final_answer"
                else:
                    return None
            return parsed

        def parse(content):
            cleaned = content.strip()
            # Try 1 — direct parse
            p = _try_parse_json(cleaned)
            if p is not None:
                n = _normalize_action(p)
                if n is not None:
                    return n
            # Try 2 — fence extraction
            json_blocks = re.findall(
                r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL
            )
            for block in json_blocks:
                p = _try_parse_json(block.strip())
                if p is not None:
                    n = _normalize_action(p)
                    if n is not None:
                        return n
            # Try 3 — brace-depth scan
            if "{" in cleaned:
                start = cleaned.index("{")
                depth = 0
                for i in range(start, len(cleaned)):
                    if cleaned[i] == "{":
                        depth += 1
                    elif cleaned[i] == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                p = json.loads(cleaned[start:i+1])
                                if isinstance(p, dict):
                                    n = _normalize_action(p)
                                    if n is not None:
                                        return n
                            except json.JSONDecodeError:
                                pass
                            break
            # Explicit failure
            return None

        return parse

    def test_clean_json_tool_call(self):
        parse = self._make_parser()
        result = parse('{"action": "tool_call", "tool": "shell", "args": {"command": "ls"}}')
        self.assertEqual(result["action"], "tool_call")
        self.assertEqual(result["tool"], "shell")

    def test_clean_json_final_answer(self):
        parse = self._make_parser()
        result = parse('{"action": "final_answer", "result": "Hello!"}')
        self.assertEqual(result["action"], "final_answer")
        self.assertEqual(result["result"], "Hello!")

    def test_fenced_json(self):
        parse = self._make_parser()
        result = parse('Here\'s the answer:\n```json\n{"action": "final_answer", "result": "42"}\n```')
        self.assertEqual(result["action"], "final_answer")
        self.assertEqual(result["result"], "42")

    def test_json_buried_in_prose(self):
        parse = self._make_parser()
        result = parse('I think we should use {"action": "tool_call", "tool": "read_file", "args": {"filepath": "x.py"}} for this.')
        self.assertEqual(result["action"], "tool_call")

    def test_plain_text_returns_none(self):
        """Phase 9.2: plain text must return None, NOT a fallback final_answer."""
        parse = self._make_parser()
        result = parse("I don't know what to do next. Let me think about it.")
        self.assertIsNone(result)

    def test_garbled_json_returns_none(self):
        """Broken JSON should fail explicitly, not silently become an answer."""
        parse = self._make_parser()
        result = parse('{"action": "tool_call", "tool": broken}')
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        parse = self._make_parser()
        result = parse("")
        self.assertIsNone(result)

    def test_inferred_tool_call(self):
        parse = self._make_parser()
        result = parse('{"tool": "shell", "args": {"command": "pwd"}}')
        self.assertEqual(result["action"], "tool_call")

    def test_inferred_final_answer(self):
        parse = self._make_parser()
        result = parse('{"result": "The answer is 42"}')
        self.assertEqual(result["action"], "final_answer")


class TestBasePyRefactored(unittest.TestCase):
    """Verify base.py source reflects Phase 9.2 refactoring."""

    def _read_base_py(self):
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(base_path, encoding="utf-8") as f:
            return f.read()

    def test_no_regex_fallback(self):
        """Verify the old Try 4 (regex for nested one-liner) was removed."""
        content = self._read_base_py()
        self.assertNotIn("regex for possibly-nested one-liner", content)

    def test_no_silent_final_answer_fallback(self):
        """Verify there's no silent 'Fallback — entire response is the answer'."""
        content = self._read_base_py()
        self.assertNotIn("Fallback — entire response is the answer", content)
        self.assertNotIn('return {"action": "final_answer", "result": content}',
                         content.replace("            ", ""))

    def test_returns_none_on_failure(self):
        """Verify _parse_agent_response returns None on failure."""
        content = self._read_base_py()
        self.assertIn("return None", content)
        self.assertIn("Explicit failure", content)

    def test_imports_validate_task_output(self):
        content = self._read_base_py()
        self.assertIn("validate_task_output", content)

    def test_has_task_type_validation_section(self):
        content = self._read_base_py()
        self.assertIn("Per-task-type output validation", content)
        self.assertIn("task_type_errors", content)


# ─── 9.3 Error Taxonomy & Retry Policies ─────────────────────────────────────

class TestRetryPolicies(unittest.TestCase):
    """Test error-specific retry policies."""

    def test_model_error_retries_with_fallback(self):
        from error_policy import get_retry_policy, ErrorCategory
        policy = get_retry_policy(ErrorCategory.MODEL_ERROR)
        self.assertTrue(policy.retry_with_fallback_model)
        self.assertGreater(policy.max_retries, 0)

    def test_tool_error_retries_same_model(self):
        from error_policy import get_retry_policy, ErrorCategory
        policy = get_retry_policy(ErrorCategory.TOOL_ERROR)
        self.assertFalse(policy.retry_with_fallback_model)
        self.assertGreater(policy.max_retries, 0)

    def test_timeout_increases_timeout(self):
        from error_policy import get_retry_policy, ErrorCategory
        policy = get_retry_policy(ErrorCategory.TIMEOUT)
        self.assertTrue(policy.increase_timeout)
        self.assertGreater(policy.timeout_multiplier, 1.0)

    def test_invalid_output_injects_correction(self):
        from error_policy import get_retry_policy, ErrorCategory
        policy = get_retry_policy(ErrorCategory.INVALID_OUTPUT)
        self.assertTrue(policy.inject_correction_prompt)

    def test_budget_exceeded_pauses(self):
        from error_policy import get_retry_policy, ErrorCategory
        policy = get_retry_policy(ErrorCategory.BUDGET_EXCEEDED)
        self.assertTrue(policy.pause_and_notify)
        self.assertEqual(policy.max_retries, 0)

    def test_dependency_failed_fails_immediately(self):
        from error_policy import get_retry_policy, ErrorCategory
        policy = get_retry_policy(ErrorCategory.DEPENDENCY_FAILED)
        self.assertTrue(policy.fail_immediately)

    def test_cancelled_no_retry(self):
        from error_policy import get_retry_policy, ErrorCategory
        policy = get_retry_policy(ErrorCategory.CANCELLED)
        self.assertTrue(policy.fail_immediately)
        self.assertEqual(policy.max_retries, 0)


class TestShouldRetry(unittest.TestCase):
    """Test should_retry decision logic."""

    def test_model_error_first_retry(self):
        from error_policy import should_retry
        self.assertTrue(should_retry("model_error", 0))

    def test_model_error_max_retries(self):
        from error_policy import should_retry
        self.assertFalse(should_retry("model_error", 3))

    def test_budget_exceeded_never_retry(self):
        from error_policy import should_retry
        self.assertFalse(should_retry("budget_exceeded", 0))

    def test_dependency_failed_never_retry(self):
        from error_policy import should_retry
        self.assertFalse(should_retry("dependency_failed", 0))

    def test_tool_error_retries(self):
        from error_policy import should_retry
        self.assertTrue(should_retry("tool_error", 0))
        self.assertTrue(should_retry("tool_error", 1))
        self.assertFalse(should_retry("tool_error", 2))


class TestAdjustedTimeout(unittest.TestCase):
    """Test timeout adjustment for retries."""

    def test_timeout_increase(self):
        from error_policy import get_adjusted_timeout
        new_timeout = get_adjusted_timeout("timeout", 100, 1)
        self.assertEqual(new_timeout, 150)  # 100 * 1.5

    def test_no_timeout_increase_first_try(self):
        from error_policy import get_adjusted_timeout
        new_timeout = get_adjusted_timeout("timeout", 100, 0)
        self.assertEqual(new_timeout, 100)

    def test_non_timeout_no_change(self):
        from error_policy import get_adjusted_timeout
        new_timeout = get_adjusted_timeout("model_error", 100, 1)
        self.assertEqual(new_timeout, 100)


class TestRetryAction(unittest.TestCase):
    """Test get_retry_action comprehensive output."""

    def test_model_error_action(self):
        from error_policy import get_retry_action
        action = get_retry_action("model_error")
        self.assertTrue(action["should_retry"])
        self.assertEqual(action["action"], "retry")
        self.assertTrue(action["use_fallback_model"])

    def test_budget_exceeded_action(self):
        from error_policy import get_retry_action
        action = get_retry_action("budget_exceeded")
        self.assertFalse(action["should_retry"])
        self.assertEqual(action["action"], "pause")

    def test_dependency_failed_action(self):
        from error_policy import get_retry_action
        action = get_retry_action("dependency_failed")
        self.assertFalse(action["should_retry"])
        self.assertEqual(action["action"], "fail")

    def test_all_categories_have_policies(self):
        from error_policy import ERROR_POLICIES
        from state_machine import ErrorCategory
        for cat in ErrorCategory:
            self.assertIn(
                cat.value, ERROR_POLICIES,
                f"Missing policy for {cat.value}",
            )


# ─── 9.4 Atomic DB Operations ────────────────────────────────────────────────

class TestAtomicWriteFile(unittest.TestCase):
    """Test atomic write_file via temp file + os.replace."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_write_creates_file(self):
        from tools.workspace import write_file
        result = run_async(write_file("test.txt", "hello world"))
        self.assertIn("✅", result)
        full_path = os.path.join(self.tmp_dir, "test.txt")
        self.assertTrue(os.path.isfile(full_path))
        with open(full_path) as f:
            self.assertEqual(f.read(), "hello world")

    def test_write_overwrites_atomically(self):
        from tools.workspace import write_file
        full_path = os.path.join(self.tmp_dir, "test.txt")
        with open(full_path, "w") as f:
            f.write("old content")
        result = run_async(write_file("test.txt", "new content"))
        self.assertIn("✅", result)
        with open(full_path) as f:
            self.assertEqual(f.read(), "new content")

    def test_no_temp_files_left(self):
        from tools.workspace import write_file
        run_async(write_file("clean.txt", "data"))
        leftovers = [f for f in os.listdir(self.tmp_dir) if f.startswith(".tmp_write_")]
        self.assertEqual(len(leftovers), 0)

    def test_append_mode_still_works(self):
        from tools.workspace import write_file
        run_async(write_file("app.txt", "line1\n"))
        run_async(write_file("app.txt", "line2\n", mode="append"))
        full_path = os.path.join(self.tmp_dir, "app.txt")
        with open(full_path) as f:
            content = f.read()
        self.assertIn("line1", content)
        self.assertIn("line2", content)

    def test_creates_parent_dirs(self):
        from tools.workspace import write_file
        result = run_async(write_file("sub/dir/file.txt", "nested"))
        self.assertIn("✅", result)
        full_path = os.path.join(self.tmp_dir, "sub", "dir", "file.txt")
        self.assertTrue(os.path.isfile(full_path))


class TestWALCheckpoint(unittest.TestCase):
    """Test that close_db calls WAL checkpoint."""

    def test_close_db_source_has_wal_checkpoint(self):
        src = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "db.py",
        )
        with open(src, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("PRAGMA wal_checkpoint(TRUNCATE)", content)


# ─── 9.5 Per-Task Cost Budgets ───────────────────────────────────────────────

class TestPerTaskBudgets(unittest.TestCase):
    """Test per-task and per-mission cost budget functions."""

    def setUp(self):
        """Set up a fresh in-memory DB for testing."""
        import db as db_mod
        self._orig_path = db_mod.DB_PATH
        self.tmp_dir = tempfile.mkdtemp()
        db_mod.DB_PATH = os.path.join(self.tmp_dir, "test.db")
        # Force new connection
        db_mod._db_connection = None
        run_async(db_mod.init_db())

    def tearDown(self):
        import db as db_mod
        run_async(db_mod.close_db())
        db_mod.DB_PATH = self._orig_path
        db_mod._db_connection = None
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_task_has_max_cost_column(self):
        import db as db_mod

        async def check():
            db = await db_mod.get_db()
            cursor = await db.execute("PRAGMA table_info(tasks)")
            cols = [row[1] for row in await cursor.fetchall()]
            return cols

        cols = run_async(check())
        self.assertIn("max_cost", cols)

    def test_task_has_error_category_column(self):
        import db as db_mod

        async def check():
            db = await db_mod.get_db()
            cursor = await db.execute("PRAGMA table_info(tasks)")
            cols = [row[1] for row in await cursor.fetchall()]
            return cols

        cols = run_async(check())
        self.assertIn("error_category", cols)

    def test_get_task_cost_empty(self):
        from db import get_task_cost
        cost = run_async(get_task_cost(999))
        self.assertEqual(cost, 0.0)

    def test_get_task_cost_with_conversations(self):
        import db as db_mod

        async def setup_and_check():
            task_id = await db_mod.add_task("Test task", "desc")
            await db_mod.log_conversation(task_id, "assistant", "resp1", cost=0.05)
            await db_mod.log_conversation(task_id, "assistant", "resp2", cost=0.10)
            return await db_mod.get_task_cost(task_id)

        cost = run_async(setup_and_check())
        self.assertAlmostEqual(cost, 0.15, places=4)

    def test_get_mission_total_cost(self):
        import db as db_mod

        async def setup_and_check():
            mission_id = await db_mod.add_mission("Test mission", "desc")
            task1 = await db_mod.add_task("T1", "d", mission_id=mission_id)
            task2 = await db_mod.add_task("T2", "d", mission_id=mission_id)
            await db_mod.log_conversation(task1, "assistant", "r1", cost=0.10)
            await db_mod.log_conversation(task2, "assistant", "r2", cost=0.20)
            return await db_mod.get_mission_total_cost(mission_id)

        cost = run_async(setup_and_check())
        self.assertAlmostEqual(cost, 0.30, places=4)

    def test_check_task_budget_no_limit(self):
        import db as db_mod

        async def check():
            task_id = await db_mod.add_task("T", "d")
            return await db_mod.check_task_budget(task_id)

        result = run_async(check())
        self.assertTrue(result["ok"])

    def test_check_task_budget_within(self):
        import db as db_mod

        async def check():
            task_id = await db_mod.add_task("T", "d")
            await db_mod.update_task(task_id, max_cost=1.0)
            await db_mod.log_conversation(task_id, "assistant", "r", cost=0.5)
            return await db_mod.check_task_budget(task_id, additional_cost=0.1)

        result = run_async(check())
        self.assertTrue(result["ok"])

    def test_check_task_budget_exceeded(self):
        import db as db_mod

        async def check():
            task_id = await db_mod.add_task("T", "d")
            await db_mod.update_task(task_id, max_cost=0.5)
            await db_mod.log_conversation(task_id, "assistant", "r", cost=0.4)
            return await db_mod.check_task_budget(task_id, additional_cost=0.2)

        result = run_async(check())
        self.assertFalse(result["ok"])
        self.assertIn("exceeded", result["reason"])


# ─── 9.6 State Machine Transition (DB Integration) ──────────────────────────

class TestTransitionTaskDB(unittest.TestCase):
    """Test transition_task with real DB."""

    def setUp(self):
        import db as db_mod
        self._orig_path = db_mod.DB_PATH
        self.tmp_dir = tempfile.mkdtemp()
        db_mod.DB_PATH = os.path.join(self.tmp_dir, "test.db")
        db_mod._db_connection = None
        run_async(db_mod.init_db())

    def tearDown(self):
        import db as db_mod
        run_async(db_mod.close_db())
        db_mod.DB_PATH = self._orig_path
        db_mod._db_connection = None
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_valid_transition(self):
        import db as db_mod
        from state_machine import transition_task

        async def check():
            task_id = await db_mod.add_task("T", "d")
            await transition_task(task_id, "processing")
            task = await db_mod.get_task(task_id)
            return task["status"]

        status = run_async(check())
        self.assertEqual(status, "processing")

    def test_invalid_transition_raises(self):
        import db as db_mod
        from state_machine import transition_task, InvalidTransition

        async def check():
            task_id = await db_mod.add_task("T", "d")
            # pending → completed is invalid
            await transition_task(task_id, "completed")

        with self.assertRaises(InvalidTransition):
            run_async(check())

    def test_transition_with_error_category(self):
        import db as db_mod
        from state_machine import transition_task

        async def check():
            task_id = await db_mod.add_task("T", "d")
            await transition_task(task_id, "processing")
            await transition_task(
                task_id, "failed",
                error="Timed out after 300s",
                error_category="timeout",
            )
            task = await db_mod.get_task(task_id)
            return task

        task = run_async(check())
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["error_category"], "timeout")
        self.assertIn("Timed out", task["error"])

    def test_transition_with_extra_fields(self):
        import db as db_mod
        from state_machine import transition_task
        from datetime import datetime

        async def check():
            task_id = await db_mod.add_task("T", "d")
            await transition_task(task_id, "processing")
            await transition_task(
                task_id, "completed",
                result="All done",
                completed_at=datetime.now().isoformat(),
            )
            task = await db_mod.get_task(task_id)
            return task

        task = run_async(check())
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["result"], "All done")

    def test_transition_task_not_found(self):
        from state_machine import transition_task

        async def check():
            await transition_task(99999, "processing")

        with self.assertRaises(ValueError):
            run_async(check())

    def test_retry_flow(self):
        """Test the full retry flow: pending → processing → failed → pending."""
        import db as db_mod
        from state_machine import transition_task

        async def check():
            task_id = await db_mod.add_task("T", "d")
            await transition_task(task_id, "processing")
            await transition_task(
                task_id, "failed",
                error="Model API error",
                error_category="model_error",
            )
            await transition_task(
                task_id, "pending",
                retry_count=1,
            )
            task = await db_mod.get_task(task_id)
            return task

        task = run_async(check())
        self.assertEqual(task["status"], "pending")
        self.assertEqual(task["retry_count"], 1)


# ─── 9.7 Module Files Exist ─────────────────────────────────────────────────

class TestPhase9Files(unittest.TestCase):

    def _src_path(self, name):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            name,
        )

    def test_state_machine_exists(self):
        self.assertTrue(os.path.isfile(self._src_path("state_machine.py")))

    def test_error_policy_exists(self):
        self.assertTrue(os.path.isfile(self._src_path("error_policy.py")))

    def test_state_machine_importable(self):
        from state_machine import (
            TaskState, ErrorCategory, TRANSITIONS,
            validate_transition, transition_task, classify_error,
            InvalidTransition,
        )
        # Just verify they all import without error

    def test_error_policy_importable(self):
        from error_policy import (
            RetryPolicy, ERROR_POLICIES,
            get_retry_policy, should_retry,
            get_adjusted_timeout, get_retry_action,
        )
        # Just verify they all import without error

    def test_db_has_wal_checkpoint(self):
        with open(self._src_path("db.py"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("wal_checkpoint", content)

    def test_workspace_has_atomic_write(self):
        with open(
            os.path.join(self._src_path("tools"), "workspace.py"),
            encoding="utf-8",
        ) as f:
            content = f.read()
        self.assertIn("os.replace", content)
        self.assertIn("tempfile", content)


if __name__ == "__main__":
    unittest.main()
