# tests/test_phase13.py
"""
Phase 13 — Agent Collaboration tests.

Covers:
  13.1 — Shared Blackboard
  13.2 — Plan Verification
  13.3 — Agent-to-Agent Queries
  13.4 — Parallel Independent Tasks
  13.5 — Interactive Plan Approval
"""
import asyncio
import json
import os
import sys
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# 13.1 — Shared Blackboard
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlackboardModule(unittest.TestCase):
    """Blackboard module exists with expected API."""

    def test_module_imports(self):
        import collaboration.blackboard as bb
        self.assertTrue(hasattr(bb, "read_blackboard"))
        self.assertTrue(hasattr(bb, "write_blackboard"))
        self.assertTrue(hasattr(bb, "update_blackboard_entry"))
        self.assertTrue(hasattr(bb, "get_or_create_blackboard"))
        self.assertTrue(hasattr(bb, "append_blackboard"))
        self.assertTrue(hasattr(bb, "format_blackboard_for_prompt"))

    def test_default_blackboard_schema(self):
        from collaboration.blackboard import DEFAULT_BLACKBOARD
        self.assertIn("architecture", DEFAULT_BLACKBOARD)
        self.assertIn("files", DEFAULT_BLACKBOARD)
        self.assertIn("decisions", DEFAULT_BLACKBOARD)
        self.assertIn("open_issues", DEFAULT_BLACKBOARD)
        self.assertIn("constraints", DEFAULT_BLACKBOARD)

    def test_default_blackboard_types(self):
        from collaboration.blackboard import DEFAULT_BLACKBOARD
        self.assertIsInstance(DEFAULT_BLACKBOARD["architecture"], dict)
        self.assertIsInstance(DEFAULT_BLACKBOARD["files"], dict)
        self.assertIsInstance(DEFAULT_BLACKBOARD["decisions"], list)
        self.assertIsInstance(DEFAULT_BLACKBOARD["open_issues"], list)
        self.assertIsInstance(DEFAULT_BLACKBOARD["constraints"], list)


class TestBlackboardOperations(unittest.TestCase):
    """In-memory blackboard operations (no DB)."""

    def setUp(self):
        from collaboration.blackboard import _BLACKBOARD_CACHE, DEFAULT_BLACKBOARD
        import json
        # Set up a fresh in-memory blackboard for goal_id=99
        _BLACKBOARD_CACHE[99] = json.loads(json.dumps(DEFAULT_BLACKBOARD))

    def tearDown(self):
        from collaboration.blackboard import clear_cache
        clear_cache(99)

    def test_write_and_read(self):
        from collaboration.blackboard import write_blackboard, read_blackboard
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                write_blackboard(99, "decisions", [
                    {"what": "Use FastAPI", "why": "performance", "by": "architect"}
                ])
            )
            result = loop.run_until_complete(read_blackboard(99, "decisions"))
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["what"], "Use FastAPI")
        finally:
            loop.close()

    def test_read_full_board(self):
        from collaboration.blackboard import read_blackboard
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(read_blackboard(99))
            self.assertIsInstance(result, dict)
            self.assertIn("architecture", result)
            self.assertIn("files", result)
        finally:
            loop.close()

    def test_read_missing_key(self):
        from collaboration.blackboard import read_blackboard
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(read_blackboard(99, "nonexistent"))
            self.assertIsNone(result)
        finally:
            loop.close()

    def test_update_entry_dict(self):
        from collaboration.blackboard import update_blackboard_entry, _BLACKBOARD_CACHE
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                update_blackboard_entry(99, "files", "app.py", {
                    "status": "implemented",
                    "interface_hash": "abc123",
                })
            )
            self.assertEqual(
                _BLACKBOARD_CACHE[99]["files"]["app.py"]["status"],
                "implemented",
            )
        finally:
            loop.close()

    def test_update_entry_list(self):
        from collaboration.blackboard import update_blackboard_entry, _BLACKBOARD_CACHE
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                update_blackboard_entry(99, "decisions", "key1", "value1")
            )
            decisions = _BLACKBOARD_CACHE[99]["decisions"]
            self.assertTrue(len(decisions) > 0)
        finally:
            loop.close()

    def test_append_blackboard(self):
        from collaboration.blackboard import append_blackboard, _BLACKBOARD_CACHE
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                append_blackboard(99, "open_issues", "Need to handle auth")
            )
            loop.run_until_complete(
                append_blackboard(99, "open_issues", "Performance concern")
            )
            issues = _BLACKBOARD_CACHE[99]["open_issues"]
            self.assertEqual(len(issues), 2)
            self.assertIn("Need to handle auth", issues)
        finally:
            loop.close()

    def test_write_architecture(self):
        from collaboration.blackboard import write_blackboard, read_blackboard
        loop = asyncio.new_event_loop()
        try:
            arch = {"framework": "FastAPI", "db": "PostgreSQL", "cache": "Redis"}
            loop.run_until_complete(write_blackboard(99, "architecture", arch))
            result = loop.run_until_complete(read_blackboard(99, "architecture"))
            self.assertEqual(result["framework"], "FastAPI")
        finally:
            loop.close()

    def test_write_constraints(self):
        from collaboration.blackboard import write_blackboard, read_blackboard
        loop = asyncio.new_event_loop()
        try:
            constraints = ["Must use Python 3.12+", "Must support PostgreSQL"]
            loop.run_until_complete(write_blackboard(99, "constraints", constraints))
            result = loop.run_until_complete(read_blackboard(99, "constraints"))
            self.assertEqual(len(result), 2)
        finally:
            loop.close()

    def test_clear_cache(self):
        from collaboration.blackboard import clear_cache, _BLACKBOARD_CACHE
        _BLACKBOARD_CACHE[200] = {"test": True}
        clear_cache(200)
        self.assertNotIn(200, _BLACKBOARD_CACHE)

    def test_clear_all_cache(self):
        from collaboration.blackboard import clear_cache, _BLACKBOARD_CACHE
        _BLACKBOARD_CACHE[201] = {"test": True}
        _BLACKBOARD_CACHE[202] = {"test": True}
        clear_cache()
        self.assertEqual(len(_BLACKBOARD_CACHE), 0)


class TestBlackboardFormatting(unittest.TestCase):
    """Blackboard prompt formatting."""

    def test_format_empty_board(self):
        from collaboration.blackboard import format_blackboard_for_prompt, DEFAULT_BLACKBOARD
        result = format_blackboard_for_prompt(DEFAULT_BLACKBOARD)
        self.assertEqual(result, "")

    def test_format_with_architecture(self):
        from collaboration.blackboard import format_blackboard_for_prompt
        board = {
            "architecture": {"framework": "FastAPI"},
            "files": {},
            "decisions": [],
            "open_issues": [],
            "constraints": [],
        }
        result = format_blackboard_for_prompt(board)
        self.assertIn("Shared Blackboard", result)
        self.assertIn("Architecture", result)
        self.assertIn("FastAPI", result)

    def test_format_with_decisions(self):
        from collaboration.blackboard import format_blackboard_for_prompt
        board = {
            "architecture": {},
            "files": {},
            "decisions": [
                {"what": "Use FastAPI", "why": "speed", "by": "architect"},
            ],
            "open_issues": [],
            "constraints": [],
        }
        result = format_blackboard_for_prompt(board)
        self.assertIn("Key Decisions", result)
        self.assertIn("Use FastAPI", result)

    def test_format_with_files(self):
        from collaboration.blackboard import format_blackboard_for_prompt
        board = {
            "architecture": {},
            "files": {"app.py": {"status": "implemented"}},
            "decisions": [],
            "open_issues": [],
            "constraints": [],
        }
        result = format_blackboard_for_prompt(board)
        self.assertIn("File Status", result)
        self.assertIn("app.py", result)

    def test_format_max_chars(self):
        from collaboration.blackboard import format_blackboard_for_prompt
        board = {
            "architecture": {"key": "x" * 5000},
            "files": {},
            "decisions": [],
            "open_issues": [],
            "constraints": [],
        }
        result = format_blackboard_for_prompt(board, max_chars=200)
        self.assertTrue(len(result) <= 230)  # 200 + truncation message

    def test_format_none_board(self):
        from collaboration.blackboard import format_blackboard_for_prompt
        result = format_blackboard_for_prompt(None)
        self.assertEqual(result, "")

    def test_format_empty_dict(self):
        from collaboration.blackboard import format_blackboard_for_prompt
        result = format_blackboard_for_prompt({})
        self.assertEqual(result, "")


def _read_source(relative_path: str) -> str:
    """Read a source file relative to project root with UTF-8 encoding."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return open(os.path.join(root, relative_path), encoding="utf-8").read()


class TestBlackboardToolRegistration(unittest.TestCase):
    """13.1 — Blackboard tools registered."""

    def test_blackboard_tools_in_registry(self):
        source = _read_source("tools/__init__.py")
        self.assertIn("read_blackboard", source)
        self.assertIn("write_blackboard", source)

    def test_blackboard_injection_in_base_agent(self):
        source = _read_source("agents/base.py")
        self.assertIn("blackboard", source.lower())
        self.assertIn("format_blackboard_for_prompt", source)


# ═══════════════════════════════════════════════════════════════════════════════
# 13.2 — Plan Verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlanVerificationModule(unittest.TestCase):
    """Plan verification module exists."""

    def test_module_imports(self):
        import collaboration.plan_verification as pv
        self.assertTrue(hasattr(pv, "verify_plan"))

    def test_verify_valid_plan(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "A", "agent_type": "coder", "depends_on_step": None},
            {"title": "B", "agent_type": "coder", "depends_on_step": 0},
            {"title": "C", "agent_type": "writer", "depends_on_step": 1},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertEqual(len(issues), 0, f"Expected no issues: {issues}")

    def test_verify_empty_plan(self):
        from collaboration.plan_verification import verify_plan
        issues = verify_plan([], goal_budget=10.0)
        self.assertEqual(len(issues), 0)


class TestPlanVerificationCycles(unittest.TestCase):
    """Cycle detection in dependency graphs."""

    def test_acyclic_chain(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "A", "agent_type": "coder", "depends_on_step": None},
            {"title": "B", "agent_type": "coder", "depends_on_step": 0},
            {"title": "C", "agent_type": "coder", "depends_on_step": 1},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        cycle_issues = [i for i in issues if "cycl" in i.lower()]
        self.assertEqual(len(cycle_issues), 0)

    def test_cyclic_deps_detected(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "A", "agent_type": "coder", "depends_on_step": 2},
            {"title": "B", "agent_type": "coder", "depends_on_step": 0},
            {"title": "C", "agent_type": "coder", "depends_on_step": 1},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertTrue(any("cycl" in i.lower() for i in issues))

    def test_self_dependency(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "A", "agent_type": "coder", "depends_on_step": 0},  # self-dep
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertTrue(any("cycl" in i.lower() for i in issues))

    def test_parallel_no_deps(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "A", "agent_type": "coder", "depends_on_step": None},
            {"title": "B", "agent_type": "coder", "depends_on_step": None},
            {"title": "C", "agent_type": "coder", "depends_on_step": None},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        cycle_issues = [i for i in issues if "cycl" in i.lower()]
        self.assertEqual(len(cycle_issues), 0)


class TestPlanVerificationAgentTypes(unittest.TestCase):
    """Agent type assignment sanity checks."""

    def test_code_task_with_writer_flagged(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "Write unit tests for auth module",
             "description": "Write unit tests for auth.py",
             "agent_type": "writer"},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertTrue(any("agent" in i.lower() for i in issues))

    def test_research_task_with_researcher_ok(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "Research best frameworks",
             "description": "Compare Flask and FastAPI",
             "agent_type": "researcher"},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        agent_issues = [i for i in issues if "agent" in i.lower()]
        self.assertEqual(len(agent_issues), 0)

    def test_executor_always_accepted(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "Build the API",
             "description": "Create endpoints",
             "agent_type": "executor"},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        agent_issues = [i for i in issues if "agent" in i.lower()]
        self.assertEqual(len(agent_issues), 0)


class TestPlanVerificationDuplicates(unittest.TestCase):
    """Duplicate subtask detection."""

    def test_duplicate_titles(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "Build the API", "agent_type": "coder"},
            {"title": "Build the API", "agent_type": "coder"},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertTrue(any("duplicate" in i.lower() for i in issues))

    def test_unique_titles(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "Build the API", "agent_type": "coder"},
            {"title": "Write the docs", "agent_type": "writer"},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        dup_issues = [i for i in issues if "duplicate" in i.lower()]
        self.assertEqual(len(dup_issues), 0)


class TestPlanVerificationBudget(unittest.TestCase):
    """Budget checking."""

    def test_within_budget(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "A", "agent_type": "coder", "tier": "cheap"},
            {"title": "B", "agent_type": "coder", "tier": "cheap"},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        budget_issues = [i for i in issues if "budget" in i.lower()]
        self.assertEqual(len(budget_issues), 0)

    def test_over_budget(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": f"Task {i}", "agent_type": "coder", "tier": "expensive"}
            for i in range(100)
        ]
        issues = verify_plan(subtasks, goal_budget=0.01)
        self.assertTrue(any("budget" in i.lower() for i in issues))


class TestPlanVerificationIntegration(unittest.TestCase):
    """Orchestrator integration."""

    def test_orchestrator_calls_verify(self):
        source = _read_source("orchestrator.py")
        self.assertIn("verify_plan", source)


# ═══════════════════════════════════════════════════════════════════════════════
# 13.3 — Agent-to-Agent Queries
# ═══════════════════════════════════════════════════════════════════════════════

class TestAskAgentAction(unittest.TestCase):
    """ask_agent action type."""

    def test_action_model_exists(self):
        from models import AskAgentAction, ACTION_MODELS
        self.assertTrue(hasattr(AskAgentAction, "action"))
        self.assertIn("ask_agent", ACTION_MODELS)

    def test_action_fields(self):
        from models import AskAgentAction
        a = AskAgentAction(target="researcher", question="What is the best ORM?")
        self.assertEqual(a.action, "ask_agent")
        self.assertEqual(a.target, "researcher")
        self.assertEqual(a.question, "What is the best ORM?")

    def test_action_validates(self):
        from models import validate_action
        parsed = {
            "action": "ask_agent",
            "target": "researcher",
            "question": "test?",
        }
        result = validate_action(parsed)
        self.assertEqual(result["action"], "ask_agent")

    def test_handled_in_base_agent(self):
        source = _read_source("agents/base.py")
        self.assertIn("ask_agent", source)

    def test_json_schema_includes_ask_agent(self):
        from models import get_action_json_schema
        schema = get_action_json_schema()
        actions = schema["json_schema"]["schema"]["properties"]["action"]["enum"]
        self.assertIn("ask_agent", actions)


# ═══════════════════════════════════════════════════════════════════════════════
# 13.4 — Parallel Independent Tasks
# ═══════════════════════════════════════════════════════════════════════════════

class TestParallelTasks(unittest.TestCase):
    """Dynamic concurrency + file overlap detection."""

    def test_file_overlap_detection_exists(self):
        source = _read_source("orchestrator.py")
        self.assertIn("_detect_file_overlap", source)

    def test_file_overlap_positive(self):
        from orchestrator import _detect_file_overlap
        task_a = {"title": "Edit app.py", "description": "Update app.py and routes.py"}
        task_c = {"title": "Fix app.py", "description": "Fix bugs in app.py"}
        self.assertTrue(_detect_file_overlap(task_a, task_c))

    def test_file_overlap_negative(self):
        from orchestrator import _detect_file_overlap
        task_a = {"title": "Edit app.py", "description": "Update routes.py"}
        task_b = {"title": "Edit models.py", "description": "Update db.py"}
        self.assertFalse(_detect_file_overlap(task_a, task_b))

    def test_extract_file_refs(self):
        from orchestrator import _extract_file_refs
        task = {"title": "Update app.py", "description": "Fix routes.py and config.json"}
        refs = _extract_file_refs(task)
        self.assertIn("app.py", refs)
        self.assertIn("routes.py", refs)
        self.assertIn("config.json", refs)

    def test_compute_max_concurrent(self):
        from orchestrator import _compute_max_concurrent
        tasks = [
            {"goal_id": 1, "title": "A"},
            {"goal_id": 2, "title": "B"},
            {"goal_id": 3, "title": "C"},
        ]
        result = _compute_max_concurrent(tasks)
        self.assertGreaterEqual(result, 2)
        self.assertLessEqual(result, 4)

    def test_compute_max_concurrent_single_goal(self):
        from orchestrator import _compute_max_concurrent
        tasks = [
            {"goal_id": 1, "title": "A"},
            {"goal_id": 1, "title": "B"},
        ]
        result = _compute_max_concurrent(tasks)
        self.assertEqual(result, 2)  # base MAX_CONCURRENT_TASKS


# ═══════════════════════════════════════════════════════════════════════════════
# 13.5 — Interactive Plan Approval
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractivePlanApproval(unittest.TestCase):
    """Interactive plan approval via Telegram."""

    def test_approval_flow_in_orchestrator(self):
        source = _read_source("orchestrator.py")
        self.assertIn("_await_plan_approval", source)

    def test_telegram_approval_buttons(self):
        source = _read_source("telegram_bot.py")
        self.assertIn("approve_plan", source)
        self.assertIn("modify_plan", source)
        self.assertIn("reject_plan", source)

    def test_send_plan_approval_method(self):
        source = _read_source("telegram_bot.py")
        self.assertIn("send_plan_approval", source)

    def test_auto_approve_timeout_constant(self):
        source = _read_source("orchestrator.py")
        self.assertIn("AUTO_APPROVE_TIMEOUT", source)


if __name__ == "__main__":
    unittest.main()
