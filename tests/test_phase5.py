# tests/test_phase5.py
"""
Tests for Phase 5: Agent Architecture Improvements

  5.1 Execution patterns (react_loop vs single_shot)
  5.2 Self-reflection mechanism
  5.3 Confidence-gated output
  5.4 Error recovery agent structure
  5.5 Tier escalation logic
  5.6 Agent registry completeness
"""
import asyncio
import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _patch_db_path(db_mod, db_path):
    import config
    config.DB_PATH = db_path
    db_mod.DB_PATH = db_path


class _DBTestBase(unittest.TestCase):
    def setUp(self):
        if not HAS_AIOSQLITE:
            self.skipTest("aiosqlite not installed")
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()

        import config
        import db as db_mod
        self._orig_config_path = config.DB_PATH
        self._orig_db_path = db_mod.DB_PATH
        self.db_mod = db_mod

        _patch_db_path(db_mod, self.db_path)
        db_mod._db_connection = None
        run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


# ────────────────────────────────────────────────────────────────────
# Replicated pure functions from agents/base.py to avoid litellm
# dependency at import time (same pattern as test_phase4.py).
# ────────────────────────────────────────────────────────────────────

ESCALATION_THRESHOLD = 3
TIER_ESCALATION_ORDER = ["cheap", "code", "medium", "expensive"]


def _escalate_tier(current_tier: str):
    """Return the next tier up, or None if already at highest."""
    try:
        idx = TIER_ESCALATION_ORDER.index(current_tier)
    except ValueError:
        return None
    if idx < len(TIER_ESCALATION_ORDER) - 1:
        return TIER_ESCALATION_ORDER[idx + 1]
    return None


def _normalize_action(parsed: dict):
    """Minimal replica of BaseAgent._normalize_action for testing."""
    action = parsed.get("action")

    _aliases = {
        "tool": "tool_call", "use_tool": "tool_call",
        "execute": "tool_call", "call": "tool_call",
        "run": "tool_call", "invoke": "tool_call",
        "answer": "final_answer", "respond": "final_answer",
        "response": "final_answer", "reply": "final_answer",
        "complete": "final_answer", "done": "final_answer",
        "output": "final_answer", "finish": "final_answer",
        "result": "final_answer", "final": "final_answer",
        "summary": "final_answer",
        "ask": "clarify", "question": "clarify",
        "clarification": "clarify",
        "plan": "decompose", "decompose": "decompose",
        "break_down": "decompose",
    }
    if action in _aliases:
        action = _aliases[action]
        parsed["action"] = action

    if action in ("think", "thinking", "reasoning", "analyze",
                   "observation", "reflect", "consider"):
        return None

    if not action:
        if "tool" in parsed:
            parsed["action"] = "tool_call"
        elif any(k in parsed for k in (
            "result", "answer", "response", "text",
            "message", "output", "content", "reply",
        )):
            parsed["action"] = "final_answer"
            for key in ("answer", "response", "text", "message",
                        "output", "content", "reply"):
                if key in parsed and "result" not in parsed:
                    parsed["result"] = parsed.pop(key)
                    break
        else:
            return None

    return parsed


def _parse_agent_response(content: str) -> dict:
    """Minimal replica of BaseAgent._parse_agent_response for testing."""
    cleaned = content.strip()

    # Try direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            norm = _normalize_action(obj)
            if norm is not None:
                return norm
    except (json.JSONDecodeError, IndexError):
        pass

    # Try ```json blocks
    json_blocks = re.findall(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL
    )
    for block in json_blocks:
        try:
            obj = json.loads(block.strip())
            if isinstance(obj, dict):
                norm = _normalize_action(obj)
                if norm is not None:
                    return norm
        except (json.JSONDecodeError, IndexError):
            pass

    # Fallback
    return {"action": "final_answer", "result": content}


def _validate_response(result: str, task: dict):
    """Minimal replica of BaseAgent._validate_response."""
    if not result or not result.strip():
        return "Your response was empty. Please provide a substantive answer."

    stripped = result.strip()
    title = task.get("title", "").lower()
    trivial_keywords = ["list", "ls", "status", "count", "version", "ping"]
    is_trivial = any(kw in title for kw in trivial_keywords)

    if not is_trivial and len(stripped) < 20:
        return (
            "Your response seems too short for this task. "
            "Please provide a more complete answer."
        )

    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "as an ai",
        "i don't have access", "i am not able",
    ]
    lower = stripped.lower()
    if any(p in lower for p in refusal_patterns) and len(stripped) < 100:
        return (
            "Your response appears to be a refusal. "
            "Try a different approach or use the available tools."
        )
    return None


def _is_action_task(task: dict) -> bool:
    """Minimal replica of BaseAgent._is_action_task."""
    text = (
        f"{task.get('title', '')} {task.get('description', '')}"
    ).lower().strip()

    question_starts = [
        "what ", "who ", "why ", "when ", "where ",
        "how does ", "how is ", "how do ",
        "explain ", "describe ", "summarize ",
        "what's ", "what is ", "do you ", "can you tell",
        "is there ", "are there ", "which ",
    ]
    if any(text.startswith(q) for q in question_starts):
        return False

    strong_verbs = [
        "fetch", "download", "install", "deploy", "execute",
        "run ", "run:", "clone", "pull ", "push ", "start ",
        "stop ", "restart", "compile", "test ", "debug",
        "setup", "set up", "configure", "scan", "scrape",
        "crawl", "ping", "ssh ", "curl ", "grep ", "find ",
        "launch", "migrate", "import ", "export ",
    ]
    if any(v in text for v in strong_verbs):
        return True

    context_verbs = [
        "list", "create", "build", "write", "read",
        "check", "update", "delete", "remove", "add ",
        "modify", "edit", "open", "search", "look up",
        "analyze", "monitor", "show",
    ]
    tech_targets = [
        "file", "folder", "directory", "repo", "repos",
        "repository", "repositories", "server", "database",
        "api", "endpoint", "package", "container", "docker",
        "service", "script", "code", "project", "workspace",
        "branch", "commit", "log ", "logs", "port", "process",
        "module", "dependency", "dependencies", "config",
    ]
    has_verb = any(v in text for v in context_verbs)
    has_target = any(t in text for t in tech_targets)
    return has_verb and has_target


# ─── 5.1 Execution Patterns ──────────────────────────────────────────────

class TestExecutionPatterns(unittest.TestCase):
    """Verify the execution_pattern attribute and routing logic."""

    def test_default_pattern_is_react_loop(self):
        """BaseAgent should default to react_loop."""
        # We can't import BaseAgent directly (litellm), so check the source
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn('execution_pattern: str = "react_loop"', src)

    def test_single_shot_pattern_exists(self):
        """execute_single_shot method must exist in BaseAgent."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("async def execute_single_shot(self, task: dict)", src)

    def test_execute_routes_on_pattern(self):
        """execute() must route based on execution_pattern."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn('if self.execution_pattern == "single_shot"', src)
        self.assertIn("return await self.execute_single_shot(task)", src)
        self.assertIn("return await self._execute_react_loop(task)", src)

    def test_react_loop_method_exists(self):
        """_execute_react_loop method must exist in BaseAgent."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("async def _execute_react_loop(self, task: dict)", src)


# ─── 5.2 Self-Reflection ─────────────────────────────────────────────────

class TestSelfReflection(unittest.TestCase):
    """Verify self-reflection mechanism."""

    def test_self_reflection_attribute_exists(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("enable_self_reflection: bool = False", src)

    def test_self_reflect_method_exists(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("async def _self_reflect(", src)

    def test_self_reflection_in_react_loop(self):
        """Self-reflection must be invoked in final_answer handler."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("if self.enable_self_reflection:", src)
        self.assertIn("await self._self_reflect(", src)

    def test_self_reflect_checks_verdict(self):
        """_self_reflect should check for 'fix' verdict."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"verdict"', src)
        self.assertIn('"fix"', src)
        self.assertIn('"corrected_result"', src)

    def test_self_reflection_prompt_has_reviewer_instructions(self):
        """The self-reflection system prompt should ask to review for errors."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        # Find the _self_reflect method
        idx = src.index("async def _self_reflect(")
        method_block = src[idx:idx + 1000]
        self.assertIn("errors", method_block.lower())
        self.assertIn("hallucinations", method_block.lower())


# ─── 5.3 Confidence-Gated Output ─────────────────────────────────────────

class TestConfidenceGating(unittest.TestCase):
    """Verify confidence-gated output mechanism."""

    def test_min_confidence_attribute(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("min_confidence: int = 0", src)

    def test_confidence_check_in_final_answer(self):
        """Low confidence should route to reviewer."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("self.min_confidence", src)
        self.assertIn('"needs_review"', src)

    def test_needs_review_has_confidence_info(self):
        """needs_review response must include the confidence score."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("review_note", src)
        self.assertIn("Agent confidence:", src)

    def test_confidence_zero_means_disabled(self):
        """When min_confidence is 0, gating should be skipped."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        # The code checks `self.min_confidence > 0`
        self.assertIn("self.min_confidence > 0", src)


# ─── 5.4 Error Recovery Agent ────────────────────────────────────────────

class TestErrorRecoveryAgent(unittest.TestCase):
    """Verify ErrorRecoveryAgent structure and registration."""

    def test_error_recovery_file_exists(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "error_recovery.py",
        )
        self.assertTrue(os.path.exists(path))

    def test_error_recovery_class_structure(self):
        """ErrorRecoveryAgent must have required attributes."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "error_recovery.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn('name = "error_recovery"', src)
        self.assertIn("class ErrorRecoveryAgent(BaseAgent)", src)
        self.assertIn("enable_self_reflection = True", src)
        self.assertIn("def get_system_prompt(self, task: dict)", src)

    def test_error_recovery_has_diagnostic_categories(self):
        """System prompt should list root cause categories."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "error_recovery.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("Bad prompt", src)
        self.assertIn("Missing tool", src)
        self.assertIn("Model too weak", src)
        self.assertIn("Missing dependency", src)
        self.assertIn("Environment issue", src)
        self.assertIn("Logic error", src)

    def test_error_recovery_registered(self):
        """ErrorRecoveryAgent must be in AGENT_REGISTRY."""
        init_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "__init__.py",
        )
        with open(init_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"error_recovery":', src)
        self.assertIn("ErrorRecoveryAgent()", src)
        self.assertIn("from agents.error_recovery import ErrorRecoveryAgent", src)

    def test_error_recovery_allowed_tools(self):
        """ErrorRecoveryAgent should have specific allowed tools."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "error_recovery.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("allowed_tools", src)
        self.assertIn("shell", src)
        self.assertIn("read_file", src)
        self.assertIn("write_file", src)


# ─── 5.5 Tier Escalation (replicated logic) ──────────────────────────────

class TestTierEscalation(unittest.TestCase):
    """Test tier escalation logic using replicated functions."""

    def test_escalate_cheap_to_code(self):
        self.assertEqual(_escalate_tier("cheap"), "code")

    def test_escalate_code_to_medium(self):
        self.assertEqual(_escalate_tier("code"), "medium")

    def test_escalate_medium_to_expensive(self):
        self.assertEqual(_escalate_tier("medium"), "expensive")

    def test_escalate_expensive_returns_none(self):
        self.assertIsNone(_escalate_tier("expensive"))

    def test_escalate_unknown_returns_none(self):
        self.assertIsNone(_escalate_tier("nonexistent"))

    def test_escalation_threshold(self):
        self.assertEqual(ESCALATION_THRESHOLD, 3)

    def test_tier_order(self):
        self.assertEqual(
            TIER_ESCALATION_ORDER,
            ["cheap", "code", "medium", "expensive"],
        )


# ─── 5.6 Response Parsing (replicated) ───────────────────────────────────

class TestResponseParsing(unittest.TestCase):
    """Test parse_agent_response with various formats."""

    def test_clean_json_final_answer(self):
        resp = _parse_agent_response(
            '{"action": "final_answer", "result": "done"}'
        )
        self.assertEqual(resp["action"], "final_answer")
        self.assertEqual(resp["result"], "done")

    def test_json_in_fences(self):
        resp = _parse_agent_response(
            '```json\n{"action": "tool_call", "tool": "shell", '
            '"args": {"command": "ls"}}\n```'
        )
        self.assertEqual(resp["action"], "tool_call")
        self.assertEqual(resp["tool"], "shell")

    def test_plain_text_fallback(self):
        resp = _parse_agent_response("Just a plain answer.")
        self.assertEqual(resp["action"], "final_answer")
        self.assertEqual(resp["result"], "Just a plain answer.")

    def test_action_aliases(self):
        resp = _parse_agent_response(
            '{"action": "done", "result": "finished"}'
        )
        self.assertEqual(resp["action"], "final_answer")

    def test_confidence_in_response(self):
        resp = _parse_agent_response(
            '{"action": "final_answer", "result": "hello", "confidence": 4}'
        )
        self.assertEqual(resp["action"], "final_answer")
        self.assertEqual(resp.get("confidence"), 4)

    def test_decompose_action(self):
        resp = _parse_agent_response(
            '{"action": "decompose", "subtasks": [{"title": "a"}]}'
        )
        self.assertEqual(resp["action"], "decompose")
        self.assertEqual(len(resp["subtasks"]), 1)


# ─── 5.7 Output Validation (replicated) ──────────────────────────────────

class TestOutputValidation(unittest.TestCase):
    """Test _validate_response logic."""

    def test_empty_response_rejected(self):
        err = _validate_response("", {"title": "task"})
        self.assertIsNotNone(err)
        self.assertIn("empty", err.lower())

    def test_short_response_rejected(self):
        err = _validate_response("ok", {"title": "write code"})
        self.assertIsNotNone(err)
        self.assertIn("short", err.lower())

    def test_trivial_task_allows_short(self):
        err = _validate_response("ok", {"title": "list files"})
        self.assertIsNone(err)

    def test_refusal_rejected(self):
        err = _validate_response(
            "I cannot do that because I don't have access to your server.",
            {"title": "deploy server"},
        )
        self.assertIsNotNone(err)
        self.assertIn("refusal", err.lower())

    def test_valid_response_passes(self):
        err = _validate_response(
            "Here is the complete implementation of the feature with all tests passing.",
            {"title": "implement feature"},
        )
        self.assertIsNone(err)


# ─── 5.8 Action Task Detection (replicated) ──────────────────────────────

class TestActionTaskDetection(unittest.TestCase):
    """Test _is_action_task heuristic."""

    def test_question_not_action(self):
        self.assertFalse(_is_action_task({"title": "What is Python?", "description": ""}))

    def test_strong_verb_is_action(self):
        self.assertTrue(_is_action_task({"title": "deploy the server", "description": ""}))
        self.assertTrue(_is_action_task({"title": "install packages", "description": ""}))

    def test_context_verb_with_tech_target(self):
        self.assertTrue(_is_action_task({"title": "create the config file", "description": ""}))

    def test_context_verb_without_tech_target(self):
        self.assertFalse(_is_action_task({"title": "create a poem", "description": ""}))

    def test_describe_not_action(self):
        self.assertFalse(_is_action_task({"title": "describe the architecture", "description": ""}))


# ─── 5.9 Single-Shot Method Structure ────────────────────────────────────

class TestSingleShotStructure(unittest.TestCase):
    """Verify execute_single_shot has proper structure."""

    def test_single_shot_returns_needs_subtasks(self):
        """single_shot should handle decompose actions."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        idx = src.index("async def execute_single_shot(")
        method_end = src.index("\n    async def ", idx + 1)
        method = src[idx:method_end]
        self.assertIn("needs_subtasks", method)
        self.assertIn("completed", method)

    def test_single_shot_has_error_handling(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        idx = src.index("async def execute_single_shot(")
        method_end = src.index("\n    async def ", idx + 1)
        method = src[idx:method_end]
        self.assertIn("except Exception", method)


# ─── 5.10 Integration: ErrorRecovery + Self-Reflection ───────────────────

class TestErrorRecoveryIntegration(unittest.TestCase):
    """Verify error_recovery agent has self_reflection enabled."""

    def test_error_recovery_uses_self_reflection(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "error_recovery.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("enable_self_reflection = True", src)

    def test_error_recovery_uses_medium_tier(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "error_recovery.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn('default_tier = "medium"', src)

    def test_error_recovery_max_iterations(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "error_recovery.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("max_iterations = 4", src)


# ─── 5.11 Hallucination Guard ────────────────────────────────────────────

class TestHallucinationGuard(unittest.TestCase):
    """Verify the hallucination guard in the react loop."""

    def test_guard_exists_in_react_loop(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("HALLUCINATION GUARD", src)
        self.assertIn("not tools_used", src)
        self.assertIn("_is_action_task", src)

    def test_guard_only_fires_early_iterations(self):
        """Guard should only activate in first 2 iterations."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("iteration < 2", src)


# ─── 5.12 Phase 5 Attributes Completeness ────────────────────────────────

class TestPhase5Attributes(unittest.TestCase):
    """Verify all Phase 5 attributes and methods exist in BaseAgent."""

    def setUp(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "base.py",
        )
        with open(src_path, encoding="utf-8") as f:
            self.src = f.read()

    def test_execution_pattern_attribute(self):
        self.assertIn("execution_pattern:", self.src)

    def test_enable_self_reflection_attribute(self):
        self.assertIn("enable_self_reflection:", self.src)

    def test_min_confidence_attribute(self):
        self.assertIn("min_confidence:", self.src)

    def test_execute_method_routes(self):
        self.assertIn("async def execute(self, task: dict)", self.src)

    def test_execute_react_loop_method(self):
        self.assertIn("async def _execute_react_loop(self, task: dict)", self.src)

    def test_execute_single_shot_method(self):
        self.assertIn("async def execute_single_shot(self, task: dict)", self.src)

    def test_self_reflect_method(self):
        self.assertIn("async def _self_reflect(", self.src)

    def test_escalate_tier_method(self):
        self.assertIn("def _escalate_tier(current_tier: str)", self.src)


if __name__ == "__main__":
    unittest.main()
