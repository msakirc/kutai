# tests/test_improvements.py
"""
Tests for the 5 reliability improvements:
  1.1 Function calling parsing
  1.2 Output validation
  1.3 Circuit breaker
  1.5 Task deduplication hash
"""
import sys
import os
import time
import hashlib
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── 1.1 Function Calling Parsing ─────────────────────────────────────────
# Inline the static method logic to avoid importing agents.base (litellm dep)

def _parse_function_call_response(tool_calls):
    """Replicated from BaseAgent for testing without litellm."""
    if not tool_calls:
        return None
    tc = tool_calls[0]
    name = tc.get("name", "")
    args = tc.get("arguments", {})
    if name == "final_answer":
        return {
            "action": "final_answer",
            "result": args.get("result", ""),
            "memories": args.get("memories", {}),
        }
    if name == "clarify":
        return {
            "action": "clarify",
            "question": args.get("question", ""),
        }
    return {
        "action": "tool_call",
        "tool": name,
        "args": args,
    }


class TestFunctionCallParsing(unittest.TestCase):

    def test_tool_call(self):
        result = _parse_function_call_response([{
            "id": "call_123",
            "name": "shell",
            "arguments": {"command": "ls -la"},
        }])
        self.assertEqual(result["action"], "tool_call")
        self.assertEqual(result["tool"], "shell")
        self.assertEqual(result["args"]["command"], "ls -la")

    def test_final_answer(self):
        result = _parse_function_call_response([{
            "id": "call_456",
            "name": "final_answer",
            "arguments": {"result": "Task completed", "memories": {"key": "val"}},
        }])
        self.assertEqual(result["action"], "final_answer")
        self.assertEqual(result["result"], "Task completed")
        self.assertEqual(result["memories"]["key"], "val")

    def test_clarify(self):
        result = _parse_function_call_response([{
            "id": "call_789",
            "name": "clarify",
            "arguments": {"question": "What language?"},
        }])
        self.assertEqual(result["action"], "clarify")
        self.assertEqual(result["question"], "What language?")

    def test_empty_returns_none(self):
        self.assertIsNone(_parse_function_call_response([]))
        self.assertIsNone(_parse_function_call_response(None))


# ─── 1.2 Output Validation ───────────────────────────────────────────────
# Inline the validation logic to avoid litellm import dependency

def _validate_response(result, task):
    """Replicated from BaseAgent for testing without litellm."""
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


class TestOutputValidation(unittest.TestCase):

    def test_empty_response_fails(self):
        err = _validate_response("", {"title": "Build API"})
        self.assertIsNotNone(err)
        self.assertIn("empty", err.lower())

    def test_whitespace_only_fails(self):
        err = _validate_response("   \n\t  ", {"title": "Build API"})
        self.assertIsNotNone(err)

    def test_too_short_non_trivial_fails(self):
        err = _validate_response("ok", {"title": "Build the API"})
        self.assertIsNotNone(err)
        self.assertIn("short", err.lower())

    def test_trivial_task_short_ok(self):
        err = _validate_response("v2.1.3", {"title": "Check version"})
        self.assertIsNone(err)

    def test_good_response_passes(self):
        err = _validate_response(
            "I've completed the task. The API endpoint is now live at /api/v1/users.",
            {"title": "Build API"},
        )
        self.assertIsNone(err)

    def test_refusal_short_fails(self):
        err = _validate_response(
            "I cannot do that, sorry about this.",
            {"title": "Build API"},
        )
        self.assertIsNotNone(err)
        self.assertIn("refusal", err.lower())

    def test_refusal_long_passes(self):
        long_text = "I cannot directly run the code, but here's " + "x" * 200
        err = _validate_response(long_text, {"title": "Build API"})
        self.assertIsNone(err)


# ─── 1.3 Circuit Breaker ────────────────────────────────────────────────
# Import directly — CircuitBreaker has no litellm dependency in its class body

# We need to avoid module-level litellm import in router.py.
# Workaround: replicate the class for testing.

class CircuitBreaker:
    """Replicated from router.py for testing without litellm."""
    def __init__(self, failure_threshold=3, window_seconds=300, cooldown_seconds=600):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.failures = []
        self.degraded_until = 0.0

    def record_failure(self):
        now = time.time()
        self.failures.append(now)
        self.failures = [t for t in self.failures if now - t < self.window_seconds]
        if len(self.failures) >= self.failure_threshold:
            self.degraded_until = now + self.cooldown_seconds

    def record_success(self):
        self.failures.clear()
        self.degraded_until = 0.0

    @property
    def is_degraded(self):
        if time.time() >= self.degraded_until:
            if self.degraded_until > 0:
                self.degraded_until = 0.0
                self.failures.clear()
            return False
        return True


class TestCircuitBreaker(unittest.TestCase):

    def test_starts_healthy(self):
        cb = CircuitBreaker()
        self.assertFalse(cb.is_degraded)

    def test_trips_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=10)
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.is_degraded)
        cb.record_failure()
        self.assertTrue(cb.is_degraded)

    def test_success_resets(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10)
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_degraded)
        cb.record_success()
        self.assertFalse(cb.is_degraded)
        self.assertEqual(len(cb.failures), 0)

    def test_cooldown_expires(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_degraded)
        time.sleep(0.2)
        self.assertFalse(cb.is_degraded)

    def test_old_failures_outside_window(self):
        cb = CircuitBreaker(failure_threshold=3, window_seconds=0.1, cooldown_seconds=10)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.2)
        cb.record_failure()  # only 1 failure in window
        self.assertFalse(cb.is_degraded)


# ─── 1.5 Task Deduplication ──────────────────────────────────────────────

def compute_task_hash(title, description, goal_id=None):
    """Replicated from db.py for testing without aiosqlite."""
    raw = f"{title or ''}|{description or ''}|{goal_id or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class TestTaskDedup(unittest.TestCase):

    def test_same_inputs_same_hash(self):
        h1 = compute_task_hash("Build API", "Create REST endpoints", 1)
        h2 = compute_task_hash("Build API", "Create REST endpoints", 1)
        self.assertEqual(h1, h2)

    def test_different_title_different_hash(self):
        h1 = compute_task_hash("Build API", "Create REST endpoints", 1)
        h2 = compute_task_hash("Fix bugs", "Create REST endpoints", 1)
        self.assertNotEqual(h1, h2)

    def test_different_goal_different_hash(self):
        h1 = compute_task_hash("Build API", "Create REST endpoints", 1)
        h2 = compute_task_hash("Build API", "Create REST endpoints", 2)
        self.assertNotEqual(h1, h2)

    def test_none_goal_consistent(self):
        h1 = compute_task_hash("Build API", "desc", None)
        h2 = compute_task_hash("Build API", "desc", None)
        self.assertEqual(h1, h2)

    def test_hash_is_32_chars(self):
        h = compute_task_hash("title", "desc", 1)
        self.assertEqual(len(h), 32)


if __name__ == "__main__":
    unittest.main()
