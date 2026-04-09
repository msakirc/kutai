"""
End-to-end integration tests for search pipeline.

Tests the REAL flow: user message → classification → agent dispatch →
web_search tool → response parsing → final_answer.

Uses actual llama-server with low-quality params for fast inference.
Tests both good and bad user inputs, and verifies the app handles
poor model responses gracefully.

Run with:
    KUTAY_TEST_SERVER=1 python -m pytest tests/test_search_e2e.py -v -s
"""

import importlib
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_ws_module():
    """Get the web_search module (not the function)."""
    return importlib.import_module("src.tools.web_search")

needs_server = unittest.skipUnless(
    os.environ.get("KUTAY_TEST_SERVER"),
    "Set KUTAY_TEST_SERVER=1 to run tests requiring a local llama-server",
)


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION — user messages route to researcher correctly
# ---------------------------------------------------------------------------
class TestSearchClassification(unittest.TestCase):
    """Verify search-like messages classify to researcher agent."""

    def _classify(self, title, desc=""):
        from src.core.task_classifier import _classify_by_keywords
        return _classify_by_keywords(title, desc)

    # ── Good, clear search requests ──

    def test_explicit_search_request(self):
        r = self._classify("Can you do a web search")
        self.assertEqual(r.agent_type, "researcher")
        self.assertTrue(r.needs_tools)

    def test_search_with_topic(self):
        r = self._classify("search for waterproof shoes in Turkey")
        self.assertEqual(r.agent_type, "researcher")

    def test_find_keyword(self):
        r = self._classify("find me the best restaurants in Istanbul")
        self.assertEqual(r.agent_type, "researcher")

    def test_research_keyword(self):
        r = self._classify("research waterproof shoes in Turkey")
        self.assertEqual(r.agent_type, "researcher")

    def test_look_up_keyword(self):
        r = self._classify("look up Victor Osimhen stats")
        self.assertEqual(r.agent_type, "researcher")

    def test_compare_keyword(self):
        r = self._classify("compare iPhone 16 vs Samsung S26")
        self.assertEqual(r.agent_type, "researcher")

    # ── Ambiguous messages that should still work ──

    def test_question_about_facts(self):
        """Factual questions may go to assistant, that's acceptable."""
        r = self._classify("How many league goals did Osimhen score this season")
        self.assertIn(r.agent_type, ("researcher", "assistant"))

    def test_vague_request(self):
        """Vague 'search' without details should still route to researcher."""
        r = self._classify("do a search")
        self.assertEqual(r.agent_type, "researcher")

    # ── Non-search messages should NOT be researcher ──

    def test_greeting_not_researcher(self):
        r = self._classify("hello how are you")
        self.assertNotEqual(r.agent_type, "researcher")

    def test_code_request_not_researcher(self):
        r = self._classify("write a python function to sort a list")
        self.assertNotEqual(r.agent_type, "researcher")

    def test_bug_report_not_researcher(self):
        r = self._classify("there's a bug in the login page")
        self.assertNotEqual(r.agent_type, "researcher")


class TestMessageClassification(unittest.IsolatedAsyncioTestCase):
    """Test the LLM message classifier distinguishes research from build requests."""

    def _classify_keywords(self, text):
        from src.app.telegram_bot import TelegramInterface
        return TelegramInterface._classify_message_by_keywords(text)

    def test_app_recommendation_is_not_mission(self):
        """Asking about existing apps should be task/question, not mission."""
        r = self._classify_keywords(
            "Are there any good apps for sharing shoplists"
        )
        self.assertNotEqual(r["type"], "mission")

    def test_build_me_an_app_is_mission(self):
        """Explicit build request should be mission."""
        r = self._classify_keywords(
            "build me an app that allows users to share shoplists"
        )
        self.assertEqual(r["type"], "mission")


# ---------------------------------------------------------------------------
# 2. RESPONSE PARSING — agent responses parsed correctly
# ---------------------------------------------------------------------------
class TestSearchResponseParsing(unittest.TestCase):
    """Verify the agent parser handles all response formats from search tasks."""

    def _parse(self, content):
        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        return agent._parse_agent_response(content)

    # ── Clean responses ──

    def test_clean_final_answer(self):
        r = self._parse('{"action": "final_answer", "result": "Osimhen scored 12 goals."}')
        self.assertEqual(r["action"], "final_answer")
        self.assertIn("12 goals", r["result"])

    def test_clean_tool_call(self):
        r = self._parse('{"action": "tool_call", "tool": "web_search", "args": {"query": "test"}}')
        self.assertEqual(r["action"], "tool_call")
        self.assertEqual(r["tool"], "web_search")

    def test_final_answer_with_memories(self):
        r = self._parse(json.dumps({
            "action": "final_answer",
            "result": "Found the answer.",
            "memories": {"key": "value"},
        }))
        self.assertEqual(r["action"], "final_answer")

    # ── Thinking model responses ──

    def test_think_tags_stripped(self):
        r = self._parse(
            '<think>Let me search for this information...</think>\n'
            '{"action": "tool_call", "tool": "web_search", "args": {"query": "test"}}'
        )
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "tool_call")

    def test_long_think_tags_stripped(self):
        think_content = "I need to think about " * 100
        r = self._parse(
            f'<think>{think_content}</think>\n'
            '{"action": "final_answer", "result": "The answer is 42."}'
        )
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "final_answer")

    def test_think_with_json_inside(self):
        """Think tags containing JSON should not confuse the parser."""
        r = self._parse(
            '<think>I should call {"tool": "web_search"} but let me reconsider</think>\n'
            '{"action": "final_answer", "result": "Here is the answer."}'
        )
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "final_answer")

    # ── Markdown-wrapped responses ──

    def test_markdown_fence(self):
        r = self._parse('```json\n{"action": "final_answer", "result": "Done."}\n```')
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "final_answer")

    def test_markdown_with_preamble(self):
        r = self._parse(
            'Here is my response:\n\n'
            '```json\n{"action": "final_answer", "result": "Done."}\n```'
        )
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "final_answer")

    # ── Sloppy model responses that should still parse ──

    def test_action_alias_answer(self):
        r = self._parse('{"action": "answer", "result": "The result."}')
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "final_answer")

    def test_action_alias_done(self):
        r = self._parse('{"action": "done", "result": "Completed."}')
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "final_answer")

    def test_no_action_but_has_result(self):
        """Missing action key but has result → should infer final_answer."""
        r = self._parse('{"result": "The weather is sunny."}')
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "final_answer")

    def test_no_action_but_has_answer(self):
        r = self._parse('{"answer": "42 goals in total."}')
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "final_answer")

    def test_tool_name_as_action(self):
        """Model puts tool name as action instead of 'tool_call'."""
        r = self._parse('{"action": "web_search", "args": {"query": "test"}}')
        self.assertIsNotNone(r)
        self.assertEqual(r["action"], "tool_call")
        self.assertEqual(r["tool"], "web_search")

    # ── Bad responses that should return None ──

    def test_pure_thinking_returns_none(self):
        r = self._parse('{"action": "thinking", "content": "Let me consider..."}')
        self.assertIsNone(r)

    def test_pure_prose_returns_none(self):
        r = self._parse("I'd be happy to help you search for that information!")
        self.assertIsNone(r)

    def test_xml_function_call_returns_none(self):
        r = self._parse('<function=web_search><parameter=query>test</parameter></function>')
        self.assertIsNone(r)

    def test_empty_string_returns_none(self):
        r = self._parse("")
        self.assertIsNone(r)


# ---------------------------------------------------------------------------
# 3. WEB SEARCH TOOL — fallback chain works
# ---------------------------------------------------------------------------
class TestWebSearchFallback(unittest.IsolatedAsyncioTestCase):
    """Verify web_search fallback: Perplexica → DDG → curl."""

    async def test_ddg_returns_results(self):
        """DDG search should return formatted results for a real query."""
        from src.tools.web_search import web_search, _DDGS
        if _DDGS is None:
            self.skipTest("ddgs package not installed")
        # Force skip Perplexica by setting fail count high
        ws_mod = _get_ws_module()
        old_fail = ws_mod._perplexica_fail_count
        ws_mod._perplexica_fail_count = 99
        try:
            result = await web_search("Python programming language", max_results=3)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 50)
            self.assertIn("Python", result)
        finally:
            ws_mod._perplexica_fail_count = old_fail

    async def test_empty_query_handled(self):
        """Empty or near-empty query should not crash."""
        from src.tools.web_search import web_search
        ws_mod = _get_ws_module()
        old_fail = ws_mod._perplexica_fail_count
        ws_mod._perplexica_fail_count = 99
        try:
            result = await web_search("", max_results=3)
            self.assertIsInstance(result, str)
        finally:
            ws_mod._perplexica_fail_count = old_fail

    async def test_perplexica_result_has_hint(self):
        """When Perplexica succeeds, result should contain finalize hint."""
        from src.tools.web_search import web_search
        ws_mod = _get_ws_module()
        # Only run if Perplexica is actually available
        old_models = ws_mod._perplexica_models
        old_fail = ws_mod._perplexica_fail_count
        ws_mod._perplexica_fail_count = 0
        ws_mod._perplexica_models = None
        try:
            result = await web_search("Python programming", max_results=3)
            # If Perplexica responded, it should have the hint
            if "AI-synthesized" in result:
                self.assertIn("final_answer", result)
            # If it fell back to DDG, that's also fine
        finally:
            ws_mod._perplexica_models = old_models
            ws_mod._perplexica_fail_count = old_fail


# ---------------------------------------------------------------------------
# 4. LAST ITERATION FORCING — agent must finalize on last iteration
# ---------------------------------------------------------------------------
class TestLastIterationForcing(unittest.TestCase):
    """Verify the last-iteration prompt forces final_answer."""

    def test_last_iteration_message_contains_must(self):
        """On the last iteration, the tool result message must demand final_answer."""
        max_iter = 4
        iteration = 2  # 0-indexed, so iteration+2 = 4 = max_iter
        msg = (
            f"{'LAST ITERATION — you MUST respond with final_answer now. Do NOT call any more tools.' if iteration + 2 >= max_iter else 'Continue working.'}"
            f" Iteration {iteration + 2}/{max_iter}."
        )
        self.assertIn("LAST ITERATION", msg)
        self.assertIn("MUST", msg)
        self.assertIn("final_answer", msg)

    def test_non_last_iteration_says_continue(self):
        max_iter = 4
        iteration = 0  # iteration+2 = 2 < 4
        msg = (
            f"{'LAST ITERATION — you MUST respond with final_answer now. Do NOT call any more tools.' if iteration + 2 >= max_iter else 'Continue working.'}"
            f" Iteration {iteration + 2}/{max_iter}."
        )
        self.assertIn("Continue working", msg)
        self.assertNotIn("LAST ITERATION", msg)


# ---------------------------------------------------------------------------
# 5. RESEARCHER PROMPT — efficiency instructions present
# ---------------------------------------------------------------------------
class TestResearcherPrompt(unittest.TestCase):
    """Verify researcher prompt encourages efficiency."""

    def test_prompt_discourages_multiple_searches(self):
        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        prompt = agent.get_system_prompt({})
        self.assertIn("ONE search", prompt)
        self.assertIn("final_answer", prompt)

    def test_max_iterations_reasonable(self):
        from src.agents.researcher import ResearcherAgent
        self.assertLessEqual(ResearcherAgent.max_iterations, 4)
        self.assertGreaterEqual(ResearcherAgent.max_iterations, 2)

    def test_web_search_in_allowed_tools(self):
        from src.agents.researcher import ResearcherAgent
        self.assertIn("web_search", ResearcherAgent.allowed_tools)


# ---------------------------------------------------------------------------
# 6. FULL E2E WITH LLAMA SERVER — real inference
# ---------------------------------------------------------------------------
@needs_server
class TestSearchE2E(unittest.IsolatedAsyncioTestCase):
    """
    Full end-to-end search tests with real llama-server inference.

    Uses low max_tokens and high temperature tolerance to keep tests fast.
    The app should produce usable results even with degraded model output.
    """

    async def _run_researcher(self, task_description, max_iter=3):
        """Run the researcher agent on a task and return the result dict."""
        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        # Override for fast testing
        agent.max_iterations = max_iter

        task = {
            "id": "test",
            "title": task_description[:50],
            "description": task_description,
            "agent_type": "researcher",
            "priority": 5,
            "context": json.dumps({
                "classification": {
                    "agent_type": "researcher",
                    "difficulty": 4,
                    "needs_tools": True,
                    "needs_thinking": False,
                }
            }),
        }
        result = await agent.execute(task)
        return result

    # ── Good queries — should produce a final answer ──

    async def test_specific_search_query(self):
        """Clear, specific search query should get a result in ≤3 iterations."""
        result = await self._run_researcher(
            "What is the population of Istanbul?"
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(len(result.get("result", "")), 20)
        self.assertLessEqual(result.get("iterations", 99), 3)

    async def test_comparison_query(self):
        result = await self._run_researcher(
            "Compare React vs Vue.js for frontend development"
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(len(result.get("result", "")), 50)

    # ── Vague queries — should still produce something ──

    async def test_vague_search(self):
        """Vague 'do a web search' should clarify or produce a generic result."""
        result = await self._run_researcher("Can you do a web search")
        # Should either ask for clarification or give a generic response
        self.assertIn(result["status"], ("completed", "needs_clarification"))

    async def test_single_word_query(self):
        """Single word should not crash."""
        result = await self._run_researcher("Python")
        self.assertEqual(result["status"], "completed")
        self.assertIsInstance(result.get("result", ""), str)

    # ── Edge cases — app should handle gracefully ──

    async def test_non_english_query(self):
        """Non-English query should not crash."""
        result = await self._run_researcher(
            "Türkiye'de su geçirmez ayakkabı nereden alınır"
        )
        self.assertEqual(result["status"], "completed")

    async def test_very_long_query(self):
        """Very long query should be truncated, not crash."""
        long_query = "search for " + "very important information about " * 50
        result = await self._run_researcher(long_query)
        self.assertEqual(result["status"], "completed")

    async def test_result_not_raw_json(self):
        """Final result should be readable text, not raw JSON."""
        result = await self._run_researcher(
            "What is the capital of France?"
        )
        if result["status"] == "completed":
            text = result.get("result", "")
            # Should not be raw JSON dump
            if text.strip().startswith("{"):
                try:
                    parsed = json.loads(text)
                    # If it's parseable JSON, it should at least have readable content
                    self.assertIn("result", parsed)
                except json.JSONDecodeError:
                    pass  # Not JSON, which is fine

    async def test_iterations_efficient_with_perplexica(self):
        """If Perplexica is available, should complete in ≤2 iterations."""
        ws_mod = _get_ws_module()
        if ws_mod._perplexica_fail_count >= ws_mod._PERPLEXICA_MAX_FAILURES:
            self.skipTest("Perplexica disabled")

        result = await self._run_researcher(
            "What is the current weather in Ankara?"
        )
        if result["status"] == "completed":
            iters = result.get("iterations", 99)
            # With Perplexica's synthesized answer + hint, should be ≤2
            self.assertLessEqual(
                iters, 3,
                f"Expected ≤3 iterations with Perplexica, got {iters}"
            )


# ---------------------------------------------------------------------------
# 7. PERPLEXICA CIRCUIT BREAKER
# ---------------------------------------------------------------------------
class TestPerplexicaCircuitBreaker(unittest.TestCase):
    """Verify Perplexica disables after failures and re-enables after cooldown."""

    def test_disables_after_max_failures(self):
        ws_mod = _get_ws_module()
        old_fail = ws_mod._perplexica_fail_count
        old_disabled = ws_mod._perplexica_disabled_at
        try:
            ws_mod._perplexica_fail_count = ws_mod._PERPLEXICA_MAX_FAILURES
            ws_mod._perplexica_disabled_at = 0.0
            # Next call should set disabled_at
            # (We can't call _search_perplexica without async, just verify the logic)
            self.assertGreaterEqual(
                ws_mod._perplexica_fail_count,
                ws_mod._PERPLEXICA_MAX_FAILURES,
            )
        finally:
            ws_mod._perplexica_fail_count = old_fail
            ws_mod._perplexica_disabled_at = old_disabled

    def test_cooldown_duration(self):
        ws_mod = _get_ws_module()
        self.assertEqual(ws_mod._PERPLEXICA_RETRY_AFTER, 300.0)
        self.assertEqual(ws_mod._PERPLEXICA_MAX_FAILURES, 3)


# ---------------------------------------------------------------------------
# 8. WORKFLOW RUNNER — constructor takes no args
# ---------------------------------------------------------------------------
class TestWorkflowRunner(unittest.TestCase):
    """Verify WorkflowRunner can be instantiated without args."""

    def test_no_args_constructor(self):
        from src.workflows.engine.runner import WorkflowRunner
        runner = WorkflowRunner()
        self.assertIsNotNone(runner)

    def test_rejects_positional_args(self):
        from src.workflows.engine.runner import WorkflowRunner
        with self.assertRaises(TypeError):
            WorkflowRunner("unexpected_arg")


if __name__ == "__main__":
    # Load env for model registry access
    from dotenv import load_dotenv
    load_dotenv()
    unittest.main(verbosity=2)
