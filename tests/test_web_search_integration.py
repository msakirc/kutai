# tests/test_web_search_integration.py
"""
Integration tests for web search functionality.

Covers:
  - DuckDuckGo (ddgs) backend
  - Perplexica/Vane backend (model discovery + search)
  - Fallback logic (Perplexica -> ddgs)
  - Researcher agent flow with web_search tool calls
"""

import asyncio
import importlib
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the web_search *module* directly.  src.tools.__init__ shadows the
# module name with the ``web_search`` function, so a plain
# ``import src.tools.web_search`` resolves to the function object.
# importlib.import_module always returns the real module.
_ws_mod = importlib.import_module("src.tools.web_search")


def run_async(coro):
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Realistic provider data for Perplexica mocks
# ---------------------------------------------------------------------------

MOCK_PROVIDERS_DATA = {
    "providers": [
        {
            "id": "groq",
            "chatModels": [
                {"key": "llama-3.3-70b-versatile"},
                {"key": "groq/compound"},
                {"key": "groq/compound-mini"},
                {"key": "whisper-large-v3"},  # should be skipped
            ],
            "embeddingModels": [],
        },
        {
            "id": "openai",
            "chatModels": [
                {"key": "gpt-4o"},
            ],
            "embeddingModels": [
                {"key": "text-embedding-3-small"},
            ],
        },
    ]
}

# Minimal providers: only one chat model, no groq/compound
MOCK_PROVIDERS_NO_GROQ = {
    "providers": [
        {
            "id": "openai",
            "chatModels": [
                {"key": "gpt-4o"},
            ],
            "embeddingModels": [
                {"key": "text-embedding-3-small"},
            ],
        },
    ]
}


# ---------------------------------------------------------------------------
# Helper: build a mock aiohttp session
# ---------------------------------------------------------------------------

def _make_aiohttp_response(status=200, json_data=None, text_data=""):
    """Create a mock aiohttp response usable as an async context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text_data)
    return resp


def _make_aiohttp_session(responses):
    """
    Build a mock aiohttp.ClientSession that yields *responses* in order.

    Each response is returned as the async context manager result of
    session.get() or session.post().
    """
    call_idx = {"i": 0}

    class _FakeCtx:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *a):
            pass

    class _FakeSession:
        def get(self, *a, **kw):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return _FakeCtx(responses[idx])

        def post(self, *a, **kw):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return _FakeCtx(responses[idx])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    return _FakeSession


def _reset_ws_state():
    """Reset web_search module global state for test isolation."""
    _ws_mod._perplexica_models = None
    _ws_mod._perplexica_fail_count = 0


# ===========================================================================
# 1. TestDDGSBackend
# ===========================================================================

class TestDDGSBackend(unittest.TestCase):
    """Tests for the DuckDuckGo ddgs backend."""

    def test_ddgs_import_available(self):
        """Verify `from ddgs import DDGS` works and _DDGS is set."""
        self.assertIsNotNone(_ws_mod._DDGS, "_DDGS should be set when ddgs is installed")
        # _DDGS may be the DDGS class or a proxy wrapper; verify it's callable
        self.assertTrue(callable(_ws_mod._DDGS), "_DDGS should be callable")

    # -- Network test (may be skipped in CI) --
    def test_ddgs_text_search_returns_results(self):
        """Actual network call: search 'Python tutorial', verify URLs in output."""
        result = run_async(_ws_mod.web_search("Python tutorial", max_results=3))
        self.assertIsInstance(result, str)
        self.assertIn("http", result, "Result should contain at least one URL")

    def test_ddgs_empty_results_handled(self):
        """Search for nonsensical query; verify returns a string, not a crash."""
        result = run_async(_ws_mod.web_search("xyzzy9999qqq_no_results_ever_12345"))
        self.assertIsInstance(result, str)


# ===========================================================================
# 2. TestPerplexicaBackend
# ===========================================================================

class TestPerplexicaBackend(unittest.TestCase):
    """Tests for the Perplexica/Vane backend (model discovery + search)."""

    def setUp(self):
        # Save and reset global state
        self._orig_models = _ws_mod._perplexica_models
        self._orig_fail_count = _ws_mod._perplexica_fail_count
        _reset_ws_state()

    def tearDown(self):
        # Restore global state
        _ws_mod._perplexica_models = self._orig_models
        _ws_mod._perplexica_fail_count = self._orig_fail_count

    def test_model_discovery(self):
        """Mock GET /api/providers, verify _discover_perplexica_models returns models."""
        resp = _make_aiohttp_response(status=200, json_data=MOCK_PROVIDERS_DATA)
        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session([resp])()):
            result = run_async(_ws_mod._discover_perplexica_models("http://localhost:3000"))

        self.assertIsNotNone(result)
        self.assertIn("chatModel", result)
        self.assertIn("embeddingModel", result)
        self.assertEqual(result["embeddingModel"]["key"], "text-embedding-3-small")

    def test_model_discovery_prefers_groq_compound(self):
        """When providers list has groq/compound, it should be preferred."""
        resp = _make_aiohttp_response(status=200, json_data=MOCK_PROVIDERS_DATA)
        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session([resp])()):
            result = run_async(_ws_mod._discover_perplexica_models("http://localhost:3000"))

        self.assertIsNotNone(result)
        self.assertEqual(result["chatModel"]["key"], "groq/compound")
        self.assertEqual(result["chatModel"]["providerId"], "groq")

    def test_model_discovery_fallback_without_groq(self):
        """Without groq/compound, the first available chat model is used."""
        resp = _make_aiohttp_response(status=200, json_data=MOCK_PROVIDERS_NO_GROQ)
        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session([resp])()):
            result = run_async(_ws_mod._discover_perplexica_models("http://localhost:3000"))

        self.assertIsNotNone(result)
        self.assertEqual(result["chatModel"]["key"], "gpt-4o")

    @patch.dict(os.environ, {"PERPLEXICA_URL": "http://localhost:3000"})
    def test_search_success(self):
        """Mock POST /api/search returning {message, sources}, verify formatted result."""
        # First call: model discovery GET /api/providers
        provider_resp = _make_aiohttp_response(status=200, json_data=MOCK_PROVIDERS_DATA)
        # Second call: POST /api/search
        search_resp = _make_aiohttp_response(status=200, json_data={
            "message": "Waterproof shoes are great for rainy weather.",
            "sources": [
                {
                    "content": "A guide to waterproof footwear...",
                    "metadata": {
                        "title": "Waterproof Shoe Guide",
                        "url": "https://example.com/shoes",
                    },
                },
            ],
        })
        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session([provider_resp, search_resp])()):
            result = run_async(_ws_mod._search_perplexica("waterproof shoes", 5, "web"))

        self.assertIsNotNone(result)
        self.assertEqual(result["answer"], "Waterproof shoes are great for rainy weather.")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["url"], "https://example.com/shoes")
        # Success should reset fail count
        self.assertEqual(_ws_mod._perplexica_fail_count, 0)

    @patch.dict(os.environ, {"PERPLEXICA_URL": "http://localhost:3000"})
    def test_search_failure_increments_counter(self):
        """Mock 500 response, verify _perplexica_fail_count increases."""
        # Model discovery succeeds
        provider_resp = _make_aiohttp_response(status=200, json_data=MOCK_PROVIDERS_DATA)
        # Search returns 500
        search_resp = _make_aiohttp_response(status=500, text_data="Internal Server Error")

        self.assertEqual(_ws_mod._perplexica_fail_count, 0)

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_session([provider_resp, search_resp])()):
            result = run_async(_ws_mod._search_perplexica("test query", 5, "web"))

        self.assertIsNone(result)
        self.assertEqual(_ws_mod._perplexica_fail_count, 1)

    @patch.dict(os.environ, {"PERPLEXICA_URL": "http://localhost:3000"})
    def test_disabled_after_max_failures(self):
        """Set fail count to max, verify returns None without making a request."""
        _ws_mod._perplexica_fail_count = 3

        # No aiohttp mock needed -- it should short-circuit before making a request
        result = run_async(_ws_mod._search_perplexica("test query", 5, "web"))

        self.assertIsNone(result)
        # Fail count should remain unchanged (no new request was made)
        self.assertEqual(_ws_mod._perplexica_fail_count, 3)


# ===========================================================================
# 3. TestWebSearchFallback
# ===========================================================================

class TestWebSearchFallback(unittest.TestCase):
    """Tests for the Perplexica -> ddgs fallback logic."""

    def setUp(self):
        self._orig_models = _ws_mod._perplexica_models
        self._orig_fail_count = _ws_mod._perplexica_fail_count
        _reset_ws_state()

    def tearDown(self):
        _ws_mod._perplexica_models = self._orig_models
        _ws_mod._perplexica_fail_count = self._orig_fail_count

    def test_perplexica_down_falls_to_ddgs(self):
        """When Perplexica returns None, ddgs should be used."""
        mock_perp = AsyncMock(return_value=None)
        with patch.object(_ws_mod, "_search_perplexica", mock_perp):
            result = run_async(_ws_mod.web_search("Python tutorial", max_results=3))

        self.assertIsInstance(result, str)
        # ddgs backend formats results as "Search results for '...'"
        self.assertIn("Search results for", result)
        mock_perp.assert_awaited_once()

    # -- Network test (may be skipped in CI) --
    def test_full_pipeline(self):
        """Actual call to web_search('waterproof shoes Turkey'), verify returns content."""
        result = run_async(_ws_mod.web_search("waterproof shoes Turkey", max_results=3))

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 20, "Result should contain meaningful content")


# ===========================================================================
# 4. TestResearcherAgentFlow (mock-heavy)
# ===========================================================================

class TestResearcherAgentFlow(unittest.TestCase):
    """Test that the researcher agent invokes web_search via its tool loop."""

    @patch("src.agents.base.format_blackboard_for_prompt", return_value="")
    @patch("src.agents.base.get_or_create_blackboard", new_callable=AsyncMock, return_value={})
    @patch("src.agents.base.record_cost", new_callable=AsyncMock)
    @patch("src.agents.base.update_task", new_callable=AsyncMock)
    @patch("src.agents.base.record_model_call", new_callable=AsyncMock)
    @patch("src.agents.base.clear_task_checkpoint", new_callable=AsyncMock)
    @patch("src.agents.base.load_task_checkpoint", new_callable=AsyncMock, return_value=None)
    @patch("src.agents.base.save_task_checkpoint", new_callable=AsyncMock)
    @patch("src.agents.base.get_completed_dependency_results", new_callable=AsyncMock, return_value=[])
    @patch("src.agents.base.recall_memory", new_callable=AsyncMock, return_value=[])
    @patch("src.agents.base.store_memory", new_callable=AsyncMock)
    @patch("src.agents.base.log_conversation", new_callable=AsyncMock)
    @patch("src.agents.base.format_project_profile", return_value="")
    @patch("src.agents.base.get_project_profile_for_task", new_callable=AsyncMock, return_value=None)
    @patch("src.agents.base.format_preferences", return_value="")
    @patch("src.agents.base.get_user_preferences", new_callable=AsyncMock, return_value={})
    @patch("src.agents.base.retrieve_context", new_callable=AsyncMock, return_value=[])
    @patch("src.agents.base.get_registry")
    @patch("src.agents.base.select_model")
    @patch("src.agents.base.call_model", new_callable=AsyncMock)
    @patch("src.agents.base.execute_tool", new_callable=AsyncMock)
    @patch("src.agents.base.grade_response", new_callable=AsyncMock, return_value=5)
    def test_researcher_calls_web_search(
        self,
        mock_grade,
        mock_execute_tool,
        mock_call_model, mock_select, mock_registry,
        mock_rag,
        mock_pref, mock_pref_fmt,
        mock_proj_profile, mock_proj_fmt,
        mock_log_conv, mock_store_mem, mock_recall,
        mock_dep_results,
        mock_save_ckpt, mock_load_ckpt, mock_clear_ckpt,
        mock_record_model, mock_update_task, mock_record_cost,
        mock_bb_get, mock_bb_fmt,
    ):
        """
        Mock the LLM to return a JSON tool_call for web_search, then
        a final_answer.  Verify the agent processes the tool call.
        """
        # Set up model registry mock
        mock_reg = MagicMock()
        mock_reg.get_schemas.return_value = []
        mock_registry.return_value = mock_reg

        # select_model returns a model identifier
        mock_select.return_value = "groq/llama-3.3-70b-versatile"

        # First LLM call: tool call for web_search
        tool_call_response = json.dumps({
            "action": "tool_call",
            "tool": "web_search",
            "args": {"query": "waterproof shoes Turkey", "max_results": 5},
        })
        # Second LLM call: final_answer
        final_response = json.dumps({
            "action": "final_answer",
            "result": "## Research: Waterproof Shoes\n\nFound several options.",
        })

        def _make_model_response(content_str):
            return {
                "content": content_str,
                "model": "test-model",
                "model_name": "test",
                "cost": 0.001,
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                "tool_calls": None,
                "latency": 0.1,
                "thinking": "",
                "is_local": False,
                "ran_on": "test",
                "provider": "test",
                "task": "test",
                "capability_score": 5.0,
                "difficulty": 3,
            }

        # Provide enough responses for multiple iterations (the agent may
        # call the model more than twice due to context rebuilds, retries, etc.)
        mock_call_model.side_effect = [
            _make_model_response(tool_call_response),
            _make_model_response(final_response),
            _make_model_response(final_response),
            _make_model_response(final_response),
        ]

        # execute_tool returns mocked search results when called with web_search
        mock_execute_tool.return_value = (
            "Search results for 'waterproof shoes Turkey':\n\n"
            "1. **Best Waterproof Shoes**\n   Great options...\n   https://example.com/shoes"
        )

        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()

        task = {
            "id": "test-task-001",
            "description": "Research waterproof shoes available in Turkey",
            "mission_id": "test-mission",
            "status": "in_progress",
            "agent": "researcher",
            "priority": 5,
            "dependencies": [],
        }

        result = run_async(agent.execute(task))

        # Verify execute_tool was called with web_search
        mock_execute_tool.assert_awaited()
        # Find the call that used web_search
        ws_calls = [
            c for c in mock_execute_tool.call_args_list
            if c[0][0] == "web_search" or c[1].get("tool_name") == "web_search"
        ]
        self.assertTrue(len(ws_calls) > 0, "execute_tool should have been called with 'web_search'")

        # Verify the agent produced a final result
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
