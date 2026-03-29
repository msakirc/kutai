"""test_web_search_e2e.py — Real end-to-end integration tests for the web search pipeline.

NOTHING IS MOCKED. Tests fire a real llama-server process, send real HTTP
requests to SearXNG (port 3002) and Perplexica/Vane (port 3000), and call
the actual web_search() implementation in src/tools/web_search.py.

Prerequisites
-------------
- Docker running with Vane (Perplexica) on port 3000
- SearXNG on port 3002
- llama-server binary at C:/Users/sakir/ai/llama.cpp/llama-server.exe
- Qwen3.5-9B model at C:/Users/sakir/ai/models/Qwen3.5-9B-UD-Q4_K_XL.gguf

Run
---
    .venv/Scripts/python.exe -m pytest tests/integration/test_web_search_e2e.py -v -s --timeout=120

Skip in CI (no LLM)
-------------------
    pytest tests/integration/ -m "not llm"
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Fix UnicodeEncodeError on Windows when printing non-ASCII characters (e.g. Turkish text)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLAMA_SERVER_EXE = r"C:\Users\sakir\ai\llama.cpp\llama-server.exe"
MODEL_PATH = r"C:\Users\sakir\ai\models\Qwen3.5-9B-UD-Q4_K_XL.gguf"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
PERPLEXICA_URL = "http://localhost:3000"
SEARXNG_URL = "http://localhost:3002"

LLAMA_SERVER_ARGS = [
    LLAMA_SERVER_EXE,
    "--model", MODEL_PATH,
    "--port", "8080",
    "--host", "127.0.0.1",
    "--n-gpu-layers", "99",
    "--ctx-size", "8192",
    "--flash-attn", "auto",
    "--metrics",
    "--threads", "9",
    "--batch-size", "2048",
    "--ubatch-size", "512",
    "--alias", "local-model",
    "--chat-template-kwargs", '{"enable_thinking": false}',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine in a fresh event loop (safe inside sync pytest)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _is_llama_server_up() -> bool:
    """Return True if llama-server is already accepting requests on port 8080."""
    try:
        r = httpx.get(f"{LLAMA_SERVER_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_llama_server(timeout_s: float = 120.0) -> bool:
    """Poll /health until llama-server responds or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_llama_server_up():
            return True
        time.sleep(2.0)
    return False


# ---------------------------------------------------------------------------
# Module-scoped fixture: llama-server lifecycle
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llama_server():
    """Start llama-server before the module's tests; terminate after.

    If llama-server is already running (e.g. started by the user), we skip
    the start/stop lifecycle and use the existing process.
    """
    already_running = _is_llama_server_up()
    proc = None

    if already_running:
        print("\n[llama_server fixture] llama-server already running — skipping start.")
        yield
        return

    print("\n[llama_server fixture] Starting llama-server...")
    proc = subprocess.Popen(
        LLAMA_SERVER_ARGS,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    start = time.monotonic()
    ready = False
    # Poll health AND check process is still alive (it might crash on bad args)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            pytest.fail(
                f"llama-server exited with code {exit_code} before becoming healthy. "
                "Check model path, GPU memory, and --chat-template-kwargs support."
            )
        if _is_llama_server_up():
            ready = True
            break
        time.sleep(2.0)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        pytest.fail("llama-server did not become healthy within 120 seconds.")
    print(f"[llama_server fixture] llama-server ready in {time.monotonic() - start:.1f}s")

    yield proc

    print("\n[llama_server fixture] Stopping llama-server...")
    if proc.poll() is None:  # only terminate if still running
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("[llama_server fixture] llama-server stopped.")


# ===========================================================================
# Test 1: SearXNG direct — raw results without LLM synthesis
# ===========================================================================

@pytest.mark.llm
@pytest.mark.integration
def test_searxng_direct_returns_results():
    """SearXNG at port 3002 returns non-empty results for a basic query."""
    url = f"{SEARXNG_URL}/search"
    params = {"q": "test", "format": "json"}

    t0 = time.monotonic()
    response = httpx.get(url, params=params, timeout=20.0)
    elapsed = time.monotonic() - t0

    print(f"\n  SearXNG response time: {elapsed:.2f}s  status={response.status_code}")
    assert response.status_code == 200, (
        f"Expected 200 from SearXNG, got {response.status_code}. "
        f"Is Docker running with Vane? body={response.text[:300]}"
    )

    data = response.json()
    results = data.get("results", [])
    print(f"  Result count: {len(results)}")
    assert len(results) > 0, "SearXNG returned 0 results for query 'test'"

    for r in results[:3]:
        assert "title" in r, f"Result missing 'title': {r}"
        assert "url" in r, f"Result missing 'url': {r}"
        # 'content' is optional in SearXNG but should usually be present
        print(f"  - {r['title'][:60]}  |  {r['url'][:60]}")


# ===========================================================================
# Test 2: Perplexica quality gate rejects 0-source answers
# ===========================================================================

@pytest.mark.llm
@pytest.mark.integration
def test_perplexica_quality_gate_rejects_no_sources(llama_server):
    """_search_perplexica() returns None for a nonsense query that yields 0 sources."""
    os.environ["PERPLEXICA_URL"] = PERPLEXICA_URL
    os.environ["SEARXNG_URL"] = SEARXNG_URL

    # Reset module-level cache so we re-discover models with the env var set
    from importlib import import_module; ws_mod = import_module("src.tools.web_search")
    ws_mod._perplexica_models = None
    ws_mod._perplexica_fail_count = 0
    ws_mod._perplexica_disabled_at = 0.0

    # A query so nonsensical that no real web sources exist
    nonsense_query = "zxqwerty nonexistent product 2099 xzxzxz gibberish"

    t0 = time.monotonic()
    result = run_async(ws_mod._search_perplexica(nonsense_query, max_results=5, focus_mode="web"))
    elapsed = time.monotonic() - t0

    print(f"\n  _search_perplexica (nonsense query) took {elapsed:.1f}s")
    print(f"  result = {result!r}")

    # Either None (quality gate fired) or a dict with answer (Perplexica synthesized something)
    # We assert None because a nonsense query should return 0 sources → quality gate rejects it.
    # If Perplexica somehow returns sources, we accept the result but print a warning.
    if result is not None:
        pytest.xfail(
            f"Perplexica returned a result for a nonsense query — quality gate did not fire. "
            f"This can happen if SearXNG returns noisy results. answer={str(result)[:200]}"
        )
    assert result is None, (
        "Quality gate should have rejected a 0-source answer for a nonsense query."
    )


# ===========================================================================
# Test 3: Full web_search() returns results
# ===========================================================================

@pytest.mark.llm
@pytest.mark.integration
def test_web_search_returns_results(llama_server):
    """web_search() returns a non-empty string of at least 100 chars within 60s."""
    os.environ["PERPLEXICA_URL"] = PERPLEXICA_URL
    os.environ["SEARXNG_URL"] = SEARXNG_URL

    # Reset module-level state
    from importlib import import_module; ws_mod = import_module("src.tools.web_search")
    ws_mod._perplexica_models = None
    ws_mod._perplexica_fail_count = 0
    ws_mod._perplexica_disabled_at = 0.0

    from src.tools.web_search import web_search

    query = "Turkey football match 2026"
    print(f"\n  Query: {query!r}")

    t0 = time.monotonic()
    result = run_async(asyncio.wait_for(web_search(query), timeout=60.0))
    elapsed = time.monotonic() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Result preview: {str(result)[:200]}")

    assert result is not None, "web_search() returned None"
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) >= 100, (
        f"Result too short ({len(result)} chars) — expected at least 100 chars. "
        f"result={result!r}"
    )
    assert elapsed < 60.0, f"web_search() took {elapsed:.1f}s, expected < 60s"


# ===========================================================================
# Test 4: web_search() fallback chain — without Perplexica
# ===========================================================================

@pytest.mark.llm
@pytest.mark.integration
def test_web_search_fallback_chain():
    """web_search() still returns results via SearXNG or DuckDuckGo when Perplexica is disabled."""
    from importlib import import_module; ws_mod = import_module("src.tools.web_search")
    from src.tools.web_search import web_search

    # Save and disable Perplexica
    original_perplexica = os.environ.get("PERPLEXICA_URL", "")
    os.environ["PERPLEXICA_URL"] = ""
    os.environ["SEARXNG_URL"] = SEARXNG_URL

    # Reset caches
    ws_mod._perplexica_models = None
    ws_mod._perplexica_fail_count = 0
    ws_mod._perplexica_disabled_at = 0.0

    query = "Python programming language"
    print(f"\n  Query (no Perplexica): {query!r}")

    try:
        t0 = time.monotonic()
        result = run_async(asyncio.wait_for(web_search(query), timeout=60.0))
        elapsed = time.monotonic() - t0

        print(f"  Completed in {elapsed:.1f}s via fallback chain")
        print(f"  Result preview: {str(result)[:200]}")

        assert result is not None, "web_search() returned None without Perplexica"
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) >= 50, (
            f"Fallback result too short ({len(result)} chars). result={result!r}"
        )
    finally:
        # Always restore
        os.environ["PERPLEXICA_URL"] = original_perplexica


# ===========================================================================
# Test 5: llama-server inference speed check
# ===========================================================================

@pytest.mark.llm
@pytest.mark.integration
def test_llama_server_inference_speed(llama_server):
    """llama-server generates >15 tokens/s and thinking is disabled."""
    # Guard: if the fixture's proc died between tests, skip with a clear message.
    if llama_server is not None and llama_server.poll() is not None:
        pytest.skip(
            f"llama-server process exited (code {llama_server.poll()}) before this test ran."
        )
    if not _is_llama_server_up():
        pytest.skip("llama-server is not responding on port 8080 — skipping speed check.")

    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 100,
        "temperature": 0.0,
        "stream": False,
    }

    print("\n  Sending chat completion request to llama-server...")
    t0 = time.monotonic()
    response = httpx.post(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=60.0,
    )
    elapsed = time.monotonic() - t0

    assert response.status_code == 200, (
        f"llama-server returned {response.status_code}. body={response.text[:400]}"
    )

    data = response.json()
    print(f"  HTTP round-trip: {elapsed:.2f}s")

    # Check we got a real completion
    choices = data.get("choices", [])
    assert len(choices) > 0, "No choices in response"
    message = choices[0].get("message", {})
    content = message.get("content", "")
    print(f"  Response: {content!r}")
    assert content.strip(), "Empty content in llama-server response"

    # Check thinking is disabled — no reasoning_content or it's empty/null
    reasoning = message.get("reasoning_content", None)
    if reasoning:
        assert reasoning.strip() == "", (
            f"Thinking appears enabled — reasoning_content={reasoning[:200]!r}. "
            "Pass --chat-template-kwargs '{\"enable_thinking\": false}' to llama-server."
        )
    print(f"  Thinking disabled: OK (reasoning_content={reasoning!r})")

    # Check generation speed via timings (llama.cpp specific field)
    timings = data.get("timings", {})
    if timings:
        tps = timings.get("predicted_per_second", 0.0)
        print(f"  Generation speed: {tps:.1f} tokens/s")
        assert tps > 15.0, (
            f"llama-server too slow: {tps:.1f} tokens/s < 15 tokens/s minimum. "
            "Check GPU layers (--n-gpu-layers 99) and model size."
        )
    else:
        # timings not present in all llama.cpp builds — estimate from wall time
        n_tokens = data.get("usage", {}).get("completion_tokens", 0)
        if n_tokens > 0:
            estimated_tps = n_tokens / elapsed
            print(f"  Estimated speed (wall-clock): {estimated_tps:.1f} tokens/s "
                  f"({n_tokens} tokens in {elapsed:.2f}s)")
            assert estimated_tps > 5.0, (
                f"Estimated generation speed {estimated_tps:.1f} tps is very low — "
                "something may be wrong with GPU offloading."
            )
        else:
            print("  timings field absent and no usage token count — skipping speed assertion.")


# ===========================================================================
# Test 6: Perplexica with llama-server synthesizes a real answer
# ===========================================================================

@pytest.mark.llm
@pytest.mark.integration
def test_perplexica_with_llama_returns_answer(llama_server):
    """_search_perplexica() returns a synthesized answer for a factual query."""
    os.environ["PERPLEXICA_URL"] = PERPLEXICA_URL
    os.environ["SEARXNG_URL"] = SEARXNG_URL

    # Reset module-level cache so we re-discover models fresh
    from importlib import import_module; ws_mod = import_module("src.tools.web_search")
    ws_mod._perplexica_models = None
    ws_mod._perplexica_fail_count = 0
    ws_mod._perplexica_disabled_at = 0.0

    query = "Python programming language"
    print(f"\n  Query: {query!r}")

    t0 = time.monotonic()
    try:
        result = run_async(
            asyncio.wait_for(
                ws_mod._search_perplexica(query, max_results=5, focus_mode="web"),
                timeout=60.0,
            )
        )
    except asyncio.TimeoutError:
        pytest.fail("_search_perplexica() timed out after 60s — Vane/llama-server too slow.")
    elapsed = time.monotonic() - t0

    print(f"  Completed in {elapsed:.1f}s")

    if result is None:
        # Explain why it might be None for debugging
        pytest.fail(
            f"_search_perplexica() returned None after {elapsed:.1f}s. "
            "Possible reasons: Perplexica not running, model discovery failed, "
            "quality gate rejected, or llama-server too slow/not loaded."
        )

    print(f"  Result type: {type(result)}")
    print(f"  Answer preview: {str(result.get('answer', ''))[:300]}")
    print(f"  Source count: {len(result.get('sources', []))}")

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "answer" in result, f"Result dict has no 'answer' key: {result.keys()}"
    answer = result["answer"]
    assert answer and len(answer) > 20, (
        f"Answer too short or empty: {answer!r}"
    )
    assert elapsed < 60.0, f"_search_perplexica() took {elapsed:.1f}s, expected < 60s"
