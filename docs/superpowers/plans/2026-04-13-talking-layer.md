# HaLLederiz Kadir Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract LLM call execution from `src/core/router.py` into a new `packages/hallederiz_kadir/` package, making router pure scoring and dispatcher the policy/orchestration layer.

**Architecture:** HaLLederiz Kadir is a KutAI package that owns the litellm call pipeline — building completion kwargs, making calls, streaming, retries, response parsing, error classification, quality checks, and metrics. Dispatcher handles candidate iteration and calls HaLLederiz Kadir once per candidate. Router becomes pure scoring with no I/O.

**Tech Stack:** Python 3.10, litellm, asyncio, DaLLaMa, KDV, Dogru mu Samet, Nerd Herd, Yazbunu

**Design spec:** `docs/superpowers/specs/2026-04-13-hallederiz-kadir-design.md`

---

## File Structure

### New files (package)

```
packages/hallederiz_kadir/
  pyproject.toml                              # Package config, deps: litellm
  src/hallederiz_kadir/
    __init__.py                               # Exports: call, CallResult, CallError
    types.py                                  # CallResult, CallError dataclasses (~50 lines)
    caller.py                                 # Main call() function — local/cloud routing,
                                              #   kwargs building, litellm call, streaming (~250 lines)
    response.py                               # Response parsing, think-tag stripping,
                                              #   thinking extraction, cost calc (~80 lines)
    retry.py                                  # Per-model retry loop, error classification,
                                              #   partial content salvage (~100 lines)
  tests/
    test_types.py                             # CallResult/CallError construction
    test_caller.py                            # Main call() tests with mocked litellm
    test_response.py                          # Response parsing, think-tag stripping
    test_retry.py                             # Retry loop, error classification
```

### Modified files

```
src/core/llm_dispatcher.py                   # Absorbs candidate iteration loop, ensure_model,
                                              #   secret redaction, thinking adaptation, fallback
                                              #   relaxation. Calls talker.call() per candidate.
src/core/router.py                           # Remove call_model(), _stream_with_accumulator(),
                                              #   _extract_thinking(), _classify_error_category(),
                                              #   ModelCallFailed, litellm imports, GPU/KDV wiring.
                                              #   call_model() becomes thin shim to dispatcher.
requirements.txt                             # Add: -e ./packages/hallederiz_kadir
```

---

## Task 1: Package Scaffold + Types

**Files:**
- Create: `packages/hallederiz_kadir/pyproject.toml`
- Create: `packages/hallederiz_kadir/src/hallederiz_kadir/__init__.py`
- Create: `packages/hallederiz_kadir/src/hallederiz_kadir/types.py`
- Create: `packages/hallederiz_kadir/tests/test_types.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create pyproject.toml**

```python
# packages/hallederiz_kadir/pyproject.toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "hallederiz_kadir"
version = "0.1.0"
description = "LLM call execution hub — litellm, streaming, retries, quality"
requires-python = ">=3.10"
dependencies = ["litellm>=1.40.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create types.py with CallResult and CallError**

```python
# packages/hallederiz_kadir/src/hallederiz_kadir/types.py
"""Result and error types for HaLLederiz Kadir."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CallResult:
    """Successful LLM call result."""
    content: str
    tool_calls: list[dict] | None
    thinking: str | None
    usage: dict                    # {"prompt_tokens": N, "completion_tokens": N}
    cost: float                    # 0.0 for local models
    latency: float                 # seconds
    model: str                     # litellm name used
    model_name: str                # human-readable name
    is_local: bool
    provider: str                  # "local" or provider name
    task: str                      # task label for logging


@dataclass
class CallError:
    """Failed LLM call with classification."""
    category: str                  # "timeout", "rate_limit", "auth", "server_error",
                                   # "loading", "gpu_busy", "quality_failure",
                                   # "daily_exhausted", "connection_error"
    message: str
    retryable: bool                # hint for dispatcher's candidate loop
    partial_content: str | None = None  # salvaged from streaming on timeout
```

- [ ] **Step 3: Create __init__.py**

```python
# packages/hallederiz_kadir/src/hallederiz_kadir/__init__.py
"""HaLLederiz Kadir — LLM call execution hub."""

from .types import CallResult, CallError

__all__ = ["CallResult", "CallError"]
```

- [ ] **Step 4: Write test for types**

```python
# packages/hallederiz_kadir/tests/test_types.py
"""Tests for CallResult and CallError dataclasses."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hallederiz_kadir.types import CallResult, CallError


def test_call_result_construction():
    r = CallResult(
        content="hello",
        tool_calls=None,
        thinking=None,
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        cost=0.0,
        latency=1.5,
        model="openai/qwen3-30b",
        model_name="qwen3-30b",
        is_local=True,
        provider="local",
        task="executor",
    )
    assert r.content == "hello"
    assert r.is_local is True
    assert r.cost == 0.0


def test_call_result_with_tool_calls():
    r = CallResult(
        content="",
        tool_calls=[{"id": "1", "name": "search", "arguments": {"q": "test"}}],
        thinking=None,
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        cost=0.001,
        latency=2.3,
        model="groq/llama-8b",
        model_name="llama-8b",
        is_local=False,
        provider="groq",
        task="executor",
    )
    assert r.tool_calls[0]["name"] == "search"
    assert r.is_local is False


def test_call_error_construction():
    e = CallError(
        category="timeout",
        message="Timeout on qwen3-30b",
        retryable=True,
    )
    assert e.category == "timeout"
    assert e.retryable is True
    assert e.partial_content is None


def test_call_error_with_partial_content():
    e = CallError(
        category="timeout",
        message="Timeout on qwen3-30b",
        retryable=True,
        partial_content="The analysis shows...",
    )
    assert e.partial_content == "The analysis shows..."
```

- [ ] **Step 5: Run test**

Run: `pytest packages/hallederiz_kadir/tests/test_types.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Add editable install to requirements.txt**

Add `-e ./packages/hallederiz_kadir` to `requirements.txt` alongside other package installs.

- [ ] **Step 7: Install the package**

Run: `pip install -e ./packages/hallederiz_kadir`

- [ ] **Step 8: Commit**

```bash
git add packages/hallederiz_kadir/ requirements.txt
git commit -m "feat(hallederiz_kadir): scaffold package with CallResult/CallError types"
```

---

## Task 2: Response Parsing Module

**Files:**
- Create: `packages/hallederiz_kadir/src/hallederiz_kadir/response.py`
- Create: `packages/hallederiz_kadir/tests/test_response.py`

This module extracts response parsing from `router.py:1321-1483` — content, tool_calls, thinking extraction, think-tag stripping, cost calculation.

- [ ] **Step 1: Write tests for response parsing**

```python
# packages/hallederiz_kadir/tests/test_response.py
"""Tests for response parsing — content, tool_calls, thinking, think-tags, cost."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock
from hallederiz_kadir.response import parse_response


def _make_response(content="hello", tool_calls=None, reasoning_content=None,
                   thinking=None, usage=None, finish_reason="stop"):
    """Build a fake litellm ModelResponse."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = reasoning_content
    msg.thinking = thinking

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = (usage or {}).get("prompt_tokens", 10)
    resp.usage.completion_tokens = (usage or {}).get("completion_tokens", 5)

    return resp


def test_parse_simple_content():
    resp = _make_response(content="Hello world")
    result = parse_response(resp, model_name="test", is_local=True,
                            is_thinking=False)
    assert result["content"] == "Hello world"
    assert result["tool_calls"] is None
    assert result["thinking"] is None


def test_parse_tool_calls():
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "search"
    tc.function.arguments = '{"query": "test"}'
    resp = _make_response(content="", tool_calls=[tc])
    result = parse_response(resp, model_name="test", is_local=False,
                            is_thinking=False)
    assert result["tool_calls"][0]["name"] == "search"
    assert result["tool_calls"][0]["arguments"] == {"query": "test"}


def test_parse_thinking_from_reasoning_content():
    resp = _make_response(content="answer", reasoning_content="let me think...")
    result = parse_response(resp, model_name="test", is_local=True,
                            is_thinking=True)
    assert result["thinking"] == "let me think..."
    assert result["content"] == "answer"


def test_parse_thinking_from_think_tags():
    resp = _make_response(content="<think>reasoning</think>The answer is 42")
    result = parse_response(resp, model_name="test", is_local=True,
                            is_thinking=True)
    assert result["thinking"] == "reasoning"


def test_strip_think_tags_when_not_requested():
    resp = _make_response(content="<think>internal</think>The answer is 42")
    result = parse_response(resp, model_name="test", is_local=True,
                            is_thinking=False)
    assert "<think>" not in result["content"]
    assert "The answer is 42" in result["content"]
    assert result["thinking"] is None


def test_rescue_reasoning_content_when_content_empty():
    """When thinking not requested but model put everything in reasoning_content."""
    resp = _make_response(content="", reasoning_content="<think>actual answer here</think>")
    result = parse_response(resp, model_name="test", is_local=True,
                            is_thinking=False)
    assert result["content"] == "actual answer here"


def test_strip_think_preserves_content_when_all_in_think():
    """When all content is in <think> tags, preserve it instead of returning empty."""
    resp = _make_response(content="<think>The only content</think>")
    result = parse_response(resp, model_name="test", is_local=True,
                            is_thinking=False)
    assert result["content"] == "The only content"


def test_strip_unclosed_think_tag():
    resp = _make_response(content="<think>reasoning that got cut off")
    result = parse_response(resp, model_name="test", is_local=True,
                            is_thinking=False)
    assert "<think>" not in result["content"]


def test_parse_malformed_tool_call_arguments():
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "search"
    tc.function.arguments = "not json"
    resp = _make_response(content="", tool_calls=[tc])
    result = parse_response(resp, model_name="test", is_local=False,
                            is_thinking=False)
    assert result["tool_calls"][0]["arguments"] == {}


def test_cost_zero_for_local():
    resp = _make_response(content="hello")
    result = parse_response(resp, model_name="test", is_local=True,
                            is_thinking=False)
    assert result["cost"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/hallederiz_kadir/tests/test_response.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_response'`

- [ ] **Step 3: Implement response.py**

```python
# packages/hallederiz_kadir/src/hallederiz_kadir/response.py
"""Response parsing — extract content, tool_calls, thinking from litellm responses."""

from __future__ import annotations

import json
import re


def parse_response(
    response,
    model_name: str,
    is_local: bool,
    is_thinking: bool,
) -> dict:
    """Parse a litellm ModelResponse into a flat dict.

    Args:
        response: litellm.ModelResponse object
        model_name: human-readable model name (for logging)
        is_local: whether this was a local model call
        is_thinking: whether thinking was requested for this call

    Returns:
        dict with keys: content, tool_calls, thinking, usage, cost
    """
    msg = response.choices[0].message

    # ── Thinking extraction ──
    thinking = _extract_thinking(msg) if is_thinking else None

    # ── Rescue reasoning_content when content is empty ──
    if not is_thinking and not (msg.content or "").strip():
        rc = getattr(msg, "reasoning_content", None) or ""
        if rc.strip():
            msg.content = re.sub(r"</?think>", "", rc).strip()

    # ── Strip think tags when thinking wasn't requested ──
    if not is_thinking and msg.content and "<think>" in msg.content:
        original = msg.content
        msg.content = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL)
        msg.content = re.sub(r"<think>.*", "", msg.content, flags=re.DOTALL)
        msg.content = re.sub(r"</?think>", "", msg.content)
        msg.content = msg.content.strip()
        if not msg.content:
            msg.content = re.sub(r"</?think>", "", original).strip()

    # ── Tool calls ──
    tool_calls = None
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            fn = tc.function
            try:
                args = json.loads(fn.arguments) if fn.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({
                "id": tc.id,
                "name": fn.name,
                "arguments": args,
            })

    # ── Usage ──
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
        }

    # ── Cost ──
    cost = 0.0
    if not is_local:
        try:
            import litellm
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

    return {
        "content": msg.content or "",
        "tool_calls": tool_calls,
        "thinking": thinking,
        "usage": usage,
        "cost": cost,
    }


def _extract_thinking(msg) -> str | None:
    """Extract thinking content from a model response message."""
    if hasattr(msg, "thinking") and msg.thinking:
        return msg.thinking
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content
    content = msg.content or ""
    match = re.search(
        r"<(?:thinking|think)>(.*?)</(?:thinking|think)>",
        content, re.DOTALL,
    )
    return match.group(1).strip() if match else None
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/hallederiz_kadir/tests/test_response.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add packages/hallederiz_kadir/src/hallederiz_kadir/response.py packages/hallederiz_kadir/tests/test_response.py
git commit -m "feat(hallederiz_kadir): response parsing — content, tool_calls, thinking, think-tags"
```

---

## Task 3: Error Classification + Retry Module

**Files:**
- Create: `packages/hallederiz_kadir/src/hallederiz_kadir/retry.py`
- Create: `packages/hallederiz_kadir/tests/test_retry.py`

Extracts error classification from `router.py:1602-1623` and retry loop from `router.py:1283-1579`.

- [ ] **Step 1: Write tests for error classification**

```python
# packages/hallederiz_kadir/tests/test_retry.py
"""Tests for error classification and retry logic."""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hallederiz_kadir.retry import classify_error


def test_classify_timeout():
    assert classify_error("Timeout on qwen3-30b") == "timeout"
    assert classify_error("Connection timed out") == "timeout"


def test_classify_rate_limit():
    assert classify_error("rate limit exceeded") == "rate_limited"
    assert classify_error("429 Too Many Requests") == "rate_limited"
    assert classify_error("tokens per minute limit") == "rate_limited"


def test_classify_auth():
    assert classify_error("Invalid API key") == "auth_failure"
    assert classify_error("Authentication failed") == "auth_failure"
    assert classify_error("billing quota exceeded") == "auth_failure"


def test_classify_gpu_busy():
    assert classify_error("GPU queue timeout for qwen3-30b") == "gpu_busy"


def test_classify_daily_exhausted():
    assert classify_error("Daily limit exhausted for groq/llama") == "daily_exhausted"


def test_classify_loading():
    assert classify_error("loading model qwen3-30b") == "loading"


def test_classify_circuit_breaker():
    assert classify_error("circuit breaker active for model") == "circuit_breaker"
    assert classify_error("Failed to load local model xyz") == "circuit_breaker"


def test_classify_no_model():
    assert classify_error("No models available") == "no_model"


def test_classify_connection():
    assert classify_error("Connection refused") == "connection_error"
    assert classify_error("DNS resolution failed") == "connection_error"


def test_classify_unknown():
    assert classify_error("something weird happened") == "unknown"


def test_classify_server_error():
    assert classify_error("500 Internal Server Error") == "server_error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/hallederiz_kadir/tests/test_retry.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement retry.py**

```python
# packages/hallederiz_kadir/src/hallederiz_kadir/retry.py
"""Retry loop and error classification for LLM calls."""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Awaitable

from .types import CallError


def classify_error(error: str) -> str:
    """Classify an error string into a category for retry/routing decisions."""
    e = error.lower()
    if "gpu queue timeout" in e or "gpu access denied" in e:
        return "gpu_busy"
    if any(k in e for k in ("rate limit", "rate_limit", "429",
                             "too many requests", "tokens per minute",
                             "resource_exhausted")):
        return "rate_limited"
    if "daily limit exhausted" in e:
        return "daily_exhausted"
    if "loading model" in e:
        return "loading"
    if "circuit breaker" in e or "failed to load local model" in e:
        return "circuit_breaker"
    if "no models available" in e or "no models matched" in e:
        return "no_model"
    if any(k in e for k in ("api key", "authentication", "unauthorized",
                             "billing")):
        return "auth_failure"
    if any(k in e for k in ("timeout", "timed out")):
        return "timeout"
    if any(k in e for k in ("connection", "network", "dns", "refused")):
        return "connection_error"
    if any(k in e for k in ("500", "internal server error")):
        return "server_error"
    return "unknown"


async def execute_with_retry(
    call_fn: Callable[[], Awaitable],
    max_retries: int,
    timeout: float,
    is_local: bool,
    model_name: str,
    health_check: Callable[[], Awaitable[bool]] | None = None,
    is_swap_in_progress: Callable[[], bool] | None = None,
    partial_content_ref: list | None = None,
) -> dict | CallError:
    """Execute an LLM call with retry, timeout, and error handling.

    Args:
        call_fn: async callable that makes the litellm call and returns parsed dict
        max_retries: max attempts (2 for local, 3 for cloud)
        timeout: seconds for asyncio.wait_for
        is_local: whether this is a local model
        model_name: for error messages
        health_check: async fn to check if local server is alive (optional)
        is_swap_in_progress: fn returning True if model swap is happening (optional)
        partial_content_ref: single-element list updated by streaming with partial content

    Returns:
        Parsed response dict on success, CallError on failure.
    """
    last_error: str | None = None

    for attempt in range(max_retries):
        try:
            result = await asyncio.wait_for(call_fn(), timeout=timeout)
            return result

        except asyncio.TimeoutError:
            last_error = f"Timeout on {model_name}"
            # For local: check if server died
            if is_local and health_check:
                alive = await health_check()
                if not alive:
                    break  # no point retrying dead server
            continue

        except Exception as e:
            error_str = str(e).lower()
            last_error = str(e)

            # Auth/billing — not retryable
            if any(kw in error_str for kw in (
                "api key", "authentication", "unauthorized",
                "billing", "credit", "quota",
            )):
                break

            # Rate limit — backoff then retry
            is_rate_limit = any(kw in error_str for kw in (
                "rate limit", "rate_limit", "429",
                "too many requests", "tokens per minute",
                "resource_exhausted",
            ))
            if is_rate_limit:
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 5)
                    continue
                break

            # Local model loading — wait for it
            if "loading model" in error_str and is_local and health_check:
                load_wait = 0
                while load_wait < 30:
                    await asyncio.sleep(5)
                    load_wait += 5
                    if await health_check():
                        break
                if await health_check():
                    continue
                break

            # Local 500 during swap — wait briefly
            is_server_error = "500" in error_str or "internal server error" in error_str
            if is_server_error and is_local and is_swap_in_progress:
                if is_swap_in_progress():
                    swap_wait = 0
                    while is_swap_in_progress() and swap_wait < 10:
                        await asyncio.sleep(2)
                        swap_wait += 2
                    break  # model changed, let dispatcher re-select

            # Generic retry with small backoff
            if attempt < 1:
                await asyncio.sleep(2)
                continue
            break

    # All retries exhausted
    category = classify_error(last_error or "Unknown")
    partial = None
    if partial_content_ref and partial_content_ref[0]:
        partial = partial_content_ref[0]

    return CallError(
        category=category,
        message=last_error or "Unknown error",
        retryable=category in ("timeout", "rate_limited", "loading",
                               "server_error", "gpu_busy", "connection_error"),
        partial_content=partial,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/hallederiz_kadir/tests/test_retry.py -v`
Expected: All PASS (classify_error tests pass; execute_with_retry tested in Task 4)

- [ ] **Step 5: Commit**

```bash
git add packages/hallederiz_kadir/src/hallederiz_kadir/retry.py packages/hallederiz_kadir/tests/test_retry.py
git commit -m "feat(hallederiz_kadir): error classification and retry loop"
```

---

## Task 4: Main Caller Module

**Files:**
- Create: `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py`
- Create: `packages/hallederiz_kadir/tests/test_caller.py`
- Modify: `packages/hallederiz_kadir/src/hallederiz_kadir/__init__.py`

The core of HaLLederiz Kadir — receives a `ModelInfo` + messages and handles the complete call pipeline: kwargs building, local vs cloud routing, streaming, quality check, metrics.

- [ ] **Step 1: Write tests for the main call function**

```python
# packages/hallederiz_kadir/tests/test_caller.py
"""Tests for the main call() function with mocked litellm and backends."""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from hallederiz_kadir.caller import call
from hallederiz_kadir.types import CallResult, CallError


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_model_info(is_local=True, litellm_name="openai/qwen3-30b",
                     name="qwen3-30b", thinking_model=False,
                     supports_function_calling=True,
                     supports_json_mode=False, api_base=None,
                     max_tokens=4096, location="local", provider="llama_cpp",
                     sampling_overrides=None, is_free=False,
                     tokens_per_second=0.0):
    """Build a fake ModelInfo."""
    m = MagicMock()
    m.is_local = is_local
    m.litellm_name = litellm_name
    m.name = name
    m.thinking_model = thinking_model
    m.supports_function_calling = supports_function_calling
    m.supports_json_mode = supports_json_mode
    m.api_base = api_base or ("http://localhost:8080" if is_local else None)
    m.max_tokens = max_tokens
    m.location = location
    m.provider = provider
    m.sampling_overrides = sampling_overrides
    m.is_free = is_free
    m.tokens_per_second = tokens_per_second
    m.has_vision = False
    return m


def _make_litellm_response(content="Hello", tool_calls=None,
                           prompt_tokens=10, completion_tokens=5):
    """Build a fake litellm response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = None
    msg.thinking = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp._hidden_params = {}

    return resp


@patch("hallederiz_kadir.caller.litellm")
def test_call_local_success(mock_litellm):
    """Local model call succeeds — returns CallResult."""
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=True)

    result = run_async(call(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        timeout=60.0,
        task="executor",
        needs_thinking=False,
        estimated_output_tokens=500,
    ))

    assert isinstance(result, CallResult)
    assert result.content == "Hello"
    assert result.is_local is True
    assert result.cost == 0.0


@patch("hallederiz_kadir.caller.litellm")
def test_call_cloud_success(mock_litellm):
    """Cloud model call succeeds — uses KDV pre/post."""
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=False, litellm_name="groq/llama-8b",
                             name="llama-8b", location="cloud", provider="groq",
                             api_base=None)

    with patch("hallederiz_kadir.caller._kdv_pre_call", return_value=(True, 0.0, False)), \
         patch("hallederiz_kadir.caller._kdv_post_call"), \
         patch("hallederiz_kadir.caller.litellm.completion_cost", return_value=0.001):
        result = run_async(call(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            timeout=60.0,
            task="executor",
            needs_thinking=False,
            estimated_output_tokens=500,
        ))

    assert isinstance(result, CallResult)
    assert result.is_local is False
    assert result.provider == "groq"


@patch("hallederiz_kadir.caller.litellm")
def test_call_timeout_returns_call_error(mock_litellm):
    """Timeout returns CallError with category='timeout'."""
    mock_litellm.acompletion = AsyncMock(side_effect=asyncio.TimeoutError)
    model = _make_model_info(is_local=False, litellm_name="groq/llama-8b",
                             location="cloud", provider="groq", api_base=None)

    with patch("hallederiz_kadir.caller._kdv_pre_call", return_value=(True, 0.0, False)), \
         patch("hallederiz_kadir.caller._kdv_record_failure"):
        result = run_async(call(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            timeout=1.0,
            task="executor",
            needs_thinking=False,
            estimated_output_tokens=500,
        ))

    assert isinstance(result, CallError)
    assert result.category == "timeout"
    assert result.retryable is True


@patch("hallederiz_kadir.caller.litellm")
def test_call_sets_api_key_for_local(mock_litellm):
    """Local models get api_key='sk-no-key'."""
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=True)

    run_async(call(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        timeout=60.0,
        task="executor",
        needs_thinking=False,
        estimated_output_tokens=500,
    ))

    kwargs = mock_litellm.acompletion.call_args[1]
    assert kwargs["api_key"] == "sk-no-key"


@patch("hallederiz_kadir.caller.litellm")
def test_call_tools_set_tool_choice(mock_litellm):
    """When tools provided and model supports FC, tool_choice='auto'."""
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=True, supports_function_calling=True)
    tools = [{"type": "function", "function": {"name": "search"}}]

    run_async(call(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        timeout=60.0,
        task="executor",
        needs_thinking=False,
        estimated_output_tokens=500,
    ))

    kwargs = mock_litellm.acompletion.call_args[1]
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


@patch("hallederiz_kadir.caller.litellm")
def test_call_json_mode_fallback(mock_litellm):
    """When tools given but model lacks FC, falls back to json_mode."""
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    model = _make_model_info(is_local=True, supports_function_calling=False,
                             supports_json_mode=True)
    tools = [{"type": "function", "function": {"name": "search"}}]

    run_async(call(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        timeout=60.0,
        task="executor",
        needs_thinking=False,
        estimated_output_tokens=500,
    ))

    kwargs = mock_litellm.acompletion.call_args[1]
    assert "tools" not in kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/hallederiz_kadir/tests/test_caller.py -v`
Expected: FAIL — `ImportError: cannot import name 'call'`

- [ ] **Step 3: Implement caller.py**

```python
# packages/hallederiz_kadir/src/hallederiz_kadir/caller.py
"""Main call function — HaLLederiz Kadir's entry point.

Receives a ModelInfo + messages from the dispatcher and handles:
  - completion_kwargs building (sampling, api_base, api_key, tools)
  - local vs cloud routing (DaLLaMa infer / KDV pre/post)
  - litellm.acompletion with streaming
  - response parsing and quality check
  - metrics, audit, logging
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import litellm

from .types import CallResult, CallError
from .response import parse_response
from .retry import execute_with_retry, classify_error

litellm.suppress_debug_info = True
litellm.return_response_headers = True
litellm.request_timeout = 120


# ─── Lazy singleton accessors ──────────────────────────────────────────────

def _get_logger():
    try:
        from src.infra.logging_config import get_logger
        return get_logger("hallederiz_kadir")
    except Exception:
        import logging
        return logging.getLogger("hallederiz_kadir")


def _get_dallama():
    """Get the DaLLaMa-backed local model manager."""
    try:
        from src.models.local_model_manager import get_local_manager
        return get_local_manager()
    except Exception:
        return None


def _kdv_pre_call(litellm_name: str, provider: str, estimated_tokens: int):
    """Check rate limits via KDV before a cloud call.

    Returns: (allowed: bool, wait_seconds: float, daily_exhausted: bool)
    """
    try:
        from src.core.router import get_kdv
        kdv = get_kdv()
        pre = kdv.pre_call(litellm_name, provider, estimated_tokens)
        return (pre.allowed, pre.wait_seconds, pre.daily_exhausted)
    except Exception:
        return (True, 0.0, False)


def _kdv_post_call(litellm_name: str, provider: str, headers: dict,
                   token_count: int):
    """Record usage via KDV after a cloud call."""
    try:
        from src.core.router import get_kdv
        get_kdv().post_call(litellm_name, provider,
                            headers=headers, token_count=token_count)
    except Exception:
        pass


def _kdv_record_failure(litellm_name: str, provider: str, reason: str):
    """Record a failure via KDV."""
    try:
        from src.core.router import get_kdv
        get_kdv().record_failure(litellm_name, provider, reason)
    except Exception:
        pass


def _get_sampling_params(task: str, sampling_overrides=None) -> dict | None:
    """Get sampling params for a task."""
    try:
        from src.models.model_profiles import get_sampling_params
        return get_sampling_params(task, sampling_overrides=sampling_overrides)
    except Exception:
        return None


def _record_metrics(model_name: str, cost: float, latency_ms: float,
                    tokens: int):
    """Record call metrics to Prometheus via Nerd Herd."""
    try:
        from src.infra.metrics import track_model_call_metrics
        track_model_call_metrics(model=model_name, cost=cost,
                                 latency_ms=latency_ms, tokens=tokens)
    except Exception:
        pass


async def _record_audit(agent_type: str, model_litellm: str, task: str,
                        cost: float, latency: float):
    """Record audit trail."""
    try:
        from src.infra.audit import audit, ACTOR_AGENT, ACTION_MODEL_CALL
        await audit(
            actor=f"{ACTOR_AGENT}:{agent_type or 'unknown'}",
            action=ACTION_MODEL_CALL,
            target=model_litellm,
            details=f"task={task} cost=${cost:.4f} latency={latency:.1f}s",
        )
    except Exception:
        pass


def _check_quality(content: str) -> tuple[bool, str]:
    """Check content quality via Dogru mu Samet.

    Returns: (is_ok, reason)
    """
    if not content or len(content) < 20:
        return (True, "")
    try:
        from dogru_mu_samet import assess
        result = assess(content)
        if result.is_degenerate:
            return (False, result.summary)
    except Exception:
        pass
    return (True, "")


# ─── Streaming ─────────────────────────────────────────────────────────────

async def _stream_with_accumulator(
    completion_kwargs: dict,
    partial_content_ref: list,
) -> "litellm.ModelResponse":
    """Call litellm with streaming, accumulate content.

    Updates partial_content_ref[0] with accumulated content so partial
    output survives timeouts.
    """
    completion_kwargs["stream"] = True
    chunks = []
    accumulated = ""
    role = "assistant"
    finish_reason = None

    async for chunk in await litellm.acompletion(**completion_kwargs):
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            if delta.content:
                accumulated += delta.content
                partial_content_ref[0] = accumulated
            if delta.role:
                role = delta.role
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        chunks.append(chunk)

    # Build synthetic non-streaming response
    last = chunks[-1] if chunks else None
    resp = litellm.ModelResponse(
        id=getattr(last, "id", "0"),
        choices=[{
            "message": {"role": role, "content": accumulated},
            "finish_reason": finish_reason or "stop",
        }],
        usage=getattr(last, "usage", None),
    )
    return resp


# ─── Main Call Function ────────────────────────────────────────────────────

async def call(
    model,
    messages: list[dict],
    tools: list[dict] | None,
    timeout: float,
    task: str,
    needs_thinking: bool,
    estimated_output_tokens: int = 1000,
) -> CallResult | CallError:
    """Execute an LLM call against a single model.

    This is HaLLederiz Kadir's main entry point. The dispatcher calls this
    once per candidate model. The HaLLederiz Kadir handles everything from
    building the litellm kwargs to returning a parsed result.

    Args:
        model: ModelInfo object from router scoring
        messages: chat messages (already redacted/adapted by dispatcher)
        tools: tool definitions or None
        timeout: seconds (computed by dispatcher)
        task: task label for sampling profile and logging
        needs_thinking: whether to enable thinking for this call
        estimated_output_tokens: for max_tokens calculation
    """
    logger = _get_logger()
    is_thinking = model.thinking_model and needs_thinking
    is_local = model.is_local
    is_ollama = model.location == "ollama"

    # ── Sampling ──
    if is_thinking:
        sampling_params = None
    else:
        sampling_params = _get_sampling_params(
            task, sampling_overrides=getattr(model, "sampling_overrides", None)
        )

    # ── Max tokens ──
    _max_tokens = min(estimated_output_tokens * 2, model.max_tokens)

    # ── HTTP timeout (slightly shorter than asyncio deadline) ──
    http_timeout = max(10.0, float(timeout) - 5.0)

    # ── Build completion kwargs ──
    completion_kwargs = dict(
        model=model.litellm_name,
        messages=messages,
        max_tokens=_max_tokens,
        timeout=http_timeout,
    )

    if sampling_params:
        for k, v in sampling_params.items():
            completion_kwargs[k] = v
    elif model.thinking_model and not is_thinking:
        completion_kwargs["temperature"] = 0.3

    if model.api_base:
        completion_kwargs["api_base"] = model.api_base

    if is_local and not is_ollama:
        completion_kwargs["api_key"] = "sk-no-key"
        completion_kwargs["num_retries"] = 0

    # ── Tools ──
    use_tools = None
    if tools and model.supports_function_calling:
        use_tools = tools
        completion_kwargs["tools"] = tools
        completion_kwargs["tool_choice"] = "auto"
    elif tools and not model.supports_function_calling and model.supports_json_mode:
        completion_kwargs["response_format"] = {"type": "json_object"}

    # ── Rate limiting (cloud only) ──
    if not is_local:
        estimated_tokens = estimated_output_tokens * 3  # rough input+output
        allowed, wait_secs, daily_exhausted = _kdv_pre_call(
            model.litellm_name, model.provider, estimated_tokens
        )
        if daily_exhausted:
            return CallError(
                category="daily_exhausted",
                message=f"Daily limit exhausted for {model.name}",
                retryable=False,
            )
        if not allowed:
            if wait_secs > 0:
                await asyncio.sleep(wait_secs)
            else:
                return CallError(
                    category="rate_limited",
                    message=f"Rate limited for {model.name}",
                    retryable=True,
                )

    # ── GPU semaphore (local only) ──
    local_manager = None
    _inf_gen = None
    if is_local and not is_ollama:
        local_manager = _get_dallama()
        if local_manager:
            gpu_timeout = min(timeout, 120.0)
            granted = await local_manager.acquire_inference_slot(
                priority=getattr(model, "priority", 5),
                task_id=task,
                agent_type=task,
                timeout=gpu_timeout,
            )
            if not granted:
                return CallError(
                    category="gpu_busy",
                    message=f"GPU queue timeout for {model.name}",
                    retryable=True,
                )

            _inf_gen = local_manager.mark_inference_start()

            # Verify model didn't change during GPU wait
            if local_manager.current_model != model.name:
                actual = local_manager.current_model
                local_manager.mark_inference_end(_inf_gen)
                local_manager.release_inference_slot()
                return CallError(
                    category="loading",
                    message=f"Model swapped during GPU wait: {model.name} → {actual}",
                    retryable=True,
                )

    # ── Streaming decision ──
    use_stream = (
        is_local
        and not is_ollama
        and not use_tools  # tool calls need full response
    )

    # ── Execute with retry ──
    max_retries = 2 if is_local else 3
    partial_content_ref = [""]
    call_start = time.time()

    def _health_check_fn():
        if local_manager:
            return local_manager._health_check()
        return asyncio.coroutine(lambda: True)()

    def _swap_check_fn():
        if local_manager:
            return local_manager.swap_started_at > 0
        return False

    async def _do_call():
        if use_stream:
            return await _stream_with_accumulator(
                completion_kwargs, partial_content_ref
            )
        else:
            return await litellm.acompletion(**completion_kwargs)

    try:
        raw_result = await execute_with_retry(
            call_fn=_do_call,
            max_retries=max_retries,
            timeout=timeout,
            is_local=is_local,
            model_name=model.name,
            health_check=_health_check_fn if local_manager else None,
            is_swap_in_progress=_swap_check_fn if local_manager else None,
            partial_content_ref=partial_content_ref,
        )
    finally:
        # Release GPU regardless of outcome
        if local_manager and _inf_gen is not None:
            local_manager.mark_inference_end(_inf_gen)
            local_manager.release_inference_slot()

    # ── Handle retry failure ──
    if isinstance(raw_result, CallError):
        if not is_local:
            _kdv_record_failure(model.litellm_name, model.provider,
                                raw_result.category)
        return raw_result

    # ── Parse response ──
    call_latency = time.time() - call_start
    parsed = parse_response(
        raw_result, model_name=model.name, is_local=is_local,
        is_thinking=is_thinking,
    )

    # ── Cloud: record usage via KDV ──
    if not is_local and raw_result.usage:
        total_tokens = (
            (raw_result.usage.prompt_tokens or 0)
            + (raw_result.usage.completion_tokens or 0)
        )
        hidden = getattr(raw_result, "_hidden_params", None)
        headers = {}
        if hidden:
            headers = dict(
                hidden.get("additional_headers")
                or hidden.get("headers")
                or {}
            )
        _kdv_post_call(model.litellm_name, model.provider,
                       headers=headers, token_count=total_tokens)

    # ── Local: update measured speed ──
    if is_local and raw_result.usage:
        output_tokens = raw_result.usage.completion_tokens or 0
        if output_tokens > 0 and call_latency > 0:
            tok_per_sec = output_tokens / call_latency
            logger.info(
                "llm performance",
                model_name=model.name,
                output_tokens=output_tokens,
                latency=f"{call_latency:.1f}s",
                speed=f"{tok_per_sec:.1f} tok/s",
            )
            try:
                from src.models.model_registry import get_registry
                get_registry().update_measured_speed(model.name, tok_per_sec)
            except Exception:
                pass
            if local_manager and local_manager.runtime_state:
                try:
                    local_manager.runtime_state.measured_tps = tok_per_sec
                except Exception:
                    pass

    # ── Quality check ──
    content = parsed["content"]
    quality_ok, quality_reason = _check_quality(content)
    if not quality_ok:
        logger.warning("quality check failed",
                       model_name=model.name, reason=quality_reason)
        return CallError(
            category="quality_failure",
            message=f"Quality check failed: {quality_reason}",
            retryable=True,
            partial_content=content,
        )

    # ── Metrics ──
    total_tokens = (parsed["usage"].get("prompt_tokens", 0)
                    + parsed["usage"].get("completion_tokens", 0))
    _record_metrics(model.name, parsed["cost"], call_latency * 1000,
                    total_tokens)

    # ── Audit ──
    try:
        await _record_audit(task, model.litellm_name, task,
                            parsed["cost"], call_latency)
    except Exception:
        pass

    # ── Build result ──
    return CallResult(
        content=content,
        tool_calls=parsed["tool_calls"],
        thinking=parsed["thinking"],
        usage=parsed["usage"],
        cost=parsed["cost"],
        latency=call_latency,
        model=model.litellm_name,
        model_name=model.name,
        is_local=is_local,
        provider=model.provider if not is_local else "local",
        task=task,
    )
```

- [ ] **Step 4: Update __init__.py to export call**

```python
# packages/hallederiz_kadir/src/hallederiz_kadir/__init__.py
"""HaLLederiz Kadir — LLM call execution hub."""

from .types import CallResult, CallError
from .caller import call

__all__ = ["call", "CallResult", "CallError"]
```

- [ ] **Step 5: Run tests**

Run: `pytest packages/hallederiz_kadir/tests/test_caller.py -v`
Expected: All PASS

- [ ] **Step 6: Run all HaLLederiz Kadir tests**

Run: `pytest packages/hallederiz_kadir/tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add packages/hallederiz_kadir/
git commit -m "feat(hallederiz_kadir): main call() function — local/cloud routing, streaming, quality"
```

---

## Task 5: Wire Dispatcher to HaLLederiz Kadir

**Files:**
- Modify: `src/core/llm_dispatcher.py`

This is the integration task. Dispatcher absorbs the candidate iteration loop from router, calls `talker.call()` per candidate, and handles ensure_model/redaction/adaptation before each call.

- [ ] **Step 1: Add candidate iteration to dispatcher**

In `src/core/llm_dispatcher.py`, replace the `_route_main_work` method. The current implementation (lines 217-270) calls `router.call_model()`. The new version calls `router.select_model()` for candidates and `talker.call()` per candidate.

Replace `_route_main_work` with:

```python
async def _route_main_work(
    self,
    reqs: "ModelRequirements",
    messages: list[dict],
    tools: list[dict] | None,
    partial_buf: object | None = None,
    on_chunk: "Callable[[str], bool] | None" = None,
) -> dict:
    """Route a MAIN_WORK call. Can trigger model swaps."""
    from src.core.router import select_model, ModelCallFailed
    from src.models.model_registry import get_registry
    from hallederiz_kadir import call as talker_call, CallResult, CallError
    import copy

    timeout = self._compute_timeout(CallCategory.MAIN_WORK, reqs)
    candidates = self._select_candidates(reqs, tools)

    if not candidates:
        raise ModelCallFailed(
            call_id=f"{reqs.agent_type}:{reqs.primary_capability}",
            last_error="No models available",
            error_category="no_model",
        )

    last_error = "Unknown"

    for scored in candidates[:5]:
        model = scored.model
        is_thinking = model.thinking_model and reqs.needs_thinking

        # ── Ensure local model loaded with correct state ──
        if model.is_local and model.location != "ollama":
            load_ok = await self._ensure_local_model(model, reqs, is_thinking)
            if not load_ok:
                last_error = f"Failed to load local model {model.name}"
                continue

        # ── Prepare messages ──
        prepared = self._prepare_messages(messages, model)

        # ── Call via HaLLederiz Kadir ──
        result = await talker_call(
            model=model,
            messages=prepared,
            tools=tools,
            timeout=timeout,
            task=reqs.effective_task or reqs.primary_capability,
            needs_thinking=reqs.needs_thinking,
            estimated_output_tokens=reqs.estimated_output_tokens,
        )

        if isinstance(result, CallResult):
            return self._result_to_dict(result, scored, reqs)

        # CallError — log and try next candidate
        last_error = result.message
        logger.warning(
            "candidate failed",
            model_name=model.name,
            error_category=result.category,
            retryable=result.retryable,
        )
        if not result.retryable:
            break

    raise ModelCallFailed(
        call_id=f"{reqs.agent_type}:{reqs.primary_capability}",
        last_error=last_error,
        error_category=_classify_error_category(last_error),
    )
```

- [ ] **Step 2: Add _route_overhead using HaLLederiz Kadir**

Replace `_route_overhead` (lines 272-338) similarly:

```python
async def _route_overhead(
    self,
    reqs: "ModelRequirements",
    messages: list[dict],
    tools: list[dict] | None,
) -> dict:
    """Route an OVERHEAD call. CANNOT trigger model swaps."""
    from src.core.router import ModelCallFailed
    from hallederiz_kadir import call as talker_call, CallResult, CallError

    timeout = self._compute_timeout(CallCategory.OVERHEAD, reqs)

    # ── Cold-start wait ──
    if not self._get_loaded_model_name() and self._should_wait_for_cold_start():
        await self._wait_for_model_load(reqs)

    reqs_safe = self._prepare_overhead_reqs(reqs)
    candidates = self._select_candidates(reqs_safe, tools)

    if not candidates:
        raise RuntimeError(
            f"OVERHEAD call failed: no models available. "
            f"Task: {reqs.effective_task or reqs.primary_capability}"
        )

    last_error = "Unknown"

    for scored in candidates[:5]:
        model = scored.model
        prepared = self._prepare_messages(messages, model)

        result = await talker_call(
            model=model,
            messages=prepared,
            tools=tools,
            timeout=timeout,
            task=reqs.effective_task or reqs.primary_capability,
            needs_thinking=reqs.needs_thinking,
            estimated_output_tokens=reqs.estimated_output_tokens,
        )

        if isinstance(result, CallResult):
            return self._result_to_dict(result, scored, reqs)

        last_error = result.message
        if not result.retryable:
            break

    raise RuntimeError(
        f"OVERHEAD call failed: all candidates exhausted. "
        f"Task: {reqs.effective_task or reqs.primary_capability}, "
        f"Error: {last_error}"
    )
```

- [ ] **Step 3: Add helper methods to dispatcher**

Add these helper methods to `LLMDispatcher`:

```python
def _select_candidates(self, reqs, tools):
    """Score and select model candidates via router."""
    import copy
    from src.core.router import select_model

    if tools:
        reqs.needs_function_calling = True

    # Direct model override
    if reqs.model_override:
        from src.models.model_registry import get_registry, ModelInfo
        from src.core.router import ScoredModel, ALL_CAPABILITIES
        registry = get_registry()
        pinned = registry.find_by_litellm_name(reqs.model_override)
        if pinned:
            return [ScoredModel(model=pinned, score=999, reasons=["pinned"])]
        return [ScoredModel(
            model=ModelInfo(
                name="override", location="cloud", provider="unknown",
                litellm_name=reqs.model_override,
                capabilities={cap: 5.0 for cap in ALL_CAPABILITIES},
                context_length=128000, max_tokens=4096,
            ),
            score=999, reasons=["pinned_raw"],
        )]

    candidates = select_model(reqs)

    # Fallback relaxation
    if not candidates:
        fallback = copy.copy(reqs)
        fallback.difficulty = 1
        fallback.min_score = 0.01
        fallback.local_only = False
        fallback.needs_thinking = False
        fallback.needs_vision = False
        fallback.needs_function_calling = False
        candidates = select_model(fallback)

    return candidates


async def _ensure_local_model(self, model, reqs, is_thinking: bool) -> bool:
    """Ensure local model is loaded with correct vision/thinking state."""
    from src.models.local_model_manager import get_local_manager
    manager = get_local_manager()

    needs_vision = reqs.needs_vision and model.has_vision
    needs_thinking_reload = (
        model.thinking_model
        and is_thinking
        and not manager._thinking_enabled
    )
    needs_reload = (
        not model.is_loaded
        or needs_thinking_reload
        or (needs_vision and not manager._vision_enabled)
    )

    if needs_reload:
        success = await manager.ensure_model(
            model.name,
            reason=f"{reqs.agent_type}:{reqs.effective_task or reqs.primary_capability}",
            enable_thinking=is_thinking,
            enable_vision=needs_vision,
            min_context=reqs.effective_context_needed,
        )
        if not success:
            # Trigger proactive replacement load
            try:
                import asyncio
                asyncio.ensure_future(
                    self.ensure_gpu_utilized(
                        [{"agent_type": reqs.agent_type, "context": "{}"}]
                    )
                )
            except Exception:
                pass
            return False
    return True


def _prepare_messages(self, messages: list[dict], model) -> list[dict]:
    """Prepare messages — secret redaction for cloud, thinking adaptation."""
    _messages = messages

    # Redact secrets for cloud models
    if not model.is_local:
        try:
            from src.security.sensitivity import redact_secrets
            _messages = []
            for msg in messages:
                _m = dict(msg)
                if isinstance(_m.get("content"), str):
                    _m["content"] = redact_secrets(_m["content"])
                _messages.append(_m)
        except Exception:
            _messages = messages

    # Thinking models reject assistant prefills
    if (model.thinking_model and _messages
            and _messages[-1].get("role") == "assistant"):
        last = _messages[-1]
        _messages = _messages[:-1]
        _messages.append({
            "role": "user",
            "content": (
                "Your previous response (continue from here, "
                "do NOT repeat this):\n\n" + last["content"]
            ),
        })

    return _messages


def _prepare_overhead_reqs(self, reqs):
    """Prepare requirements for OVERHEAD — exclude unloaded local models."""
    import copy
    reqs_safe = copy.copy(reqs)

    sv = self._swap_version()
    loaded = self._get_loaded_model_name()

    if sv > 0 and not loaded:
        reqs_safe = self._exclude_all_local(reqs_safe)
    else:
        reqs_safe = self._exclude_unloaded_local(reqs_safe)

    return reqs_safe


def _result_to_dict(self, result: "CallResult", scored, reqs) -> dict:
    """Convert CallResult to the legacy response dict format."""
    return {
        "content": result.content,
        "model": result.model,
        "model_name": result.model_name,
        "cost": result.cost,
        "usage": result.usage,
        "tool_calls": result.tool_calls,
        "latency": result.latency,
        "thinking": result.thinking,
        "is_local": result.is_local,
        "ran_on": "local" if result.is_local else result.provider,
        "provider": result.provider,
        "task": result.task,
        "capability_score": scored.capability_score,
        "difficulty": reqs.difficulty,
    }
```

- [ ] **Step 4: Import _classify_error_category from HaLLederiz Kadir**

Add at the top of dispatcher (lazy import pattern):

```python
def _classify_error_category(error: str) -> str:
    from hallederiz_kadir.retry import classify_error
    return classify_error(error)
```

- [ ] **Step 5: Remove old call_model imports from dispatcher**

Remove the two `from src.core.router import call_model` lines (current lines 230 and 301) since they're replaced by `hallederiz_kadir.call`.

- [ ] **Step 6: Run existing dispatcher tests**

Run: `pytest tests/test_llm_dispatcher.py -v`
Expected: Tests that mock `src.core.router.call_model` will need patch path updates. Fix any that fail by updating mock targets.

- [ ] **Step 7: Commit**

```bash
git add src/core/llm_dispatcher.py
git commit -m "feat(dispatcher): absorb candidate iteration, call via HaLLederiz Kadir"
```

---

## Task 6: Router Shim + Cleanup

**Files:**
- Modify: `src/core/router.py`

Replace `call_model()` with a thin shim that routes through the dispatcher. Remove the 640 lines of call execution code. Keep all scoring logic intact.

- [ ] **Step 1: Replace call_model with shim**

Replace `call_model()` (lines 957-1597) and supporting functions with:

```python
# ─── Legacy Shim ──────────────────────────────────────────────────────────

async def call_model(
    reqs: ModelRequirements,
    messages: list[dict],
    tools: list[dict] | None = None,
    timeout_override: float | None = None,
    partial_buf: object | None = None,
    on_chunk: "Callable[[str], bool] | None" = None,
) -> dict:
    """Legacy shim — routes through dispatcher.

    All new code should call dispatcher.request() directly.
    This shim preserves backward compatibility during migration.
    """
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    return await get_dispatcher().request(
        category=CallCategory.MAIN_WORK,
        reqs=reqs,
        messages=messages,
        tools=tools,
    )
```

- [ ] **Step 2: Remove extracted code from router**

Remove these functions/code that now live in HaLLederiz Kadir:
- `_stream_with_accumulator()` (lines 43-93)
- `_extract_thinking()` (lines 1628-1638)
- `_classify_error_category()` (lines 1602-1623)
- The `litellm` import and configuration (lines 32-41)
- The `ModelCallFailed` class (lines 17-29) — move to HaLLederiz Kadir types or keep as re-export

Keep `ModelCallFailed` as a re-export from HaLLederiz Kadir for backward compatibility:

```python
# At top of router.py, after removing litellm import:
from hallederiz_kadir.retry import classify_error as _classify_error_category  # shim


class ModelCallFailed(RuntimeError):
    """All model candidates exhausted. Kept here for import compatibility."""
    def __init__(self, call_id: str, last_error: str, error_category: str):
        super().__init__(f"All models failed for '{call_id}': {last_error}")
        self.call_id = call_id
        self.last_error = last_error
        self.error_category = error_category
```

- [ ] **Step 3: Remove unused imports from router**

Remove imports that were only used by call execution:
- `litellm` (line 32)
- `re` (if only used in think-tag stripping — check first; used in `_make_adhoc_profile` too)
- `time` (if only used in call_model — check first)

Keep all imports used by scoring logic.

- [ ] **Step 4: Run router scoring tests**

Run: `pytest tests/ -k "router" -v`
Expected: Scoring tests pass. Tests that used `call_model` directly should go through the shim.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v --timeout=30`
Expected: All existing tests pass. The shim ensures backward compatibility.

- [ ] **Step 6: Verify import check**

Run: `python -c "from src.core.router import call_model, select_model, ModelCallFailed, ModelRequirements, ScoredModel; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/core/router.py
git commit -m "refactor(router): replace call_model with dispatcher shim, remove call execution code"
```

---

## Task 7: Integration Test

**Files:**
- Create: `tests/test_hallederiz_kadir_integration.py`

End-to-end test that verifies the full pipeline: dispatcher → HaLLederiz Kadir → mocked litellm.

- [ ] **Step 1: Write integration test**

```python
# tests/test_hallederiz_kadir_integration.py
"""Integration test: dispatcher → HaLLederiz Kadir → mocked litellm.

Verifies the full call pipeline works end-to-end without a real LLM.
"""

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_litellm_response(content="The answer is 42"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.reasoning_content = None
    msg.thinking = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 50
    resp.usage.completion_tokens = 10
    resp._hidden_params = {}

    return resp


def _make_model_info(name="test-model", is_local=False):
    m = MagicMock()
    m.name = name
    m.litellm_name = f"openai/{name}"
    m.is_local = is_local
    m.location = "local" if is_local else "cloud"
    m.provider = "llama_cpp" if is_local else "test_provider"
    m.thinking_model = False
    m.has_vision = False
    m.supports_function_calling = True
    m.supports_json_mode = False
    m.api_base = "http://localhost:8080" if is_local else None
    m.max_tokens = 4096
    m.sampling_overrides = None
    m.is_free = True
    m.tokens_per_second = 0.0
    m.is_loaded = True
    m.capabilities = {}
    m.context_length = 32000
    return m


def _make_scored(model):
    scored = MagicMock()
    scored.model = model
    scored.score = 8.5
    scored.capability_score = 7.2
    scored.reasons = ["test"]
    return scored


@patch("hallederiz_kadir.caller.litellm")
@patch("hallederiz_kadir.caller._kdv_pre_call", return_value=(True, 0.0, False))
@patch("hallederiz_kadir.caller._kdv_post_call")
@patch("hallederiz_kadir.caller._record_metrics")
@patch("hallederiz_kadir.caller._record_audit", new_callable=AsyncMock)
def test_full_pipeline_cloud(mock_audit, mock_metrics, mock_kdv_post,
                              mock_kdv_pre, mock_litellm):
    """Full pipeline: dispatcher → HaLLederiz Kadir → cloud model."""
    mock_litellm.acompletion = AsyncMock(return_value=_make_litellm_response())
    mock_litellm.completion_cost = MagicMock(return_value=0.001)

    model = _make_model_info(is_local=False)
    scored = _make_scored(model)

    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    dispatcher = LLMDispatcher()

    with patch.object(dispatcher, "_select_candidates", return_value=[scored]), \
         patch.object(dispatcher, "_prepare_messages", return_value=[{"role": "user", "content": "test"}]):

        from dataclasses import dataclass, field

        @dataclass
        class FakeReqs:
            task: str = "executor"
            primary_capability: str = "general"
            difficulty: int = 5
            estimated_output_tokens: int = 500
            estimated_input_tokens: int = 1000
            needs_thinking: bool = False
            needs_vision: bool = False
            needs_function_calling: bool = False
            local_only: bool = False
            prefer_speed: bool = False
            min_score: float = 0.0
            agent_type: str = "executor"
            effective_task: str = "executor"
            model_override: str | None = None
            priority: int = 5
            exclude_models: list = field(default_factory=list)

        reqs = FakeReqs()

        result = run_async(dispatcher.request(
            category=CallCategory.MAIN_WORK,
            reqs=reqs,
            messages=[{"role": "user", "content": "What is 6*7?"}],
        ))

    assert result["content"] == "The answer is 42"
    assert result["capability_score"] == 7.2
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_hallederiz_kadir_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --timeout=30 -x`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_hallederiz_kadir_integration.py
git commit -m "test(hallederiz_kadir): integration test — full dispatcher → HaLLederiz Kadir pipeline"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `docs/architecture-modularization.md`

- [ ] **Step 1: Update system layers diagram**

In `docs/architecture-modularization.md`, update the System Layers diagram to include HaLLederiz Kadir:

```
Telegram / API
       │
  Orchestrator ─── picks tasks from queue by priority, dependencies
       │
  LLM Dispatcher ─── candidate iteration, swap budget, model selection (via Router)
       │
  HaLLederiz Kadir ─── litellm call, streaming, retry, quality check, metrics
       │
  ┌────┴────┐
  │         │
DaLLaMa   KDV
(local)   (cloud)
  │
llama-server
```

- [ ] **Step 2: Update extracted packages table**

Add HaLLederiz Kadir to the Extracted Packages table:

```markdown
| **hallederiz_kadir** | LLM call execution hub: litellm, streaming, retries, quality | `packages/hallederiz_kadir/` | New | litellm, dogru_mu_samet |
```

- [ ] **Step 3: Update data flow section**

Update the "Data Flow: Local LLM Call" section to show HaLLederiz Kadir:

```
Agent needs LLM call
  → Dispatcher.request(messages, MAIN_WORK)
    → Router scores models, picks "qwen3-30b"
    → Dispatcher: ensure_model via DaLLaMa, prepare messages
    → TalkingLayer.call(model, messages, ...)
       → DaLLaMa.infer() → GPU acquire → endpoint URL
       → litellm.acompletion(api_base=url, ...)
       → Parse response, quality check, metrics
    → Response flows back: TalkingLayer → Dispatcher → Agent
  → DaLLaMa marks inference done, resets idle timer
```

- [ ] **Step 4: Update "What Agents Need to Know" section**

Add a HaLLederiz Kadir entry:

```markdown
**If you're fixing an LLM call error (timeout, retry, streaming, response parsing):**
The real logic is in `packages/hallederiz_kadir/src/hallederiz_kadir/`. The caller in dispatcher just iterates candidates. Don't touch dispatcher or router for call execution bugs — fix HaLLederiz Kadir.
```

- [ ] **Step 5: Update "Don't Extract" table**

Change the LLM Router / Dispatcher entry to clarify the split:

```markdown
| LLM Router / Dispatcher | Router: 15-dimension scoring is KutAI-shaped. Dispatcher: swap budget + candidate orchestration is KutAI-shaped. Call execution extracted to hallederiz_kadir. |
```

- [ ] **Step 6: Commit**

```bash
git add docs/architecture-modularization.md
git commit -m "docs: update architecture for HaLLederiz Kadir extraction"
```

---

## Summary

| Task | What | Estimated Size |
|------|------|---------------|
| 1 | Package scaffold + types | ~100 lines |
| 2 | Response parsing module | ~120 lines |
| 3 | Error classification + retry | ~130 lines |
| 4 | Main caller module | ~280 lines |
| 5 | Wire dispatcher to HaLLederiz Kadir | ~200 lines changed |
| 6 | Router shim + cleanup | ~600 lines removed |
| 7 | Integration test | ~120 lines |
| 8 | Documentation update | ~50 lines changed |

**Net effect:** Router drops from ~1,700 to ~500 lines. Dispatcher grows from ~550 to ~750 lines. New package is ~400 lines. Call execution is cleanly separated with hard package boundaries.
