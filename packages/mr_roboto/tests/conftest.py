"""Test-suite conftest for the mr_roboto package.

Why this exists
---------------
mr_roboto verbs dispatch LLM-bound work through Beckman's singular-admission
queue. Both paths converge on the same wait:

    LLMDispatcher.request()  ->  general_beckman.enqueue(await_inline=True)
    critic-gate pre-hook     ->  LLMDispatcher.request()  ->  (same)

`enqueue(await_inline=True)` parks an asyncio.Future in `_inline_waiters` and
awaits `wait_for(fut, timeout=INLINE_TIMEOUT)`. In the live system a Beckman
worker drains the queue and calls `resolve_inline()`. Unit tests run no
worker — so any un-mocked inline-wait blocks for the full production
`INLINE_TIMEOUT` (600s) before raising TimeoutError. Across ~70 test files
that turns the suite into an effectively-infinite hang.

Two autouse guards keep the suite finite:

1. `INLINE_TIMEOUT` is shrunk to a few seconds. The module-level comment on
   it ("Monkeypatchable in tests") invites exactly this. Any test that leaves
   a real inline-wait un-mocked now fails fast and visibly instead of hanging.

2. `KUTAI_CRITIC_GATE=off`. The critic gate is a pre-hook on `git_commit` /
   `notify_user` that fires a SECOND LLM call via the dispatcher *and* stages
   the repo. It is incidental infrastructure for every test except
   `test_critic_gate.py`, which manages the env var itself (delenv/setenv)
   and so overrides this guard for its own cases.

Neither guard changes production behavior — they only bound the hang and skip
incidental infra. A test that genuinely exercises an LLM-bound verb must
still mock the dispatcher / `beckman.enqueue`; this conftest only ensures the
gap surfaces as a fast failure rather than a 10-minute stall.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _mr_roboto_no_llm_hang(monkeypatch):
    # Guard 1 — shrink the inline-wait ceiling so an un-resolved Beckman
    # Future fails fast rather than stalling for the production 600s.
    try:
        import general_beckman
        monkeypatch.setattr(general_beckman, "INLINE_TIMEOUT", 3.0)
    except Exception:
        pass

    # Guard 2 — skip the critic-gate LLM pre-hook on irreversible verbs.
    # test_critic_gate.py overrides this with its own monkeypatch.
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    yield
