"""Test-suite conftest for the mr_roboto package.

Why this exists
---------------
SP5 (2026-06-11) deleted the blocking ``enqueue(await_inline=True)`` primitive,
so the old inline-wait hang (a parked ``_inline_waiters`` Future stalling for
the 600s ``INLINE_TIMEOUT`` when no Beckman worker drains the queue) can no
longer occur — all enqueues are fire-and-continue. One autouse guard remains:

  `KUTAI_CRITIC_GATE=off`. The critic gate is a pre-hook on `git_commit` /
  `notify_user` that fires a SECOND LLM call via the dispatcher *and* stages
  the repo. It is incidental infrastructure for every test except
  `test_critic_gate.py`, which manages the env var itself (delenv/setenv) and
  so overrides this guard for its own cases.

This guard does not change production behavior — it only skips incidental infra.
A test that genuinely exercises an LLM-bound verb must still mock the dispatcher
/ `beckman.enqueue`.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _mr_roboto_no_llm_hang(monkeypatch):
    # Skip the critic-gate LLM pre-hook on irreversible verbs.
    # test_critic_gate.py overrides this with its own monkeypatch.
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    yield
