# test_requirements_builder_estimate.py
"""RC-A change B (mission 74): the worker's input-token estimate must come
from the same source as admission — estimate_for (btable chain) — NOT the
char-based (desc+ctx)//4 heuristic.

Ground-truth data (9998 calls) showed char-based floors to 1000 while the
real assembled prompt is ~9k (RAG + system + tools added at call time), so
char under-counts ~9000 tokens and never predicts a per-call TPM overflow.
btable's agent-default (~8000) tracks actual ~2x better. Unifying the
worker on estimate_for makes admission's pick == the worker's pick.
"""
from __future__ import annotations

import asyncio

from fatih_hoca.requirements_builder import requirements_for
from fatih_hoca.requirements import AGENT_REQUIREMENTS
from fatih_hoca.estimates import estimate_for
from general_beckman.btable_cache import get_btable


def _run(coro):
    return asyncio.run(coro)


class _Shim:
    __slots__ = ("agent_type", "context")


def _admission_estimate(agent: str, ctx: dict) -> int:
    """The exact value admission's estimate_for would produce for parity."""
    sh = _Shim(); sh.agent_type = agent; sh.context = ctx
    return estimate_for(sh, btable=get_btable()).in_tokens


def test_short_task_uses_btable_estimate_not_char_floor():
    """A short task floors char-based to 1000; the worker must instead use
    the estimate_for agent-default (parity with admission)."""
    task = {"id": 1, "title": "x", "description": "do a thing", "priority": 5}
    reqs = _run(requirements_for(task, {}, agent_name="coder"))
    expected = _admission_estimate("coder", {})
    assert expected > 1000, "precondition: coder default must exceed char floor"
    assert reqs.estimated_input_tokens == expected


def test_huge_context_task_still_uses_btable_estimate():
    """Decision (founder-confirmed): drop char entirely. Even a large stored
    context resolves to the estimate_for value — admission uses the same, so
    they cannot diverge into the admit-then-no_candidates trap."""
    big_ctx_str = "x" * 40000  # char-based would yield ~10000
    task = {"id": 2, "title": "y", "description": "z", "priority": 5,
            "context": big_ctx_str}
    reqs = _run(requirements_for(task, {}, agent_name="executor"))
    expected = _admission_estimate("executor", {})
    assert reqs.estimated_input_tokens == expected
    # prove char-based (≈10000) is NOT what we returned
    assert reqs.estimated_input_tokens != (len("z") + len(big_ctx_str)) // 4
