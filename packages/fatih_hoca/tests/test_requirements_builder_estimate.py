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


def test_review_step_escalates_estimate_by_full_artifact_size(monkeypatch):
    """Mission 90 task 567399: review steps fetch their input_artifacts in FULL
    (not the lossy _summary), so the input estimate must escalate by the real
    full-artifact size — else the summary-learned btable value (~4k) lets
    selection pick a small-ctx model that can't hold the full docs. Escalating
    estimated_input_tokens floors effective_context_length()."""
    big = "X" * 40000  # ~10k tok per artifact (char//4)

    class _Store:
        async def retrieve(self, mid, name):
            return "tiny" if name.endswith("_summary") else big

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store()
    )
    ctx = {
        "workflow_step_id": "1.13",
        "mission_id": 90,
        "input_artifacts": ["market_research_report", "product_charter"],
        # set output override so the builder skips its DB step-refresh read
        "estimated_output_tokens": 2000,
    }
    task = {"id": 9, "mission_id": 90, "priority": 5}
    reqs = _run(requirements_for(task, ctx, agent_name="reviewer"))

    base = _admission_estimate("reviewer", ctx)
    # two ~10k-tok artifacts → estimate must escalate well above base
    assert reqs.estimated_input_tokens > base
    assert reqs.estimated_input_tokens >= 20000
    # effective ctx floor follows the escalated estimate (no explicit min set)
    assert reqs.effective_context_needed > 25000


def test_review_step_no_artifacts_keeps_base_estimate(monkeypatch):
    """No input_artifacts → nothing to escalate; base estimate preserved."""
    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store",
        lambda: (_ for _ in ()).throw(AssertionError("store should not be read")),
    )
    ctx = {"workflow_step_id": "1.13", "mission_id": 90}
    task = {"id": 10, "mission_id": 90, "priority": 5}
    reqs = _run(requirements_for(task, ctx, agent_name="reviewer"))
    assert reqs.estimated_input_tokens == _admission_estimate("reviewer", ctx)
