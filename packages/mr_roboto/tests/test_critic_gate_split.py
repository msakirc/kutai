"""critic_gate split tests (SP3b Task 8 + SP6).

The MECHANICAL confirm gate must make NO dispatcher call. The inline
``produce_verdict`` producer was deleted in SP6 T5 — the verdict now travels
through Beckman's admitted-child path built by ``_build_critic_spec``.

These tests pin the surviving shape:
  * ``confirm_gate(...)``  — mechanical, LLM-free, reads a persisted verdict.
  * ``_build_critic_spec(...)`` — redacts secrets, builds a raw_dispatch
    overhead spec for the admitted critic child.
  * the module never imports/uses LLMDispatcher.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

import mr_roboto.critic_gate as cg


_PERSIST_PATCH = "mr_roboto.critic_gate._persist"


# ─── Mechanical confirm gate: NO dispatcher access ───────────────────────


@pytest.mark.asyncio
async def test_mechanical_gate_makes_no_llm_call(monkeypatch):
    """The mechanical confirm gate must never touch the dispatcher."""
    # The package conftest sets KUTAI_CRITIC_GATE=off autouse; this case
    # exercises the persisted-verdict path, not the opt-out, so clear it.
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    # Any dispatcher access (singleton OR constructor) must raise — proving
    # the mechanical path is LLM-free.
    monkeypatch.setattr(
        "src.core.llm_dispatcher.get_dispatcher",
        lambda: (_ for _ in ()).throw(AssertionError("mechanical called dispatcher")),
    )
    monkeypatch.setattr(
        "src.core.llm_dispatcher.LLMDispatcher",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("mechanical constructed dispatcher")),
    )
    with patch(_PERSIST_PATCH, new_callable=AsyncMock):
        verdict = await cg.confirm_gate(
            action_name="git_commit",
            payload={"x": 1},
            mission_id=1,
            persisted_verdict={"verdict": "pass", "reasons": []},
        )
    assert verdict["verdict"] == "pass"
    assert verdict["bypassed"] is False


@pytest.mark.asyncio
async def test_mechanical_gate_blocks_on_persisted_veto(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    with patch(_PERSIST_PATCH, new_callable=AsyncMock):
        verdict = await cg.confirm_gate(
            action_name="notify_user",
            payload={"x": 1},
            persisted_verdict={"verdict": "veto", "reasons": ["leaks PII"]},
        )
    assert verdict["verdict"] == "veto"
    assert "leaks PII" in verdict["reasons"][0]


@pytest.mark.asyncio
async def test_mechanical_gate_opt_out_passes(monkeypatch):
    """KUTAI_CRITIC_GATE=off → pass + bypassed, no dispatcher, no verdict needed."""
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    monkeypatch.setattr(
        "src.core.llm_dispatcher.get_dispatcher",
        lambda: (_ for _ in ()).throw(AssertionError("opt-out called dispatcher")),
    )
    with patch(_PERSIST_PATCH, new_callable=AsyncMock):
        verdict = await cg.confirm_gate(
            action_name="git_commit", payload={"x": 1}, persisted_verdict=None
        )
    assert verdict["verdict"] == "pass"
    assert verdict["bypassed"] is True


@pytest.mark.asyncio
async def test_mechanical_gate_missing_verdict_fails_closed(monkeypatch):
    """No persisted verdict (producer never ran / failed) → veto (fail-closed).

    SP6 T1: the gate is fail-CLOSED. A missing verdict is not safe to pass;
    the only way to bypass without a real verdict is KUTAI_CRITIC_GATE=off.
    The gate must still make no LLM/dispatcher call — it is purely mechanical.
    """
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    monkeypatch.setattr(
        "src.core.llm_dispatcher.get_dispatcher",
        lambda: (_ for _ in ()).throw(AssertionError("missing-verdict called dispatcher")),
    )
    with patch(_PERSIST_PATCH, new_callable=AsyncMock):
        verdict = await cg.confirm_gate(
            action_name="git_commit", payload={"x": 1}, persisted_verdict=None
        )
    assert verdict["verdict"] == "veto"
    assert verdict["bypassed"] is False


# ─── Critic-child spec build: redaction + admitted-overhead shape ────────
# (The inline ``produce_verdict`` producer was deleted in SP6 T5; the verdict
#  now travels via Beckman's admitted-child path. The spec builder is the
#  surviving seam — pin its redaction + raw_dispatch/overhead shape here.)


def test_build_critic_spec_redacts_and_is_overhead():
    spec = cg._build_critic_spec(
        "notify_user",
        cg._redact_payload(
            {"message": "your api_key=sk-1234567890abcdef1234567890abcdef"}
        ),
    )
    llm = spec["context"]["llm_call"]
    sent = str(llm["messages"])
    assert "sk-1234567890abcdef1234567890abcdef" not in sent
    assert "REDACTED" in sent
    assert llm["raw_dispatch"] is True
    assert llm["call_category"] == "overhead"
    assert llm["agent_type"] == "critic"


def test_critic_gate_no_longer_imports_dispatcher():
    """The module must not import LLMDispatcher nor call .request(...).

    AST-based so docstring mentions of the old shape don't false-trip.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cg))

    # No import of LLMDispatcher (from src.core.llm_dispatcher import ...).
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "LLMDispatcher", (
                    "critic_gate must not import LLMDispatcher"
                )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            # llm_dispatcher must not be imported at all (no execute/request).
            assert "llm_dispatcher" not in mod, (
                "critic_gate must not import from llm_dispatcher"
            )

    # No `.request(...)` attribute call anywhere in real code.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "request":
                raise AssertionError("critic_gate must not call .request(...)")


# ─── SP6 Task 1: confirm_gate fail-CLOSED ────────────────────────────────


@pytest.mark.asyncio
async def test_confirm_gate_fail_closed_on_missing_verdict(monkeypatch):
    """Gate ENABLED + no usable verdict → VETO (fail-closed), not pass."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)  # gate on
    res = await cg.confirm_gate("git_commit", {"x": 1}, persisted_verdict=None)
    assert res["verdict"] == "veto"
    assert res["bypassed"] is False


@pytest.mark.asyncio
async def test_confirm_gate_fail_closed_on_garbage_verdict(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    res = await cg.confirm_gate("git_commit", {"x": 1},
                                persisted_verdict={"verdict": "banana"})
    assert res["verdict"] == "veto"


@pytest.mark.asyncio
async def test_confirm_gate_optout_still_passes_without_verdict(monkeypatch):
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    res = await cg.confirm_gate("git_commit", {"x": 1}, persisted_verdict=None)
    assert res["verdict"] == "pass"
    assert res["bypassed"] is True


@pytest.mark.asyncio
async def test_confirm_gate_honours_real_verdict(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    pv = {"verdict": "veto", "reasons": ["leaks a secret"]}
    res = await cg.confirm_gate("git_commit", {"x": 1}, persisted_verdict=pv)
    assert res["verdict"] == "veto"
    assert res["reasons"] == ["leaks a secret"]
    pv2 = {"verdict": "pass", "reasons": []}
    res2 = await cg.confirm_gate("git_commit", {"x": 1}, persisted_verdict=pv2)
    assert res2["verdict"] == "pass"
