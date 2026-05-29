"""SP3b Task 8 — critic_gate split tests.

The MECHANICAL confirm gate must make NO dispatcher call. The verdict
PRODUCER routes through the admitted single-call worker (husam), never
``LLMDispatcher().request(...)`` directly, and persists the verdict.

These tests pin the new shape:
  * ``confirm_gate(...)``  — mechanical, LLM-free, reads a persisted verdict.
  * ``produce_verdict(...)`` — admitted producer, goes through husam.run,
    persists via ``_persist``.
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
async def test_mechanical_gate_missing_verdict_default_passes(monkeypatch):
    """No persisted verdict (producer never ran / failed) → default-pass.

    Never block work on a missing verdict — fail-open, matching the legacy
    'broken critic never blocks' contract.
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
    assert verdict["verdict"] == "pass"


# ─── Producer: routes through husam, persists ────────────────────────────


@pytest.mark.asyncio
async def test_producer_persists_verdict(monkeypatch):
    """produce_verdict goes through husam.run (NOT dispatcher.request) and
    persists the parsed verdict."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    captured = {}

    async def _fake_husam_run(spec):
        captured["spec"] = spec
        return {"content": '{"verdict": "veto", "reasons": ["leaks API key"]}'}

    persist_mock = AsyncMock()
    with patch("husam.run", _fake_husam_run), patch(_PERSIST_PATCH, persist_mock):
        result = await cg.produce_verdict(
            "git_commit",
            {"commit_message": "leak", "diff": "+sk-abc"},
            mission_id=42,
        )
    assert result["verdict"] == "veto"
    assert "leaks API key" in result["reasons"][0]
    # Persisted with the parsed verdict.
    persist_mock.assert_awaited_once()
    pa = persist_mock.await_args
    assert pa.args[2] == "veto"  # (mission_id, action_name, verdict, reasons, hash)
    # The producer built a raw_dispatch overhead spec for husam.
    llm = captured["spec"]["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    assert llm["call_category"] == "overhead"
    assert llm["agent_type"] == "critic"


@pytest.mark.asyncio
async def test_producer_redacts_before_husam(monkeypatch):
    """Secrets must not appear in the messages handed to husam."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    captured = {}

    async def _fake_husam_run(spec):
        captured["spec"] = spec
        return {"content": '{"verdict": "pass", "reasons": []}'}

    with patch("husam.run", _fake_husam_run), patch(_PERSIST_PATCH, new_callable=AsyncMock):
        await cg.produce_verdict(
            "notify_user",
            {"message": "your api_key=sk-1234567890abcdef1234567890abcdef"},
        )
    sent = str(captured["spec"]["context"]["llm_call"]["messages"])
    assert "sk-1234567890abcdef1234567890abcdef" not in sent
    assert "REDACTED" in sent


@pytest.mark.asyncio
async def test_producer_husam_failure_default_passes(monkeypatch):
    """When husam raises, default-pass — never block on a broken critic."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)

    async def _boom(spec):
        raise RuntimeError("model down")

    with patch("husam.run", _boom), patch(_PERSIST_PATCH, new_callable=AsyncMock):
        result = await cg.produce_verdict("git_commit", {"x": 1})
    assert result["verdict"] == "pass"
    assert any("model down" in r for r in result["reasons"])


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
