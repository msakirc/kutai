"""Tests for the B4 Critic gate."""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, patch

import mr_roboto
import mr_roboto.critic_gate as CRITIC_MODULE
from mr_roboto.critic_gate import (
    critic_gate,
    _redact,
    _redact_payload,
    _parse_verdict,
    _opt_out,
)

_PERSIST_PATCH = "mr_roboto.critic_gate._persist"


# ─── Unit: redaction ─────────────────────────────────────────────────────


def test_redact_strips_api_key():
    text = "use api_key=sk-1234567890abcdefABCDEF1234567890ab to call"
    out = _redact(text)
    assert "sk-1234567890" not in out
    assert "REDACTED" in out


def test_redact_strips_email_and_phone():
    text = "contact alice@example.com or +90 532 555 1212"
    out = _redact(text)
    assert "alice@example.com" not in out
    assert "555 1212" not in out and "5551212" not in out


def test_redact_payload_recursive():
    payload = {
        "msg": "Bearer abcdef0123456789abcdef0123456789",
        "nested": {"email": "x@y.com", "ok": "fine"},
        "list": ["api_key=secret123", "hello"],
    }
    out = _redact_payload(payload)
    assert "Bearer abcdef" not in str(out)
    assert "x@y.com" not in str(out)
    # Non-secret strings preserved
    assert out["nested"]["ok"] == "fine"


# ─── Unit: verdict parsing ───────────────────────────────────────────────


def test_parse_verdict_clean_json():
    out = _parse_verdict('{"verdict": "veto", "reasons": ["leaks PII"]}')
    assert out["verdict"] == "veto"
    assert out["reasons"] == ["leaks PII"]


def test_parse_verdict_fenced_json():
    raw = 'Here you go:\n```json\n{"verdict":"pass","reasons":[]}\n```'
    out = _parse_verdict(raw)
    assert out["verdict"] == "pass"


def test_parse_verdict_garbage_defaults_to_pass():
    out = _parse_verdict("totally not json")
    assert out["verdict"] == "pass"
    assert out["reasons"]  # has at least one explanatory reason


def test_parse_verdict_unknown_verdict_normalises_to_pass():
    out = _parse_verdict('{"verdict": "maybe", "reasons": ["uncertain"]}')
    assert out["verdict"] == "pass"


# ─── Unit: parse_verdict_strict (SP6 gate-side, FAIL-CLOSED) ──────────────
# Surface A (and Tasks 3-4 surface B) stamp the verdict via this helper.
# Unlike _parse_verdict, garbage/empty/non-enum MUST veto.


def test_parse_verdict_strict_good_pass():
    from mr_roboto.critic_gate import parse_verdict_strict
    out = parse_verdict_strict('{"verdict": "pass", "reasons": []}')
    assert out["verdict"] == "pass"


def test_parse_verdict_strict_good_veto_keeps_reasons():
    from mr_roboto.critic_gate import parse_verdict_strict
    out = parse_verdict_strict('{"verdict": "veto", "reasons": ["leaks a token"]}')
    assert out["verdict"] == "veto"
    assert out["reasons"] == ["leaks a token"]


def test_parse_verdict_strict_fenced_json():
    from mr_roboto.critic_gate import parse_verdict_strict
    out = parse_verdict_strict('```json\n{"verdict": "veto", "reasons": ["x"]}\n```')
    assert out["verdict"] == "veto"


@pytest.mark.parametrize("raw", [
    "",
    "   ",
    "not json at all",
    '{"foo": 1}',                              # no verdict key
    '{"verdict": "maybe"}',                    # non-enum verdict
    '["verdict", "pass"]',                     # not an object
    '{"verdict": "pass"',                      # truncated / unparseable
])
def test_parse_verdict_strict_fails_closed(raw):
    from mr_roboto.critic_gate import parse_verdict_strict
    assert parse_verdict_strict(raw)["verdict"] == "veto", (
        f"strict parse must veto on {raw!r} (fail-closed)"
    )


# ─── Unit: opt-out ───────────────────────────────────────────────────────


def test_opt_out_recognises_off(monkeypatch):
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    assert _opt_out() is True
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "0")
    assert _opt_out() is True


def test_opt_out_default_on(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    assert _opt_out() is False


# ─── Unit: critic_gate happy paths ───────────────────────────────────────


# SP3b Task 8: the LLM hop now travels through ``husam.run`` (the admitted
# single-call worker), NOT ``LLMDispatcher().request(...)``. These behavioral
# tests are repointed to patch ``husam.run`` accordingly.


@pytest.mark.asyncio
async def test_critic_gate_pass_branch(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    fake_resp = {"content": '{"verdict": "pass", "reasons": []}'}
    husam_run = AsyncMock(return_value=fake_resp)
    with patch("husam.run", husam_run), patch(_PERSIST_PATCH, new_callable=AsyncMock):
        result = await critic_gate(
            "git_commit", {"commit_message": "fix typo", "diff": "+ 1 line"}
        )
    assert result["verdict"] == "pass"
    assert result["bypassed"] is False
    husam_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_critic_gate_veto_branch(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    fake_resp = {
        "content": '{"verdict": "veto", "reasons": ["leaks API key"]}'
    }
    husam_run = AsyncMock(return_value=fake_resp)
    with patch("husam.run", husam_run), patch(_PERSIST_PATCH, new_callable=AsyncMock):
        result = await critic_gate(
            "notify_user",
            {"message": "Your token is sk-abc123", "chat_id": 1},
        )
    assert result["verdict"] == "veto"
    assert "leaks API key" in result["reasons"][0]


@pytest.mark.asyncio
async def test_critic_gate_opt_out_short_circuits(monkeypatch):
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    husam_run = AsyncMock()
    with patch("husam.run", husam_run), patch(_PERSIST_PATCH, new_callable=AsyncMock):
        result = await critic_gate("git_commit", {"x": 1})
    # The worker should NEVER be called when opted out.
    husam_run.assert_not_awaited()
    assert result["verdict"] == "pass"
    assert result["bypassed"] is True


@pytest.mark.asyncio
async def test_critic_gate_dispatcher_failure_default_passes(monkeypatch):
    """When the critic producer call itself raises, default-pass — never block."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    husam_run = AsyncMock(side_effect=RuntimeError("model down"))
    with patch("husam.run", husam_run), patch(_PERSIST_PATCH, new_callable=AsyncMock):
        result = await critic_gate("git_commit", {"x": 1})
    assert result["verdict"] == "pass"
    assert any("model down" in r for r in result["reasons"])


@pytest.mark.asyncio
async def test_critic_gate_redacts_payload_before_sending(monkeypatch):
    """Secrets must not appear in the prompt sent to the critic LLM."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    captured = {}

    async def _capture(spec):
        captured["spec"] = spec
        return {"content": '{"verdict": "pass", "reasons": []}'}

    with patch("husam.run", _capture), patch(_PERSIST_PATCH, new_callable=AsyncMock):
        await critic_gate(
            "notify_user",
            {"message": "your api_key=sk-1234567890abcdef1234567890abcdef"},
        )
    sent = str(captured["spec"]["context"]["llm_call"]["messages"])
    assert "sk-1234567890abcdef1234567890abcdef" not in sent
    assert "REDACTED" in sent


# ─── Router wiring: git_commit post-hook ─────────────────────────────────


@pytest.mark.asyncio
async def test_router_git_commit_pass1_parks_no_commit(monkeypatch):
    """SP6 two-pass: first entry (no persisted verdict) enqueues an admitted
    critic child + parks the gated task; auto_commit must NOT run yet."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    task = {
        "id": 5,
        "mission_id": 1,
        "title": "x",
        "context": "{}",
        "payload": {"action": "git_commit"},
    }
    enq = AsyncMock(return_value=1234)
    upd = AsyncMock()
    with patch("mr_roboto.enqueue", enq), patch(
        "mr_roboto.update_task", upd
    ), patch(
        "mr_roboto.auto_commit",
        new_callable=AsyncMock,
        return_value={"committed": True, "empty": False, "message": "x"},
    ) as mock_commit, patch(
        "src.tools.git_ops._run_git",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ), patch(
        "src.tools.git_ops.ensure_git_repo", new_callable=AsyncMock
    ), patch(
        "src.tools.git_ops._resolve_repo", return_value="/tmp/repo"
    ):
        action = await mr_roboto.run(task)
    assert action.status == "needs_clarification"
    enq.assert_awaited_once()
    assert any(
        c.kwargs.get("status") == "waiting_human" for c in upd.await_args_list
    )
    mock_commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_git_commit_pass2_veto_aborts(monkeypatch):
    """SP6 two-pass: pass-2 (persisted veto in context) → LLM-free confirm_gate
    blocks; auto_commit NEVER called, stage is reset."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    import json as _json
    # SP6 T3: empty payload_hash now fail-closes into re-gate; to exercise the
    # confirm (veto-honoured) path this test intends, stub _hash_payload to a
    # fixed value and give the context a matching anchor so the drift guard
    # passes.
    monkeypatch.setattr("mr_roboto.critic_gate._hash_payload", lambda p: "MATCH")
    ctx = {
        "critic_verdict": {
            "verdict": "veto",
            "reasons": ["commit message contains a leaked secret"],
            "payload_hash": "MATCH",
        }
    }
    task = {
        "id": 5,
        "mission_id": 1,
        "title": "leak token",
        "context": _json.dumps(ctx),
        "payload": {"action": "git_commit"},
    }
    with patch(
        "mr_roboto.auto_commit", new_callable=AsyncMock
    ) as mock_commit, patch(
        "src.tools.git_ops._run_git",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_run_git, patch(
        "src.tools.git_ops.ensure_git_repo", new_callable=AsyncMock
    ), patch(
        "src.tools.git_ops._resolve_repo", return_value="/tmp/repo"
    ):
        action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert "veto" in (action.error or "")
    mock_commit.assert_not_awaited()
    # `git reset` must have been issued to unstage
    reset_calls = [
        c for c in mock_run_git.await_args_list if "reset" in c.args[0]
    ]
    assert reset_calls, "expected `git reset` rollback after veto"


@pytest.mark.asyncio
async def test_router_git_commit_critic_opt_out_skips_gate(monkeypatch):
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    task = {
        "id": 5,
        "mission_id": 1,
        "title": "x",
        "payload": {"action": "git_commit"},
    }
    with patch(
        "mr_roboto.critic_gate.critic_gate",
        new_callable=AsyncMock,
    ) as mock_gate, patch(
        "mr_roboto.auto_commit",
        new_callable=AsyncMock,
        return_value={"committed": True, "empty": False, "message": "x"},
    ):
        action = await mr_roboto.run(task)
    assert action.status == "completed"
    mock_gate.assert_not_awaited()


# ─── Router wiring: notify_user post-hook ────────────────────────────────


@pytest.mark.asyncio
async def test_router_notify_user_critic_pass_sends(monkeypatch):
    # SP6 T4: notify_user is now a TWO-PASS self-park gated ONLY when
    # mission-scoped. A mission task with a persisted PASS verdict (pass 2)
    # → LLM-free confirm_gate passes → message sent.
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    import json as _json
    ctx = {"critic_verdict": {"verdict": "pass", "reasons": [], "payload_hash": ""}}
    task = {
        "id": 7,
        "mission_id": 1,
        "context": _json.dumps(ctx),
        "payload": {
            "action": "notify_user",
            "chat_id": 222,
            "text": "Mission done",
        },
    }
    fake_tg = AsyncMock()
    with patch(
        "mr_roboto.notify_user.get_telegram", return_value=fake_tg
    ):
        action = await mr_roboto.run(task)
    assert action.status == "completed"
    fake_tg.app.bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_router_notify_user_critic_veto_drops(monkeypatch):
    # SP6 T4: pass-2 with a persisted VETO verdict on a mission-scoped task →
    # LLM-free confirm_gate vetoes → failed, message NOT sent. (No drift guard
    # / no `git reset` — notify_user stages nothing; it just doesn't send.)
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    import json as _json
    ctx = {"critic_verdict": {"verdict": "veto",
                              "reasons": ["leaks credential"],
                              "payload_hash": ""}}
    task = {
        "id": 7,
        "mission_id": 1,
        "context": _json.dumps(ctx),
        "payload": {
            "action": "notify_user",
            "chat_id": 222,
            "text": "your secret is sk-xyz",
        },
    }
    fake_tg = AsyncMock()
    with patch(
        "mr_roboto.notify_user.get_telegram", return_value=fake_tg
    ):
        action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert "veto" in (action.error or "")
    # Message must NOT have been sent
    fake_tg.app.bot.send_message.assert_not_awaited()
    fake_tg.send_notification.assert_not_awaited()


# ─── Standalone critic_gate action ───────────────────────────────────────


@pytest.mark.asyncio
async def test_router_standalone_critic_gate_pass(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    task = {
        "id": 1,
        "mission_id": 9,
        "payload": {
            "action": "critic_gate",
            "action_name": "deploy",
            "target_payload": {"env": "production"},
        },
    }
    with patch(
        "mr_roboto.critic_gate.critic_gate",
        new_callable=AsyncMock,
        return_value={"verdict": "pass", "reasons": []},
    ):
        action = await mr_roboto.run(task)
    assert action.status == "completed"


@pytest.mark.asyncio
async def test_router_standalone_critic_gate_veto(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    task = {
        "id": 1,
        "mission_id": 9,
        "payload": {
            "action": "critic_gate",
            "action_name": "deploy",
            "target_payload": {"env": "production", "force": True},
        },
    }
    with patch(
        "mr_roboto.critic_gate.critic_gate",
        new_callable=AsyncMock,
        return_value={"verdict": "veto", "reasons": ["force deploy"]},
    ):
        action = await mr_roboto.run(task)
    assert action.status == "failed"
