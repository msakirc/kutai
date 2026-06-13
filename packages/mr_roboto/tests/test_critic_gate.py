"""Tests for the B4 Critic gate."""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, patch

import mr_roboto
import mr_roboto.critic_gate as CRITIC_MODULE
from mr_roboto.critic_gate import (
    _redact,
    _redact_payload,
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


# ─── Unit: parse_verdict_strict (SP6 gate-side, FAIL-CLOSED) ──────────────
# Surface A (and Tasks 3-4 surface B) stamp the verdict via this helper.
# Garbage/empty/non-enum MUST veto (the producer-side _parse_verdict was
# deleted in SP6 T5; parse_verdict_strict is the only verdict parser now).


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


# ─── Unit: _build_critic_spec redaction ──────────────────────────────────


def test_build_critic_spec_redacts_payload():
    """Secrets must not appear in the admitted critic-child spec messages."""
    spec = CRITIC_MODULE._build_critic_spec(
        "notify_user",
        _redact_payload(
            {"message": "your api_key=sk-1234567890abcdef1234567890abcdef"}
        ),
    )
    sent = str(spec["context"]["llm_call"]["messages"])
    assert "sk-1234567890abcdef1234567890abcdef" not in sent
    assert "REDACTED" in sent
    assert spec["context"]["llm_call"]["raw_dispatch"] is True
    assert spec["context"]["llm_call"]["call_category"] == "overhead"


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
    # Opt-out (gate disabled): the executor must NOT enqueue an admitted critic
    # child nor run confirm_gate — it commits directly.
    enq = AsyncMock(return_value=1234)
    confirm = AsyncMock()
    with patch("mr_roboto.enqueue", enq), patch(
        "mr_roboto.critic_gate.confirm_gate", confirm
    ), patch(
        "mr_roboto.auto_commit",
        new_callable=AsyncMock,
        return_value={"committed": True, "empty": False, "message": "x"},
    ):
        action = await mr_roboto.run(task)
    assert action.status == "completed"
    enq.assert_not_awaited()
    confirm.assert_not_awaited()


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
# Deleted in SP6 T5: the standalone ``action == "critic_gate"`` mechanical
# executor is gone — critic_gate is now an admitted posthook LLM child (T2).
# No standalone-action tests remain.
