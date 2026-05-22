"""Z1 — spec-patch review+apply loop: SURFACE side.

After ``propose_spec_patch_from_html_diff`` writes its proposal markdown,
the dispatch branch must surface it back into Telegram as a ``notify_user``
follow-up task carrying Apply/Reject inline buttons. The callback tokens
are ``sp_apply:<mid>:<ts>`` / ``sp_rej:<mid>:<ts>`` (mission_id + integer
ts) — kept short so they stay under Telegram's 64-byte callback_data cap.

These tests drive the dispatch branch (and the ``_surface_spec_patch_proposal``
helper directly) and assert the notify task is enqueued with the right
buttons. ``general_beckman.enqueue`` is monkeypatched to capture the spec,
so no DB / live Telegram is touched.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest


_ORIGINAL = """<!DOCTYPE html>
<html><body>
  <header data-oid="screen_5_3:header"><h1>Welcome to KutAI</h1></header>
  <main data-oid="screen_5_3:main">
    <section data-oid="screen_5_3:hero" style="background:#0066ff">
      <h2>Sign in</h2>
      <p>Use your email and password.</p>
    </section>
  </main>
</body></html>
"""

# Color shift + copy change in the hero → at least one change.
_EDITED = """<!DOCTYPE html>
<html><body>
  <header data-oid="screen_5_3:header"><h1>Welcome to KutAI</h1></header>
  <main data-oid="screen_5_3:main">
    <section data-oid="screen_5_3:hero" style="background:#00cc99">
      <h2>Sign in</h2>
      <p>Use email or social login.</p>
    </section>
  </main>
</body></html>
"""


def _install_fake_beckman(monkeypatch):
    """Install a stub ``general_beckman`` module whose ``enqueue`` records
    every spec it receives. Returns the capture list.

    mr_roboto does ``import general_beckman`` lazily inside the branch, so a
    module object in ``sys.modules`` is sufficient — no real package needed.
    """
    captured: list[dict] = []

    async def _fake_enqueue(spec, *args, **kwargs):
        captured.append(spec)
        return {"task_id": len(captured)}

    fake = types.ModuleType("general_beckman")
    fake.enqueue = _fake_enqueue  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "general_beckman", fake)
    return captured


def _notify_specs(captured):
    return [
        s for s in captured
        if (((s.get("context") or {}).get("payload") or {}).get("action")
            == "notify_user")
    ]


def test_surface_helper_enqueues_notify_with_apply_and_reject(monkeypatch):
    """The helper enqueues a notify_user task whose inline_buttons carry
    both sp_apply and sp_rej tokens, each ≤64 bytes."""
    from mr_roboto import _surface_spec_patch_proposal

    captured = _install_fake_beckman(monkeypatch)

    payload = {"mission_id": 42, "ts": 1234}
    res = {
        "ok": True,
        "changes": [{"data_oid": "screen_5_3:hero", "kinds": ["color"],
                     "detail": {}, "suggested_target": "design_tokens"}],
        "missing_oids": [],
        "proposal_md": "# Spec Patch Proposal\n\nbody body body\n",
        "proposal_path": "mission_42/.propagation/spec_patch_proposal_1234.md",
    }

    asyncio.run(_surface_spec_patch_proposal(payload, res))

    notifies = _notify_specs(captured)
    assert len(notifies) == 1, captured
    pl = notifies[0]["context"]["payload"]
    buttons = pl["inline_buttons"]
    cbs = {b["callback_data"] for b in buttons}
    assert "sp_apply:42:1234" in cbs
    assert "sp_rej:42:1234" in cbs
    for b in buttons:
        assert len(b["callback_data"].encode("utf-8")) <= 64, b
    # message carries the proposal body + a header mentioning the mission.
    assert "42" in pl["message"]
    assert "Spec Patch Proposal" in pl["message"]


def test_surface_helper_zero_changes_has_no_apply_button(monkeypatch):
    """With no changes there is nothing to apply — notify still fires but
    carries no Apply button."""
    from mr_roboto import _surface_spec_patch_proposal

    captured = _install_fake_beckman(monkeypatch)

    payload = {"mission_id": 7, "ts": 999}
    res = {
        "ok": True,
        "changes": [],
        "missing_oids": [],
        "proposal_md": "# Spec Patch Proposal\n\n_No changes._\n",
        "proposal_path": "mission_7/.propagation/spec_patch_proposal_999.md",
    }

    asyncio.run(_surface_spec_patch_proposal(payload, res))

    notifies = _notify_specs(captured)
    assert len(notifies) == 1
    buttons = notifies[0]["context"]["payload"].get("inline_buttons") or []
    cbs = {b["callback_data"] for b in buttons}
    assert "sp_apply:7:999" not in cbs
    # A reject button with nothing to apply is pointless — neither token.
    assert all(not c.startswith("sp_apply:") for c in cbs)


def test_surface_helper_missing_ids_skips_notify(monkeypatch):
    """Missing mid/ts → skip the notify enqueue gracefully (no crash)."""
    from mr_roboto import _surface_spec_patch_proposal

    captured = _install_fake_beckman(monkeypatch)
    res = {"ok": True, "changes": [{"data_oid": "x", "kinds": ["copy"],
                                    "detail": {}, "suggested_target": "plan"}],
           "missing_oids": [], "proposal_md": "# P\n", "proposal_path": "p.md"}

    # No mission_id / ts in payload.
    asyncio.run(_surface_spec_patch_proposal({}, res))
    assert _notify_specs(captured) == []


def test_surface_message_truncated_to_telegram_headroom(monkeypatch):
    """A huge proposal body must be truncated well under the 4096 limit."""
    from mr_roboto import _surface_spec_patch_proposal

    captured = _install_fake_beckman(monkeypatch)
    big = "x" * 10000
    payload = {"mission_id": 1, "ts": 5}
    res = {"ok": True,
           "changes": [{"data_oid": "a", "kinds": ["copy"], "detail": {},
                        "suggested_target": "plan"}],
           "missing_oids": [], "proposal_md": big, "proposal_path": "p.md"}

    asyncio.run(_surface_spec_patch_proposal(payload, res))
    msg = _notify_specs(captured)[0]["context"]["payload"]["message"]
    assert len(msg) <= 3600  # header + ≤3500 body + ellipsis headroom


def test_dispatch_branch_surfaces_after_proposer(tmp_path: Path, monkeypatch):
    """End-to-end through ``mr_roboto.run``: a real HTML diff with changes
    runs the proposer AND enqueues a notify_user with Apply/Reject."""
    from mr_roboto import run as mr_run

    captured = _install_fake_beckman(monkeypatch)

    orig = tmp_path / "o.html"
    edit = tmp_path / "e.html"
    out = tmp_path / "p.md"
    orig.write_text(_ORIGINAL, encoding="utf-8")
    edit.write_text(_EDITED, encoding="utf-8")

    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "propose_spec_patch_from_html_diff",
            "html_path": str(orig),
            "edited_html_path": str(edit),
            "out_path": str(out),
            "mission_id": 42,
            "ts": 1234,
        },
    }
    res = asyncio.run(mr_run(task))
    # The proposer result is STILL the Action result — notify is a side-effect.
    assert res.status == "completed", res.error
    assert res.result["ok"] is True
    assert out.exists()

    notifies = _notify_specs(captured)
    assert len(notifies) == 1, captured
    cbs = {b["callback_data"]
           for b in notifies[0]["context"]["payload"]["inline_buttons"]}
    assert "sp_apply:42:1234" in cbs
    assert "sp_rej:42:1234" in cbs
