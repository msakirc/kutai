# tests/integration/test_image_i2p_swap_e2e.py
"""Plan 3 — end-to-end placeholder swap, CPS drive (SP5: await_inline gone).

Drives the host path: ``mr_roboto.run(task)`` with
``action=swap_placeholder_images`` against a temp ``.web/`` tree (including a
subdirectory screen). The kickoff enqueues a prompt_writer child with
on_complete/on_error continuations and returns immediately; this test then
plays Beckman's terminal-fire role — it pops each captured enqueue, fabricates
the child result (writing a real PNG for image children), and invokes the
registered continuation handler by NAME via the beckman registry (so the
registration wiring is exercised, not just the functions). End state must
match the old blocking e2e expectations: HTML rewritten to subdir-correct
relative refs, PNGs renamed to stable ``<pid>.png``, graceful degrade
preserved. Also asserts the verify mechanic accepts the MID-FLIGHT kickoff
shape (surviving placehold.co is acceptable while the chain runs) and still
fails on a legacy inconsistent result.
"""
import json
import os

import pytest
from PIL import Image


_HTML_HOME = """<!DOCTYPE html>
<html><body class="w-[390px] min-h-[844px]">
  <img src="https://placehold.co/390x220/E07A5F/FFF?text=hero"
       alt="smiling barista handing over a takeaway cup">
  <img src="https://placehold.co/260x180/3D405B/FFF?text=feat"
       alt="ai-powered task triage dashboard">
  <img src="/assets/already_real.png" alt="something already swapped">
</body></html>
"""

# Subdir screen — exercises the recursive os.walk (multi-screen prototypes).
_HTML_SCREEN = """<!DOCTYPE html>
<html><body>
  <img src="https://placehold.co/64x64/264653/FFF?text=u" alt="user portrait">
</body></html>
"""


@pytest.mark.asyncio
async def test_i2p_swap_e2e_cps_chain(monkeypatch, tmp_path, temp_db):
    """Full CPS chain: kickoff → prompts_done → image_done ×3 → finalize.

    Uses the ``temp_db`` fixture so the audit-log path
    (mr_roboto.run → record_action_event → src/infra/db.py) hits an isolated,
    schema-initialised SQLite file rather than crashing on a missing
    registry_events table against the live DB."""
    ws = tmp_path / "mission_777"
    web = ws / ".web"
    (web / "screens").mkdir(parents=True)
    (web / "home.html").write_text(_HTML_HOME, encoding="utf-8")
    (web / "screens" / "onboarding.html").write_text(
        _HTML_SCREEN, encoding="utf-8"
    )
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(ws),
    )

    # placeholder_ids are slug-derived: <html-stem>__<occurrence>.
    # home.html → home__0 (hero), home__1 (feat); /assets/already_real.png
    # is NOT a placeholder and is skipped by the scanner.
    # onboarding.html → onboarding__0 (user portrait).
    prompt_envelope = {
        "_schema_version": "1",
        "prompts": [
            {"placeholder_id": "home__0", "prompt": "coral barista"},
            {"placeholder_id": "home__1", "prompt": "slate dashboard"},
            {"placeholder_id": "onboarding__0", "prompt": "teal portrait"},
        ],
    }

    queued: list[dict] = []
    next_id = {"n": 1000}

    async def _fake_enqueue(spec, **kwargs):
        assert "await_inline" not in kwargs, "CPS regression: await_inline used"
        assert kwargs.get("on_complete"), "child enqueued without continuation"
        assert kwargs.get("on_error"), "child enqueued without on_error"
        next_id["n"] += 1
        queued.append({"spec": spec, "kwargs": kwargs, "id": next_id["n"]})
        return next_id["n"]

    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fake_enqueue,
    )

    import mr_roboto
    from general_beckman.continuations import _HANDLERS

    task = {
        "id": 12345, "mission_id": 777, "title": "swap_e2e",
        "context": {"payload": {
            "action": "swap_placeholder_images",
            "design_tokens": {"primary": "#E07A5F"},
            "brand_voice": "warm, neighborhood coffee shop",
        }},
    }
    action = await mr_roboto.run(task)

    # ── kickoff: immediate Action-compatible completed result ────────────
    assert action.status == "completed"
    res = action.result
    assert res["ok"] is True
    assert res["queued"] is True
    assert res["chain"] == "started"
    assert res["placeholder_count"] == 3
    assert res["html_files_seen"] == 2

    # 5.35 threads its own task id → prompt_writer child gets parent_id.
    assert queued[0]["kwargs"]["parent_id"] == 12345

    # ── MID-FLIGHT: the verify mechanic accepts the kickoff shape even
    # though every <img> still points at placehold.co (chain in flight). ──
    from mr_roboto.verify_swap_placeholder_images_shape import (
        verify_swap_placeholder_images_shape,
    )
    verdict = verify_swap_placeholder_images_shape(
        workspace_path=str(ws), swap_result=json.dumps(res),
    )
    assert verdict["ok"] is True
    assert verdict["surviving_placeholders"] == 3

    # ── play Beckman: fire each child's continuation by NAME ─────────────
    call_log: list[str] = []
    guard = 0
    while queued:
        guard += 1
        assert guard < 20, "chain did not terminate"
        child = queued.pop(0)
        spec, kwargs = child["spec"], child["kwargs"]
        agent_type = spec.get("agent_type")
        call_log.append(agent_type or "")
        # cont_state survives a DB JSON round-trip — simulate it.
        state = json.loads(json.dumps(kwargs["cont_state"]))
        handler = _HANDLERS[kwargs["on_complete"]]

        if agent_type == "prompt_writer":
            # The continuation fires at TRUE terminal — post constrained_emit
            # repair — with the persisted result (JSON-string body tolerated).
            await handler(child["id"],
                          {"result": json.dumps(prompt_envelope)}, state)
            continue
        if agent_type == "image":
            ic = spec["context"]["image_call"]
            assert spec["mission_id"] == 777
            os.makedirs(ic["out_dir"], exist_ok=True)
            # paintress writes a timestamp-suffixed file; simulate that.
            path = os.path.join(
                ic["out_dir"], f"{ic['filename_hint']}_raw.png",
            )
            Image.new(
                "RGB", (ic["width"], ic["height"]), (100, 150, 200)
            ).save(path, "PNG")
            # The image lane returns {content, path, provider, ...}.
            await handler(child["id"], {
                "content": path, "path": path, "provider": "pollinations",
                "model": "pollinations/flux", "cost": 0.0,
            }, state)
            continue
        raise AssertionError(f"unexpected agent_type: {agent_type!r}")

    # Sequential chain: 1 prompt_writer then 3 images, one at a time.
    assert call_log == ["prompt_writer", "image", "image", "image"]

    # ── end state: identical to the old blocking e2e expectations ────────
    assets = ws / ".web" / "assets"
    pngs = sorted(p.name for p in assets.glob("*.png"))
    assert pngs == ["home__0.png", "home__1.png", "onboarding__0.png"]
    for png in pngs:
        assert (assets / png).stat().st_size > 0

    home = (web / "home.html").read_text(encoding="utf-8")
    assert "placehold.co" not in home
    # Root HTML: relpath from .web/ to .web/assets/ → "assets/<pid>.png".
    assert 'src="assets/home__0.png"' in home
    assert 'src="assets/home__1.png"' in home
    assert "/assets/already_real.png" in home  # untouched real src

    onboarding = (web / "screens" / "onboarding.html").read_text(
        encoding="utf-8"
    )
    assert "placehold.co" not in onboarding
    # Subdir screen: relpath from .web/screens/ to .web/assets/ →
    # "../assets/<pid>.png" — resolves correctly in a static file server,
    # whereas a flat "assets/<pid>.png" would 404 (→ .web/screens/assets/...).
    assert 'src="../assets/onboarding__0.png"' in onboarding
    assert 'src="assets/onboarding__0.png"' not in onboarding

    # Chain ledger finalized: full success, deep shape check passed. The
    # ledger lives OUTSIDE the served .web root (never published/tunneled).
    assert not (web / ".swap_chain.json").exists()
    ledger = json.loads(
        (ws / ".swap_state" / "swap_chain.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "done"
    assert ledger["replaced"] == 3
    assert ledger["skipped"] == 0
    assert ledger["errors"] == []
    assert ledger["shape_check"]["ok"] is True

    # POST-CHAIN: the verify mechanic's live shape (empty swap_result) is
    # meaningful — all rewritten refs exist on disk.
    verdict = verify_swap_placeholder_images_shape(
        workspace_path=str(ws), swap_result={},
    )
    assert verdict["ok"] is True
    assert verdict["surviving_placeholders"] == 0
    assert verdict["broken_asset_refs"] == []


@pytest.mark.asyncio
async def test_verify_fails_on_inconsistent_result(monkeypatch, tmp_path):
    """The verify mechanic FAILS when a (legacy, chain-less) swap_result
    claims everything replaced but a placehold.co URL still survives in the
    HTML and errors is empty — i.e. an internally inconsistent result."""
    ws = tmp_path / "mission_888"
    web = ws / ".web"
    web.mkdir(parents=True)
    # One real swap + one surviving placehold.co, but swap_result lies that all
    # three were replaced with zero skips / zero errors.
    (web / "home.html").write_text(
        '<html><body>'
        '<img src="assets/home__0.png" alt="hero">'
        '<img src="https://placehold.co/260x180/3D405B/FFF?text=feat" alt="feat">'
        '</body></html>',
        encoding="utf-8",
    )
    (web / "assets").mkdir()
    (web / "assets" / "home__0.png").write_bytes(b"\x89PNG\r\n\x1a\nFAKE")

    from mr_roboto.verify_swap_placeholder_images_shape import (
        verify_swap_placeholder_images_shape,
    )
    verdict = verify_swap_placeholder_images_shape(
        workspace_path=str(ws),
        swap_result={"replaced_count": 2, "skipped_count": 0, "errors": []},
    )
    assert verdict["ok"] is False
    assert "inconsistent" in (verdict.get("error") or "").lower()
    assert verdict["surviving_placeholders"] == 1
