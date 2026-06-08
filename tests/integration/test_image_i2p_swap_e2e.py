# tests/integration/test_image_i2p_swap_e2e.py
"""Plan 3 v2 — end-to-end placeholder swap with PRODUCTION TaskResult shape.

Drives the host path: ``mr_roboto.run(task)`` with
``action=swap_placeholder_images`` against a temp ``.web/`` tree (including a
subdirectory screen), with beckman.enqueue mocked at the swap module's
namespace so the production JSON-STRING ``TaskResult.result`` shape is
exercised (orchestrator ``json.dumps``). Image generation is mocked (no real
network / GPU). Also asserts the verify mechanic passes on the consistent
result and fails on an inconsistent one.
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
async def test_i2p_swap_e2e_with_json_string_result(monkeypatch, tmp_path):
    """Mocks beckman.enqueue with the production JSON-string TaskResult.result
    shape and asserts the full host path: HTML rewritten to assets/<pid>.png,
    PNGs written, recursive subdir screen handled, and the verify mechanic
    accepts the consistent result."""
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

    call_log: list[str] = []

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

    async def _fake_enqueue(spec, **kwargs):
        agent_type = spec.get("agent_type")
        call_log.append(agent_type or "")
        if agent_type == "prompt_writer":
            assert kwargs.get("await_inline") is True
            class _R:
                status = "completed"
                # PRODUCTION SHAPE — JSON STRING (orchestrator json.dumps).
                result = json.dumps(prompt_envelope)
                error = None
            return _R()
        if agent_type == "image":
            ic = spec["context"]["image_call"]
            os.makedirs(ic["out_dir"], exist_ok=True)
            # paintress writes a timestamp-suffixed file; simulate that exactly.
            path = os.path.join(
                ic["out_dir"], f"{ic['filename_hint']}_raw.png",
            )
            Image.new(
                "RGB", (ic["width"], ic["height"]), (100, 150, 200)
            ).save(path, "PNG")
            class _R:
                status = "completed"
                # PRODUCTION SHAPE — JSON STRING.
                result = json.dumps({
                    "path": path, "provider": "pollinations",
                    "model": "pollinations/flux", "cost": 0.0,
                })
                error = None
            return _R()
        raise AssertionError(f"unexpected agent_type: {agent_type!r}")

    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fake_enqueue,
    )

    import mr_roboto

    task = {
        "id": 12345, "mission_id": 777, "title": "swap_e2e",
        "context": {"payload": {
            "action": "swap_placeholder_images",
            "design_tokens": {"primary": "#E07A5F"},
            "brand_voice": "warm, neighborhood coffee shop",
        }},
    }
    action = await mr_roboto.run(task)

    assert action.status == "completed"
    res = action.result
    assert res["ok"] is True
    assert res["replaced_count"] == 3
    assert res["skipped_count"] == 0
    assert res["html_files_seen"] == 2
    assert res["html_files_changed"] == 2

    assets = ws / ".web" / "assets"
    pngs = sorted(p.name for p in assets.glob("*.png"))
    # Stable <pid>.png names (no timestamp suffix after rename).
    assert pngs == ["home__0.png", "home__1.png", "onboarding__0.png"]
    for png in pngs:
        assert (assets / png).stat().st_size > 0

    home = (web / "home.html").read_text(encoding="utf-8")
    assert "placehold.co" not in home
    assert 'src="assets/home__0.png"' in home
    assert 'src="assets/home__1.png"' in home
    assert "/assets/already_real.png" in home  # untouched real src

    onboarding = (web / "screens" / "onboarding.html").read_text(
        encoding="utf-8"
    )
    assert "placehold.co" not in onboarding
    assert 'src="assets/onboarding__0.png"' in onboarding

    assert call_log.count("prompt_writer") == 1
    assert call_log.count("image") == 3

    # The verify mechanic accepts this consistent result (0 surviving == 0
    # skipped, assets/ present).
    from mr_roboto.verify_swap_placeholder_images_shape import (
        verify_swap_placeholder_images_shape,
    )
    verdict = verify_swap_placeholder_images_shape(
        workspace_path=str(ws), swap_result=res,
    )
    assert verdict["ok"] is True
    assert verdict["surviving_placeholders"] == 0


@pytest.mark.asyncio
async def test_verify_fails_on_inconsistent_result(monkeypatch, tmp_path):
    """The verify mechanic FAILS when swap_result claims everything replaced
    but a placehold.co URL still survives in the HTML and errors is empty —
    i.e. an internally inconsistent result."""
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
