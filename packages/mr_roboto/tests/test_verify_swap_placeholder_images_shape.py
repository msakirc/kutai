import json

from mr_roboto.verify_swap_placeholder_images_shape import (
    verify_swap_placeholder_images_shape,
)


_HTML_REWRITTEN = """<!DOCTYPE html>
<html><body>
  <img src="assets/home__0.png" alt="hero">
  <img src="assets/home__1.png" alt="feat">
  <img src="assets/home__2.png" alt="user">
</body></html>"""

_HTML_PARTIAL = """<!DOCTYPE html>
<html><body>
  <img src="assets/home__0.png" alt="hero">
  <img src="https://placehold.co/260x180/3D405B/FFF?text=feat" alt="feat">
  <img src="assets/home__2.png" alt="user">
</body></html>"""


def test_passes_when_all_placeholders_replaced(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__1.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 3, "skipped_count": 0, "errors": []},
    )
    assert res["ok"] is True


def test_passes_when_skipped_matches_surviving_placeholders(tmp_path):
    """Graceful degrade: 1 placeholder skipped → 1 placehold.co survives in
    HTML. That matches; verifier accepts."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PARTIAL, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 2, "skipped_count": 1,
                     "errors": ["image gen failed for home__1"]},
    )
    assert res["ok"] is True


def test_fails_when_replaced_count_disagrees_with_html(tmp_path):
    """If swap_result claims 3 replaced but 1 placehold.co survives and
    errors is empty, the result is internally inconsistent — fail.

    The rewritten refs in _HTML_PARTIAL must exist on disk so the layer-1
    broken-asset-ref check passes cleanly and the layer-2 consistency check
    is the thing under test."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PARTIAL, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 3, "skipped_count": 0, "errors": []},
    )
    assert res["ok"] is False
    assert "inconsistent" in (res.get("error") or "").lower()


def test_fails_when_assets_dir_missing_but_replaced_count_positive(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 3, "skipped_count": 0, "errors": []},
    )
    assert res["ok"] is False
    assert "assets" in (res.get("error") or "").lower()


def test_passes_when_swap_skipped_entirely(tmp_path):
    """Swap reported 0 replaced + 0 skipped (no placeholders existed) →
    verifier passes (no work expected)."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text("<html><body>no img</body></html>",
                                   encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 0, "skipped_count": 0, "errors": []},
    )
    assert res["ok"] is True


def test_accepts_json_string_swap_result(tmp_path):
    """PRODUCTION SHAPE: the swap step result arrives as a JSON STRING.
    The verifier must json.loads it before reading fields, otherwise every
    field reads 0 and a fully-replaced prototype is flagged inconsistent."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__1.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    swap_result_str = json.dumps(
        {"replaced_count": 3, "skipped_count": 0, "errors": []}
    )
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result=swap_result_str,
    )
    assert res["ok"] is True
    assert res["expected_replaced"] == 3


def test_garbage_json_string_coerces_to_empty(tmp_path):
    """A non-JSON string degrades to {} (replaced=0, skipped=0); a clean
    prototype whose rewritten refs all exist on disk still passes (layer-2
    consistency is skipped for the empty result; layer-1 finds no broken
    refs)."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__1.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result="not json {",
    )
    assert res["ok"] is True
    assert res["surviving_placeholders"] == 0


def test_live_empty_swap_result_passes_when_rewritten_assets_exist(tmp_path):
    """LIVE i2p case: swap_result is empty (no cross-step injection). A
    prototype whose rewritten asset refs all exist on disk passes — and the
    gate is MEANINGFUL (it actually walked the workspace and checked the
    refs), not vacuous."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__1.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={},
    )
    assert res["ok"] is True
    assert res["broken_asset_refs"] == []


def test_live_empty_swap_result_fails_on_broken_asset_ref(tmp_path):
    """LIVE i2p case: a rewritten ref points at a file that does NOT exist on
    disk (the real corruption mode). With an empty swap_result the gate must
    still FAIL — proving the gate is not vacuous in live."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    (web / "assets").mkdir()
    # Only 2 of 3 referenced assets exist; home__1.png is missing.
    for n in ("home__0.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={},
    )
    assert res["ok"] is False
    assert "broken asset ref" in (res.get("error") or "").lower()
    assert "assets/home__1.png" in res["broken_asset_refs"]


def test_root_relative_ref_not_flagged_broken(tmp_path):
    """A root-relative ("/assets/x.png") ref is ROOT-ANCHORED, not a
    locally-rewritten relative asset ref. The executor leaves such pre-existing
    refs untouched, so the verifier must NOT resolve it against the HTML dir
    (which would yield a bogus path) — even when the file is absent on disk it
    must NOT be flagged broken."""
    html = (
        "<!DOCTYPE html><html><body>"
        '<img src="/assets/already_real.png" alt="real">'
        "</body></html>"
    )
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(html, encoding="utf-8")
    # Note: NO file written at /assets/already_real.png anywhere.
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={},
    )
    assert res["ok"] is True
    assert res["broken_asset_refs"] == []


def test_subdir_dotdot_ref_resolves_correctly(tmp_path):
    """After Fix 1 the executor emits "../assets/<pid>.png" for subdir
    screens. The verifier resolves each ref against the HTML file's own dir,
    so .web/screens/onboarding.html + "../assets/x.png" → .web/assets/x.png,
    which exists → NOT broken. Confirms verifier agrees with the executor for
    subdir HTML."""
    web = tmp_path / ".web"
    (web / "screens").mkdir(parents=True)
    (web / "screens" / "onboarding.html").write_text(
        '<html><body><img src="../assets/onboarding__0.png" alt="u"></body></html>',
        encoding="utf-8",
    )
    (web / "assets").mkdir()
    (web / "assets" / "onboarding__0.png").write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={},
    )
    assert res["ok"] is True
    assert res["broken_asset_refs"] == []


def test_subdir_flat_ref_is_flagged_broken(tmp_path):
    """Regression lock for the original bug: a subdir screen with the OLD flat
    "assets/<pid>.png" ref resolves to .web/screens/assets/<pid>.png (missing)
    → broken. This is exactly what the live verify gate caught."""
    web = tmp_path / ".web"
    (web / "screens").mkdir(parents=True)
    (web / "screens" / "onboarding.html").write_text(
        '<html><body><img src="assets/onboarding__0.png" alt="u"></body></html>',
        encoding="utf-8",
    )
    (web / "assets").mkdir()
    # The asset exists in the FLAT dir, but the subdir-relative resolution
    # points at .web/screens/assets/... which does not exist.
    (web / "assets" / "onboarding__0.png").write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={},
    )
    assert res["ok"] is False
    assert "broken asset ref" in (res.get("error") or "").lower()


def test_live_empty_swap_result_passes_with_only_surviving_placeholders(tmp_path):
    """LIVE i2p graceful-degrade: nothing was rewritten (all images still
    point at placehold.co). With an empty swap_result the gate passes —
    surviving placeholders alone never fail the gate."""
    html = (
        "<!DOCTYPE html><html><body>"
        '<img src="https://placehold.co/600x400/000/FFF?text=hero" alt="h">'
        '<img src="https://placehold.co/260x180/3D405B/FFF?text=feat" alt="f">'
        "</body></html>"
    )
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(html, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={},
    )
    assert res["ok"] is True
    assert res["surviving_placeholders"] == 2
    assert res["broken_asset_refs"] == []
