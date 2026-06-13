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


# ── CPS kickoff shape (chain="started") ─────────────────────────────────

_HTML_PENDING = (
    "<!DOCTYPE html><html><body>"
    '<img src="https://placehold.co/600x400/000/FFF?text=hero" alt="h">'
    '<img src="https://placehold.co/260x180/3D405B/FFF?text=feat" alt="f">'
    "</body></html>"
)


def _write_ledger(tmp_path, *, n=2, status="prompts_pending",
                  shape_check=None, errors=None, replaced=None):
    ledger = {
        "mission_id": 1,
        "status": status,
        "placeholders": [
            {"placeholder_id": f"home__{i}", "tag_span": [0, 1],
             "html_path": "x"} for i in range(n)
        ],
        "prompt_map": {},
        "results": {},
    }
    if shape_check is not None:
        ledger["shape_check"] = shape_check
    if errors is not None:
        ledger["errors"] = errors
    if replaced is not None:
        ledger["replaced"] = replaced
    # Ledger lives OUTSIDE the served .web root (it carries prompts/paths/
    # exception strings; .web is tunnel-served + gh-pages-published).
    state_dir = tmp_path / ".swap_state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "swap_chain.json").write_text(
        json.dumps(ledger), encoding="utf-8",
    )


def test_chain_started_mid_flight_passes_despite_surviving_placeholders(tmp_path):
    """5.35.verify may run while the image chain is still in flight — the
    kickoff shape + a valid ledger pass even though every <img> still points
    at placehold.co."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(tmp_path, n=2, status="prompts_pending")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "queued": True, "chain": "started",
                     "placeholder_count": 2, "html_files_seen": 1},
    )
    assert res["ok"] is True
    assert res["surviving_placeholders"] == 2


def test_chain_started_accepts_done_ledger(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(tmp_path, n=2, status="done")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "chain": "started", "placeholder_count": 2},
    )
    assert res["ok"] is True


def test_chain_in_flight_accepted_like_started(tmp_path):
    """FIX 1.2d: a re-run kickoff that found the chain mid-flight returns
    chain='in_flight' (no overwrite, no duplicate enqueue). The verifier
    must validate it like 'started' — ledger exists, count matches, sane
    status — NOT fall through to the legacy surviving==skipped branch
    (which would fail every mid-flight re-run)."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(tmp_path, n=2, status="images_pending")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "chain": "in_flight",
                     "placeholder_count": 2, "html_files_seen": 1},
    )
    assert res["ok"] is True
    assert res["surviving_placeholders"] == 2


def test_chain_in_flight_fails_when_ledger_missing(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "chain": "in_flight", "placeholder_count": 2},
    )
    assert res["ok"] is False
    assert "ledger" in (res.get("error") or "").lower()


def test_chain_started_fails_when_ledger_missing(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "chain": "started", "placeholder_count": 2},
    )
    assert res["ok"] is False
    assert "ledger" in (res.get("error") or "").lower()


def test_chain_started_fails_on_placeholder_count_mismatch(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(tmp_path, n=3)
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "chain": "started", "placeholder_count": 2},
    )
    assert res["ok"] is False
    assert "placeholder_count" in (res.get("error") or "")


def test_chain_started_fails_on_bad_ledger_status(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(tmp_path, n=2, status="exploded")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "chain": "started", "placeholder_count": 2},
    )
    assert res["ok"] is False
    assert "status" in (res.get("error") or "").lower()


def test_chain_started_still_fails_on_broken_asset_ref(tmp_path):
    """Layer 1 outranks the chain shape: a rewritten ref pointing at a
    missing file fails even mid-flight."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(
        '<html><body><img src="assets/home__0.png" alt="h"></body></html>',
        encoding="utf-8",
    )
    _write_ledger(tmp_path, n=1, status="images_pending")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "chain": "started", "placeholder_count": 1},
    )
    assert res["ok"] is False
    assert "broken asset ref" in (res.get("error") or "").lower()


def test_producer_ok_false_fails(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text("<html><body></body></html>",
                                   encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": False, "chain": "started", "placeholder_count": 2},
    )
    assert res["ok"] is False
    assert "ok=false" in (res.get("error") or "").lower()


def test_chain_none_keeps_legacy_consistency_semantics(tmp_path):
    """Degrade kickoff (chain 'none', everything skipped): surviving ==
    skipped passes; nothing was rewritten."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"ok": True, "chain": "none", "replaced_count": 0,
                     "skipped_count": 2,
                     "errors": ["prompt_writer enqueue raised: x"]},
    )
    assert res["ok"] is True
    assert res["surviving_placeholders"] == 2


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


# ── FIX 4.1(b): layer-1 scoped to swap-written refs ─────────────────────

def test_agent_authored_missing_ref_is_warning_not_failure(tmp_path):
    """An agent-authored relative ref (img/logo.png) the swap never touched
    must NOT fail the gate even when missing on disk — it becomes a WARNING
    note in the result. Only swap-plausible refs (assets/<x>.png) hard-fail."""
    html = (
        "<!DOCTYPE html><html><body>"
        '<img src="img/logo.png" alt="logo">'
        "</body></html>"
    )
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(html, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is True
    assert res["broken_asset_refs"] == []
    assert any("img/logo.png" in w for w in res["warnings"])


def test_non_png_assets_ref_is_warning_not_failure(tmp_path):
    """assets/photo.jpg does not match the swap's <pid>.png pattern — the
    swap never writes non-PNG assets, so a missing one is agent-authored →
    warning, not failure."""
    html = (
        "<!DOCTYPE html><html><body>"
        '<img src="assets/photo.jpg" alt="photo">'
        "</body></html>"
    )
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(html, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is True
    assert res["broken_asset_refs"] == []
    assert any("assets/photo.jpg" in w for w in res["warnings"])


def test_agent_ref_existing_on_disk_produces_no_warning(tmp_path):
    """An agent-authored ref that EXISTS is fine — no warning noise."""
    html = (
        "<!DOCTYPE html><html><body>"
        '<img src="img/logo.png" alt="logo">'
        "</body></html>"
    )
    web = tmp_path / ".web"; web.mkdir()
    (web / "img").mkdir()
    (web / "img" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    (web / "home.html").write_text(html, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is True
    # No agent-ref warning (the live no-ledger note is unrelated).
    assert not any("img/logo.png" in w for w in res["warnings"])


def test_dotdot_assets_png_missing_still_fails(tmp_path):
    """../assets/<pid>.png is the swap's subdir-screen shape — missing on
    disk must still hard-fail layer 1."""
    web = tmp_path / ".web"
    (web / "screens").mkdir(parents=True)
    (web / "screens" / "onboarding.html").write_text(
        '<html><body><img src="../assets/onboarding__0.png" alt="u"></body></html>',
        encoding="utf-8",
    )
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is False
    assert "broken asset ref" in (res.get("error") or "").lower()


# ── FIX 4.1(a): live path (empty swap_result) reads the ledger itself ───

def test_live_no_ledger_passes_with_note(tmp_path):
    """Live + no ledger: the kickoff may have found nothing to swap — pass,
    but record a note so the verdict is auditable."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text("<html><body>no img</body></html>",
                                   encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is True
    assert any("ledger" in w.lower() for w in res["warnings"])


def test_live_ledger_mid_flight_passes(tmp_path):
    """Live + ledger mid-flight: surviving placehold.co URLs are expected —
    tolerant pass (the chain is still generating)."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(tmp_path, n=2, status="images_pending")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is True
    assert res["surviving_placeholders"] == 2


def test_live_ledger_bad_status_fails(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(tmp_path, n=2, status="exploded")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is False
    assert "status" in (res.get("error") or "").lower()


def test_live_ledger_no_placeholders_fails(tmp_path):
    """A ledger is only ever written with a non-empty placeholder list — an
    empty one is corrupt state."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(tmp_path, n=0, status="prompts_pending")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is False
    assert "placeholder" in (res.get("error") or "").lower()


def test_live_ledger_unreadable_fails(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    state_dir = tmp_path / ".swap_state"
    state_dir.mkdir()
    (state_dir / "swap_chain.json").write_text("not json {", encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is False
    assert "ledger" in (res.get("error") or "").lower()


def test_live_ledger_done_shape_check_ok_passes(tmp_path):
    """Live post-finalize: ledger done + recorded shape_check ok → pass."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__1.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    _write_ledger(
        tmp_path, n=3, status="done", replaced=3,
        shape_check={"ok": True, "surviving_placeholders": 0,
                     "broken_asset_refs": []},
    )
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is True
    assert res["expected_replaced"] == 3


def test_live_ledger_done_shape_check_failed_fails(tmp_path):
    """ENFORCEMENT: finalize's deep shape check finally has a consumer —
    when verify runs post-finalize and the recorded shape_check failed, the
    gate FAILS and surfaces the recorded errors."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(
        tmp_path, n=2, status="done", replaced=2,
        shape_check={"ok": False, "surviving_placeholders": 2,
                     "broken_asset_refs": []},
        errors=["inconsistent: surviving placeholders=2 but skipped=0"],
    )
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is False
    assert "shape check" in (res.get("error") or "").lower()
    assert "inconsistent" in (res.get("error") or "").lower()


def test_live_ledger_done_without_shape_check_passes_with_note(tmp_path):
    """The degraded kickoff (prompt_writer enqueue raised) writes status=done
    WITHOUT a shape_check — tolerate (placehold.co survives by design), but
    record a note."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PENDING, encoding="utf-8")
    _write_ledger(
        tmp_path, n=2, status="done",
        errors=["prompt_writer enqueue raised: x"],
    )
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path), swap_result={},
    )
    assert res["ok"] is True
    assert any("shape_check" in w for w in res["warnings"])
