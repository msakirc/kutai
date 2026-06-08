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
    errors is empty, the result is internally inconsistent — fail."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PARTIAL, encoding="utf-8")
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
    prototype with no surviving placeholders still passes."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result="not json {",
    )
    assert res["ok"] is True
    assert res["surviving_placeholders"] == 0
