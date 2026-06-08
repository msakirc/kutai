import json
import os
import pytest
from mr_roboto.swap_placeholder_images import (
    swap_placeholder_images,
    _scan_placeholders,
    _list_html_files,
    _PLACEHOLDER_HOST_RE,
    _parse_task_result,
)


def test_placeholder_host_regex():
    assert _PLACEHOLDER_HOST_RE.search("https://placehold.co/64x64/eee/333?text=x")
    assert _PLACEHOLDER_HOST_RE.search("http://placehold.co/256x256")
    assert not _PLACEHOLDER_HOST_RE.search("/assets/hero_1.png")
    assert not _PLACEHOLDER_HOST_RE.search("assets/hero_1.png")
    assert not _PLACEHOLDER_HOST_RE.search("https://example.com/real.png")


_HTML = """<!DOCTYPE html>
<html><body class="w-[390px] min-h-[844px]">
  <img src="https://placehold.co/390x220/E07A5F/FFF?text=hero"
       alt="smiling barista handing over a takeaway cup">
  <img src="https://placehold.co/260x180/3D405B/FFF?text=feat"
       alt="ai-powered task triage dashboard">
  <img src="/assets/already_real.png" alt="something already swapped">
  <img src="https://placehold.co/64x64/264653/FFF?text=u"
       alt="user portrait">
</body></html>"""


def test_scan_finds_three(tmp_path):
    p = tmp_path / "home.html"
    p.write_text(_HTML, encoding="utf-8")
    hits = _scan_placeholders(str(p))
    assert len(hits) == 3
    ids = {h["placeholder_id"] for h in hits}
    assert ids == {"home__0", "home__1", "home__2"}
    assert all(h["alt"] for h in hits)
    assert all(h["width"] > 0 and h["height"] > 0 for h in hits)


def test_scan_handles_missing(tmp_path):
    assert _scan_placeholders(str(tmp_path / "missing.html")) == []


def test_scan_handles_no_placeholders(tmp_path):
    p = tmp_path / "empty.html"
    p.write_text("<html><body>no images</body></html>", encoding="utf-8")
    assert _scan_placeholders(str(p)) == []


def test_list_html_recursive(tmp_path):
    """v2 fix: walks subdirectories so multi-screen prototypes work."""
    web = tmp_path / ".web"
    (web / "screens").mkdir(parents=True)
    (web / "home.html").write_text("<html></html>", encoding="utf-8")
    (web / "screens" / "onboarding.html").write_text("<html></html>", encoding="utf-8")
    (web / "screens" / "settings.html").write_text("<html></html>", encoding="utf-8")
    (web / "assets").mkdir(exist_ok=True)
    (web / "assets" / "ignored.png").write_bytes(b"\x89PNG")  # not an HTML
    files = _list_html_files(str(tmp_path))
    names = sorted(os.path.basename(f) for f in files)
    assert names == ["home.html", "onboarding.html", "settings.html"]


def test_parse_task_result_handles_json_string():
    """v2 fix: TaskResult.result is a JSON string in production."""
    class _TR:
        result = json.dumps({"path": "/x/y.png", "provider": "p"})
    parsed = _parse_task_result(_TR())
    assert parsed == {"path": "/x/y.png", "provider": "p"}


def test_parse_task_result_handles_dict():
    """Defensive: tests may pass dicts."""
    class _TR:
        result = {"path": "/x/y.png"}
    parsed = _parse_task_result(_TR())
    assert parsed == {"path": "/x/y.png"}


def test_parse_task_result_handles_none():
    class _TR:
        result = None
    assert _parse_task_result(_TR()) == {}


def test_parse_task_result_handles_garbage_string():
    class _TR:
        result = "not json {"
    assert _parse_task_result(_TR()) == {}


@pytest.mark.asyncio
async def test_swap_no_html_files(monkeypatch, tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["replaced_count"] == 0
    assert res["html_files_seen"] == 0
