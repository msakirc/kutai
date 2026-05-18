"""Z1 Tier 3 (C9+A11) — verify_html_prototype_shape contract tests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mr_roboto.verify_html_prototype_shape import verify_html_prototype_shape
from mr_roboto import run as mr_roboto_run


_HERE = Path(__file__).resolve().parent
_FIX_DIR = _HERE / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _load(stem: str) -> dict:
    return json.loads((_FIX_DIR / f"{stem}.json").read_text(encoding="utf-8"))


def test_good_html_prototype_passes():
    fx = _load("good_html_prototype")
    res = verify_html_prototype_shape(
        html_text=fx["html_text"], design_tokens=fx["design_tokens"]
    )
    assert res["ok"] is True, res
    assert res["doctype_present"] is True
    assert res["has_390_width"] is True
    assert res["has_844_height"] is True
    assert res["has_tailwind"] is True
    assert res["img_problems"] == []
    assert res["color_offenders"] == []


def test_bad_html_rejected():
    fx = _load("bad_html_prototype")
    res = verify_html_prototype_shape(
        html_text=fx["html_text"], design_tokens=fx["design_tokens"]
    )
    assert res["ok"] is False
    assert res["has_390_width"] is False
    # img without alt should be flagged
    assert res["img_problems"]
    # hardcoded color #FF00FF outside tokens
    assert any(
        offender.lower().lstrip("#") == "ff00ff" for offender in res["color_offenders"]
    )


def test_missing_doctype_rejected():
    html = (
        "<html><head><script src=\"https://cdn.tailwindcss.com\"></script>"
        "</head><body class=\"w-[390px] min-h-[844px]\">"
        "<img src=\"https://placehold.co/100x100\" alt=\"x\">"
        "</body></html>"
    )
    res = verify_html_prototype_shape(html_text=html)
    assert res["ok"] is False
    assert res["doctype_present"] is False


def test_img_without_alt_rejected():
    html = (
        "<!DOCTYPE html><html><head>"
        "<script src=\"https://cdn.tailwindcss.com\"></script></head>"
        "<body class=\"w-[390px] min-h-[844px]\">"
        "<img src=\"https://placehold.co/100x100\">"
        "</body></html>"
    )
    res = verify_html_prototype_shape(html_text=html)
    assert res["ok"] is False
    assert res["img_problems"]
    assert any("alt" in p["issue"] for p in res["img_problems"])


def test_empty_src_rejected():
    html = (
        "<!DOCTYPE html><html><head>"
        "<script src=\"https://cdn.tailwindcss.com\"></script></head>"
        "<body class=\"w-[390px] min-h-[844px]\">"
        "<img src=\"about:blank\" alt=\"placeholder\">"
        "</body></html>"
    )
    res = verify_html_prototype_shape(html_text=html)
    assert res["ok"] is False


def test_no_design_tokens_skips_color_check():
    """When design_tokens is None, hardcoded colors are tolerated."""
    html = (
        "<!DOCTYPE html><html><head>"
        "<script src=\"https://cdn.tailwindcss.com\"></script></head>"
        "<body class=\"w-[390px] min-h-[844px]\" style=\"color:#FF00FF\">"
        "<img src=\"https://placehold.co/100x100\" alt=\"x\">"
        "</body></html>"
    )
    res = verify_html_prototype_shape(html_text=html, design_tokens=None)
    assert res["ok"] is True
    assert res["color_offenders"] == []


def test_dispatcher_completes_on_good():
    fx = _load("good_html_prototype")
    task = {
        "id": 0, "mission_id": 0,
        "payload": {
            "action": "verify_html_prototype_shape",
            "html_text": fx["html_text"],
            "design_tokens": fx["design_tokens"],
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "completed", result.error
    assert result.result["ok"] is True


def test_dispatcher_fails_on_bad():
    fx = _load("bad_html_prototype")
    task = {
        "id": 0, "mission_id": 0,
        "payload": {
            "action": "verify_html_prototype_shape",
            "html_text": fx["html_text"],
            "design_tokens": fx["design_tokens"],
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "failed"


def test_path_mode(tmp_path: Path):
    fx = _load("good_html_prototype")
    p = tmp_path / "home.html"
    p.write_text(fx["html_text"], encoding="utf-8")
    res = verify_html_prototype_shape(
        html_paths=[str(p)], design_tokens=fx["design_tokens"]
    )
    assert res["ok"] is True
    assert len(res["per_file"]) == 1
