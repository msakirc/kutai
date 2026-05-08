"""Tests for mr_roboto.parse_og_tags — the meta-tag verifier verb.

The pure regex extraction is exercised offline. The HTTP-bound tests use
aiohttp's ClientSession with a tmpdir-served HTML or by monkeypatching
``mr_roboto.parse_og_tags._fetch`` so the suite stays hermetic.
"""
from __future__ import annotations

import pytest

import importlib

og_mod = importlib.import_module("mr_roboto.parse_og_tags")
from mr_roboto.parse_og_tags import _extract_meta, parse_og_tags
import mr_roboto


SAMPLE_HTML_GOOD = """
<html><head>
<meta property="og:title" content="Hello World" />
<meta property="og:description" content="A test page" />
<meta property="og:image" content="https://example.com/img.png" />
<meta name="twitter:card" content="summary_large_image" />
<title>Fallback</title>
</head><body></body></html>
"""

SAMPLE_HTML_MISSING_IMAGE = """
<html><head>
<meta property="og:title" content="Hello" />
<meta property="og:description" content="x" />
</head></html>
"""

SAMPLE_HTML_REVERSED_ATTRS = """
<meta content="Reversed Title" property="og:title">
<meta content="Reversed Desc" property="og:description">
<meta content="https://example.com/r.png" property="og:image">
"""


def test_extract_meta_property_first():
    out = _extract_meta(SAMPLE_HTML_GOOD)
    assert out["og:title"] == "Hello World"
    assert out["og:description"] == "A test page"
    assert out["og:image"] == "https://example.com/img.png"
    assert out["twitter:card"] == "summary_large_image"


def test_extract_meta_content_first():
    out = _extract_meta(SAMPLE_HTML_REVERSED_ATTRS)
    assert out["og:title"] == "Reversed Title"
    assert out["og:image"] == "https://example.com/r.png"


def test_extract_meta_title_fallback():
    out = _extract_meta("<html><head><title>Just title</title></head></html>")
    assert out.get("title") == "Just title"
    assert "og:title" not in out


@pytest.mark.asyncio
async def test_parse_og_tags_invalid_url():
    res = await parse_og_tags("not-a-url")
    assert res["ok"] is False
    assert "invalid url" in res["errors"]


@pytest.mark.asyncio
async def test_parse_og_tags_happy(monkeypatch):
    async def fake_fetch(url, t):
        return 200, SAMPLE_HTML_GOOD, None

    async def fake_head(url, t):
        return 200, None

    monkeypatch.setattr(og_mod, "_fetch", fake_fetch)
    monkeypatch.setattr(og_mod, "_head", fake_head)

    res = await parse_og_tags("https://example.com/")
    assert res["ok"] is True
    assert res["missing"] == []
    assert res["image_reachable"] is True
    assert res["tags"]["og:title"] == "Hello World"


@pytest.mark.asyncio
async def test_parse_og_tags_missing_required(monkeypatch):
    async def fake_fetch(url, t):
        return 200, SAMPLE_HTML_MISSING_IMAGE, None

    monkeypatch.setattr(og_mod, "_fetch", fake_fetch)

    res = await parse_og_tags("https://example.com/", check_image=False)
    assert res["ok"] is False
    assert "og:image" in res["missing"]


@pytest.mark.asyncio
async def test_parse_og_tags_image_unreachable(monkeypatch):
    async def fake_fetch(url, t):
        return 200, SAMPLE_HTML_GOOD, None

    async def fake_head(url, t):
        return 404, None

    monkeypatch.setattr(og_mod, "_fetch", fake_fetch)
    monkeypatch.setattr(og_mod, "_head", fake_head)

    res = await parse_og_tags("https://example.com/")
    assert res["ok"] is False
    assert res["image_reachable"] is False


@pytest.mark.asyncio
async def test_parse_og_tags_fetch_failure(monkeypatch):
    async def fake_fetch(url, t):
        return 0, "", "ConnectionError: refused"

    monkeypatch.setattr(og_mod, "_fetch", fake_fetch)

    res = await parse_og_tags("https://example.com/")
    assert res["ok"] is False
    assert any("ConnectionError" in e for e in res["errors"])


@pytest.mark.asyncio
async def test_parse_og_tags_via_dispatcher(monkeypatch):
    async def fake_fetch(url, t):
        return 200, SAMPLE_HTML_GOOD, None

    async def fake_head(url, t):
        return 200, None

    monkeypatch.setattr(og_mod, "_fetch", fake_fetch)
    monkeypatch.setattr(og_mod, "_head", fake_head)

    action = await mr_roboto.run({
        "mission_id": None,
        "payload": {
            "action": "parse_og_tags",
            "url": "https://example.com/",
        },
    })
    assert action.status == "completed", action
    assert action.result["ok"] is True
