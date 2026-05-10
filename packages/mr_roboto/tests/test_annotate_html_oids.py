"""Z1 Tier 4 (T4B / C17 + A20) — annotate_html_oids contract tests.

The post-processor walks the DOM via BeautifulSoup and tags semantic
blocks (``<header>``, ``<main>``, ``<nav>``, ``<footer>``, ``<section>``,
plus ``<div>`` whose class hints at a section role) with
``data-oid="<artifact_slug>:<section>"``. The annotation is the anchor
the spec-patch proposer uses to reverse-look-up which spec node a DOM
diff touches (Onlook ``data-oid`` pattern, async-adapted).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mr_roboto.annotate_html_oids import annotate_html_oids


_HTML_BASIC = """<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
  <header><h1>Welcome</h1></header>
  <main>
    <section><h2>Sign in</h2><p>copy</p></section>
    <section><h2>Forgot</h2></section>
  </main>
  <footer><p>2026</p></footer>
</body>
</html>
"""


def test_semantic_blocks_get_data_oid():
    res = annotate_html_oids(html_text=_HTML_BASIC, artifact_slug="login")
    assert res["ok"] is True
    out = res["annotated_html"]
    # header / main / footer / two sections — all get data-oid
    assert 'data-oid="login:header"' in out
    assert 'data-oid="login:main"' in out
    assert 'data-oid="login:footer"' in out
    assert 'data-oid="login:section"' in out  # first
    assert 'data-oid="login:section_2"' in out  # second (deduped)
    # count
    assert res["annotated_count"] >= 5


def test_existing_data_oid_preserved():
    html = (
        '<!DOCTYPE html><html><body>'
        '<header data-oid="custom:hero">x</header>'
        '<main>m</main>'
        '</body></html>'
    )
    res = annotate_html_oids(html_text=html, artifact_slug="screen5")
    out = res["annotated_html"]
    assert 'data-oid="custom:hero"' in out
    assert 'data-oid="screen5:main"' in out


def test_idempotent_second_pass_no_op():
    res1 = annotate_html_oids(html_text=_HTML_BASIC, artifact_slug="login")
    res2 = annotate_html_oids(
        html_text=res1["annotated_html"], artifact_slug="login"
    )
    # No new oids on second pass: they're already there.
    assert res2["annotated_count"] == 0
    assert res2["annotated_html"] == res1["annotated_html"]


def test_path_in_place_rewrite(tmp_path: Path):
    p = tmp_path / "5_3.html"
    p.write_text(_HTML_BASIC, encoding="utf-8")
    res = annotate_html_oids(html_paths=[str(p)], artifact_slug="screen_plan_5_3")
    assert res["ok"] is True
    rewritten = p.read_text(encoding="utf-8")
    assert 'data-oid="screen_plan_5_3:header"' in rewritten
    assert res["per_file"][0]["annotated_count"] >= 5


def test_run_dispatch_routes_to_annotate(tmp_path: Path):
    """The mr_roboto.run dispatcher must route action='annotate_html_oids'."""
    from mr_roboto import run as mr_run
    p = tmp_path / "5_3.html"
    p.write_text(_HTML_BASIC, encoding="utf-8")
    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "annotate_html_oids",
            "html_paths": [str(p)],
            "artifact_slug": "screen_5_3",
        },
    }
    res = asyncio.run(mr_run(task))
    assert res.status == "completed", res.error
    assert 'data-oid="screen_5_3:header"' in p.read_text(encoding="utf-8")
