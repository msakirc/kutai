"""Z1 Tier 4 (T4B / C17 + A20) — propose_spec_patch_from_html_diff tests.

Founder edits an annotated HTML offline (e.g. tweaks copy, swaps a
color, reorders sections) and re-uploads. The mechanical diff:

1. Parses original + edited HTML via BeautifulSoup
2. Walks both DOMs by ``data-oid`` (Onlook-style stable anchor)
3. For each oid where text/attr/structure changed, proposes a patch
   to the upstream artifact (style guide for color, screen plan for
   layout, copy doc for text)

Output: ``spec_patch_proposal.md`` for founder review.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mr_roboto.propose_spec_patch import propose_spec_patch_from_html_diff


_ORIGINAL = """<!DOCTYPE html>
<html><body>
  <header data-oid="screen_5_3:header"><h1>Welcome to KutAI</h1></header>
  <main data-oid="screen_5_3:main">
    <section data-oid="screen_5_3:hero" style="background:#0066ff">
      <h2>Sign in</h2>
      <p>Use your email and password.</p>
    </section>
    <section data-oid="screen_5_3:cta">
      <button>Continue</button>
    </section>
  </main>
</body></html>
"""

# Edits: copy change in hero <p>, color change on hero (blue -> teal),
# button text change in cta. Header unchanged.
_EDITED = """<!DOCTYPE html>
<html><body>
  <header data-oid="screen_5_3:header"><h1>Welcome to KutAI</h1></header>
  <main data-oid="screen_5_3:main">
    <section data-oid="screen_5_3:hero" style="background:#00cc99">
      <h2>Sign in</h2>
      <p>Use email or social login.</p>
    </section>
    <section data-oid="screen_5_3:cta">
      <button>Get started</button>
    </section>
  </main>
</body></html>
"""


def test_detects_text_color_and_button_changes(tmp_path: Path):
    orig = tmp_path / "orig.html"
    edit = tmp_path / "edit.html"
    orig.write_text(_ORIGINAL, encoding="utf-8")
    edit.write_text(_EDITED, encoding="utf-8")

    res = propose_spec_patch_from_html_diff(
        html_path=str(orig),
        edited_html_path=str(edit),
    )
    assert res["ok"] is True
    changes = res["changes"]
    by_oid = {c["data_oid"]: c for c in changes}
    # hero touched — both color attr + nested copy text
    assert "screen_5_3:hero" in by_oid
    hero = by_oid["screen_5_3:hero"]
    kinds = set(hero["kinds"])
    # at least one of style or text recognized
    assert kinds & {"style", "text", "copy"}
    # cta changed (button text)
    assert "screen_5_3:cta" in by_oid
    # header unchanged → not in diff
    assert "screen_5_3:header" not in by_oid


def test_unchanged_html_yields_no_changes(tmp_path: Path):
    orig = tmp_path / "a.html"
    orig.write_text(_ORIGINAL, encoding="utf-8")
    edit = tmp_path / "b.html"
    edit.write_text(_ORIGINAL, encoding="utf-8")

    res = propose_spec_patch_from_html_diff(
        html_path=str(orig),
        edited_html_path=str(edit),
    )
    assert res["ok"] is True
    assert res["changes"] == []


def test_emits_proposal_markdown(tmp_path: Path):
    orig = tmp_path / "o.html"
    edit = tmp_path / "e.html"
    out = tmp_path / "spec_patch_proposal.md"
    orig.write_text(_ORIGINAL, encoding="utf-8")
    edit.write_text(_EDITED, encoding="utf-8")

    res = propose_spec_patch_from_html_diff(
        html_path=str(orig),
        edited_html_path=str(edit),
        out_path=str(out),
    )
    assert res["ok"] is True
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "spec patch proposal" in body.lower()
    assert "screen_5_3:hero" in body


def test_missing_oid_on_edited_node_falls_back_to_text_match(tmp_path: Path):
    """If edited HTML strips data-oid (founder used a tool that doesn't
    preserve it), the diff should still surface SOMETHING — fallback
    matches by sibling-position when oid missing."""
    orig = tmp_path / "o.html"
    edit = tmp_path / "e.html"
    orig.write_text(_ORIGINAL, encoding="utf-8")
    stripped = _EDITED.replace(' data-oid="screen_5_3:hero"', "")
    edit.write_text(stripped, encoding="utf-8")

    res = propose_spec_patch_from_html_diff(
        html_path=str(orig),
        edited_html_path=str(edit),
    )
    assert res["ok"] is True
    # we don't require detection here — just that the call doesn't crash
    # and surfaces a structured "missing_oids" warning so the founder
    # knows the diff is incomplete.
    assert isinstance(res.get("missing_oids"), list)


def test_run_dispatch_routes(tmp_path: Path):
    from mr_roboto import run as mr_run
    orig = tmp_path / "o.html"
    edit = tmp_path / "e.html"
    out = tmp_path / "p.md"
    orig.write_text(_ORIGINAL, encoding="utf-8")
    edit.write_text(_EDITED, encoding="utf-8")

    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "propose_spec_patch_from_html_diff",
            "html_path": str(orig),
            "edited_html_path": str(edit),
            "out_path": str(out),
        },
    }
    res = asyncio.run(mr_run(task))
    assert res.status == "completed", res.error
    assert out.exists()
