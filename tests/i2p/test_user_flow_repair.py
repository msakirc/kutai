"""Deterministic repair of user_flow.md before verify_user_flow_shape.

m90 task 567452: the analyst model authored a semantically-correct flow graph
but dropped the `surfaces:` frontmatter line and the ```mermaid fence (bare
`graph TD`). The shape is mechanical — the engine repairs it deterministically
(inject surfaces from surfaces.json, fence bare mermaid blocks) so a compliant,
renderable artifact is guaranteed regardless of model compliance. Repair never
fabricates missing per-surface diagrams (multi-surface under-production still
correctly fails).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr_roboto.verify_user_flow_shape import normalize_user_flow, verify_user_flow_shape


# ── pure normalize() unit tests ──────────────────────────────────────────────

def test_injects_surfaces_when_frontmatter_lacks_it():
    text = "---\nmission_id: 90\n---\n\n```mermaid\ngraph TD\n  A --> B\n```\n"
    out, changed = normalize_user_flow(text, ["web"])
    assert changed
    assert "surfaces: [web]" in out
    # existing frontmatter key preserved
    assert "mission_id: 90" in out


def test_fences_bare_graph_block():
    text = "---\nsurfaces: [web]\n---\n\ngraph TD\n    A[\"a\"] --> B[\"b\"]\n"
    out, changed = normalize_user_flow(text, ["web"])
    assert changed
    assert "```mermaid" in out
    assert out.count("```") == 2  # exactly one opening + one closing fence
    # node content preserved inside the fence
    assert 'A["a"] --> B["b"]' in out


def test_leaves_already_fenced_untouched():
    text = "---\nsurfaces: [web]\n---\n\n```mermaid\ngraph TD\n  A --> B\n```\n"
    out, changed = normalize_user_flow(text, ["web"])
    assert not changed
    assert out.count("```") == 2  # no double-fencing


def test_idempotent():
    text = "---\nmission_id: 90\n---\n\ngraph TD\n    A --> B\n"
    once, _ = normalize_user_flow(text, ["web"])
    twice, changed2 = normalize_user_flow(once, ["web"])
    assert not changed2
    assert once == twice


def test_no_surfaces_available_does_not_inject_empty():
    text = "---\nmission_id: 90\n---\n\ngraph TD\n    A --> B\n"
    out, _ = normalize_user_flow(text, [])
    assert "surfaces: []" not in out  # never inject an empty surfaces list


# ── end-to-end: the exact 567452 case, surfaces from surfaces.json ───────────

@pytest.mark.asyncio
async def test_verify_repairs_567452_and_passes(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / ".flow").mkdir(parents=True)
    (ws / ".charter").mkdir(parents=True)
    (ws / ".charter" / "surfaces.json").write_text(
        json.dumps({"surfaces": ["web"]}), encoding="utf-8"
    )
    # 567452 on-disk shape: frontmatter has mission_id but NO surfaces; bare graph.
    (ws / ".flow" / "user_flow.md").write_text(
        "---\nmission_id: 90\n---\n\n"
        'graph TD\n    Start["Landing<br/>/"] --> Auth["Sign Up<br/>/auth"]\n',
        encoding="utf-8",
    )
    res = await verify_user_flow_shape(
        mission_id=90, path=".flow/user_flow.md", workspace_path=str(ws),
    )
    assert res["ok"], res
    # the on-disk file was repaired: renderable + shape-compliant
    fixed = (ws / ".flow" / "user_flow.md").read_text(encoding="utf-8")
    assert fixed.startswith("---")
    assert "surfaces: [web]" in fixed
    assert "```mermaid" in fixed
