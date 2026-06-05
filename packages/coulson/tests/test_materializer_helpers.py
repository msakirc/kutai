"""Materializer pure helpers (deterministic-materializer spec, 2026-06-05)."""
from __future__ import annotations

import json

from coulson.grounding import stamp_front_matter, select_canonical


# ── stamp_front_matter — markdown ──────────────────────────────────────────

def test_md_no_frontmatter_prepends_block():
    out = stamp_front_matter("# Charter\n\nbody", 81, "md")
    assert out.startswith("---\n")
    assert "mission_id: 81" in out.split("---")[1]
    assert "# Charter" in out


def test_md_frontmatter_missing_mission_id_injects():
    src = '---\ntitle: "x"\n---\n\n# Charter\nbody'
    out = stamp_front_matter(src, 81, "md")
    fm = out.split("---")[1]
    assert "mission_id: 81" in fm
    assert 'title: "x"' in fm
    # still exactly one front-matter block (two `---` fences)
    assert out.count("---") == 2


def test_md_frontmatter_with_mission_id_is_unchanged():
    src = '---\nmission_id: 81\ntitle: "x"\n---\n\n# Charter'
    assert stamp_front_matter(src, 81, "md") == src  # idempotent, no double-stamp


# ── stamp_front_matter — json ──────────────────────────────────────────────

def test_json_injects_mission_id_when_absent():
    out = stamp_front_matter('{"items": [1, 2]}', 81, "json")
    assert json.loads(out)["mission_id"] == 81
    assert json.loads(out)["items"] == [1, 2]


def test_json_with_mission_id_is_unchanged():
    src = '{"mission_id": 81, "items": []}'
    assert json.loads(stamp_front_matter(src, 81, "json"))["mission_id"] == 81


def test_json_unparseable_returned_as_is():
    assert stamp_front_matter("{ broken", 81, "json") == "{ broken"


# ── select_canonical ───────────────────────────────────────────────────────

_NARRATION_WRAP = (
    "## Analysis\n\n### Corrected Artifact Content\n\n"
    "```yaml\n---\nmission_id: 81\n---\n\n## Landscape\nx\n\n## Notes\ny\n```\n"
)
_DISK_NARRATION = "## Analysis\n### Findings\n- listed\n### Recommendations\nready."


def _needs(*sections):
    def _ok(c: str) -> bool:
        return all(f"## {s}" in c for s in sections)
    return _ok


def test_prefers_unwrapped_artifact_over_narration_wrapper():
    schema_ok = _needs("Landscape", "Notes")
    got = select_canonical([_NARRATION_WRAP, _DISK_NARRATION], schema_ok)
    assert got is not None
    assert got.strip().startswith("---")          # cleaned, front-matter first
    assert "## Landscape" in got and "```" not in got
    assert "### Corrected Artifact Content" not in got


def test_keeps_raw_doc_when_no_fence():
    schema_ok = _needs("Vision")
    doc = "# Charter\n\n## Vision\nbody"
    assert select_canonical([doc, None], schema_ok) == doc


def test_returns_most_substantial_when_none_conform():
    schema_ok = _needs("DoesNotExist")
    short, long = "## A\nx", "## B\n" + ("word " * 50)
    assert select_canonical([short, long], schema_ok) == long


def test_none_when_no_usable_candidate():
    assert select_canonical([None, "", "   "], _needs("X")) is None
