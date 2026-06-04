# test_recanonicalize.py
"""Canonicalize-override recovery (mission 81 step 1.4a DLQ).

The analyst buried the correct competitive_positioning document inside a
```yaml fenced block in ``final_answer.result`` but wrote a *narration*
("Findings / Recommendations" report — no document) to the declared
``produces`` path. The existing ``autopersist_candidate`` only rescues a
*totally unwritten* path, so a path written with the WRONG content slipped
through; ``verify_competitive_positioning_shape`` then read the narration
file and false-DLQ'd a correct artifact.

``unwrap_fenced_artifact`` + ``recanonicalize_candidate`` are the pure
decision: when the single declared text path was written but its content
FAILS the artifact schema while the result carries a schema-valid artifact
(usually fence-wrapped), return ``(path, canonical_content)`` to overwrite.
"""
from __future__ import annotations

from coulson.grounding import (
    recanonicalize_candidate,
    unwrap_fenced_artifact,
)


# ─── unwrap_fenced_artifact ──────────────────────────────────────────────────

_WRAPPED = """## Analysis: Competitive Positioning Lock

### Corrected Artifact Content

```yaml
---
_schema_version: "1"
mission_id: 81
named_competitors: ["Habitica", "TickTick"]
---

## Landscape
Habitica and TickTick serve habit + productivity audiences.

## Notes
- https://habitica.com
```
"""


def test_unwrap_returns_fenced_body():
    body = unwrap_fenced_artifact(_WRAPPED)
    assert body is not None
    assert body.startswith("---")
    assert "## Landscape" in body
    # The narration wrapper and the fence markers are gone.
    assert "### Corrected Artifact Content" not in body
    assert "```" not in body


def test_unwrap_no_fence_returns_none():
    assert unwrap_fenced_artifact("# Just a plain doc\n\nno fence here") is None


def test_unwrap_non_string_returns_none():
    assert unwrap_fenced_artifact(None) is None
    assert unwrap_fenced_artifact({"a": 1}) is None


def test_unwrap_picks_largest_artifact_block():
    txt = (
        "intro\n```bash\nls -la\n```\nmiddle\n"
        "```md\n# Real Doc\n\n## Section\nbody body body body\n```\n"
    )
    body = unwrap_fenced_artifact(txt)
    assert body is not None
    assert body.startswith("# Real Doc")
    assert "ls -la" not in body


# ─── recanonicalize_candidate ────────────────────────────────────────────────

_PRODUCES = ["mission_81/.prd/competitive_positioning.md"]
_NARRATION = "## Analysis\n\n### Findings\n- named competitors listed\n\n### Recommendations\nThe file is ready."


def _md_sections_ok(needles):
    """Fake schema_ok: passes iff every needle header is present."""
    def _ok(content: str) -> bool:
        return all(f"## {n}" in content for n in needles)
    return _ok


def test_override_when_disk_fails_and_result_passes():
    schema_ok = _md_sections_ok(["Landscape", "Notes"])
    got = recanonicalize_candidate(
        _PRODUCES,
        {_PRODUCES[0]},          # path WAS written
        _WRAPPED,                # result carries the fenced artifact
        disk_content=_NARRATION,  # on-disk content is narration → fails
        schema_ok=schema_ok,
    )
    assert got is not None
    path, content = got
    assert path == _PRODUCES[0]
    assert content.startswith("---")
    assert "## Landscape" in content and "```" not in content


def test_no_override_when_disk_already_valid():
    schema_ok = _md_sections_ok(["Landscape", "Notes"])
    good_disk = "---\n_schema_version: \"1\"\n---\n\n## Landscape\nx\n\n## Notes\ny"
    got = recanonicalize_candidate(
        _PRODUCES, {_PRODUCES[0]}, _WRAPPED,
        disk_content=good_disk, schema_ok=schema_ok,
    )
    assert got is None  # disk already conforms — never clobber a valid file


def test_no_override_when_path_unwritten():
    # Unwritten path is the autopersist_candidate path, not this one.
    schema_ok = _md_sections_ok(["Landscape", "Notes"])
    got = recanonicalize_candidate(
        _PRODUCES, set(), _WRAPPED,
        disk_content=None, schema_ok=schema_ok,
    )
    assert got is None


def test_no_override_when_result_candidate_fails_schema():
    schema_ok = _md_sections_ok(["Landscape", "Notes", "DoesNotExist"])
    got = recanonicalize_candidate(
        _PRODUCES, {_PRODUCES[0]}, _WRAPPED,
        disk_content=_NARRATION, schema_ok=schema_ok,
    )
    assert got is None  # result doesn't satisfy schema either → don't persist


def test_no_override_multi_file_produces():
    schema_ok = _md_sections_ok(["Landscape"])
    got = recanonicalize_candidate(
        ["a.md", "b.md"], {"a.md", "b.md"}, _WRAPPED,
        disk_content=_NARRATION, schema_ok=schema_ok,
    )
    assert got is None


def test_no_override_non_text_extension():
    schema_ok = _md_sections_ok(["Landscape"])
    got = recanonicalize_candidate(
        ["m/logo.png"], {"m/logo.png"}, _WRAPPED,
        disk_content=_NARRATION, schema_ok=schema_ok,
    )
    assert got is None


def test_json_override_requires_parseable_candidate():
    schema_ok = lambda c: '"items"' in c  # noqa: E731
    produces = ["m/x.json"]
    bad_disk = "this is not json, just a note"
    # Candidate fence is invalid JSON → reject (never land garbage).
    wrapped_bad = "report\n```json\n{ broken\n```"
    assert recanonicalize_candidate(
        produces, {produces[0]}, wrapped_bad,
        disk_content=bad_disk, schema_ok=schema_ok,
    ) is None
    # Valid JSON candidate → override.
    wrapped_ok = 'report\n```json\n{"items": [1, 2]}\n```'
    got = recanonicalize_candidate(
        produces, {produces[0]}, wrapped_ok,
        disk_content=bad_disk, schema_ok=schema_ok,
    )
    assert got is not None
    assert got[0] == "m/x.json"
