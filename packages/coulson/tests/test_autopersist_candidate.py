# test_autopersist_candidate.py
"""Auto-persist recovery for inline final_answer artifacts (mission 75 DLQ).

Step 0.0a.draft (intake_todo_draft) uses the "return JSON as final_answer,
engine persists to the produces path" contract with write_file disabled.
The recovery in react.py only persisted `.md` produces, so the `.json`
artifact was never written, the grounding guard never cleared, and the
writer agent looped to max_iterations → DLQ (task 165064).

`autopersist_candidate(produces, written, result)` is the pure decision:
return (relative_path, content_to_write) when a single still-unwritten
text artifact (.md / .json) is dumped inline, else None.
"""
from __future__ import annotations

import json

from coulson.grounding import autopersist_candidate


# ─── .json (the bug) ──────────────────────────────────────────────────────────

def test_json_produces_persists_valid_json_string():
    produces = ["mission_75/.intake/intake_todo_draft.json"]
    result = '{"_schema_version": "1", "items": [{"n": 1, "category": "Audience"}]}'
    got = autopersist_candidate(produces, set(), result)
    assert got == (produces[0], result)


def test_json_produces_serializes_dict_result():
    produces = ["m/x.json"]
    result = {"_schema_version": "1", "items": [{"n": 1}]}
    got = autopersist_candidate(produces, set(), result)
    assert got is not None
    path, content = got
    assert path == "m/x.json"
    assert json.loads(content) == result  # round-trips


def test_json_produces_rejects_invalid_json():
    # Truncated / malformed → do NOT persist garbage; let the guard re-prompt.
    got = autopersist_candidate(["m/x.json"], set(), '{"items": [')
    assert got is None


def test_json_produces_rejects_empty_result():
    assert autopersist_candidate(["m/x.json"], set(), "") is None
    assert autopersist_candidate(["m/x.json"], set(), "   ") is None


# ─── .md (regression — existing behavior preserved) ──────────────────────────

def test_md_produces_persists_long_markdown():
    produces = ["m/charter.md"]
    result = "# Charter\n\n" + ("lorem ipsum " * 60)  # >500 chars
    got = autopersist_candidate(produces, set(), result)
    assert got == (produces[0], result)


def test_md_produces_rejects_short_markdown():
    assert autopersist_candidate(["m/charter.md"], set(), "too short") is None


# ─── guards: only when the single produces is still unwritten ─────────────────

def test_skips_when_already_written():
    produces = ["m/x.json"]
    written = {"m/x.json"}
    valid = '{"items": []}'
    assert autopersist_candidate(produces, written, valid) is None


def test_skips_multi_file_produces():
    got = autopersist_candidate(["a.json", "b.json"], set(), '{"x": 1}')
    assert got is None


def test_skips_non_text_extension():
    assert autopersist_candidate(["m/logo.png"], set(), "x" * 600) is None


# ─── schema-aware unwrap (mission 81 §4 — unwritten narration-wrap hole) ──────
# When the agent dumps a narration report that BURIES the real artifact in a
# fenced block, persisting the raw narration passes the loose
# validate_artifact_schema header scan (it finds `## Section` *inside* the
# fence) but fails the stricter verify_* front-matter gate (`---` not at file
# start) → false DLQ. With a schema_ok validator injected, auto-persist must
# write the UNWRAPPED artifact instead of the raw narration.

_AP_WRAPPED = """## Analysis: Competitive Positioning Lock

### Corrected Artifact Content

```yaml
---
_schema_version: "1"
mission_id: 81
---

## Landscape
Habitica and TickTick serve habit + productivity audiences.

## Notes
- https://habitica.com
```
"""


def _md_needs(*needles):
    """Fake schema_ok: passes iff every `## needle` header is present."""
    def _ok(content: str) -> bool:
        return all(f"## {n}" in content for n in needles)
    return _ok


def test_md_schema_ok_persists_unwrapped_artifact():
    # Raw passes validate (headers found inside fence) AND body passes — must
    # still prefer the UNWRAPPED body so front-matter lands at file start.
    schema_ok = _md_needs("Landscape", "Notes")
    got = autopersist_candidate(
        ["m/cp.md"], set(), _AP_WRAPPED, schema_ok=schema_ok,
    )
    assert got is not None
    path, content = got
    assert path == "m/cp.md"
    assert content.startswith("---")            # front-matter at file start
    assert "## Landscape" in content
    assert "```" not in content                  # fence markers stripped
    assert "### Corrected Artifact Content" not in content  # narration gone


def test_md_schema_ok_keeps_raw_doc_with_incidental_code_fence():
    # A real doc that merely *contains* a non-artifact fenced snippet must NOT
    # be replaced by that snippet. body=`ls -la` fails schema → keep raw doc.
    doc = "# Real Doc\n\n## Section\n" + ("body " * 120) + "\n```bash\nls -la\n```\n"
    schema_ok = _md_needs("Section")
    got = autopersist_candidate(["m/d.md"], set(), doc, schema_ok=schema_ok)
    assert got is not None
    path, content = got
    assert content == doc          # raw doc preserved verbatim
    assert "ls -la" in content


def test_md_schema_ok_keeps_raw_doc_embedding_valid_example_fence():
    # Doc embeds a complete JSON example in a fence; the md schema needs
    # `## Vision`, which the JSON body lacks → body fails → keep the raw doc.
    doc = (
        "---\nx: 1\n---\n\n# Charter\n\n## Vision\n" + ("v " * 120)
        + '\n```json\n{"example": true}\n```\n'
    )
    schema_ok = _md_needs("Vision")
    got = autopersist_candidate(["m/c.md"], set(), doc, schema_ok=schema_ok)
    assert got is not None
    assert got[1] == doc


def test_md_schema_ok_length_fallback_when_neither_passes():
    # No fence, raw fails schema but is substantial → length heuristic still
    # persists raw (lets grade/verify give precise feedback, not loop).
    doc = "# Draft\n\n" + ("filler " * 100)   # >500 chars, lacks required header
    schema_ok = _md_needs("Vision")
    got = autopersist_candidate(["m/c.md"], set(), doc, schema_ok=schema_ok)
    assert got == ("m/c.md", doc)


def test_md_no_schema_ok_keeps_legacy_length_behavior():
    # Without schema_ok the function cannot safely distinguish narration from
    # artifact → falls back to the raw length heuristic (unchanged contract):
    # it persists the RAW wrapper, never the unwrapped body.
    raw = _AP_WRAPPED + ("\nmore narration " * 40)   # push past the 500 floor
    got = autopersist_candidate(["m/cp.md"], set(), raw)
    assert got == ("m/cp.md", raw)   # raw, not unwrapped — no schema to decide


def test_json_unwraps_fenced_artifact_when_raw_unparseable():
    # Narration wrapping a ```json fence: raw doesn't parse → unwrap + persist
    # the inner JSON instead of looping to DLQ.
    wrapped = '## Summary\n\nHere is the artifact:\n```json\n{"items": [1, 2]}\n```\n'
    got = autopersist_candidate(["m/x.json"], set(), wrapped)
    assert got is not None
    path, content = got
    assert path == "m/x.json"
    assert json.loads(content) == {"items": [1, 2]}


def test_json_unwrap_rejects_unparseable_fence():
    wrapped = "## Summary\n```json\n{ broken json\n```"
    assert autopersist_candidate(["m/x.json"], set(), wrapped) is None
