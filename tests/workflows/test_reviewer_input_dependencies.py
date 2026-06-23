"""Reviewer step 1.13 (research_quality_review) must WAIT for the producers of
every artifact it reviews. Mission 89 halted because 1.13 consumed
prior_art_report / interview_script / competitive_positioning whose producers
(1.0c / 0.0c / 1.4a) were NOT in 1.13's dependency closure — so 1.13 raced ahead
and flagged the not-yet-produced inputs as "does not exist" blockers.

A reviewer reviewing an input it does not depend on is a structural bug: the
input may legitimately be absent only because its producer hasn't run yet. This
test asserts every UNCONDITIONAL (skip_when is None) input producer of 1.13 is
in its transitive depends_on closure. (Conditional producers stay optional so we
never deadlock 1.13 on a skipped lane.)
"""
import json
from pathlib import Path

WF = Path("src/workflows/i2p/i2p_v3.json")


def _load():
    data = json.loads(WF.read_text(encoding="utf-8"))
    return {s["id"]: s for s in data["steps"]}, data["steps"]


def _closure(by_id, sid):
    seen, stack = set(), list(by_id.get(sid, {}).get("depends_on", []))
    while stack:
        d = stack.pop()
        if d in seen:
            continue
        seen.add(d)
        stack += by_id.get(d, {}).get("depends_on", [])
    return seen


def test_1_13_waits_for_all_unconditional_input_producers():
    by_id, steps = _load()
    producers: dict[str, list[str]] = {}
    for s in steps:
        for art in s.get("output_artifacts", []) or []:
            producers.setdefault(art, []).append(s["id"])

    closure = _closure(by_id, "1.13")
    missing = []
    for art in by_id["1.13"].get("input_artifacts", []):
        for p in producers.get(art, []):
            if p == "1.13":
                continue
            if by_id.get(p, {}).get("skip_when") is None and p not in closure:
                missing.append((art, p))

    assert missing == [], (
        "1.13 reviews artifacts whose unconditional producers it does not wait "
        f"for (race -> spurious 'does not exist' blockers): {missing}"
    )
