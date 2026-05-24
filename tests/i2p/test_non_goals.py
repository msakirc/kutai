"""Z1 Tier 2 (A2) — non_goals.md shape + non-goals overlap tests."""
from __future__ import annotations

import asyncio

from mr_roboto.verify_non_goals_shape import verify_non_goals_shape
from mr_roboto.check_against_non_goals import check_against_non_goals
from mr_roboto import run as mr_roboto_run


_GOOD_NG = """\
---
_schema_version: "1"
mission_id: 42
non_goals:
  - "No real-time multiplayer features."
  - "No mobile-app version (web-only at launch)."
  - "No support for under-13 users."
  - "No vendor onboarding workflow this iteration."
---

# Non-goals

- No real-time multiplayer features.
- No mobile-app version (web-only at launch).
- No support for under-13 users.
- No vendor onboarding workflow this iteration.
"""


_NO_FRONTMATTER = """\
# Non-goals

- No real-time multiplayer.
- No mobile.
- No under-13.
"""


_TOO_FEW = """\
---
_schema_version: "1"
non_goals:
  - "Only one non-goal."
---

# Non-goals

- Only one non-goal.
"""


_PLACEHOLDER = """\
---
_schema_version: "1"
non_goals:
  - "TODO add non-goal"
  - "No mobile."
  - "No under-13."
---

# Non-goals

- TODO add non-goal
- No mobile.
- No under-13.
"""


_WRONG_VERSION = """\
---
_schema_version: "2"
non_goals:
  - "No A."
  - "No B."
  - "No C."
---

# Non-goals

- No A.
- No B.
- No C.
"""


# ────────────────────────────────────────────────────────────────────────────
# verify_non_goals_shape
# ────────────────────────────────────────────────────────────────────────────


def test_good_shape_passes():
    res = verify_non_goals_shape(non_goals_text=_GOOD_NG)
    assert res["ok"] is True, res
    assert res["frontmatter_present"] is True
    assert res["schema_version"] == "1"
    assert res["yaml_count"] == 4
    assert res["bullet_count"] == 4
    assert res["placeholders"] == []
    assert res["problems"] == []


def test_missing_frontmatter_rejects():
    res = verify_non_goals_shape(non_goals_text=_NO_FRONTMATTER)
    assert res["ok"] is False
    assert res["frontmatter_present"] is False
    assert any("frontmatter" in p.lower() for p in res["problems"])


def test_too_few_bullets_rejects():
    res = verify_non_goals_shape(non_goals_text=_TOO_FEW)
    assert res["ok"] is False
    assert any("bullet_count" in p or "yaml_count" in p for p in res["problems"])


def test_placeholders_rejected():
    res = verify_non_goals_shape(non_goals_text=_PLACEHOLDER)
    assert res["ok"] is False
    assert res["placeholders"]
    assert any("placeholder" in p.lower() for p in res["problems"])


def test_wrong_schema_version_rejected():
    res = verify_non_goals_shape(non_goals_text=_WRONG_VERSION)
    assert res["ok"] is False
    assert res["schema_version"] == "2"
    assert any("_schema_version" in p for p in res["problems"])


def test_empty_rejects():
    res = verify_non_goals_shape(non_goals_text="")
    assert res["ok"] is False
    assert "empty non_goals.md" in res["problems"]


# ────────────────────────────────────────────────────────────────────────────
# check_against_non_goals
# ────────────────────────────────────────────────────────────────────────────


def test_overlap_returns_match():
    target = (
        "FR-014: implement WebSocket multiplayer rooms for real-time chat "
        "between players. Use a Redis pub/sub bus for the multiplayer events."
    )
    res = check_against_non_goals(
        non_goals_text=_GOOD_NG, target_text=target
    )
    assert res["matches"], res
    # The "real-time multiplayer" non-goal should be flagged.
    flagged = " ".join(m["non_goal"] for m in res["matches"]).lower()
    assert "multiplayer" in flagged


def test_no_overlap_returns_empty():
    target = (
        "FR-001: nightly batch job aggregates pricing data from supplier "
        "feeds. No realtime requirement; runs at 02:00 TR-time."
    )
    res = check_against_non_goals(
        non_goals_text=_GOOD_NG, target_text=target
    )
    # Heuristic may catch token overlap on "non-goal" tokens like "feature";
    # what matters is not flagging the multiplayer non-goal here.
    matches_text = " ".join(m["non_goal"] for m in res["matches"]).lower()
    assert "multiplayer" not in matches_text


def test_missing_non_goals_returns_empty():
    res = check_against_non_goals(
        non_goals_text="", target_text="anything goes"
    )
    assert res["matches"] == []
    assert res["non_goals_present"] is False


def test_literal_substring_flags_match():
    target = (
        "Architecture: NO MOBILE-APP VERSION (web-only at launch). "
        "We host React via Vite + Cloudflare Pages."
    )
    res = check_against_non_goals(
        non_goals_text=_GOOD_NG, target_text=target.lower()
    )
    assert any(m["literal_substring"] for m in res["matches"])


# ────────────────────────────────────────────────────────────────────────────
# Mechanical dispatch tests
# ────────────────────────────────────────────────────────────────────────────


def test_dispatch_verify_non_goals_shape_completed():
    task = {
        "id": 0,
        "mission_id": 0,
        "payload": {
            "action": "verify_non_goals_shape",
            "non_goals_text": _GOOD_NG,
        },
    }
    res = asyncio.run(mr_roboto_run(task))
    assert res.status == "completed"
    assert res.result["ok"] is True


def test_dispatch_verify_non_goals_shape_failed():
    task = {
        "id": 0,
        "mission_id": 0,
        "payload": {
            "action": "verify_non_goals_shape",
            "non_goals_text": _NO_FRONTMATTER,
        },
    }
    res = asyncio.run(mr_roboto_run(task))
    assert res.status == "failed"


def test_dispatch_check_against_non_goals_completes_on_overlap():
    """check_against_non_goals always returns completed; matches list is signal."""
    task = {
        "id": 0,
        "mission_id": 0,
        "payload": {
            "action": "check_against_non_goals",
            "non_goals_text": _GOOD_NG,
            "target_text": "WebSocket multiplayer real-time gameplay.",
        },
    }
    res = asyncio.run(mr_roboto_run(task))
    assert res.status == "completed"
    assert res.result["matches"]


def test_workflow_step_0_6a_present():
    """i2p_v3.json must define 0.6a non_goals_lock + 0.6a.verify."""
    import json
    from pathlib import Path

    wf_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "workflows"
        / "i2p"
        / "i2p_v3.json"
    )
    wf = json.loads(wf_path.read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in wf["steps"]}

    assert "0.6a" in by_id
    assert by_id["0.6a"]["agent"] == "mechanical"
    assert "non_goals_lock" == by_id["0.6a"]["name"]
    assert "non_goals" in by_id["0.6a"]["output_artifacts"]
    # legacy_pre_non_goals gate was removed; step is now unconditional
    sw = by_id["0.6a"].get("skip_when") or ""
    assert not sw or "legacy_pre_" not in sw

    assert "0.6a.verify" in by_id
    assert by_id["0.6a.verify"]["payload"]["action"] == "verify_non_goals_shape"


def test_schema_version_carried_on_artifacts():
    """All new artifacts emit _schema_version='1'."""
    import json
    from pathlib import Path

    wf_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "workflows"
        / "i2p"
        / "i2p_v3.json"
    )
    wf = json.loads(wf_path.read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in wf["steps"]}

    for sid, art in (
        ("0.6a", "non_goals"),
        ("0.6a.verify", "non_goals_shape_result"),
        ("3.1.verify", "functional_requirements_falsification_result"),
        ("3.7.verify", "business_rules_falsification_result"),
    ):
        schema = by_id[sid]["artifact_schema"][art]
        assert schema.get("_schema_version") == "1", (
            f"step {sid} artifact {art} missing _schema_version='1'"
        )
