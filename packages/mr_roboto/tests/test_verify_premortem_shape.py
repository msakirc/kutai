"""Z1 Tier 5B (A6) — verify_premortem_shape contract tests."""
from __future__ import annotations

import json

import pytest

from mr_roboto.verify_premortem_shape import verify_premortem_shape


def _good_envelope() -> dict:
    return {
        "_schema_version": "1",
        "scenarios": [
            {
                "kind": "technical",
                "obituary": "The cron pipeline silently dropped 12% of orders.",
                "plausibility": 4,
                "cause": "Unhandled retry exhaustion in the queue worker.",
                "mapped_monitoring_rule": "rule_queue_drop_rate",
            },
            {
                "kind": "market",
                "obituary": "Three competitors shipped the same feature first.",
                "plausibility": 3,
                "cause": "Slow feedback loop with target users.",
                "mapped_monitoring_rule": None,
            },
            {
                "kind": "founder",
                "obituary": "Founder lost interest after 5 weeks of no traction.",
                "plausibility": 5,
                "cause": "No early-validation milestone landed.",
                "mapped_monitoring_rule": "rule_weekly_active_user_growth",
            },
        ],
    }


def test_happy_path_envelope():
    res = verify_premortem_shape(premortem=_good_envelope())
    assert res["ok"] is True, res
    assert set(res["kinds_seen"]) == {"technical", "market", "founder"}
    assert res["missing_kinds"] == []
    assert res["problems"] == []


def test_happy_path_text_with_fenced_block():
    env = _good_envelope()
    md = (
        "# Premortem\n\nIt is one year from now.\n\n"
        "```json\n" + json.dumps(env, indent=2) + "\n```\n"
    )
    res = verify_premortem_shape(premortem_text=md)
    assert res["ok"] is True


def test_too_few_scenarios_rejected():
    env = _good_envelope()
    env["scenarios"] = env["scenarios"][:2]  # only 2
    res = verify_premortem_shape(premortem=env)
    assert res["ok"] is False
    assert any("too_few_scenarios" in p for p in res["problems"])
    assert "founder" in res["missing_kinds"]


def test_missing_kind_rejected():
    env = _good_envelope()
    # Remove the founder one — leaves technical + market only.
    env["scenarios"] = [s for s in env["scenarios"] if s["kind"] != "founder"]
    # Replace with a duplicate technical so we still have 3 entries.
    env["scenarios"].append({
        "kind": "technical",
        "obituary": "Same cron blew up again.",
        "plausibility": 2,
        "cause": "Different reason same surface.",
        "mapped_monitoring_rule": None,
    })
    res = verify_premortem_shape(premortem=env)
    assert res["ok"] is False
    assert "founder" in res["missing_kinds"]


def test_plausibility_out_of_range_rejected():
    env = _good_envelope()
    env["scenarios"][0]["plausibility"] = 7  # > 5
    res = verify_premortem_shape(premortem=env)
    assert res["ok"] is False
    assert any(
        "plausibility_out_of_range" in p
        for entry in res["per_item_problems"]
        for p in entry["problems"]
    )


def test_plausibility_zero_rejected():
    env = _good_envelope()
    env["scenarios"][1]["plausibility"] = 0
    res = verify_premortem_shape(premortem=env)
    assert res["ok"] is False


def test_plausibility_non_int_rejected():
    env = _good_envelope()
    env["scenarios"][2]["plausibility"] = "5"  # string
    res = verify_premortem_shape(premortem=env)
    assert res["ok"] is False


def test_missing_obituary_rejected():
    env = _good_envelope()
    env["scenarios"][0]["obituary"] = ""
    res = verify_premortem_shape(premortem=env)
    assert res["ok"] is False


def test_missing_cause_rejected():
    env = _good_envelope()
    env["scenarios"][1]["cause"] = "   "
    res = verify_premortem_shape(premortem=env)
    assert res["ok"] is False


def test_path_read_when_file_missing():
    res = verify_premortem_shape(premortem_path="/nonexistent/premortem.md")
    assert res["ok"] is False
    assert any("read_error" in p for p in res["problems"])


def test_no_input_rejected():
    res = verify_premortem_shape()
    assert res["ok"] is False
    assert "no_input" in res["problems"]


def test_unparseable_text_rejected():
    res = verify_premortem_shape(premortem_text="just some prose, no json here")
    assert res["ok"] is False
    assert any(p.startswith("parse_error") for p in res["problems"])


@pytest.mark.asyncio
async def test_dispatch_via_mr_roboto_run_happy_path():
    import mr_roboto
    env = _good_envelope()
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "verify_premortem_shape",
            "premortem": env,
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed", action


@pytest.mark.asyncio
async def test_dispatch_via_mr_roboto_run_bad_envelope():
    import mr_roboto
    env = _good_envelope()
    env["scenarios"][0]["plausibility"] = 99
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "verify_premortem_shape",
            "premortem": env,
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert "verify_premortem_shape" in (action.error or "")


def test_path_round_trip(tmp_path):
    env = _good_envelope()
    p = tmp_path / "premortem.md"
    p.write_text(
        "# Premortem\n\n```json\n" + json.dumps(env) + "\n```\n",
        encoding="utf-8",
    )
    res = verify_premortem_shape(premortem_path=str(p))
    assert res["ok"] is True
