"""self_critique verdict-resolution tests (SP4-prereq bugfix).

Root cause of mission_79 step 0.1 DLQ: the self_critique sub-iter guard
re-prompts the agent with a "respond with {\"verdict\": ...}" schema, and the
react loop then consumed that verdict ENVELOPE as the task's final result —
clobbering the real artifact (the product charter the agent had already
written to disk). Grade then evaluated the verdict, not the charter, and
failed it every attempt → DLQ.

These tests pin the pure resolver that decides what to do with the agent's
reply to the critique prompt:

  - verdict "clean"  → keep the pre-critique artifact (is_clean True)
  - verdict "issues" + findings → re-emit (is_clean False, fix_message set)
  - unparseable / empty / ambiguous → FAIL OPEN to clean (never destroy a
    good artifact because the critic reply was malformed)

All sync — resolver is a pure function like the rest of self_critique.py.
"""
from __future__ import annotations

import json

from coulson.self_critique import (
    SelfCritiqueResolution,
    parse_self_critique_verdict,
)


def _clean(findings=None):
    return json.dumps({"verdict": "clean", "findings": findings or []})


def _issues(findings):
    return json.dumps({"verdict": "issues", "findings": findings})


# ── clean ────────────────────────────────────────────────────────────────

def test_clean_verdict_is_clean():
    r = parse_self_critique_verdict(_clean(), produces=["a.md"])
    assert isinstance(r, SelfCritiqueResolution)
    assert r.is_clean is True
    assert r.is_verdict is True  # → react restores the pre-critique artifact
    assert r.fix_message is None


def test_clean_verdict_in_prose_is_clean():
    raw = "Here is my review:\n```json\n" + _clean() + "\n```\nLooks good."
    r = parse_self_critique_verdict(raw, produces=["a.md"])
    assert r.is_clean is True
    assert r.is_verdict is True


def test_exact_mission79_clobber_envelope_restores():
    """The literal envelope that clobbered mission_79 step 0.1."""
    r = parse_self_critique_verdict('{"verdict": "clean", "findings": []}', produces=["x.md"])
    assert r.is_verdict is True and r.is_clean is True


# ── issues ───────────────────────────────────────────────────────────────

def test_issues_with_findings_not_clean():
    findings = [{"severity": "error", "file": "a.md", "why": "missing section X"}]
    r = parse_self_critique_verdict(_issues(findings), produces=["a.md"])
    assert r.is_clean is False
    assert r.is_verdict is True
    assert r.fix_message is not None
    assert "missing section X" in r.fix_message
    # re-emit prompt must demand the FULL artifact + the declared path
    assert "a.md" in r.fix_message


def test_issues_empty_findings_treated_clean():
    """verdict=issues but nothing actionable → restore artifact, don't re-emit.

    Still a verdict envelope, so is_verdict True (react must NOT keep the
    envelope — it restores the pre-critique artifact)."""
    r = parse_self_critique_verdict(_issues([]), produces=["a.md"])
    assert r.is_clean is True
    assert r.is_verdict is True


# ── fail-open ──────────────────────────────────────────────────────────────

def test_unparseable_reply_fails_open_clean():
    r = parse_self_critique_verdict("totally not json", produces=["a.md"])
    assert r.is_clean is True
    assert r.is_verdict is False  # not a verdict → react keeps the reply
    assert r.fix_message is None


def test_empty_reply_fails_open_clean():
    for raw in ("", None):
        r = parse_self_critique_verdict(raw, produces=["a.md"])
        assert r.is_clean is True
        assert r.is_verdict is False


def test_missing_verdict_key_fails_open_clean():
    r = parse_self_critique_verdict(json.dumps({"findings": []}), produces=["a.md"])
    assert r.is_clean is True
    assert r.is_verdict is False


def test_artifact_shaped_reply_is_kept_not_a_verdict():
    """If the agent ignored the critique schema and re-sent the artifact
    (markdown, no verdict envelope), it is NOT a verdict — react keeps it."""
    r = parse_self_critique_verdict("# Product Charter\n\n## Positioning\n...", produces=["a.md"])
    assert r.is_clean is True
    assert r.is_verdict is False


# ── loop-mirror integration: prove the clobber is resolved end-to-end ──────
#
# Mirrors the react.run sub-iteration slice (the real loop needs DB +
# dispatcher + model pool). Replicates the stash → critique → resolve control
# flow added in react.py so a behavioral change there must be reflected here.

def _mirror_react_result(replies, *, produces):
    """Return the task result react.run would land on, given queued replies.

    `replies` = list of parsed final_answer dicts the model emits in order:
      replies[0] = the real artifact emission
      replies[1] = the agent's answer to the self_critique prompt
      replies[2:] = any re-emissions
    """
    from coulson.self_critique import check_self_critique_sub_iter

    sub_corrections = 0
    self_critique_passes = 0
    awaiting = False
    pre_parsed = None
    idx = 0
    parsed = None
    for _iter in range(6):
        parsed = replies[min(idx, len(replies) - 1)]
        idx += 1

        if awaiting:
            awaiting = False
            reply = parsed.get("result", "")
            crit = parse_self_critique_verdict(reply, produces=produces)
            if crit.is_verdict and crit.is_clean:
                parsed = pre_parsed            # restore artifact
            elif not crit.is_clean:
                if sub_corrections < 3:
                    sub_corrections += 1
                    continue                   # re-emit (consume next reply)
                parsed = pre_parsed
            # else: not a verdict → keep parsed
            return parsed.get("result")

        corr = check_self_critique_sub_iter(
            parsed, task={"context": json.dumps({"produces": produces})},
            agent_type="writer", self_critique_passes=self_critique_passes,
            tool_calls=[{"name": "write_file", "args": {"filepath": produces[0]}, "ok": True}],
        )
        if corr is not None:
            self_critique_passes += 1
            pre_parsed = parsed
            awaiting = True
            continue
        return parsed.get("result")
    return parsed.get("result") if parsed else None


CHARTER = "# HabitFlow Product Charter\n\n## Product Positioning\n..."


def test_loop_clean_verdict_does_not_clobber_artifact():
    """The mission_79 bug: writer emits charter, critic says clean, result must
    stay the charter — NOT the {"verdict":"clean"} envelope."""
    result = _mirror_react_result(
        [
            {"action": "final_answer", "result": CHARTER},
            {"action": "final_answer", "result": '{"verdict": "clean", "findings": []}'},
        ],
        produces=["mission_79/.charter/product_charter.md"],
    )
    assert result == CHARTER
    assert "verdict" not in result


def test_loop_issues_verdict_reemits_full_artifact():
    fixed = CHARTER + "\n\n## Solutions We Own\n..."
    result = _mirror_react_result(
        [
            {"action": "final_answer", "result": CHARTER},
            {"action": "final_answer", "result": json.dumps(
                {"verdict": "issues", "findings": [
                    {"severity": "error", "file": "x.md", "why": "missing Solutions"}]})},
            {"action": "final_answer", "result": fixed},
        ],
        produces=["x.md"],
    )
    assert result == fixed
