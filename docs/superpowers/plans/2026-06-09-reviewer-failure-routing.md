# Reviewer-Failure Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When an i2p reviewer step emits `status=fail`, the system automatically localises the at-fault producer(s) from the reviewer's structured issues and re-pends them with feedback through the existing retry rail; the founder is escalated only when no target is localisable or a producer exhausts its normal attempts.

**Architecture:** Reviewer output becomes routable (structured `issues[{target_artifact, severity, problem}]` + `status` enum that permits `fail`). A mechanical `verify_review_verdict` check on each reviewer routes a `fail` verdict to a new `route_review_failure` handler: tagged issues map deterministically to producers via an artifact→producer index; untagged issues go to a router LLM; each implicated producer's **existing task row** is re-pended with feedback (so `worker_attempts` carries forward and the existing cap bounds the loop). Founder-halt (regenerate-producer / accept-anyway card) is the last resort.

**Tech Stack:** Python 3.10, pytest, aiosqlite. Packages: `general_beckman` (apply/posthooks), `mr_roboto` (mechanical verifiers), `coulson` (LLM-child prompts), `src/workflows/engine` (workflow index), `src/app/telegram_bot.py` (founder UX). Spec: `docs/superpowers/specs/2026-06-08-reviewer-failure-routing-design.md`.

---

## File structure

- `src/workflows/i2p/i2p_v3.json` — reviewer `status` enums, structured `issues` schema, `checks` wiring, per-issue instruction text (all 11 reviewers).
- `tests/i2p/reviewer_regression/test_reviewer_regression.py` — schema-permits-verdict invariant (drafted in Task 1).
- `src/workflows/engine/producer_index.py` *(new)* — artifact→producer-step index.
- `packages/coulson/src/coulson/posthooks/review_router.py` *(new)* — router LLM prompt + parse + pure tag-mapping.
- `packages/mr_roboto/src/mr_roboto/verify_review_verdict.py` *(new)* — the mechanical verdict reader.
- `packages/mr_roboto/src/mr_roboto/__init__.py` — dispatch arm for `verify_review_verdict`.
- `packages/general_beckman/src/general_beckman/review_routing.py` *(new)* — `route_review_failure` (tag→LLM→group→re-pend; escalation).
- `packages/general_beckman/src/general_beckman/apply.py` — route the `verify_review_verdict` kind to `route_review_failure` (not the default blocker).
- `src/app/telegram_bot.py` — founder-halt card + `rr:` callbacks (regenerate / accept-anyway).

The 11 reviewer step ids: `0.6, 1.7, 1.13, 3.11, 4.16, 6.6, 7.16, 10.5, 11.5, 12.5, 14.2`.

---

## Phase 1 — Schema reconcile + invariant guard

### Task 1: status enum permits the reject verdict (3 confirmed reviewers)

Already implemented in the working tree (uncommitted). This task commits it.

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` (3.11 status enum `["pass"]→["pass","fail"]`; 4.16 + 6.6 `["pass","approved"]→["pass","approved","fail"]`)
- Test: `tests/i2p/reviewer_regression/test_reviewer_regression.py` (new `test_reviewer_schema_permits_emitted_verdict`)

- [ ] **Step 1: Confirm the invariant test is present** (added this session). It parametrizes over `_FIXTURES`, resolves the step's enum'd verdict field, and asserts every non-null fixture verdict is in the enum:

```python
@pytest.mark.parametrize("fixture", _FIXTURES, ids=_fixture_id)
def test_reviewer_schema_permits_emitted_verdict(fixture):
    ver, step_id, path = fixture
    payload = json.loads(path.read_text(encoding="utf-8"))
    wf = _load_workflow()
    step = _step_by_id(wf, step_id)
    schema = step["artifact_schema"][step["output_artifacts"][0]]
    fields = schema.get("fields") or {}
    verdict_field = next(
        (k for k in ("verdict", "status") if k in fields and "equals" in fields[k]),
        None,
    )
    if verdict_field is None:
        pytest.skip("no enum-constrained verdict field")
    actual = (payload.get("expected_verdict") or {}).get(verdict_field)
    if actual is None:
        pytest.skip("fixture does not exercise the verdict field")
    allowed = list(fields[verdict_field].get("equals") or [])
    assert actual in allowed, (
        f"{step_id} {path.name}: {verdict_field}={actual!r} not in schema enum "
        f"{allowed} — the deterministic gate would DLQ this valid verdict"
    )
```

- [ ] **Step 2: Run the invariant test — verify it passes with the enum edits**

Run: `python -m pytest tests/i2p/reviewer_regression/test_reviewer_regression.py -k permits_emitted_verdict -q`
Expected: all pass (the 3.11/4.16/6.6 enum edits are in the tree).

- [ ] **Step 3: Run the full reviewer_regression suite**

Run: `python -m pytest tests/i2p/reviewer_regression/test_reviewer_regression.py -q`
Expected: 42 passed.

- [ ] **Step 4: Validate JSON + commit**

```bash
python -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('ok')"
git add src/workflows/i2p/i2p_v3.json tests/i2p/reviewer_regression/test_reviewer_regression.py
git commit -m "fix(i2p): reviewer status enum permits 'fail' + schema-permits-verdict invariant"
```

### Task 2: audit remaining reviewers for a reject verdict + reconcile

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`
- Test: `tests/i2p/reviewer_regression/test_reviewer_regression.py`

- [ ] **Step 1: Audit each reviewer instruction for an emittable reject verdict**

Run:
```bash
python -c "
import json
wf=json.load(open('src/workflows/i2p/i2p_v3.json',encoding='utf-8'))
by={s['id']:s for s in wf['steps']}
for sid in ('0.6','1.7','1.13','3.11','4.16','6.6','7.16','10.5','11.5','12.5','14.2'):
    s=by[sid]; instr=s.get('instruction','').lower()
    rejects = 'fail' in instr or 'reject' in instr
    out=s['output_artifacts'][0]; flds=s['artifact_schema'][out].get('fields',{})
    vf=next((k for k in ('verdict','status') if k in flds and 'equals' in flds[k]),None)
    enum=flds[vf]['equals'] if vf else None
    print(sid,'rejects=',rejects,'field=',vf,'enum=',enum)
"
```
Expected: prints, per reviewer, whether the instruction can reject and its current enum. Record which reviewers can reject but lack `fail` in the enum.

- [ ] **Step 2: Add `fail` to the `status`/`verdict` enum for every reviewer flagged `rejects=True` whose enum lacks it**

For each such step, edit its enum block in `src/workflows/i2p/i2p_v3.json`, appending `"fail"` (mirror the Task 1 edits). Leave reviewers with `rejects=False` (e.g. 7.16, 12.5, 14.2) unchanged.

- [ ] **Step 3: Run the invariant + full suite**

Run: `python -m pytest tests/i2p/reviewer_regression/test_reviewer_regression.py -q`
Expected: all pass (no fixture verdict is outside its enum).

- [ ] **Step 4: Commit**

```bash
python -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json',encoding='utf-8'))"
git add src/workflows/i2p/i2p_v3.json
git commit -m "fix(i2p): reconcile reject verdict enums across all rejecting reviewers"
```

### Task 3: structured `issues` schema on all 11 reviewers

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` (each reviewer's `artifact_schema[...].fields.issues`)
- Test: `tests/i2p/test_reviewer_issues_schema.py` *(new)*

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path
import pytest

_WF = Path(__file__).resolve().parents[1] / "src" / "workflows" / "i2p" / "i2p_v3.json"
_REVIEWERS = ["0.6","1.7","1.13","3.11","4.16","6.6","7.16","10.5","11.5","12.5","14.2"]

def _by_id():
    wf = json.loads(_WF.read_text(encoding="utf-8"))
    return {s["id"]: s for s in wf["steps"]}

@pytest.mark.parametrize("sid", _REVIEWERS)
def test_reviewer_issues_are_structured(sid):
    s = _by_id()[sid]
    schema = s["artifact_schema"][s["output_artifacts"][0]]
    issues = schema["fields"]["issues"]
    assert issues.get("type") == "array", f"{sid} issues must be an array"
    item = issues.get("items") or {}
    item_fields = set((item.get("fields") or {}).keys())
    assert {"target_artifact", "severity", "problem"} <= item_fields, (
        f"{sid} issue items must carry target_artifact/severity/problem, got {item_fields}"
    )
```

- [ ] **Step 2: Run it — verify it fails**

Run: `python -m pytest tests/i2p/test_reviewer_issues_schema.py -q`
Expected: FAIL — current `issues` is `{}` (not an array with structured items).

- [ ] **Step 3: Edit each reviewer's `issues` schema to the structured shape**

For each of the 11 reviewers, replace its `"issues": {}` (or current shape) with:

```json
"issues": {
  "type": "array",
  "items": {
    "type": "object",
    "fields": {
      "target_artifact": { "type": "string" },
      "severity": { "type": "string", "equals": ["blocker", "major", "minor"] },
      "problem": { "type": "string" }
    }
  }
}
```

`target_artifact` is intentionally NOT in a `required_fields` list (it may be null/absent when an issue is systemic — the router LLM handles those).

- [ ] **Step 4: Run the test — verify it passes**

Run: `python -m pytest tests/i2p/test_reviewer_issues_schema.py -q`
Expected: 11 passed.

- [ ] **Step 5: Update reviewer instructions to emit the structured issues**

For each reviewer, append to its `instruction` string a sentence:

> "Emit `issues` as an array; each issue is `{target_artifact, severity (blocker|major|minor), problem}`. Set `target_artifact` to the name of the artifact the issue is about (one of your input artifacts); use null only when the issue is systemic and cannot be attributed to one artifact."

- [ ] **Step 6: Run prompt-quality + JSON validity, then commit**

```bash
python -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json',encoding='utf-8'))"
python -m pytest tests/i2p/test_reviewer_issues_schema.py tests/i2p/reviewer_regression/test_reviewer_regression.py -q
git add src/workflows/i2p/i2p_v3.json tests/i2p/test_reviewer_issues_schema.py
git commit -m "feat(i2p): structured reviewer issues (target_artifact/severity/problem) on all 11 reviewers"
```

> NOTE: existing reviewer_regression fixtures may carry free-form `issues`. If Step 6 surfaces fixture failures, update each fixture's `expected_verdict.issues` to the structured shape in the same commit (the fixtures are the contract).

---

## Phase 2 — Producer index (pure)

### Task 4: artifact→producer-step index

**Files:**
- Create: `src/workflows/engine/producer_index.py`
- Test: `tests/workflows/test_producer_index.py` *(new)*

- [ ] **Step 1: Write the failing test**

```python
from src.workflows.engine.producer_index import build_producer_index, producers_for_reviewer

def _wf():
    return {"steps": [
        {"id": "a", "output_artifacts": ["x", "y"]},
        {"id": "b", "output_artifacts": ["z"]},
        {"id": "rev", "input_artifacts": ["x", "z"], "output_artifacts": ["rev_result"]},
    ]}

def test_build_index_maps_artifact_to_producers():
    idx = build_producer_index(_wf())
    assert idx["x"] == ["a"]
    assert idx["z"] == ["b"]

def test_producers_for_reviewer_dedups_and_resolves():
    idx = build_producer_index(_wf())
    assert sorted(producers_for_reviewer(_wf(), "rev", idx)) == ["a", "b"]

def test_artifact_maps_to_artifact_producer():
    idx = build_producer_index(_wf())
    assert idx.get("missing") is None
```

- [ ] **Step 2: Run it — verify it fails**

Run: `python -m pytest tests/workflows/test_producer_index.py -q`
Expected: FAIL — `No module named producer_index`.

- [ ] **Step 3: Implement**

```python
"""Map workflow artifacts to the step(s) that produce them, and resolve the
producer set a reviewer step reviews (its input_artifacts → producers)."""
from __future__ import annotations


def build_producer_index(workflow: dict) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for step in workflow.get("steps", []):
        for art in step.get("output_artifacts") or []:
            index.setdefault(art, []).append(step["id"])
    return index


def producers_for_reviewer(
    workflow: dict, reviewer_id: str, index: dict[str, list[str]] | None = None
) -> list[str]:
    index = index or build_producer_index(workflow)
    reviewer = next(
        (s for s in workflow.get("steps", []) if s.get("id") == reviewer_id), None
    )
    if reviewer is None:
        return []
    out: list[str] = []
    for art in reviewer.get("input_artifacts") or []:
        for pid in index.get(art, []):
            if pid not in out and pid != reviewer_id:
                out.append(pid)
    return out


def producer_for_artifact(
    artifact: str, index: dict[str, list[str]]
) -> str | None:
    producers = index.get(artifact) or []
    return producers[0] if producers else None
```

- [ ] **Step 4: Run — verify pass**

Run: `python -m pytest tests/workflows/test_producer_index.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/workflows/engine/producer_index.py tests/workflows/test_producer_index.py
git commit -m "feat(workflow): artifact->producer index + reviewer producer resolution"
```

---

## Phase 3 — Hybrid router (tag mapping pure, LLM fallback)

### Task 5: deterministic tag mapping — issues → producer groups

**Files:**
- Create: `packages/coulson/src/coulson/posthooks/review_router.py`
- Test: `packages/coulson/tests/test_review_router.py` *(new)*

- [ ] **Step 1: Write the failing test**

```python
from coulson.posthooks.review_router import map_tagged_issues

def test_tagged_issues_group_by_producer():
    index = {"requirements_spec": ["3.4"], "prd_final": ["2.11"]}
    issues = [
        {"target_artifact": "requirements_spec", "severity": "blocker", "problem": "no traceability"},
        {"target_artifact": "prd_final", "severity": "major", "problem": "thin NFRs"},
        {"target_artifact": "requirements_spec", "severity": "minor", "problem": "typo"},
    ]
    grouped, unresolved = map_tagged_issues(issues, index)
    assert set(grouped) == {"3.4", "2.11"}
    assert len(grouped["3.4"]) == 2
    assert unresolved == []

def test_untagged_and_unmappable_go_to_unresolved():
    index = {"requirements_spec": ["3.4"]}
    issues = [
        {"target_artifact": None, "severity": "blocker", "problem": "systemic"},
        {"target_artifact": "ghost", "severity": "blocker", "problem": "no producer"},
    ]
    grouped, unresolved = map_tagged_issues(issues, index)
    assert grouped == {}
    assert len(unresolved) == 2
```

- [ ] **Step 2: Run — verify fail**

Run: `python -m pytest packages/coulson/tests/test_review_router.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the pure tag mapping**

```python
"""Route a reviewer's fail issues to the producer step(s) to re-pend.

Tag path (deterministic): issue.target_artifact -> producer via the
artifact->producer index. Untagged/unmappable issues are returned for the
LLM fallback (map_with_llm)."""
from __future__ import annotations

from typing import Any


def map_tagged_issues(
    issues: list[dict], index: dict[str, list[str]]
) -> tuple[dict[str, list[dict]], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    unresolved: list[dict] = []
    for issue in issues:
        art = issue.get("target_artifact")
        producers = index.get(art) if art else None
        if not producers:
            unresolved.append(issue)
            continue
        grouped.setdefault(producers[0], []).append(issue)
    return grouped, unresolved
```

- [ ] **Step 4: Run — verify pass**

Run: `python -m pytest packages/coulson/tests/test_review_router.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/coulson/src/coulson/posthooks/review_router.py packages/coulson/tests/test_review_router.py
git commit -m "feat(coulson): deterministic tag mapping for reviewer-fail routing"
```

### Task 6: router LLM fallback — assign unresolved issues to a producer

**Files:**
- Modify: `packages/coulson/src/coulson/posthooks/review_router.py`
- Test: `packages/coulson/tests/test_review_router.py`

- [ ] **Step 1: Write the failing test (parse + prompt builder, LLM stubbed)**

```python
from coulson.posthooks.review_router import build_router_prompt, parse_router_assignment

def test_build_router_prompt_lists_candidates():
    prompt = build_router_prompt(
        issue={"problem": "auth flow undefined", "severity": "blocker"},
        candidates=[("3.4", "requirements_spec"), ("2.11", "prd_final")],
    )
    assert "auth flow undefined" in prompt
    assert "3.4" in prompt and "2.11" in prompt
    assert "unknown" in prompt.lower()

def test_parse_router_assignment_picks_step():
    assert parse_router_assignment("STEP: 3.4", ["3.4", "2.11"]) == "3.4"

def test_parse_router_assignment_unknown():
    assert parse_router_assignment("STEP: unknown", ["3.4"]) is None

def test_parse_router_rejects_hallucinated_step():
    assert parse_router_assignment("STEP: 9.9", ["3.4"]) is None
```

- [ ] **Step 2: Run — verify fail**

Run: `python -m pytest packages/coulson/tests/test_review_router.py -q`
Expected: FAIL — functions missing.

- [ ] **Step 3: Implement prompt builder + parser**

```python
def build_router_prompt(issue: dict, candidates: list[tuple[str, str]]) -> str:
    lines = [f"- {sid}: produces {art}" for sid, art in candidates]
    return (
        "A reviewer flagged a problem. Pick the single producer step whose "
        "output most likely caused it, or 'unknown' if it cannot be attributed.\n\n"
        f"Problem (severity {issue.get('severity')}): {issue.get('problem')}\n\n"
        "Candidate producer steps:\n" + "\n".join(lines) + "\n\n"
        "Reply with exactly one line: 'STEP: <step_id>' or 'STEP: unknown'."
    )


def parse_router_assignment(raw: str, candidate_ids: list[str]) -> str | None:
    import re
    m = re.search(r"STEP:\s*([^\s]+)", raw or "", re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).strip()
    return val if val in candidate_ids else None
```

The async driver (assigns each unresolved issue via a Beckman OVERHEAD child) is wired in Task 8 where it has access to the dispatch path; the pure prompt/parse are unit-tested here.

- [ ] **Step 4: Run — verify pass**

Run: `python -m pytest packages/coulson/tests/test_review_router.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/coulson/src/coulson/posthooks/review_router.py packages/coulson/tests/test_review_router.py
git commit -m "feat(coulson): router LLM prompt + strict parse for unresolved reviewer issues"
```

---

## Phase 4 — Mechanical verdict reader

### Task 7: `verify_review_verdict` mr_roboto verifier

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/verify_review_verdict.py`
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (dispatch arm, mirror `verify_surfaces_shape` at __init__.py ~1421)
- Test: `packages/mr_roboto/tests/test_verify_review_verdict.py` *(new)*

- [ ] **Step 1: Write the failing test**

```python
import pytest

@pytest.mark.parametrize("status,expected_ok", [("pass", True), ("approved", True)])
def test_pass_class_ok(status, expected_ok):
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={"status": status, "issues": []})
    assert res["ok"] is expected_ok
    assert res["verdict_class"] == "pass"

def test_fail_class_flags_route():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={
        "status": "fail",
        "issues": [{"target_artifact": "x", "severity": "blocker", "problem": "p"}],
    })
    assert res["ok"] is False
    assert res["verdict_class"] == "fail"
    assert res["issues"]

def test_unparseable_is_task_failure_not_route():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result=None)
    assert res["ok"] is False
    assert res["verdict_class"] == "malformed"
```

- [ ] **Step 2: Run — verify fail**

Run: `python -m pytest packages/mr_roboto/tests/test_verify_review_verdict.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
"""Read a reviewer's *_review_result verdict and classify it.

pass-class -> the reviewer accepted the artifact, step completes.
fail-class -> route to general_beckman.review_routing.route_review_failure.
malformed -> the reviewer task itself failed (no parseable verdict): normal DLQ,
not the routing path."""
from __future__ import annotations

from typing import Any

_PASS_CLASS = {"pass", "approved", "needs_minor_fixes"}
_FAIL_CLASS = {"fail"}


def verify_review_verdict(*, review_result: Any) -> dict[str, Any]:
    if not isinstance(review_result, dict) or "status" not in review_result:
        return {"ok": False, "verdict_class": "malformed",
                "error": "no parseable review verdict", "issues": []}
    status = str(review_result.get("status") or "").lower()
    issues = review_result.get("issues") or []
    if status in _FAIL_CLASS:
        return {"ok": False, "verdict_class": "fail", "issues": issues}
    if status in _PASS_CLASS:
        return {"ok": True, "verdict_class": "pass", "issues": issues}
    return {"ok": False, "verdict_class": "malformed",
            "error": f"unknown verdict status {status!r}", "issues": issues}
```

- [ ] **Step 4: Wire the dispatch arm in `__init__.py`** (mirror the `verify_surfaces_shape` arm). Locate the `if action == "verify_surfaces_shape":` block (~line 1421) and add an analogous block:

```python
    if action == "verify_review_verdict":
        from mr_roboto.verify_review_verdict import verify_review_verdict
        res = verify_review_verdict(review_result=payload.get("review_result"))
        if res["verdict_class"] == "pass":
            return Action(status="completed", result=res)
        # fail or malformed: surface so beckman routes it (fail -> route,
        # malformed -> normal DLQ). Beckman's apply path inspects verdict_class.
        return Action(status="failed", error=str(res.get("error") or "review verdict not pass"),
                      result=res)
```

- [ ] **Step 5: Run the verifier + a dispatch smoke test**

Run: `python -m pytest packages/mr_roboto/tests/test_verify_review_verdict.py -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/verify_review_verdict.py packages/mr_roboto/src/mr_roboto/__init__.py packages/mr_roboto/tests/test_verify_review_verdict.py
git commit -m "feat(mr_roboto): verify_review_verdict — classify reviewer verdict (pass/fail/malformed)"
```

---

## Phase 5 — Routing apply path (general_beckman)

> **Executor note:** Tasks 8–9 integrate with `general_beckman/apply.py` internals. Before writing, read: the verdict dispatch around `apply.py:4866` (`if a.kind in _CHECK_KINDS`), `_apply_simple_blocker_verdict`, and `_stamp_retry_feedback` (`apply.py:793`). The re-pend template to mirror is the producer re-pend at `apply.py:2776–2834` (reads `worker_attempts`, `+1`, `_stamp_retry_feedback(ctx, attempts)`, `update_task(status="pending", worker_attempts=attempts, ...)`). Key requirement: re-pend the **producer's** task row, found by `mission_id + resume_name/step_id`, NOT the reviewer's `source` row.

### Task 8: `route_review_failure` — group + re-pend producers

**Files:**
- Create: `packages/general_beckman/src/general_beckman/review_routing.py`
- Test: `packages/general_beckman/tests/test_review_routing.py` *(new)*

- [ ] **Step 1: Write the failing test (DB + dispatch stubbed)**

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_route_repends_tagged_producers_existing_rows():
    from general_beckman.review_routing import route_review_failure
    wf = {"steps": [
        {"id": "3.4", "output_artifacts": ["requirements_spec"]},
        {"id": "3.11", "input_artifacts": ["requirements_spec"], "output_artifacts": ["rr"]},
    ]}
    review_result = {"status": "fail", "issues": [
        {"target_artifact": "requirements_spec", "severity": "blocker", "problem": "no traceability"},
    ]}
    repended = []
    async def fake_repend(mission_id, step_id, feedback):
        repended.append((step_id, feedback))
        return True
    with patch("general_beckman.review_routing._repend_producer", new=fake_repend), \
         patch("general_beckman.review_routing._escalate_to_founder", new=AsyncMock()) as halt:
        outcome = await route_review_failure(
            mission_id=1, reviewer_id="3.11", review_result=review_result, workflow=wf,
        )
    assert ("3.4", ) == tuple(s for s, _ in repended)
    assert "no traceability" in repended[0][1]
    assert outcome["routed"] == ["3.4"]
    halt.assert_not_awaited()

@pytest.mark.asyncio
async def test_route_escalates_when_all_unresolved():
    from general_beckman.review_routing import route_review_failure
    wf = {"steps": [{"id": "3.11", "input_artifacts": [], "output_artifacts": ["rr"]}]}
    review_result = {"status": "fail", "issues": [
        {"target_artifact": None, "severity": "blocker", "problem": "systemic"},
    ]}
    with patch("general_beckman.review_routing._assign_unresolved", new=AsyncMock(return_value={})), \
         patch("general_beckman.review_routing._repend_producer", new=AsyncMock()) as rp, \
         patch("general_beckman.review_routing._escalate_to_founder", new=AsyncMock()) as halt:
        outcome = await route_review_failure(
            mission_id=1, reviewer_id="3.11", review_result=review_result, workflow=wf,
        )
    rp.assert_not_awaited()
    halt.assert_awaited_once()
    assert outcome["escalated"] is True
```

- [ ] **Step 2: Run — verify fail**

Run: `python -m pytest packages/general_beckman/tests/test_review_routing.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the orchestration (pure control flow; IO via the 3 helpers the tests patch)**

```python
"""Autonomous routing of a reviewer 'fail' to the at-fault producer(s).

tag-map (deterministic) -> LLM fallback for unresolved -> re-pend each producer's
EXISTING task row with feedback. Escalate to the founder-halt only when nothing
is localisable. Per-producer attempt bounding is the existing retry rail
(worker_attempts) — no separate budget here."""
from __future__ import annotations

from typing import Any

from src.workflows.engine.producer_index import build_producer_index, producers_for_reviewer
from coulson.posthooks.review_router import map_tagged_issues


def _feedback_text(issues: list[dict]) -> str:
    lines = [f"- [{i.get('severity')}] {i.get('problem')}" for i in issues]
    return "Reviewer rejected this artifact. Fix:\n" + "\n".join(lines)


async def route_review_failure(
    *, mission_id: int, reviewer_id: str, review_result: dict, workflow: dict,
) -> dict[str, Any]:
    issues = review_result.get("issues") or []
    index = build_producer_index(workflow)
    grouped, unresolved = map_tagged_issues(issues, index)

    if unresolved:
        candidates = [
            (pid, art)
            for art in (next((s for s in workflow["steps"] if s["id"] == reviewer_id), {}).get("input_artifacts") or [])
            for pid in index.get(art, [])
        ]
        assigned = await _assign_unresolved(unresolved, candidates)
        for pid, issue in assigned.items() if isinstance(assigned, dict) else []:
            grouped.setdefault(pid, []).extend(issue if isinstance(issue, list) else [issue])

    if not grouped:
        await _escalate_to_founder(
            mission_id=mission_id, reviewer_id=reviewer_id,
            review_result=review_result, workflow=workflow, reason="no_localisable_target",
        )
        return {"routed": [], "escalated": True}

    routed: list[str] = []
    for pid, pissues in grouped.items():
        ok = await _repend_producer(mission_id, pid, _feedback_text(pissues))
        if ok:
            routed.append(pid)
        else:
            await _escalate_to_founder(
                mission_id=mission_id, reviewer_id=reviewer_id,
                review_result=review_result, workflow=workflow,
                reason="producer_exhausted", producer=pid,
            )
    return {"routed": routed, "escalated": False}
```

- [ ] **Step 4: Run — verify pass** (helpers `_assign_unresolved`, `_repend_producer`, `_escalate_to_founder` are patched in the tests; their real bodies land in Task 9)

Run: `python -m pytest packages/general_beckman/tests/test_review_routing.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/review_routing.py packages/general_beckman/tests/test_review_routing.py
git commit -m "feat(beckman): route_review_failure orchestration (tag->LLM->re-pend, escalate)"
```

### Task 9: routing IO helpers + wire into apply dispatch

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/review_routing.py` (`_repend_producer`, `_assign_unresolved`, `_escalate_to_founder`)
- Modify: `packages/general_beckman/src/general_beckman/apply.py` (route `verify_review_verdict` verdict to `route_review_failure`; `malformed` → normal DLQ)
- Test: `packages/general_beckman/tests/test_review_routing_io.py` *(new)*

- [ ] **Step 1: Write the failing test for `_repend_producer` (real DB via the test harness)**

```python
import pytest

@pytest.mark.asyncio
async def test_repend_increments_existing_worker_attempts(tmp_db_mission_with_producer):
    # Fixture seeds a mission with a COMPLETED producer task '3.4' at worker_attempts=1.
    from general_beckman.review_routing import _repend_producer
    from src.infra.db import get_task_by_step
    ok = await _repend_producer(mission_id=tmp_db_mission_with_producer, step_id="3.4",
                                feedback="fix X")
    assert ok is True
    row = await get_task_by_step(tmp_db_mission_with_producer, "3.4")
    assert row["status"] == "pending"
    assert int(row["worker_attempts"]) == 2  # incremented, not reset
    assert "fix X" in (row.get("retry_feedback") or "")
```

> The executor adapts `get_task_by_step` to the actual db helper (search `db.py` for the existing "task by mission_id + step/resume_name" query; if none, the producer re-pend keys on `mission_id` + `context.workflow_step_id`). The fixture mirrors existing real-DB beckman tests under `packages/general_beckman/tests/`.

- [ ] **Step 2: Run — verify fail**

Run: `python -m pytest packages/general_beckman/tests/test_review_routing_io.py -q`
Expected: FAIL — `_repend_producer` raises NotImplementedError / returns None.

- [ ] **Step 3: Implement the helpers** by mirroring the existing producer re-pend (`apply.py:2776–2834`): look up the producer task row by `mission_id`+step id; if found and below its `max_worker_attempts`, `attempts = worker_attempts+1`, build `ctx`, call `_stamp_retry_feedback(ctx, attempts)`, `update_task(status="pending", worker_attempts=attempts, ...)` and return True; if at cap, return False (→ escalate). `_assign_unresolved` runs an OVERHEAD Beckman child per unresolved issue using `build_router_prompt` + `parse_router_assignment` (Task 6). `_escalate_to_founder` enqueues the founder-halt (Task 10) by setting the reviewer task `waiting_human` and sending the card.

```python
from general_beckman.review_routing import map_tagged_issues  # already imported
from coulson.posthooks.review_router import build_router_prompt, parse_router_assignment
from general_beckman.posthooks import _stamp_retry_feedback  # adjust import to its real home (apply.py)


async def _repend_producer(mission_id: int, step_id: str, feedback: str) -> bool:
    from src.infra.db import find_task_by_step, update_task
    row = await find_task_by_step(mission_id, step_id)
    if not row:
        return False
    attempts = int(row.get("worker_attempts") or 0) + 1
    max_attempts = int(row.get("max_worker_attempts") or 15)
    if attempts > max_attempts:
        return False
    ctx = dict(row.get("context") or {})
    ctx["retry_feedback"] = feedback
    _stamp_retry_feedback(ctx, attempts)
    await update_task(row["id"], status="pending", worker_attempts=attempts,
                      max_worker_attempts=max_attempts, context=ctx)
    return True
```

(`find_task_by_step` — use the real db helper; if absent, add a thin query `SELECT * FROM tasks WHERE mission_id=? AND json_extract(context,'$.workflow_step_id')=? ORDER BY id DESC LIMIT 1`.)

- [ ] **Step 4: Route the verdict in `apply.py`** — at the `verify_review_verdict` kind, branch on `result.verdict_class`:

```python
    if a.kind == "verify_review_verdict":
        res = (a.result or {}) if hasattr(a, "result") else {}
        vclass = res.get("verdict_class")
        if vclass == "fail":
            from general_beckman.review_routing import route_review_failure
            from src.workflows.engine.loader import load_workflow
            wf = load_workflow(source.get("workflow") or "i2p_v3").raw  # adapt to loader API
            await route_review_failure(
                mission_id=source["mission_id"], reviewer_id=_step_id_of(source),
                review_result=res.get("review_result") or {}, workflow=wf,
            )
            return
        # malformed -> the reviewer task genuinely failed: normal DLQ path.
        await _apply_z1_mechanical_verdict(source=source, ctx=ctx, pending=pending, verdict=a)
        return
```

(`_step_id_of` / loader `.raw` — adapt to the real helpers; the executor verifies the verdict object carries `review_result` by having the check payload include it, see Task 11.)

- [ ] **Step 5: Run the IO test + routing test**

Run: `python -m pytest packages/general_beckman/tests/test_review_routing_io.py packages/general_beckman/tests/test_review_routing.py -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/src/general_beckman/review_routing.py packages/general_beckman/src/general_beckman/apply.py packages/general_beckman/tests/test_review_routing_io.py
git commit -m "feat(beckman): re-pend producers on reviewer fail + wire verdict dispatch"
```

---

## Phase 6 — Founder-halt fallback (Telegram)

### Task 10: founder-halt card + regenerate/accept callbacks

**Files:**
- Modify: `src/app/telegram_bot.py` (`send_review_halt_keyboard`, `_handle_review_halt` for `rr:` callbacks; register handler — mirror the `sc:` surface_choice pattern added this session)
- Modify: `packages/general_beckman/src/general_beckman/review_routing.py` (`_escalate_to_founder` sends the card + sets `waiting_human`)
- Test: `tests/app/test_review_halt_callback.py` *(new)*

- [ ] **Step 1: Write the failing test (mirror test_surface_choice_callback.py)**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_regenerate_repends_chosen_producer():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}
    update = MagicMock()
    update.effective_chat.id = 7
    update.callback_query.data = "rr:regen:9:3.4"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    with patch("general_beckman.review_routing._repend_producer", new=AsyncMock(return_value=True)) as rp:
        await iface._handle_review_halt(update, MagicMock())
    rp.assert_awaited_once()
    assert rp.call_args.kwargs.get("step_id") == "3.4" or rp.call_args.args[1] == "3.4"

@pytest.mark.asyncio
async def test_accept_anyway_overrides_and_completes():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}
    update = MagicMock()
    update.effective_chat.id = 7
    update.callback_query.data = "rr:accept:9:3.11"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    with patch("src.infra.db.update_task", new=AsyncMock()) as upd, \
         patch("src.infra.db.record_action_event", new=AsyncMock()) as audit:
        await iface._handle_review_halt(update, MagicMock())
    upd.assert_awaited()  # reviewer task completed (overridden)
    audit.assert_awaited()  # override recorded
```

- [ ] **Step 2: Run — verify fail**

Run: `python -m pytest tests/app/test_review_halt_callback.py -q`
Expected: FAIL — handler missing.

- [ ] **Step 3: Implement `send_review_halt_keyboard` + `_handle_review_halt`** (mirror `send_surface_keyboard`/`_handle_surface_choice` added this session). callback_data: `rr:regen:{reviewer_task_id}:{producer_step}` and `rr:accept:{reviewer_task_id}:{reviewer_step}`. Register `CallbackQueryHandler(self._handle_review_halt, pattern=r"^rr:")` next to the `sc:` handler. `regen` → `await _repend_producer(mission_id, producer_step, feedback)`; `accept` → `update_task(reviewer_task_id, status="completed")` + `record_action_event(verb="review_override", ...)`.

- [ ] **Step 4: Run — verify pass**

Run: `python -m pytest tests/app/test_review_halt_callback.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/app/telegram_bot.py packages/general_beckman/src/general_beckman/review_routing.py tests/app/test_review_halt_callback.py
git commit -m "feat(telegram): reviewer-fail founder-halt card + regenerate/accept callbacks"
```

---

## Phase 7 — Wire the check onto reviewers + integration

### Task 11: attach `verify_review_verdict` check to all 11 reviewers

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` (add a `checks` entry per reviewer)
- Test: `tests/i2p/test_reviewer_checks_wired.py` *(new)*

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path
import pytest

_WF = Path(__file__).resolve().parents[1] / "src" / "workflows" / "i2p" / "i2p_v3.json"
_REVIEWERS = ["0.6","1.7","1.13","3.11","4.16","6.6","7.16","10.5","11.5","12.5","14.2"]

@pytest.mark.parametrize("sid", _REVIEWERS)
def test_reviewer_has_verdict_check(sid):
    wf = json.loads(_WF.read_text(encoding="utf-8"))
    s = next(x for x in wf["steps"] if x["id"] == sid)
    actions = {(c.get("payload") or {}).get("action") for c in (s.get("checks") or [])}
    assert "verify_review_verdict" in actions, f"{sid} missing verify_review_verdict check"
```

- [ ] **Step 2: Run — verify fail**

Run: `python -m pytest tests/i2p/test_reviewer_checks_wired.py -q`
Expected: FAIL.

- [ ] **Step 3: Add the check to each reviewer.** For each of the 11, add to its `checks` array (create the array if absent):

```json
{
  "kind": "verify_review_verdict",
  "payload": {
    "action": "verify_review_verdict",
    "review_result_artifact": "<the step's output_artifacts[0]>"
  }
}
```

The expander must populate `payload.review_result` from the produced artifact at apply time (so the verdict reader + router see it). If the check-payload pipeline doesn't already inline the produced artifact, extend the expander/posthook to read `review_result_artifact` from the blackboard — mirror how `verify_surfaces_shape` reads its file. Verify against an existing artifact-consuming check.

- [ ] **Step 4: Run — verify pass + full reviewer suites**

Run: `python -m pytest tests/i2p/test_reviewer_checks_wired.py tests/i2p/test_reviewer_issues_schema.py tests/i2p/reviewer_regression/test_reviewer_regression.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
python -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json',encoding='utf-8'))"
git add src/workflows/i2p/i2p_v3.json tests/i2p/test_reviewer_checks_wired.py
git commit -m "feat(i2p): wire verify_review_verdict check onto all 11 reviewers"
```

### Task 12: end-to-end integration test

**Files:**
- Test: `tests/integration/test_reviewer_fail_routing_e2e.py` *(new)*

- [ ] **Step 1: Write the integration test** — seed a mission with a producer (`3.4`, completed, `worker_attempts=1`) and reviewer (`3.11`); feed a `fail` review_result tagging `requirements_spec`; run the verdict dispatch; assert the producer row is re-pended (`status=pending`, `worker_attempts=2`, feedback present) and no founder card was sent. Then feed a `fail` with `target_artifact=null` and assert the founder-halt path fires (card sent, reviewer `waiting_human`).

```python
import pytest

@pytest.mark.asyncio
async def test_tagged_fail_repends_producer_no_founder(seed_mission_3_11):
    # seed_mission_3_11: mission with producer 3.4 (completed, attempts=1) + reviewer 3.11
    from general_beckman.review_routing import route_review_failure
    from src.workflows.engine.loader import load_workflow
    wf = load_workflow("i2p_v3").raw
    rr = {"status": "fail", "issues": [
        {"target_artifact": "requirements_spec", "severity": "blocker", "problem": "no traceability"}]}
    out = await route_review_failure(mission_id=seed_mission_3_11, reviewer_id="3.11",
                                     review_result=rr, workflow=wf)
    assert out["escalated"] is False and out["routed"]
```

- [ ] **Step 2: Run — confirm it fails for the right reason, implement any missing seam, then pass**

Run: `python -m pytest tests/integration/test_reviewer_fail_routing_e2e.py -q`
Expected: pass after wiring gaps closed.

- [ ] **Step 3: Run the whole reviewer + routing surface**

Run: `python -m pytest tests/i2p/ packages/general_beckman/tests/test_review_routing.py packages/general_beckman/tests/test_review_routing_io.py packages/mr_roboto/tests/test_verify_review_verdict.py tests/app/test_review_halt_callback.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_reviewer_fail_routing_e2e.py
git commit -m "test(i2p): e2e reviewer-fail routing — tagged re-pend + founder escalation"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** schema reconcile (T1–2), structured issues (T3), producer index (T4), tag map + LLM fallback (T5–6), verdict reader (T7), routing + re-pend + escalation (T8–9), founder-halt card (T10), check wiring (T11), e2e (T12). All spec sections map to a task.
- **Known seams to verify at execution (flagged, not placeholders):** (a) the db helper that finds a task by mission+step (`find_task_by_step` / `json_extract(context,'$.workflow_step_id')`); (b) `load_workflow(...).raw` vs the loader's real accessor; (c) how a check payload inlines the produced artifact (`review_result`) — mirror `verify_surfaces_shape`'s artifact read; (d) the re-pend cascade actually re-runs the reviewer after its producer completes (the reviewer `depends_on` the producer — confirm the pump re-pends dependents, else add the reviewer to the re-pend set). Each is a concrete lookup with a named template, not an open design choice.
- **Type consistency:** `verdict_class ∈ {pass, fail, malformed}`; `route_review_failure(mission_id, reviewer_id, review_result, workflow)`; `_repend_producer(mission_id, step_id, feedback) -> bool`; callback ids `rr:regen:<task>:<producer>` / `rr:accept:<task>:<reviewer>`.
- **Bot hazard:** the live bot git-add-A commits to main. Execute this plan in a git worktree (superpowers:using-git-worktrees), commit per task with explicit paths.
