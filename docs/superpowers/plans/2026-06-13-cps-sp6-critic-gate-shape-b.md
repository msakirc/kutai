# CPS SP6 — Critic Gate shape-(b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the critic gate (second-LLM veto before irreversible actions) an *admitted* task that runs before the gated action, and make it *fail-closed* — so a broken/unavailable critic blocks the action instead of waving it through, and the verdict LLM call is visible to admission.

**Architecture:** Three surfaces. **A (posthook `post_hooks:["critic_gate"]`)** converts from a mechanical child that calls `husam.run` inline → a raw_dispatch LLM child + CPS resume handler (mirrors grade/code_review; the existing `_Z1_MECHANICAL_KINDS` verdict rail already DLQs the source on veto). **B (inline mechanical gates `git_commit`/`notify_user`)** converts from an inline `husam.run` mid-executor → a two-pass self-park: pass-1 enqueues an admitted critic child + parks the gated task (`waiting_human`+`needs_clarification`, the proven human-confirm-gate mechanic); a CPS resume handler stamps the verdict + re-pends; pass-2 calls the LLM-free `confirm_gate`. **C (standalone `critic_gate` action)** is deleted (its only prod caller was A). `confirm_gate` rule 3 flips fail-open → fail-closed. The inline `produce_verdict`/`critic_gate` orchestrator funcs become dead and are deleted.

**Tech Stack:** Python 3.10 async, aiosqlite, pytest (`timeout` prefix MANDATORY — never run pytest without it; never run two pytest invocations concurrently — SQLite WAL lock crash-loops live KutAI). Packages: `general_beckman` (admission/continuations/verdict rail), `mr_roboto` (mechanical executors + critic_gate), `husam` (single-call worker), `src/core` (orchestrator).

**Source of truth spec:** `docs/superpowers/specs/2026-06-13-cps-sp6-critic-gate-shape-b-design.md` (read it first; this plan implements it).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `packages/mr_roboto/src/mr_roboto/critic_gate.py` | critic prompt build + verdict parse + redact + persist + `confirm_gate` | MODIFY: fail-closed flip; DELETE inline `produce_verdict`+`critic_gate` orchestrator |
| `packages/mr_roboto/src/mr_roboto/critic_continuations.py` | **NEW** — surface-B CPS resume handlers (`verdict_done`/`verdict_err`) + `register_continuations()` | CREATE |
| `packages/mr_roboto/src/mr_roboto/__init__.py` | mechanical dispatcher | MODIFY: `git_commit` + `notify_user` → two-pass; DELETE standalone `critic_gate` action |
| `packages/general_beckman/src/general_beckman/apply.py` | posthook spawn + verdict rail | MODIFY: route `critic_gate` posthook to LLM child; add `critic_gate` kind to `_enqueue_posthook_llm_child` |
| `packages/general_beckman/src/general_beckman/posthook_continuations.py` | posthook CPS resume handlers | MODIFY: add `_critic_resume`/`_critic_resume_err` + register |
| `packages/general_beckman/src/general_beckman/continuations.py` | `_HANDLER_MODULES` registry | MODIFY: append `mr_roboto.critic_continuations` |
| `src/core/orchestrator.py` (sweep) | `on_task_finished` legacy straggler shim | MODIFY: delete `__init__.py:1309-1329`-style shim per sweep |
| `src/tools/vision.py` (sweep) | comment only | MODIFY |
| `CLAUDE.md` (sweep) | one-line note | MODIFY |
| `tests/` + `packages/*/tests/` | coverage | CREATE/MODIFY/DELETE |

**Task order:** Task 1 (safety flip, isolated) → Task 2 (surface A) → Task 3 (surface B git_commit) → Task 4 (surface B notify_user) → Task 5 (delete dead inline funcs + surface C + tests) → Task 6 (sweep companions). Tasks 1–4 each leave the tree green; Task 5 only runs after A+B prove the inline funcs are dead.

---

### Task 1: `confirm_gate` fail-CLOSED flip

The single safety change. Today `confirm_gate` rule 3 (missing/garbage verdict) returns default-`pass`. Flip it: when the gate is ENABLED and no usable verdict is present → return `verdict="veto"`. The `KUTAI_CRITIC_GATE=off` opt-out (rule 1) stays the ONLY pass-without-verdict path.

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/critic_gate.py:326-333`
- Test: `packages/mr_roboto/tests/test_critic_gate_split.py`

- [ ] **Step 1: Write the failing test**

Add to `packages/mr_roboto/tests/test_critic_gate_split.py`:

```python
import os
import pytest
from mr_roboto import critic_gate as cg


@pytest.mark.asyncio
async def test_confirm_gate_fail_closed_on_missing_verdict(monkeypatch):
    """Gate ENABLED + no usable verdict → VETO (fail-closed), not pass."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)  # gate on
    res = await cg.confirm_gate("git_commit", {"x": 1}, persisted_verdict=None)
    assert res["verdict"] == "veto"
    assert res["bypassed"] is False


@pytest.mark.asyncio
async def test_confirm_gate_fail_closed_on_garbage_verdict(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    res = await cg.confirm_gate("git_commit", {"x": 1},
                                persisted_verdict={"verdict": "banana"})
    assert res["verdict"] == "veto"


@pytest.mark.asyncio
async def test_confirm_gate_optout_still_passes_without_verdict(monkeypatch):
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    res = await cg.confirm_gate("git_commit", {"x": 1}, persisted_verdict=None)
    assert res["verdict"] == "pass"
    assert res["bypassed"] is True


@pytest.mark.asyncio
async def test_confirm_gate_honours_real_verdict(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    pv = {"verdict": "veto", "reasons": ["leaks a secret"]}
    res = await cg.confirm_gate("git_commit", {"x": 1}, persisted_verdict=pv)
    assert res["verdict"] == "veto"
    assert res["reasons"] == ["leaks a secret"]
    pv2 = {"verdict": "pass", "reasons": []}
    res2 = await cg.confirm_gate("git_commit", {"x": 1}, persisted_verdict=pv2)
    assert res2["verdict"] == "pass"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_gate_split.py::test_confirm_gate_fail_closed_on_missing_verdict -v`
Expected: FAIL — current code returns `verdict="pass"` for a missing verdict.

- [ ] **Step 3: Flip rule 3 to fail-closed**

In `packages/mr_roboto/src/mr_roboto/critic_gate.py`, replace the rule-3 block (currently lines ~326-333):

```python
    if verdict not in {"pass", "veto"}:
        # Missing/garbage verdict → fail-open.
        return {
            "verdict": "pass",
            "reasons": reasons or ["no critic verdict available — default-passing"],
            "bypassed": False,
            "payload_hash": payload_hash,
        }
```

with the fail-CLOSED version:

```python
    if verdict not in {"pass", "veto"}:
        # SP6: gate is ENABLED (opt-out handled above) but no usable verdict
        # is present (producer never ran / failed / garbage). FAIL-CLOSED — a
        # broken critic must BLOCK the irreversible action, never wave it
        # through. KUTAI_CRITIC_GATE=off (rule 1) is the only pass-without-
        # verdict path.
        return {
            "verdict": "veto",
            "reasons": reasons or ["no critic verdict available — fail-closed"],
            "bypassed": False,
            "payload_hash": payload_hash,
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_gate_split.py -v -k confirm_gate`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/critic_gate.py packages/mr_roboto/tests/test_critic_gate_split.py
git commit -m "feat(critic_gate): confirm_gate fails CLOSED on missing verdict (SP6 T1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Surface A — posthook `critic_gate` → admitted LLM child

Convert the `post_hooks:["critic_gate"]` posthook (only live use: i2p `init_mission_github_repo`, i2p_v3.json:5707) from a mechanical child that re-enters `husam.run` inline → a raw_dispatch LLM child + CPS resume. The verdict rail already DLQs on veto (`critic_gate` ∈ `_Z1_MECHANICAL_KINDS`), so no new verdict branch is needed.

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/apply.py` (route at ~1417; add kind branch at ~1860)
- Modify: `packages/general_beckman/src/general_beckman/posthook_continuations.py` (add handlers + register)
- Test: `packages/general_beckman/tests/test_critic_posthook_cps.py` (new)

- [ ] **Step 1: Write the failing test (resume builds a DLQ-ing verdict on veto)**

Create `packages/general_beckman/tests/test_critic_posthook_cps.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
import general_beckman.posthook_continuations as pc


@pytest.mark.asyncio
async def test_critic_resume_veto_builds_failing_verdict():
    """A veto child result → PostHookVerdict(kind='critic_gate', passed=False)
    applied to the source (the _Z1_MECHANICAL rail then DLQs it)."""
    captured = {}

    async def _fake_apply(child_task, verdict):
        captured["verdict"] = verdict

    result = {"result": {"content": '{"verdict": "veto", "reasons": ["leaks a token"]}'}}
    state = {"source_task_id": 42, "action_name": "git_commit", "mission_id": 7}
    with patch.object(pc, "_apply_posthook_verdict", _fake_apply), \
         patch.object(pc, "_persist_critic_log", AsyncMock()):
        await pc._critic_resume(child_task_id=99, result=result, state=state)

    v = captured["verdict"]
    assert v.kind == "critic_gate"
    assert v.passed is False
    assert v.source_task_id == 42


@pytest.mark.asyncio
async def test_critic_resume_pass_builds_passing_verdict():
    captured = {}

    async def _fake_apply(child_task, verdict):
        captured["verdict"] = verdict

    result = {"result": {"content": '{"verdict": "pass", "reasons": []}'}}
    state = {"source_task_id": 42, "action_name": "git_commit"}
    with patch.object(pc, "_apply_posthook_verdict", _fake_apply), \
         patch.object(pc, "_persist_critic_log", AsyncMock()):
        await pc._critic_resume(child_task_id=99, result=result, state=state)
    assert captured["verdict"].passed is True


@pytest.mark.asyncio
async def test_critic_resume_err_fail_closed():
    """Terminal producer error → fail-closed: passed=False verdict applied."""
    captured = {}

    async def _fake_apply(child_task, verdict):
        captured["verdict"] = verdict

    state = {"source_task_id": 42, "action_name": "notify_user"}
    with patch.object(pc, "_apply_posthook_verdict", _fake_apply):
        await pc._critic_resume_err(child_task_id=99,
                                    result={"error": "no candidates"}, state=state)
    assert captured["verdict"].kind == "critic_gate"
    assert captured["verdict"].passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/general_beckman/tests/test_critic_posthook_cps.py -v`
Expected: FAIL — `pc._critic_resume` / `_critic_resume_err` / `_persist_critic_log` do not exist.

- [ ] **Step 3: Add the resume handlers + register them**

In `packages/general_beckman/src/general_beckman/posthook_continuations.py`, add before `def register_continuations()`:

```python
# ──────────────────────────────────────────────────────────────────────────
# SP6 — critic_gate posthook resume (admitted LLM child, fail-closed).
#
# Unlike grade (which retry-escalates), critic_gate is a SINGLE-SHOT veto:
# critic_gate ∈ _Z1_BLOCKER_KINDS ⊂ _Z1_MECHANICAL_KINDS, so
# _apply_posthook_verdict_locked routes a passed=False verdict through
# _apply_z1_mechanical_verdict → single-shot DLQ of the source. The handler's
# only job is to build the right PostHookVerdict.
# ──────────────────────────────────────────────────────────────────────────


async def _persist_critic_log(state: dict, verdict: str, reasons: list) -> None:
    """Persist one critic_log row (best-effort; never raises)."""
    from mr_roboto.critic_gate import _persist
    await _persist(
        state.get("mission_id"),
        str(state.get("action_name") or "unknown"),
        verdict,
        list(reasons or []),
        str(state.get("payload_hash") or ""),
    )


def _make_critic_verdict(source_task_id, passed: bool, reasons: list):
    from general_beckman.result_router import PostHookVerdict
    return PostHookVerdict(
        source_task_id=source_task_id, kind="critic_gate",
        passed=passed, raw={"reasons": list(reasons or [])},
    )


async def _critic_resume(child_task_id: int, result: dict, state: dict) -> None:
    """Resume after a critic child completed. Parse verdict; persist log;
    apply a single-shot gate verdict (veto→passed=False→source DLQ)."""
    from mr_roboto.critic_gate import _parse_verdict
    source_task_id = state.get("source_task_id")
    parsed = _parse_verdict(_extract_content(result))
    passed = parsed["verdict"] != "veto"
    await _persist_critic_log(state, parsed["verdict"], parsed.get("reasons") or [])
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_critic_verdict(source_task_id, passed, parsed.get("reasons") or []),
    )


async def _critic_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    """On_error: the critic child failed terminally. FAIL-CLOSED — apply a
    passed=False verdict so the source DLQs (action blocked), never default-pass."""
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    reasons = [f"critic verdict unavailable (producer error: {str(err)[:120]}) — fail-closed"]
    await _persist_critic_log(state, "veto", reasons)
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_critic_verdict(source_task_id, False, reasons),
    )
```

Then inside `register_continuations()`, add after the self_reflect lines:

```python
        # SP6 — critic_gate posthook (surface A).
        register_resume("posthook.critic.resume", _critic_resume)
        register_resume("posthook.critic.resume_err", _critic_resume_err)
```

- [ ] **Step 4: Run resume tests — verify they pass**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/general_beckman/tests/test_critic_posthook_cps.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Add the `critic_gate` kind branch to `_enqueue_posthook_llm_child`**

In `packages/general_beckman/src/general_beckman/apply.py`, in `_enqueue_posthook_llm_child` (the if/elif kind-chain ending with the `else: raise ValueError(...)` at ~1860), add a new `elif` before the `else`:

```python
    elif kind == "critic_gate":
        # SP6 — admitted critic child. action_name + payload come from the
        # source step context (set by the workflow step that declared
        # post_hooks:["critic_gate"]). Build the redacted critic spec via the
        # mr_roboto helper; resume single-shots a gate verdict.
        from mr_roboto.critic_gate import _build_critic_spec, _redact_payload, _hash_payload
        action_name = str(
            source_ctx.get("critic_action_name")
            or source_ctx.get("step_id")
            or "unknown"
        )
        raw_payload = source_ctx.get("critic_target_payload")
        redacted = _redact_payload(raw_payload)
        built = _build_critic_spec(action_name, redacted)
        on_complete = "posthook.critic.resume"
        on_error = "posthook.critic.resume_err"
        cont_state = {
            "source_task_id": source_id,
            "kind": "critic_gate",
            "action_name": action_name,
            "payload_hash": _hash_payload(redacted),
            "mission_id": source.get("mission_id"),
        }
```

(Match the local variable names already used by the sibling branches — `source_id`, `built`, `on_complete`, `on_error`, `cont_state` — so the shared `await enqueue(built, ...)` tail at ~1871 picks them up. Verify those names against the grade branch before editing.)

- [ ] **Step 6: Route the posthook spawn from mechanical → LLM child**

In `packages/general_beckman/src/general_beckman/apply.py`, find the LLM-child route condition (~1417):

```python
    if (
        a.kind == "grade"
        or a.kind == "code_review"
        or a.kind.startswith("summary:")
        or a.kind in _REWRITE_POSTHOOK_KINDS
    ):
        await _enqueue_posthook_llm_child(a.kind, source, posthook_ctx)
        return
```

Add `critic_gate` to it:

```python
    if (
        a.kind == "grade"
        or a.kind == "code_review"
        or a.kind == "critic_gate"          # SP6 — admitted critic child
        or a.kind.startswith("summary:")
        or a.kind in _REWRITE_POSTHOOK_KINDS
    ):
        await _enqueue_posthook_llm_child(a.kind, source, posthook_ctx)
        return
```

Then DELETE the now-dead mechanical-tuple block `if a.kind == "critic_gate": return ("mechanical", {...})` (apply.py:2357-2376) — the LLM-child route above now handles it before this code is reached. (Re-verify the route runs first; if the mechanical builder is a separate function reached independently, leave a `# critic_gate handled by LLM-child route` breadcrumb instead of a dangling branch.)

- [ ] **Step 7: Write a real-DB pump integration test (admitted proof + veto DLQs source)**

Create `packages/general_beckman/tests/test_critic_posthook_pump.py` — drive a source task with `post_hooks:["critic_gate"]` through enqueue→pick, assert (a) a child task row exists with `lane='oneshot'` (NOT a phantom lane) and `agent_type='critic'`, and (b) feeding a veto child result through `on_task_finished` DLQs the source. Use the existing real-DB pump test in `packages/general_beckman/tests/` as the template (the one added for the SP3b `lane="overhead"` regression — grep `pick_ready_top_k` in tests). Mock `husam.run` to return `{"content": '{"verdict":"veto","reasons":["x"]}'}`.

```python
import pytest
# (follow the existing real-DB pump fixture in this dir; pattern:)
#   src = await enqueue({..., "context": {"output_artifacts": [...],
#                        "_pending_posthooks": ["critic_gate"],
#                        "critic_action_name": "init_mission_github_repo",
#                        "critic_target_payload": {"repo_visibility": "public"}}})
#   run post_execute / verdict path; assert child row lane == "oneshot"
#   then fire on_task_finished(child_id, veto_result); assert source status DLQ
```

- [ ] **Step 8: Run the full surface-A suite + beckman regression**

Run: `timeout 120 .venv/Scripts/python -m pytest packages/general_beckman/tests/test_critic_posthook_cps.py packages/general_beckman/tests/test_critic_posthook_pump.py -v`
Then: `timeout 120 .venv/Scripts/python -m pytest packages/general_beckman/tests/ -x -q`
Expected: PASS (new) + no regressions in beckman suite (known pre-existing reds noted in memory `project_sp5_await_inline_closed_20260611` — `test_admission_cache`×2 — are harness artifacts, NOT introduced here; confirm they were red before your change with `git stash`).

- [ ] **Step 9: Commit**

```bash
git add packages/general_beckman/src/general_beckman/apply.py \
        packages/general_beckman/src/general_beckman/posthook_continuations.py \
        packages/general_beckman/tests/test_critic_posthook_cps.py \
        packages/general_beckman/tests/test_critic_posthook_pump.py
git commit -m "feat(critic_gate): surface A posthook -> admitted LLM child + CPS resume (SP6 T2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Surface B — `git_commit` two-pass self-park

Convert the inline-gate `git_commit` executor (mr_roboto/__init__.py:810-981) to two passes: pass-1 captures the diff, enqueues an admitted critic child, parks the gated task; a CPS resume handler stamps the verdict + re-pends; pass-2 calls the LLM-free `confirm_gate` and commits or vetoes. Bounded re-gate guards staging drift.

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/critic_continuations.py`
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (module-level imports + git_commit executor)
- Modify: `packages/general_beckman/src/general_beckman/continuations.py` (`_HANDLER_MODULES`)
- Test: `packages/mr_roboto/tests/test_critic_two_pass.py` (new)

- [ ] **Step 0: Add module-level `enqueue`/`update_task` imports (B1 — Opus review 2026-06-13)**

The two-pass executors call bare `enqueue(...)` / `update_task(...)`. Today `mr_roboto/__init__.py` has **NO module-level binding** for either — `enqueue` is only reached via function-local `import general_beckman`, and `update_task` via a function-local `from general_beckman import update_task` inside the human-confirm gate (`__init__.py:769`). Bare calls would `NameError`, and the tests' `monkeypatch.setattr("mr_roboto.enqueue", ...)` would have nothing to patch. Add a module-level import near the top of `packages/mr_roboto/src/mr_roboto/__init__.py` (verified circular-import-safe — `general_beckman` never imports `mr_roboto` at module load):

```python
from general_beckman import enqueue, update_task  # SP6: module-level so two-pass executors + tests resolve them
```

This makes the bare calls resolve AND makes `mr_roboto.enqueue` / `mr_roboto.update_task` valid monkeypatch targets. (The human-confirm gate's existing function-local `from general_beckman import update_task` at ~769 can stay or be removed — harmless either way.)

- [ ] **Step 1: Write the failing test for the resume handlers**

Create `packages/mr_roboto/tests/test_critic_two_pass.py`:

```python
import json
import pytest
from unittest.mock import AsyncMock, patch
import mr_roboto.critic_continuations as cc


@pytest.mark.asyncio
async def test_verdict_done_stamps_and_repends():
    """verdict_done: persist log, stamp critic_verdict into the gated task
    context, flip it back to pending."""
    updates = []

    async def _fake_update(task_id, **kw):
        updates.append((task_id, kw))

    async def _fake_get_task(tid):
        return {"id": tid, "context": json.dumps({"action": "git_commit"})}

    result = {"result": {"content": '{"verdict": "pass", "reasons": []}'}}
    state = {"gated_task_id": 55, "action_name": "git_commit",
             "mission_id": 3, "payload_hash": "abc123"}
    with patch.object(cc, "update_task", _fake_update), \
         patch.object(cc, "get_task", _fake_get_task), \
         patch.object(cc, "_persist_critic_log", AsyncMock()):
        await cc._verdict_done(child_task_id=99, result=result, state=state)

    assert updates, "gated task must be updated"
    tid, kw = updates[0]
    assert tid == 55
    assert kw["status"] == "pending"
    ctx = json.loads(kw["context"])
    assert ctx["critic_verdict"]["verdict"] == "pass"
    assert ctx["critic_verdict"]["payload_hash"] == "abc123"


@pytest.mark.asyncio
async def test_verdict_err_fails_closed_gated_task():
    """verdict_err: terminal producer error → fail the gated task (blocked)."""
    updates = []

    async def _fake_update(task_id, **kw):
        updates.append((task_id, kw))

    state = {"gated_task_id": 55, "action_name": "git_commit"}
    with patch.object(cc, "update_task", _fake_update), \
         patch.object(cc, "_persist_critic_log", AsyncMock()):
        await cc._verdict_err(child_task_id=99,
                              result={"error": "no candidates"}, state=state)
    tid, kw = updates[0]
    assert tid == 55
    assert kw["status"] == "failed"
    assert "blocked" in kw["error"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_two_pass.py -v`
Expected: FAIL — module `mr_roboto.critic_continuations` does not exist.

- [ ] **Step 3: Create the resume-handler module**

Create `packages/mr_roboto/src/mr_roboto/critic_continuations.py`:

```python
"""CPS SP6 — surface-B critic gate resume handlers.

Two-pass self-park for inline mechanical gates (git_commit / notify_user):
pass-1 enqueues an admitted critic child + parks the gated task; this module's
handlers stamp the verdict back into the gated task and re-pend it (verdict_done)
or fail it closed (verdict_err). The gated executor's pass-2 then reads
context['critic_verdict'] and calls the LLM-free confirm_gate.
"""
from __future__ import annotations

import json

from general_beckman import update_task
from src.infra.db import get_task
from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.critic_continuations")


def _extract_content(result: dict) -> str:
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = inner.get("content", "")
    elif inner is not None:
        content = inner
    else:
        content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


def _parse_ctx(task: dict) -> dict:
    raw = (task or {}).get("context") or "{}"
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


async def _persist_critic_log(state: dict, verdict: str, reasons: list) -> None:
    from mr_roboto.critic_gate import _persist
    await _persist(
        state.get("mission_id"),
        str(state.get("action_name") or "unknown"),
        verdict,
        list(reasons or []),
        str(state.get("payload_hash") or ""),
    )


async def _verdict_done(child_task_id: int, result: dict, state: dict) -> None:
    """Critic child completed → stamp verdict into the gated task + re-pend."""
    from mr_roboto.critic_gate import _parse_verdict

    gated_id = state.get("gated_task_id")
    parsed = _parse_verdict(_extract_content(result))
    await _persist_critic_log(state, parsed["verdict"], parsed.get("reasons") or [])

    gated = await get_task(int(gated_id)) if gated_id is not None else None
    if gated is None:
        logger.warning("critic verdict_done: gated task missing", gated_id=gated_id)
        return
    ctx = _parse_ctx(gated)
    ctx["critic_verdict"] = {
        "verdict": parsed["verdict"],
        "reasons": parsed.get("reasons") or [],
        "payload_hash": str(state.get("payload_hash") or ""),
    }
    await update_task(int(gated_id), status="pending", context=json.dumps(ctx))


async def _verdict_err(child_task_id: int, result: dict, state: dict) -> None:
    """Critic child failed terminally → FAIL the gated task CLOSED (action blocked)."""
    gated_id = state.get("gated_task_id")
    action = str(state.get("action_name") or "action")
    err = (result or {}).get("error", "unknown")
    await _persist_critic_log(
        state, "veto", [f"producer error: {str(err)[:120]}"]
    )
    if gated_id is None:
        return
    await update_task(
        int(gated_id),
        status="failed",
        error=f"critic verdict unavailable ({str(err)[:80]}) — {action} blocked (fail-closed)",
    )


def register_continuations() -> None:
    """Register SP6 surface-B critic resume handlers. Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("mr_roboto.critic.verdict_done", _verdict_done)
        register_resume("mr_roboto.critic.verdict_err", _verdict_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("critic continuation registration deferred", error=str(exc))


register_continuations()
```

- [ ] **Step 4: Run resume tests — verify they pass**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_two_pass.py -v`
Expected: PASS (both).

- [ ] **Step 5: Register the module in `_HANDLER_MODULES`**

In `packages/general_beckman/src/general_beckman/continuations.py`, append to the `_HANDLER_MODULES` list (after `"mr_roboto.swap_placeholder_images"`):

```python
    "mr_roboto.critic_continuations",  # SP6 — surface-B critic two-pass resumes
```

- [ ] **Step 6: Write the failing executor test (two-pass git_commit)**

Add to `packages/mr_roboto/tests/test_critic_two_pass.py`:

```python
import mr_roboto


@pytest.mark.asyncio
async def test_git_commit_pass1_parks_and_enqueues_critic(monkeypatch, tmp_path):
    """Pass 1 (no verdict in ctx): enqueue an admitted critic child + park
    the gated task (needs_clarification + waiting_human). NO commit yet."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)  # gate on
    enq = AsyncMock(return_value=1234)
    upd = AsyncMock()
    monkeypatch.setattr("mr_roboto.enqueue", enq, raising=False)
    monkeypatch.setattr("mr_roboto.update_task", upd, raising=False)
    # Stub git capture so no real repo is needed.
    monkeypatch.setattr("src.tools.git_ops._resolve_repo", lambda p: str(tmp_path))
    monkeypatch.setattr("src.tools.git_ops._run_git",
                        AsyncMock(return_value=(0, "stat", "")))
    monkeypatch.setattr("src.tools.git_ops.ensure_git_repo", AsyncMock())
    auto = AsyncMock()
    monkeypatch.setattr("mr_roboto.auto_commit", auto, raising=False)

    task = {"id": 55, "title": "t", "mission_id": 3,
            "context": json.dumps({}),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)

    assert action.status == "needs_clarification"
    assert enq.await_count == 1               # critic child enqueued
    # parked as waiting_human
    assert any(kw.get("status") == "waiting_human"
               for _, kw in [c.args and (c.args[0], c.kwargs) for c in upd.await_args_list]) \
        or any(c.kwargs.get("status") == "waiting_human" for c in upd.await_args_list)
    auto.assert_not_called()                  # NO commit in pass 1


@pytest.mark.asyncio
async def test_git_commit_pass2_veto_blocks_commit(monkeypatch, tmp_path):
    """Pass 2 (verdict=veto in ctx): NO commit; rollback + failed."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    monkeypatch.setattr("src.tools.git_ops._resolve_repo", lambda p: str(tmp_path))
    run_git = AsyncMock(return_value=(0, "stat", ""))
    monkeypatch.setattr("src.tools.git_ops._run_git", run_git)
    monkeypatch.setattr("src.tools.git_ops.ensure_git_repo", AsyncMock())
    auto = AsyncMock()
    monkeypatch.setattr("mr_roboto.auto_commit", auto, raising=False)

    ctx = {"critic_verdict": {"verdict": "veto", "reasons": ["leaks token"],
                              "payload_hash": ""}}
    task = {"id": 55, "title": "t", "mission_id": 3,
            "context": json.dumps(ctx),
            "payload": {"action": "git_commit", "workspace_path": str(tmp_path)}}
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    auto.assert_not_called()
```

(Note: the exact monkeypatch targets for `enqueue`/`update_task` depend on how `mr_roboto/__init__.py` imports them — adjust to match the import site you add in Step 7. Run the test, read the failure, fix the patch target.)

- [ ] **Step 7: Rewrite the `git_commit` executor as two-pass**

In `packages/mr_roboto/src/mr_roboto/__init__.py`, replace the gate portion of the `if action == "git_commit":` block (the `from mr_roboto.critic_gate import critic_gate ...` through the veto-return, lines ~815-875) with the two-pass structure. Keep the post-gate commit tail (auto_commit + provenance + mark_green + require_diff) UNCHANGED. New gate logic:

```python
    if action == "git_commit":
        from mr_roboto.critic_gate import (
            confirm_gate as _confirm_gate, _opt_out as _critic_opt_out,
            _build_critic_spec, _redact_payload, _hash_payload,
        )
        from src.tools.workspace import get_mission_workspace_relative
        from src.tools.git_ops import _run_git, ensure_git_repo, _resolve_repo

        ctx = task.get("context") or {}
        if isinstance(ctx, str):
            try:
                import json as _json
                ctx = _json.loads(ctx)
            except (ValueError, TypeError):
                ctx = {}
        ctx_verdict = ctx.get("critic_verdict") if isinstance(ctx, dict) else None

        gate_enabled = (not _critic_opt_out()) and bool(payload.get("critic_gate", True))

        # Capture the staged diff (needed for both the producer payload in pass 1
        # and the drift-recheck in pass 2).
        async def _capture():
            mid = task.get("mission_id")
            repo_path = get_mission_workspace_relative(mid) if mid else ""
            await ensure_git_repo(repo_path)
            target = _resolve_repo(repo_path) or ""
            diff_stat = diff_full = ""
            if target:
                await _run_git(["add", "-A"], cwd=target)
                _, diff_stat, _ = await _run_git(["diff", "--cached", "--stat"], cwd=target)
                _, diff_full, _ = await _run_git(["diff", "--cached"], cwd=target)
            planned = f"Task #{task.get('id')}: {(task.get('title') or 'untitled')[:60]}"
            cargo = {"commit_message": planned,
                     "diff_stat": (diff_stat or "")[:2000],
                     "diff_excerpt": (diff_full or "")[:4000]}
            return target, cargo

        gate_result = None
        if gate_enabled and ctx_verdict is None:
            # ── PASS 1: capture, enqueue admitted critic child, park. ──
            try:
                _target, cargo = await _capture()
                redacted = _redact_payload(cargo)
                spec = _build_critic_spec("git_commit", redacted)
                mid = task.get("mission_id")
                if mid is not None:
                    spec["mission_id"] = mid
                await enqueue(
                    spec, parent_id=task.get("id"),
                    on_complete="mr_roboto.critic.verdict_done",
                    on_error="mr_roboto.critic.verdict_err",
                    cont_state={"gated_task_id": task.get("id"),
                                "action_name": "git_commit",
                                "mission_id": mid,
                                "payload_hash": _hash_payload(redacted)},
                )
                await update_task(int(task["id"]), status="waiting_human")
                return Action(status="needs_clarification",
                              result={"awaiting_critic": True, "action": "git_commit"})
            except Exception as e:
                # Park/enqueue failed → fail-closed (do NOT silently commit).
                from src.infra.logging_config import get_logger as _gl
                _gl("mr_roboto.critic_gate").warning(f"git_commit gate park failed: {e}")
                return Action(status="failed",
                              error=f"critic gate could not be scheduled: {e}")

        if gate_enabled and ctx_verdict is not None:
            # ── PASS 2: confirm against the persisted verdict (LLM-free). ──
            # Drift guard: re-capture and compare hash; bounded re-gate.
            _target, cargo2 = await _capture()
            new_hash = _hash_payload(_redact_payload(cargo2))
            judged_hash = str(ctx_verdict.get("payload_hash") or "")
            if judged_hash and new_hash != judged_hash:
                regate_n = int(ctx.get("critic_regate_n") or 0)
                if regate_n >= 2:
                    if _target:
                        await _run_git(["reset"], cwd=_target)
                    return Action(status="failed",
                                  error="critic re-gate exhausted (tree kept changing)")
                # Drop the stale verdict, bump counter, re-park for a fresh gate.
                import json as _json
                ctx.pop("critic_verdict", None)
                ctx["critic_regate_n"] = regate_n + 1
                await update_task(int(task["id"]), context=_json.dumps(ctx),
                                  status="pending")
                return Action(status="needs_clarification",
                              result={"regate": ctx["critic_regate_n"]})
            gate_result = await _confirm_gate("git_commit", cargo2,
                                              mission_id=task.get("mission_id"),
                                              persisted_verdict=ctx_verdict)
            if gate_result.get("verdict") == "veto":
                if _target:
                    await _run_git(["reset"], cwd=_target)
                return Action(status="failed",
                              error=f"critic_gate vetoed git_commit: {gate_result.get('reasons')}",
                              result={"critic": gate_result})

        # ── pass-through (gate disabled OR pass-2 approved): real commit. ──
        commit_info = await auto_commit(task, payload.get("result") or {})
        if gate_result is not None:
            (commit_info or {}).setdefault("critic", gate_result)
        # … (UNCHANGED provenance + mark_green + require_diff tail follows) …
```

Ensure `enqueue` and `update_task` are imported at the top of `mr_roboto/__init__.py` (or import locally inside the block — match the test's monkeypatch target from Step 6). `update_task` is already used by the human-confirm gate path, so the import exists; add `from general_beckman import enqueue` near it.

- [ ] **Step 8: Run the executor tests; fix patch targets as needed**

Run: `timeout 90 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_two_pass.py -v`
Expected: PASS. If a monkeypatch misses, read the error, align the patch target to the real import site, re-run.

- [ ] **Step 9: Run the mr_roboto regression suite**

Run: `timeout 120 .venv/Scripts/python -m pytest packages/mr_roboto/tests/ -x -q`
Expected: no NEW failures. Known pre-existing red: `test_reversibility_registry` (per memory `project_cps_migration_20260527` SP3b note) — confirm red before your change via `git stash`.

- [ ] **Step 10: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/critic_continuations.py \
        packages/mr_roboto/src/mr_roboto/__init__.py \
        packages/general_beckman/src/general_beckman/continuations.py \
        packages/mr_roboto/tests/test_critic_two_pass.py
git commit -m "feat(critic_gate): git_commit two-pass self-park, admitted critic child (SP6 T3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Surface B — `notify_user` two-pass self-park (MISSION-SCOPED ONLY)

Symmetric to git_commit but simpler: no staging, no rollback (nothing is sent until pass 2), so no drift-recheck is needed.

> **⚠️ B2 CARVE-OUT (Opus review 2026-06-13) — gate `notify_user` ONLY when `mission_id` is present.**
> `notify_user` is the single outbound boundary for ROUTINE non-mission alerts (VRAM/health pings, cron/uptime, follow-up reminders, mention digests — many with NO `mission_id`, firing during normal operation; see `general_beckman/sweep.py`, `cron.py`, `nerd_herd/health_summary.py`, `src/app/jobs/follow_up_reminder.py`). A uniform fail-closed two-pass gate would (a) add a critic LLM round-trip to every alert and (b) **DROP** the alert if the critic is unavailable — the pathological case being a "VRAM 95%" alert vetoed *because* the critic can't run under that very pressure (the #261969 history is exactly why notify_user fail-OPEN was deliberate). The critic gate's purpose is guarding **autonomous agent-produced** content (mission comms), NOT internal system health pings.
> **Rule:** `notify_user` enters the two-pass gate ONLY when `task.get("mission_id") is not None`. Routine alerts (no mission_id) **bypass the gate entirely** — send directly, never parked, never dropped (preserves today's fire-and-forget). `git_commit` stays unconditionally gated (commits are always autonomous-agent output, and are reversible via `git reset`/retry so fail-closed is safe there).

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (notify_user executor, ~2166-2213)
- Test: `packages/mr_roboto/tests/test_critic_two_pass.py` (extend)

- [ ] **Step 1: Write the failing tests**

Add to `packages/mr_roboto/tests/test_critic_two_pass.py`:

```python
@pytest.mark.asyncio
async def test_notify_user_pass1_parks(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    enq = AsyncMock(return_value=1234)
    upd = AsyncMock()
    send = AsyncMock()
    monkeypatch.setattr("mr_roboto.enqueue", enq, raising=False)
    monkeypatch.setattr("mr_roboto.update_task", upd, raising=False)
    monkeypatch.setattr("mr_roboto.notify_user.notify_user", send, raising=False)
    task = {"id": 60, "mission_id": 2, "context": json.dumps({}),
            "payload": {"action": "notify_user", "message": "hi founder"}}
    action = await mr_roboto.run(task)
    assert action.status == "needs_clarification"
    assert enq.await_count == 1
    send.assert_not_called()                 # nothing sent in pass 1


@pytest.mark.asyncio
async def test_notify_user_pass2_pass_sends(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    send = AsyncMock(return_value={"sent": True})
    monkeypatch.setattr("mr_roboto.notify_user.notify_user", send, raising=False)
    ctx = {"critic_verdict": {"verdict": "pass", "reasons": [], "payload_hash": ""}}
    task = {"id": 60, "mission_id": 2, "context": json.dumps(ctx),
            "payload": {"action": "notify_user", "message": "hi founder"}}
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_notify_user_pass2_veto_drops(monkeypatch):
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)
    send = AsyncMock()
    monkeypatch.setattr("mr_roboto.notify_user.notify_user", send, raising=False)
    ctx = {"critic_verdict": {"verdict": "veto", "reasons": ["leaks PII"],
                              "payload_hash": ""}}
    task = {"id": 60, "mission_id": 2, "context": json.dumps(ctx),
            "payload": {"action": "notify_user", "message": "ssn 123"}}
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    send.assert_not_called()


@pytest.mark.asyncio
async def test_notify_user_routine_alert_bypasses_gate(monkeypatch):
    """B2 carve-out: a notify_user with NO mission_id is a routine alert —
    it must SEND DIRECTLY, never enqueue a critic child, never park. A flaky
    critic must never silence a health/VRAM/cron alert."""
    monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)  # gate on globally
    enq = AsyncMock()
    send = AsyncMock(return_value={"sent": True})
    monkeypatch.setattr("mr_roboto.enqueue", enq, raising=False)
    monkeypatch.setattr("mr_roboto.notify_user.notify_user", send, raising=False)
    task = {"id": 61, "context": json.dumps({}),   # NO mission_id
            "payload": {"action": "notify_user", "message": "VRAM at 95%"}}
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    send.assert_awaited_once()       # sent directly
    enq.assert_not_called()          # no critic child, no park
```

- [ ] **Step 2: Run to verify failure**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_two_pass.py -v -k notify_user`
Expected: FAIL (current notify_user still inline-gates).

- [ ] **Step 3: Rewrite the notify_user gate as two-pass**

In `packages/mr_roboto/src/mr_roboto/__init__.py`, replace the gate portion of `if action == "notify_user":` (lines ~2170-2208, from the `from mr_roboto.critic_gate import ...` through the veto-return) with:

```python
    if action == "notify_user":
        from mr_roboto.notify_user import notify_user
        from mr_roboto.critic_gate import (
            confirm_gate as _confirm_gate, _opt_out as _critic_opt_out,
            _build_critic_spec, _redact_payload, _hash_payload,
        )
        import json as _json

        ctx = task.get("context") or {}
        if isinstance(ctx, str):
            try:
                ctx = _json.loads(ctx)
            except (ValueError, TypeError):
                ctx = {}
        ctx_verdict = ctx.get("critic_verdict") if isinstance(ctx, dict) else None
        # B2 carve-out: gate ONLY mission-scoped notify_user (autonomous agent
        # comms). Routine alerts (no mission_id) bypass entirely — never parked,
        # never dropped, no critic latency.
        gate_enabled = (
            (not _critic_opt_out())
            and bool(payload.get("critic_gate", True))
            and task.get("mission_id") is not None
        )
        text = payload.get("message") or payload.get("text") or ""

        if gate_enabled and ctx_verdict is None:
            # PASS 1 — enqueue admitted critic child, park. Nothing sent yet.
            try:
                redacted = _redact_payload({"message": text})
                spec = _build_critic_spec("notify_user", redacted)
                mid = task.get("mission_id")
                if mid is not None:
                    spec["mission_id"] = mid
                await enqueue(
                    spec, parent_id=task.get("id"),
                    on_complete="mr_roboto.critic.verdict_done",
                    on_error="mr_roboto.critic.verdict_err",
                    cont_state={"gated_task_id": task.get("id"),
                                "action_name": "notify_user",
                                "mission_id": mid,
                                "payload_hash": _hash_payload(redacted)},
                )
                await update_task(int(task["id"]), status="waiting_human")
                return Action(status="needs_clarification",
                              result={"awaiting_critic": True, "action": "notify_user"})
            except Exception as e:
                from src.infra.logging_config import get_logger as _gl
                _gl("mr_roboto.critic_gate").warning(f"notify_user gate park failed: {e}")
                return Action(status="failed",
                              error=f"critic gate could not be scheduled: {e}")

        if gate_enabled and ctx_verdict is not None:
            # PASS 2 — confirm (LLM-free). No drift guard: nothing staged.
            gate_result = await _confirm_gate("notify_user", {"message": text},
                                              mission_id=task.get("mission_id"),
                                              persisted_verdict=ctx_verdict)
            if gate_result.get("verdict") == "veto":
                return Action(status="failed",
                              error=f"critic_gate vetoed notify_user: {gate_result.get('reasons')}",
                              result={"critic": gate_result, "sent": False})

        try:
            res = await notify_user(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))
```

- [ ] **Step 4: Run notify_user tests — verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_two_pass.py -v -k notify_user`
Expected: PASS (all four — park, pass-sends, veto-drops, routine-bypass).

- [ ] **Step 5: Run mr_roboto regression**

Run: `timeout 120 .venv/Scripts/python -m pytest packages/mr_roboto/tests/ -x -q`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/__init__.py packages/mr_roboto/tests/test_critic_two_pass.py
git commit -m "feat(critic_gate): notify_user two-pass self-park (SP6 T4)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Delete dead inline funcs + surface C + test cleanup

After A+B migrate, the inline `produce_verdict` + `critic_gate` orchestrator + the standalone `action=="critic_gate"` executor have no live callers. Re-grep, delete, fix tests.

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/critic_gate.py` (delete `produce_verdict` + `critic_gate`)
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (delete standalone `critic_gate` action, ~983-1002)
- Modify/Delete: `packages/mr_roboto/tests/test_critic_gate.py`, `test_critic_gate_split.py`

- [ ] **Step 1: Re-grep all callers (must be zero in prod)**

Run: `rg -n "produce_verdict|critic_gate\(" packages src --glob '!*/tests/*'`
Expected: ZERO hits for `produce_verdict(` and the `critic_gate(` orchestrator (the surface-A/B code uses `_build_critic_spec` + `confirm_gate`, not these). If any prod hit remains, STOP — a surface was missed; do not delete.

Run: `rg -n '"critic_gate"|action == "critic_gate"' packages/mr_roboto/src` — confirm the only `action=="critic_gate"` executor is the standalone block (~983); confirm apply.py no longer routes a mechanical critic_gate child.

- [ ] **Step 2: Update tests that exercise the deleted funcs (make them RED→handled)**

Identify the tests that call `produce_verdict` / `critic_gate` orchestrator / the standalone action:
- `packages/mr_roboto/tests/test_critic_gate.py` — `test_router_standalone_critic_gate_pass/veto` (~339,360), plus `produce_verdict`/`critic_gate` orchestrator tests (~106-164).
- `packages/mr_roboto/tests/test_critic_gate_split.py` — `produce_verdict`/`critic_gate` tests (~106-164).

Delete the tests that validate the deleted symbols; KEEP the `confirm_gate` tests (Task 1) and any `_parse_verdict`/`_redact`/`_build_critic_spec` unit tests (those funcs survive). Run the suite first to enumerate exactly which tests reference the deleted names:

Run: `rg -ln "produce_verdict|critic_gate\(|action_name=" packages/mr_roboto/tests/`

- [ ] **Step 3: Delete the standalone `critic_gate` executor action**

In `packages/mr_roboto/src/mr_roboto/__init__.py`, delete the `if action == "critic_gate":` block (~983-1002). Leave a one-line breadcrumb comment: `# critic_gate is now an admitted posthook LLM child (SP6) — no standalone mechanical action.`

- [ ] **Step 4: Delete the inline `produce_verdict` + `critic_gate` orchestrator funcs**

In `packages/mr_roboto/src/mr_roboto/critic_gate.py`, delete `async def produce_verdict(...)` (~222-278) and `async def critic_gate(...)` (~343-381). KEEP: `_redact*`, `_hash_payload`, `_opt_out`, `_PROMPT_TEMPLATE`, `_parse_verdict`, `_persist`, `_build_critic_spec`, `confirm_gate`. Update the module docstring to describe the surviving API (admitted producer via `_build_critic_spec`; `confirm_gate` is the only public gate).

- [ ] **Step 5: Run the critic_gate test files + import smoke**

Run: `timeout 90 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_gate.py packages/mr_roboto/tests/test_critic_gate_split.py packages/mr_roboto/tests/test_critic_two_pass.py -v`
Then: `timeout 30 .venv/Scripts/python -c "import mr_roboto; from mr_roboto import critic_gate, critic_continuations; print('ok')"`
Expected: PASS + `ok`.

- [ ] **Step 6: Full structural check — no inline husam in gated executors**

Add a structural test `packages/mr_roboto/tests/test_critic_no_inline_husam.py`:

```python
import pathlib


def test_gated_executors_have_no_inline_husam_or_produce_verdict():
    src = pathlib.Path("packages/mr_roboto/src/mr_roboto/__init__.py").read_text(encoding="utf-8")
    # The two-pass executors must use confirm_gate, never produce_verdict/husam inline.
    assert "produce_verdict" not in src
    assert "husam.run" not in src  # gated executors never call husam directly


def test_critic_gate_module_has_no_orchestrator():
    src = pathlib.Path("packages/mr_roboto/src/mr_roboto/critic_gate.py").read_text(encoding="utf-8")
    assert "async def produce_verdict" not in src
    assert "async def critic_gate(" not in src
```

Run: `timeout 30 .venv/Scripts/python -m pytest packages/mr_roboto/tests/test_critic_no_inline_husam.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/critic_gate.py \
        packages/mr_roboto/src/mr_roboto/__init__.py \
        packages/mr_roboto/tests/
git commit -m "refactor(critic_gate): delete inline produce_verdict/orchestrator + standalone action (SP6 T5)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Sweep companions

Light cleanup folded into the sprint. Each is independent; commit separately if preferred.

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (legacy straggler shim, ~1309-1329)
- Modify: `src/tools/vision.py` (comment)
- Modify: `CLAUDE.md` (one-line note)
- Delete: `.claude/worktrees/sp5-deletion-sweep` (dir)
- Modify/Delete: stale tests

- [ ] **Step 1: Locate + remove the SP5 legacy straggler shim**

Run: `rg -n "removable post-SP5|Legacy straggler" packages/general_beckman/src`
Read the matched block in `general_beckman/__init__.py` (`on_task_finished`'s legacy-context fallback, ~1309-1329). Confirm via grep that the durable `continuations` table is the sole live path (no code writes legacy continuation context anymore). If confirmed dead, delete the shim block. If ANY caller still depends on it, leave it and note why in the commit — do NOT force the deletion.

Run: `timeout 120 .venv/Scripts/python -m pytest packages/general_beckman/tests/ -x -q`
Expected: green (the shim is dead; removing it changes nothing).

- [ ] **Step 2: Reconcile the two stale tests named in the SP5 handoff**

Per `docs/handoff/2026-06-10-sp5-await-inline-remaining-handoff.md` Residual section:
- `tests/test_mission_workflow_integration.py::TestLLMClassification` (3 tests) — assert a return value from `_classify_user_message`, which SP2 changed to return `None`. Delete these 3 tests (the CPS classifier contract is covered elsewhere) OR rewrite against the `on_complete="telegram.message_route_resume"` path. Prefer delete unless a quick rewrite is obvious.
- `tests/integration/test_agent_basic.py` (ReAct-iteration test ~line 300) — patches `src.agents.base.execute_tool`, removed by Runtime Phase A. Delete or rewrite against `coulson.execute`. Prefer delete.

Run each targeted file with `timeout 60 .venv/Scripts/python -m pytest <file> -q` to confirm no collection errors after edits.

- [ ] **Step 3: vision.py comment + CLAUDE.md note**

In `src/tools/vision.py`, near the `husam.run` call (~83), add:

```python
        # NOTE (SP6): this direct husam.run is the sanctioned shape-(a) exception
        # — vision is a mid-ReAct DYNAMIC tool call that cannot be pre-scheduled
        # as an admitted task (unlike the critic gate, which is now admitted +
        # fail-closed, SP6). Migrating this needs CPS-for-tools (future). Ruling-#1.
```

In `CLAUDE.md`, under "### LLM Dispatch & Model Routing" or the critic-gate-relevant section, add one line:

```markdown
- **Critic gate is admitted + fail-closed (SP6 2026-06-13)**: the second-LLM veto on irreversible actions (git_commit/notify_user/posthook `critic_gate`) runs as an ADMITTED task before the action (not an inline husam call); a missing/failed verdict BLOCKS the action (`confirm_gate` fail-closed). Opt-out: `KUTAI_CRITIC_GATE=off`. `src/tools/vision.py` keeps an inline `husam.run` as the sanctioned mid-ReAct exception.
```

- [ ] **Step 4: Delete the orphaned worktree dir**

Run: `rm -rf .claude/worktrees/sp5-deletion-sweep` (already git-pruned per SP5 handoff; safe). Verify: `git worktree list` does not show it.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py \
        src/tools/vision.py CLAUDE.md tests/
git commit -m "chore(sp6): sweep — drop SP5 legacy shim, stale tests, vision/critic-gate docs (SP6 T6)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final verification (after all tasks)

- [ ] **Re-grep guards:**
  - `rg -n "await_inline\s*=\s*True" src packages --glob '!*/tests/*'` → ZERO.
  - `rg -n "produce_verdict\(|husam.run" packages/mr_roboto/src` → ZERO (vision.py is in src/tools, outside mr_roboto).
- [ ] **Targeted suites green:** `timeout 180 .venv/Scripts/python -m pytest packages/general_beckman/tests/ packages/mr_roboto/tests/ -q` (ONE invocation — never two concurrent). Known pre-existing reds (`test_admission_cache`×2, `test_reversibility_registry`) confirmed red on the base commit, NOT introduced here.
- [ ] **Import smoke:** `timeout 30 .venv/Scripts/python -c "import mr_roboto, general_beckman; from mr_roboto import critic_gate, critic_continuations; from general_beckman import posthook_continuations; print('ok')"`.
- [ ] **DEPLOY GATE (founder, NOT done by the worker):** all of SP6 is restart-gated. Live KutAI runs old code until a founder `/restart` via Telegram. After restart, run ONE graded mission that hits a `git_commit` step to confirm: critic child dispatches on `oneshot`, gated task parks then resumes, commit lands on pass, and a forced-veto blocks the commit. Do NOT push without founder sign-off (project convention: restart-gated, founder pushes).

---

## Self-review notes (author)

- **Spec coverage:** Task 1 = confirm_gate fail-closed (spec §"confirm_gate → fail-CLOSED"). Task 2 = surface A (spec §"Surface A"). Tasks 3–4 = surface B (spec §"Surface B"). Task 5 = delete inline funcs + surface C (spec §"Surface A" deletion note + §"confirm_gate" dead-funcs para). Task 6 = sweep companions (spec §"SP6 sweep companions"). Vision deferral = spec §"Out of scope". All spec sections mapped.
- **Park status:** uses the verified `needs_clarification`+`waiting_human` mechanic (orchestrator.py:310-323 skips on_task_finished for mechanical tasks) — NOT the non-existent `needs_followup`.
- **Verdict rail:** surface A reuses the existing `_Z1_MECHANICAL_KINDS` single-shot rail (critic_gate already a member) — no new branch, confirmed against apply.py:3949,3958,5165.
- **Open follow-on (not blocking):** the drift-recheck bound (N=2) in Task 3 is conservative; if i2p serialization makes drift impossible in practice, it can be simplified later — left in per spec risk note.
