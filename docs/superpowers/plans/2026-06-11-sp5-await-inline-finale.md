# SP5 finale — delete `await_inline` (CPS-migrate the 2 carve-outs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `await_inline=True` blocking primitive from `general_beckman.enqueue` entirely by CPS-migrating its last two live callers, then deleting the parameter + inline-waiter machinery.

**Architecture:** `await_inline` blocks the calling coroutine on an `asyncio.Future` until a child task reaches terminal, resolved from the terminal hook. CPS (continuation-passing style) replaces this with: enqueue a child carrying a named `on_complete` continuation handler + persisted `cont_state`; the handler fires asynchronously when the child finishes. We migrate both remaining callers to CPS (preserving their LLM "intelligence" per founder ruling — not deleting), then delete the blocking path.

**Tech Stack:** Python 3.10 async, `general_beckman` continuation substrate (`continuations.py`: `register_resume`, `dispatch_on_complete`, `_HANDLER_MODULES`), SQLite `continuations` table, pytest.

---

## Context the executor needs (read before starting)

**The handoff premise was WRONG.** `docs/handoff/2026-06-10-sp5-await-inline-remaining-handoff.md` scoped carve-out 1 (task_classifier) as a "task-admission redesign" because a code comment claims `add_task` consumes `classify_task`'s result synchronously. **It does not.** Verified 2026-06-11:
- `src/infra/db.py:4351 add_task()` is dedup+insert; it takes `agent_type` as a parameter and never calls `classify_task`.
- `classify_task` / `_classify_with_llm` / `_enqueue_inline_classifier` / `TaskClassification` have **zero live callers** in `src/` or `packages/` — only tests. The only live import from the module is `_extract_json` (used by `src/app/telegram_bot.py:437`).
- Live message classification is `telegram_bot._classify_user_message` (already CPS, `on_complete="telegram.message_route_resume"`, no `await_inline`).

So carve-out 1 is a **dead-but-kept** chain. Founder decision (2026-06-11): **CPS-migrate both carve-outs** (keep the intelligence in a CPS-callable shape; do not delete). Founder explicitly accepted that the migrated classifier is "a CPS shell nobody invokes" — its value is being wireable later.

**The two carve-outs (verified the ONLY live `await_inline=True` callers):**
1. `src/core/task_classifier.py:284` — `_enqueue_inline_classifier`.
2. `src/app/jobs/investor_bullets.py:211` — `_call_llm_anomaly_hypothesis` (dormant; `missions.product_id` is NULL in prod so `collect_metrics` returns `{}` and the anomaly loop never fires).

**CPS substrate cheat-sheet (`packages/general_beckman/src/general_beckman/continuations.py`):**
- Handler signature: `async def handler(child_task_id: int, result: dict, state: dict) -> None`.
- Register at import time via a module-level `def register_continuations()` that calls `register_resume("name", handler)`, plus a bare `register_continuations()` call at module bottom.
- **Every new handler module MUST be added to `_HANDLER_MODULES`** (line ~175) or it is absent after restart and rows stay pending. `src.app.jobs.investor_bullets` is already noted there as deferred (line ~183) — un-defer it.
- `result` passed to a resume handler is the top-level-JSON-decoded `tasks.result`. For `raw_dispatch` LLM children the LLM text is at `result["content"]` (may be a list of `{type,text}` parts — coalesce).
- `cont_state` (dict) is persisted with the continuation row and handed back as `state`.

**Discipline (carried from prior SP5 sessions):**
- Work in a git worktree (`superpowers:using-git-worktrees`). Parallel `main` sessions have crossed past deletion work.
- **Editable-package trap:** pytest in a worktree imports the `pip install -e` package from the **main** checkout, not the worktree. Run package tests with sources forced ahead:
  `PYTHONPATH=".;$(printf '%s;' packages/*/src)" .venv/Scripts/python -m pytest <path> -p no:cacheprovider`
- **conftest collision:** never put `tests/` and `packages/*/tests/` in ONE pytest invocation. Separate runs.
- `timeout` on every pytest run; never run two pytest processes concurrently (SQLite WAL lock crash-loops live KutAI).
- Re-grep every symbol before deleting (`feedback_audit_call_sites`).

---

## File Structure

- `src/core/task_classifier.py` — **modify.** Add sync `parse_classification(result: dict) -> TaskClassification` (extracted intelligence) + `_classify_resume` continuation handler + `register_continuations()`; convert `classify_task` to a CPS kickoff `classify_task(title, desc, *, on_complete="task_classifier.classify.resume", cont_state=None) -> int | None`; delete `_enqueue_inline_classifier`. Keep `_extract_json`, `TaskClassification`, keyword fallback.
- `src/app/jobs/investor_bullets.py` — **modify.** Split `run_investor_bullets` into a kickoff that enqueues a sequential CPS chain over anomalies; add `_hypothesis_resume` handler that threads accumulated hypotheses via `cont_state` and on the last anomaly renders+emits; delete the `await_inline=True` path from `_enqueue_overhead` / `_call_llm_anomaly_hypothesis`; add `register_continuations()`.
- `packages/general_beckman/src/general_beckman/continuations.py` — **modify.** Add `src.core.task_classifier` to `_HANDLER_MODULES`; un-defer `src.app.jobs.investor_bullets`.
- `packages/general_beckman/src/general_beckman/__init__.py` — **modify.** Delete `await_inline` param + mutual-exclusion check + inline-wait path + `resolve_inline` + `_inline_waiters` + `INLINE_TIMEOUT` + `TaskResult` (if unused after) + terminal-hook resolve block (lines ~1132-1140) + `__all__` entries.
- `packages/general_beckman/README.md` — **modify.** Remove `await_inline` documentation.
- `packages/general_beckman/tests/test_no_inline_deadlock.py` — **modify.** Strengthen to "param is gone / no caller passes it".
- Test files — **modify/rewrite** the `classify_task` integration tests + add unit tests for `parse_classification` and the resume handlers.

---

## Task 1: Worktree + baseline green

**Files:** none (setup)

- [ ] **Step 1: Create the worktree**

Use `superpowers:using-git-worktrees` to create an isolated worktree off `main` named `sp5-await-inline`. All subsequent work happens there.

- [ ] **Step 2: Baseline — confirm the two carve-outs are the only live callers**

Run: `rg -n "await_inline\s*=\s*True" src packages | rg -v "/tests/|\.md:"`
Expected: exactly two hits — `src/core/task_classifier.py` and `src/app/jobs/investor_bullets.py`.

- [ ] **Step 3: Baseline test runs (record green/red before touching anything)**

Run (separately):
`timeout 180 .venv/Scripts/python -m pytest packages/general_beckman/tests/test_no_inline_deadlock.py -q -p no:cacheprovider`
`timeout 120 .venv/Scripts/python -m pytest tests/core/test_task_classifier_picks.py -q -p no:cacheprovider` (LLM-marked; expect skip/deselect without a model — that is fine, record it).
Expected: deadlock-guard green; note any pre-existing reds so they are not blamed on this work.

---

## Task 2: Carve-out 1 — extract `parse_classification` (sync intelligence, no LLM)

**Files:**
- Modify: `src/core/task_classifier.py`
- Test: `tests/core/test_parse_classification.py` (create)

- [ ] **Step 1: Write the failing unit test**

```python
# tests/core/test_parse_classification.py
"""parse_classification: pure (no LLM, no Beckman) mapping of an LLM result
dict → TaskClassification. This is the 'intelligence' extracted from the old
inline _classify_with_llm post-await block so it is testable synchronously."""
from src.core.task_classifier import parse_classification, TaskClassification


def test_parse_basic_fields():
    cls = parse_classification(
        {"content": '{"agent_type": "coder", "difficulty": 7, "needs_tools": true}'},
        title="build a parser", description="write a JSON parser module",
    )
    assert isinstance(cls, TaskClassification)
    assert cls.agent_type == "coder"
    assert cls.difficulty == 7
    assert cls.needs_tools is True
    assert cls.method == "llm"


def test_parse_clamps_difficulty():
    cls = parse_classification(
        {"content": '{"agent_type": "executor", "difficulty": 99}'},
        title="x", description="y",
    )
    assert cls.difficulty == 10


def test_parse_vision_guarded_to_visual_reviewer():
    # needs_vision only honored when agent_type == visual_reviewer
    cls = parse_classification(
        {"content": '{"agent_type": "coder", "needs_vision": true}'},
        title="ui work", description="design the layout",
    )
    assert cls.needs_vision is False


def test_parse_shopping_sub_intent_attached():
    cls = parse_classification(
        {"content": '{"agent_type": "shopping_advisor"}'},
        title="coffee machine", description="en ucuz kahve makinesi",
    )
    assert cls.agent_type == "shopping_advisor"
    assert cls.shopping_sub_intent is not None


def test_parse_falls_back_to_keywords_on_bad_json():
    cls = parse_classification(
        {"content": "not json at all"},
        title="fix the bug", description="error in auth",
    )
    # bad JSON → keyword fallback, still a valid classification
    assert cls.method == "keyword"
    assert cls.agent_type == "fixer"
```

- [ ] **Step 2: Run it — verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_parse_classification.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'parse_classification'`.

- [ ] **Step 3: Implement `parse_classification`**

In `src/core/task_classifier.py`, add this function (it lifts the body of the existing `_classify_with_llm` from `raw = response.get("content"...)` through the `return cls`, plus the keyword-fallback-on-parse-error). Place it AFTER `classify_task`'s helpers, BEFORE the keyword section:

```python
def parse_classification(result: dict, *, title: str, description: str) -> "TaskClassification":
    """Map a raw classifier-LLM result dict → TaskClassification.

    Pure + synchronous (no LLM, no Beckman). On any parse failure, degrades to
    the keyword classifier. This is the intelligence formerly inlined in
    _classify_with_llm's post-await block; extracted so it is unit-testable and
    reusable by the CPS resume handler.
    """
    content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    raw = str(content or "").strip()
    try:
        parsed = _extract_json(raw)
    except Exception:
        cls = _classify_by_keywords(title, description)
        if cls.agent_type == "shopping_advisor":
            cls.shopping_sub_intent = _classify_shopping_sub_intent(f"{title} {description}")
        return cls

    search_depth = parsed.get("search_depth") or _classify_search_depth(title + " " + description)
    agent_type = parsed.get("agent_type", "executor")
    # Only visual_reviewer actually uses analyze_image; a false needs_vision
    # triggers a 60s mmproj reload — guard it.
    needs_vision = parsed.get("needs_vision", False) and agent_type == "visual_reviewer"

    cls = TaskClassification(
        agent_type=agent_type,
        difficulty=max(1, min(10, int(parsed.get("difficulty", 5)))),
        needs_tools=parsed.get("needs_tools", False),
        needs_vision=needs_vision,
        needs_thinking=parsed.get("needs_thinking", False),
        local_only=parsed.get("local_only", False),
        priority=PRIORITY_MAP.get(parsed.get("priority", "normal"), 5),
        confidence=0.85,
        method="llm",
        search_depth=search_depth,
    )
    if cls.agent_type == "shopping_advisor":
        cls.shopping_sub_intent = _classify_shopping_sub_intent(f"{title} {description}")
    return cls
```

Note: this references `PRIORITY_MAP`, `_classify_by_keywords`, `_classify_shopping_sub_intent`, `_classify_search_depth`, `_extract_json`, `TaskClassification` — all already defined in this module. Confirm `PRIORITY_MAP` exists (grep it) before running.

- [ ] **Step 4: Run it — verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_parse_classification.py -q -p no:cacheprovider`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/core/task_classifier.py tests/core/test_parse_classification.py
git commit -m "refactor(task_classifier): extract sync parse_classification from inline classifier"
```

---

## Task 3: Carve-out 1 — CPS kickoff + resume handler; delete `_enqueue_inline_classifier`

**Files:**
- Modify: `src/core/task_classifier.py`
- Test: `tests/core/test_classify_cps.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_classify_cps.py
"""classify_task is now a CPS kickoff (no await_inline). It enqueues the
classifier LLM child with an on_complete continuation and returns the child id.
The resume handler reconstructs the TaskClassification via parse_classification."""
import pytest
from src.core import task_classifier as tc


@pytest.mark.asyncio
async def test_classify_task_kicks_off_with_on_complete(monkeypatch):
    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return 4242  # child task id

    monkeypatch.setattr(tc, "_enqueue", fake_enqueue, raising=False)
    child_id = await tc.classify_task("build a parser", "write a JSON parser")
    assert child_id == 4242
    assert captured["kwargs"]["on_complete"] == "task_classifier.classify.resume"
    # the kickoff must NOT pass await_inline
    assert "await_inline" not in captured["kwargs"]
    # cont_state carries title/description for the resume to rebuild classification
    st = captured["kwargs"]["cont_state"]
    assert st["title"] == "build a parser"
    assert "parser" in st["description"]
    # the enqueued LLM spec is a raw_dispatch classifier call
    assert captured["spec"]["context"]["llm_call"]["raw_dispatch"] is True


@pytest.mark.asyncio
async def test_classify_resume_builds_classification(monkeypatch):
    seen = {}
    monkeypatch.setattr(tc, "_on_classified", lambda cls, state: seen.update(cls=cls, state=state))
    await tc._classify_resume(
        99,
        {"content": '{"agent_type": "fixer", "difficulty": 6}'},
        {"title": "fix bug", "description": "auth error"},
    )
    assert seen["cls"].agent_type == "fixer"
    assert seen["cls"].difficulty == 6
```

- [ ] **Step 2: Run it — verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_classify_cps.py -q -p no:cacheprovider`
Expected: FAIL — `classify_task` still returns a `TaskClassification`/uses inline path; `_classify_resume` / `_enqueue` / `_on_classified` undefined.

- [ ] **Step 3: Implement the CPS kickoff + resume; delete the inline path**

In `src/core/task_classifier.py`:

(a) Add a module-level monkeypatchable enqueue alias near the top (after imports):

```python
async def _enqueue(spec: dict, **kwargs):
    """Thin, monkeypatchable wrapper over general_beckman.enqueue."""
    import general_beckman
    return await general_beckman.enqueue(spec, **kwargs)
```

(b) Replace `_classify_with_llm` and `_enqueue_inline_classifier` and the old async `classify_task` with the CPS kickoff. Delete `_enqueue_inline_classifier` entirely. New `classify_task`:

```python
async def classify_task(
    title: str,
    description: str,
    *,
    on_complete: str = "task_classifier.classify.resume",
    cont_state: dict | None = None,
) -> int | None:
    """CPS kickoff: enqueue the classifier LLM child and return its task id.

    The classification result is delivered asynchronously to the named
    ``on_complete`` continuation handler (default: this module's
    ``_classify_resume``), which rebuilds the TaskClassification via
    ``parse_classification``. No synchronous return value, no await_inline.

    Keyword-only callers that want a synchronous classification with no LLM
    can call ``_classify_by_keywords`` directly.
    """
    messages = [{
        "role": "user",
        "content": CLASSIFIER_PROMPT.format(
            task_description=f"{title}: {description[:500]}"
        ),
    }]
    state = dict(cont_state or {})
    state.setdefault("title", title)
    state.setdefault("description", description)
    spec = {
        "title": "task-classifier",
        "description": f"Classify task: {title[:80]!r}",
        "agent_type": "classifier",
        "kind": "classifier",
        "context": {"llm_call": {
            "raw_dispatch": True,
            "task": "router",
            "agent_type": "classifier",
            "difficulty": 3,
            "messages": messages,
            "prefer_speed": True,
            "needs_json_mode": True,
            "priority": 3,
            "estimated_input_tokens": 500,
            "estimated_output_tokens": 200,
            "call_category": "overhead",
        }},
    }
    return await _enqueue(spec, on_complete=on_complete, cont_state=state)
```

(c) Add the resume handler + a default consumer hook + registration at module bottom:

```python
def _on_classified(cls: "TaskClassification", state: dict) -> None:
    """Default consumer for a completed classification. No live caller wires a
    real consumer yet (telegram + /task admit typed tasks directly), so this
    logs. Future wiring: pass your own on_complete to classify_task and consume
    the TaskClassification there."""
    logger.info(
        "task classified (cps)",
        agent_type=cls.agent_type, difficulty=cls.difficulty,
        method=cls.method, title=str(state.get("title", ""))[:60],
    )


async def _classify_resume(child_task_id: int, result: dict, state: dict) -> None:
    """Continuation: rebuild the classification from the classifier LLM result."""
    cls = parse_classification(
        result or {},
        title=str(state.get("title", "")),
        description=str(state.get("description", "")),
    )
    _on_classified(cls, state)


def register_continuations() -> None:
    """Register CPS handlers for the task classifier (called at import + by
    general_beckman.continuations.register_startup_handlers)."""
    from general_beckman.continuations import register_resume
    register_resume("task_classifier.classify.resume", _classify_resume)


register_continuations()
```

Note: keep `_extract_json`, `TaskClassification`, `parse_classification`, `_classify_by_keywords`, `_classify_shopping_sub_intent`, `_classify_search_depth`, `CLASSIFIER_PROMPT`, `PRIORITY_MAP`. Delete only `_classify_with_llm` (logic now lives in `parse_classification`) and `_enqueue_inline_classifier`.

- [ ] **Step 4: Run it — verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_classify_cps.py tests/core/test_parse_classification.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Re-grep — confirm carve-out 1 no longer uses await_inline**

Run: `rg -n "await_inline" src/core/task_classifier.py`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add src/core/task_classifier.py tests/core/test_classify_cps.py
git commit -m "refactor(task_classifier): CPS-migrate classify_task off await_inline (carve-out 1)"
```

---

## Task 4: Carve-out 1 — reconcile the old integration tests

**Files:**
- Modify: `tests/test_mission_workflow_integration.py`, `tests/integration/test_classification.py`, `tests/integration/test_e2e_llm_pipeline.py`, `tests/integration/test_shopping_flow.py`, `tests/core/test_task_classifier_picks.py`

These call `cls = await classify_task(...)` and assert on `cls.agent_type` etc. `classify_task` now returns an `int | None` (child id), not a `TaskClassification`.

- [ ] **Step 1: Inventory the call sites**

Run: `rg -n "await classify_task\(|classify_task," tests`
Expected: the files above.

- [ ] **Step 2: Migrate each call site to the CPS contract**

For each test that wants to assert a classification *outcome*, replace the synchronous-return assertion with the resume-handler path. Pattern:

```python
# OLD:
#   cls = await classify_task(title, desc)
#   assert cls.agent_type == "coder"
# NEW: capture the kickoff, then drive the resume with a fixed LLM result.
import src.core.task_classifier as tc

captured = {}
async def _fake_enqueue(spec, **kw):
    captured["spec"] = spec; captured["kw"] = kw
    return 1
monkeypatch.setattr(tc, "_enqueue", _fake_enqueue, raising=False)

result_holder = {}
monkeypatch.setattr(tc, "_on_classified", lambda c, s: result_holder.update(cls=c))

await tc.classify_task(title, desc)
await tc._classify_resume(1, {"content": '{"agent_type": "coder", "difficulty": 6}'}, captured["kw"]["cont_state"])
assert result_holder["cls"].agent_type == "coder"
```

For tests that purely exercised real-LLM classification quality (`tests/core/test_task_classifier_picks.py`, the `@pytest.mark.llm` ones in `test_e2e_llm_pipeline.py`): retarget them to call `parse_classification` against a captured LLM result, OR if they were end-to-end-by-design and now redundant, delete them — but only after confirming the behavior is covered by `test_parse_classification.py` / `test_classify_cps.py`. Document each deletion in the commit message.

- [ ] **Step 3: Run the migrated tests (each file separately, all non-LLM-marked)**

Run e.g.:
`timeout 120 .venv/Scripts/python -m pytest tests/test_mission_workflow_integration.py tests/integration/test_classification.py -q -p no:cacheprovider -m "not llm"`
Expected: PASS (or pre-recorded baseline reds unrelated to classify_task — verify each red predates this work).

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test(task_classifier): migrate classify_task tests to CPS contract"
```

---

## Task 5: Carve-out 2 — sequential-chain CPS for investor_bullets anomalies

**Files:**
- Modify: `src/app/jobs/investor_bullets.py`
- Test: `tests/app/jobs/test_investor_bullets_cps.py` (create)

**Design:** `run_investor_bullets` currently loops `for ... in anomaly_items[:5]: hyp = await _call_llm_anomaly_hypothesis(...)` (each an await_inline call), then renders. CPS version:
- Kickoff (`run_investor_bullets`): collect metrics, detect anomalies. If **no** anomalies → render+emit immediately (synchronous, no LLM) and return. If anomalies → enqueue the **first** hypothesis child with `on_complete="investor_bullets.hypothesis.resume"` and `cont_state` = `{product_id, mission_id, anomalies: [[name,current,history],...], idx: 0, hypotheses: {}}`. Return `{"ok": True, "pending": True}`.
- Resume (`_hypothesis_resume`): extract the hypothesis string from `result`, store it into `state["hypotheses"][name]`. Increment `idx`. If more anomalies remain → enqueue the next hypothesis child with the updated `cont_state`. If done → call the shared `_finalize_bullets(state)` which fetches gaps, renders, emits variants. State (≤5 small entries) easily fits a continuation row; **no new table**.

This keeps the LLM intelligence, removes `await_inline`, and uses only the existing substrate.

- [ ] **Step 1: Write the failing test**

```python
# tests/app/jobs/test_investor_bullets_cps.py
"""investor_bullets anomaly hypotheses are produced via a sequential CPS chain
(no await_inline). Each resume stores one hypothesis and either enqueues the
next anomaly's child or finalizes."""
import pytest
import src.app.jobs.investor_bullets as ib


@pytest.mark.asyncio
async def test_resume_enqueues_next_anomaly(monkeypatch):
    enq = {}
    async def fake_enqueue(spec, **kw):
        enq["spec"] = spec; enq["kw"] = kw
        return 7
    monkeypatch.setattr(ib, "_enqueue_hypothesis_child", fake_enqueue, raising=False)
    finalized = {}
    async def fake_final(state):
        finalized["state"] = state
    monkeypatch.setattr(ib, "_finalize_bullets", fake_final, raising=False)

    state = {
        "product_id": "p1", "mission_id": 0,
        "anomalies": [["mrr", 100.0, [50.0, 60.0]], ["churn", 9.0, [3.0, 4.0]]],
        "idx": 0, "hypotheses": {},
    }
    await ib._hypothesis_resume(1, {"content": "MRR jumped on a new enterprise deal."}, state)
    # stored first hypothesis, enqueued the second anomaly, did NOT finalize
    assert "mrr" in enq["kw"]["cont_state"]["hypotheses"]
    assert enq["kw"]["cont_state"]["idx"] == 1
    assert "state" not in finalized


@pytest.mark.asyncio
async def test_resume_finalizes_on_last(monkeypatch):
    monkeypatch.setattr(ib, "_enqueue_hypothesis_child",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not enqueue")),
                        raising=False)
    finalized = {}
    async def fake_final(state):
        finalized["state"] = state
    monkeypatch.setattr(ib, "_finalize_bullets", fake_final, raising=False)

    state = {
        "product_id": "p1", "mission_id": 0,
        "anomalies": [["mrr", 100.0, [50.0]]],
        "idx": 0, "hypotheses": {},
    }
    await ib._hypothesis_resume(1, {"content": "One-off annual prepay."}, state)
    assert finalized["state"]["hypotheses"]["mrr"] == "One-off annual prepay."


@pytest.mark.asyncio
async def test_run_no_anomalies_renders_immediately(monkeypatch):
    async def fake_collect(pid):
        return ({}, [])
    monkeypatch.setattr(ib, "collect_metrics", fake_collect)
    called = {}
    async def fake_final(state):
        called["yes"] = True
        return {"ok": True}
    monkeypatch.setattr(ib, "_finalize_bullets", fake_final, raising=False)
    out = await ib.run_investor_bullets("p1")
    assert called.get("yes") is True
```

- [ ] **Step 2: Run it — verify it fails**

Run: `timeout 90 .venv/Scripts/python -m pytest tests/app/jobs/test_investor_bullets_cps.py -q -p no:cacheprovider`
Expected: FAIL — `_hypothesis_resume`, `_enqueue_hypothesis_child`, `_finalize_bullets` undefined.

- [ ] **Step 3: Implement the CPS chain**

In `src/app/jobs/investor_bullets.py`:

(a) Delete `await_inline` from `_enqueue_overhead` (drop the param + pass-through) and **delete** `_call_llm_anomaly_hypothesis` (the await_inline caller). Add `_enqueue_hypothesis_child` and the prompt-building helper:

```python
async def _enqueue_overhead(spec: dict, *, lane: str, **kwargs) -> Any:
    """Thin wrapper around general_beckman.enqueue (monkeypatchable)."""
    from general_beckman import enqueue
    return await enqueue(spec, lane=lane, **kwargs)


def _hypothesis_prompt(metric_name: str, current: float, history: list[float]) -> str:
    median_val = statistics.median(history) if len(history) >= 2 else 0.0
    direction = "above" if current > median_val else "below"
    return (
        f"You are surfacing a data anomaly for a founder's investor update.\n"
        f"Metric: {metric_name}\nThis month: {current}\n"
        f"Trailing 3-month median: {round(median_val, 4)}\n"
        f"Direction: {direction} the median (significant deviation).\n\n"
        f"In ONE sentence, state the most plausible business reason for this change. "
        f"Do NOT guess if uncertain — say 'needs founder explanation'. "
        f"No prose, no preamble. Just the hypothesis sentence."
    )


async def _enqueue_hypothesis_child(state: dict) -> Any:
    """Enqueue the LLM hypothesis child for anomaly state['idx'] with a CPS
    continuation back into _hypothesis_resume."""
    import time, uuid
    from general_beckman.lanes import LANE_ONESHOT
    name, current, history = state["anomalies"][state["idx"]]
    _suffix = f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"
    spec = {
        "title": f"investor_bullets:hypothesis:{name}:{_suffix}",
        "description": f"One-sentence anomaly hypothesis for {name}.",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 2,
        "context": {"llm_call": {
            "raw_dispatch": True, "call_category": "overhead",
            "task": "reviewer", "agent_type": "reviewer", "difficulty": 3,
            "messages": [{"role": "user", "content": _hypothesis_prompt(name, current, history)}],
            "failures": [], "estimated_input_tokens": 300, "estimated_output_tokens": 100,
        }},
    }
    return await _enqueue_overhead(
        spec, lane=LANE_ONESHOT,
        on_complete="investor_bullets.hypothesis.resume",
        on_error="investor_bullets.hypothesis.resume_err",
        cont_state=state,
    )
```

(b) Add `_extract_hypothesis`, `_hypothesis_resume`, `_hypothesis_resume_err`, and `_finalize_bullets` (extracting steps 3-5 of the old `run_investor_bullets`):

```python
def _extract_hypothesis(result: dict) -> str:
    content = (result or {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "").strip()


async def _hypothesis_resume(child_task_id: int, result: dict, state: dict) -> None:
    name = state["anomalies"][state["idx"]][0]
    hyp = _extract_hypothesis(result)
    if hyp:
        state.setdefault("hypotheses", {})[name] = hyp
    state["idx"] = state.get("idx", 0) + 1
    if state["idx"] < len(state["anomalies"]):
        await _enqueue_hypothesis_child(state)
    else:
        await _finalize_bullets(state)


async def _hypothesis_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    # A failed hypothesis child must not stall the chain — skip it and continue.
    logger.warning("investor_bullets: hypothesis child failed; skipping",
                   metric=state["anomalies"][state["idx"]][0])
    state["idx"] = state.get("idx", 0) + 1
    if state["idx"] < len(state["anomalies"]):
        await _enqueue_hypothesis_child(state)
    else:
        await _finalize_bullets(state)


async def _finalize_bullets(state: dict) -> dict:
    """Fetch gaps, render bullets, emit segmented variants. Shared by the
    no-anomaly fast path and the end of the CPS chain."""
    product_id = state["product_id"]
    metrics = state.get("metrics") or {}
    hypotheses = state.get("hypotheses", {})
    gaps = await _fetch_gaps(product_id)
    bullets_md = await render_bullets(metrics, hypotheses, gaps)
    contacts = await _list_contacts(product_id)
    variants = emit_segmented_variants(bullets_md, contacts)
    logger.info("investor_bullets: variants emitted",
                product_id=product_id, variant_count=len(variants))
    return {"ok": True, "variants": len(variants)}
```

(c) Rewrite `run_investor_bullets` as the kickoff (replace its steps 2-5 body):

```python
async def run_investor_bullets(product_id: str = "default", *, mission_id: int = 0) -> dict:
    """Monthly investor bullets kickoff (mr_roboto executor). Anomaly
    hypotheses run as a sequential CPS chain; the chain's tail finalizes."""
    try:
        metrics, missing = await collect_metrics(product_id)
        logger.info("investor_bullets: metrics collected", product_id=product_id,
                    metric_count=len(metrics), missing_sources=missing)

        anomalies = []
        for name, data in metrics.items():
            current = data.get("current", 0.0)
            history = data.get("history", [])
            if _detect_anomaly(name, current, history)["is_anomaly"]:
                anomalies.append([name, current, history])
        anomalies = anomalies[:5]  # cap LLM calls

        state = {
            "product_id": product_id, "mission_id": mission_id,
            "metrics": metrics, "anomalies": anomalies,
            "idx": 0, "hypotheses": {},
        }
        if not anomalies:
            return await _finalize_bullets(state)
        await _enqueue_hypothesis_child(state)
        return {"ok": True, "pending": True, "anomalies": len(anomalies)}
    except Exception as exc:
        logger.error("investor_bullets failed", product_id=product_id, error=str(exc))
        return {"ok": False, "error": str(exc)}
```

(Adjust the trailing lines of the original `run_investor_bullets` — the old steps 3-5 and its return — are now in `_finalize_bullets`; delete them from the kickoff. Preserve any existing exception/return shape the original had after line 819 if it differs.)

(d) Add registration at module bottom:

```python
def register_continuations() -> None:
    from general_beckman.continuations import register_resume
    register_resume("investor_bullets.hypothesis.resume", _hypothesis_resume)
    register_resume("investor_bullets.hypothesis.resume_err", _hypothesis_resume_err)


register_continuations()
```

- [ ] **Step 4: Run it — verify pass**

Run: `timeout 90 .venv/Scripts/python -m pytest tests/app/jobs/test_investor_bullets_cps.py -q -p no:cacheprovider`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the existing investor_bullets tests; reconcile fallout**

Run: `rg -l "investor_bullets" tests` then run each file separately with `timeout 120 ... -m "not llm"`. Any test asserting the old `_call_llm_anomaly_hypothesis` signature or the inline loop must be migrated to drive `_hypothesis_resume` (pattern as in Step 1) or deleted if redundant. Commit message documents deletions.

- [ ] **Step 6: Re-grep — confirm carve-out 2 no longer uses await_inline**

Run: `rg -n "await_inline" src/app/jobs/investor_bullets.py`
Expected: no matches.

- [ ] **Step 7: Commit**

```bash
git add src/app/jobs/investor_bullets.py tests/app/jobs/test_investor_bullets_cps.py tests/
git commit -m "refactor(investor_bullets): CPS-migrate anomaly hypotheses off await_inline (carve-out 2)"
```

---

## Task 6: Register both handler modules for restart-recovery

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/continuations.py`
- Test: `packages/general_beckman/tests/test_handler_modules_registered.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# packages/general_beckman/tests/test_handler_modules_registered.py
"""The two SP5 CPS migrations must be in _HANDLER_MODULES so their resume
handlers exist after a restart (else continuation rows stay pending forever)."""
from general_beckman.continuations import _HANDLER_MODULES, _HANDLERS, register_startup_handlers


def test_sp5_modules_listed():
    assert "src.core.task_classifier" in _HANDLER_MODULES
    assert "src.app.jobs.investor_bullets" in _HANDLER_MODULES


def test_sp5_handlers_register_on_startup():
    register_startup_handlers()
    assert "task_classifier.classify.resume" in _HANDLERS
    assert "investor_bullets.hypothesis.resume" in _HANDLERS
    assert "investor_bullets.hypothesis.resume_err" in _HANDLERS
```

- [ ] **Step 2: Run it — verify it fails**

Run: `PYTHONPATH=".;$(printf '%s;' packages/*/src)" timeout 90 .venv/Scripts/python -m pytest packages/general_beckman/tests/test_handler_modules_registered.py -q -p no:cacheprovider`
Expected: FAIL — `src.core.task_classifier` not in the list; investor_bullets present only as a comment.

- [ ] **Step 3: Edit `_HANDLER_MODULES`**

Replace the deferred comment + add the classifier:

```python
    # CPS SP2 — edge-group migrations:
    "src.app.telegram_bot",
    "src.app.interview",
    "src.app.meetings",
    "src.app.jobs.faq_regen",
    # CPS SP5 — await_inline finale (both carve-outs migrated 2026-06-11):
    "src.core.task_classifier",
    "src.app.jobs.investor_bullets",
    # CPS SP3 - in-task deadlock set:
    "general_beckman.posthook_continuations",
```

- [ ] **Step 4: Run it — verify pass**

Run: `PYTHONPATH=".;$(printf '%s;' packages/*/src)" timeout 90 .venv/Scripts/python -m pytest packages/general_beckman/tests/test_handler_modules_registered.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/continuations.py packages/general_beckman/tests/test_handler_modules_registered.py
git commit -m "feat(beckman): register SP5 CPS handler modules for restart recovery"
```

---

## Task 7: Delete `await_inline` from `enqueue` + inline-waiter machinery

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Modify: `packages/general_beckman/README.md`

- [ ] **Step 1: Confirm zero remaining callers**

Run: `rg -n "await_inline\s*=\s*True" src packages | rg -v "/tests/|\.md:"`
Expected: NO matches (both carve-outs migrated). If anything appears, stop and migrate it first.

- [ ] **Step 2: Check `TaskResult` / `resolve_inline` external usage before deleting**

Run: `rg -n "resolve_inline|_inline_waiters|INLINE_TIMEOUT|TaskResult" src packages --glob '!**/__init__.py'`
- If `TaskResult` is imported anywhere live (outside the inline machinery), KEEP the dataclass but remove only the inline-wait plumbing. If not, delete `TaskResult` too.
- Record findings in the commit message.

- [ ] **Step 3: Remove the machinery**

In `packages/general_beckman/src/general_beckman/__init__.py`:
- Delete the `await_inline: bool = False` param (line ~1281), its docstring block (~1297-1299), and the mutual-exclusion `ValueError` check (~1319-1323).
- Delete the inline-wait path: from `if not await_inline: return task_id` collapse to always `return task_id`; delete lines ~1377-1401 (the `_inline_keepalive_cm` / future / `wait_for` block).
- Delete `resolve_inline` (~83-87), `_inline_waiters` (~80), `INLINE_TIMEOUT` (~76-77).
- Delete the terminal-hook resolve block (~1132-1140: `# await_inline resolve...` through `resolve_inline(task_id, _tr)`).
- Remove `"resolve_inline"`, `"INLINE_TIMEOUT"`, `"_inline_waiters"` from `__all__` (~19-20), and `"TaskResult"` if deleted in Step 2. Update the `enqueue(...)` signature line in the module docstring (~6) and the `resolve_inline` line (~7).
- Return type of `enqueue` becomes `int | None` (no longer `int | TaskResult`).

- [ ] **Step 4: Update the README**

In `packages/general_beckman/README.md`, delete the `await_inline=True` examples and the "mutually exclusive" bullet (both EN ~76-85,~200 and TR ~312-321,~437 sections). Replace with a one-line note that blocking inline waits were removed in SP5 — all enqueues are fire-and-continue with optional `on_complete`/`on_error` continuations.

- [ ] **Step 5: Import smoke + targeted Beckman tests**

Run:
`PYTHONPATH=".;$(printf '%s;' packages/*/src)" timeout 60 .venv/Scripts/python -c "import general_beckman; print(general_beckman.enqueue)"`
`PYTHONPATH=".;$(printf '%s;' packages/*/src)" timeout 180 .venv/Scripts/python -m pytest packages/general_beckman/tests -q -p no:cacheprovider`
Expected: import OK; tests green except `test_no_inline_deadlock.py` assertions that referenced the now-deleted param — fixed in Task 8.

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py packages/general_beckman/README.md
git commit -m "refactor(beckman)!: delete await_inline blocking primitive (SP5 finale)"
```

---

## Task 8: Strengthen the deadlock guards

**Files:**
- Modify: `packages/general_beckman/tests/test_no_inline_deadlock.py`

The existing guards assert no posthook child passes `await_inline=True`. With the param gone, passing it would be a `TypeError`. Convert the guards to assert the param no longer exists on `enqueue` and the source is clean.

- [ ] **Step 1: Add a signature guard**

Add to `test_no_inline_deadlock.py`:

```python
import inspect
import general_beckman


def test_enqueue_has_no_await_inline_param():
    """SP5: the blocking inline-wait primitive is deleted. enqueue must not
    accept await_inline anymore."""
    sig = inspect.signature(general_beckman.enqueue)
    assert "await_inline" not in sig.parameters


def test_no_inline_waiter_machinery():
    assert not hasattr(general_beckman, "resolve_inline")
    assert not hasattr(general_beckman, "_inline_waiters")
```

Keep the existing `_assert_no_await_inline` content checks (still valid — they assert no source line contains `await_inline=True`). If any existing guard now imports `resolve_inline` / `INLINE_TIMEOUT`, delete that import and the assertion that used it.

- [ ] **Step 2: Run the guards**

Run: `PYTHONPATH=".;$(printf '%s;' packages/*/src)" timeout 120 .venv/Scripts/python -m pytest packages/general_beckman/tests/test_no_inline_deadlock.py packages/general_beckman/tests/test_handler_modules_registered.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add packages/general_beckman/tests/test_no_inline_deadlock.py
git commit -m "test(beckman): guard that await_inline param is deleted"
```

---

## Task 9: Full verification + code review + merge

**Files:** none (verification)

- [ ] **Step 1: Re-grep the whole tree**

Run: `rg -n "await_inline" src packages | rg -v "\.md:"`
Expected: only doc-comment/test references describing the deletion; NO `await_inline=True` in non-test source.

- [ ] **Step 2: Run the affected suites (separate invocations, never concurrent)**

Run, each separately with `timeout`:
- `packages/general_beckman/tests` (with the PYTHONPATH hack)
- `tests/core/test_parse_classification.py tests/core/test_classify_cps.py`
- `tests/app/jobs/test_investor_bullets_cps.py`
- the migrated integration files from Tasks 4 & 5 (`-m "not llm"`)
- import smoke: `python -c "import src.core.task_classifier, src.app.jobs.investor_bullets, general_beckman"`

Record the full pass count. Any red must be a pre-recorded baseline red (Task 1 Step 3) — verify.

- [ ] **Step 3: Self-review the diff**

Run: `git diff main...HEAD --stat` and read the full diff. Confirm: no `await_inline` survivors; both modules registered; `_extract_json` still importable by telegram_bot; no orphaned imports of deleted symbols.

- [ ] **Step 4: Request code review**

Use `superpowers:requesting-code-review`. Focus areas to flag for the reviewer:
- carve-out 2 sequential chain: does a failed hypothesis child (`resume_err`) correctly continue the chain, and does the final anomaly always reach `_finalize_bullets`? (no anomaly → still finalizes via the kickoff fast path).
- continuation-state size: `cont_state` carries `metrics` + `anomalies` — bounded (≤5 anomalies, capped) but confirm it serializes (JSON) without blowing the row.
- did anything outside the two modules rely on `classify_task` returning a `TaskClassification` or on `TaskResult`? (re-grep evidence).

- [ ] **Step 5: Finish the branch**

Use `superpowers:finishing-a-development-branch`. Merge to `main` (project convention: direct to main, no PR). Note in the handoff that the change is **restart-gated** — live KutAI runs old code until a founder `/restart`.

- [ ] **Step 6: Write the closing handoff + memory**

Create `docs/handoff/2026-06-11-sp5-await-inline-DONE.md`: SP5 fully closed; the handoff premise (carve-out 1 = redesign) was a stale-comment artifact — it was dead code, CPS-migrated as a kept shell; both carve-outs migrated; `await_inline` deleted; restart-gated. Update memory `project_sp5_request_deleted_20260610` / add an SP5-complete note.

---

## Self-Review (author checklist — completed)

**Spec coverage:** Both carve-outs (Tasks 2-5), restart registration (Task 6), param+machinery deletion (Task 7), guards (Task 8), verification/merge (Task 9). The handoff's deletion-order steps 1-4 all map to tasks. ✓

**Placeholder scan:** Each code step shows full code. Integration-test migration (Tasks 4/5 Step 2) gives the exact transform pattern rather than rewriting every assertion — acceptable because the call sites are mechanical and numerous; the pattern is complete. ✓

**Type consistency:** `classify_task` → `int | None` everywhere (Tasks 3, 4, 7). Resume handler signature `(child_task_id, result, state)` matches the substrate contract (Tasks 3, 5, 6). `_finalize_bullets(state)` consumes the same `state` keys the kickoff/resume write (`product_id`, `metrics`, `anomalies`, `idx`, `hypotheses`). Handler names identical across register + test (`task_classifier.classify.resume`, `investor_bullets.hypothesis.resume`/`.resume_err`). ✓
