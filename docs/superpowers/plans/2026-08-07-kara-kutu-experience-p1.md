# Kara Kutu Experience — P1 (success round-trip MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove per-step success experience is captured at grade-pass (`apply.py:6073`) under a single-source-of-truth operation key and retrieved by the LIVE exemplar reader (`context.py:~1349`) at the same step.

**Architecture:** Extend the existing `kara_kutu` package with `capture`/`recall` verbs + a pure key-builder `i2p_step_key`. Reuse the already-shipped `workflow_exemplars` store (do NOT move it in P1 — see Task 2 justification), wrapping it behind `kara_kutu.recall`/`capture`. Wire `capture` at the grade-pass hook and re-key the reader so both sides build their key with the same helper on `task['agent_type']` (not `profile.name`). The `quality` score derives from `GradeResult` booleans with a default-safe helper.

**Tech Stack:** Python 3.10 (async), `dabidabi` (aiosqlite SQLite engine), `yazbunu` (logging), pytest + pytest-asyncio, Windows venv at `.venv`.

---

## Scope / out of scope

**In scope (P1 = SUCCESS round-trip MVP only):** the smallest change that proves per-step success experience is captured at grade-pass and retrieved by the LIVE reader. Concretely: key-builder, `recall`, `capture` (success mode only), the quality helper, the grade-pass wiring, the reader re-key, and a round-trip integration test.

**OUT of scope (later plans — do NOT implement here):**
- Failure-mode capture (`pattern`/`fix`/`severity`) and migrating `_maybe_emit_lesson_from_posthook_fail`. The `capture` signature includes the failure params + the exactly-one-mode guard **stub only** (Task 3) so the interface is stable, but the failure branch raises `NotImplementedError`.
- `mission_lessons` absorption / routing new writes through `capture`.
- The `intersect` multi-source refactor / `Experience`-adapter injection (that is P2).
- Chroma deletion / rebuild-swap / reclaim / hygiene (P3/P4).

---

## Key facts established by reading the code (do not re-derive)

- **Hook site** `packages/general_beckman/src/general_beckman/apply.py:6073` — `if a.kind == "grade" and a.passed:` inside `async def _apply_posthook_verdict_locked(task: dict, a: PostHookVerdict)` (def at `apply.py:5498`).
  - `source = await get_task(a.source_task_id)` is bound at `apply.py:5503`; `source` is a task dict with `source.get("result")`, `source.get("agent_type")`, `source.get("title")`, `source.get("mission_id")`.
  - `a` is a `PostHookVerdict` (`packages/general_beckman/src/general_beckman/result_router.py:86`): fields `source_task_id:int`, `kind:str`, `passed:bool`, `raw:dict`, `action`, `new_result`.
  - **`a.raw` for a grade verdict carries the grade booleans directly** as dict keys `relevant`/`complete`/`well_formed`/`coherent`/`passed` (built by `_grade_raw_dict` in `packages/general_beckman/src/general_beckman/posthook_continuations.py:75-93`).
  - **`a.raw` does NOT carry the graded RESULT text.** The result text lives in `source.get("result")`. (Confirmed at `posthook_continuations.py:81-92` — `raw` only has verdict booleans + prose fields `insight`/`strategy`/`situation`.) The plan uses `source.get("result")` for the exemplar body. **This is a spec deviation to flag** — see the final report.
- **Live reader** `packages/coulson/src/coulson/context.py:1346-1350` calls `lookup_exemplars(workflow=_wf_name, step_id=_step_id, agent_type=profile.name)`. The `agent_type=profile.name` at line **1349** is the mis-key. Line **1241** already computes `agent_type = task.get("agent_type") or profile.name` in the same function — the fix is to reuse `task['agent_type']` here.
- **`_wf_name`** is read at `context.py:1332-1343` via `SELECT workflow FROM missions WHERE id=?`. **On the live DB `missions.workflow` is `''` for mission 90** (probe below). So the workflow key component is `''`, not `'i2p'`. Capture MUST read the workflow the identical way so both sides key on `''`.
- **Grade booleans** `packages/coulson/src/coulson/posthooks/grading.py:20-32` — `@dataclass GradeResult` with `relevant/complete/well_formed/coherent: Optional[bool] = None` (each may be `None`).
- **Exemplar store** `src/memory/workflow_exemplars.py` — `capture_exemplar(*, workflow, step_id, agent_type, result, quality_score, task_id, mission_id=None)`, `lookup_exemplars(*, workflow, step_id, agent_type)`, `extract_step_id(title)`, `_tool_recipe_for_task(task_id)`, `MAX_PER_KEY = 3`. It imports `from src.infra.db import get_db` (an alias shim → `dabidabi`). **Zero non-test callers of `capture_exemplar`** confirmed by grep.
- **kara_kutu DB pattern**: `mission_lessons.py` uses `from dabidabi import get_db` and `from yazbunu import get_logger` (NOT `src.infra`).
- **Test temp-DB pattern** (repo-canonical, from `tests/test_beckman_posthooks.py:80-89`): monkeypatch `DB_PATH` env + monkeypatch `src.infra.db.DB_PATH`, close any cached connection, then `from src.infra.db import init_db, add_task, ...; await init_db()`. Use this verbatim.

---

## Task 0: Hinge verification (investigation — NOT TDD)

Settles the design hinge (spec §11): for completed i2p `[X.Y]` steps, does `tasks.agent_type` equal the `profile.name` coulson resolves? This decides whether re-keying the reader (Task 6) is load-bearing.

**Files:** none created. Read-only probe against the live DB.

- [ ] **Step 1: Run the read-only probe**

The live DB is `C:\Users\sakir\ai\kutai\kutai.db` (from `.env` `DB_PATH`). Open it `?mode=ro` so the running bot is never touched.

```bash
python - <<'PY'
import sqlite3, json
db = r"C:\Users\sakir\ai\kutai\kutai.db"
uri = f"file:///{db.replace(chr(92),'/')}?mode=ro"
con = sqlite3.connect(uri, uri=True); cur = con.cursor()

# (a) agent_type distribution for completed [X.Y] steps
cur.execute("""SELECT agent_type, COUNT(*) FROM tasks
  WHERE status='completed' AND title GLOB '[[]*.*]*'
  GROUP BY agent_type ORDER BY 2 DESC""")
print("agent_type dist:", cur.fetchall())

# (b) tasks.agent_type vs context.agent_type divergence
cur.execute("""SELECT id, agent_type, context FROM tasks
  WHERE status='completed' AND title GLOB '[[]*.*]*' ORDER BY id DESC LIMIT 50""")
div = 0
for tid, at, ctx in cur.fetchall():
    try: cat = (json.loads(ctx or "{}") or {}).get("agent_type")
    except Exception: cat = None
    if cat and cat != at:
        div += 1; print("DIVERGE", tid, at, cat)
print("ctx-vs-column divergences:", div)

# (c) workflow key component: what does missions.workflow hold?
cur.execute("SELECT workflow, COUNT(*) FROM missions GROUP BY workflow")
print("missions.workflow dist:", cur.fetchall())

# (d) existing seeded exemplars (if any)
try:
    cur.execute("SELECT workflow, step_id, agent_type, COUNT(*) FROM workflow_exemplars GROUP BY 1,2,3")
    print("workflow_exemplars:", cur.fetchall())
except Exception as e:
    print("workflow_exemplars:", e)
con.close()
PY
```

- [ ] **Step 2: Interpret**

**Expected (already observed 2026-08-07):**
- (a) real profile names: `analyst 42, mechanical 19, writer 14, coder 11, architect 11, ...` — so `tasks.agent_type` IS populated for completed `[X.Y]` steps.
- (b) `ctx-vs-column divergences: 0` — `tasks.agent_type` is authoritative and stable.
- (c) `missions.workflow dist: [('', 1)]` — the workflow component is the empty string on the live mission, NOT `'i2p'`.
- (d) 3 stale seed rows keyed `('i2p','3.10a','writer')`.

**Interpretation / decision:**
- `context.py:1241` resolves `agent_type = task.get("agent_type") or profile.name`. Since `task['agent_type']` is populated for every sampled `[X.Y]` step, the reader's `profile.name` at 1349 currently *coincides* with `task['agent_type']` for these tasks — so re-keying is **safe** (no regression for populated tasks) and **correct** (removes the divergence for any future task where `profile.name != task['agent_type']`, e.g. a fallback profile). Re-keying is load-bearing as a *consistency guarantee between capture and read*, even though it is behavior-preserving for today's populated rows. **Proceed with Task 6.**
- The **workflow component** must be derived identically on both sides. Both capture (Task 5) and reader (Task 6) read `SELECT workflow FROM missions WHERE id=?` → both get `''` → they match. The stale `'i2p'` seed rows will NOT match the live reader and are irrelevant. **Do NOT hardcode `'i2p'` in the key-builder.**

---

## Task 1: Key-builder `i2p_step_key`

Single source of truth for the per-step operation key. Both capture and the live reader call this so the key can never drift.

**Files:**
- Create: `packages/kara_kutu/src/kara_kutu/experience.py`
- Modify: `packages/kara_kutu/src/kara_kutu/__init__.py` (export `i2p_step_key`)
- Test: `packages/kara_kutu/tests/test_experience_key.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/kara_kutu/tests/test_experience_key.py
"""P1 — pure key-builder for the per-step operation key."""
from __future__ import annotations


def test_i2p_step_key_roundtrip_deterministic():
    from kara_kutu import i2p_step_key
    k1 = i2p_step_key("", "7.4a", "coder")
    k2 = i2p_step_key("", "7.4a", "coder")
    assert k1 == k2
    assert isinstance(k1, str)


def test_i2p_step_key_components_distinguish():
    from kara_kutu import i2p_step_key
    base = i2p_step_key("", "7.4a", "coder")
    assert i2p_step_key("i2p", "7.4a", "coder") != base   # workflow matters
    assert i2p_step_key("", "7.4b", "coder") != base       # step matters
    assert i2p_step_key("", "7.4a", "writer") != base      # agent matters


def test_i2p_step_key_normalizes_none_and_whitespace():
    from kara_kutu import i2p_step_key
    # None workflow == "" workflow; agent/step stripped.
    assert i2p_step_key(None, "7.4a", "coder") == i2p_step_key("", "7.4a", "coder")
    assert i2p_step_key("", " 7.4a ", " coder ") == i2p_step_key("", "7.4a", "coder")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 python -m pytest packages/kara_kutu/tests/test_experience_key.py -v`
Expected: FAIL — `ImportError: cannot import name 'i2p_step_key' from 'kara_kutu'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/kara_kutu/src/kara_kutu/experience.py
"""kara_kutu — per-step SUCCESS experience (P1).

Single source of truth for the operation key + the capture/recall verbs over
the workflow-step exemplar store. P1 wires ONLY the success round-trip; the
failure mode (pattern/fix) is stubbed and raises NotImplementedError.

DB access follows kara_kutu convention: dabidabi / yazbunu, not src.infra.
"""
from __future__ import annotations

from yazbunu import get_logger

logger = get_logger("kara_kutu.experience")

# key component separator — a control char that cannot appear in a step id,
# workflow name, or agent_type, so the key is unambiguous and reversible.
_KEY_SEP = "\x1f"


def i2p_step_key(workflow, step_id: str, agent_type: str) -> str:
    """Build the per-step operation key `(workflow, step_id, agent_type)`.

    SINGLE source of truth: capture (apply.py grade-pass) AND the live reader
    (coulson.context) both build their lookup key with this helper so the two
    sides can never drift. ``agent_type`` MUST be ``task['agent_type']`` (NOT
    ``profile.name``) on both sides. ``workflow`` MUST be read the same way on
    both sides (``SELECT workflow FROM missions``) — do NOT hardcode 'i2p'.
    """
    wf = (workflow or "").strip()
    sid = (step_id or "").strip()
    ag = (agent_type or "").strip()
    return _KEY_SEP.join((wf, sid, ag))
```

- [ ] **Step 4: Export from the package**

```python
# packages/kara_kutu/src/kara_kutu/__init__.py — add to imports block after the mission_lessons import
from .experience import i2p_step_key
```

And add `"i2p_step_key"` to the `__all__` list in the same file (append after `"emit_lessons_from_dlq_patterns",`).

- [ ] **Step 5: Run test to verify it passes**

Run: `timeout 30 python -m pytest packages/kara_kutu/tests/test_experience_key.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
rtk git add packages/kara_kutu/src/kara_kutu/experience.py packages/kara_kutu/src/kara_kutu/__init__.py packages/kara_kutu/tests/test_experience_key.py
rtk git commit -m "feat(kara_kutu): add i2p_step_key single-source key-builder (P1 exp round-trip)"
```

---

## Task 2: `recall` + `Experience` dataclass

Wrap the already-shipped `workflow_exemplars.lookup_exemplars` behind `kara_kutu.recall`. **Decision: WRAP, do not move the store in P1.** Justification: moving `workflow_exemplars.py` into kara_kutu forces repointing its `src.infra.db` import to `dabidabi`, updating the reader import in `context.py:1324`, and re-homing its dedicated tests — a larger diff that risks the LIVE reader path for zero P1 benefit. Wrapping is the smaller diff and keeps P1 focused on the round-trip. (Spec §7 "move into kara_kutu" is deferred to a later hygiene pass; noted as a gap in the report.)

**Files:**
- Modify: `packages/kara_kutu/src/kara_kutu/experience.py` (add `Experience`, `recall`)
- Modify: `packages/kara_kutu/src/kara_kutu/__init__.py` (export `Experience`, `recall`)
- Test: `packages/kara_kutu/tests/test_experience_recall.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/kara_kutu/tests/test_experience_recall.py
"""P1 — recall reads exemplars for the exact key, best-first."""
from __future__ import annotations
import pytest


@pytest.mark.asyncio
async def test_recall_returns_seeded_row(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, get_db
    await init_db()

    # Seed one exemplar directly via the store's ensure+insert path.
    from src.memory.workflow_exemplars import _ensure_table
    await _ensure_table()
    db = await get_db()
    await db.execute(
        "INSERT INTO workflow_exemplars "
        "(workflow, step_id, agent_type, result, tool_recipe, quality_score, task_id, mission_id) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("", "7.4a", "coder", "the passing output", "[]", 0.95, 4242, 90),
    )
    await db.commit()

    from kara_kutu import recall, i2p_step_key, Experience
    key = i2p_step_key("", "7.4a", "coder")
    out = await recall(key, limit=3)
    assert len(out) == 1
    exp = out[0]
    assert isinstance(exp, Experience)
    assert exp.kind == "step_success"
    assert exp.key == key
    assert exp.text == "the passing output"
    assert exp.quality == 0.95
    assert exp.task_id == 4242


@pytest.mark.asyncio
async def test_recall_empty_key_returns_nothing(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    from src.infra.db import init_db
    await init_db()

    from kara_kutu import recall, i2p_step_key
    from src.memory.workflow_exemplars import _ensure_table
    await _ensure_table()
    out = await recall(i2p_step_key("", "9.9z", "nobody"), limit=3)
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 python -m pytest packages/kara_kutu/tests/test_experience_recall.py -v`
Expected: FAIL — `ImportError: cannot import name 'recall' from 'kara_kutu'` (and `Experience`).

- [ ] **Step 3: Write minimal implementation**

Add to `packages/kara_kutu/src/kara_kutu/experience.py` (after `i2p_step_key`):

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experience:
    """One retrieved unit of internal experience.

    P1 populates only the success-mode fields. ``fix``/``pattern`` stay None
    for step_success; ``occurrences``/``severity``/``suppressed`` default to the
    success-mode neutrals and are carried for the P2 mission_lessons path.
    """
    kind: str
    key: str
    text: str
    tool_recipe: list = field(default_factory=list)
    quality: float = 0.0
    task_id: int = 0
    created_at: str = ""
    fix: "str | None" = None
    occurrences: int = 1
    severity: str = "info"
    suppressed: bool = False


def _parse_key(key: str) -> tuple[str, str, str]:
    """Reverse i2p_step_key back to (workflow, step_id, agent_type)."""
    parts = (key or "").split(_KEY_SEP)
    if len(parts) != 3:
        return ("", "", "")
    return (parts[0], parts[1], parts[2])


async def recall(key: str, *, limit: int = 3) -> "list[Experience]":
    """Per-step exact-key recall: top-N by quality, best first.

    P1: dispatches only the step-success namespace (the per-step exemplar
    store). The per-error-domain namespace is P2.
    """
    from src.memory.workflow_exemplars import lookup_exemplars

    workflow, step_id, agent_type = _parse_key(key)
    if not step_id or not agent_type:
        return []
    rows = await lookup_exemplars(
        workflow=workflow, step_id=step_id, agent_type=agent_type,
    )
    out: list[Experience] = []
    for r in rows[:limit]:
        out.append(Experience(
            kind="step_success",
            key=key,
            text=r.get("result", "") or "",
            tool_recipe=r.get("tool_recipe") or [],
            quality=float(r.get("quality_score") or 0.0),
            task_id=int(r.get("task_id") or 0),
        ))
    return out
```

- [ ] **Step 4: Export from the package**

```python
# packages/kara_kutu/src/kara_kutu/__init__.py — extend the experience import
from .experience import i2p_step_key, Experience, recall
```

Add `"Experience"` and `"recall"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `timeout 60 python -m pytest packages/kara_kutu/tests/test_experience_recall.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
rtk git add packages/kara_kutu/src/kara_kutu/experience.py packages/kara_kutu/src/kara_kutu/__init__.py packages/kara_kutu/tests/test_experience_recall.py
rtk git commit -m "feat(kara_kutu): recall(key) + Experience over workflow_exemplars (P1)"
```

---

## Task 3: `capture` (success mode) + exactly-one-mode guard stub

Success capture writes an exemplar. Recipe is reconstructed from `task_id` internally (the store's `capture_exemplar` already calls `_tool_recipe_for_task`). The failure mode is a stub (raises) — its params exist so the interface is stable for P2.

**Files:**
- Modify: `packages/kara_kutu/src/kara_kutu/experience.py` (add `capture`)
- Modify: `packages/kara_kutu/src/kara_kutu/__init__.py` (export `capture`)
- Test: `packages/kara_kutu/tests/test_experience_capture.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/kara_kutu/tests/test_experience_capture.py
"""P1 — capture(success) writes an exemplar row; mode guard is exactly-one."""
from __future__ import annotations
import pytest


@pytest.mark.asyncio
async def test_capture_success_writes_row(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    from src.infra.db import init_db, get_db
    await init_db()

    from kara_kutu import capture, recall, i2p_step_key
    key = i2p_step_key("", "7.4a", "coder")
    ok = await capture(key, task_id=555, result="captured output", quality=0.8, mission_id=90)
    assert ok is True

    got = await recall(key, limit=3)
    assert len(got) == 1
    assert got[0].text == "captured output"
    assert got[0].quality == 0.8
    assert got[0].task_id == 555


@pytest.mark.asyncio
async def test_capture_rejects_both_modes(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    from src.infra.db import init_db
    await init_db()

    from kara_kutu import capture, i2p_step_key
    key = i2p_step_key("", "7.4a", "coder")
    # both success (result) AND failure (pattern/fix) → guard raises
    with pytest.raises(ValueError):
        await capture(key, task_id=1, result="r", pattern="p", fix="f")
    # neither mode → guard raises
    with pytest.raises(ValueError):
        await capture(key, task_id=1)


@pytest.mark.asyncio
async def test_capture_failure_mode_stubbed(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    from src.infra.db import init_db
    await init_db()

    from kara_kutu import capture, i2p_step_key
    key = i2p_step_key("", "7.4a", "coder")
    with pytest.raises(NotImplementedError):
        await capture(key, task_id=1, pattern="boom", fix="do X")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 python -m pytest packages/kara_kutu/tests/test_experience_capture.py -v`
Expected: FAIL — `ImportError: cannot import name 'capture' from 'kara_kutu'`

- [ ] **Step 3: Write minimal implementation**

Add to `packages/kara_kutu/src/kara_kutu/experience.py` (after `recall`):

```python
async def capture(
    key: str,
    *,
    task_id: int,
    mission_id: "int | None" = None,
    result: "str | None" = None,
    quality: "float | None" = None,           # success mode
    pattern: "str | None" = None,
    fix: "str | None" = None,
    severity: str = "info",                     # failure mode (P2)
) -> bool:
    """Capture one unit of experience under ``key``.

    GUARD exactly one mode: success == (result is not None); failure ==
    (pattern is not None and fix is not None). Neither / both → ValueError.

    P1 implements SUCCESS only. The per-step exemplar store reconstructs the
    tool-recipe from ``task_id`` internally (capture_exemplar → _tool_recipe_for_task).
    """
    is_success = result is not None
    is_failure = pattern is not None and fix is not None
    if is_success == is_failure:
        raise ValueError(
            "capture: exactly one mode required — success(result=...) XOR "
            "failure(pattern=..., fix=...)"
        )

    if is_failure:
        # P2: reuse kara_kutu.upsert_mission_lesson. Stubbed in P1.
        raise NotImplementedError("capture failure-mode lands in P2")

    # Success mode.
    from src.memory.workflow_exemplars import capture_exemplar
    workflow, step_id, agent_type = _parse_key(key)
    if not step_id or not agent_type:
        logger.debug("capture: unkeyable success drop key=%r", key)
        return False
    return await capture_exemplar(
        workflow=workflow,
        step_id=step_id,
        agent_type=agent_type,
        result=result or "",
        quality_score=float(quality if quality is not None else 0.0),
        task_id=int(task_id or 0),
        mission_id=mission_id,
    )
```

- [ ] **Step 4: Export from the package**

```python
# packages/kara_kutu/src/kara_kutu/__init__.py — extend the experience import
from .experience import i2p_step_key, Experience, recall, capture
```

Add `"capture"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `timeout 60 python -m pytest packages/kara_kutu/tests/test_experience_capture.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
rtk git add packages/kara_kutu/src/kara_kutu/experience.py packages/kara_kutu/src/kara_kutu/__init__.py packages/kara_kutu/tests/test_experience_capture.py
rtk git commit -m "feat(kara_kutu): capture(success) + exactly-one-mode guard (failure stubbed for P2)"
```

---

## Task 4: `quality` from grade booleans (default-safe pure helper)

Derive a scalar quality from the `GradeResult` booleans carried in `a.raw`. Default-safe: all-true → high; missing/None → mid; explicit failures pull it down; never 0 (a grade-PASS is never worthless). This is a **pure function of a dict** so it is trivially testable and has no DB/LLM dependency.

**Files:**
- Modify: `packages/kara_kutu/src/kara_kutu/experience.py` (add `quality_from_grade`)
- Modify: `packages/kara_kutu/src/kara_kutu/__init__.py` (export `quality_from_grade`)
- Test: `packages/kara_kutu/tests/test_experience_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/kara_kutu/tests/test_experience_quality.py
"""P1 — quality derived from grade booleans, default-safe."""
from __future__ import annotations
import pytest


def test_all_true_is_high():
    from kara_kutu import quality_from_grade
    q = quality_from_grade({"relevant": True, "complete": True,
                            "well_formed": True, "coherent": True})
    assert q == pytest.approx(1.0)


def test_all_missing_is_mid_never_zero():
    from kara_kutu import quality_from_grade
    # short-circuit grade-pass builds raw without the axis booleans
    q = quality_from_grade({"passed": True})
    assert q == pytest.approx(0.7)
    assert q > 0.0


def test_none_axes_treated_as_pass():
    from kara_kutu import quality_from_grade
    q = quality_from_grade({"relevant": None, "complete": None,
                            "well_formed": None, "coherent": None})
    assert q == pytest.approx(0.7)


def test_explicit_false_lowers_but_never_zero():
    from kara_kutu import quality_from_grade
    q = quality_from_grade({"relevant": True, "complete": False,
                            "well_formed": True, "coherent": True})
    assert 0.0 < q < 1.0
    assert q < quality_from_grade({"relevant": True, "complete": True,
                                   "well_formed": True, "coherent": True})


def test_empty_or_non_dict_is_mid():
    from kara_kutu import quality_from_grade
    assert quality_from_grade({}) == pytest.approx(0.7)
    assert quality_from_grade(None) == pytest.approx(0.7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 python -m pytest packages/kara_kutu/tests/test_experience_quality.py -v`
Expected: FAIL — `ImportError: cannot import name 'quality_from_grade' from 'kara_kutu'`

- [ ] **Step 3: Write minimal implementation**

Add to `packages/kara_kutu/src/kara_kutu/experience.py` (after `capture`):

```python
# Grade axes carried in the grade verdict raw dict (posthook_continuations._grade_raw_dict).
_GRADE_AXES = ("relevant", "complete", "well_formed", "coherent")

# A grade-PASS with no axis evidence is worth this much: not perfect, never 0.
_QUALITY_DEFAULT = 0.7


def quality_from_grade(raw: "dict | None") -> float:
    """Map grade booleans → quality in (0, 1].

    - all four axes True (or None) → 1.0
    - axes absent / raw empty / non-dict → _QUALITY_DEFAULT (mid, never 0)
    - each explicit False subtracts a fixed penalty, floored so a grade-PASS
      never reads as 0.

    None is treated as "not evaluated" == pass (short-circuit grade-passes build
    raw without the axis booleans; missing must never read as failure).
    """
    if not isinstance(raw, dict):
        return _QUALITY_DEFAULT
    present = [raw.get(a) for a in _GRADE_AXES if a in raw]
    if not present:
        return _QUALITY_DEFAULT
    falses = sum(1 for v in present if v is False)
    if falses == 0:
        return 1.0
    # 0.25 penalty per explicit False, floored at 0.1 (never 0 on a pass).
    return max(0.1, 1.0 - 0.25 * falses)
```

- [ ] **Step 4: Export from the package**

```python
# packages/kara_kutu/src/kara_kutu/__init__.py — extend the experience import
from .experience import i2p_step_key, Experience, recall, capture, quality_from_grade
```

Add `"quality_from_grade"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `timeout 30 python -m pytest packages/kara_kutu/tests/test_experience_quality.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
rtk git add packages/kara_kutu/src/kara_kutu/experience.py packages/kara_kutu/src/kara_kutu/__init__.py packages/kara_kutu/tests/test_experience_quality.py
rtk git commit -m "feat(kara_kutu): quality_from_grade default-safe helper (P1)"
```

---

## Task 5: Wire `capture` at the grade-pass hook (`apply.py:6073`)

On grade-PASS, best-effort capture a step-success exemplar. Wrapped in `try/except` like the neighbouring hooks (a capture miss must NEVER block or DLQ the source). Key built via `i2p_step_key` on `source['agent_type']`, `extract_step_id(source['title'])`, and the workflow read the same way the reader reads it.

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/apply.py` — inside the `if a.kind == "grade" and a.passed:` block (opens at line 6073), insert the capture call after the block sets `ctx["_pending_posthooks"] = pending` at line 6080 (before the `if not pending:` branch at 6081), so it runs on every grade-PASS regardless of pending summaries.
- Test: `packages/general_beckman/tests/test_grade_pass_captures_experience.py`

- [ ] **Step 1: Write the failing test**

The test patches `kara_kutu.capture` on the `general_beckman.apply` module namespace and asserts a grade-PASS fires it with the correct key. It seeds a real source task so `source` fields resolve.

```python
# packages/general_beckman/tests/test_grade_pass_captures_experience.py
"""P1 — a grade-PASS at apply.py:6073 captures a step-success experience."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_grade_pass_calls_capture_with_step_key(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, update_task
    await init_db()

    # Source: an i2p [7.4a] coder step that has passed grading.
    source_id = await add_task(
        title="[7.4a] db_schema_author",
        description="",
        agent_type="coder",
        mission_id=1,
        context={"_pending_posthooks": ["grade"]},
    )
    await update_task(source_id, status="ungraded", result="the schema output")

    from general_beckman.result_router import PostHookVerdict
    from general_beckman import apply as _apply_mod

    verdict = PostHookVerdict(
        source_task_id=source_id, kind="grade", passed=True,
        raw={"passed": True, "relevant": True, "complete": True,
             "well_formed": True, "coherent": True},
    )
    source = await _db_mod.get_task(source_id)

    with patch.object(_apply_mod, "capture", AsyncMock(return_value=True)) as cap:
        await _apply_mod._apply_posthook_verdict_locked(source, verdict)

    cap.assert_awaited_once()
    kwargs = cap.call_args.kwargs
    args = cap.call_args.args
    # key is the first positional arg
    from kara_kutu import i2p_step_key
    # workflow read from missions.workflow (mission 1 has none) → "" component
    expected_key = i2p_step_key("", "7.4a", "coder")
    assert args[0] == expected_key
    assert kwargs["task_id"] == source_id
    assert kwargs["result"] == "the schema output"
    assert kwargs["mission_id"] == 1
    assert 0.0 < kwargs["quality"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 90 python -m pytest packages/general_beckman/tests/test_grade_pass_captures_experience.py -v`
Expected: FAIL — `AttributeError: <module 'general_beckman.apply'> does not have the attribute 'capture'` (the name is not yet imported/used in apply.py).

- [ ] **Step 3: Write minimal implementation**

In `packages/general_beckman/src/general_beckman/apply.py`, inside the `if a.kind == "grade" and a.passed:` block, immediately after `ctx["_pending_posthooks"] = pending` (line 6080) and before `if not pending:` (line 6081), insert:

```python
        # P1 — capture per-step SUCCESS experience (best-effort; a miss must
        # never block or DLQ the source). Key via the single builder on
        # source['agent_type'] (NOT profile.name) so capture and the live
        # reader (coulson.context) share one key. Workflow is read the SAME
        # way the reader reads it (missions.workflow) so both key on the same
        # value.
        try:
            from kara_kutu import capture, i2p_step_key, quality_from_grade
            from src.memory.workflow_exemplars import extract_step_id
            from dabidabi import get_db as _get_db_exp
            _exp_step = extract_step_id(source.get("title", "") or "")
            if _exp_step:
                _exp_wf = ""
                _exp_mid = source.get("mission_id")
                if _exp_mid is not None:
                    try:
                        _edb = await _get_db_exp()
                        async with _edb.execute(
                            "SELECT workflow FROM missions WHERE id=?",
                            (_exp_mid,),
                        ) as _ecur:
                            _erow = await _ecur.fetchone()
                            if _erow:
                                _exp_wf = _erow[0] or ""
                    except Exception:
                        pass
                _exp_key = i2p_step_key(
                    _exp_wf, _exp_step, source.get("agent_type") or "",
                )
                await capture(
                    _exp_key,
                    task_id=a.source_task_id,
                    mission_id=_exp_mid,
                    result=source.get("result") or "",
                    quality=quality_from_grade(a.raw),
                )
        except Exception as _exp_err:
            logger.debug("experience capture (grade pass) skipped",
                         task_id=a.source_task_id, error=str(_exp_err))
```

> **Import note (verified):** `_apply_posthook_verdict_locked` imports `from dabidabi import get_task, update_task, add_task` at `apply.py:5500` — it does NOT import `get_db`, and there is NO module-level `get_db` in apply.py (it is imported locally inside other functions at lines 345/695/1699). The try-block above therefore imports it locally as `from dabidabi import get_db as _get_db_exp` and calls `_edb = await _get_db_exp()`. This is already reflected in the code block above — implement it verbatim.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 90 python -m pytest packages/general_beckman/tests/test_grade_pass_captures_experience.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Regression — grade-pass tests still green**

Run: `timeout 120 python -m pytest tests/test_beckman_posthooks.py -v`
Expected: PASS (all existing grade/summary posthook tests unchanged — the capture is additive + try-wrapped).

- [ ] **Step 6: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/apply.py packages/general_beckman/tests/test_grade_pass_captures_experience.py
rtk git commit -m "feat(beckman): wire kara_kutu.capture on grade-PASS (apply.py:6073, best-effort) (P1)"
```

---

## Task 6: Re-key the live reader (`context.py:1349`) to `task['agent_type']`

The single-source-of-truth fix: the reader must build its lookup key with the same `agent_type` capture used — `task['agent_type']` (line 1241's `agent_type` var), NOT `profile.name`. This closes the shipped key-mismatch trap (spec §0, §5).

**Files:**
- Modify: `packages/coulson/src/coulson/context.py:1349` — change `agent_type=profile.name,` to `agent_type=agent_type,` (the `agent_type` local computed at line 1241 = `task.get("agent_type") or profile.name`). Also update the info log at line 1361 to log the same value.
- Test: `packages/coulson/tests/test_context_exemplar_rekey.py`

- [ ] **Step 1: Write the failing test**

The reader lives inside `build_user_context` (a large function). Rather than drive the whole builder, assert the reader calls `lookup_exemplars` with `agent_type == task['agent_type']` (not `profile.name`) when they differ. Patch `lookup_exemplars` to a recorder and give a task whose `agent_type` differs from `profile.name`.

```python
# packages/coulson/tests/test_context_exemplar_rekey.py
"""P1 — the live exemplar reader keys on task['agent_type'], not profile.name."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch


class _Profile:
    name = "executor"          # deliberately != task['agent_type']
    allowed_tools = []


@pytest.mark.asyncio
async def test_reader_keys_on_task_agent_type(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    from src.infra.db import init_db, get_db
    await init_db()
    # Mission row so the workflow read succeeds ("" workflow).
    db = await get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status, workflow) VALUES (1, 't', 'active', '')"
    )
    await db.commit()

    from coulson.context import build_user_context

    task = {
        "id": 999,
        "title": "[7.4a] db_schema_author",
        "agent_type": "coder",        # capture side
        "mission_id": 1,
        "context": "{}",
    }

    recorder = AsyncMock(return_value=[])
    with patch("src.memory.workflow_exemplars.lookup_exemplars", recorder):
        await build_user_context(_Profile(), task, model_ctx=4096)

    recorder.assert_awaited()
    # Find the exemplar lookup call (there is exactly one).
    call = recorder.call_args
    assert call.kwargs["step_id"] == "7.4a"
    assert call.kwargs["agent_type"] == "coder"        # NOT "executor"
    assert call.kwargs["workflow"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 90 python -m pytest packages/coulson/tests/test_context_exemplar_rekey.py -v`
Expected: FAIL — `AssertionError: assert 'executor' == 'coder'` (reader currently passes `profile.name` = `"executor"`).

- [ ] **Step 3: Write minimal implementation**

In `packages/coulson/src/coulson/context.py`, change line 1349 from:

```python
                    agent_type=profile.name,
```

to:

```python
                    agent_type=agent_type,
```

And change the log at lines 1359-1362 from:

```python
                        logger.info(
                            "Workflow exemplars injected: step=%s agent=%s n=%d",
                            _step_id, profile.name, len(exemplars),
                        )
```

to:

```python
                        logger.info(
                            "Workflow exemplars injected: step=%s agent=%s n=%d",
                            _step_id, agent_type, len(exemplars),
                        )
```

> `agent_type` is the local defined at `context.py:1241` (`agent_type = task.get("agent_type") or profile.name`), which is in scope at 1349. No new import.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 90 python -m pytest packages/coulson/tests/test_context_exemplar_rekey.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
rtk git add packages/coulson/src/coulson/context.py packages/coulson/tests/test_context_exemplar_rekey.py
rtk git commit -m "fix(coulson): exemplar reader keys on task['agent_type'] not profile.name (P1 single-key)"
```

---

## Task 7: Round-trip integration test on the REAL reader path (P1 exit criterion)

Capture a `[X.Y]` success via `kara_kutu.capture`, then assert the LIVE reader (`context.py` exemplar block) returns it for the same step — proving the two sides now share one key. This is P1's exit criterion (spec §8-P1, §9).

**Files:**
- Test: `packages/kara_kutu/tests/test_experience_roundtrip.py`

- [ ] **Step 1: Write the failing test (write BEFORE Task 6 is deployed to confirm it captures the round-trip; here it validates the finished P1)**

```python
# packages/kara_kutu/tests/test_experience_roundtrip.py
"""P1 EXIT CRITERION — a captured [X.Y] success is returned by the REAL reader."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch


class _Profile:
    name = "executor"          # != task['agent_type'] on purpose
    allowed_tools = []


@pytest.mark.asyncio
async def test_capture_then_real_reader_returns_it(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    from src.infra.db import init_db, get_db
    await init_db()

    db = await get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status, workflow) VALUES (1, 't', 'active', '')"
    )
    await db.commit()

    # 1) capture a success under the coder [7.4a] key
    from kara_kutu import capture, i2p_step_key
    key = i2p_step_key("", "7.4a", "coder")
    ok = await capture(key, task_id=123, result="ROUND-TRIP-MARKER output",
                       quality=0.9, mission_id=1)
    assert ok is True

    # 2) the REAL reader path: build_user_context for the same step.
    #    lookup_exemplars is the actual store fn (NOT mocked) — we assert the
    #    captured text lands in the rendered context string.
    from coulson.context import build_user_context
    task = {
        "id": 999,
        "title": "[7.4a] db_schema_author",
        "agent_type": "coder",
        "mission_id": 1,
        "context": "{}",
    }
    context_str, _tools = await build_user_context(_Profile(), task, model_ctx=8192)

    assert "ROUND-TRIP-MARKER output" in context_str, (
        "captured exemplar not surfaced by the real reader — key mismatch"
    )
```

- [ ] **Step 2: Run test to verify it fails (if run before Task 6)**

Run: `timeout 120 python -m pytest packages/kara_kutu/tests/test_experience_roundtrip.py -v`
Expected (before Task 6 lands): FAIL — the reader keys on `profile.name="executor"`, capture keyed on `"coder"`, so the marker is absent. After Tasks 1-6 land: this drives the exit proof.

> If Tasks 1-6 are already committed, this test PASSES on first run — that is the intended P1 exit state. Keep the test regardless (it is the durable round-trip regression guard).

- [ ] **Step 3: (no implementation — Tasks 1-6 already provide the behavior)**

This task adds no production code; it is the integration proof. If it fails after Tasks 1-6, debug with `superpowers:systematic-debugging` — the likely cause is a key-component mismatch (workflow read path or agent_type source differing between capture and reader). Both must resolve to `("", "7.4a", "coder")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 120 python -m pytest packages/kara_kutu/tests/test_experience_roundtrip.py -v`
Expected: PASS (1 passed) — **P1 exit criterion met.**

- [ ] **Step 5: Full P1 suite green**

Run:
```bash
timeout 180 python -m pytest packages/kara_kutu/tests/ packages/general_beckman/tests/test_grade_pass_captures_experience.py packages/coulson/tests/test_context_exemplar_rekey.py -v
```
Expected: PASS (all P1 tests).

- [ ] **Step 6: Commit**

```bash
rtk git add packages/kara_kutu/tests/test_experience_roundtrip.py
rtk git commit -m "test(kara_kutu): P1 exit — captured [X.Y] success returned by the real reader"
```

---

## Self-review checklist (run before handoff)

- **Spec coverage (§8-P1):** capture into kara_kutu (Task 3) ✅; fix the live key (Task 6) ✅; round-trip proof via the REAL reader (Task 7) ✅; quality from booleans default-safe (Task 4) ✅; key via single builder on `task['agent_type']` (Task 1, used in Tasks 5+6) ✅; hook at `apply.py:6073` with `source`/`a.source_task_id`/`a.raw` in scope (Task 5) ✅. **Deferred by design (flagged as gaps):** failure-mode capture / `_maybe_emit_lesson_from_posthook_fail` migration (stub only, Task 3); moving `workflow_exemplars` into kara_kutu (wrapped instead, Task 2).
- **Placeholder scan:** every code step contains real code; no TBD/TODO. The one explicit VERIFY-before-implement note is in Task 5 Step 3 (the `get_db` import in apply.py) — that is a deliberate guard, not a placeholder, because the exact import line must be confirmed at edit time.
- **Type consistency:** `i2p_step_key(workflow, step_id, agent_type)->str`, `Experience(kind,key,text,tool_recipe,quality,task_id,created_at,fix,occurrences,severity,suppressed)`, `recall(key,*,limit)->list[Experience]`, `capture(key,*,task_id,mission_id,result,quality,pattern,fix,severity)->bool`, `quality_from_grade(raw)->float` — used consistently across Tasks 1-7.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-07-kara-kutu-experience-p1.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints.

**Which approach?**
