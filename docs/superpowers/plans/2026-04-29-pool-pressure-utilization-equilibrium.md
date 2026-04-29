# Pool Pressure — Utilization Equilibrium Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-axis pool pressure with intelligent multi-signal fusion. Capacity magnitude, token cost, queue lookahead, and perishability all become first-class signals. Same scalar consumed by Beckman admission and Hoca utilization layer; equilibrium falls out of perishability-conditional dampening.

**Architecture:** Ten independent signals + four modifiers + bucketed worst-wins combination + gated abundance → scalar ∈ [-1, +1]. New tables `model_call_tokens` (per-call telemetry) + `step_token_stats` (rolled up B-table) drive a `(step_id) → (agent, phase) → AGENT_REQUIREMENTS` lookup chain for token estimates. `RateLimits` widens to a `RateLimitMatrix` of axis × time × granularity cells; only populated cells participate in scoring.

**Tech Stack:** Python 3.10, aiosqlite (SQLite WAL), litellm. Packages: `packages/nerd_herd/`, `packages/fatih_hoca/`, `packages/general_beckman/`, `packages/kuleden_donen_var/`, `packages/hallederiz_kadir/`.

**Reference spec:** `docs/superpowers/specs/2026-04-29-pool-pressure-utilization-equilibrium-design.md`

**Reference telemetry:** `docs/research/2026-04-28-token-distribution.md`

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `packages/nerd_herd/src/nerd_herd/signals/__init__.py` | Public re-exports |
| `packages/nerd_herd/src/nerd_herd/signals/s1_remaining.py` | S1 per-axis remaining pressure |
| `packages/nerd_herd/src/nerd_herd/signals/s2_call_burden.py` | S2 per-call token burden |
| `packages/nerd_herd/src/nerd_herd/signals/s3_task_burden.py` | S3 cumulative task burden |
| `packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py` | S4 queue token pressure |
| `packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py` | S5 queue request pressure |
| `packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py` | S6 capable-capacity overlap |
| `packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py` | S7 burn rate extrapolation |
| `packages/nerd_herd/src/nerd_herd/signals/s9_perishability.py` | S9 universal perishability |
| `packages/nerd_herd/src/nerd_herd/signals/s10_failure.py` | S10 failure state |
| `packages/nerd_herd/src/nerd_herd/signals/s11_cost.py` | S11 cost burden |
| `packages/nerd_herd/src/nerd_herd/modifiers.py` | M1 amplifier, M2 perish-conditional dampener, M3 difficulty weights |
| `packages/nerd_herd/src/nerd_herd/breakdown.py` | `PressureBreakdown` dataclass |
| `packages/nerd_herd/src/nerd_herd/combine.py` | Combination logic (bucket worst-wins + gated abundance) |
| `packages/nerd_herd/src/nerd_herd/burn_log.py` | Rolling burn-rate counter (calls + tokens / 5min window) |
| `packages/fatih_hoca/src/fatih_hoca/estimates.py` | `Estimates` dataclass + `estimate_for(task)` lookup chain |
| `packages/fatih_hoca/src/fatih_hoca/step_overrides.py` | Static `STEP_TOKEN_OVERRIDES` table |
| `packages/general_beckman/src/general_beckman/btable_rollup.py` | Hourly rollup `model_call_tokens` → `step_token_stats` |

### Modified files

| Path | Change |
|---|---|
| `src/infra/db.py:55-770` | New tables: `model_call_tokens`, `step_token_stats`. New helper: `record_call_tokens()`. New column: `model_pick_log.outcome` |
| `packages/nerd_herd/src/nerd_herd/types.py` | `RateLimits` → `RateLimitMatrix` rename + axis cells expansion. `QueueProfile` widening. `SystemSnapshot.pressure_for` signature change |
| `packages/nerd_herd/src/nerd_herd/pool_pressure.py` | Becomes thin orchestrator; signal computation extracted to `signals/` |
| `packages/nerd_herd/src/nerd_herd/__init__.py` | Re-exports for new types |
| `packages/fatih_hoca/src/fatih_hoca/scarcity.py` | Deleted; replaced by signals + combine |
| `packages/fatih_hoca/src/fatih_hoca/ranking.py:240+` | Utilization layer call site uses new pressure_for signature; remove inline fit_excess (M2 absorbs it) |
| `packages/fatih_hoca/src/fatih_hoca/requirements.py:60-181` | `AGENT_REQUIREMENTS` recalibrated to telemetry p90; new `AVG_ITERATIONS_BY_AGENT` constant |
| `packages/fatih_hoca/src/fatih_hoca/selector.py` | Propagate `Estimates` through `select()` |
| `packages/general_beckman/src/general_beckman/admission.py` | No change to threshold; admission caller in `__init__.py` updates pressure_for signature |
| `packages/general_beckman/src/general_beckman/__init__.py:108-200` | Admission gate uses new signature; logs admission_decision with breakdown |
| `packages/general_beckman/src/general_beckman/queue_profile_push.py` | Dep-resolution + by_difficulty + by_capability + projections; cached completed_ids |
| `packages/kuleden_donen_var/src/kuleden_donen_var/nerd_herd_adapter.py` | Drop "rpd-only" restriction; copy all populated cells |
| `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py:461-500` | Call `record_call_tokens()` after every successful call |

### Test files (one per signal/module + integration)

`packages/nerd_herd/tests/signals/test_s{1..11}.py`, `packages/nerd_herd/tests/test_modifiers.py`, `packages/nerd_herd/tests/test_combine.py`, `packages/nerd_herd/tests/test_breakdown.py`, `packages/nerd_herd/tests/test_burn_log.py`, `packages/fatih_hoca/tests/test_estimates.py`, `packages/general_beckman/tests/test_btable_rollup.py`, `packages/general_beckman/tests/test_queue_profile_dep_resolution.py`, `tests/test_record_call_tokens.py`.

---

## Phase 1 — Schema + Telemetry

### Task 1: Add `model_call_tokens` table

**Files:**
- Modify: `src/infra/db.py:55-770` (init_db function, schema creation block)
- Test: `tests/test_record_call_tokens.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_record_call_tokens.py
import os
import tempfile
from pathlib import Path

import pytest
import aiosqlite


@pytest.mark.asyncio
async def test_model_call_tokens_table_created(monkeypatch):
    """init_db must create model_call_tokens with expected schema."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        monkeypatch.setenv("DB_PATH", str(db_path))
        # Reset module-level singleton
        import src.infra.db as db_mod
        db_mod._db = None
        await db_mod.init_db()
        async with aiosqlite.connect(str(db_path)) as conn:
            cur = await conn.execute("PRAGMA table_info(model_call_tokens)")
            cols = {row[1]: row[2] for row in await cur.fetchall()}
        expected = {
            "id": "INTEGER",
            "timestamp": "TIMESTAMP",
            "task_id": "INTEGER",
            "agent_type": "TEXT",
            "workflow_step_id": "TEXT",
            "workflow_phase": "TEXT",
            "call_category": "TEXT",
            "model": "TEXT",
            "provider": "TEXT",
            "is_streaming": "INTEGER",
            "prompt_tokens": "INTEGER",
            "completion_tokens": "INTEGER",
            "reasoning_tokens": "INTEGER",
            "total_tokens": "INTEGER",
            "duration_ms": "INTEGER",
            "iteration_n": "INTEGER",
            "success": "INTEGER",
        }
        for k, v in expected.items():
            assert k in cols, f"missing column {k}"
            assert cols[k] == v, f"column {k}: expected {v}, got {cols[k]}"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
timeout 30 pytest tests/test_record_call_tokens.py::test_model_call_tokens_table_created -v
```
Expected: FAIL with `OperationalError: no such table` or empty cols.

- [ ] **Step 3: Add table to `init_db()`**

In `src/infra/db.py`, locate the `init_db()` function (around line 55) and add this CREATE TABLE block after the `model_stats` block (after line 256):

```python
    await db.execute("""
        CREATE TABLE IF NOT EXISTS model_call_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL DEFAULT (datetime('now')),
            task_id INTEGER,
            agent_type TEXT,
            workflow_step_id TEXT,
            workflow_phase TEXT,
            call_category TEXT,
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            is_streaming INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            reasoning_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER NOT NULL,
            duration_ms INTEGER,
            iteration_n INTEGER,
            success INTEGER NOT NULL
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mct_task ON model_call_tokens(task_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mct_step ON model_call_tokens(agent_type, workflow_step_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mct_recent ON model_call_tokens(timestamp)")
```

- [ ] **Step 4: Run test, verify it passes**

```bash
timeout 30 pytest tests/test_record_call_tokens.py::test_model_call_tokens_table_created -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infra/db.py tests/test_record_call_tokens.py
git commit -m "feat(db): add model_call_tokens table for per-call telemetry"
```

---

### Task 2: Add `step_token_stats` table + `model_pick_log.outcome` column

**Files:**
- Modify: `src/infra/db.py:55-770`
- Test: `tests/test_record_call_tokens.py`

- [ ] **Step 1: Append failing tests**

```python
# tests/test_record_call_tokens.py — append
@pytest.mark.asyncio
async def test_step_token_stats_table_created(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        monkeypatch.setenv("DB_PATH", str(db_path))
        import src.infra.db as db_mod
        db_mod._db = None
        await db_mod.init_db()
        async with aiosqlite.connect(str(db_path)) as conn:
            cur = await conn.execute("PRAGMA table_info(step_token_stats)")
            cols = {row[1]: row[2] for row in await cur.fetchall()}
        expected = {"agent_type", "workflow_step_id", "workflow_phase",
                    "samples_n", "in_p50", "in_p90", "in_p99",
                    "out_p50", "out_p90", "out_p99",
                    "iters_p50", "iters_p90", "iters_p99", "updated_at"}
        assert expected.issubset(set(cols.keys()))


@pytest.mark.asyncio
async def test_model_pick_log_has_outcome_column(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        monkeypatch.setenv("DB_PATH", str(db_path))
        import src.infra.db as db_mod
        db_mod._db = None
        await db_mod.init_db()
        async with aiosqlite.connect(str(db_path)) as conn:
            cur = await conn.execute("PRAGMA table_info(model_pick_log)")
            cols = {row[1] for row in await cur.fetchall()}
        assert "outcome" in cols
```

- [ ] **Step 2: Run, verify both fail**

```bash
timeout 30 pytest tests/test_record_call_tokens.py -v
```
Expected: 2 FAILs (new tests).

- [ ] **Step 3: Add `step_token_stats` table to `init_db()`**

In `src/infra/db.py::init_db()`, after the `model_call_tokens` block from Task 1:

```python
    await db.execute("""
        CREATE TABLE IF NOT EXISTS step_token_stats (
            agent_type TEXT NOT NULL,
            workflow_step_id TEXT NOT NULL,
            workflow_phase TEXT NOT NULL,
            samples_n INTEGER NOT NULL,
            in_p50 INTEGER, in_p90 INTEGER, in_p99 INTEGER,
            out_p50 INTEGER, out_p90 INTEGER, out_p99 INTEGER,
            iters_p50 REAL, iters_p90 REAL, iters_p99 REAL,
            updated_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (agent_type, workflow_step_id, workflow_phase)
        )
    """)
```

- [ ] **Step 4: Add `outcome` column to `model_pick_log`**

In `src/infra/db.py::init_db()`, find the `model_pick_log` table (search for `CREATE TABLE IF NOT EXISTS model_pick_log`) and add:

```python
    # outcome column added 2026-04-29 for pressure calibration loop
    try:
        await db.execute(
            "ALTER TABLE model_pick_log ADD COLUMN outcome TEXT"
        )
    except Exception:
        pass  # column already exists
```

Place this **after** the `CREATE TABLE` for `model_pick_log` and any existing migrations on it. Mirror the pattern of the existing `provider` column ALTER if present.

- [ ] **Step 5: Run, verify both pass**

```bash
timeout 30 pytest tests/test_record_call_tokens.py -v
```
Expected: PASS x4.

- [ ] **Step 6: Commit**

```bash
git add src/infra/db.py tests/test_record_call_tokens.py
git commit -m "feat(db): add step_token_stats table and model_pick_log.outcome column"
```

---

### Task 3: `record_call_tokens` helper + caller.py instrumentation

**Files:**
- Modify: `src/infra/db.py` (new helper near `record_model_call` at line 2044)
- Modify: `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py:461-500`
- Test: `tests/test_record_call_tokens.py` (extend)

- [ ] **Step 1: Append failing test for `record_call_tokens`**

```python
# tests/test_record_call_tokens.py — append
@pytest.mark.asyncio
async def test_record_call_tokens_inserts_row(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        monkeypatch.setenv("DB_PATH", str(db_path))
        import src.infra.db as db_mod
        db_mod._db = None
        await db_mod.init_db()
        await db_mod.record_call_tokens(
            task_id=42,
            agent_type="analyst",
            workflow_step_id="3.5",
            workflow_phase="phase_3",
            call_category="main_work",
            model="gpt-4",
            provider="openai",
            is_streaming=False,
            prompt_tokens=1000,
            completion_tokens=500,
            reasoning_tokens=0,
            total_tokens=1500,
            duration_ms=2500,
            iteration_n=1,
            success=True,
        )
        async with aiosqlite.connect(str(db_path)) as conn:
            cur = await conn.execute(
                "SELECT task_id, total_tokens FROM model_call_tokens WHERE task_id=42"
            )
            row = await cur.fetchone()
        assert row == (42, 1500)
```

- [ ] **Step 2: Run, verify FAIL**

```bash
timeout 30 pytest tests/test_record_call_tokens.py::test_record_call_tokens_inserts_row -v
```
Expected: FAIL with `AttributeError: module ... has no attribute 'record_call_tokens'`.

- [ ] **Step 3: Add helper to `src/infra/db.py`**

Insert after the `record_model_call` function (around line 2120):

```python
async def record_call_tokens(
    *,
    task_id: int | None,
    agent_type: str | None,
    workflow_step_id: str | None,
    workflow_phase: str | None,
    call_category: str,
    model: str,
    provider: str,
    is_streaming: bool,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    total_tokens: int,
    duration_ms: int,
    iteration_n: int,
    success: bool,
) -> None:
    """Persist per-call token usage. Single INSERT, no upsert.

    Feeds step_token_stats rollup (Beckman cron) and offline calibration.
    """
    db = await get_db()
    await db.execute(
        """INSERT INTO model_call_tokens
           (task_id, agent_type, workflow_step_id, workflow_phase, call_category,
            model, provider, is_streaming, prompt_tokens, completion_tokens,
            reasoning_tokens, total_tokens, duration_ms, iteration_n, success)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (task_id, agent_type, workflow_step_id, workflow_phase, call_category,
         model, provider, int(is_streaming), prompt_tokens, completion_tokens,
         reasoning_tokens, total_tokens, duration_ms, iteration_n, int(success)),
    )
    await db.commit()
```

- [ ] **Step 4: Run helper test, verify PASS**

```bash
timeout 30 pytest tests/test_record_call_tokens.py::test_record_call_tokens_inserts_row -v
```
Expected: PASS.

- [ ] **Step 5: Wire into `caller.py`**

In `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py`, locate the post-call block (around line 461-500). After `_record_metrics(...)` (line 500), add:

```python
        # Per-call token telemetry (feeds B-table rollup + calibration)
        try:
            from src.infra.db import record_call_tokens
            await record_call_tokens(
                task_id=getattr(task, "id", None) if task else None,
                agent_type=getattr(task, "agent_type", None) if task else None,
                workflow_step_id=(task.context or {}).get("workflow_step_id") if task and getattr(task, "context", None) else None,
                workflow_phase=(task.context or {}).get("workflow_phase") if task and getattr(task, "context", None) else None,
                call_category=getattr(reqs, "call_category", "main_work"),
                model=model.name,
                provider=model.provider,
                is_streaming=stream_used,
                prompt_tokens=(raw_result.usage.prompt_tokens or 0) if not stream_used and getattr(raw_result, "usage", None) else 0,
                completion_tokens=(raw_result.usage.completion_tokens or 0) if not stream_used and getattr(raw_result, "usage", None) else 0,
                reasoning_tokens=getattr(raw_result.usage, "reasoning_tokens", 0) if not stream_used and getattr(raw_result, "usage", None) else 0,
                total_tokens=total_tokens,
                duration_ms=int(call_latency * 1000),
                iteration_n=iteration_index if "iteration_index" in dir() else 0,
                success=True,
            )
        except Exception:
            pass  # telemetry best-effort
```

`iteration_index` is a new parameter on the call path. Find the call signature `async def call_model(...)` around line 257; add `iteration_n: int = 0` to its parameters and propagate from `LLMDispatcher.request()` and `BaseAgent.run` ReAct loop. (Specific signature changes laid out in Task 17.)

For now, leave `iteration_index = 0` as fallback if the variable isn't defined in caller scope (using `if "iteration_index" in dir() else 0`).

- [ ] **Step 6: Run hallederiz tests for regression**

```bash
timeout 60 pytest packages/hallederiz_kadir/tests/ -v
```
Expected: existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/infra/db.py packages/hallederiz_kadir/src/hallederiz_kadir/caller.py tests/test_record_call_tokens.py
git commit -m "feat(telemetry): record_call_tokens helper + caller.py instrumentation"
```

---

## Phase 2 — Estimates Infrastructure

### Task 4: `Estimates` dataclass + `STEP_TOKEN_OVERRIDES` static table

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/estimates.py`
- Create: `packages/fatih_hoca/src/fatih_hoca/step_overrides.py`
- Test: `packages/fatih_hoca/tests/test_estimates.py`

- [ ] **Step 1: Write failing test for `Estimates` dataclass**

```python
# packages/fatih_hoca/tests/test_estimates.py
from fatih_hoca.estimates import Estimates


def test_estimates_total_tokens_computed():
    e = Estimates(in_tokens=1000, out_tokens=2000, iterations=5)
    assert e.total_tokens == (1000 + 2000) * 5


def test_estimates_per_call_tokens():
    e = Estimates(in_tokens=1000, out_tokens=2000, iterations=5)
    assert e.per_call_tokens == 3000


def test_step_token_overrides_known_step():
    from fatih_hoca.step_overrides import STEP_TOKEN_OVERRIDES
    e = STEP_TOKEN_OVERRIDES["4.5b"]
    assert e.out_tokens >= 100_000  # openapi_spec is heavy
```

- [ ] **Step 2: Run, verify FAIL**

```bash
timeout 30 pytest packages/fatih_hoca/tests/test_estimates.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `estimates.py`**

```python
# packages/fatih_hoca/src/fatih_hoca/estimates.py
"""Token estimates per task.

Lookup chain: B-table (learned, step_token_stats) → A (STEP_TOKEN_OVERRIDES)
→ AGENT_REQUIREMENTS default. See `estimate_for(task)` in this module.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Estimates:
    in_tokens: int
    out_tokens: int
    iterations: int

    @property
    def per_call_tokens(self) -> int:
        return self.in_tokens + self.out_tokens

    @property
    def total_tokens(self) -> int:
        return self.per_call_tokens * self.iterations
```

- [ ] **Step 4: Create `step_overrides.py`**

```python
# packages/fatih_hoca/src/fatih_hoca/step_overrides.py
"""Static curated step-level token overrides.

Hand-seeded from 2026-04-28 telemetry sweep. Only known-heavy steps get
an entry. Entries override AGENT_REQUIREMENTS defaults until the learned
B-table (step_token_stats) has ≥ MIN_SAMPLES for that step.

Source: docs/research/2026-04-28-token-distribution.md §6 outliers.
"""
from __future__ import annotations

from fatih_hoca.estimates import Estimates


STEP_TOKEN_OVERRIDES: dict[str, Estimates] = {
    # i2p_v3 — known-heavy artifact-emit steps
    "4.5b":   Estimates(in_tokens=10_000, out_tokens=100_000, iterations=12),  # openapi_spec
    "5.4b":   Estimates(in_tokens=6_000,  out_tokens=92_000,  iterations=8),   # forms_and_states
    "3.5":    Estimates(in_tokens=10_000, out_tokens=58_000,  iterations=24),  # integration_requirements
    "4.15a1": Estimates(in_tokens=20_000, out_tokens=44_000,  iterations=6),   # backend_core_design
    "5.11b":  Estimates(in_tokens=28_000, out_tokens=43_000,  iterations=8),   # design_handoff_document
    "3.6":    Estimates(in_tokens=11_000, out_tokens=27_000,  iterations=8),   # platform_and_accessibility_requirements
    "4.5a":   Estimates(in_tokens=10_000, out_tokens=25_000,  iterations=8),   # api_resource_model
    "5.11a":  Estimates(in_tokens=13_000, out_tokens=25_000,  iterations=8),   # design_system_handoff
    "5.7":    Estimates(in_tokens=5_000,  out_tokens=23_000,  iterations=8),   # component_specs
    "3.7":    Estimates(in_tokens=10_000, out_tokens=23_000,  iterations=8),   # business_rules_extraction
}
```

- [ ] **Step 5: Run tests, verify PASS**

```bash
timeout 30 pytest packages/fatih_hoca/tests/test_estimates.py -v
```
Expected: PASS x3.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/estimates.py packages/fatih_hoca/src/fatih_hoca/step_overrides.py packages/fatih_hoca/tests/test_estimates.py
git commit -m "feat(fatih_hoca): Estimates dataclass + STEP_TOKEN_OVERRIDES"
```

---

### Task 5: `AGENT_REQUIREMENTS` recalibration + `AVG_ITERATIONS_BY_AGENT`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/requirements.py:60-181`
- Test: `packages/fatih_hoca/tests/test_estimates.py` (extend)

- [ ] **Step 1: Append failing test**

```python
# packages/fatih_hoca/tests/test_estimates.py — append
def test_avg_iterations_by_agent_seeded_from_telemetry():
    from fatih_hoca.requirements import AVG_ITERATIONS_BY_AGENT
    # 2026-04-28 telemetry: analyst avg 7.1, architect 11.8, researcher 23.4
    assert 6 <= AVG_ITERATIONS_BY_AGENT["analyst"] <= 9
    assert 10 <= AVG_ITERATIONS_BY_AGENT["architect"] <= 14
    assert 20 <= AVG_ITERATIONS_BY_AGENT["researcher"] <= 28


def test_agent_requirements_calibrated_to_p90():
    from fatih_hoca.requirements import AGENT_REQUIREMENTS
    # Telemetry showed analyst p90 = 25k tokens; old default 3k under-reserves 8x
    analyst = AGENT_REQUIREMENTS["analyst"]
    assert analyst.estimated_output_tokens >= 15_000  # at least p75
```

- [ ] **Step 2: Run, verify FAIL**

```bash
timeout 30 pytest packages/fatih_hoca/tests/test_estimates.py::test_avg_iterations_by_agent_seeded_from_telemetry packages/fatih_hoca/tests/test_estimates.py::test_agent_requirements_calibrated_to_p90 -v
```
Expected: FAIL.

- [ ] **Step 3: Add `AVG_ITERATIONS_BY_AGENT` constant**

In `packages/fatih_hoca/src/fatih_hoca/requirements.py`, after the `AGENT_REQUIREMENTS` dict (line 181):

```python
# ─── Per-agent iteration estimates (cold-start values from 2026-04-28 telemetry) ─
# Refined automatically by step_token_stats once samples accumulate.
AVG_ITERATIONS_BY_AGENT: dict[str, int] = {
    # Telemetry-backed (i2p ReAct path)
    "analyst":          8,
    "architect":        12,
    "writer":           6,
    "researcher":       24,
    "reviewer":         12,
    # Cold-start by analogy (will refine when telemetry catches up)
    "planner":          8,
    "coder":            6,
    "implementer":      6,
    "fixer":            5,
    "test_generator":   5,
    "executor":         4,
    "summarizer":       3,
    "visual_reviewer":  4,
    # Shopping
    "shopping_advisor":     4,
    "product_researcher":   5,
    "deal_analyst":         4,
    "shopping_clarifier":   2,
    # Default
    "assistant":        4,
    "classifier":       1,
    "grader":           2,
}
```

- [ ] **Step 4: Recalibrate `AGENT_REQUIREMENTS` to telemetry p90**

In `packages/fatih_hoca/src/fatih_hoca/requirements.py:160-181`, replace the existing `AGENT_REQUIREMENTS` dict entries with telemetry-calibrated p90 values:

```python
AGENT_REQUIREMENTS: dict[str, ModelRequirements] = {
    # ── Difficult / sensitive — calibrated to telemetry p90 ──
    "planner":        ModelRequirements(task="planner",        difficulty=5, estimated_input_tokens=8_000,  estimated_output_tokens=20_000, prefer_quality=True),
    "architect":      ModelRequirements(task="architect",      difficulty=5, estimated_input_tokens=8_000,  estimated_output_tokens=20_000, prefer_quality=True),
    "coder":          ModelRequirements(task="coder",          difficulty=6, estimated_input_tokens=8_000,  estimated_output_tokens=15_000, needs_function_calling=True),
    "fixer":          ModelRequirements(task="fixer",          difficulty=6, estimated_input_tokens=8_000,  estimated_output_tokens=12_000, needs_function_calling=True),
    "reviewer":       ModelRequirements(task="reviewer",       difficulty=6, estimated_input_tokens=10_000, estimated_output_tokens=8_000),
    "analyst":        ModelRequirements(task="analyst",        difficulty=6, estimated_input_tokens=8_000,  estimated_output_tokens=25_000, needs_function_calling=True),
    # ── Moderate ──
    "implementer":    ModelRequirements(task="implementer",    difficulty=5, estimated_input_tokens=8_000,  estimated_output_tokens=15_000, needs_function_calling=True),
    "test_generator": ModelRequirements(task="test_generator", difficulty=5, estimated_input_tokens=8_000,  estimated_output_tokens=10_000, needs_function_calling=True),
    "writer":         ModelRequirements(task="writer",         difficulty=5, estimated_input_tokens=8_000,  estimated_output_tokens=15_000),
    "visual_reviewer": ModelRequirements(task="visual_reviewer", difficulty=5, estimated_input_tokens=4_000, estimated_output_tokens=4_000, needs_vision=True),
    # ── Token-heavy / conversational ──
    "researcher":     ModelRequirements(task="researcher",     difficulty=4, estimated_input_tokens=8_000,  estimated_output_tokens=5_000, needs_function_calling=True, prefer_local=True, prefer_speed=True),
    "assistant":      ModelRequirements(task="assistant",      difficulty=3, estimated_input_tokens=4_000,  estimated_output_tokens=3_000, prefer_local=True, prefer_speed=True),
    "executor":       ModelRequirements(task="executor",       difficulty=3, estimated_input_tokens=4_000,  estimated_output_tokens=2_000, needs_function_calling=True, prefer_speed=True, prefer_local=True),
    "summarizer":     ModelRequirements(task="summarizer",     difficulty=4, estimated_input_tokens=4_000,  estimated_output_tokens=3_000, prefer_speed=True, prefer_local=True),
    # ── Shopping ──
    "shopping_advisor":    ModelRequirements(task="shopping_advisor",    difficulty=5, estimated_input_tokens=4_000, estimated_output_tokens=4_000, needs_function_calling=True, prefer_local=True, prefer_speed=True),
    "product_researcher":  ModelRequirements(task="shopping_advisor",    difficulty=4, estimated_input_tokens=4_000, estimated_output_tokens=3_000, needs_function_calling=True, prefer_local=True, prefer_speed=True),
    "deal_analyst":        ModelRequirements(task="shopping_advisor",    difficulty=5, estimated_input_tokens=4_000, estimated_output_tokens=3_000, needs_function_calling=True, prefer_local=True),
    "shopping_clarifier":  ModelRequirements(task="shopping_advisor",    difficulty=3, estimated_input_tokens=2_000, estimated_output_tokens=1_500, prefer_local=True, prefer_speed=True),
}
```

- [ ] **Step 5: Run tests, verify PASS**

```bash
timeout 30 pytest packages/fatih_hoca/tests/test_estimates.py -v
```
Expected: PASS.

- [ ] **Step 6: Run wider regression on fatih_hoca**

```bash
timeout 120 pytest packages/fatih_hoca/tests/ -v
```
Expected: existing tests pass (some may need re-baselining if they hard-code old `estimated_output_tokens` values — fix inline).

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/requirements.py packages/fatih_hoca/tests/test_estimates.py
git commit -m "feat(fatih_hoca): recalibrate AGENT_REQUIREMENTS to telemetry p90, add AVG_ITERATIONS_BY_AGENT"
```

---

### Task 6: `estimate_for(task)` lookup chain

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/estimates.py`
- Test: `packages/fatih_hoca/tests/test_estimates.py` (extend)

- [ ] **Step 1: Append failing tests**

```python
# packages/fatih_hoca/tests/test_estimates.py — append
import asyncio


class FakeTask:
    def __init__(self, agent_type, step_id=None, phase=None):
        self.agent_type = agent_type
        self.context = {}
        if step_id:
            self.context["workflow_step_id"] = step_id
        if phase:
            self.context["workflow_phase"] = phase


def test_estimate_for_uses_static_overrides_when_step_known():
    from fatih_hoca.estimates import estimate_for
    task = FakeTask("architect", step_id="4.5b")
    e = estimate_for(task, btable={})
    assert e.out_tokens >= 100_000


def test_estimate_for_falls_back_to_agent_requirements():
    from fatih_hoca.estimates import estimate_for
    task = FakeTask("analyst")
    e = estimate_for(task, btable={})
    assert e.out_tokens >= 15_000


def test_estimate_for_uses_btable_when_samples_sufficient():
    from fatih_hoca.estimates import estimate_for, Estimates
    task = FakeTask("analyst", step_id="2.6", phase="phase_2")
    btable = {
        ("analyst", "2.6", "phase_2"): {
            "samples_n": 10,
            "in_p90": 5000, "out_p90": 4000, "iters_p90": 7,
        }
    }
    e = estimate_for(task, btable=btable)
    assert e.in_tokens == 5000
    assert e.out_tokens == 4000
    assert e.iterations == 7


def test_estimate_for_skips_btable_when_samples_below_threshold():
    from fatih_hoca.estimates import estimate_for
    task = FakeTask("analyst", step_id="4.5b", phase="phase_4")
    btable = {
        ("analyst", "4.5b", "phase_4"): {
            "samples_n": 2,  # below MIN_SAMPLES=5
            "in_p90": 100, "out_p90": 100, "iters_p90": 1,
        }
    }
    e = estimate_for(task, btable=btable)
    assert e.out_tokens >= 100_000  # falls through to STEP_TOKEN_OVERRIDES
```

- [ ] **Step 2: Run, verify FAIL**

```bash
timeout 30 pytest packages/fatih_hoca/tests/test_estimates.py -v
```
Expected: 4 FAILs (`AttributeError: ... has no attribute 'estimate_for'`).

- [ ] **Step 3: Add `estimate_for` to `estimates.py`**

Append to `packages/fatih_hoca/src/fatih_hoca/estimates.py`:

```python
from typing import Any

MIN_SAMPLES = 5
THINKING_OUT_SCALE = 2.0  # cloud thinking models: char-derived telemetry under-reports


def _btable_lookup(btable: dict, agent_type: str, step_id: str, phase: str) -> dict | None:
    return btable.get((agent_type, step_id or "", phase or ""))


def estimate_for(task: Any, *, btable: dict, model_is_thinking: bool = False) -> Estimates:
    """Token estimate for a task via the lookup chain.

    Order:
      1. step_token_stats (B-table) when samples_n >= MIN_SAMPLES
      2. STEP_TOKEN_OVERRIDES (curated static A-table)
      3. AGENT_REQUIREMENTS default + AVG_ITERATIONS_BY_AGENT

    Args:
        task: object with .agent_type and .context (dict-like with
              workflow_step_id / workflow_phase keys, optional).
        btable: dict keyed by (agent_type, step_id, phase) returning a row
                with samples_n, in_p90, out_p90, iters_p90.
        model_is_thinking: when True, scale out_tokens up to compensate for
                under-reporting from char-based telemetry.
    """
    from fatih_hoca.requirements import AGENT_REQUIREMENTS, AVG_ITERATIONS_BY_AGENT
    from fatih_hoca.step_overrides import STEP_TOKEN_OVERRIDES

    agent_type = getattr(task, "agent_type", "") or "assistant"
    ctx = getattr(task, "context", None) or {}
    step_id = ctx.get("workflow_step_id") if isinstance(ctx, dict) else None
    phase = ctx.get("workflow_phase") if isinstance(ctx, dict) else None

    # Level 1 — learned
    row = _btable_lookup(btable, agent_type, step_id, phase) if step_id else None
    if row and (row.get("samples_n") or 0) >= MIN_SAMPLES:
        out_p90 = int(row.get("out_p90", 0) or 0)
        if model_is_thinking:
            out_p90 = int(out_p90 * THINKING_OUT_SCALE)
        return Estimates(
            in_tokens=int(row.get("in_p90", 0) or 0),
            out_tokens=out_p90,
            iterations=int(round(float(row.get("iters_p90", 1) or 1))),
        )

    # Level 2 — static overrides
    if step_id and step_id in STEP_TOKEN_OVERRIDES:
        return STEP_TOKEN_OVERRIDES[step_id]

    # Level 3 — AGENT_REQUIREMENTS + AVG_ITERATIONS
    reqs = AGENT_REQUIREMENTS.get(agent_type) or AGENT_REQUIREMENTS["assistant"]
    return Estimates(
        in_tokens=reqs.estimated_input_tokens,
        out_tokens=reqs.estimated_output_tokens,
        iterations=AVG_ITERATIONS_BY_AGENT.get(agent_type, 6),
    )
```

- [ ] **Step 4: Run tests, verify PASS**

```bash
timeout 30 pytest packages/fatih_hoca/tests/test_estimates.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/estimates.py packages/fatih_hoca/tests/test_estimates.py
git commit -m "feat(fatih_hoca): estimate_for lookup chain (B → A → AGENT_REQUIREMENTS)"
```

---

## Phase 3 — Type System Updates

### Task 7: Rename `RateLimits` → `RateLimitMatrix` + add axis cells

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py:80-110`
- Modify: `packages/nerd_herd/src/nerd_herd/__init__.py` (re-export)
- Modify: every consumer that imports `RateLimits` (use grep — `packages/`, `src/`, `tests/`)
- Test: `packages/nerd_herd/tests/test_types_in_flight.py` (update existing) + new `packages/nerd_herd/tests/test_rate_limit_matrix.py`

- [ ] **Step 1: Find all references**

```bash
grep -rn "RateLimits\b" packages/ src/ tests/ | head -40
```
Note results — every occurrence must be renamed.

- [ ] **Step 2: Write failing test for new cells**

```python
# packages/nerd_herd/tests/test_rate_limit_matrix.py
from nerd_herd.types import RateLimit, RateLimitMatrix


def test_matrix_has_all_axes():
    m = RateLimitMatrix()
    # Request axes
    for axis in ("rpm", "rph", "rpd", "rpw", "rpmonth"):
        assert isinstance(getattr(m, axis), RateLimit)
    # Token axes
    for axis in ("tpm", "tph", "tpd", "tpw", "tpmonth", "itpm", "itpd", "otpm", "otpd"):
        assert isinstance(getattr(m, axis), RateLimit)
    # Cost axes
    for axis in ("cpd", "cpmonth"):
        assert isinstance(getattr(m, axis), RateLimit)


def test_populated_cells_iterator_returns_only_filled():
    m = RateLimitMatrix(
        rpm=RateLimit(limit=30),
        tpm=RateLimit(limit=6000),
        # rpd left empty
    )
    populated = dict(m.populated_cells())
    assert "rpm" in populated
    assert "tpm" in populated
    assert "rpd" not in populated


def test_token_cells_filters_correctly():
    m = RateLimitMatrix(
        rpm=RateLimit(limit=30),
        tpm=RateLimit(limit=6000),
        tpd=RateLimit(limit=1_000_000),
    )
    token_axes = {n for n, _ in m.token_cells()}
    assert token_axes == {"tpm", "tpd"}


def test_request_cells_filters_correctly():
    m = RateLimitMatrix(
        rpm=RateLimit(limit=30),
        tpm=RateLimit(limit=6000),
        rpd=RateLimit(limit=14_400),
    )
    req_axes = {n for n, _ in m.request_cells()}
    assert req_axes == {"rpm", "rpd"}
```

- [ ] **Step 3: Run, verify FAIL**

```bash
timeout 30 pytest packages/nerd_herd/tests/test_rate_limit_matrix.py -v
```

- [ ] **Step 4: Replace `RateLimits` with `RateLimitMatrix` in `types.py`**

In `packages/nerd_herd/src/nerd_herd/types.py:80-90`, replace:

```python
@dataclass
class RateLimit:
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None
    in_flight: int = 0


@dataclass
class RateLimitMatrix:
    # Request-axis cells
    rpm: RateLimit = field(default_factory=RateLimit)
    rph: RateLimit = field(default_factory=RateLimit)
    rpd: RateLimit = field(default_factory=RateLimit)
    rpw: RateLimit = field(default_factory=RateLimit)
    rpmonth: RateLimit = field(default_factory=RateLimit)

    # Token-axis cells (total)
    tpm: RateLimit = field(default_factory=RateLimit)
    tph: RateLimit = field(default_factory=RateLimit)
    tpd: RateLimit = field(default_factory=RateLimit)
    tpw: RateLimit = field(default_factory=RateLimit)
    tpmonth: RateLimit = field(default_factory=RateLimit)

    # Token-axis cells (split)
    itpm: RateLimit = field(default_factory=RateLimit)
    itpd: RateLimit = field(default_factory=RateLimit)
    otpm: RateLimit = field(default_factory=RateLimit)
    otpd: RateLimit = field(default_factory=RateLimit)

    # Cost-axis cells
    cpd: RateLimit = field(default_factory=RateLimit)
    cpmonth: RateLimit = field(default_factory=RateLimit)

    def populated_cells(self):
        for name, rl in vars(self).items():
            if isinstance(rl, RateLimit) and rl.limit is not None and rl.limit > 0:
                yield name, rl

    def token_cells(self):
        for name, rl in self.populated_cells():
            if name.startswith(("tp", "itp", "otp")):
                yield name, rl

    def request_cells(self):
        for name, rl in self.populated_cells():
            if name.startswith("rp"):
                yield name, rl

    def cost_cells(self):
        for name, rl in self.populated_cells():
            if name.startswith("cp"):
                yield name, rl
```

Update `CloudModelState` and `CloudProviderState` to use `RateLimitMatrix`:

```python
@dataclass
class CloudModelState:
    model_id: str = ""
    utilization_pct: float = 0.0
    limits: RateLimitMatrix = field(default_factory=RateLimitMatrix)
    pool_pressure: PoolPressure | None = None


@dataclass
class CloudProviderState:
    provider: str = ""
    utilization_pct: float = 0.0
    consecutive_failures: int = 0
    last_failure_at: int | None = None
    limits: RateLimitMatrix = field(default_factory=RateLimitMatrix)
    models: dict[str, CloudModelState] = field(default_factory=dict)
```

- [ ] **Step 5: Update consumers**

Run sed-style replace across all hit files from Step 1. Typical files: `packages/kuleden_donen_var/src/kuleden_donen_var/nerd_herd_adapter.py`, `packages/general_beckman/`, tests in `packages/nerd_herd/tests/`. Replace `RateLimits` → `RateLimitMatrix` everywhere.

- [ ] **Step 6: Update `__init__.py` re-exports**

In `packages/nerd_herd/src/nerd_herd/__init__.py`:

```python
from nerd_herd.types import (
    RateLimit,
    RateLimitMatrix,
    CloudModelState,
    CloudProviderState,
    LocalModelState,
    QueueProfile,
    SystemSnapshot,
    InFlightCall,
)
```

- [ ] **Step 7: Run all nerd_herd + adapter tests**

```bash
timeout 60 pytest packages/nerd_herd/ packages/kuleden_donen_var/ -v
```
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add packages/nerd_herd/ packages/kuleden_donen_var/ tests/
git commit -m "refactor(nerd_herd): RateLimits → RateLimitMatrix with axis cell expansion"
```

---

### Task 8: KDV adapter — copy populated cells

**Files:**
- Modify: `packages/kuleden_donen_var/src/kuleden_donen_var/nerd_herd_adapter.py`
- Modify: `packages/kuleden_donen_var/src/kuleden_donen_var/rate_limiter.py` (add cell accessors if needed)
- Test: `packages/kuleden_donen_var/tests/test_nerd_herd_adapter.py` (extend)

- [ ] **Step 1: Append failing test**

```python
# packages/kuleden_donen_var/tests/test_nerd_herd_adapter.py — append
def test_adapter_copies_tpm_cell_when_present():
    from kuleden_donen_var.kdv import KuledenDonenVar
    from kuleden_donen_var.nerd_herd_adapter import build_cloud_provider_state
    kdv = KuledenDonenVar()
    kdv.register_model("groq/llama", "groq", rpm=30, tpm=6000, rpd=14_400)
    state = build_cloud_provider_state(kdv, "groq")
    m = state.models["groq/llama"]
    assert m.limits.tpm.limit == 6000
    assert m.limits.rpm.limit == 30
    assert m.limits.rpd.limit == 14_400
```

- [ ] **Step 2: Run, verify FAIL**

(KDV's `register_model` may need rpm/tpm support; ensure tests target current API.)

- [ ] **Step 3: Update adapter**

Replace `_rl` and the cell-building logic in `packages/kuleden_donen_var/src/kuleden_donen_var/nerd_herd_adapter.py` with:

```python
def _rl(state, axis: str):
    """Build a RateLimit for axis from KDV state (returns empty RateLimit when missing)."""
    from nerd_herd.types import RateLimit
    if state is None:
        return RateLimit()
    return RateLimit(
        limit=getattr(state, f"{axis}_limit", None),
        remaining=getattr(state, f"{axis}_remaining", None),
        reset_at=int(getattr(state, f"{axis}_reset_at", 0) or 0) or None,
    )


# Axes adapter forwards (header_parser populates these on rate_limiter state objects).
_ADAPTER_AXES = (
    "rpm", "rph", "rpd", "rpw", "rpmonth",
    "tpm", "tph", "tpd", "tpw", "tpmonth",
    "itpm", "itpd", "otpm", "otpd",
    "cpd", "cpmonth",
)


def build_cloud_provider_state(kdv, provider):
    from nerd_herd.types import (
        CloudModelState, CloudProviderState, RateLimitMatrix,
    )
    model_ids = kdv._providers.get(provider)
    if not model_ids:
        return None

    def _matrix(state):
        m = RateLimitMatrix()
        for axis in _ADAPTER_AXES:
            setattr(m, axis, _rl(state, axis))
        return m

    models = {}
    for mid in model_ids:
        mstate = kdv._rate_limiter.model_limits.get(mid)
        models[mid] = CloudModelState(model_id=mid, limits=_matrix(mstate))

    prov_state = kdv._rate_limiter._provider_limits.get(provider)
    return CloudProviderState(
        provider=provider,
        models=models,
        limits=_matrix(prov_state),
    )
```

- [ ] **Step 4: Ensure rate_limiter exposes axis attrs**

In `packages/kuleden_donen_var/src/kuleden_donen_var/rate_limiter.py`, the `ModelLimits` dataclass currently only has `rpd_*`/`rpm_*`/`tpm_*`. Add minimum stubs (None defaults) for the axes the adapter expects, so `getattr` returns None for unsupported cells without crashing:

```python
@dataclass
class ModelLimits:
    # Existing fields preserved
    # Add new None-defaulted fields for axes adapter looks up:
    rph_limit: int | None = None
    rph_remaining: int | None = None
    rph_reset_at: float | None = None
    rpw_limit: int | None = None
    rpw_remaining: int | None = None
    rpw_reset_at: float | None = None
    rpmonth_limit: int | None = None
    rpmonth_remaining: int | None = None
    rpmonth_reset_at: float | None = None
    tph_limit: int | None = None
    tph_remaining: int | None = None
    tph_reset_at: float | None = None
    tpd_limit: int | None = None
    tpd_remaining: int | None = None
    tpd_reset_at: float | None = None
    tpw_limit: int | None = None
    tpw_remaining: int | None = None
    tpw_reset_at: float | None = None
    tpmonth_limit: int | None = None
    tpmonth_remaining: int | None = None
    tpmonth_reset_at: float | None = None
    itpm_limit: int | None = None
    itpm_remaining: int | None = None
    itpm_reset_at: float | None = None
    itpd_limit: int | None = None
    itpd_remaining: int | None = None
    itpd_reset_at: float | None = None
    otpm_limit: int | None = None
    otpm_remaining: int | None = None
    otpm_reset_at: float | None = None
    otpd_limit: int | None = None
    otpd_remaining: int | None = None
    otpd_reset_at: float | None = None
    cpd_limit: int | None = None
    cpd_remaining: int | None = None
    cpd_reset_at: float | None = None
    cpmonth_limit: int | None = None
    cpmonth_remaining: int | None = None
    cpmonth_reset_at: float | None = None
```

(Existing `rpd_*`, `rpm_*`, `tpm_*` fields stay — these are additions only.)

- [ ] **Step 5: Run, verify PASS**

```bash
timeout 60 pytest packages/kuleden_donen_var/tests/ -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/kuleden_donen_var/
git commit -m "refactor(kdv): adapter forwards all populated rate-limit cells"
```

---

### Task 9: `QueueProfile` widening + `PressureBreakdown` dataclass

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py` (`QueueProfile`)
- Create: `packages/nerd_herd/src/nerd_herd/breakdown.py`
- Test: `packages/nerd_herd/tests/test_breakdown.py`

- [ ] **Step 1: Failing tests**

```python
# packages/nerd_herd/tests/test_breakdown.py
from nerd_herd.breakdown import PressureBreakdown


def test_breakdown_serializable():
    b = PressureBreakdown(
        scalar=-0.4,
        signals={"S1": -0.3, "S2": -0.1, "S9": 0.2},
        modifiers={"M1": 1.5, "M3_difficulty": 5},
        bucket_totals={"burden": -0.05, "queue": 0.0, "other": -0.3},
        positive_total=0.0,
        negative_total=-0.4,
    )
    d = b.to_dict()
    assert d["scalar"] == -0.4
    assert d["signals"]["S1"] == -0.3
    assert d["bucket_totals"]["other"] == -0.3
```

```python
# packages/nerd_herd/tests/test_queue_profile_widening.py (new file)
from nerd_herd.types import QueueProfile


def test_queue_profile_widened_fields():
    qp = QueueProfile(
        total_ready_count=10,
        hard_tasks_count=3,
        by_difficulty={3: 4, 7: 3, 9: 1},
        by_capability={"vision": 2, "thinking": 1, "function_calling": 8},
        projected_tokens=120_000,
        projected_calls=80,
    )
    assert qp.by_difficulty[7] == 3
    assert qp.by_capability["vision"] == 2
    assert qp.projected_tokens == 120_000
    assert qp.projected_calls == 80
```

- [ ] **Step 2: Run, verify FAIL**

```bash
timeout 30 pytest packages/nerd_herd/tests/test_breakdown.py packages/nerd_herd/tests/test_queue_profile_widening.py -v
```

- [ ] **Step 3: Create `breakdown.py`**

```python
# packages/nerd_herd/src/nerd_herd/breakdown.py
"""Diagnostic struct returned alongside the pressure scalar."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PressureBreakdown:
    """Per-(model, task) signal-by-signal contribution. Logged to model_pick_log."""
    scalar: float
    signals: dict[str, float] = field(default_factory=dict)
    modifiers: dict[str, float] = field(default_factory=dict)
    bucket_totals: dict[str, float] = field(default_factory=dict)
    positive_total: float = 0.0
    negative_total: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Widen `QueueProfile`**

In `packages/nerd_herd/src/nerd_herd/types.py:130-135`, replace the existing `QueueProfile`:

```python
@dataclass
class QueueProfile:
    hard_tasks_count: int = 0
    total_ready_count: int = 0
    # New fields (2026-04-29 — pressure utilization equilibrium)
    by_difficulty: dict[int, int] = field(default_factory=dict)
    by_capability: dict[str, int] = field(default_factory=dict)
    projected_tokens: int = 0
    projected_calls: int = 0
```

- [ ] **Step 5: Run, verify PASS**

```bash
timeout 30 pytest packages/nerd_herd/tests/ -v
```
Expected: PASS for the new tests; existing QueueProfile usages unaffected (additions are field defaults).

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/
git commit -m "feat(nerd_herd): widen QueueProfile + add PressureBreakdown dataclass"
```

---

## Phase 4 — Signal Modules

Each signal task follows TDD: failing tests covering boundary cases, then minimal implementation, then commit. All signal modules live under `packages/nerd_herd/src/nerd_herd/signals/`.

### Task 10: S1 — per-axis remaining pressure

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/__init__.py`
- Create: `packages/nerd_herd/src/nerd_herd/signals/s1_remaining.py`
- Test: `packages/nerd_herd/tests/signals/__init__.py` (empty)
- Test: `packages/nerd_herd/tests/signals/test_s1.py`

- [ ] **Step 1: Failing tests**

```python
# packages/nerd_herd/tests/signals/test_s1.py
import pytest

from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s1_remaining import s1_remaining


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s1_empty_matrix_returns_zero():
    assert s1_remaining(_matrix(), reset_in_secs=0, in_flight=0) == 0.0


def test_s1_50pct_remaining_negative_below_depletion_threshold():
    # depletion_threshold for paid (per_call) profile is 0.15;
    # 50% remaining is above threshold → 0 (or positive abundance if enabled)
    m = _matrix(rpd=RateLimit(limit=100, remaining=50))
    p = s1_remaining(m, reset_in_secs=3600, in_flight=0, profile="per_call")
    assert -0.05 <= p <= 1.0


def test_s1_5pct_remaining_negative():
    m = _matrix(rpd=RateLimit(limit=100, remaining=5))
    p = s1_remaining(m, reset_in_secs=3600, in_flight=0, profile="per_call")
    # Below 15% threshold → depletion arm fires, magnitude proportional to depth
    assert p < -0.5


def test_s1_multi_cell_returns_worst_axis():
    # rpm flush, tpm critical
    m = _matrix(
        rpm=RateLimit(limit=30, remaining=25),     # 83% remaining
        tpm=RateLimit(limit=6000, remaining=600),  # 10% remaining
    )
    p = s1_remaining(m, reset_in_secs=60, in_flight=0, profile="per_call")
    assert p < -0.3  # worst-axis (tpm) wins


def test_s1_in_flight_overlay_drops_effective():
    m = _matrix(rpd=RateLimit(limit=100, remaining=10, in_flight=8))
    p = s1_remaining(m, reset_in_secs=3600, in_flight=0, profile="per_call")
    # effective = 10 - 8 = 2 → 2% remaining → strong negative
    assert p < -0.7


def test_s1_abundance_uses_max_when_no_negative_cell():
    m = _matrix(
        rpm=RateLimit(limit=30, remaining=29),
        rpd=RateLimit(limit=14_400, remaining=14_300),
    )
    p = s1_remaining(m, reset_in_secs=600, in_flight=0, profile="time_bucketed")
    # All cells flush + reset imminent → strong positive abundance
    assert p > 0.3
```

- [ ] **Step 2: Run, verify FAIL**

```bash
timeout 30 pytest packages/nerd_herd/tests/signals/test_s1.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create signals package**

```python
# packages/nerd_herd/src/nerd_herd/signals/__init__.py
"""Pressure signals — pure functions returning float ∈ [-1, +1]."""
```

```python
# packages/nerd_herd/tests/signals/__init__.py
```

- [ ] **Step 4: Implement S1**

```python
# packages/nerd_herd/src/nerd_herd/signals/s1_remaining.py
"""S1 — per-axis remaining pressure.

For every populated cell in a RateLimitMatrix:
  effective = max(0, remaining - in_flight)
  frac = effective / limit
  Map to two-arm pressure curve based on pool profile.

Fold:
  - If any cell is negative: take min (worst-axis-wins for depletion).
  - Else: take max (best-axis-wins for abundance).
"""
from __future__ import annotations

import math

from nerd_herd.types import RateLimit, RateLimitMatrix


# Profile-driven thresholds (matches existing pool_pressure.py constants)
PROFILE_PARAMS: dict[str, dict[str, float]] = {
    "per_call":      {"depletion_threshold": 0.15, "depletion_max": -1.0,
                      "abundance_mode": "flat", "abundance_max": 1.0,
                      "time_scale_secs": 86400.0, "exhausted_neutral": False},
    "time_bucketed": {"depletion_threshold": 0.30, "depletion_max": -0.5,
                      "abundance_mode": "time_decay", "abundance_max": 1.0,
                      "time_scale_secs": 86400.0, "exhausted_neutral": True},
}


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _cell_pressure(rl: RateLimit, *, reset_in_secs: float, profile: str) -> float:
    if rl.limit is None or rl.limit <= 0:
        return 0.0
    p = PROFILE_PARAMS[profile]
    effective = max(0, (rl.remaining or 0) - rl.in_flight)
    if p["exhausted_neutral"] and effective <= 0:
        return 0.0
    frac = min(1.0, effective / rl.limit)

    if frac < p["depletion_threshold"]:
        intensity = (p["depletion_threshold"] - frac) / p["depletion_threshold"]
        return _clamp(p["depletion_max"] * intensity, -1.0, 0.0)

    # Abundance arm
    if p["abundance_mode"] == "time_decay":
        if rl.reset_at and rl.reset_at > 0:
            time_weight = math.exp(-max(0.0, reset_in_secs) / p["time_scale_secs"])
            return _clamp(frac * time_weight * p["abundance_max"], 0.0, 1.0)
        return 0.0
    elif p["abundance_mode"] == "flat":
        return _clamp(p["abundance_max"], 0.0, 1.0)
    return 0.0


def s1_remaining(
    matrix: RateLimitMatrix,
    *,
    reset_in_secs: float = 0,
    in_flight: int = 0,
    profile: str = "per_call",
) -> float:
    """Compute S1 across all populated cells. Fold = worst-of-negatives or max-of-positives."""
    cell_pressures = [_cell_pressure(rl, reset_in_secs=reset_in_secs, profile=profile)
                      for _, rl in matrix.populated_cells()]
    if not cell_pressures:
        return 0.0
    negs = [p for p in cell_pressures if p < 0]
    if negs:
        return min(negs)
    return max(cell_pressures, default=0.0)
```

- [ ] **Step 5: Run tests, verify PASS**

```bash
timeout 30 pytest packages/nerd_herd/tests/signals/test_s1.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/ packages/nerd_herd/tests/signals/
git commit -m "feat(nerd_herd): S1 per-axis remaining pressure signal"
```

---

### Task 11: S2 — per-call burden

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s2_call_burden.py`
- Test: `packages/nerd_herd/tests/signals/test_s2.py`

- [ ] **Step 1: Failing tests**

```python
# packages/nerd_herd/tests/signals/test_s2.py
import pytest
from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s2_call_burden import s2_call_burden


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s2_zero_when_no_call():
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000))
    assert s2_call_burden(m, est_per_call_tokens=0) == 0.0


def test_s2_zero_when_below_30pct_threshold():
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000))
    assert s2_call_burden(m, est_per_call_tokens=1500) == 0.0  # 25%


def test_s2_negative_at_60pct_bite():
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000))
    p = s2_call_burden(m, est_per_call_tokens=3600)  # 60%
    assert -0.5 < p < -0.3


def test_s2_max_negative_at_100pct_bite():
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000))
    p = s2_call_burden(m, est_per_call_tokens=6000)
    assert p == pytest.approx(-1.0, abs=0.05)


def test_s2_picks_largest_bite_across_windows():
    m = _matrix(
        tpm=RateLimit(limit=6000, remaining=6000),       # 50% bite of 3000 tokens
        tpd=RateLimit(limit=1_000_000, remaining=1_000_000),  # 0.3% bite
    )
    p = s2_call_burden(m, est_per_call_tokens=3000)
    # tpm bite (50%) wins over tpd bite (0.3%)
    assert -0.5 < p < -0.1
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/signals/s2_call_burden.py
"""S2 — per-call token burden.

For each token-axis cell with remaining > 0:
    bite_frac = est_per_call_tokens / remaining
    pressure = -clip(bite_frac - 0.30, 0, 0.70) / 0.70
Fold: largest bite (most-stressed window) wins.
"""
from __future__ import annotations

from nerd_herd.types import RateLimitMatrix


BITE_THRESHOLD = 0.30
BITE_RANGE = 0.70  # 0.30 → 0, 1.00 → -1


def s2_call_burden(matrix: RateLimitMatrix, *, est_per_call_tokens: int) -> float:
    if est_per_call_tokens <= 0:
        return 0.0
    worst = 0.0
    for _, rl in matrix.token_cells():
        remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        bite = est_per_call_tokens / remaining
        excess = max(0.0, bite - BITE_THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess / BITE_RANGE)
        if pressure < worst:
            worst = pressure
    return worst
```

- [ ] **Step 4: Run, PASS**

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s2_call_burden.py packages/nerd_herd/tests/signals/test_s2.py
git commit -m "feat(nerd_herd): S2 per-call burden signal"
```

---

### Task 12: S3 — per-task burden

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s3_task_burden.py`
- Test: `packages/nerd_herd/tests/signals/test_s3.py`

- [ ] **Step 1: Tests**

```python
# packages/nerd_herd/tests/signals/test_s3.py
import pytest
from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s3_task_burden import s3_task_burden


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s3_zero_when_no_task_tokens():
    assert s3_task_burden(_matrix(tpm=RateLimit(limit=6000, remaining=6000)), est_per_task_tokens=0) == 0.0


def test_s3_uses_largest_window_bite():
    # tpm = 6000, tpd = 1M; 50k task = 8x tpm but 5% tpd
    m = _matrix(
        tpm=RateLimit(limit=6000, remaining=6000),
        tpd=RateLimit(limit=1_000_000, remaining=1_000_000),
    )
    p = s3_task_burden(m, est_per_task_tokens=50_000)
    # tpm window can't even hold one task; overwhelming negative
    assert p < -0.5
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/signals/s3_task_burden.py
"""S3 — cumulative per-task token burden across iterations.

Same shape as S2 but uses `est_per_task_tokens = est_per_call_tokens × est_iterations`.
Captures the full task's hit on each token-axis budget.
"""
from __future__ import annotations

from nerd_herd.signals.s2_call_burden import BITE_RANGE, BITE_THRESHOLD
from nerd_herd.types import RateLimitMatrix


def s3_task_burden(matrix: RateLimitMatrix, *, est_per_task_tokens: int) -> float:
    if est_per_task_tokens <= 0:
        return 0.0
    worst = 0.0
    for _, rl in matrix.token_cells():
        remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        bite = est_per_task_tokens / remaining
        excess = max(0.0, bite - BITE_THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess / BITE_RANGE)
        if pressure < worst:
            worst = pressure
    return worst
```

- [ ] **Step 4: PASS, Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s3_task_burden.py packages/nerd_herd/tests/signals/test_s3.py
git commit -m "feat(nerd_herd): S3 cumulative task burden signal"
```

---

### Task 13: S4 — queue token pressure

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py`
- Test: `packages/nerd_herd/tests/signals/test_s4.py`

- [ ] **Step 1: Tests**

```python
# packages/nerd_herd/tests/signals/test_s4.py
import pytest
from nerd_herd.types import QueueProfile, RateLimit, RateLimitMatrix
from nerd_herd.signals.s4_queue_tokens import s4_queue_tokens


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s4_zero_when_no_queue():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=0)
    assert s4_queue_tokens(m, queue=qp) == 0.0


def test_s4_zero_below_70pct_demand_ratio():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=300_000)  # 60% of 500k
    assert s4_queue_tokens(m, queue=qp) == 0.0


def test_s4_negative_at_95pct_demand():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=475_000)  # 95% of remaining
    p = s4_queue_tokens(m, queue=qp)
    assert -0.6 < p < -0.4


def test_s4_clipped_at_oversubscription():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=750_000)  # 150% — over budget
    p = s4_queue_tokens(m, queue=qp)
    assert p == pytest.approx(-1.0, abs=0.05)
```

- [ ] **Step 2-3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py
"""S4 — queue token pressure.

Sum est_per_task_tokens over unblocked + pending + dep-resolved tasks (queue
projection done at queue_profile_push). Compare to remaining token budget
across windows. Fold: most-stressed window wins.

Threshold: 70% projected → 0; 100% → -0.5; 120%+ → -1.0.
"""
from __future__ import annotations

from nerd_herd.types import QueueProfile, RateLimitMatrix


THRESHOLD = 0.70
SLOPE = 2.0  # demand 70% → 0; 95% → -0.5; 120% → -1.0


def s4_queue_tokens(matrix: RateLimitMatrix, *, queue: QueueProfile) -> float:
    projected = queue.projected_tokens
    if projected <= 0:
        return 0.0
    worst = 0.0
    for _, rl in matrix.token_cells():
        remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        ratio = projected / remaining
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
```

- [ ] **Step 4-5: PASS, Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py packages/nerd_herd/tests/signals/test_s4.py
git commit -m "feat(nerd_herd): S4 queue token pressure signal"
```

---

### Task 14: S5 — queue request pressure

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py`
- Test: `packages/nerd_herd/tests/signals/test_s5.py`

- [ ] **Step 1: Tests**

```python
# packages/nerd_herd/tests/signals/test_s5.py
import pytest
from nerd_herd.types import QueueProfile, RateLimit, RateLimitMatrix
from nerd_herd.signals.s5_queue_calls import s5_queue_calls


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s5_zero_when_no_queue():
    m = _matrix(rpd=RateLimit(limit=1000, remaining=500))
    assert s5_queue_calls(m, queue=QueueProfile(projected_calls=0)) == 0.0


def test_s5_negative_at_120pct_demand():
    m = _matrix(rpd=RateLimit(limit=1000, remaining=500))
    qp = QueueProfile(projected_calls=600)  # 120%
    p = s5_queue_calls(m, queue=qp)
    assert p == pytest.approx(-1.0, abs=0.05)
```

- [ ] **Step 2-3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py
"""S5 — queue request pressure.

Same shape as S4 but on the request axis. Per-task call cost = est_iterations
(folded into queue.projected_calls at queue_profile_push time).
"""
from __future__ import annotations

from nerd_herd.signals.s4_queue_tokens import SLOPE, THRESHOLD
from nerd_herd.types import QueueProfile, RateLimitMatrix


def s5_queue_calls(matrix: RateLimitMatrix, *, queue: QueueProfile) -> float:
    projected = queue.projected_calls
    if projected <= 0:
        return 0.0
    worst = 0.0
    for _, rl in matrix.request_cells():
        remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        ratio = projected / remaining
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
```

- [ ] **Step 4-5: PASS, Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py packages/nerd_herd/tests/signals/test_s5.py
git commit -m "feat(nerd_herd): S5 queue request pressure signal"
```

---

### Task 15: S6 — capable-supply

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py`
- Test: `packages/nerd_herd/tests/signals/test_s6.py`

- [ ] **Step 1: Tests**

```python
# packages/nerd_herd/tests/signals/test_s6.py
import pytest
from nerd_herd.signals.s6_capable_supply import s6_capable_supply


class FakeModel:
    def __init__(self, name, capabilities, rpd_remaining, supports_difficulty):
        self.name = name
        self.capabilities = capabilities
        self.rpd_remaining = rpd_remaining
        self.supports_difficulty = supports_difficulty


def test_s6_zero_when_no_demand():
    model = FakeModel("a", {"vision"}, 100, lambda d: True)
    queue = {"by_capability": {}, "by_difficulty": {}}
    eligible = [model]
    p = s6_capable_supply(model, queue=queue, eligible_models=eligible)
    assert p == 0.0


def test_s6_zero_when_supply_meets_demand():
    m = FakeModel("a", {"vision"}, 100, lambda d: True)
    queue = {"by_capability": {"vision": 5}, "by_difficulty": {}}
    p = s6_capable_supply(m, queue=queue, eligible_models=[m], iter_avg=10)
    # demand 5 calls × 10 iters = 50; supply 100; ratio 0.5 → no pressure
    assert p == 0.0


def test_s6_negative_when_demand_exceeds_supply():
    m = FakeModel("a", {"vision"}, 50, lambda d: True)
    queue = {"by_capability": {"vision": 50}, "by_difficulty": {}}
    p = s6_capable_supply(m, queue=queue, eligible_models=[m], iter_avg=10)
    # demand 500; supply 50; ratio 10x → strong negative
    assert p == pytest.approx(-1.0, abs=0.05)


def test_s6_zero_when_model_not_eligible():
    m = FakeModel("a", {"function_calling"}, 100, lambda d: True)  # no vision
    queue = {"by_capability": {"vision": 50}, "by_difficulty": {}}
    p = s6_capable_supply(m, queue=queue, eligible_models=[], iter_avg=10)
    assert p == 0.0
```

- [ ] **Step 2-3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py
"""S6 — capable-capacity overlap.

For each capability the queue needs (vision, thinking, function_calling, hard
difficulty tier), compute the ratio of queue demand vs available capacity
across models that can serve it. If the model under evaluation IS eligible
for an under-supplied capability, it carries conserve-pressure proportional
to the shortage.

Today's S6 is a coarse first pass: takes the worst capability ratio across
demands the model can serve. Per-capability attribution weighting is left
for calibration Phase 2.
"""
from __future__ import annotations

from typing import Any


THRESHOLD = 0.70
SLOPE = 2.0


def _model_capabilities(model: Any) -> set[str]:
    caps = getattr(model, "capabilities", None) or set()
    if isinstance(caps, dict):
        return {k for k, v in caps.items() if v}
    return set(caps)


def _supply_for(capability: str, eligible_models: list, iter_avg: float) -> float:
    """Sum of remaining call-capacity across models capable of capability."""
    total = 0.0
    for m in eligible_models:
        if capability not in _model_capabilities(m):
            continue
        rem = float(getattr(m, "rpd_remaining", 0) or 0)
        total += rem * max(1.0, iter_avg)
    return total


def s6_capable_supply(
    model: Any,
    *,
    queue: dict | Any,
    eligible_models: list,
    iter_avg: float = 8.0,
) -> float:
    """Compute S6 for `model` given the current queue and capable-model pool.

    queue: dict-like with `by_capability: {cap: count}` and `by_difficulty: {d: count}`.
    eligible_models: full list of models eligible for the relevant capabilities.
    iter_avg: average iterations per task (multiplies queue counts to call demand).
    """
    by_capability = (
        queue.get("by_capability", {}) if isinstance(queue, dict)
        else getattr(queue, "by_capability", {}) or {}
    )
    if not by_capability:
        return 0.0
    model_caps = _model_capabilities(model)
    worst = 0.0
    for capability, count in by_capability.items():
        if capability not in model_caps:
            continue
        if count <= 0:
            continue
        demand = float(count) * max(1.0, iter_avg)
        supply = _supply_for(capability, eligible_models, iter_avg)
        if supply <= 0:
            continue
        ratio = demand / supply
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
```

- [ ] **Step 4-5: PASS, Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py packages/nerd_herd/tests/signals/test_s6.py
git commit -m "feat(nerd_herd): S6 capable-supply overlap signal"
```

---

### Task 16: S7 — burn rate + supporting `burn_log` module

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/burn_log.py`
- Create: `packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py`
- Test: `packages/nerd_herd/tests/test_burn_log.py`
- Test: `packages/nerd_herd/tests/signals/test_s7.py`

- [ ] **Step 1: Tests for burn_log**

```python
# packages/nerd_herd/tests/test_burn_log.py
import time

from nerd_herd.burn_log import BurnLog


def test_burn_log_records_and_extrapolates():
    log = BurnLog(window_secs=300)
    now = time.time()
    log.record(provider="groq", model="llama-8b", tokens=1000, calls=1, now=now - 60)
    log.record(provider="groq", model="llama-8b", tokens=2000, calls=1, now=now - 30)
    log.record(provider="groq", model="llama-8b", tokens=1500, calls=1, now=now - 10)
    rate = log.rate(provider="groq", model="llama-8b", now=now)
    assert rate.tokens_per_min > 0
    assert rate.calls_per_min > 0


def test_burn_log_drops_old_entries():
    log = BurnLog(window_secs=60)
    log.record(provider="x", model="y", tokens=100, calls=1, now=time.time() - 3600)
    rate = log.rate(provider="x", model="y", now=time.time())
    assert rate.tokens_per_min == 0
    assert rate.calls_per_min == 0
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Implement burn_log**

```python
# packages/nerd_herd/src/nerd_herd/burn_log.py
"""Rolling burn-rate log per (provider, model). Used by S7."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class BurnRate:
    tokens_per_min: float
    calls_per_min: float


class BurnLog:
    def __init__(self, window_secs: float = 300.0):
        self._window = window_secs
        self._entries: dict[tuple[str, str], deque] = {}
        self._lock = Lock()

    def record(self, *, provider: str, model: str, tokens: int, calls: int = 1, now: float | None = None):
        ts = now if now is not None else time.time()
        key = (provider, model)
        with self._lock:
            d = self._entries.setdefault(key, deque())
            d.append((ts, tokens, calls))
            self._evict(d, ts)

    def _evict(self, d: deque, now: float):
        cutoff = now - self._window
        while d and d[0][0] < cutoff:
            d.popleft()

    def rate(self, *, provider: str, model: str, now: float | None = None) -> BurnRate:
        ts = now if now is not None else time.time()
        key = (provider, model)
        with self._lock:
            d = self._entries.get(key)
            if not d:
                return BurnRate(0.0, 0.0)
            self._evict(d, ts)
            tot_tok = sum(e[1] for e in d)
            tot_calls = sum(e[2] for e in d)
        # rate normalized to per-minute
        return BurnRate(
            tokens_per_min=(tot_tok * 60.0) / self._window,
            calls_per_min=(tot_calls * 60.0) / self._window,
        )


# Module-level singleton (process-scoped); main.py wires up before snapshot use
_GLOBAL_BURN_LOG: BurnLog | None = None


def get_burn_log() -> BurnLog:
    global _GLOBAL_BURN_LOG
    if _GLOBAL_BURN_LOG is None:
        _GLOBAL_BURN_LOG = BurnLog(window_secs=300.0)
    return _GLOBAL_BURN_LOG
```

- [ ] **Step 4: Tests for S7**

```python
# packages/nerd_herd/tests/signals/test_s7.py
import time
import pytest

from nerd_herd.burn_log import BurnLog
from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s7_burn_rate import s7_burn_rate


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s7_zero_when_no_history():
    log = BurnLog(window_secs=300)
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000, reset_at=int(time.time())+60))
    p = s7_burn_rate(m, provider="groq", model="x", burn_log=log, now=time.time())
    assert p == 0.0


def test_s7_negative_when_burn_extrapolates_over_remaining():
    log = BurnLog(window_secs=300)
    now = time.time()
    # 1000 tok consumed over last 60s = 200 tok/sec → 12k/min
    log.record(provider="groq", model="x", tokens=1000, calls=5, now=now - 60)
    log.record(provider="groq", model="x", tokens=2000, calls=8, now=now - 10)
    m = _matrix(tpm=RateLimit(limit=6000, remaining=2000,
                              reset_at=int(now)+30))  # 30s to reset, low remaining
    p = s7_burn_rate(m, provider="groq", model="x", burn_log=log, now=now)
    assert p < -0.3
```

- [ ] **Step 5: Implement S7**

```python
# packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py
"""S7 — burn-rate extrapolation.

Compare recent (5min) consumption rate × seconds-to-reset against remaining
budget. Independent of S4/S5 — captures off-queue demand and validates queue
projections against historical truth.

Cold-start (no history) → 0.
"""
from __future__ import annotations

import time

from nerd_herd.burn_log import BurnLog
from nerd_herd.types import RateLimitMatrix


THRESHOLD = 0.70
SLOPE = 2.0


def s7_burn_rate(
    matrix: RateLimitMatrix,
    *,
    provider: str,
    model: str,
    burn_log: BurnLog,
    now: float | None = None,
) -> float:
    ts = now if now is not None else time.time()
    rate = burn_log.rate(provider=provider, model=model, now=ts)
    if rate.tokens_per_min <= 0 and rate.calls_per_min <= 0:
        return 0.0
    worst = 0.0
    for axis, rl in matrix.populated_cells():
        if rl.reset_at is None or rl.reset_at <= ts:
            continue
        secs_to_reset = max(0.0, rl.reset_at - ts)
        if axis.startswith("tp") or axis.startswith("itp") or axis.startswith("otp"):
            extrapolated = rate.tokens_per_min * (secs_to_reset / 60.0)
        elif axis.startswith("rp"):
            extrapolated = rate.calls_per_min * (secs_to_reset / 60.0)
        else:
            continue
        remaining = max(1, (rl.remaining or 0) - rl.in_flight)
        ratio = extrapolated / remaining
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
```

- [ ] **Step 6: PASS, Step 7: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/burn_log.py packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py packages/nerd_herd/tests/test_burn_log.py packages/nerd_herd/tests/signals/test_s7.py
git commit -m "feat(nerd_herd): S7 burn-rate signal + BurnLog rolling counter"
```

---

### Task 17: S9 — universal perishability

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s9_perishability.py`
- Test: `packages/nerd_herd/tests/signals/test_s9.py`

- [ ] **Step 1: Tests covering all 6 branches**

```python
# packages/nerd_herd/tests/signals/test_s9.py
import time
import pytest

from nerd_herd.types import (
    LocalModelState, RateLimit, RateLimitMatrix,
)
from nerd_herd.signals.s9_perishability import s9_perishability


class FakeModel:
    def __init__(self, *, is_local=False, is_free=False, name="x", size_mb=0):
        self.is_local = is_local
        self.is_free = is_free
        self.name = name
        self.size_mb = size_mb
        self.is_loaded = False


# Loaded local + idle
def test_s9_loaded_local_idle_positive():
    m = FakeModel(is_local=True, name="loaded-x")
    m.is_loaded = True
    local = LocalModelState(model_name="loaded-x", idle_seconds=30, requests_processing=0)
    p = s9_perishability(m, local=local, vram_avail_mb=8000, matrix=RateLimitMatrix(),
                         task_difficulty=5, now=time.time())
    assert p == pytest.approx(0.25, abs=0.05)


# Loaded local + processing → busy penalty
def test_s9_loaded_local_busy_negative():
    m = FakeModel(is_local=True, name="loaded-x")
    m.is_loaded = True
    local = LocalModelState(model_name="loaded-x", idle_seconds=0, requests_processing=1)
    p = s9_perishability(m, local=local, vram_avail_mb=8000, matrix=RateLimitMatrix(),
                         task_difficulty=5, now=time.time())
    assert p == pytest.approx(-0.10, abs=0.01)


# Cold local + VRAM available
def test_s9_cold_local_vram_available_positive():
    m = FakeModel(is_local=True, name="cold-x", size_mb=4000)
    m.is_loaded = False
    local = LocalModelState(model_name="other-loaded")
    p = s9_perishability(m, local=local, vram_avail_mb=8000, matrix=RateLimitMatrix(),
                         task_difficulty=5, now=time.time())
    assert p == pytest.approx(0.4, abs=0.05)


# Cold local + VRAM unavailable
def test_s9_cold_local_no_vram_negative():
    m = FakeModel(is_local=True, name="cold-x", size_mb=8000)
    m.is_loaded = False
    local = LocalModelState(model_name="other-loaded")
    p = s9_perishability(m, local=local, vram_avail_mb=4000, matrix=RateLimitMatrix(),
                         task_difficulty=5, now=time.time())
    assert p == pytest.approx(-0.5, abs=0.05)


# Free cloud + reset imminent + flush
def test_s9_free_cloud_reset_imminent_positive():
    m = FakeModel(is_free=True, name="groq-x")
    now = time.time()
    matrix = RateLimitMatrix(
        rpd=RateLimit(limit=1000, remaining=950, reset_at=int(now + 600)),  # 10min
    )
    p = s9_perishability(m, local=None, vram_avail_mb=0, matrix=matrix,
                         task_difficulty=5, now=now)
    assert p > 0.7  # strong abundance


# Paid cloud + budget flush + hard task
def test_s9_paid_cloud_flush_hard_task_positive():
    m = FakeModel(is_free=False, name="claude-opus")
    matrix = RateLimitMatrix(
        rpd=RateLimit(limit=200, remaining=190),
    )
    p = s9_perishability(m, local=None, vram_avail_mb=0, matrix=matrix,
                         task_difficulty=9, now=time.time())
    assert p == pytest.approx(1.0, abs=0.05)


# Paid cloud + budget flush + easy task → 0 (no perishability fires)
def test_s9_paid_cloud_flush_easy_task_zero():
    m = FakeModel(is_free=False, name="claude-opus")
    matrix = RateLimitMatrix(
        rpd=RateLimit(limit=200, remaining=190),
    )
    p = s9_perishability(m, local=None, vram_avail_mb=0, matrix=matrix,
                         task_difficulty=3, now=time.time())
    assert p == 0.0
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/signals/s9_perishability.py
"""S9 — universal perishability signal.

Equilibrium core. Same signal across pool types; per-pool computation differs.
Returns "how strongly does NOT using this model right now waste capacity
we'll lose."
"""
from __future__ import annotations

import math
import time
from typing import Any

from nerd_herd.types import LocalModelState, RateLimitMatrix


LOCAL_IDLE_SAT_SECS = 60.0
LOCAL_IDLE_MAX = 0.5
LOCAL_BUSY_PENALTY = -0.10
COLD_LOCAL_VRAM_OK = 0.4
COLD_LOCAL_NO_VRAM = -0.5
TIME_DECAY_SCALE_SECS = 86400.0
PAID_RIGHT_TOOL_DIFFICULTY_THRESHOLD = 7
FLUSH_THRESHOLD = 0.7


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def s9_perishability(
    model: Any,
    *,
    local: LocalModelState | None,
    vram_avail_mb: int,
    matrix: RateLimitMatrix,
    task_difficulty: int,
    now: float | None = None,
) -> float:
    ts = now if now is not None else time.time()

    # ── Local branches ─────────────────────────────────────────────
    if getattr(model, "is_local", False):
        loaded_name = (local.model_name or "") if local else ""
        if getattr(model, "is_loaded", False) and loaded_name == getattr(model, "name", ""):
            if int(getattr(local, "requests_processing", 0) or 0) > 0:
                return LOCAL_BUSY_PENALTY
            idle = float(getattr(local, "idle_seconds", 0.0) or 0.0)
            return _clamp(min(1.0, idle / LOCAL_IDLE_SAT_SECS) * LOCAL_IDLE_MAX, 0, LOCAL_IDLE_MAX)
        # Cold local
        size_mb = int(getattr(model, "size_mb", 0) or 0)
        if size_mb <= 0 or vram_avail_mb >= size_mb:
            return COLD_LOCAL_VRAM_OK
        return COLD_LOCAL_NO_VRAM

    # ── Cloud branches ─────────────────────────────────────────────
    # Find the strongest perishability cell across populated request-axis cells
    strongest = 0.0
    if getattr(model, "is_free", False):
        for _, rl in matrix.request_cells():
            if rl.limit is None or rl.limit <= 0:
                continue
            effective = max(0, (rl.remaining or 0) - rl.in_flight)
            frac = effective / rl.limit
            if frac < FLUSH_THRESHOLD:
                continue
            if rl.reset_at is None or rl.reset_at <= ts:
                continue
            secs_to_reset = max(0.0, rl.reset_at - ts)
            time_weight = math.exp(-secs_to_reset / TIME_DECAY_SCALE_SECS)
            v = _clamp(frac * time_weight, 0.0, 1.0)
            if v > strongest:
                strongest = v
        return strongest

    # Paid cloud — right-tool perishability when budget flush + hard task
    if task_difficulty < PAID_RIGHT_TOOL_DIFFICULTY_THRESHOLD:
        return 0.0
    for _, rl in matrix.request_cells():
        if rl.limit is None or rl.limit <= 0:
            continue
        effective = max(0, (rl.remaining or 0) - rl.in_flight)
        frac = effective / rl.limit
        if frac >= FLUSH_THRESHOLD:
            return 1.0
    return 0.0
```

- [ ] **Step 4-5: PASS, Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s9_perishability.py packages/nerd_herd/tests/signals/test_s9.py
git commit -m "feat(nerd_herd): S9 universal perishability signal"
```

---

### Task 18: S10 — failure state

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s10_failure.py`
- Test: `packages/nerd_herd/tests/signals/test_s10.py`

- [ ] **Step 1: Tests + Implementation in one task** (signal is tiny)

```python
# packages/nerd_herd/tests/signals/test_s10.py
import pytest
from nerd_herd.signals.s10_failure import s10_failure


def test_s10_zero_failures():
    assert s10_failure(consecutive_failures=0) == 0.0


def test_s10_one_failure():
    assert s10_failure(consecutive_failures=1) == pytest.approx(-0.2, abs=0.01)


def test_s10_three_failures():
    assert s10_failure(consecutive_failures=3) == pytest.approx(-0.5, abs=0.01)


def test_s10_clamps_at_minus_half():
    assert s10_failure(consecutive_failures=10) == pytest.approx(-0.5, abs=0.01)
```

```python
# packages/nerd_herd/src/nerd_herd/signals/s10_failure.py
"""S10 — failure state. Linear ramp from 0 → -0.5 with consecutive failures."""
from __future__ import annotations


def s10_failure(*, consecutive_failures: int) -> float:
    if consecutive_failures <= 0:
        return 0.0
    if consecutive_failures <= 2:
        return -0.2
    return -0.5
```

- [ ] **Step 2: PASS, Step 3: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s10_failure.py packages/nerd_herd/tests/signals/test_s10.py
git commit -m "feat(nerd_herd): S10 failure-state signal"
```

---

### Task 19: S11 — cost burden

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s11_cost.py`
- Test: `packages/nerd_herd/tests/signals/test_s11.py`

- [ ] **Step 1: Tests**

```python
# packages/nerd_herd/tests/signals/test_s11.py
import pytest
from nerd_herd.signals.s11_cost import s11_cost


def test_s11_zero_when_no_cap():
    p = s11_cost(est_call_cost=0.05, daily_cost_remaining=0.0)
    assert p == 0.0


def test_s11_zero_when_below_threshold():
    p = s11_cost(est_call_cost=0.05, daily_cost_remaining=10.0)
    assert p == 0.0  # 0.5% of remaining


def test_s11_negative_when_call_eats_majority():
    p = s11_cost(est_call_cost=8.0, daily_cost_remaining=10.0)
    assert p < -0.3
```

- [ ] **Step 2-3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/signals/s11_cost.py
"""S11 — cost burden.

Compare est_call_cost against daily/monthly cost remaining. Same shape as
S2 but on cost axis. Zero when no cost cap configured.
"""
from __future__ import annotations


THRESHOLD = 0.30
SLOPE = 1.0 / 0.70


def s11_cost(*, est_call_cost: float, daily_cost_remaining: float) -> float:
    if est_call_cost <= 0 or daily_cost_remaining <= 0:
        return 0.0
    bite = est_call_cost / daily_cost_remaining
    excess = max(0.0, bite - THRESHOLD)
    if excess <= 0:
        return 0.0
    return -min(1.0, excess * SLOPE)
```

- [ ] **Step 4-5: PASS, Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s11_cost.py packages/nerd_herd/tests/signals/test_s11.py
git commit -m "feat(nerd_herd): S11 cost-burden signal"
```

---

## Phase 5 — Modifiers + Combination

### Task 20: M1, M2, M3 modifiers

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/modifiers.py`
- Test: `packages/nerd_herd/tests/test_modifiers.py`

- [ ] **Step 1: Tests**

```python
# packages/nerd_herd/tests/test_modifiers.py
import pytest

from nerd_herd.modifiers import (
    M1_capacity_amplifier,
    M2_perishability_dampener,
    M3_difficulty_weights,
)


# M1
def test_m1_small_pool_amplifies():
    assert M1_capacity_amplifier(limit=10) > 1.3
    assert M1_capacity_amplifier(limit=10) <= 2.0


def test_m1_medium_pool_neutral():
    assert M1_capacity_amplifier(limit=100) == pytest.approx(1.0, abs=0.05)


def test_m1_large_pool_dampens():
    assert M1_capacity_amplifier(limit=1000) < 0.7


def test_m1_clamps():
    assert M1_capacity_amplifier(limit=1) == 2.0
    assert M1_capacity_amplifier(limit=1_000_000) == 0.5


# M2
def test_m2_strong_perishability_no_damp():
    assert M2_perishability_dampener(fit_excess=0.5, s9_value=0.7) == 1.0


def test_m2_no_perishability_full_damp():
    # fit_excess=0.5 → 1 - 0.5*0.5 = 0.75
    assert M2_perishability_dampener(fit_excess=0.5, s9_value=0.0) == pytest.approx(0.75, abs=0.01)


def test_m2_mild_perishability_partial_damp():
    # 0.2 < S9=0.3 < 0.5 → partial: 1 - 0.5*0.25 = 0.875
    assert M2_perishability_dampener(fit_excess=0.5, s9_value=0.3) == pytest.approx(0.875, abs=0.01)


# M3
def test_m3_easy_downweights_burden():
    w = M3_difficulty_weights(difficulty=2)
    assert w["S2"] == 0.5
    assert w["S3"] == 0.5
    assert w["S4"] == 1.5


def test_m3_hard_upweights_burden():
    w = M3_difficulty_weights(difficulty=9)
    assert w["S2"] == 1.5
    assert w["S4"] == 0.7


def test_m3_mid_neutral():
    w = M3_difficulty_weights(difficulty=5)
    assert all(v == 1.0 for k, v in w.items() if k in ("S2", "S3", "S4"))
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/modifiers.py
"""Modifiers — reshape signal values without computing their own pressure."""
from __future__ import annotations

import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ── M1: Capacity amplifier ─────────────────────────────────────────
def M1_capacity_amplifier(*, limit: int) -> float:
    """Small pools amplify negative signals; large pools dampen them.

    factor = clip(2.0 - 0.5 * log10(limit), 0.5, 2.0)
      limit=10  → 1.5
      limit=100 → 1.0
      limit=1000→ 0.5
    """
    if limit <= 0:
        return 1.0
    return _clamp(2.0 - 0.5 * math.log10(limit), 0.5, 2.0)


# ── M2: Perishability-conditional fit-excess dampener ──────────────
def M2_perishability_dampener(*, fit_excess: float, s9_value: float) -> float:
    """Reduce positive signals when model overqualified, UNLESS perishability fires.

    s9_value > 0.5  → no damp (1.0)
    s9_value > 0.2  → partial damp (clip(1 - fit_excess * 0.25, 0.5, 1.0))
    else            → full damp   (clip(1 - fit_excess * 0.5,  0.0, 1.0))
    """
    excess = max(0.0, fit_excess)
    if s9_value > 0.5:
        return 1.0
    if s9_value > 0.2:
        return _clamp(1.0 - excess * 0.25, 0.5, 1.0)
    return _clamp(1.0 - excess * 0.5, 0.0, 1.0)


# ── M3: Difficulty re-weights ──────────────────────────────────────
def M3_difficulty_weights(*, difficulty: int, model_is_paid: bool = False) -> dict[str, float]:
    """Per-signal weights driven by task difficulty.

    Easy (d≤3):   down-weight burden, up-weight queue & abundance
    Hard (d≥7):   up-weight burden, down-weight queue, S9 inverts on paid
    Mid:          all 1.0
    """
    if difficulty <= 3:
        s9_w = 0.7 if model_is_paid else 1.5
        return {
            "S1": 1.0, "S2": 0.5, "S3": 0.5,
            "S4": 1.5, "S5": 1.5, "S6": 1.5,
            "S7": 1.0, "S9": s9_w,
            "S10": 1.0, "S11": 1.5,
        }
    if difficulty >= 7:
        s9_w = 1.5 if model_is_paid else 0.7
        return {
            "S1": 1.0, "S2": 1.5, "S3": 1.5,
            "S4": 0.7, "S5": 0.7, "S6": 0.7,
            "S7": 1.0, "S9": s9_w,
            "S10": 1.0, "S11": 0.7,
        }
    return {k: 1.0 for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11")}
```

- [ ] **Step 4-5: PASS, Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/modifiers.py packages/nerd_herd/tests/test_modifiers.py
git commit -m "feat(nerd_herd): M1/M2/M3 modifiers"
```

---

### Task 21: combine() — bucket worst-wins + gated abundance

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/combine.py`
- Test: `packages/nerd_herd/tests/test_combine.py`

- [ ] **Step 1: Tests**

```python
# packages/nerd_herd/tests/test_combine.py
import pytest

from nerd_herd.combine import combine_signals


def test_all_neutral_returns_zero():
    sigs = {k: 0.0 for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11")}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    assert breakdown.scalar == 0.0


def test_burden_bucket_takes_min():
    sigs = {"S1": 0, "S2": -0.3, "S3": -0.5, "S4": 0, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    # burden_neg = min(-0.3, -0.5) = -0.5; W_burden=0.5 → -0.25
    assert breakdown.bucket_totals["burden"] == pytest.approx(-0.25, abs=0.01)


def test_abundance_gated_off_by_significant_negative():
    sigs = {"S1": 0.6, "S2": 0, "S3": 0, "S4": -0.5, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0.4, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    # queue_neg=-0.5; weighted=-0.35 (<-0.2 gate). abundance gated off.
    assert breakdown.positive_total == 0.0
    assert breakdown.negative_total < -0.2


def test_abundance_fires_with_no_significant_negative():
    sigs = {"S1": 0.6, "S2": 0, "S3": 0, "S4": -0.1, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0.4, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    # queue_neg=-0.1; weighted=-0.07 (>-0.2). Abundance fires: max(S1, S9) = 0.6
    assert breakdown.positive_total == pytest.approx(0.6, abs=0.05)
    assert breakdown.scalar > 0.4


def test_scalar_clipped():
    sigs = {"S1": 1.0, "S2": -1.0, "S3": -1.0, "S4": -1.0, "S5": -1.0, "S6": -1.0,
            "S7": -1.0, "S9": -1.0, "S10": -1.0, "S11": -1.0}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    assert breakdown.scalar == pytest.approx(-1.0, abs=0.01)
```

- [ ] **Step 2-3: Implement**

```python
# packages/nerd_herd/src/nerd_herd/combine.py
"""Combination logic: signals → scalar.

Worst-wins per bucket; weighted sum across buckets; gated abundance.
"""
from __future__ import annotations

from nerd_herd.breakdown import PressureBreakdown


W_BURDEN = 0.5
W_QUEUE = 0.7
W_OTHER = 1.0
ABUNDANCE_GATE = -0.2

BURDEN_BUCKET = ("S2", "S3")
QUEUE_BUCKET = ("S4", "S5", "S6")
OTHER_BUCKET = ("S1", "S7", "S9", "S10", "S11")
POSITIVE_ARM_SIGNALS = ("S1", "S9")


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def combine_signals(*, signals: dict[str, float], weights: dict[str, float]) -> PressureBreakdown:
    """Combine ten weighted signals into a single scalar plus diagnostic struct.

    Args:
      signals: {"S1": ..., "S2": ..., ...} ∈ [-1, +1]
      weights: per-signal weight from M3 (already including M1 amplifier on negatives if applicable)

    Returns:
      PressureBreakdown with scalar + per-bucket totals + per-signal contributions.
    """
    weighted = {k: signals.get(k, 0.0) * weights.get(k, 1.0) for k in OTHER_BUCKET + BURDEN_BUCKET + QUEUE_BUCKET}

    burden_neg = min((weighted[k] for k in BURDEN_BUCKET if weighted[k] < 0), default=0.0)
    queue_neg = min((weighted[k] for k in QUEUE_BUCKET if weighted[k] < 0), default=0.0)
    other_neg = min((weighted[k] for k in OTHER_BUCKET if weighted[k] < 0), default=0.0)

    bucket_totals = {
        "burden": W_BURDEN * burden_neg,
        "queue": W_QUEUE * queue_neg,
        "other": W_OTHER * other_neg,
    }

    negative_total = sum(bucket_totals.values())

    positive_total = 0.0
    if negative_total > ABUNDANCE_GATE:
        positives = [weighted[k] for k in POSITIVE_ARM_SIGNALS if weighted[k] > 0]
        positive_total = max(positives, default=0.0)

    scalar = _clamp(negative_total + positive_total)

    return PressureBreakdown(
        scalar=scalar,
        signals=dict(signals),
        modifiers={"weights": dict(weights)},
        bucket_totals=bucket_totals,
        positive_total=positive_total,
        negative_total=negative_total,
    )
```

- [ ] **Step 4-5: PASS, Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/combine.py packages/nerd_herd/tests/test_combine.py
git commit -m "feat(nerd_herd): combine_signals — bucket worst-wins + gated abundance"
```

---

## Phase 6 — Wire It Up

### Task 22: `pressure_for` orchestrator + signature

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py` (`SystemSnapshot.pressure_for` rewrite)
- Modify: `packages/nerd_herd/src/nerd_herd/__init__.py` (re-export)
- Test: `packages/nerd_herd/tests/test_pressure_for.py` (extend or replace)

- [ ] **Step 1: Failing integration test**

```python
# packages/nerd_herd/tests/test_pressure_for.py — append
import pytest
from nerd_herd.types import (
    CloudModelState, CloudProviderState, LocalModelState, RateLimit,
    RateLimitMatrix, SystemSnapshot, QueueProfile,
)


class FakeModel:
    def __init__(self, name, provider, *, is_local=False, is_free=False, size_mb=0,
                 cap_score=5.0, capabilities=None):
        self.name = name
        self.provider = provider
        self.is_local = is_local
        self.is_free = is_free
        self.is_loaded = False
        self.size_mb = size_mb
        self.cap_score = cap_score
        self.capabilities = capabilities or set()


def test_pressure_for_full_path_returns_breakdown():
    """End-to-end smoke test: pressure_for with all 10 signals."""
    snap = SystemSnapshot(
        vram_available_mb=8000,
        local=LocalModelState(),
        cloud={
            "groq": CloudProviderState(
                provider="groq",
                limits=RateLimitMatrix(rpd=RateLimit(limit=14_400, remaining=14_000)),
                models={
                    "groq/llama": CloudModelState(
                        model_id="groq/llama",
                        limits=RateLimitMatrix(
                            rpm=RateLimit(limit=30, remaining=29),
                            tpm=RateLimit(limit=6000, remaining=5800),
                            rpd=RateLimit(limit=14_400, remaining=14_000),
                        ),
                    ),
                },
            ),
        },
        queue_profile=QueueProfile(
            total_ready_count=5, hard_tasks_count=1,
            by_difficulty={3: 4, 7: 1},
            by_capability={"function_calling": 5},
            projected_tokens=20_000, projected_calls=15,
        ),
    )
    model = FakeModel("groq/llama", "groq", is_free=True, cap_score=5.0)
    breakdown = snap.pressure_for(
        model,
        task_difficulty=5,
        est_per_call_tokens=2000,
        est_per_task_tokens=20_000,
        est_iterations=10,
        est_call_cost=0.0,
        cap_needed=5.0,
        consecutive_failures=0,
    )
    assert -1.0 <= breakdown.scalar <= 1.0
    # Smoke: all 10 signal keys populated
    assert all(k in breakdown.signals for k in
               ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11"))
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Rewrite `SystemSnapshot.pressure_for`**

In `packages/nerd_herd/src/nerd_herd/types.py:144+`, replace the existing `pressure_for` method:

```python
    def pressure_for(
        self,
        model,
        *,
        task_difficulty: int = 5,
        est_per_call_tokens: int = 0,
        est_per_task_tokens: int = 0,
        est_iterations: int = 1,
        est_call_cost: float = 0.0,
        cap_needed: float = 5.0,
        consecutive_failures: int = 0,
    ):
        """Compute pressure breakdown via 10 signals + 4 modifiers.

        Returns a PressureBreakdown (use .scalar for the scalar value).
        """
        from nerd_herd.breakdown import PressureBreakdown
        from nerd_herd.burn_log import get_burn_log
        from nerd_herd.combine import combine_signals
        from nerd_herd.modifiers import (
            M1_capacity_amplifier, M2_perishability_dampener, M3_difficulty_weights,
        )
        from nerd_herd.signals.s1_remaining import s1_remaining
        from nerd_herd.signals.s2_call_burden import s2_call_burden
        from nerd_herd.signals.s3_task_burden import s3_task_burden
        from nerd_herd.signals.s4_queue_tokens import s4_queue_tokens
        from nerd_herd.signals.s5_queue_calls import s5_queue_calls
        from nerd_herd.signals.s6_capable_supply import s6_capable_supply
        from nerd_herd.signals.s7_burn_rate import s7_burn_rate
        from nerd_herd.signals.s9_perishability import s9_perishability
        from nerd_herd.signals.s10_failure import s10_failure
        from nerd_herd.signals.s11_cost import s11_cost

        # Resolve matrix for this model (model-specific cell wins; provider is fallback)
        provider = getattr(model, "provider", "")
        prov = self.cloud.get(provider)
        model_state = (prov.models.get(getattr(model, "name", "")) if prov else None)
        matrix = (model_state.limits if model_state else
                  prov.limits if prov else
                  RateLimitMatrix())

        # Profile selection (free vs paid)
        profile = "time_bucketed" if getattr(model, "is_free", False) else "per_call"

        # Time-to-reset for the model's RPD cell (fall back to provider rpd)
        rpd_cell = matrix.rpd if matrix.rpd.limit else (
            prov.limits.rpd if prov else RateLimit()
        )
        import time as _time
        now = _time.time()
        reset_in = max(0.0, (rpd_cell.reset_at - now)) if rpd_cell.reset_at else 0.0

        # Total in-flight count for the model's pool
        in_flight_n = sum(
            1 for c in self.in_flight_calls
            if not c.is_local and c.provider == provider and c.model == getattr(model, "name", "")
        )

        # Compute signals
        sig = {
            "S1": s1_remaining(matrix, reset_in_secs=reset_in, in_flight=in_flight_n, profile=profile),
            "S2": s2_call_burden(matrix, est_per_call_tokens=est_per_call_tokens),
            "S3": s3_task_burden(matrix, est_per_task_tokens=est_per_task_tokens),
            "S4": s4_queue_tokens(matrix, queue=self.queue_profile or QueueProfile()),
            "S5": s5_queue_calls(matrix, queue=self.queue_profile or QueueProfile()),
            "S6": s6_capable_supply(model, queue=self.queue_profile or QueueProfile(),
                                    eligible_models=[], iter_avg=float(est_iterations or 8)),
            "S7": s7_burn_rate(matrix, provider=provider, model=getattr(model, "name", ""),
                               burn_log=get_burn_log(), now=now),
            "S9": s9_perishability(model, local=self.local, vram_avail_mb=self.vram_available_mb,
                                   matrix=matrix, task_difficulty=task_difficulty, now=now),
            "S10": s10_failure(consecutive_failures=consecutive_failures),
            "S11": s11_cost(est_call_cost=est_call_cost,
                            daily_cost_remaining=(matrix.cpd.remaining or 0.0)),
        }

        # Modifiers
        weights = M3_difficulty_weights(
            difficulty=task_difficulty,
            model_is_paid=not getattr(model, "is_free", False) and not getattr(model, "is_local", False),
        )

        # Apply M1 to negative-arm signals: amplify by limit-aware factor
        # Use the smallest-limit populated cell as the amplifier basis (worst-axis-wins)
        smallest_limit = min(
            (rl.limit for _, rl in matrix.populated_cells() if rl.limit), default=100,
        )
        m1 = M1_capacity_amplifier(limit=smallest_limit)
        for k in ("S1", "S2", "S3", "S4", "S5"):
            if sig[k] < 0:
                sig[k] *= m1

        # Apply M2 to positive S9 (over-qualification dampener, perishability-conditional)
        fit_excess = max(0.0, getattr(model, "cap_score", 5.0) - cap_needed)
        m2 = M2_perishability_dampener(fit_excess=fit_excess, s9_value=sig["S9"])
        if sig["S9"] > 0:
            sig["S9"] *= m2

        breakdown = combine_signals(signals=sig, weights=weights)
        breakdown.modifiers = {"M1": m1, "M2": m2, "M3_difficulty": task_difficulty}
        return breakdown
```

The old `pressure_for(self, model)` returning a float is gone. All callers must be updated (next tasks).

- [ ] **Step 4: Run, verify PASS**

```bash
timeout 60 pytest packages/nerd_herd/ -v
```
Expected: PASS for the integration test; existing pressure_for tests will FAIL — they call the old signature. Update them in the next task.

- [ ] **Step 5: Commit (interim — broken state acceptable for next tasks)**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/tests/test_pressure_for.py
git commit -m "feat(nerd_herd): rewrite pressure_for as 10-signal orchestrator (callers TBD next)"
```

---

### Task 23: Update `ranking.py` utilization layer

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py`
- Modify: `packages/fatih_hoca/src/fatih_hoca/scarcity.py` — DELETE
- Test: `packages/fatih_hoca/tests/test_ranking_utilization.py` (new or update)

- [ ] **Step 1: Find all `pool_scarcity` and old-signature `pressure_for` callers**

```bash
grep -rn "pool_scarcity\|snapshot.pressure_for\b" packages/ src/ tests/
```

- [ ] **Step 2: Update `_apply_utilization_layer` in ranking.py**

In `packages/fatih_hoca/src/fatih_hoca/ranking.py`, find the function that calls `pool_scarcity` (search for `pool_scarcity`). Replace its usage:

```python
        # Old:
        # scarcity = pool_scarcity(model, snapshot, task_difficulty=...)
        # composite *= 1 + UTILIZATION_K * scarcity * (1 - max(0, fit_excess))

        # New:
        from fatih_hoca.estimates import estimate_for
        from fatih_hoca.capability_curve import CAP_NEEDED_BY_DIFFICULTY
        estimates = estimate_for(task, btable=btable, model_is_thinking=getattr(model, "is_thinking", False))
        prov_state = snapshot.cloud.get(getattr(model, "provider", ""))
        breakdown = snapshot.pressure_for(
            model,
            task_difficulty=task_difficulty,
            est_per_call_tokens=estimates.per_call_tokens,
            est_per_task_tokens=estimates.total_tokens,
            est_iterations=estimates.iterations,
            est_call_cost=getattr(model, "estimated_cost", lambda *_: 0.0)(estimates.in_tokens, estimates.out_tokens),
            cap_needed=CAP_NEEDED_BY_DIFFICULTY.get(task_difficulty, 5.0),
            consecutive_failures=getattr(prov_state, "consecutive_failures", 0) if prov_state else 0,
        )
        composite *= 1 + UTILIZATION_K * breakdown.scalar
```

The `(1 - max(0, fit_excess))` damper is removed — M2 absorbs it inside `pressure_for`.

`btable` is fetched from a module-level cache (loaded from `step_token_stats` table — see Task 25 for the rollup; for now empty dict is acceptable as cold-start).

- [ ] **Step 3: Delete `scarcity.py`**

```bash
git rm packages/fatih_hoca/src/fatih_hoca/scarcity.py
```

If `scarcity.py` is imported anywhere else, those imports must be removed.

- [ ] **Step 4: Run fatih_hoca tests**

```bash
timeout 120 pytest packages/fatih_hoca/ -v
```
Expected: PASS (some old scarcity tests removed, ranking tests updated).

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/
git commit -m "refactor(fatih_hoca): ranking uses pressure_for breakdown; scarcity.py deleted"
```

---

### Task 24: Update Beckman admission gate signature

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (next_task admission code, around lines 100-200)
- Test: `packages/general_beckman/tests/test_next_task_admission.py` (update)

- [ ] **Step 1: Find the existing admission call**

```bash
grep -n "snapshot.pressure_for\|pressure_for" packages/general_beckman/src/general_beckman/__init__.py
```

- [ ] **Step 2: Update the call site**

Locate the admission loop (around line 200 in `__init__.py`). Replace the existing pressure call with:

```python
                from fatih_hoca.estimates import estimate_for
                # btable: dict module-level cache (empty in MVP cold-start; refreshed by Task 25 rollup)
                from general_beckman.btable_cache import get_btable
                estimates = estimate_for(task, btable=get_btable(),
                                         model_is_thinking=getattr(pick.model, "is_thinking", False))
                breakdown = snap.pressure_for(
                    pick.model,
                    task_difficulty=difficulty,
                    est_per_call_tokens=estimates.per_call_tokens,
                    est_per_task_tokens=estimates.total_tokens,
                    est_iterations=estimates.iterations,
                    est_call_cost=getattr(pick.model, "estimated_cost", lambda *_: 0.0)(
                        estimates.in_tokens, estimates.out_tokens
                    ),
                    cap_needed=5.0,
                    consecutive_failures=0,  # populated below from cloud state
                )
                pressure = breakdown.scalar
                _log.info(
                    "admission_decision",
                    extra={
                        "task_id": task["id"],
                        "model": pick.model.name,
                        "pressure": pressure,
                        "urgency": urgency,
                        "threshold": threshold(urgency),
                        "admitted": pressure >= threshold(urgency),
                        "breakdown": breakdown.to_dict(),
                    },
                )
                if pressure >= threshold(urgency):
                    # ... admit logic unchanged
```

- [ ] **Step 3: Add tiny `btable_cache.py`**

```python
# packages/general_beckman/src/general_beckman/btable_cache.py
"""Module-level B-table cache. Refreshed by btable_rollup cron (Task 25)."""
from __future__ import annotations

_BTABLE: dict[tuple[str, str, str], dict] = {}


def get_btable() -> dict:
    return _BTABLE


def set_btable(rows: dict) -> None:
    global _BTABLE
    _BTABLE = rows
```

- [ ] **Step 4: Run beckman tests**

```bash
timeout 60 pytest packages/general_beckman/ -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/
git commit -m "refactor(beckman): admission gate uses pressure_for breakdown + btable cache"
```

---

### Task 25: queue_profile_push — dep-resolution + projections

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/queue_profile_push.py`
- Test: `packages/general_beckman/tests/test_queue_profile_dep_resolution.py` (new)

- [ ] **Step 1: Failing tests**

```python
# packages/general_beckman/tests/test_queue_profile_dep_resolution.py
import json
import os
import tempfile
from pathlib import Path

import pytest
import aiosqlite


@pytest.mark.asyncio
async def test_queue_profile_excludes_blocked_tasks(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute(
                """CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY, status TEXT, agent_type TEXT,
                    next_retry_at TIMESTAMP, depends_on TEXT, completed_at TIMESTAMP,
                    context TEXT
                )"""
            )
            # Two pending: t1 unblocked, t2 depends on t1 (still pending → blocked)
            await conn.execute(
                "INSERT INTO tasks (id, status, agent_type, depends_on, context) VALUES (1, 'pending', 'analyst', '[]', '{\"workflow_step_id\":\"s1\",\"workflow_phase\":\"p1\",\"difficulty\":5}')"
            )
            await conn.execute(
                "INSERT INTO tasks (id, status, agent_type, depends_on, context) VALUES (2, 'pending', 'analyst', '[1]', '{\"difficulty\":7}')"
            )
            await conn.commit()
        os.environ["DB_PATH"] = str(db_path)
        from general_beckman.queue_profile_push import build_profile
        profile = await build_profile(str(db_path))
        # only t1 unblocked
        assert profile.total_ready_count == 1


@pytest.mark.asyncio
async def test_queue_profile_includes_dep_resolved(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute(
                """CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY, status TEXT, agent_type TEXT,
                    next_retry_at TIMESTAMP, depends_on TEXT, completed_at TIMESTAMP,
                    context TEXT
                )"""
            )
            await conn.execute(
                "INSERT INTO tasks VALUES (1, 'completed', 'analyst', NULL, '[]', datetime('now'), '{}')"
            )
            await conn.execute(
                "INSERT INTO tasks VALUES (2, 'pending', 'analyst', NULL, '[1]', NULL, '{\"workflow_step_id\":\"s2\",\"workflow_phase\":\"p2\",\"difficulty\":7}')"
            )
            await conn.commit()
        from general_beckman.queue_profile_push import build_profile
        profile = await build_profile(str(db_path))
        assert profile.total_ready_count == 1
        assert profile.hard_tasks_count == 1
        # difficulty bucket
        assert profile.by_difficulty.get(7) == 1


@pytest.mark.asyncio
async def test_queue_profile_projects_tokens_and_calls(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute(
                """CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY, status TEXT, agent_type TEXT,
                    next_retry_at TIMESTAMP, depends_on TEXT, completed_at TIMESTAMP,
                    context TEXT
                )"""
            )
            await conn.execute(
                "INSERT INTO tasks VALUES (1, 'pending', 'analyst', NULL, '[]', NULL, '{\"workflow_step_id\":\"4.5b\",\"workflow_phase\":\"phase_4\",\"difficulty\":7}')"
            )
            await conn.commit()
        from general_beckman.queue_profile_push import build_profile
        profile = await build_profile(str(db_path))
        # 4.5b is in STEP_TOKEN_OVERRIDES with iters=12, in=10k, out=100k
        # projected_tokens ≈ (10k+100k)*12 = 1.32M
        assert profile.projected_tokens >= 1_000_000
        assert profile.projected_calls >= 12
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Implement**

Replace the contents of `packages/general_beckman/src/general_beckman/queue_profile_push.py`:

```python
"""Build current QueueProfile from queue tables and push to nerd_herd.

Latency target: <5 ms per push. Profile build runs every 2-3 seconds.
"""
from __future__ import annotations

import json
import os
import time

import aiosqlite

from nerd_herd.types import QueueProfile


# In-process completed-id cache (TTL 30s; invalidated on task completion via on_finish)
_COMPLETED_IDS: set[int] = set()
_COMPLETED_AT: float = 0.0
_CACHE_TTL_SECS = 30.0


async def _refresh_completed_ids(db_path: str) -> set[int]:
    global _COMPLETED_IDS, _COMPLETED_AT
    now = time.time()
    if now - _COMPLETED_AT < _CACHE_TTL_SECS and _COMPLETED_IDS:
        return _COMPLETED_IDS
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT id FROM tasks WHERE status='completed' "
                "AND (completed_at IS NULL OR completed_at > datetime('now', '-7 days'))"
            ) as cur:
                rows = await cur.fetchall()
        _COMPLETED_IDS = {int(r[0]) for r in rows}
        _COMPLETED_AT = now
    except Exception:
        # On error, retain previous cache (don't blow away)
        pass
    return _COMPLETED_IDS


def invalidate_completed_id_cache(task_id: int) -> None:
    """Hook for on_task_finished: add id without forcing a DB read."""
    _COMPLETED_IDS.add(int(task_id))


_NEEDS_VISION_AGENTS = {"visual_reviewer"}
_NEEDS_THINKING_AGENTS = {"analyst", "architect", "planner", "reviewer"}
_NEEDS_TOOLS_AGENTS = {"analyst", "implementer", "executor", "researcher", "test_generator", "fixer", "coder"}


async def build_profile(db_path: str | None = None) -> QueueProfile:
    db_path = db_path or os.environ.get("DB_PATH", "kutai.db")
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """SELECT id, agent_type, depends_on, context
               FROM tasks
               WHERE status='pending'
                 AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))"""
        ) as cur:
            ready_rows = await cur.fetchall()

    completed = await _refresh_completed_ids(db_path)

    # Lazy import to avoid cycle at module load
    from fatih_hoca.estimates import Estimates, estimate_for

    class _TaskShim:
        def __init__(self, agent_type, ctx):
            self.agent_type = agent_type
            self.context = ctx

    unblocked: list[_TaskShim] = []
    for tid, agent_type, deps_json, ctx_json in ready_rows:
        try:
            deps = json.loads(deps_json or "[]")
        except Exception:
            deps = []
        if not all(int(d) in completed for d in deps):
            continue
        try:
            ctx = json.loads(ctx_json or "{}")
        except Exception:
            ctx = {}
        unblocked.append(_TaskShim(agent_type, ctx))

    by_difficulty: dict[int, int] = {}
    by_capability: dict[str, int] = {"vision": 0, "thinking": 0, "function_calling": 0}
    projected_tokens = 0
    projected_calls = 0
    hard = 0

    for shim in unblocked:
        ctx = shim.context if isinstance(shim.context, dict) else {}
        d = ctx.get("difficulty")
        if d is None and "classification" in ctx and isinstance(ctx["classification"], dict):
            d = ctx["classification"].get("difficulty")
        d = int(d or 5)
        by_difficulty[d] = by_difficulty.get(d, 0) + 1
        if d >= 7:
            hard += 1
        if shim.agent_type in _NEEDS_VISION_AGENTS:
            by_capability["vision"] += 1
        if shim.agent_type in _NEEDS_THINKING_AGENTS:
            by_capability["thinking"] += 1
        if shim.agent_type in _NEEDS_TOOLS_AGENTS:
            by_capability["function_calling"] += 1
        e = estimate_for(shim, btable={})
        projected_tokens += e.total_tokens
        projected_calls += e.iterations

    return QueueProfile(
        total_ready_count=len(unblocked),
        hard_tasks_count=hard,
        by_difficulty=by_difficulty,
        by_capability=by_capability,
        projected_tokens=projected_tokens,
        projected_calls=projected_calls,
    )


async def build_and_push(db_path: str | None = None) -> None:
    """Fire-and-forget: build profile and push to nerd_herd. Swallows exceptions."""
    try:
        profile = await build_profile(db_path)
    except Exception:
        return
    try:
        import nerd_herd
        nerd_herd.push_queue_profile(profile)
    except Exception:
        return
```

- [ ] **Step 4: Wire `invalidate_completed_id_cache` into `on_task_finished`**

In `packages/general_beckman/src/general_beckman/__init__.py`, in the `on_task_finished` function, add:

```python
    try:
        from general_beckman.queue_profile_push import invalidate_completed_id_cache
        invalidate_completed_id_cache(task_id)
    except Exception:
        pass
```

- [ ] **Step 5: Run tests, verify PASS**

```bash
timeout 60 pytest packages/general_beckman/tests/test_queue_profile_dep_resolution.py -v
```

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/
git commit -m "feat(beckman): queue_profile_push with dep-resolution + token/call projections"
```

---

## Phase 7 — Rollup, Cleanup, Simulator

### Task 26: B-table rollup job + Beckman cron

**Files:**
- Create: `packages/general_beckman/src/general_beckman/btable_rollup.py`
- Modify: Beckman scheduled_jobs registration in main entry
- Test: `packages/general_beckman/tests/test_btable_rollup.py`

- [ ] **Step 1: Tests**

```python
# packages/general_beckman/tests/test_btable_rollup.py
import os
import tempfile
from pathlib import Path

import pytest
import aiosqlite


@pytest.mark.asyncio
async def test_rollup_writes_step_token_stats():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        os.environ["DB_PATH"] = str(db_path)
        import src.infra.db as db_mod
        db_mod._db = None
        await db_mod.init_db()
        # Seed model_call_tokens with 10 rows for one (agent, step, phase) key
        async with aiosqlite.connect(str(db_path)) as conn:
            for i in range(10):
                await conn.execute(
                    """INSERT INTO model_call_tokens (
                        agent_type, workflow_step_id, workflow_phase, call_category,
                        model, provider, is_streaming, prompt_tokens, completion_tokens,
                        total_tokens, duration_ms, iteration_n, success
                    ) VALUES ('analyst','3.5','phase_3','main_work','gpt','openai',0, ?, ?, ?, 1000, ?, 1)""",
                    (1000 + i*100, 2000 + i*100, 3000 + i*200, i + 1),
                )
            await conn.commit()
        from general_beckman.btable_rollup import run_rollup
        rows_written = await run_rollup(str(db_path))
        assert rows_written >= 1
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT samples_n, in_p90 FROM step_token_stats WHERE agent_type='analyst' AND workflow_step_id='3.5'"
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == 10
        assert row[1] > 0
```

- [ ] **Step 2-3: Implement**

```python
# packages/general_beckman/src/general_beckman/btable_rollup.py
"""B-table rollup: model_call_tokens → step_token_stats.

Hourly cron registered in Beckman scheduled_jobs. 14-day rolling window.
Computes p50/p90/p99 in Python (SQLite lacks percentile_disc).
"""
from __future__ import annotations

import os
import statistics

import aiosqlite

from general_beckman.btable_cache import set_btable


WINDOW_DAYS = 14


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


async def run_rollup(db_path: str | None = None) -> int:
    db_path = db_path or os.environ.get("DB_PATH", "kutai.db")
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            f"""SELECT agent_type, workflow_step_id, workflow_phase,
                       prompt_tokens, completion_tokens, iteration_n
                FROM model_call_tokens
                WHERE timestamp > datetime('now', '-{WINDOW_DAYS} days')
                  AND is_streaming = 0
                  AND agent_type IS NOT NULL
                  AND workflow_step_id IS NOT NULL
                  AND workflow_phase IS NOT NULL"""
        ) as cur:
            rows = await cur.fetchall()

    # Group per key
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    iter_max: dict[tuple[str, str, str], dict[int, int]] = {}
    for agent_type, step_id, phase, in_tok, out_tok, iter_n in rows:
        key = (agent_type, step_id, phase)
        bucket = grouped.setdefault(key, {"in": [], "out": []})
        bucket["in"].append(float(in_tok or 0))
        bucket["out"].append(float(out_tok or 0))
        # Track max iter per task — approximate by iter_n value
        # (multi-iteration calls share task_id; we don't fetch task_id here so we
        #  use max iter_n observed per (agent, step, phase) as the proxy.)
        per_task = iter_max.setdefault(key, {})
        per_task[iter_n] = per_task.get(iter_n, 0) + 1

    rows_written = 0
    btable_dict: dict[tuple[str, str, str], dict] = {}
    async with aiosqlite.connect(db_path) as db:
        for key, vals in grouped.items():
            in_sorted = sorted(vals["in"])
            out_sorted = sorted(vals["out"])
            samples_n = len(out_sorted)
            iter_n_vals = sorted(iter_max.get(key, {1: 1}).keys())
            iters_p50 = _percentile([float(v) for v in iter_n_vals], 0.50)
            iters_p90 = _percentile([float(v) for v in iter_n_vals], 0.90)
            iters_p99 = _percentile([float(v) for v in iter_n_vals], 0.99)
            row = {
                "samples_n": samples_n,
                "in_p50": int(_percentile(in_sorted, 0.50)),
                "in_p90": int(_percentile(in_sorted, 0.90)),
                "in_p99": int(_percentile(in_sorted, 0.99)),
                "out_p50": int(_percentile(out_sorted, 0.50)),
                "out_p90": int(_percentile(out_sorted, 0.90)),
                "out_p99": int(_percentile(out_sorted, 0.99)),
                "iters_p50": iters_p50,
                "iters_p90": iters_p90,
                "iters_p99": iters_p99,
            }
            await db.execute(
                """INSERT INTO step_token_stats
                    (agent_type, workflow_step_id, workflow_phase,
                     samples_n, in_p50, in_p90, in_p99, out_p50, out_p90, out_p99,
                     iters_p50, iters_p90, iters_p99, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(agent_type, workflow_step_id, workflow_phase) DO UPDATE SET
                     samples_n=excluded.samples_n,
                     in_p50=excluded.in_p50, in_p90=excluded.in_p90, in_p99=excluded.in_p99,
                     out_p50=excluded.out_p50, out_p90=excluded.out_p90, out_p99=excluded.out_p99,
                     iters_p50=excluded.iters_p50, iters_p90=excluded.iters_p90, iters_p99=excluded.iters_p99,
                     updated_at=datetime('now')""",
                (key[0], key[1], key[2], samples_n,
                 row["in_p50"], row["in_p90"], row["in_p99"],
                 row["out_p50"], row["out_p90"], row["out_p99"],
                 iters_p50, iters_p90, iters_p99),
            )
            rows_written += 1
            btable_dict[key] = row
        await db.commit()

    set_btable(btable_dict)  # refresh in-memory cache for admission gate
    return rows_written
```

- [ ] **Step 4: Register cron in Beckman scheduled_jobs**

In `packages/general_beckman/src/general_beckman/__init__.py` (or wherever scheduled_jobs registration happens), add an hourly job that calls `run_rollup()`. Pattern depends on existing scheduled_jobs structure — typically a tuple `(name, interval_secs, callable)`. Adapt to existing convention:

```python
# In Beckman startup or scheduled_jobs registration:
from general_beckman.btable_rollup import run_rollup

# 3600s = hourly
beckman.register_job("btable_rollup", interval_secs=3600, callback=run_rollup)
```

If the project doesn't have an explicit scheduled_jobs API, run rollup at startup AND on each `on_task_finished` (cheap when grouped is small).

- [ ] **Step 5: PASS, Step 6: Commit**

```bash
git add packages/general_beckman/
git commit -m "feat(beckman): B-table rollup cron — model_call_tokens → step_token_stats"
```

---

### Task 27: Delete `pool_pressure.py` legacy + tidy

**Files:**
- Delete: `packages/nerd_herd/src/nerd_herd/pool_pressure.py` (or trim to thin wrapper)
- Modify: `packages/nerd_herd/src/nerd_herd/__init__.py`
- Update: any remaining test files referencing `compute_pool_pressure` or `PoolPressure` directly

- [ ] **Step 1: Find references**

```bash
grep -rn "compute_pool_pressure\|PoolPressure\b" packages/ src/ tests/
```

- [ ] **Step 2: Remove unused imports + delete file**

If `compute_pool_pressure` is not used anywhere outside its own tests + `types.py:144` (which we already replaced):

```bash
git rm packages/nerd_herd/src/nerd_herd/pool_pressure.py
git rm packages/nerd_herd/tests/test_pool_pressure.py packages/nerd_herd/tests/test_pressure_for.py 2>/dev/null || true
```

If `PoolPressure` dataclass is referenced (e.g. as `CloudModelState.pool_pressure`), keep the dataclass but move it to `types.py`.

- [ ] **Step 3: Update `__init__.py`**

In `packages/nerd_herd/src/nerd_herd/__init__.py`, remove the `compute_pool_pressure` and `PoolPressure` re-exports if they no longer exist. Add:

```python
from nerd_herd.breakdown import PressureBreakdown
from nerd_herd.signals.s1_remaining import s1_remaining
# ... etc, or leave as importable from nerd_herd.signals.* path
```

- [ ] **Step 4: Run full test suite**

```bash
timeout 300 pytest packages/nerd_herd/ packages/fatih_hoca/ packages/general_beckman/ packages/kuleden_donen_var/ packages/hallederiz_kadir/ -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/
git commit -m "refactor(nerd_herd): remove legacy pool_pressure module"
```

---

### Task 28: Simulator scenarios + acceptance gate

**Files:**
- Modify: `packages/fatih_hoca/tests/sim/scenarios.py` (or wherever Phase 2d sim lives)
- Modify: `packages/fatih_hoca/tests/sim/run_scenarios.py`

- [ ] **Step 1: Add 8 new scenarios**

In `packages/fatih_hoca/tests/sim/scenarios.py`, append:

```python
def _make_pool_pressure_scenario_1_fat_vs_tiny():
    """Two pools, 50% remaining each. 10-RPD vs 1000-RPD; tiny should empty faster."""
    return Scenario(
        name="pp_fat_vs_tiny",
        # Setup: provider A 10 RPD limit / 5 remaining; provider B 1000 / 500
        # Run a 20-task queue, mostly cloud-bound, mid difficulty.
        # Assert: A's effective drop rate (per task admitted) is greater than B's
        ...
    )

def _make_pool_pressure_scenario_2_token_filter():
    """30k call vs 25k tpm; model excluded by token-fit filter."""
    ...

def _make_pool_pressure_scenario_3_cold_local_vram():
    """Cold local + free VRAM admits default-urgency task."""
    ...

def _make_pool_pressure_scenario_4_free_cloud_burn():
    """Free cloud near reset + flush quota wins admissions on easy work."""
    ...

def _make_pool_pressure_scenario_5_paid_cloud_reserve():
    """Paid cloud flush + no hard queue. M2 dampens; local wins."""
    ...

def _make_pool_pressure_scenario_6_capability_shortage():
    """50 vision tasks, only Gemini+gpt5 capable, combined 80 RPD. S6 fires -1."""
    ...

def _make_pool_pressure_scenario_7_difficulty_lookahead():
    """8 d=9 tasks queued; current d=3 candidate. Cloud reserved; local wins."""
    ...

def _make_pool_pressure_scenario_8_full_mission():
    """30-task mission across d=3/5/7/9; ASSERT: cloud RPD never exhausts before reset;
    local never sits idle while cloud has flush quota; no pool > 80% utilized."""
    ...


POOL_PRESSURE_SCENARIOS = [
    _make_pool_pressure_scenario_1_fat_vs_tiny(),
    _make_pool_pressure_scenario_2_token_filter(),
    _make_pool_pressure_scenario_3_cold_local_vram(),
    _make_pool_pressure_scenario_4_free_cloud_burn(),
    _make_pool_pressure_scenario_5_paid_cloud_reserve(),
    _make_pool_pressure_scenario_6_capability_shortage(),
    _make_pool_pressure_scenario_7_difficulty_lookahead(),
    _make_pool_pressure_scenario_8_full_mission(),
]
```

Each scenario constructor must follow the existing `Scenario` API conventions in this file. Concrete setup blocks depend on the simulator's scaffolding — read `run_scenarios.py` to understand how scenarios construct snapshots, run iterations, and assert outcomes. Implement each scenario as a small unit test that asserts the expected emergent behavior.

- [ ] **Step 2: Update `run_scenarios.py` to include new scenarios**

```python
# In run_scenarios.py main:
from packages.fatih_hoca.tests.sim.scenarios import POOL_PRESSURE_SCENARIOS

ALL_SCENARIOS = EXISTING_SCENARIOS + POOL_PRESSURE_SCENARIOS

failures = []
for scenario in ALL_SCENARIOS:
    try:
        scenario.run()
        print(f"[PASS] {scenario.name}")
    except AssertionError as e:
        failures.append((scenario.name, str(e)))
        print(f"[FAIL] {scenario.name}: {e}")

if failures:
    print(f"\n{len(failures)} scenarios failed.")
    sys.exit(1)
print("All scenarios passed.")
```

- [ ] **Step 3: Run scenarios**

```bash
timeout 300 python packages/fatih_hoca/tests/sim/run_scenarios.py
```
Expected: all PASS. **Scenario 8 is the merge-acceptance gate** — if it fails, calibrate weights before merging.

- [ ] **Step 4: Run full test suite once more**

```bash
timeout 300 pytest packages/ tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/tests/sim/
git commit -m "test(sim): add 8 pool-pressure equilibrium scenarios"
```

---

## Self-Review Checklist

After completing all 28 tasks, verify:

- [ ] **Spec coverage:** Every section of the spec has a task implementing it. Specifically:
  - §3 Architecture (Tier 1/2/3) → Tasks 22 (orchestrator) + 23 (utilization integration)
  - §4 Signals S1-S11 → Tasks 10-19
  - §5 Modifiers M1-M3 → Task 20 (M4 unchanged in admission.py)
  - §6 Estimates → Tasks 4-6
  - §7 Token logging → Tasks 1-3
  - §8 Queue profile → Task 25
  - §9 RateLimitMatrix → Task 7-8
  - §10 Combination → Task 21
  - §11 Beckman/Hoca consumption → Tasks 23-24
  - §12 Migration → Task order matches spec §12
  - §13 Testing → Tests in every task + Task 28
  - §14 Calibration → Task 26 rollup is the data feed; calibration scripts deferred to Phase 2

- [ ] **Placeholder scan:** No "TBD", "implement later", "similar to Task X" without code. Each step has runnable code or commands.

- [ ] **Type consistency:**
  - `RateLimitMatrix` (not `RateLimits`) used everywhere after Task 7
  - `Estimates` dataclass shape consistent across Tasks 4, 5, 6, 23, 24, 25
  - `PressureBreakdown` shape consistent: `signals`, `modifiers`, `bucket_totals`, `positive_total`, `negative_total`, `scalar`
  - `pressure_for` signature consistent: `(model, *, task_difficulty, est_per_call_tokens, est_per_task_tokens, est_iterations, est_call_cost, cap_needed, consecutive_failures)`

- [ ] **No spec gaps:** Spec §15 open items already flagged in task notes (per-provider parser PRs, auto-calibration script, etc.).

---

## Execution Notes

### Risks during execution

- **Task 22 leaves system briefly broken.** `pressure_for` rewrite must be followed by Tasks 23 + 24 to update callers. Don't deploy between these tasks.
- **Task 7 rename touches many files.** Use grep + sed; verify with full test suite before commit.
- **Task 23 deletes `scarcity.py`** — if anything else imports it (check earlier `grep -rn`), update those imports first.
- **Task 25's `estimate_for(shim, btable={})` cold-start** — falls through to `STEP_TOKEN_OVERRIDES` then `AGENT_REQUIREMENTS`; expected behavior, not a bug. B-table populates after Task 26's first cron run.

### Performance verification points

- After Task 25: `build_profile` should return in <5ms for n<200 pending tasks. Add timing assertion in test if needed.
- After Task 22: `pressure_for` should return in <1ms (no I/O, snapshot-derived). Add timing assertion.

### Calibration deferred

Phase 0 calibration of S9 max values, M3 weight matrix, bucket weights — seeded from analytical reasoning. Real tuning happens during Phase 1 soak window using `model_pick_log.snapshot_summary` breakdowns (post-merge work, not in this plan).
