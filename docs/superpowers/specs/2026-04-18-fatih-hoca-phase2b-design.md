# Fatih Hoca Phase 2b — Design Spec

**Date:** 2026-04-18
**Status:** Approved for planning
**Predecessors:** Phase 1 (selection-intelligence), Phase 2a (hygiene)
**Worktree:** `.worktrees/fatih-hoca-phase2b`
**Branch:** `feat/fatih-hoca-phase2b`

## Goal

Land the data-free infrastructure for selection-intelligence tuning, plus a
synthetic-workload simulator that previews task/model distribution without
waiting two weeks of real telemetry.

## Scope

Four independent items, ordered by dependency:

1. Scheduled benchmark refresh
2. Fire-and-forget asyncio audit
3. `/bench_picks` Telegram command
4. i2p dry-run simulator

Out of scope: weight rebalance, grading-derived perf_score, phantom "a" model
archaeology, ChromaDB `Error finding id`.

---

## Item 1 — Scheduled Benchmark Refresh

### What
Add `tick_benchmark_refresh()` sibling in `src/app/scheduled_jobs.py`. Wired
into the orchestrator heartbeat.

### Behavior
- **Skip when fresh:** If `.benchmark_cache/_bulk_*.json` max mtime is < 24h
  old, return immediately with DEBUG log.
- **Kick refresh:** Otherwise, call `src.models.benchmark.benchmark_cli.refresh_all()`
  asynchronously (offloaded via `asyncio.to_thread` or similar, since the CLI
  is sync).
- **Coverage delta:** Before refresh, count matched entries; after refresh,
  count again. Log at INFO: `benchmark refresh: matched N→M (+K)`.
- **Idempotent:** Module-level `_refresh_in_flight: bool` flag prevents
  overlapping refreshes when the tick fires again mid-run.
- **Error handling:** All exceptions caught; log at WARNING with traceback;
  tick never crashes the orchestrator.

### Test
- Monkeypatch `time.time()` / filesystem mtime to simulate fresh vs stale
  cache, spy on `refresh_all()` invocation count.
- Simulate an in-flight refresh; verify second tick is noop.
- Inject an exception from `refresh_all()`; verify WARNING logged, tick
  returns cleanly.

---

## Item 2 — Fire-and-Forget Asyncio Audit

### What
Find every `asyncio.create_task(...)` call with no assignment (task reference
dropped immediately → GC-reapable mid-flight). Apply the Phase 2a Task 0
pattern: module-level `_in_flight: set[asyncio.Task]`, add on create, remove
on done-callback.

### Known Sites
- `src/tools/web_search.py:252` — `_record_fetch_quality_fire_and_forget`
- `packages/fatih_hoca/src/fatih_hoca/selector.py:314` — pick telemetry `_write()`

### Method
1. Grep `asyncio.create_task\(` across `src/` and `packages/` for content mode.
2. Filter to unassigned usages (no `= ` before the call, no append to a list).
3. For each site, add module-level task set with done-callback cleanup.

### Test
- For each fixed site: schedule multiple tasks rapidly, force GC, assert all
  tasks still ran to completion (no "Task was destroyed but it is pending"
  warnings in captured logs).

---

## Item 3 — `/bench_picks` Telegram Command

### What
New command handler in `src/app/telegram_bot.py` that surfaces the
`model_pick_log` aggregation without opening sqlite.

### Query
```sql
SELECT task_name, picked_model, COUNT(*) AS n, AVG(picked_score) AS avg_score
FROM model_pick_log
WHERE timestamp > datetime('now', '-7 days')
GROUP BY task_name, picked_model
ORDER BY task_name, n DESC;
```

### Output
Monospace table formatted into a code block. If result set is large, cap at
~40 rows and note truncation. If empty, reply
`No pick log entries in last 7 days.`

### Permissions
Follow existing command-handler patterns; use `_reply()` helper to preserve
`REPLY_KEYBOARD`.

### Test
- Seed `model_pick_log` in a tmp sqlite; invoke handler with a fake Telegram
  update; assert reply contains expected rows.
- Empty DB → assert "no entries" message.
- >40 distinct (task, model) pairs → assert truncation note.

---

## Item 4 — i2p Dry-Run Simulator

### What
New CLI: `python -m packages.fatih_hoca.simulate_i2p` (module:
`packages/fatih_hoca/src/fatih_hoca/simulate_i2p.py`).

Walks the 182 steps of `src/workflows/i2p/i2p_v3.json`, feeds each through
`Selector.select()` against a fresh Selector instance with a pinned fake
snapshot, and reports the task × model distribution.

### Key Design Decisions

**Dry-run is implicit.** Telemetry is opt-in (`enable_telemetry()`); the
simulator simply does not call it. DB writes silently no-op.

**Fresh Selector per run.** `Selector._swap_budget` is stateful — a reused
production selector would mutate. Construct a new `Selector(registry,
FakeNerdHerd())` inside the simulator.

**Pinned snapshot.** Live VRAM jitter makes results non-reproducible. Fake
Nerd Herd returns a fixed snapshot: plausible VRAM free (e.g. 7000 MB), no
loaded local model, inference metrics empty. A future `--loaded MODEL` flag
can override, but MVP uses a single pinned scenario.

**Difficulty mapping.** i2p uses `"easy"|"medium"|"hard"` strings; Selector
expects int 1-10. Map: easy→3, medium→5, hard→8.

### Inputs per Step
From each `steps[i]` entry:
- `agent` → `agent_type`
- `difficulty` (mapped) → `difficulty`
- `tools_hint` contains `"vision"` or similar? → for MVP, default
  `needs_function_calling=True, needs_vision=False, needs_thinking=True`
- `task` = `name`
- `call_category="main_work"`

### Aggregation
For each step, record: `step_id`, `task_name`, `agent`, `difficulty`,
`picked_model` (or `"<none>"` if `select()` returned None), `picked_score`,
`top3_candidates` (list of `(model, score)`).

### Output

**Stdout (always):**
- Header summary: total steps, unique picks, coverage (steps with non-None pick).
- Table 1 — pick distribution: `model | count | pct` (sorted by count desc).
- Table 2 — by agent: `agent | top_model | n_steps`.
- Table 3 — by difficulty: `difficulty | top_model | n_steps`.

**JSON (optional, `--json <path>`):** full per-step records as JSON array,
suitable for diffing between runs after weight changes.

### CLI
```
python -m packages.fatih_hoca.simulate_i2p [--json OUT.json] [--workflow PATH]
```
`--workflow` defaults to `src/workflows/i2p/i2p_v3.json`.

### Test
- Unit test: instantiate simulator with a tiny fake workflow (3 steps) and
  an in-memory registry of 2 models; assert aggregation counts match
  expectations.
- Smoke test: run against real `i2p_v3.json` with the real registry; assert
  coverage > 90% (most steps pick something) and exits 0. No assertions on
  specific model names (those shift with registry).

---

## Non-Functional Constraints

- **No `pip install -e`** — the shared venv is pinned to main. Tests invoke
  via editable install already in place.
- **TDD per item** — failing test first, implementation, green, commit.
- **Commit after each green item** — four commits minimum on the feature
  branch, messages: `feat(fatih-hoca): scheduled benchmark refresh` etc.
- **Subagent-driven** — delegate each item to a subagent with crisp scope.
- **Test timeouts** — all pytest invocations use `timeout 30 pytest ...`
  (targeted) or `timeout 120 pytest ...` (full).

## Dependencies Between Items

All four items are independent. Suggested build order: 2 → 1 → 3 → 4.
(Audit first because it's the smallest diff and teaches the asyncio pattern
fresh; refresh second because it's the most load-bearing; command third;
simulator last because it's the largest surface area.)

## Success Criteria

- `tick_benchmark_refresh()` runs on heartbeat; `.benchmark_cache/` stays
  < 24h old without manual CLI runs.
- Grep for unassigned `asyncio.create_task(` returns zero hits in `src/`
  and `packages/` after the audit.
- `/bench_picks` returns a formatted table in Telegram.
- `python -m packages.fatih_hoca.simulate_i2p` produces a reproducible
  distribution report for all 182 i2p steps.
- All existing tests still pass; four new test modules added.
