# Handoff — Yalayut demand-signal subsystem: 6 of 7 signal types unwired

**Date:** 2026-05-16
**Found by:** Phase 4 review pass (4-phase wiring audit)
**Status:** open gap — needs a dedicated follow-up plan
**Severity:** P1 — the autonomous on-demand discovery loop is functionally dead

## Summary

Phase 4 Task 1 built the demand-signal subsystem
(`packages/yalayut/src/yalayut/discovery/demand.py`): the `DemandSignal`
dataclass, `record_signal()`, `stack_confidence()`, `pending_signals()`,
`mark_discovered()`, dedupe + 7-day cooldown. All of that works and is tested.

What is **not** built: the firing sites. `record_signal()` is invoked from
exactly one production location, for exactly one of the seven signal types.
The other six are defined constants with no caller. The chain
`signal fires → stacks → on_demand_discovery consumes it` therefore only ever
runs when the founder manually types `/yalayut discover <intent>`.

The plan never contained a task to instrument the pipeline. This is a genuine
scope gap, not a regression.

## Current state

`SIGNAL_TYPES` (demand.py) — 4 proactive + 3 reactive:

| Signal type | Category | Fired in production? | Where it should fire |
|---|---|---|---|
| `founder` | proactive | **YES** — `telegram_bot.py` `/yalayut discover` | (done) |
| `planning_miss` | proactive | no | planner/expander: a workflow step whose intent matches no enabled catalog artifact |
| `step_entry_miss` | proactive | no | `intersect.flash()`: task enters dispatch, `yalayut.query()` returns nothing |
| `tool_call` | proactive | no | `coulson/react.py` runtime tool dispatch: agent requests a capability with no backing skill/tool |
| `hint_miss` | reactive | no | `yalayut/capture.py`: repeated `internal_hint` capture on one pattern — signals a reusable *external* skill would beat re-deriving it |
| `dlq` | reactive | no | `general_beckman/apply.py` DLQ path: task exhausted — record the task's intent as a demand |
| `repeat_pattern` | reactive | no | periodic aggregation: same `source_step_pattern` seen N times across tasks |

Consumers are ready and unused:
- `pending_signals(limit)` — returns distinct patterns ordered by stacked confidence.
- `stack_confidence(pattern)` — `1 - Π(1-c_i)`.
- `on_demand_discovery(demand)` — `discovery/on_demand.py`, real body, works.
- The `yalayut_discovery` mr_roboto executor already dispatches an `on_demand`
  mode (`executors/yalayut_discovery.py`).

## The second half of the gap — no autonomous trigger

Even with signals firing, nothing periodically *drains* `pending_signals()`.
`on_demand_discovery` is reachable today only via the immediate enqueue that
`/yalayut discover` does right after recording the founder signal. There is no
scheduled check that says "stacked confidence on pattern X crossed the
threshold → enqueue an on-demand discovery run for X".

## Proposed follow-up plan (sketch — not yet written)

Two work units. Do them in this order.

### Unit A — fire the six dead signal types

One small, isolated change per site. Each is a single `await
demand.record_signal(DemandSignal(...))` call, best-effort (wrapped so a
signal failure never affects the host path). `yalayut` must not be imported
into core loop files directly — fire through a thin lazy import, matching how
`orchestrator.py` already lazy-imports for the periodic checks, or route via a
mechanical task. Decide per site:

- `step_entry_miss` — `intersect/flash.py`, where `query()` returns `[]`.
  `intersect` already imports `yalayut`, so a direct call is fine here.
- `planning_miss` — planner/expander, when a step's `tools_hint` / intent has
  no catalog match.
- `tool_call` — `coulson/react.py`, on an unresolved tool request.
- `hint_miss` — `yalayut/capture.py`, when `capture_hint` writes its Nth
  `internal_hint` for the same pattern.
- `dlq` — `general_beckman/apply.py` DLQ write path. apply.py must not import
  `yalayut`; enqueue a mechanical task or use a lazy import behind a flag.
- `repeat_pattern` — derive inside a periodic scan rather than a hot-path
  call; cheapest to compute from existing `yalayut_demand_signals` rows or
  task history.

Keep `intent_keywords` meaningful at each site — they drive the untrusted
source match in `on_demand_discovery`.

### Unit B — autonomous on-demand trigger

Add a periodic drain. Two viable shapes:

1. New orchestrator check `_check_on_demand_discovery()` — mirrors
   `_check_yalayut_discovery` / `_check_source_scout`: timestamp-gated, zero
   yalayut import, enqueues a plain dict
   `{mechanical, action:"yalayut_discovery", mode:"on_demand"}`.
2. Or fold the drain into the existing `yalayut_discovery` daily executor:
   after `daily_discovery()`, call `pending_signals()`, and for each pattern
   with `stacked_confidence` ≥ threshold run `on_demand_discovery(demand)`
   then `mark_discovered(pattern)`.

Option 2 is less wiring (no new orchestrator method, no new cadence row) and
keeps discovery in one executor. Recommend option 2 unless on-demand needs a
faster cadence than daily.

Pick a confidence threshold (spec implies ~0.5–0.6; `source_scout`'s
`_scan_demand_websearch` already filters at `stacked_confidence ≥ 0.5` — reuse
that constant for consistency).

## Acceptance

- Each of the 6 signal types has a production call site, proven by a test that
  exercises the host path and asserts a row lands in `yalayut_demand_signals`.
- A test proves the autonomous trigger drains `pending_signals()` and calls
  `on_demand_discovery` once a pattern crosses the threshold, then
  `mark_discovered` flips `resulted_in_discovery`.
- No core-loop file imports `yalayut` directly except `intersect` (which
  already does).

## References

- Subsystem: `packages/yalayut/src/yalayut/discovery/demand.py`
- On-demand body: `packages/yalayut/src/yalayut/discovery/on_demand.py`
- Existing founder firing site: `src/app/telegram_bot.py` (`/yalayut discover`)
- Phase 4 plan: `docs/superpowers/plans/2026-05-16-yalayut-phase4-discovery.md`
- Review-pass fixes that shipped alongside this finding: commits `98dc3e03`,
  `49211450`, `b878895b`, `cb1576a8`, `2e7f1d19` (capture_hint verdict).
