# Audit — `record_model_call` double-counting

Anchor: 2026-05-04. Continues Phase A/B handoff
(`docs/handoff/2026-05-04-runtime-phase-a-complete.md` line 100). Audit
performed standalone; **fix deferred to Phase C** per the original plan.

## Finding

For every ReAct iteration, in-memory Prometheus-style counters
(`_counters` in `src/infra/metrics.py`) are incremented **twice** for the
same call:

1. `hallederiz_kadir.caller.call()` → `track_model_call_metrics(...)`
   directly. Lines 157-158 of `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py`.
2. `coulson.react` after the call returns →
   `await record_model_call(...)` (db.py 2496) →
   `track_model_call_metrics(...)` internally (db.py 2512).

Both fire on every iteration. The DB persistence to `model_stats` is
still single-counted — that path lives only in `record_model_call`. The
double-count is in-memory only.

## Affected counters

`src/infra/metrics.py::_counters` keys:

- `model_calls:{model}` — call count, doubled
- `cost:{model}` — cost sum, doubled
- `latency_sum:{model}` — latency sum, doubled
- `tokens:{model}` — token sum, **single-counted** (record_model_call
  passes 0 for tokens, so this dimension is fine)

`token_count` for a given model also lives in `model_call_tokens` table
(written by hallederiz directly) — DB metric. Unaffected.

## Other call sites of `record_model_call`

- `src/core/grading.py:425-426` — post-task grade record. Single call
  per task, no in-flight overlap with hallederiz (grading is its own
  hallederiz call, separately tracked). Adds an extra metrics tick on
  top of hallederiz's own.
- `src/core/metrics_push.py:62,79` — mission summary push. Synthesizes
  iteration×latency. Adds a ghost metrics tick.
- `src/memory/feedback.py:72-73` — feedback channel. Adds a ghost
  metrics tick.

All four call sites compound the double-count. Net inflation factor on
`model_calls:{model}` and `cost:{model}` is roughly 2.5× of true
hallederiz calls (1× per real call from hallederiz, plus 1× from
react.py per ReAct iter, plus periodic +1 from grading / metrics_push /
feedback for some tasks).

## Why the test enshrines the bug

`tests/test_architecture_fixes.py::TestRecordModelCallUnified::test_db_record_model_call_calls_metrics_internally`
(line 67-74) asserts the source of `db.py::record_model_call` contains
`track_model_call_metrics`. This was added when `metrics.py` got its
own `record_model_call` deleted (item #14 in `plans/plan_v5.md`) and
the metrics tracking was folded into `db.py::record_model_call` for
unification. At that time hallederiz didn't yet emit metrics directly.
The hallederiz call was added later (predates the runtime extraction)
and the unification test was never revisited.

## Recommended fix (Phase C)

1. Remove `track_model_call_metrics` call from `db.record_model_call`.
   db.py becomes a pure persistence path; hallederiz is the single owner
   of in-memory counters.
2. Flip the architecture test:

   ```python
   def test_db_record_model_call_does_not_emit_metrics(self):
       """db.record_model_call must NOT call track_model_call_metrics —
       hallederiz_kadir.caller is the single emitter to keep counters
       single-counted across all call paths.
       """
       source = _read_source("src/infra/db.py")
       # Find record_model_call body
       start = source.find("async def record_model_call")
       next_def = source.find("\nasync def ", start + 1)
       body = source[start:next_def] if next_def != -1 else source[start:]
       self.assertNotIn(
           "track_model_call_metrics",
           body,
           "record_model_call must not emit metrics; hallederiz owns that.",
       )
   ```

3. Audit the four `record_model_call` callers (coulson.react,
   grading.py, metrics_push.py, feedback.py) and decide whether each
   *should* keep calling it. coulson.react's call is the obvious
   redundancy — hallederiz already saw the call. Grading, metrics_push,
   feedback are post-hoc telemetry pushes, also redundant once
   hallederiz is the truth source. If all four go away, `record_model_call`
   becomes dead code and `model_stats` table dies too.

   Conservative path: remove only coulson.react's call. Keep the others
   because they synthesize aggregate statistics (per-task grade,
   mission summary) that hallederiz can't see.

4. Tighten via Prometheus dashboard sanity check before+after — values
   for active models should drop to ~50% of pre-fix.

## Why not fix now

Phase A/B handoff explicitly defers this to Phase C dispatcher
slim-down (line 100 of the runtime-phase-a handoff). Phase C will move
per-iter Hoca selection out of dispatcher and into coulson.react,
replacing the existing two-retry-layer architecture with a single retry
surface. That work is the right time to also collapse the metrics
double-emission path, since dispatcher's retry-recurse currently calls
`_record_pick` (DB telemetry, separate from `record_model_call`) on
every attempt, and consolidating retry into one layer simplifies the
metrics question simultaneously.

Standalone fix today would either:
- Touch coulson.react after just-shipping it (Phase B yesterday), OR
- Touch dispatcher right before Phase C rewrites it.

Either way the patch will get rewritten in Phase C. Wait.

## Quick verification commands

```bash
# Sample current counter inflation (run while traffic flowing)
.venv/Scripts/python.exe -c "
from src.infra.metrics import _counters
import json
print(json.dumps(
    {k:v for k,v in _counters.items() if k.startswith('model_calls:')},
    indent=2
))"

# Compare to ground-truth hallederiz call count
.venv/Scripts/python.exe -c "
import sqlite3
c = sqlite3.connect(r'C:\\Users\\sakir\\ai\\kutai\\kutai.db')
print(dict(c.execute('SELECT model, COUNT(*) FROM model_call_tokens GROUP BY model').fetchall()))
"
```

The first numbers should be ~2× the second per active model.

## Status

Read-only audit. No code or tests changed. Finding entered into the
Phase C scope.
