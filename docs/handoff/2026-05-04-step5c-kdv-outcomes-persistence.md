# Handoff — Step 5c: KDV outcomes persistence

User: sakircimen@gmail.com
Anchor: 2026-05-04 (drafted during Step 5 session, deferred to its own commit)

Prerequisite: Steps 5a + 5b shipped (SQLite registry replacing
`.dead_models.json`). 5c is strictly additive — no dependency on
Step 6.

---

## Goal

Persist `KuledenDonenVar._outcomes` deque (per-model rolling success/
failure window) across process restarts. Currently the deque is
in-memory only; restart wipes it; selector's S10_failure signal
returns 0 (neutral) until 5+ samples re-accumulate per model.

The 24h outcome window (`_OUTCOME_MAX_AGE_SECONDS = 86400`, see
2026-05-03 handoff Step 1) is meaningless if it dies on every reboot.
Cold-start gap is the limiting factor on S10's signal usefulness.

---

## Why this is its own commit (not Step 5a/5b)

- 5a/5b is `fatih_hoca/registry.py` + new `src/infra/registry_store.py`
  — kill-switch state.
- 5c is `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py` +
  `src/infra/kdv_persistence.py` — reliability state.

Different layer, different package, different concern. Bundling would
muddy review. 5c can ship before, after, or independently of Step 6.

---

## Current state (no changes needed to find these)

### KDV `_outcomes` deque

`packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py:50`

```python
self._outcomes: dict[str, deque] = {}
```

Populated by `_record_outcome(model_id, success)` at line 427. Read by
`recent_success_rate(model_id)` at line 437 and `recent_samples_n` at
line 456. Both consumers age-filter against `_OUTCOME_MAX_AGE_SECONDS`
on read.

### KDV snapshot/restore mechanism

`packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py:599-644`

```python
def snapshot_state(self) -> dict:
    return {
        "models":         {mid: state.snapshot_state() for ...},
        "providers":      {prov: state.snapshot_state() for ...},
        "breakers":       {prov: cb.snapshot_state() for ...},
        "enabled_at":     dict(self._provider_enabled_at),
        "call_count":     dict(self._provider_call_count),
        "attempt_count":  dict(self._provider_attempt_count),
    }
```

`outcomes` is NOT in the dict — that's the gap.

### Persistence bridge

`src/infra/kdv_persistence.py:34-74`

`save()` iterates the snapshot dict, packs each scope into a row:

```python
for mid, model_snap in snap.get("models", {}).items():
    rows.append(("model", mid, json.dumps(model_snap), now))
# ... and same for providers, breakers
# meta row holds enabled_at, call_count, attempt_count
```

`load()` (line 78+) reads back, filters stale rows (>24h), feeds
`kdv.restore_state(snap)`. Already wired at orchestrator startup.

### `kdv_state` table

`src/infra/db.py:778`

```sql
CREATE TABLE IF NOT EXISTS kdv_state (
    scope         TEXT NOT NULL,
    scope_key     TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    last_persisted REAL NOT NULL,
    PRIMARY KEY (scope, scope_key)
);
```

No schema change needed — `outcomes` rows fit existing shape.

---

## Changes (3 files, ~40 LOC + tests)

### 1. `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py`

#### Extend `snapshot_state()`

Add `outcomes` key holding per-model serialized deques. Each entry is
a list of `[ts, success_bool]` tuples. JSON-safe.

```python
def snapshot_state(self) -> dict:
    return {
        # ... existing keys ...
        "outcomes": {
            mid: [[ts, bool(s)] for ts, s in dq]
            for mid, dq in self._outcomes.items()
        },
    }
```

Don't pre-trim aged entries here — reader does that on restore. Keeps
the snapshot reproducible.

#### Extend `restore_state()`

Repopulate `self._outcomes` deques. Apply age filter on the way in
(don't restore entries older than `_OUTCOME_MAX_AGE_SECONDS`):

```python
def restore_state(self, snap: dict) -> None:
    # ... existing restore logic ...
    from collections import deque
    import time as _time
    cutoff = _time.time() - self._OUTCOME_MAX_AGE_SECONDS
    for mid, entries in snap.get("outcomes", {}).items():
        dq = deque(maxlen=self._OUTCOME_MAX_LEN)
        for entry in entries or []:
            try:
                ts, success = float(entry[0]), bool(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if ts < cutoff:
                continue
            dq.append((ts, success))
        if dq:
            self._outcomes[mid] = dq
```

Filter on restore, not save: a process that crashed mid-window should
still write what it had. Loading-side decides what's stale.

### 2. `src/infra/kdv_persistence.py`

#### `save()` — add new scope iteration

```python
for mid, entries in snap.get("outcomes", {}).items():
    rows.append(("outcomes", mid, json.dumps(entries), now))
```

Insert after the `breakers` loop, before the `meta` row append at
line 51.

#### `load()` — add scope handling

The current loader (read it at `src/infra/kdv_persistence.py:78+`)
groups rows by scope and feeds them into `snap_for_kdv["models"]`,
`snap_for_kdv["providers"]`, etc. Add:

```python
elif scope == "outcomes":
    snap_for_kdv.setdefault("outcomes", {})[scope_key] = json.loads(snapshot_json)
```

Stale-row filter (`last_persisted > now - stale_hours * 3600`) already
applies — don't duplicate.

### 3. Tests

**`packages/kuleden_donen_var/tests/test_kdv.py`** — extend with:

```python
def test_snapshot_includes_outcomes():
    """snapshot_state captures _outcomes deques for restore."""
    # populate via _record_outcome, snapshot, assert key present

def test_restore_outcomes_round_trip():
    """restore_state(snapshot_state()) preserves recent_success_rate."""
    # populate, snapshot, fresh KDV instance, restore, assert rate matches

def test_restore_outcomes_drops_aged_entries():
    """Entries older than _OUTCOME_MAX_AGE_SECONDS are filtered on restore."""
    # build snapshot with mixed-age entries, restore, assert aged ones gone
```

**`tests/integration/test_kdv_persistence.py`** (or wherever the existing
save/load round-trip test lives — check first; some integration tests
already cover the rate-limiter persistence path) — add an outcomes
round-trip case to confirm the bridge wiring.

---

## Pitfalls / things to watch

### Snapshot size

A model with 1000 outcomes (deque maxlen — find the constant in
`kdv.py`, search `_OUTCOME_MAX_LEN`) serializes to ~30KB JSON. Times
~30 active cloud models = ~900KB across the kdv_state table. Fine
for SQLite, but worth confirming `_OUTCOME_MAX_LEN` isn't pathologically
large. If it is, consider per-model max entries on snapshot (cap
oldest first).

### Save cadence

Find where `save()` is called from. Likely orchestrator periodic
task — check `src/app/run.py` or `src/core/orchestrator.py` for
`kdv_persistence.save`. Expected cadence: every ~5min + on graceful
shutdown. Outcomes changing every call means we're saving frequently
either way; the existing cadence is the right cadence.

### Restore order

`restore_state()` runs early at orchestrator boot, before any LLM
call. Verify the `_outcomes` restore happens AFTER per-model
RateLimitState restore (order in `restore_state` matters because
some logic — none currently, but be safe — could depend on rate-limit
state existing before outcome state).

### Aged-entry trimming on read still required

Don't remove the existing `while dq and dq[0][0] < cutoff: dq.popleft()`
in `recent_success_rate` / `recent_samples_n`. Restore-side filter
handles startup; runtime filter handles the case where the deque was
populated, then sat idle for >24h before next read.

### JSON encoding of deque entries

Python `deque((1.5, True))` doesn't JSON-serialize directly via
`json.dumps`. Hence the `[[ts, bool(s)] for ts, s in dq]` conversion
in `snapshot_state()`. On restore, JSON gives lists; cast back to
tuples when appending into the deque (`dq.append((ts, success))`).

### Schema migration

None. `kdv_state` table accepts any (scope, scope_key) — no DDL change.
Existing rows untouched. New scope `"outcomes"` just starts appearing.

### Cross-cut with Step 6

Step 6 (provider_prior_rate) reads the same `_outcomes` deques
aggregated by provider (or openrouter sub-key). Step 5c persisting
those deques means provider_prior also survives restart. Strict
strengthening, no conflict.

---

## Sequencing

Single commit. Order within the commit:

1. Extend `KuledenDonenVar.snapshot_state` + `restore_state`.
2. Extend `kdv_persistence.save` + `load`.
3. Tests.

Run after change:

```bash
timeout 120 .venv/Scripts/python.exe -W ignore -m pytest \
    packages/kuleden_donen_var/tests/ tests/integration/ -q
```

Expected: prior count + ~3 new tests passing.

---

## Don't

- DON'T persist `_outcomes` from `RateLimitState` — wrong layer. The
  deque lives on `KuledenDonenVar` itself, not per-RateLimitState.
- DON'T pre-trim aged entries inside `snapshot_state`. Trim on restore.
- DON'T add a separate `outcomes_state` table. Reuse `kdv_state` with
  scope=`"outcomes"`.
- DON'T touch `_OUTCOME_MAX_AGE_SECONDS` (currently 86400). Step 1 of
  the prior session set this; persistence inherits the same window.
- DON'T forget the cast back to tuples on restore — `(ts, success)`,
  not `[ts, success]`. The age-trim loop indexes `dq[0][0]` which
  works on both, but the matching `while ... popleft` semantics rely
  on tuple shape consistency for `bool` checks.

---

## Verification after shipping

1. Orchestrator restart: `recent_success_rate` for a model with prior
   data should return its real rate, not 1.0.
2. Query `SELECT scope_key, length(snapshot_json) FROM kdv_state
   WHERE scope='outcomes'` after a few minutes of runtime — confirm
   per-model rows present, sizes reasonable.
3. Set `_OUTCOME_MAX_AGE_SECONDS` temporarily lower, restart, confirm
   restore drops aged entries (delete after verifying).
