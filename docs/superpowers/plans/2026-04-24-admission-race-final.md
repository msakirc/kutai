# Admission Race — Final Architectural Plan

**Date:** 2026-04-24
**Status:** Implementing

## Root cause

The in-flight registry represents only two of three GPU-slot lifecycle states:

1. **Admitted** — Beckman claimed the DB row and chose a local model. The task conceptually owns the slot but has not made an LLM call yet.
2. **Calling** — Dispatcher ran `_begin_call`. `_task_slots[task_id]` populated.
3. **Between calls (ReAct iteration gap)** — Call finished; tool execution in progress; next call pending. Task-slot survives via `task-{id}` key.

Current design handles (2) and (3). **(1) has no writer.** The 22-second window between `claim_task` returning and the dispatcher's first `_begin_call` is invisible to `pressure_for()`.

Prior patches — pre-warming the embedder, moving `_begin_call` earlier inside dispatcher, killing the hard cap — all try to shrink or dodge the gap. None represent state (1) explicitly. The gap is **structural**, not latency-driven.

## The fix

Extract the in-flight registry into its own peer module and add a `reserve_task` primitive.

### Module: `src/core/in_flight.py` (new)

Holds:
- `_task_slots: dict[int, _InFlightEntry]`
- `_call_entries: dict[str, _InFlightEntry]`
- `_InFlightEntry` dataclass

Exposes:
- `reserve_task(task_id, pick)` — **Beckman calls this** after `_claim_task` succeeds
- `begin_call(category, model_name, provider, is_local, task_id)` — **Dispatcher calls this** on each request
- `end_call(call_id)` — **Dispatcher calls this** in finally
- `release_task(task_id)` — **Orchestrator calls this** in `_dispatch` finally
- `in_flight_snapshot() -> list` — read-only accessor
- `_push()` — internal, pushes merged list to nerd_herd on every mutation

### Constraint satisfaction

| Constraint | Honored |
|---|---|
| Nerd_herd purity | Unchanged — still receives `push_in_flight(list)` |
| Single producer to nerd_herd | `src.core.in_flight` is the ONLY module that calls `nerd_herd.push_in_flight` |
| No Beckman→dispatcher import | Beckman imports `src.core.in_flight`, not dispatcher |
| Single concept | One registry, one `task-{id}` key schema, same upsert semantics |
| ReAct gap survival | `end_call` preserves `task-*` keys (unchanged) |
| Mid-task model accuracy | `begin_call` upserts same slot with new model on retry (unchanged) |

### Lifecycle

```
t0   Beckman.next_task() claims row + calls reserve_task(2879, pick)
     → _task_slots[2879] populated with admission-time model
     → in_flight.push() fires
     → nerd_herd sidecar updated
t0+ε Orchestrator.create_task(_dispatch)
t0+3s Beckman tick N+1: refresh_snapshot sees task-2879 in list
     → pressure_for(local_model) = -1.0
     → REJECT second local candidate
...
t0+22s Dispatcher _begin_call: UPSERTs task-2879 with current model
t0+Nm  _end_call fires; task-* preserved
       (iteration gap; slot still present)
t_done Orchestrator _dispatch finally: release_task(2879)
       → slot removed, nerd_herd pushed empty
       → admission re-opens local
```

## Migration steps

### Step A — Extract registry

1. Create `src/core/in_flight.py`. Move from `src/core/llm_dispatcher.py`:
   - `_task_slots`, `_call_entries`, `_InFlightEntry`
   - `_push_in_flight` → rename `_push`
   - `_begin_call` → public `begin_call`
   - `_end_call` → public `end_call`
   - `release_task`
   - `in_flight_snapshot`
2. Add new public `reserve_task(task_id, pick)`:
   ```python
   async def reserve_task(task_id: int, pick) -> None:
       model = pick.model
       _task_slots[task_id] = _InFlightEntry(
           call_id=f"task-{task_id}",
           task_id=task_id,
           category="main_work",  # provisional; upsert on real call
           model=model.name,
           provider=model.provider,
           is_local=model.is_local,
           started_at=time.time(),
       )
       await _push()
   ```
3. Leave `_call_entries` path for standalone calls untouched.

### Step B — Dispatcher updates

`src/core/llm_dispatcher.py`:
- Remove `_task_slots`, `_call_entries`, `_InFlightEntry`, `_push_in_flight`.
- Replace `_begin_call` / `_end_call` / `release_task` calls with `from src.core.in_flight import begin_call, end_call, release_task as _release_task`.

### Step C — Orchestrator updates

`src/core/orchestrator.py`:
- Update import: `from src.core.in_flight import release_task`.

### Step D — Beckman wires reserve

`packages/general_beckman/src/general_beckman/__init__.py` in `next_task()`:
- After successful `_claim_task`, before `return task`:
  ```python
  try:
      from src.core.in_flight import reserve_task
      await reserve_task(task["id"], pick)
  except Exception as e:
      _log.warning(f"reserve_task failed #{task['id']}: {e}")
  ```
- Fail-open: if reserve raises, admit anyway. (Nerd_herd being down shouldn't halt admission.)

### Step E — Skip mechanical

Mechanical tasks don't use GPU → skip reserve for them. Already short-circuited via mechanical bypass block above the reserve site.

## Verification

### Unit tests
- `reserve_task` populates slot with pick's model.
- `begin_call` with same task_id upserts (replaces model).
- `end_call` with `task-*` call_id is no-op on the dict.
- `release_task` clears and pushes empty list.

### Integration test
- Stub dispatcher to never call `begin_call` (simulate 22s gap).
- Admit A → reserve fires.
- Tick Beckman with fresh refresh_snapshot → assert B REJECTED.
- Call `release_task(A)` → tick Beckman → assert B ADMITTED.

### Production log assertions
- After every `admission: task #N ADMIT model=<local>`, next admission within 30s must be REJECT (or non-local) — count violations post-deploy, expect 0.
- `main_work candidate failed.*context_overflow` should drop to near-zero.

### Regression guard
- Grep assertion: `_task_slots` and `_call_entries` appear only in `src/core/in_flight.py`.

## What NOT to do (failed patch checklist)

- Do not add `BECKMAN_HARD_CAP` or artificial caps.
- Do not let Beckman import `src.core.llm_dispatcher`.
- Do not re-add `requests_processing` fallback in `_local_pressure`.
- Do not move `begin_call` earlier inside dispatcher — the gap is before dispatcher runs.
- Do not split registry into "admission-level" and "call-level" dicts.
- Do not pin `context_length` in models.yaml.
- Do not put reserve call in orchestrator's `_dispatch` — too late; `_dispatch` runs on its own asyncio.Task after pump sleep.
- Do not add business logic to `reserve_task` — pure state write.
