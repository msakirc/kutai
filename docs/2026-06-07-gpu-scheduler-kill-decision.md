# gpu_scheduler — kill-or-fold decision (spike output)
*2026-06-07. Sub-task 2a of `docs/2026-05-31-modularization-finish-plan.md` P2. Founder directive: "ideally none" — spike before any move.*

## Decision: **KILL** (delete `src/models/gpu_scheduler.py` + the dead slot API)

## Evidence
- `GPUScheduler.acquire()` / `release()` are reachable ONLY via
  `LocalModelManager.acquire_inference_slot()` / `release_inference_slot()`
  (`src/models/local_model_manager.py` L415–443).
- Those two methods have **zero prod callers** (whole-repo grep). Only refs:
  - `tests/test_dallama_shim.py` — `hasattr` existence asserts (L21-22) + `"inference_busy" in status` (L88)
  - `tests/unit/test_idle_watchdog_race.py` — patches `get_gpu_scheduler` (defensive, not a real user)
- Consequence: a slot is never acquired ⇒ `_current` is always `None` ⇒
  `is_busy` is constant `False`, queue is always empty, and the
  `schedule_accelerate_retries("gpu_available")` wake on release never fires.
- `local_model_manager.get_status()` L470 exposes `"inference_busy":
  self._scheduler.is_busy` — a constant-`False` field with no prod reader
  (only the shim test asserts the key exists).

## Why it's redundant (not a lost feature to restore)
- Concurrent inference requests to a loaded model are serialized/batched by
  **llama-server itself** (its own request slots).
- Concurrent model **swaps** are serialized by **DaLLaMa's swap lock**.
- The app-layer priority queue duplicated a concern already owned lower down,
  and was disconnected. Founder: "ideally none" → don't reintroduce an
  app-layer GPU arbiter.

## Tension noted
`docs`/memory `pressure_concurrency_20260524`: `ONESHOT_CONCURRENCY=4`
thrashes one GPU. If priority-ordered serialization is ever wanted, it should
live in DaLLaMa (the GPU owner), driven by Beckman task priority — NOT as an
uncalled module under `src/models/`. That is a *new* feature decision, not a
reason to keep dead code.

## Execution plan (pure subtraction)
1. Delete `src/models/gpu_scheduler.py`.
2. `local_model_manager.py`: drop `self._scheduler = get_gpu_scheduler()`
   (init), `acquire_inference_slot` + `release_inference_slot` (L415–443),
   and the `"inference_busy"` status field (L470).
3. Tests: drop the two `hasattr` asserts + `inference_busy` assert in
   `test_dallama_shim.py`; drop the `get_gpu_scheduler` patch in
   `test_idle_watchdog_race.py`.
4. Verify: import smoke + dallama-shim + idle-watchdog + a load/swap test.

End state: `src/models/` loses gpu_scheduler entirely (matches finish-plan
P2 "Done" target). No behavioral change (removed paths were never executed).
