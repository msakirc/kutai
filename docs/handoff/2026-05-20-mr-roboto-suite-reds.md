# Handoff — 7 mr_roboto suite reds (post wiring-sweep)

> **RESOLVED 2026-05-21** (commit `ae004547`). All 3 causes fixed: conftest
> `_PACKAGE_SRCS`/eviction now include `safety_guard` (+ `intersect`/`yalayut`);
> `synthetic_check` tagged `"full"` in `VERB_REVERSIBILITY`; the stale
> `test_offline_sync_has_no_flow` flipped to pin smoke-flow presence. Full
> mr_roboto suite: **732 passed, 0 failed**. Kept for history.

**Date:** 2026-05-20
**Branch:** `main` (HEAD `a0b83166`)
**Scope:** new failures in `packages/mr_roboto/tests/` introduced after
commit `46d5a334` (2026-05-16, the suite-hang fix). The hang itself stays
fixed — suite runs to completion in ~42s, 725 tests collected.
**Result:** 7 failed / 725 passed.

These reds are NOT listed in the broader `2026-05-19-wiring-sweep-residuals.md`
§3 pre-existing failures table. They surfaced after the wiring-sweep batch
merged and `safety_guard` was introduced as a new `pre_action` hook in
`mr_roboto.run`.

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/ -q
# 7 failed, 725 passed, 2 skipped, 7 warnings in ~42s
```

---

## §1 — Summary: 3 root causes, 7 failures

| Cause | Tests | Fix complexity |
|---|---|---|
| **A** — `safety_guard` package not on sys.path during tests | 5 | trivial (1-line conftest add) |
| **B** — `synthetic_check` dispatcher action untagged | 1 | trivial (1-line registry add) |
| **C** — `test_offline_sync_has_no_flow` asserts state that Z5 P2 deliberately reversed | 1 | small (update or drop the test) |

---

## §2 — Cause A: `safety_guard` import fails in tests

### Affected tests (5)

- `test_action_reversibility.py::test_run_cmd_default_partial`
- `test_action_reversibility.py::test_run_cmd_override_irreversible_wins`
- `test_run_cmd.py::test_run_router_run_cmd_completed`
- `test_run_cmd.py::test_run_router_run_cmd_failed_on_nonzero_when_required`
- `test_z7_audit_log_realpath.py::test_run_non_publish_verb_writes_no_audit_row`

### Where it dies

```
packages/mr_roboto/src/mr_roboto/__init__.py:411
    guard_result = await _safety_guard_check(task)

packages/mr_roboto/src/mr_roboto/__init__.py:258
    from safety_guard import pre_action, Allow, WaitForFounder, Block
E   ModuleNotFoundError: No module named 'safety_guard'
```

### Root cause

`_safety_guard_check` was added to `mr_roboto.run` (called before every
dispatch). It imports from the `safety_guard` package at
`packages/safety_guard/src/safety_guard/`. That package exists on disk —
it's only invisible to pytest.

The worktree-root `conftest.py` prepends each `packages/*/src` to
`sys.path` and evicts the matching `sys.modules` entries so pytest reads
the worktree source instead of any pre-installed editable copy. Its
`_PACKAGE_SRCS` list omits `safety_guard`:

```python
# conftest.py
_PACKAGE_SRCS = [
    _ROOT / "packages" / "fatih_hoca" / "src",
    _ROOT / "packages" / "nerd_herd" / "src",
    ...
    _ROOT / "packages" / "c21_paraflow_diff" / "src",
]
# (no safety_guard entry)
```

The sibling sys.modules-eviction set has the same omission.

Tests that exercise verbs where `_safety_guard_check` actually runs —
the `run_cmd` path and any verb without a short-circuit before line 411 —
hit the import and ModuleNotFoundError. Tests that mock the executor or
take a different code path silently skip the import.

### Fix

Two lines in worktree-root `conftest.py`:

```python
_PACKAGE_SRCS = [
    ...
    _ROOT / "packages" / "c21_paraflow_diff" / "src",
    _ROOT / "packages" / "safety_guard" / "src",   # add
]

for _mod in list(sys.modules):
    root = _mod.split(".", 1)[0]
    if root in {
        ...
        "c21_paraflow_diff",
        "safety_guard",   # add
    }:
        del sys.modules[_mod]
```

No production-code change needed — `safety_guard` is reachable in the
real KutAI runtime because it's an editable install. Only the test
harness was skipped.

### Verification

```
.venv/Scripts/python -m pytest \
  packages/mr_roboto/tests/test_action_reversibility.py \
  packages/mr_roboto/tests/test_run_cmd.py \
  packages/mr_roboto/tests/test_z7_audit_log_realpath.py \
  -q
```
Expect all green.

### Est. effort

5 minutes.

---

## §3 — Cause B: `synthetic_check` missing from VERB_REVERSIBILITY

### Affected test (1)

- `test_reversibility_registry.py::test_every_dispatcher_action_is_in_registry`

### Where it dies

```
E AssertionError: dispatcher actions missing from VERB_REVERSIBILITY:
  ['synthetic_check']
```

### Root cause

Commit `28433d5d` ("fix(z8): synthetic_check dispatch + 5 ops cron seed
rows") added the `if action == "synthetic_check":` block in
`mr_roboto/__init__.py` but did not add a reversibility tag to
`reversibility.py::VERB_REVERSIBILITY`. The guard test is doing its job.

### Fix

One line in `packages/mr_roboto/src/mr_roboto/reversibility.py`, in the
"Scanners / reviewers / checks" section (or alongside the other Z8
ops-cron verbs):

```python
"synthetic_check": "full",  # Z8 P1 — read-only HTTP probe; advisory founder_action only
```

Tag rationale: synthetic monitoring is a probe — read-only HTTP +
optional founder_action, no external side-effect. Same shape as the
other Z8 ops crons (`backup_verify`, `dependency_scan`, `cve_scan`,
`secret_scan`, `cost_pull`) which are all "full". Confirm by reading
the dispatcher block at `__init__.py` (look for `synthetic_check`); if
the verb actually emits a Telegram message rather than a founder_action
row, promote to `irreversible`.

### Verification

```
.venv/Scripts/python -m pytest \
  packages/mr_roboto/tests/test_reversibility_registry.py -q
```
Expect 6 passed (the file has 6 collected tests).

### Est. effort

5 minutes once the dispatcher block is read.

---

## §4 — Cause C: `test_offline_sync_has_no_flow` stale relative to Z5 P2

### Affected test (1)

- `test_z5_t4b_maestro.py::TestMaestroFlowTemplates::test_offline_sync_has_no_flow`

### Where it dies

```
packages/mr_roboto/tests/test_z5_t4b_maestro.py:536
    assert "smoke_flow" not in recipe.templates
E   AssertionError: assert 'smoke_flow' not in {... 'smoke_flow':
    'flows/offline_sync_smoke.flow.yaml', ...}
```

### Root cause

The test was authored when the `mobile_offline_sync` recipe had no
`smoke_flow` template — its whole point was to pin the absence ("offline
sync should NOT carry a maestro flow template, only smoke_flow recipes
do"). Commit `461a90d0` ("Z5 P2 mobile_smoke flows + workspace
auto-discovery") deliberately added `smoke_flow:
flows/offline_sync_smoke.flow.yaml` to the recipe so the offline-sync
scaffold ships its own smoke flow.

The recipe is now correct by design; the test is the stale side.

### Fix

Need a brief read of the surrounding test class to confirm intent.
Likely one of:

1. **Update the assertion** to match the new contract: offline-sync
   recipes legitimately ship a smoke_flow now. The assertion should be
   `assert recipe.templates["smoke_flow"].endswith(".flow.yaml")` or
   similar shape check, not absence.
2. **Delete the test** outright if `test_offline_sync_has_no_flow` was
   meant to enforce a separation that Z5 P2 removed intentionally. The
   sibling tests in `TestMaestroFlowTemplates` probably already cover
   the "smoke_flow file exists + parses" angle.

Confirm by reading `461a90d0`'s commit body / diff and the rest of
`TestMaestroFlowTemplates` to see which option matches the Z5 author's
intent. If unclear, ping the Z5 P2 author.

### Verification

```
.venv/Scripts/python -m pytest \
  packages/mr_roboto/tests/test_z5_t4b_maestro.py -q
```

### Est. effort

15 minutes including reading `461a90d0` to pick option 1 vs 2.

---

## §5 — Suggested order

1. **§2 Cause A** — conftest one-line; unblocks 5 tests at once. Do
   first; it's the smallest fix with the biggest blast radius.
2. **§3 Cause B** — registry one-line. Trivial after A.
3. **§4 Cause C** — needs author intent confirmation, smallest scope
   but slowest decision; do last.

Total: ~25 minutes of work plus the Z5 P2 intent check.

---

## §6 — Verification gate after all 3 land

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/ -q
```
Expect: **732 passed, 2 skipped, 0 failed.**
(725 currently passing + 7 newly green.)

If any test regresses, the change either weakened a guard (Cause A
conftest edits should be additive only) or the synthetic_check tag is
wrong (re-read the dispatcher block — see §3 fix).

---

## §7 — Things this handoff does NOT touch

- The hang fix from `46d5a334` is intact — conftest still at
  `packages/mr_roboto/tests/conftest.py`, `INLINE_TIMEOUT=3` +
  `KUTAI_CRITIC_GATE=off` autouse fixture present.
- The 4 pre-existing reds I closed in `46d5a334` (notify_user mock,
  vision_multi loops, verify_artifacts router, reversibility detector)
  are all still green.
- Pre-existing main-branch failures listed in
  `2026-05-19-wiring-sweep-residuals.md` §3 (founder_actions/lifecycle,
  z5_t5_distribution, z6_t6a_reversibility, intake_todo, repo
  list_by_mission_orders_desc) are separate — owners flagged there.
