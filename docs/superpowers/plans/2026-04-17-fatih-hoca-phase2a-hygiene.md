# Fatih Hoca Phase 2a — Scoring Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate two scoring-hygiene issues in Fatih Hoca surfaced by Phase 1: the `asyncio.get_event_loop()` deprecation in pick telemetry, and the 1.15× specialty-bonus multiplier that now double-counts with blended benchmark+profile signal.

**Architecture:** Two fully independent changes, both confined to `packages/fatih_hoca/`. Task 1 replaces deprecated asyncio API in `selector.py` with forward-compatible `get_running_loop()` pattern (sync callers silently skip telemetry instead of attempting to run a loop). Task 2 deletes the `composite *= 1.15` specialty bonus in `ranking.py` — the `specialty` field stays on `ModelInfo` for eligibility filtering but no longer bends the composite score.

**Tech Stack:** Python 3.10+, pytest (sync + `pytest-asyncio`), existing fatih_hoca package. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-17-fatih-hoca-phase2a-hygiene.md` (commit `cab2053`).

---

## File Structure

**Modified:**
- `packages/fatih_hoca/src/fatih_hoca/selector.py:295–302` (Task 1) — replace `get_event_loop()` block
- `packages/fatih_hoca/src/fatih_hoca/ranking.py:412–422` (Task 2) — drop `composite *= 1.15`, keep reason string

**Modified (tests):**
- `tests/fatih_hoca/test_pick_telemetry.py` (Task 1) — add sync-context test
- `tests/fatih_hoca/test_scoring_hygiene.py` (Task 2) — add TestSpecialtyBonusRemoval class

**No new files.** No touching of `src/core/`, `src/app/`, or any other package.

---

## Task 0: Tighten telemetry write guard (emergency — Phase 1 regression)

**Context:** During Phase 2a development, `model_pick_log` in `C:\Users\sakir\ai\kutai\kutai.db` was found polluted with ~184 rows of test-fixture model names (`a`, `b`, `good`, `alpha`, `thinker`, etc.). Root cause: `asyncio.create_task(_write())` in `_persist_pick_telemetry` schedules the DB write on the event loop. In tests, the write runs **after** `monkeypatch.setenv("DB_PATH", ...)` has reverted via fixture teardown. When it eventually runs, `os.getenv("DB_PATH")` reads the OS-level production value (set system-wide on this Windows host). The existing guard (`if "src.app.config" not in sys.modules: return`) doesn't help — OS env is set independently of module imports.

**Consequence in production:** orchestrator picked a fake-path model from polluted telemetry → tried to load `/fake/a.gguf` → failed load → 5-min demotion → user saw `🟠 [ERROR] models.local_model_manager Failed to load a (demoted for 5 min)`. Polluted rows already purged; this task prevents recurrence.

**Fix:** require explicit production opt-in via `enable_telemetry(db_path)`. Default `_telemetry_db_path = None` → skip write. Tests monkeypatch the module attribute directly.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py`
- Modify: `src/app/run.py` (one-line opt-in at startup)
- Modify: `tests/fatih_hoca/test_pick_telemetry.py` (new leak test + update existing)

- [ ] **Step 1: Read current `_write` block**

```bash
sed -n '245,305p' packages/fatih_hoca/src/fatih_hoca/selector.py
```

- [ ] **Step 2: Write the failing leak test**

Append to `tests/fatih_hoca/test_pick_telemetry.py`:

```python
@pytest.mark.asyncio
async def test_telemetry_does_not_leak_when_os_env_has_db_path(tmp_path, monkeypatch):
    """Repro for 2026-04-17 production pollution: OS-level DB_PATH must not
    drive telemetry writes. Requires explicit enable_telemetry() opt-in."""
    prod_db = tmp_path / "fake_production.db"
    monkeypatch.setenv("DB_PATH", str(prod_db))

    from src.infra.db import init_db
    await init_db()

    import fatih_hoca
    from fatih_hoca.registry import ModelInfo, ModelRegistry
    from fatih_hoca.selector import Selector

    fatih_hoca._registry = None
    fatih_hoca._selector = None
    reg = ModelRegistry()
    reg._models["leaky_fixture"] = ModelInfo(
        name="leaky_fixture", location="local",
        provider="llama_cpp", litellm_name="openai/leaky_fixture",
        path="/fake/leaky_fixture.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 7.0 for c in ["reasoning","code_generation","analysis","instruction_adherence"]},
    )

    class _Nh:
        def snapshot(self):
            from nerd_herd.types import SystemSnapshot
            return SystemSnapshot()
    fatih_hoca._registry = reg
    fatih_hoca._selector = Selector(registry=reg, nerd_herd=_Nh())

    fatih_hoca.select(
        task="coder", agent_type="coder", difficulty=5,
        estimated_input_tokens=500, estimated_output_tokens=500,
        call_category="main_work",
    )

    import asyncio
    await asyncio.sleep(0.2)

    import sqlite3
    conn = sqlite3.connect(prod_db)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM model_pick_log WHERE picked_model='leaky_fixture'"
        )
        count = cur.fetchone()[0]
    finally:
        conn.close()

    assert count == 0, (
        f"telemetry leaked: {count} rows. tests must not write without "
        f"explicit enable_telemetry() opt-in."
    )
```

- [ ] **Step 3: Run the test — expect FAIL**

```bash
cd ".worktrees/fatih-hoca-phase2a" && source ../../.venv/Scripts/activate
timeout 30 python -m pytest tests/fatih_hoca/test_pick_telemetry.py::test_telemetry_does_not_leak_when_os_env_has_db_path -v
```

Expected: FAIL (`telemetry leaked: 1 rows for leaky_fixture`).

- [ ] **Step 4: Add module-level marker + helpers in `selector.py`**

After imports, before the `Selector` class:

```python
# ─── Telemetry DB Path (explicit opt-in for production) ──────────────────────
#
# Tests must never leak to the real DB. The previous guard relied on
# os.getenv("DB_PATH") + "src.app.config in sys.modules" — but the OS env
# var is set on the production host independently of module imports, so
# tests that inherit it would still write (see 2026-04-17 incident).
_telemetry_db_path: str | None = None


def enable_telemetry(db_path: str) -> None:
    """Opt telemetry in. Call once at production startup."""
    global _telemetry_db_path
    _telemetry_db_path = db_path


def disable_telemetry() -> None:
    """Opt telemetry out (test teardown)."""
    global _telemetry_db_path
    _telemetry_db_path = None
```

- [ ] **Step 5: Rewrite the `_write` DB-resolution block**

Find (around line 248–265):

```python
                import os
                import sys
                import aiosqlite
                db_path = os.getenv("DB_PATH")
                if not db_path:
                    cfg_mod = sys.modules.get("src.app.config")
                    if cfg_mod is None:
                        return
                    db_path = getattr(cfg_mod, "DB_PATH", None)
                if not db_path:
                    return
```

Replace with:

```python
                import aiosqlite
                db_path = _telemetry_db_path
                if not db_path:
                    return
```

- [ ] **Step 6: Update `test_select_persists_pick_to_db`**

Before the `with caplog.at_level(...)` block, add:

```python
    from fatih_hoca import selector as _sel_mod
    monkeypatch.setattr(_sel_mod, "_telemetry_db_path", str(tmp_path / "test.db"))
```

Remove the now-unused `monkeypatch.setenv("DB_PATH", ...)` line from that test.

- [ ] **Step 7: Wire production opt-in in `src/app/run.py`**

```bash
grep -n "def main\|init_db\|DB_PATH\|load_dotenv" src/app/run.py | head
```

After `init_db()` completes, add:

```python
    try:
        from fatih_hoca.selector import enable_telemetry
        from src.app.config import DB_PATH as _DB_PATH
        enable_telemetry(_DB_PATH)
    except Exception:
        pass
```

- [ ] **Step 8: Re-run leak test — expect PASS**

```bash
timeout 30 python -m pytest tests/fatih_hoca/test_pick_telemetry.py::test_telemetry_does_not_leak_when_os_env_has_db_path -v
```

- [ ] **Step 9: Re-run existing persist test**

```bash
timeout 30 python -m pytest tests/fatih_hoca/test_pick_telemetry.py::test_select_persists_pick_to_db -v
```

Expected: PASS.

- [ ] **Step 10: Full suite regression check**

```bash
timeout 120 python -m pytest packages/fatih_hoca/tests/ tests/fatih_hoca/ tests/test_benchmark_fetcher.py -q 2>&1 | tail -5
```

Expected: 206 passed (Phase 1 baseline 205 + new leak test).

- [ ] **Step 11: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py src/app/run.py tests/fatih_hoca/test_pick_telemetry.py
git commit -m "fix(fatih_hoca): telemetry writes require explicit enable_telemetry() opt-in"
```

---

## Task 1: Replace deprecated `asyncio.get_event_loop()`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py:295–302`
- Modify: `tests/fatih_hoca/test_pick_telemetry.py` (add new test case)

- [ ] **Step 1: Write failing test for sync-context behavior**

Append to `tests/fatih_hoca/test_pick_telemetry.py` (NOT wrapped in `@pytest.mark.asyncio` — this test runs in a pure sync context so there's no running loop):

```python
def test_select_in_sync_context_skips_telemetry_cleanly(tmp_path, monkeypatch):
    """When fatih_hoca.select() is invoked from a sync caller with no event loop
    running, pick telemetry must be silently skipped — no crash, no DeprecationWarning,
    no DB row. This is the path Python 3.12+ will break if we keep get_event_loop().
    """
    import warnings
    monkeypatch.setenv("DB_PATH", str(tmp_path / "sync_test.db"))

    # We need the DB schema present so that IF telemetry accidentally wrote, we'd see it.
    import asyncio
    from src.infra.db import init_db
    asyncio.run(init_db())

    import fatih_hoca
    from fatih_hoca.registry import ModelInfo, ModelRegistry
    from fatih_hoca.selector import Selector

    fatih_hoca._registry = None
    fatih_hoca._selector = None

    reg = ModelRegistry()
    reg._models["a"] = ModelInfo(
        name="a", location="local",
        provider="llama_cpp", litellm_name="openai/a",
        path="/fake/a.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 7.0 for c in ["reasoning", "code_generation", "analysis", "instruction_adherence"]},
    )

    class _Nh:
        def snapshot(self):
            from nerd_herd.types import SystemSnapshot
            return SystemSnapshot()

    fatih_hoca._registry = reg
    fatih_hoca._selector = Selector(registry=reg, nerd_herd=_Nh())

    # Sync call — no asyncio.run wrapper.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pick = fatih_hoca.select(
            task="coder", agent_type="coder", difficulty=5,
            estimated_input_tokens=500, estimated_output_tokens=500,
            call_category="main_work",
        )

    assert pick is not None, "select() must return a Pick even when telemetry is skipped"
    assert pick.model.name == "a"

    # Verify no row was written (telemetry skipped in sync context is the contract)
    import sqlite3
    conn = sqlite3.connect(tmp_path / "sync_test.db")
    try:
        cur = conn.execute("SELECT COUNT(*) FROM model_pick_log")
        count = cur.fetchone()[0]
    finally:
        conn.close()
    assert count == 0, f"sync-context select() must not write telemetry, got {count} rows"
```

- [ ] **Step 2: Run the test — expect failure**

```bash
cd ".worktrees/fatih-hoca-phase2a" && source ../../.venv/Scripts/activate
timeout 30 python -m pytest tests/fatih_hoca/test_pick_telemetry.py::test_select_in_sync_context_skips_telemetry_cleanly -v
```

Expected: FAIL. The current code calls `asyncio.get_event_loop()` which in Python 3.10+ emits `DeprecationWarning` when no loop is running (our `warnings.simplefilter("error", DeprecationWarning)` promotes that to a raised exception). The test may also fail via `run_until_complete` path writing a row.

- [ ] **Step 3: Apply the fix in `selector.py:295–302`**

Read the current block first to confirm exact content:

```bash
sed -n '293,305p' packages/fatih_hoca/src/fatih_hoca/selector.py
```

Expected current content:

```python
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_write())
            else:
                loop.run_until_complete(_write())
        except RuntimeError:
            pass  # no event loop available — skip this call
```

Replace with:

```python
        try:
            asyncio.get_running_loop()
            asyncio.create_task(_write())
        except RuntimeError:
            # No running loop — sync caller. Telemetry is best-effort and
            # only meaningful inside the orchestrator's asyncio context;
            # skip silently rather than stall on a background loop.
            pass
```

Do not change anything above or below this block. The `_write` coroutine and its encompassing helper remain untouched.

- [ ] **Step 4: Re-run the sync-context test**

```bash
timeout 30 python -m pytest tests/fatih_hoca/test_pick_telemetry.py::test_select_in_sync_context_skips_telemetry_cleanly -v
```

Expected: PASS.

- [ ] **Step 5: Re-run the existing async-context test**

```bash
timeout 30 python -m pytest tests/fatih_hoca/test_pick_telemetry.py::test_select_persists_pick_to_db -v
```

Expected: PASS (unchanged behavior in the async path — `get_running_loop()` succeeds there, `create_task` scheduled exactly as before).

- [ ] **Step 6: Run the full fatih_hoca suite with DeprecationWarning promoted to error**

```bash
timeout 120 python -W error::DeprecationWarning -m pytest packages/fatih_hoca/tests/ tests/fatih_hoca/ -q 2>&1 | tail -6
```

Expected: all tests pass. If an unrelated DeprecationWarning surfaces (e.g., from `pynvml`), the test will fail — that's out of Task 1 scope; narrow the filter:

```bash
timeout 120 python -W "error::DeprecationWarning:fatih_hoca" -m pytest packages/fatih_hoca/tests/ tests/fatih_hoca/ -q 2>&1 | tail -6
```

If that still fails, the code has a remaining fatih_hoca-internal deprecation to track — report it and leave as a follow-up.

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py tests/fatih_hoca/test_pick_telemetry.py
git commit -m "fix(fatih_hoca): replace deprecated asyncio.get_event_loop with get_running_loop"
```

---

## Task 2: Remove specialty-bonus composite multiplier

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py:412–422`
- Modify: `tests/fatih_hoca/test_scoring_hygiene.py` (add `TestSpecialtyBonusRemoval` class)

**Context:** The current code at `ranking.py:412–422`:

```python
        # Group B: Specialty alignment
        if model.specialty and effective_task:
            _specialty_tasks = {
                "coding": {"coder", "implementer", "fixer", "test_generator"},
                "research": {"researcher", "analyst"},
                "vision": {"visual_reviewer"},
            }
            matched = _specialty_tasks.get(model.specialty, set())
            if effective_task in matched:
                composite *= 1.15
                reasons.append(f"specialty={model.specialty}")
```

After Phase 1 blended AA benchmark signal into `ModelInfo.capabilities`, this multiplier double-counts: a coding-specialty model on a coder task already gets high `code_generation` and `code_reasoning` via the profile dot-product and now via benchmarks. The 1.15× stacks on top, beating general models that have objectively better benchmark coder scores.

- [ ] **Step 1: Write the failing test — specialty bonus must not tip the winner**

Append to `tests/fatih_hoca/test_scoring_hygiene.py`:

```python
class TestSpecialtyBonusRemoval:
    """After Phase 2a, the 1.15× specialty multiplier is gone. The specialty
    field is kept for eligibility filtering (in selector.py) and observability
    reasons — but it must not bend the composite score."""

    def test_equal_caps_specialty_does_not_win(self):
        """Two models with identical capabilities — one specialty=coding, one None
        on a coder task. They must tie on composite. Before removal, specialty
        model would get 1.15× bonus."""
        from fatih_hoca.ranking import rank_candidates
        from fatih_hoca.requirements import ModelRequirements
        from nerd_herd.types import SystemSnapshot

        caps = {"reasoning": 7.0, "code_generation": 7.0, "code_reasoning": 7.0,
                "analysis": 6.5, "instruction_adherence": 7.0}
        specialist = _make_model("specialist", caps, specialty="coding")
        general = _make_model("general", caps)  # specialty=None default

        # Speed parity so the test isolates specialty-bonus behavior.
        specialist.tokens_per_second = 20.0
        general.tokens_per_second = 20.0

        reqs = ModelRequirements(
            primary_capability="code_generation",
            difficulty=5,
            estimated_input_tokens=500,
            estimated_output_tokens=500,
        )
        reqs.effective_task = "coder"  # triggers specialty matching path

        snap = SystemSnapshot()
        ranked = rank_candidates([specialist, general], reqs, snap, failures=[])
        by_name = {r.model.name: r for r in ranked}

        specialist_score = by_name["specialist"].composite
        general_score = by_name["general"].composite

        # Must be equal (no bonus to tip the balance)
        assert abs(specialist_score - general_score) < 0.01, (
            f"specialty must not bend composite: specialist={specialist_score:.3f} "
            f"vs general={general_score:.3f} — bonus not removed"
        )

        # Specialty reason may still appear in reasons (observability only, no *= operation)
        # but there must not be a 1.15× multiplier residue.

    def test_benchmark_signal_beats_specialty_tag(self):
        """A general model with strong AA coder caps must beat a specialty
        coding model with weaker caps on a coder task. This is the core
        business outcome Phase 2a enables."""
        from fatih_hoca.ranking import rank_candidates
        from fatih_hoca.requirements import ModelRequirements
        from nerd_herd.types import SystemSnapshot

        weak_caps = {"reasoning": 6.0, "code_generation": 5.5, "code_reasoning": 5.5,
                     "analysis": 6.0, "instruction_adherence": 6.5}
        strong_caps = {"reasoning": 7.5, "code_generation": 8.5, "code_reasoning": 8.0,
                       "analysis": 7.0, "instruction_adherence": 7.0}

        weak_specialist = _make_model("weak_coder_specialist", weak_caps, specialty="coding")
        strong_general = _make_model("strong_general", strong_caps)

        weak_specialist.tokens_per_second = 20.0
        strong_general.tokens_per_second = 20.0

        reqs = ModelRequirements(
            primary_capability="code_generation",
            difficulty=6,
            estimated_input_tokens=500,
            estimated_output_tokens=500,
        )
        reqs.effective_task = "coder"

        snap = SystemSnapshot()
        ranked = rank_candidates([weak_specialist, strong_general], reqs, snap, failures=[])

        winner = ranked[0].model.name
        assert winner == "strong_general", (
            f"benchmark signal must dominate specialty tag: winner={winner}, "
            f"scores={[(r.model.name, round(r.composite, 2)) for r in ranked]}"
        )
```

- [ ] **Step 2: Run the failing tests**

```bash
cd ".worktrees/fatih-hoca-phase2a" && source ../../.venv/Scripts/activate
timeout 30 python -m pytest tests/fatih_hoca/test_scoring_hygiene.py::TestSpecialtyBonusRemoval -v
```

Expected: both fail. `test_equal_caps_specialty_does_not_win` fails because the specialist gets a 1.15× bonus making its composite ~15% higher. `test_benchmark_signal_beats_specialty_tag` may or may not fail depending on how large the capability gap is vs the 1.15× bonus — if it passes, that's fine (the stronger assertion is the first one).

- [ ] **Step 3: Read the current specialty block to confirm exact content**

```bash
sed -n '410,425p' packages/fatih_hoca/src/fatih_hoca/ranking.py
```

Expected:

```python
        # Group B: Specialty alignment
        if model.specialty and effective_task:
            _specialty_tasks = {
                "coding": {"coder", "implementer", "fixer", "test_generator"},
                "research": {"researcher", "analyst"},
                "vision": {"visual_reviewer"},
            }
            matched = _specialty_tasks.get(model.specialty, set())
            if effective_task in matched:
                composite *= 1.15
                reasons.append(f"specialty={model.specialty}")
```

- [ ] **Step 4: Remove the multiplier; keep observability**

Replace that block with:

```python
        # Group B: Specialty alignment (observability only, no composite effect).
        # The 1.15× multiplier was removed in Phase 2a: after Phase 1 blended AA
        # benchmark signal into ModelInfo.capabilities, a coding-specialty model
        # on a coder task already gets its code_* dims heavily boosted via the
        # profile dot-product; multiplying again caused specialty models to beat
        # general models with objectively stronger benchmark coder scores.
        # Hard filtering for coding specialty on non-coder tasks still lives
        # in selector._check_eligibility (line ~251), which is unchanged.
        if model.specialty and effective_task:
            _specialty_tasks = {
                "coding": {"coder", "implementer", "fixer", "test_generator"},
                "research": {"researcher", "analyst"},
                "vision": {"visual_reviewer"},
            }
            matched = _specialty_tasks.get(model.specialty, set())
            if effective_task in matched:
                reasons.append(f"specialty={model.specialty}")
```

- [ ] **Step 5: Re-run the specialty tests**

```bash
timeout 30 python -m pytest tests/fatih_hoca/test_scoring_hygiene.py::TestSpecialtyBonusRemoval -v
```

Expected: both pass.

- [ ] **Step 6: Run the full fatih_hoca ranking + E2E suite**

```bash
timeout 120 python -m pytest packages/fatih_hoca/tests/test_ranking.py tests/fatih_hoca/ -q
```

Expected: all pass. The Phase 1 E2E test (`test_aa_signal_promotes_dense_32b_for_coder_over_base_a3b`) does not use specialty (neither seeded model has `specialty="coding"`), so it should be unaffected.

**If an existing `packages/fatih_hoca/tests/test_ranking.py` test fails** because it asserted the specialty bonus boosted a specialist by ~15%, that test was measuring the bonus itself. Options:
- Update the test to assert the specialist + general tie (new correct behavior).
- If the test's intent was "specialty models should beat general models on specialty tasks" — that's the exact business behavior we're rejecting. Convert it to a TODO-commented skip with a reference to this plan, or delete it entirely if no one else relies on it.
- Document any test change in the commit message.

Do NOT reintroduce the multiplier to make a test pass.

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py tests/fatih_hoca/test_scoring_hygiene.py
git commit -m "fix(fatih_hoca): remove 1.15× specialty bonus — double-counts with blended benchmark signal"
```

---

## Task 3: Verification sweep

**Files:** none modified (sweep only).

- [ ] **Step 1: Run the full suite with deprecation warnings as errors**

```bash
cd ".worktrees/fatih-hoca-phase2a" && source ../../.venv/Scripts/activate
timeout 300 python -W "error::DeprecationWarning:fatih_hoca" -m pytest packages/fatih_hoca/tests/ tests/fatih_hoca/ tests/test_benchmark_fetcher.py -q 2>&1 | tail -6
```

Expected: 207+ passed (Phase 1 baseline 205 + 3 new tests from Tasks 1 & 2). No deprecation warnings from fatih_hoca.

- [ ] **Step 2: Smoke-test the init flow to confirm nothing regressed**

```bash
cd ".worktrees/fatih-hoca-phase2a" && python -c "
import os, sys
# Resolve paths relative to the main-repo root (shares .benchmark_cache/)
root = '../..'
sys.path.insert(0, root)
os.chdir(root)
import fatih_hoca
names = fatih_hoca.init(
    models_dir='models' if os.path.exists('models') else None,
    catalog_path='src/models/models.yaml' if os.path.exists('src/models/models.yaml') else None,
)
print(f'registered {len(names)} models')
for m in fatih_hoca.all_models()[:5]:
    bench = bool(m.benchmark_scores)
    code = m.capabilities.get('code_generation', 0)
    print(f'  {m.name}: benchmark={bench} code_gen={code:.2f} specialty={m.specialty or \"-\"}')
" 2>&1 | tail -10
```

Expected: non-zero model count, at least some `benchmark=True`, log output includes `benchmark coverage: N/M matched`.

- [ ] **Step 3: Final commit if the sweep surfaced any doc touchups**

None expected. If Task 2's commit message already documents the behavior change, no additional docs change needed. If you'd like to note the bonus removal in CLAUDE.md's Fatih Hoca bullet, add a parenthetical mention:

```
- **Fatih Hoca** owns all model knowledge: catalog (YAML+GGUF), benchmark enrichment ... 15-dimension scoring (no specialty bonus — benchmark signal dominates; specialty used for eligibility filtering only), task profiles, ...
```

If adding that tweak, commit:

```bash
git add CLAUDE.md
git commit -m "docs: note specialty-bonus removal in Fatih Hoca summary"
```

- [ ] **Step 4: Ship-readiness check**

```bash
git log --oneline -5
```

Expected: 2–3 commits from this plan (Task 1, Task 2, optional CLAUDE.md touch).

---

## Post-plan handoff

After the branch merges to main:

1. The pre-Phase-2a Python 3.12+ risk is gone. `python -W error::DeprecationWarning` runs clean on fatih_hoca.
2. Selection quality for coder tasks: general models with strong AA coder benchmarks will now rise above specialty-tagged models that have weaker actual benchmark scores. Monitor `model_pick_log` — specifically `SELECT picked_model, picked_score, picked_reasons FROM model_pick_log WHERE task_name='coder' ORDER BY timestamp DESC LIMIT 20` to spot-check.
3. If coder picks regress subjectively, don't reinstate the 1.15× multiplier. The correct fix is to (a) verify benchmark coverage on your specialty models (they may be missing AA entries in which case profile-only fallback underscores them), or (b) bump the profile weight on `code_generation`/`code_reasoning` in `TASK_PROFILES["coder"]`. The whole point of Phase 2a is making signal routing explicit rather than hidden in a multiplier.
4. The deferred Phase 2b item (`tick_benchmark_refresh` in `src/app/scheduled_jobs.py`) can now land independently — it touches orchestrator scheduling only and has no Fatih Hoca dependencies.
