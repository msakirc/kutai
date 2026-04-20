"""Tests for simulator runner (Phase 2d)."""
from types import SimpleNamespace

from sim.state import SimState, SimLocalModel, SimPoolCounter
from sim.runner import SimTask, SimRun, run_simulation


def _fake_select(state, task):
    """Always pick 'loaded-local'. Deterministic stub for testing runner mechanics."""
    return SimpleNamespace(
        model_name="loaded-local",
        pool="local",
        estimated_output_tokens=1000,
        tokens_per_second=20.0,
    )


def _fake_snapshot_factory(state):
    return SimpleNamespace(
        local=SimpleNamespace(
            model_name="loaded-local",
            idle_seconds=state.locals.get("loaded-local", SimLocalModel()).idle_seconds,
            measured_tps=20.0,
            requests_processing=0,
            thinking_enabled=False,
        ),
        cloud={},
    )


def test_runner_records_pick_per_task():
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=100.0)
    tasks = [SimTask(idx=i, difficulty=3) for i in range(5)]
    run: SimRun = run_simulation(
        tasks=tasks,
        initial_state=state,
        select_fn=_fake_select,
        snapshot_factory=_fake_snapshot_factory,
    )
    assert len(run.picks) == 5
    assert all(p.model_name == "loaded-local" for p in run.picks)


def test_runner_advances_clock_per_pick():
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True)
    tasks = [SimTask(idx=i, difficulty=3, estimated_output_tokens=1000) for i in range(3)]
    run: SimRun = run_simulation(
        tasks=tasks,
        initial_state=state,
        select_fn=_fake_select,
        snapshot_factory=_fake_snapshot_factory,
    )
    # 1000 tokens / 20 tps = 50s per task, 3 tasks = 150s
    assert run.final_state.virtual_clock == 150.0


def test_runner_resets_used_local_idle():
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    tasks = [SimTask(idx=0, difficulty=3, estimated_output_tokens=1000)]
    run: SimRun = run_simulation(
        tasks=tasks,
        initial_state=state,
        select_fn=_fake_select,
        snapshot_factory=_fake_snapshot_factory,
    )
    # Used → idle resets
    assert run.final_state.locals["loaded-local"].idle_seconds == 0.0
