"""Runnable scenario harness — prints the equilibrium table for all 7 + 8 scenarios.

Future agents: this is the canonical way to see the Phase 2d utilization
equation in action. Run from the repo root (or any worktree):

    /c/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe \
        packages/fatih_hoca/tests/sim/run_scenarios.py

Output columns (Phase 2d scenarios):
  hard    % of d≥7 tasks where picked cap_score ≥ cap_needed_for_difficulty
  waste   % of d≤4 tasks where a per_call (paid) model picked with fit_excess>0.4
          (free-pool over-qualified picks don't count — they are intended burn)
  free_q  avg across time_bucketed pools of min(1.0, picks_to_pool / pool.limit)
  picks   most-picked models with counts

Pool-pressure scenarios (Task 28) print PASS/FAIL per assertion.

No DB writes, no llama-server launches, no network. Pure in-memory simulation.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


def _setup_paths() -> None:
    """Make sim.* and fatih_hoca.* importable when run as a script."""
    worktree = Path(__file__).resolve().parents[4]
    src = worktree / "packages" / "fatih_hoca" / "src"
    tests = worktree / "packages" / "fatih_hoca" / "tests"
    nerd_herd_src = worktree / "packages" / "nerd_herd" / "src"
    for p in (str(src), str(tests), str(nerd_herd_src)):
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_paths()

from sim.scenarios import (  # noqa: E402
    baseline,
    claude_constrained,
    groq_near_reset,
    diverse_pool,
    exhaustion_sequence,
    back_to_back_i2p,
    staggered_i2p,
    POOL_PRESSURE_SCENARIOS,
    POOL_PRESSURE_ASSERTIONS,
)
from sim.runner import run_simulation  # noqa: E402
from sim.report import compute_metrics  # noqa: E402


SCENARIOS = [
    ("baseline", baseline),
    ("claude_constrained", claude_constrained),
    ("groq_near_reset", groq_near_reset),
    ("diverse_pool", diverse_pool),
    ("exhaustion_sequence", exhaustion_sequence),
    ("back_to_back_i2p", back_to_back_i2p),
    ("staggered_i2p", staggered_i2p),
]


def main() -> int:
    # ── Phase 2d scenarios ────────────────────────────────────────────────────
    print(f"{'scenario':24s} {'hard':>6s} {'waste':>6s} {'free_q':>7s}  picks")
    print(f"{'-' * 24} {'-' * 6} {'-' * 6} {'-' * 7}  {'-' * 40}")
    for name, factory in SCENARIOS:
        scenario = factory()
        run = run_simulation(
            tasks=scenario.tasks,
            initial_state=scenario.initial_state,
            select_fn=scenario.select_fn,
            snapshot_factory=scenario.snapshot_factory,
        )
        m = compute_metrics(run)
        picks = Counter(p.model_name for p in run.picks)
        picks_str = " ".join(
            f"{k.split('/')[-1]}={v}" for k, v in picks.most_common()
        )
        print(
            f"{name:24s} "
            f"{m.hard_task_satisfaction:>6.1%} "
            f"{m.easy_task_waste:>6.1%} "
            f"{m.free_quota_utilization:>7.1%}  "
            f"{picks_str}"
        )

    # ── Pool-pressure scenarios (Task 28) ─────────────────────────────────────
    print()
    print("Pool-pressure scenarios (Task 28)")
    print(f"{'scenario':36s} {'result':>6s}  details")
    print(f"{'-' * 36} {'-' * 6}  {'-' * 50}")
    all_pp_passed = True
    for name, factory in POOL_PRESSURE_SCENARIOS:
        scenario = factory()
        assert_fn = POOL_PRESSURE_ASSERTIONS.get(name)
        if assert_fn is None:
            print(f"{name:36s} {'SKIP':>6s}")
            continue
        try:
            failures = assert_fn(scenario)
        except Exception as exc:
            failures = [f"EXCEPTION: {exc}"]
        if failures:
            all_pp_passed = False
            result = "FAIL"
            detail = "; ".join(failures[:2])
        else:
            result = "PASS"
            detail = ""
        print(f"{name:36s} {result:>6s}  {detail}")

    print()
    if all_pp_passed:
        print("All pool-pressure scenarios PASSED.")
        return 0
    else:
        print("One or more pool-pressure scenarios FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
