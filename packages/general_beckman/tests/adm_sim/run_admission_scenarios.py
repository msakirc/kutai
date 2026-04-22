"""CLI runner for admission scenarios — mirrors fatih_hoca's run_scenarios.py.

Usage:
    python packages/general_beckman/tests/sim/run_admission_scenarios.py

Prints a table: scenario | admits | unclaimed_end | result.
"""
from __future__ import annotations

import asyncio
import os
import sys


def _setup_paths():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    for pkg in (
        "fatih_hoca", "nerd_herd", "general_beckman",
        "kuleden_donen_var", "dallama", "hallederiz_kadir", "salako",
    ):
        src = os.path.join(root, "packages", pkg, "src")
        if src not in sys.path:
            sys.path.insert(0, src)
    tests_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if tests_root not in sys.path:
        sys.path.insert(0, tests_root)
    if root not in sys.path:
        sys.path.insert(0, root)


_setup_paths()

from adm_sim.runner import run_ticks  # noqa: E402
from adm_sim.scenarios import SCENARIOS  # noqa: E402


async def main():
    print(f"{'scenario':<28} {'admits':>6} {'end_q':>6}  result")
    print("-" * 28, "-" * 6, "-" * 6, "-" * 30, sep="  ")
    any_fail = False
    for sc in SCENARIOS:
        state = sc.state_factory()
        metrics = await run_ticks(state, ticks=sc.ticks)
        err = sc.assertion(metrics, state)
        status = "OK" if err is None else f"FAIL: {err}"
        if err is not None:
            any_fail = True
        print(f"{sc.name:<28} {metrics['admits']:>6} {metrics['unclaimed_at_end']:>6}  {status}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
