"""Swap-storm diagnostic — exercises real-registry locals with cloud disabled.

Verifies that stickiness's original purpose (preventing swap thrashing
between close-capability locals) still works under Phase 2d's dialed-down
stickiness magnitude (1.10×). Does NOT run via pytest because it needs the
real GGUF registry from `src/models/models.yaml` + `$MODEL_DIR`, which is a
machine-specific setup. Safe to run — no llama-server launches, no DB
writes, no network.

Usage (from repo root OR a worktree):

    MODEL_DIR=/c/Users/sakir/ai/models \
        python packages/fatih_hoca/tests/sim/run_swap_storm_check.py

Set `MODEL_DIR` to the directory containing GGUF files. If not set, falls
back to `C:\\Users\\sakir\\ai\\models`.

Output: per-starting-loaded-model, how many swaps happened across 200
synthetic tasks drawn from the i2p v3 difficulty distribution.

Pass criteria (manual inspection):
- Swap rate should stay under ~5% (the storm-prevention threshold).
- Grossly-under-qualified loaded models should swap exactly once on the
  first hard-enough task, then stay put.
- Well-fit loaded models should almost never swap.
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def _setup_paths() -> Path:
    """Return repo root + inject fatih_hoca src into sys.path."""
    here = Path(__file__).resolve()
    worktree = here.parents[4]
    src = worktree / "packages" / "fatih_hoca" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return worktree


def _resolve_repo_root(worktree: Path) -> Path:
    """Prefer worktree, fall back to main repo for models.yaml."""
    if (worktree / "src" / "models" / "models.yaml").exists():
        return worktree
    # Most common case: we're in a worktree. Main repo is two levels up
    # of `.worktrees/<branch>/`.
    main_repo = worktree.parent.parent
    if (main_repo / "src" / "models" / "models.yaml").exists():
        return main_repo
    raise FileNotFoundError(
        f"Could not locate src/models/models.yaml from {worktree} "
        f"or {main_repo}"
    )


def _collect_pool(registry) -> list[str]:
    """Return 7 general-purpose locals spanning the cap range."""
    wanted = [
        "Qwen3.5-9B-UD-Q4_K_XL",
        "GigaChat3.1-Lightning-Uncensored.i1-Q4_K_M",
        "gpt-oss-20b-UD-Q4_K_XL",
        "gemma-4-26B-A4B-it-UD-IQ4_NL",
        "Qwen3.5-27B.Q4_K_M",
        "Qwen3.5-35B-A3B-UD-Q4_K_XL",
        "GLM-4.7-Flash-UD-Q4_K_XL",
    ]
    return [n for n in wanted if registry.get(n) is not None]


def _make_candidate(model_info, is_loaded: bool):
    """Clone a ModelInfo into a mutable namespace for ranking."""
    attrs = {
        k: getattr(model_info, k)
        for k in dir(model_info)
        if not k.startswith("_") and not callable(getattr(model_info, k))
    }
    c = SimpleNamespace(**attrs)
    c.operational_dict = model_info.operational_dict
    c.estimated_cost = model_info.estimated_cost
    c.is_loaded = is_loaded
    return c


def _snapshot(loaded_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        local=SimpleNamespace(
            model_name=loaded_name,
            idle_seconds=300.0,
            measured_tps=15.0,
            thinking_enabled=False,
            requests_processing=0,
        ),
        cloud={},
    )


def _run(registry, pool_names: list[str], start_loaded: str, n_tasks: int = 200):
    from fatih_hoca.ranking import rank_candidates
    from fatih_hoca.requirements import ModelRequirements, QueueProfile, get_quota_planner

    get_quota_planner().set_queue_profile(QueueProfile())
    loaded = start_loaded
    rng = random.Random(42)
    swaps = 0
    picks: Counter[str] = Counter()
    swap_events: list[tuple[int, int, str, str]] = []

    for i in range(n_tasks):
        r = rng.random()
        if r < 0.50:
            d = rng.choice([1, 2, 3])
        elif r < 0.80:
            d = rng.choice([4, 5, 6])
        elif r < 0.95:
            d = rng.choice([7, 8])
        else:
            d = rng.choice([9, 10])

        candidates = [
            _make_candidate(registry.get(n), is_loaded=(n == loaded))
            for n in pool_names
        ]
        reqs = ModelRequirements(
            task="coder", difficulty=d, estimated_output_tokens=1500
        )
        scored = rank_candidates(
            candidates=candidates,
            reqs=reqs,
            snapshot=_snapshot(loaded),
            failures=[],
            remaining_budget=300.0,
        )
        if not scored:
            continue
        top = scored[0].model.name
        picks[top] += 1
        if top != loaded:
            swaps += 1
            swap_events.append((i, d, loaded[:20], top[:20]))
            loaded = top

    return swaps, picks, swap_events


def main() -> int:
    worktree = _setup_paths()
    repo_root = _resolve_repo_root(worktree)

    os.environ.setdefault(
        "MODEL_DIR", r"C:\Users\sakir\ai\models"
    )
    model_dir = os.environ["MODEL_DIR"]

    from fatih_hoca.registry import ModelRegistry

    reg = ModelRegistry()
    reg.load_yaml(str(repo_root / "src" / "models" / "models.yaml"))
    reg.load_gguf_dir(model_dir)

    pool_names = _collect_pool(reg)
    if len(pool_names) < 3:
        print(
            f"ERROR: only {len(pool_names)} general-purpose locals found. "
            f"Need at least 3 for a meaningful swap test.",
            file=sys.stderr,
        )
        return 1

    print(f"Pool ({len(pool_names)} general-purpose locals, cloud disabled):")
    for n in pool_names:
        m = reg.get(n)
        print(f"  {n:50s}  params={m.total_params_b:5.1f}B")

    print()
    configs = [
        ("healthy start (Qwen-9B)", "Qwen3.5-9B-UD-Q4_K_XL"),
        ("weak start (GigaChat)", "GigaChat3.1-Lightning-Uncensored.i1-Q4_K_M"),
        ("best start (Qwen-27B)", "Qwen3.5-27B.Q4_K_M"),
    ]
    for label, start in configs:
        if start not in pool_names:
            print(f"  [skip] {label}: {start} not available")
            continue
        swaps, picks, swap_events = _run(reg, pool_names, start)
        rate = 100.0 * swaps / 200
        top_picks = {k.split("/")[-1]: v for k, v in picks.most_common(4)}
        print(f"  {label:30s}  swaps={swaps}/200 ({rate:.1f}%)  picks={top_picks}")
        if swap_events:
            for idx, d, src_, dst in swap_events[:3]:
                print(f"    tick={idx:3d} d={d:2d}  {src_:25s} -> {dst}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
