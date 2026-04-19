"""Dry-run simulator: walk i2p step profiles through Selector.select().

Outputs a task x model distribution report. No DB writes: telemetry is
opt-in via enable_telemetry() and the simulator never calls it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from fatih_hoca.registry import ModelRegistry
from fatih_hoca.selector import Selector
from nerd_herd.types import LocalModelState, SystemSnapshot


DIFFICULTY_MAP = {"easy": 3, "medium": 5, "hard": 8}

# Walk up from this file to find the repo root, then into src/workflows.
# packages/fatih_hoca/src/fatih_hoca/simulate_i2p.py
#  parents[0] = fatih_hoca/
#  parents[1] = src/
#  parents[2] = fatih_hoca/ (package root)
#  parents[3] = packages/
#  parents[4] = repo root
_DEFAULT_WORKFLOW = (
    Path(__file__).resolve().parents[4] / "src" / "workflows" / "i2p" / "i2p_v3.json"
)


class _FakeNerdHerd:
    """Returns a pinned snapshot so simulation is reproducible.

    Uses the real nerd_herd.types.SystemSnapshot to match what Selector expects.

    Default fixture represents the realistic mid-workflow production state:
    - vram_mb=7000: GPU has ~7 GB free (typical for 8 GB card with model loaded)
    - idle_seconds=300: GPU has been idle 300 s since last inference
      → local urgency = min(1, 300/600) = 0.50 → +12.5% urgency bonus for locals
    - loaded=None: no model marked loaded; pass a model name to trigger
      swap-stickiness (1.40×) in the ranker, which reflects the mid-workflow
      state where a model is already in VRAM between consecutive calls.

    idle_seconds > 0 only makes sense when a model is actually loaded; the
    counter ticks between calls once a model is resident. Combining
    idle_seconds=300 with a loaded model name reproduces the typical production
    scenario where local urgency has maximum observable effect.
    """

    def __init__(
        self,
        vram_mb: int = 7000,
        loaded: str | None = None,
        idle_seconds: float = 300.0,
    ):
        # idle_seconds=300 → urgency=0.5 (halfway to LOCAL_IDLE_SATURATION_SECS=600)
        # This produces a realistic +12.5% urgency bonus for local models so the
        # urgency layer has an observable effect in simulation.  Production sees
        # idle values in the 0-600s range whenever the GPU is between calls.
        local = LocalModelState(model_name=loaded, idle_seconds=idle_seconds)
        self._snapshot = SystemSnapshot(
            vram_available_mb=vram_mb,
            local=local,
        )

    def snapshot(self) -> SystemSnapshot:
        return self._snapshot


def _load_steps(workflow_path: Path) -> list[dict]:
    data = json.loads(workflow_path.read_text(encoding="utf-8"))
    steps = data.get("steps") or data.get("phases") or []
    if not isinstance(steps, list):
        raise ValueError(f"workflow steps is not a list: {type(steps)}")
    return steps


def _find_models_yaml(workflow_path: Path) -> Path | None:
    """Walk up from workflow path (and from this file) to find models.yaml.

    Layout assumption: this file lives at
    packages/fatih_hoca/src/fatih_hoca/simulate_i2p.py
    so parents[4] is the repo root.  The workflow JSON typically lives at
    src/workflows/i2p/i2p_v3.json (3 levels up from that dir → repo root).
    """
    candidates = [
        # Relative to workflow file (e.g. src/workflows/i2p/i2p_v3.json -> repo/src/models/models.yaml)
        workflow_path.parent.parent.parent / "src" / "models" / "models.yaml",
        # Relative to this source file: parents[4] = repo root
        Path(__file__).resolve().parents[4] / "src" / "models" / "models.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def simulate(
    workflow_path: Path | str,
    model_dir: Path | str | None = None,
    loaded_model: str | None = None,
) -> list[dict]:
    """Run each step through Selector.select() and return per-step records.

    Parameters
    ----------
    workflow_path : Path | str
        Path to the workflow JSON (default: i2p_v3.json).
    model_dir : Path | str | None
        Directory of GGUF files to register as local models.
    loaded_model : str | None
        Registry name of the model to simulate as already loaded in VRAM.
        When set, the fake snapshot reports this model as the current
        ``local.model_name`` AND the registry entry is marked ``is_loaded=True``
        (triggering the 1.40× swap-stickiness bonus in the ranker).
        Represents the realistic mid-workflow state where a local model is
        already resident between consecutive i2p calls.
        Example: ``"Qwen3.5-35B-A3B-UD-Q4_K_XL-thinking"``
    """
    workflow_path = Path(workflow_path)
    steps = _load_steps(workflow_path)

    registry = ModelRegistry()
    # ModelRegistry() does NOT auto-load on init; load() uses a hardcoded path
    # relative to registry.py which can be wrong in worktrees. Resolve explicitly.
    models_yaml = _find_models_yaml(workflow_path)
    if models_yaml:
        try:
            registry.load(models_yaml)
        except Exception:
            pass
    else:
        # Fall back to registry default path resolution
        try:
            registry.load()
        except Exception:
            pass

    if model_dir is not None:
        try:
            registry.load_gguf_dir(Path(model_dir))
        except Exception as exc:
            print(f"WARNING: failed to load GGUF dir {model_dir}: {exc}", file=sys.stderr)

    # Mark a model as loaded so the ranker sees swap-stickiness (1.40×) for it.
    # This reproduces the mid-workflow scenario where a model is already in VRAM.
    if loaded_model:
        m = registry.get(loaded_model)
        if m is not None:
            registry.mark_loaded(loaded_model, api_base="http://localhost:8080")
            print(
                f"NOTE: simulating with loaded_model={loaded_model!r} "
                f"(swap-stickiness 1.40× active for this model)",
                file=sys.stderr,
            )
        else:
            print(
                f"WARNING: --loaded-model {loaded_model!r} not found in registry; "
                f"ignoring (no stickiness will apply)",
                file=sys.stderr,
            )

    total = len(list(registry.all_models()))
    local_count = sum(1 for m in registry.all_models() if getattr(m, "is_local", False))
    cloud_count = total - local_count
    if total == 0:
        print(
            "WARNING: registry is empty — no models loaded. "
            "Simulator will produce 100% '<none>' picks. "
            "Check that models.yaml was discoverable.",
            file=sys.stderr,
        )
    elif local_count == 0:
        print(
            f"NOTE: registry has {cloud_count} cloud models and 0 local GGUF models. "
            f"Production may have additional local models that would outscore "
            f"cloud picks on speed+cost for easy/medium steps. Run with "
            f"--model-dir to include local models.",
            file=sys.stderr,
        )

    selector = Selector(
        registry=registry,
        nerd_herd=_FakeNerdHerd(loaded=loaded_model),
    )

    records: list[dict] = []
    for step in steps:
        difficulty_raw = step.get("difficulty", "medium")
        difficulty = DIFFICULTY_MAP.get(
            difficulty_raw if isinstance(difficulty_raw, str) else "medium",
            5,
        )
        agent_type = step.get("agent", "") or ""
        task_name = step.get("name", step.get("id", "unknown"))
        # Production callers pass a TASK_PROFILES key (same string as agent name)
        # for `task`, not the step name. Passing the step name falls back to the
        # "assistant" profile and triggers coding_specialty_mismatch for every step.
        pick = selector.select(
            task=agent_type,
            agent_type=agent_type,
            difficulty=difficulty,
            call_category="main_work",
        )

        if pick is None:
            records.append({
                "step_id": step.get("id", ""),
                "task_name": task_name,
                "agent": agent_type,
                "difficulty": difficulty,
                "picked_model": "<none>",
                "picked_score": 0.0,
                "top3": [],
            })
        else:
            records.append({
                "step_id": step.get("id", ""),
                "task_name": task_name,
                "agent": agent_type,
                "difficulty": difficulty,
                "picked_model": pick.model.name,
                "picked_score": round(getattr(pick.model, "score", 0.0) or 0.0, 2),
                "top3": [],  # Selector.select() returns only top pick; scored list not exposed by API
            })
    return records


def build_report(records: list[dict]) -> dict:
    total = len(records)
    covered = sum(1 for r in records if r["picked_model"] != "<none>")

    pick_counter: Counter[str] = Counter(r["picked_model"] for r in records)
    by_agent: dict[str, Counter] = {}
    by_difficulty: dict[int, Counter] = {}
    for r in records:
        by_agent.setdefault(r["agent"] or "?", Counter())[r["picked_model"]] += 1
        by_difficulty.setdefault(r["difficulty"], Counter())[r["picked_model"]] += 1

    distribution = sorted(pick_counter.items(), key=lambda x: -x[1])
    agent_top = sorted(
        (agent, c.most_common(1)[0][0] if c else "<none>", sum(c.values()))
        for agent, c in by_agent.items()
    )
    difficulty_top = sorted(
        (d, c.most_common(1)[0][0] if c else "<none>", sum(c.values()))
        for d, c in by_difficulty.items()
    )
    return {
        "total_steps": total,
        "coverage": covered,
        "distribution": distribution,
        "by_agent": agent_top,
        "by_difficulty": difficulty_top,
    }


def _format_report(report: dict) -> str:
    lines = [
        f"Total steps: {report['total_steps']}",
        f"Covered: {report['coverage']} "
        f"({report['coverage'] * 100 / max(report['total_steps'], 1):.0f}%)",
        "",
        "- Pick distribution -",
        f"{'model':<40} {'count':>6} {'pct':>6}",
    ]
    total = max(report['total_steps'], 1)
    for model, count in report["distribution"]:
        lines.append(f"{model[:40]:<40} {count:>6} {count * 100 / total:>5.1f}%")
    lines += ["", "- By agent -", f"{'agent':<20} {'top_model':<40} {'n':>4}"]
    for agent, top, n in report["by_agent"]:
        lines.append(f"{agent[:20]:<20} {top[:40]:<40} {n:>4}")
    lines += ["", "- By difficulty -", f"{'diff':>4} {'top_model':<40} {'n':>4}"]
    for d, top, n in report["by_difficulty"]:
        lines.append(f"{d:>4} {top[:40]:<40} {n:>4}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default=str(_DEFAULT_WORKFLOW),
                        help="Path to workflow JSON (default: i2p_v3.json)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Write per-step records as JSON array to this path")
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=None,
        help="Directory of GGUF files to register as local models (default: none)",
    )
    parser.add_argument(
        "--loaded-model",
        dest="loaded_model",
        default=None,
        help=(
            "Registry name of a local model to simulate as already loaded in VRAM. "
            "Triggers swap-stickiness (1.40x) and idle_seconds urgency. "
            "Example: 'Qwen3.5-35B-A3B-UD-Q4_K_XL-thinking'"
        ),
    )
    args = parser.parse_args(argv)

    records = simulate(args.workflow, model_dir=args.model_dir, loaded_model=args.loaded_model)
    report = build_report(records)
    print(_format_report(report))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(records, indent=2))
        print(f"\nWrote {len(records)} records to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
