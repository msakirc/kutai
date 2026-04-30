"""Matrix-population audit — answers 'are the cells the signals read
actually populated?'.

Pool pressure is only as good as the data feeding it. Any axis whose
matrix cell is empty is a dead signal — selector treats the model as
unconstrained on that dimension. This tool walks every cloud model and
prints which of the 16 cells have (limit, remaining) populated.

Usage:
    python -m fatih_hoca.debug.matrix_audit
    python -m fatih_hoca.debug.matrix_audit --provider gemini

Empty cells aren't always wrong (no provider exposes ALL 16 axes), but
pattern-matching across providers + models reveals plumbing gaps.
"""
from __future__ import annotations

import argparse
import asyncio
import sys


_AXES = [
    "rpm", "rph", "rpd", "rpw", "rpmonth",
    "tpm", "tph", "tpd", "tpw", "tpmonth",
    "itpm", "itpd", "otpm", "otpd",
    "cpd", "cpmonth",
]


def _cell_state(rl) -> str:
    """One-char status. L=limit only, R=remaining only, B=both, .=empty."""
    has_limit = rl.limit is not None and rl.limit > 0
    has_remaining = rl.remaining is not None
    if has_limit and has_remaining:
        return "B"
    if has_limit:
        return "L"
    if has_remaining:
        return "R"
    return "."


async def audit(provider_filter: str | None = None) -> None:
    import fatih_hoca
    import nerd_herd

    if fatih_hoca._registry is None:
        print("ERROR: fatih_hoca._registry is None — run inside live KutAI process.")
        sys.exit(1)

    snap = await nerd_herd.refresh_snapshot()
    if snap is None:
        print("ERROR: snapshot is None")
        sys.exit(1)

    print(f"\n=== Matrix Cell Audit "
          f"(provider_filter={provider_filter or 'all'}) ===\n")
    print(f"Legend: B=limit+remaining (good), L=limit only, "
          f"R=remaining only (rare), .=empty\n")

    header = f"{'model':<48} {'pool':<6} " + " ".join(f"{a:>5}" for a in _AXES)
    print(header)
    print("-" * len(header))

    providers = (
        {provider_filter: snap.cloud.get(provider_filter)}
        if provider_filter else snap.cloud
    )
    if not providers or all(v is None for v in providers.values()):
        print("(no cloud providers in snapshot)")
        return

    for prov_name, prov_state in providers.items():
        if prov_state is None:
            continue
        for model_id, model_state in sorted(prov_state.models.items()):
            cells = model_state.limits
            row = (
                f"{model_id[:48]:<48} {prov_name[:6]:<6} "
                + " ".join(f"{_cell_state(getattr(cells, a)):>5}" for a in _AXES)
            )
            print(row)

    # Summary: which axes are populated for ANY model on each provider?
    print("\n=== Per-provider axis coverage ===")
    for prov_name, prov_state in providers.items():
        if prov_state is None:
            continue
        populated = set()
        for ms in prov_state.models.values():
            for axis in _AXES:
                rl = getattr(ms.limits, axis)
                if rl.limit is not None and rl.limit > 0:
                    populated.add(axis)
        missing = [a for a in _AXES if a not in populated]
        print(f"  {prov_name}: populated={sorted(populated) or 'NONE'}")
        if missing:
            print(f"     missing axes={missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", type=str, default=None)
    args = ap.parse_args()
    asyncio.run(audit(provider_filter=args.provider))


if __name__ == "__main__":
    main()
