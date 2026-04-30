"""Pressure-breakdown CLI — full signal-by-signal explainer.

Live diagnostic against the running registry + nerd_herd snapshot. Prints
per-(model, task_difficulty) breakdown so an operator can see EXACTLY
which signal pushed which model up/down.

Usage:
    python -m fatih_hoca.debug.pressure_dump
    python -m fatih_hoca.debug.pressure_dump --difficulty 7
    python -m fatih_hoca.debug.pressure_dump --provider gemini --difficulty 7

Columns:
    model, pool, S1..S11, M1, M2, scalar, decision

Decision: "ADMIT" if pressure >= threshold for typical urgency 0.5
(threshold=0.0), else "REJECT" with the dominant negative signal name.
Mirrors what beckman.admission gate would do for a typical task.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any


def _format_signal(v: float) -> str:
    """Two-char-width signed formatter. -1.00, +0.50, +0.00."""
    if v == 0:
        return " 0.00"
    return f"{v:+.2f}"


def _dominant(signals: dict[str, float]) -> str:
    """Name of the signal with the largest |contribution|."""
    if not signals:
        return ""
    name, val = max(signals.items(), key=lambda kv: abs(kv[1]))
    return f"{name}={val:+.2f}"


async def dump(
    difficulty: int = 5,
    provider_filter: str | None = None,
    est_tokens: int = 5000,
    est_iterations: int = 4,
    threshold: float = 0.0,
) -> None:
    import fatih_hoca
    import nerd_herd

    if fatih_hoca._registry is None:
        print("ERROR: fatih_hoca._registry is None — call fatih_hoca.init() first.")
        print("       Run this from inside a live KutAI process or wrap with init().")
        sys.exit(1)

    snap = await nerd_herd.refresh_snapshot()
    if snap is None:
        print("ERROR: nerd_herd.refresh_snapshot() returned None")
        sys.exit(1)

    models = [m for m in fatih_hoca._registry.all_models()
              if not provider_filter or getattr(m, "provider", "") == provider_filter]
    if not models:
        print(f"No models registered (provider_filter={provider_filter!r}).")
        return

    print(f"\n=== Pressure Breakdown — difficulty={difficulty} threshold={threshold:+.2f} ===\n")
    cols = ["model", "pool", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
            "S9", "S10", "S11", "M1", "M2", "scalar", "decision"]
    print("  ".join(f"{c:>14}" if i == 0 else f"{c:>6}"
                    for i, c in enumerate(cols)))
    print("  ".join("-" * 14 if i == 0 else "-" * 6 for i in range(len(cols))))

    for m in sorted(models, key=lambda x: (getattr(x, "is_local", False),
                                            getattr(x, "provider", ""),
                                            getattr(x, "name", ""))):
        try:
            br = snap.pressure_for(
                m,
                task_difficulty=difficulty,
                est_per_call_tokens=est_tokens,
                est_per_task_tokens=est_tokens * est_iterations,
                est_iterations=est_iterations,
                est_call_cost=getattr(m, "estimated_cost",
                                      lambda *_: 0.0)(est_tokens, est_tokens),
                cap_needed=5.0,
                consecutive_failures=0,
            )
        except Exception as e:
            print(f"{getattr(m,'name','?'):>14}  pressure_for raised: {e!r}")
            continue

        sigs = br.signals
        mods = br.modifiers or {}
        pool = ("local" if getattr(m, "is_local", False)
                else "free" if getattr(m, "is_free", False)
                else "paid")
        decision = "ADMIT" if br.scalar >= threshold else f"REJECT ({_dominant(sigs)})"

        row = [
            f"{getattr(m,'name','?')[:14]:>14}",
            f"{pool:>6}",
            *[f"{_format_signal(sigs.get(k, 0.0)):>6}" for k in
              ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11")],
            f"{mods.get('M1', 1.0):>6.2f}",
            f"{mods.get('M2', 1.0):>6.2f}",
            f"{br.scalar:>+6.2f}",
            decision,
        ]
        print("  ".join(row))

    print("\nLegend:")
    print("  S1=remaining    S2=call_burden  S3=task_burden  S4=queue_tokens")
    print("  S5=queue_calls  S6=capable_supply  S7=burn_rate  S9=perishability")
    print("  S10=failure     S11=cost        M1=capacity_amp  M2=perish_dampener")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=5)
    ap.add_argument("--provider", type=str, default=None)
    ap.add_argument("--est-tokens", type=int, default=5000)
    ap.add_argument("--est-iterations", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.0)
    args = ap.parse_args()
    asyncio.run(dump(
        difficulty=args.difficulty,
        provider_filter=args.provider,
        est_tokens=args.est_tokens,
        est_iterations=args.est_iterations,
        threshold=args.threshold,
    ))


if __name__ == "__main__":
    main()
