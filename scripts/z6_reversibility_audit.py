"""Z6 T6A — reversibility audit for i2p_v3.json.

Walks every step in ``src/workflows/i2p/i2p_v3.json`` and proposes a
``reversibility`` tag using simple, conservative heuristics:

  needs_real_tools=True            → ``irreversible``
  instruction mentions one of      → ``partial``
    {deploy, publish, push, production, apply migration,
     send email, create_repo, init repo}
  otherwise                        → ``full`` (safest default)

The script is idempotent: when re-run after the proposals are applied,
all rows surface ``current == proposed`` and nothing changes.

Usage:
    python scripts/z6_reversibility_audit.py --report
    python scripts/z6_reversibility_audit.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = _REPO_ROOT / "src" / "workflows" / "i2p" / "i2p_v3.json"

# Substrings that indicate a non-local, semi-reversible action. Lowercased.
PARTIAL_TRIGGERS: tuple[str, ...] = (
    "deploy",
    "publish",
    "push",
    "production",
    "apply migration",
    "send email",
    "create_repo",
    "init repo",
    "github repo",
    "real vendor",
    "vendor write",
)


def _instruction_text(step: dict) -> str:
    """Pull every textual hint we can off a step. Lowercased."""
    parts: list[str] = []
    for k in ("instruction", "name", "done_when"):
        v = step.get(k)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(x) for x in v)
    ctx = step.get("context")
    if isinstance(ctx, dict):
        action = ctx.get("action") or ctx.get("executor")
        if isinstance(action, str):
            parts.append(action)
    return " ".join(parts).lower()


def propose(step: dict) -> str:
    """Return the heuristic reversibility tag for one step."""
    if step.get("needs_real_tools"):
        return "irreversible"
    ctx = step.get("context")
    if isinstance(ctx, dict) and ctx.get("needs_real_tools"):
        return "irreversible"
    blob = _instruction_text(step)
    for trig in PARTIAL_TRIGGERS:
        if trig in blob:
            return "partial"
    return "full"


def walk(workflow: dict) -> Iterable[tuple[str, str, str]]:
    """Yield ``(step_id, current, proposed)`` for each step."""
    for step in workflow.get("steps", []):
        sid = step.get("id") or "?"
        current = step.get("reversibility") or ""
        yield sid, current, propose(step)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--report", action="store_true",
                   help="Print proposals only (default).")
    g.add_argument("--apply", action="store_true",
                   help="Write proposals back into i2p_v3.json.")
    p.add_argument("--workflow", type=Path, default=_WORKFLOW)
    args = p.parse_args(argv)

    with args.workflow.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    rows = list(walk(wf))
    changes = [(s, c, n) for s, c, n in rows if c != n]

    if not args.apply:
        print(f"# i2p_v3 reversibility audit — {len(rows)} steps, "
              f"{len(changes)} need updating")
        print(f"{'step':<10} {'current':<14} {'proposed':<14}")
        for sid, cur, nxt in rows:
            mark = "*" if cur != nxt else " "
            print(f"{mark}{sid:<9} {cur or '-':<14} {nxt:<14}")
        return 0

    # Apply.
    proposed: dict[str, str] = {sid: nxt for sid, _, nxt in rows}
    for step in wf.get("steps", []):
        sid = step.get("id")
        if sid in proposed:
            step["reversibility"] = proposed[sid]

    with args.workflow.open("w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Applied reversibility to {len(rows)} steps in {args.workflow}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
