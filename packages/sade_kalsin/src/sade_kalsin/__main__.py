"""CLI: ``python -m sade_kalsin audit [--quarter Q] [--layer NAME] [--root R]``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sade_kalsin.audit_report import run_audit


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sade_kalsin")
    sub = p.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="Run a bash-audit pass")
    audit.add_argument("--quarter", default=None, help="e.g. 2026-Q2 (default: current)")
    audit.add_argument("--layer", default=None, help="Single-layer focus")
    audit.add_argument(
        "--root",
        default=None,
        help="Repo root (default: cwd)",
    )
    audit.add_argument(
        "--out-dir",
        default=None,
        help="Override docs/audits/ output dir",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd != "audit":
        return 2
    root = Path(args.root) if args.root else Path.cwd()
    out_dir = Path(args.out_dir) if args.out_dir else None
    result = run_audit(
        root=root,
        quarter=args.quarter,
        out_dir=out_dir,
        layer_filter=args.layer,
    )
    print(f"sade_kalsin audit: wrote {result['report_path']} "
          f"({result['layer_count']} layers, {result['total_loc']:,} LOC)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
