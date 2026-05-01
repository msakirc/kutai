"""Re-sync artifact_schema in pending F-* feature tasks.

Bug: expander.py prior to 2026-05-01 prefixed output_artifacts (e.g.
"F-001__code_review_result") but copied artifact_schema with unprefixed
keys ("code_review_result"). Loader.get_step does not index expanded F-*
step IDs, so step-refresh in base.py never updates the schema for these
tasks. They keep stale, unprefixed schemas.

This script:
  1. Loads i2p_v3 workflow + locates feature_implementation_template.
  2. Walks pending F-* tasks (status pending or processing).
  3. Re-derives artifact_schema with the correct ``F-XXX__`` prefix from
     template's current schema (picks up any structural fixes too).
  4. Writes back to task.context.

Dry-run by default. Pass --apply to commit.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import sqlite3
from pathlib import Path

DB_PATH = r"C:\Users\sakir\ai\kutai\kutai.db"
WORKFLOW_JSON = Path("src/workflows/i2p/i2p_v3.json")

STEP_ID_RE = re.compile(r"^(?P<phase>\d+)\.(?P<feature>F-\d+)\.feat\.(?P<n>\d+)$")


def load_template() -> dict:
    data = json.loads(WORKFLOW_JSON.read_text(encoding="utf-8"))
    for tpl in data.get("templates", []):
        if tpl.get("template_id") == "feature_implementation_template":
            return tpl
    raise RuntimeError("feature_implementation_template not found in i2p_v3.json")


def index_template_steps(tpl: dict) -> dict[str, dict]:
    """Map template_step_id (e.g. 'feat.9') -> step dict."""
    return {s["template_step_id"]: s for s in tpl.get("steps", [])}


def derive_target_schema(tpl_step: dict, feature_id: str) -> dict | None:
    raw = tpl_step.get("artifact_schema")
    if not isinstance(raw, dict):
        return None
    prefix = f"{feature_id}__"
    return {f"{prefix}{k}": v for k, v in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = no limit; else stop after N rewrites")
    args = ap.parse_args()

    tpl = load_template()
    tpl_steps = index_template_steps(tpl)
    print(f"template steps: {sorted(tpl_steps.keys())}")

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT id, status, context FROM tasks
        WHERE status IN ('pending','processing','failed')
          AND json_extract(context, '$.workflow_step_id') LIKE '%.F-%.feat.%'
    """).fetchall()
    print(f"candidate tasks: {len(rows)}")

    rewrites: list[tuple[int, str]] = []
    skipped_no_template = 0
    skipped_unchanged = 0
    skipped_unparseable = 0
    rewrite_summary: dict[str, int] = {}

    for r in rows:
        try:
            ctx = json.loads(r["context"])
        except (json.JSONDecodeError, TypeError):
            skipped_unparseable += 1
            continue
        sid = ctx.get("workflow_step_id", "")
        m = STEP_ID_RE.match(sid)
        if not m:
            skipped_unparseable += 1
            continue
        feature_id = m.group("feature")
        feat_key = f"feat.{m.group('n')}"
        tpl_step = tpl_steps.get(feat_key)
        if not tpl_step:
            skipped_no_template += 1
            continue
        target_schema = derive_target_schema(tpl_step, feature_id)
        if target_schema is None:
            skipped_no_template += 1
            continue
        current_schema = ctx.get("artifact_schema") or {}
        if current_schema == target_schema:
            skipped_unchanged += 1
            continue
        ctx["artifact_schema"] = target_schema
        new_ctx = json.dumps(ctx)
        rewrites.append((r["id"], new_ctx))
        sname = ctx.get("step_name", "?")
        rewrite_summary[sname] = rewrite_summary.get(sname, 0) + 1
        if args.limit and len(rewrites) >= args.limit:
            break

    print(f"rewrites: {len(rewrites)}")
    print(f"skipped (no template): {skipped_no_template}")
    print(f"skipped (unchanged): {skipped_unchanged}")
    print(f"skipped (unparseable): {skipped_unparseable}")
    print(f"\nby step_name:")
    for k, n in sorted(rewrite_summary.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {k}")

    if not rewrites:
        return 0
    if not args.apply:
        print("\n[dry-run] no changes written. Re-run with --apply to commit.")
        return 0

    cur = db.cursor()
    for tid, new_ctx in rewrites:
        cur.execute("UPDATE tasks SET context = ? WHERE id = ?", (new_ctx, tid))
    db.commit()
    print(f"committed {len(rewrites)} rewrites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
