"""One-shot repair for mission-57 corrupted multi-artifact envelopes.

Bug (hooks.py:1221-1226 prior to 2026-05-01 fix): post-execute stored the
whole LLM envelope JSON string into EVERY output_artifact slot. Effect:
multi-artifact steps end up with art1 == art2 == art3 == full envelope.

This script:
  1. Loads blackboards.data for mission_id=57.
  2. Detects slots whose value parses as a dict-envelope containing ALL of
     a known sibling group (heuristic: groups of slots whose stored string
     is byte-identical and the parsed dict's keys cover every group member).
  3. For each detected envelope, extracts per-key value and overwrites slot.
  4. Same treatment for `<name>_summary` slots if they share the corruption.
  5. Dry-run by default. Pass --apply to write.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DB_PATH = r"C:\Users\sakir\ai\kutai\kutai.db"
MISSION_ID = 57


def detect_envelopes(arts: dict) -> list[tuple[set[str], str, dict]]:
    """Return list of (member_keys, envelope_string, parsed_dict).

    Group keys by identical string value. For any group whose value parses
    as a dict and whose dict keys are a superset of the group, this is a
    corrupted envelope.
    """
    by_value: dict[str, list[str]] = defaultdict(list)
    for k, v in arts.items():
        if isinstance(v, str) and v.lstrip().startswith("{"):
            by_value[v].append(k)

    found = []
    for envelope_str, members in by_value.items():
        if len(members) < 2:
            continue
        try:
            parsed = json.loads(envelope_str)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        member_set = set(members)
        if not member_set.issubset(parsed.keys()):
            continue
        found.append((member_set, envelope_str, parsed))
    return found


def split_value(parsed: dict, key: str) -> str:
    v = parsed[key]
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes (default dry-run)")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--mission", type=int, default=MISSION_ID)
    ap.add_argument("--backup-dir", default="logs/migrations")
    args = ap.parse_args()

    import sqlite3
    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT data FROM blackboards WHERE mission_id = ?", (args.mission,)).fetchone()
    if not row:
        print(f"mission {args.mission}: no blackboard row")
        return 1

    data = json.loads(row["data"])
    arts = data.get("artifacts", {})
    if not isinstance(arts, dict):
        print(f"mission {args.mission}: artifacts is not a dict, abort")
        return 1

    envelopes = detect_envelopes(arts)
    print(f"mission {args.mission}: artifacts total={len(arts)}  detected envelope groups={len(envelopes)}")
    total_slot_rewrites = 0
    rewrites: dict[str, str] = {}
    for member_set, env_str, parsed in envelopes:
        env_keys = sorted(parsed.keys())
        members_sorted = sorted(member_set)
        is_summary_group = all(m.endswith("_summary") for m in members_sorted)
        print(f"  group: {members_sorted}  parsed_keys={env_keys}  envelope_len={len(env_str)}  summary={is_summary_group}")
        for k in members_sorted:
            new_v = split_value(parsed, k.removesuffix("_summary") if is_summary_group and k.removesuffix("_summary") in parsed else k) if not is_summary_group else split_value(parsed, k)
            # If this is a summary slot but parsed envelope contains the unsuffixed name, prefer that
            if is_summary_group:
                base = k.removesuffix("_summary")
                if base in parsed:
                    new_v = split_value(parsed, base)
                elif k in parsed:
                    new_v = split_value(parsed, k)
                else:
                    print(f"    WARN: summary slot {k} has no matching base in envelope, skip")
                    continue
            rewrites[k] = new_v
            total_slot_rewrites += 1
            print(f"    {k}: {len(env_str)}c -> {len(new_v)}c")

    if not rewrites:
        print("nothing to rewrite")
        return 0

    print(f"\ntotal slot rewrites: {total_slot_rewrites}")

    if not args.apply:
        print("\n[dry-run] no changes written. Re-run with --apply to commit.")
        return 0

    # Backup
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = backup_dir / f"blackboard_m{args.mission}_{stamp}.json"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"backup written: {backup_path}")

    # Apply rewrites
    for k, v in rewrites.items():
        arts[k] = v
    data["artifacts"] = arts
    new_blob = json.dumps(data, ensure_ascii=False)
    db.execute(
        "UPDATE blackboards SET data = ?, updated_at = datetime('now') WHERE mission_id = ?",
        (new_blob, args.mission),
    )
    db.commit()
    print(f"committed {len(rewrites)} slot rewrites to mission {args.mission}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
