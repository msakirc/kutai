"""Z10 T3C — Migrate flat Chroma collections to per-mission namespace.

Default (dry-run): list every collection under ``data/chroma`` and the
planned rename:

  - Collections already namespaced (``mission_*__*`` or ``global__*``):
    left as-is.
  - Other collections: planned rename to ``global__<old_name>``. Operator
    can override with an explicit mapping file (``--mapping path.json``).

Pass ``--apply`` to actually rename. Chroma has no rename primitive, so
"rename" = create new + bulk copy rows + drop old. Per-row copy uses
``get(include=[documents,embeddings,metadatas])`` then ``upsert`` into the
new collection so embeddings survive without re-encoding.

Usage:
    python scripts/migrate_chroma_to_per_mission.py            # dry-run
    python scripts/migrate_chroma_to_per_mission.py --apply    # commit
    python scripts/migrate_chroma_to_per_mission.py --mapping map.json --apply

Mapping file format (JSON):
    {"semantic": "global__semantic", "ad_hoc_table": "mission_47__ad_hoc"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _client_for(persist_dir: str):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, allow_reset=False),
    )


def _plan(client, mapping: dict[str, str]) -> list[tuple[str, str]]:
    """Return list of (old_name, new_name) ops."""
    ops: list[tuple[str, str]] = []
    cols = client.list_collections()
    for c in cols:
        name = getattr(c, "name", None) or (c if isinstance(c, str) else None)
        if not name:
            continue
        if name.startswith("mission_") and "__" in name:
            continue
        if name.startswith("global__"):
            continue
        target = mapping.get(name, f"global__{name}")
        if target == name:
            continue
        ops.append((name, target))
    return ops


def _copy_collection(client, src_name: str, dst_name: str) -> int:
    src = client.get_collection(src_name)
    rows = src.get(include=["documents", "embeddings", "metadatas"])
    ids = rows.get("ids") or []
    if not ids:
        # Empty source — just create the dst placeholder.
        client.get_or_create_collection(
            name=dst_name,
            metadata=dict(src.metadata or {}),
        )
        return 0

    docs = rows.get("documents") or [None] * len(ids)
    metas = rows.get("metadatas") or [{}] * len(ids)
    embs = rows.get("embeddings") or [None] * len(ids)

    keep = [i for i, e in enumerate(embs) if e is not None]
    if not keep:
        return 0

    ids = [ids[i] for i in keep]
    docs = [docs[i] for i in keep]
    metas = [metas[i] for i in keep]
    embs = [embs[i] for i in keep]

    # Use the same metadata so dimension/hnsw config carries over.
    dst = client.get_or_create_collection(
        name=dst_name,
        metadata=dict(src.metadata or {}),
    )
    dst.upsert(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embs,
    )
    return len(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--persist-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "chroma",
        ),
    )
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--mapping", help="JSON file mapping old → new names")
    args = ap.parse_args()

    if not os.path.isdir(args.persist_dir):
        print(f"persist-dir not found: {args.persist_dir}")
        return 1

    mapping: dict[str, str] = {}
    if args.mapping:
        with open(args.mapping, "r", encoding="utf-8") as f:
            mapping = json.load(f)

    client = _client_for(args.persist_dir)
    ops = _plan(client, mapping)

    if not ops:
        print("No collections to migrate. Everything already namespaced.")
        return 0

    print(f"Plan ({len(ops)} renames):")
    for src, dst in ops:
        print(f"  {src!r} → {dst!r}")

    if not args.apply:
        print("\nDry-run. Pass --apply to commit.")
        return 0

    for src, dst in ops:
        try:
            n = _copy_collection(client, src, dst)
            client.delete_collection(src)
            print(f"  OK: {src!r} → {dst!r} ({n} rows)")
        except Exception as e:
            print(f"  FAIL: {src!r} → {dst!r}: {e}", file=sys.stderr)

    print("\nMigration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
