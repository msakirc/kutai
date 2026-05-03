"""One-shot migration: move .dead_models.json entries to the SQLite
provider/model registry.

Idempotent — safe to run twice. After successful migration the JSON
file is renamed to .dead_models.json.archived (kept on disk so the
pre-migration state is recoverable).

Run:
    .venv/Scripts/python.exe scripts/migrate_dead_models_to_sqlite.py

Notes:
- Old JSON format has no `cause` column. Treat existing entries as
  '404_permanent' (CAUSE_POLICY: 24h TTL, auto-expires). The 24h
  window is generous enough that any genuinely-dead id won't escape;
  any genuinely-live id heals on its own.
- expires_at on the new row honors the original JSON expiration when
  it falls within the cause's TTL window; otherwise capped to the
  CAUSE_POLICY default. Trade-off: small possibility a transiently-
  dead id stays out a few minutes longer than the JSON would have
  said. Acceptable.
- Provider rows are NOT created here — providers populate themselves
  on first auth-failure or first call.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the workspace importable when run directly.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.infra import registry_store  # noqa: E402


_JSON_PATH = Path(_repo_root) / ".dead_models.json"
_ARCHIVE_SUFFIX = ".archived"


def main() -> int:
    if not _JSON_PATH.exists():
        print(f"No file at {_JSON_PATH} — nothing to migrate.")
        return 0

    try:
        data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"FAIL: cannot parse {_JSON_PATH}: {e}", file=sys.stderr)
        return 2

    if not isinstance(data, dict):
        print(
            f"FAIL: expected dict, got {type(data).__name__} in {_JSON_PATH}",
            file=sys.stderr,
        )
        return 2

    now = time.time()
    cause = "404_permanent"
    cause_ttl = registry_store.CAUSE_POLICY[cause]["ttl_seconds"]

    migrated = 0
    skipped_expired = 0
    skipped_invalid = 0

    for litellm_name, expiration_ts in data.items():
        if not isinstance(litellm_name, str) or not litellm_name:
            skipped_invalid += 1
            continue
        try:
            exp = float(expiration_ts)
        except (TypeError, ValueError):
            skipped_invalid += 1
            continue
        if exp <= now:
            # Already expired in the old format — don't bring it forward.
            skipped_expired += 1
            continue
        # Mark dead with default cause. Re-marking is fine — registry_store
        # will refresh marked_at + expires_at if the row already exists.
        registry_store.mark_dead(
            litellm_name, cause=cause, actor="migration",
            payload={
                "source_file": str(_JSON_PATH.name),
                "original_expires_at_epoch": exp,
                "cause_policy_ttl_seconds": cause_ttl,
            },
        )
        migrated += 1

    archive_path = _JSON_PATH.with_suffix(_JSON_PATH.suffix + _ARCHIVE_SUFFIX)
    try:
        _JSON_PATH.rename(archive_path)
        archived_to = str(archive_path)
    except OSError as e:
        archived_to = f"<failed: {e}>"

    print(
        f"Migration done. migrated={migrated} "
        f"skipped_expired={skipped_expired} skipped_invalid={skipped_invalid} "
        f"archived_to={archived_to}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
