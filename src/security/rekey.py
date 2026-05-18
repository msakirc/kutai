# src/security/rekey.py
"""Rekey / re-encrypt all credential rows from one key version to another.

Usage::

    python -m src.security.rekey --from-version 1 --to-version 2 [--dry-run]

Both ``KUTAY_MASTER_KEY_v<from>`` and ``KUTAY_MASTER_KEY_v<to>`` (or the
legacy ``KUTAY_MASTER_KEY`` aliased to v1) must be set in the environment.
Each row is read with the source key, re-encrypted with the destination
key, written back with ``key_version=<to>``, and an audit row is emitted
with ``action='rotate'``.

Idempotent within a single run: rows already at ``<to>`` are skipped. Safe
to rerun if interrupted — the second pass picks up the unfinished rows.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

from src.infra.logging_config import get_logger

from . import credential_audit, credential_store as cs

logger = get_logger("security.rekey")


async def rekey(
    from_version: int,
    to_version: int,
    dry_run: bool = False,
) -> dict[str, int]:
    """Re-encrypt every `key_version=from_version` row using the *to_version* key.

    Returns a stats dict: ``{"scanned": N, "rekeyed": M, "skipped": K, "errors": E}``.
    """
    if from_version == to_version:
        raise ValueError("from-version and to-version must differ")

    def _env_lookup(name: str) -> str | None:
        # Windows uppercases env names — check both spellings.
        return os.getenv(name) or os.getenv(name.upper())

    if not _env_lookup(f"KUTAY_MASTER_KEY_v{from_version}") and not (
        from_version == 1 and os.getenv("KUTAY_MASTER_KEY")
    ):
        raise RuntimeError(
            f"KUTAY_MASTER_KEY_v{from_version} env var not set "
            "(legacy KUTAY_MASTER_KEY is treated as v1 only)"
        )
    if not _env_lookup(f"KUTAY_MASTER_KEY_v{to_version}"):
        raise RuntimeError(f"KUTAY_MASTER_KEY_v{to_version} env var not set")

    # Force key discovery and verify both versions are present in
    # _fernet_by_version. Calling _get_fernet() populates the dict.
    cs._reset_key_state()
    cs._get_fernet()
    missing = []
    for v in (from_version, to_version):
        if v not in cs._fernet_by_version:
            missing.append(v)
    if missing:
        raise RuntimeError(
            f"could not build Fernet for versions {missing} — "
            "check env var values"
        )

    from ..infra.db import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT service_name, encrypted_data, key_version FROM credentials"
    )
    rows = await cur.fetchall()
    stats = {"scanned": 0, "rekeyed": 0, "skipped": 0, "errors": 0}

    src_fernet = cs._fernet_by_version[from_version]
    dst_fernet = cs._fernet_by_version[to_version]

    for row in rows:
        stats["scanned"] += 1
        service_name = row[0] if isinstance(row, tuple) else row["service_name"]
        encrypted_data = row[1] if isinstance(row, tuple) else row["encrypted_data"]
        existing_version = row[2] if isinstance(row, tuple) else row["key_version"]

        if existing_version == to_version:
            stats["skipped"] += 1
            continue
        if existing_version != from_version:
            logger.warning(
                f"skipping {service_name}: key_version="
                f"{existing_version}, expected {from_version}",
                service=service_name,
            )
            stats["skipped"] += 1
            continue

        try:
            plaintext = src_fernet.decrypt(encrypted_data.encode()).decode()
            reencrypted = dst_fernet.encrypt(plaintext.encode()).decode()
        except Exception as exc:
            logger.error(
                f"rekey decrypt/encrypt failed for {service_name}: {exc}",
                service=service_name,
                error=str(exc),
            )
            stats["errors"] += 1
            continue

        if dry_run:
            stats["rekeyed"] += 1
            logger.info(
                f"[dry-run] would rekey {service_name} "
                f"v{from_version}→v{to_version}"
            )
            continue

        now = datetime.now(timezone.utc).isoformat()
        try:
            cs._rekey_in_progress = True
            await db.execute(
                "UPDATE credentials SET "
                " encrypted_data = ?, key_version = ?, "
                " rotated_at = ?, updated_at = ? "
                "WHERE service_name = ?",
                (reencrypted, to_version, now, now, service_name),
            )
            await db.commit()
            stats["rekeyed"] += 1
            await credential_audit.log_access(
                service_name, "rotate", True, error=None
            )
        finally:
            cs._rekey_in_progress = False

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.security.rekey",
        description="Re-encrypt all credentials from one master-key version "
        "to another.",
    )
    parser.add_argument("--from-version", type=int, required=True)
    parser.add_argument("--to-version", type=int, required=True)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing to the DB.",
    )
    args = parser.parse_args(argv)

    try:
        stats = asyncio.run(
            rekey(args.from_version, args.to_version, dry_run=args.dry_run)
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    label = "[dry-run] " if args.dry_run else ""
    print(
        f"{label}scanned={stats['scanned']} rekeyed={stats['rekeyed']} "
        f"skipped={stats['skipped']} errors={stats['errors']}"
    )
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
