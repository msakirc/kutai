"""yalayut.secrets — fernet-encrypted auth secret store + env_status lifecycle.

API and MCP artifacts declare ``auth_env`` / ``mcp.env_required``. yalayut needs
to know whether the required keys are available so it can:

  * write ``env_status`` ('ready' | 'missing_<KEY>') on ``yalayut_index`` at
    vet time and on a daily re-check;
  * let intersect filter artifacts whose ``env_status != 'ready'`` at match
    time.

A key is "available" if it is present in ``os.environ`` OR stored (encrypted)
in the ``yalayut_secrets`` table. Founder writes via ``/yalayut auth set`` →
``set_secret``. Encryption uses ``cryptography.fernet`` with the key in ``.env``
under ``YALAYUT_SECRET_KEY``.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("yalayut.secrets")

_SECRET_KEY_ENV = "YALAYUT_SECRET_KEY"


class SecretsError(RuntimeError):
    """Raised when the secrets store cannot operate (missing fernet key)."""


def _fernet():
    """Build a Fernet instance from the env key. Raises if absent/invalid."""
    from cryptography.fernet import Fernet

    raw = os.getenv(_SECRET_KEY_ENV)
    if not raw:
        raise SecretsError(
            f"{_SECRET_KEY_ENV} not set in .env — cannot encrypt yalayut secrets"
        )
    try:
        return Fernet(raw.encode() if isinstance(raw, str) else raw)
    except Exception as e:
        raise SecretsError(f"{_SECRET_KEY_ENV} is not a valid fernet key: {e}") from e


async def _db_upsert_secret(key_name: str, encrypted_value: bytes) -> None:
    """Upsert one encrypted secret row. Patched in tests."""
    from dabidabi import get_db

    db = await get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO yalayut_secrets (key_name, encrypted_value, added_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(key_name) DO UPDATE SET encrypted_value = excluded.encrypted_value",
        (key_name, encrypted_value, now),
    )
    await db.commit()


async def _db_fetch_secret(key_name: str) -> bytes | None:
    """Fetch one encrypted secret blob. Patched in tests."""
    from dabidabi import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT encrypted_value FROM yalayut_secrets WHERE key_name = ?",
        (key_name,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    return row[0]


async def set_secret(key_name: str, value: str) -> None:
    """Encrypt ``value`` and store it under ``key_name`` in yalayut_secrets."""
    blob = _fernet().encrypt(value.encode("utf-8"))
    await _db_upsert_secret(key_name, blob)
    logger.info("secret stored", key_name=key_name)


async def get_secret(key_name: str) -> str | None:
    """Return the decrypted secret, or ``None`` if not stored."""
    blob = await _db_fetch_secret(key_name)
    if blob is None:
        return None
    try:
        return _fernet().decrypt(bytes(blob)).decode("utf-8")
    except Exception as e:
        logger.warning("secret decrypt failed", key_name=key_name, err=str(e))
        return None


async def resolve_env(key_name: str) -> str | None:
    """Return a key's value from os.environ first, then the encrypted store."""
    val = os.getenv(key_name)
    if val:
        return val
    return await get_secret(key_name)


async def compute_env_status(required_keys: list[str] | None) -> str:
    """Return 'ready' if all keys resolve, else 'missing_<first-missing-KEY>'."""
    if not required_keys:
        return "ready"
    for key in required_keys:
        if not await resolve_env(key):
            return f"missing_{key}"
    return "ready"
