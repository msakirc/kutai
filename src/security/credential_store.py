# credential_store.py
"""
Encrypted credential storage for external service integrations.

Uses cryptography.fernet with a master key from KUTAY_MASTER_KEY env var.
Falls back to base64 encoding if cryptography is not installed or key is unset.
"""

import base64
import json
import logging
import os
import warnings
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encryption backend — prefer cryptography.fernet, fall back to base64
# ---------------------------------------------------------------------------

_MASTER_KEY: bytes | None = None
_fernet = None

try:
    from cryptography.fernet import Fernet

    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False
    Fernet = None  # type: ignore[assignment,misc]


def _get_fernet():
    """Return a Fernet instance using the master key, or None for fallback."""
    global _fernet, _MASTER_KEY

    if _fernet is not None:
        return _fernet

    if not _HAS_CRYPTOGRAPHY:
        logger.warning(
            "cryptography package not installed — credentials stored with "
            "base64 encoding only (NOT secure for production)"
        )
        return None

    raw_key = os.getenv("KUTAY_MASTER_KEY", "")
    if not raw_key:
        warnings.warn(
            "KUTAY_MASTER_KEY not set — using auto-generated fallback key. "
            "Set KUTAY_MASTER_KEY for persistent encrypted storage.",
            stacklevel=2,
        )
        # Generate a deterministic-ish fallback for dev/testing
        # (NOT secure — credentials won't survive restarts with different key)
        raw_key = base64.urlsafe_b64encode(b"kutay-dev-fallback-key-00000000").decode()

    # Ensure the key is valid Fernet format (32 url-safe base64 bytes)
    try:
        _fernet = Fernet(raw_key.encode() if isinstance(raw_key, str) else raw_key)
        _MASTER_KEY = raw_key.encode() if isinstance(raw_key, str) else raw_key
        return _fernet
    except Exception:
        # Key isn't valid Fernet format — derive one from it
        import hashlib

        derived = hashlib.sha256(raw_key.encode()).digest()
        key = base64.urlsafe_b64encode(derived)
        _fernet = Fernet(key)
        _MASTER_KEY = key
        return _fernet


def _encrypt(data: str) -> str:
    """Encrypt a string, returning base64-encoded ciphertext."""
    f = _get_fernet()
    if f is not None:
        return f.encrypt(data.encode()).decode()
    # Fallback: simple base64 (NOT secure)
    return base64.urlsafe_b64encode(data.encode()).decode()


def _decrypt(token: str) -> str:
    """Decrypt a token back to the original string."""
    f = _get_fernet()
    if f is not None:
        return f.decrypt(token.encode()).decode()
    # Fallback: simple base64
    return base64.urlsafe_b64decode(token.encode()).decode()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def store_credential(service_name: str, data: dict) -> None:
    """Encrypt and store a credential in the database."""
    from ..infra.db import get_db

    encrypted = _encrypt(json.dumps(data))
    now = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    await db.execute(
        """INSERT INTO credentials (service_name, encrypted_data, created_at, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(service_name) DO UPDATE SET
               encrypted_data = excluded.encrypted_data,
               updated_at = excluded.updated_at""",
        (service_name, encrypted, now, now),
    )
    await db.commit()
    logger.info(f"Stored credential for service: {service_name}")


async def get_credential(service_name: str) -> dict | None:
    """Retrieve and decrypt a credential from the database."""
    from ..infra.db import get_db

    db = await get_db()
    cursor = await db.execute(
        "SELECT encrypted_data FROM credentials WHERE service_name = ?",
        (service_name,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None

    try:
        decrypted = _decrypt(row[0] if isinstance(row, tuple) else row["encrypted_data"])
        return json.loads(decrypted)
    except Exception as e:
        logger.error(f"Failed to decrypt credential for {service_name}: {e}")
        return None


async def delete_credential(service_name: str) -> bool:
    """Remove a credential from the database."""
    from ..infra.db import get_db

    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM credentials WHERE service_name = ?",
        (service_name,),
    )
    await db.commit()
    deleted = cursor.rowcount > 0
    if deleted:
        logger.info(f"Deleted credential for service: {service_name}")
    return deleted


async def list_credentials() -> list[str]:
    """List service names that have stored credentials (no secrets returned)."""
    from ..infra.db import get_db

    db = await get_db()
    cursor = await db.execute("SELECT service_name FROM credentials ORDER BY service_name")
    rows = await cursor.fetchall()
    return [
        (row[0] if isinstance(row, tuple) else row["service_name"])
        for row in rows
    ]
