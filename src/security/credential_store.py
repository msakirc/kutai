# credential_store.py
"""
Encrypted credential storage for external service integrations.

Uses cryptography.fernet with a master key from KUTAY_MASTER_KEY env var.
Falls back to base64 encoding if cryptography is not installed or key is unset.
"""

import base64
import json
import os
import warnings
from datetime import datetime, timezone

from src.infra.logging_config import get_logger

logger = get_logger("security.credential_store")

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
        env = os.getenv("KUTAY_ENV", "development").lower()
        if env in ("production", "prod", "staging"):
            raise RuntimeError(
                "KUTAY_MASTER_KEY environment variable is required in "
                f"{env} mode. Set a Fernet-compatible key or a passphrase."
            )
        warnings.warn(
            "KUTAY_MASTER_KEY not set — using dev-only fallback key. "
            "This is NOT secure. Set KUTAY_MASTER_KEY for production.",
            stacklevel=2,
        )
        # Dev-only deterministic key — credentials won't survive key changes
        raw_key = base64.urlsafe_b64encode(b"kutay-dev-fallback-key-00000000").decode()
        logger.warning("🔓 Using insecure dev fallback key for credential encryption")

    # Ensure the key is valid Fernet format (32 url-safe base64 bytes)
    try:
        _fernet = Fernet(raw_key.encode() if isinstance(raw_key, str) else raw_key)
        _MASTER_KEY = raw_key.encode() if isinstance(raw_key, str) else raw_key
        return _fernet
    except Exception:
        # Key isn't valid Fernet format — derive one using PBKDF2
        import hashlib

        # Use PBKDF2 with 480k iterations (OWASP 2023 recommendation for SHA-256)
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            raw_key.encode(),
            salt=b"kutay-credential-store-v1",  # static salt is OK here
            iterations=480_000,
        )
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

class CredentialSchemaError(ValueError):
    """Raised when a credential payload fails per-vendor schema validation."""


async def store_credential(
    service_name: str,
    data: dict,
    expires_at: str | None = None,
    scope: str | None = None,
    schema_id: str | None = None,
    *,
    unsafe: bool = False,
) -> None:
    """Encrypt and store a credential in the database.

    Parameters
    ----------
    service_name:
        Unique identifier for the service (e.g. "github", "vercel").
    data:
        Credential payload to encrypt (e.g. {"token": "..."}).
    expires_at:
        Optional ISO-8601 expiration timestamp.  When set, ``get_credential``
        will return ``None`` after this time and log a warning. Stored both
        inside the encrypted envelope (tamper-proof) and in the indexable
        ``expires_at`` column (cheap pre-check).
    scope:
        Optional scope label (e.g. ``read_only``, ``read_write``, ``admin``).
        Defaults to ``read_write`` at the column level.
    schema_id:
        Optional pointer to a ``credential_schemas/<service_name>.json`` entry
        that the payload was validated against. When omitted, the loader
        auto-uses *service_name* if a schema exists.
    unsafe:
        Bypass per-vendor schema validation. Required when storing a service
        that has no schema and the caller explicitly opts out. Without this
        flag, a payload that fails validation raises
        :class:`CredentialSchemaError`.
    """
    from . import credential_schemas as _cs_schemas
    from ..infra.db import get_db

    # Validate against per-vendor schema if one is registered.
    auto_schema_id = schema_id
    if not unsafe:
        ok, errors = _cs_schemas.validate_payload(
            service_name, data, scope=scope
        )
        if not ok:
            raise CredentialSchemaError(
                f"credential payload for '{service_name}' failed validation: "
                + "; ".join(errors)
            )
        if auto_schema_id is None and _cs_schemas.load_schema(service_name):
            auto_schema_id = service_name
    schema_id = auto_schema_id

    # Embed expiration inside the encrypted payload so it's tamper-proof
    envelope = {"_data": data}
    if expires_at:
        envelope["_expires_at"] = expires_at

    encrypted = _encrypt(json.dumps(envelope))
    now = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    # Detect whether this is a fresh write or an in-place rotation so the
    # audit log uses the right action verb.
    pre_cur = await db.execute(
        "SELECT 1 FROM credentials WHERE service_name = ?", (service_name,)
    )
    existed = await pre_cur.fetchone() is not None
    # UPSERT: insert with full metadata; on conflict update only the fields
    # the caller cared about so a partial update doesn't reset, e.g., scope
    # back to the column default. ``rotated_at`` is bumped on every update
    # since re-storing is effectively a rotation event.
    # Pass scope as NULL into excluded when the caller omitted it so the
    # ON CONFLICT branch can preserve any previously set scope via COALESCE.
    # On a fresh INSERT, the column DEFAULT 'read_write' will fill NULLs.
    await db.execute(
        """INSERT INTO credentials (
               service_name, encrypted_data, created_at, updated_at,
               scope, expires_at, schema_id
           )
           VALUES (?, ?, ?, ?, COALESCE(?, 'read_write'), ?, ?)
           ON CONFLICT(service_name) DO UPDATE SET
               encrypted_data = excluded.encrypted_data,
               updated_at = excluded.updated_at,
               rotated_at = excluded.updated_at,
               scope = COALESCE(?, credentials.scope),
               expires_at = excluded.expires_at,
               schema_id = COALESCE(?, credentials.schema_id)""",
        (
            service_name,
            encrypted,
            now,
            now,
            scope,
            expires_at,
            schema_id,
            scope,
            schema_id,
        ),
    )
    await db.commit()
    logger.info(
        f"Stored credential for service: {service_name}",
        service=service_name,
        scope=scope or "read_write",
        has_expires_at=bool(expires_at),
    )
    # Audit trail (T2C). First store is "write"; re-store is "rotate".
    try:
        from . import credential_audit
        await credential_audit.log_access(
            service_name,
            "rotate" if existed else "write",
            True,
            scope=scope or "read_write",
        )
    except Exception:
        pass


async def get_credential(service_name: str) -> dict | None:
    """Retrieve and decrypt a credential from the database.

    Returns ``None`` if the credential doesn't exist, can't be decrypted,
    or has expired.

    Cheap path: the ``expires_at`` column is consulted before decryption.
    Tamper-proof path: after decrypt, the envelope's ``_expires_at`` is also
    checked (defends against an attacker tampering with the plaintext column).
    """
    from ..infra.db import get_db

    async def _audit(success: bool, scope: str | None, error: str | None):
        try:
            from . import credential_audit
            await credential_audit.log_access(
                service_name, "read", success, scope=scope, error=error
            )
        except Exception:
            pass

    db = await get_db()
    cursor = await db.execute(
        "SELECT encrypted_data, expires_at, scope FROM credentials "
        "WHERE service_name = ?",
        (service_name,),
    )
    row = await cursor.fetchone()
    if row is None:
        await _audit(False, None, "not_found")
        return None

    try:
        encrypted_data = row[0] if isinstance(row, tuple) else row["encrypted_data"]
        col_expires_at = row[1] if isinstance(row, tuple) else row["expires_at"]
        row_scope = row[2] if isinstance(row, tuple) else row["scope"]
    except (IndexError, KeyError):
        encrypted_data = row[0]
        col_expires_at = None
        row_scope = None

    # Cheap pre-check via the indexed column — skip decrypt entirely if expired
    if col_expires_at:
        try:
            exp_dt = datetime.fromisoformat(col_expires_at)
            if exp_dt < datetime.now(timezone.utc):
                logger.warning(
                    f"Credential for '{service_name}' expired at "
                    f"{col_expires_at} (column pre-check). "
                    "Please refresh it with /credential add.",
                    service=service_name,
                    expires_at=col_expires_at,
                )
                await _audit(False, row_scope, "expired")
                return None
        except (ValueError, TypeError):
            pass

    try:
        decrypted = _decrypt(encrypted_data)
        payload = json.loads(decrypted)
    except Exception as e:
        logger.error(
            f"Failed to decrypt credential for {service_name}: {e}",
            service=service_name,
            error=str(e),
        )
        await _audit(False, row_scope, f"decrypt_failed: {e}")
        return None

    # Handle both new envelope format and legacy flat format
    if isinstance(payload, dict) and "_data" in payload:
        # New envelope format: tamper-proof expiration recheck
        env_expires_at = payload.get("_expires_at")
        if env_expires_at:
            try:
                exp_dt = datetime.fromisoformat(env_expires_at)
                if exp_dt < datetime.now(timezone.utc):
                    logger.warning(
                        f"Credential for '{service_name}' expired at "
                        f"{env_expires_at}. "
                        "Please refresh it with /credential add.",
                        service=service_name,
                        expires_at=env_expires_at,
                    )
                    await _audit(False, row_scope, "expired_envelope")
                    return None
            except (ValueError, TypeError):
                pass  # malformed expiration — treat as non-expiring
        await _audit(True, row_scope, None)
        return payload["_data"]

    # Legacy flat format (no envelope) — return as-is
    await _audit(True, row_scope, None)
    return payload


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
        logger.info(f"Deleted credential for service: {service_name}", service=service_name)
    try:
        from . import credential_audit
        await credential_audit.log_access(
            service_name,
            "delete",
            deleted,
            error=None if deleted else "not_found",
        )
    except Exception:
        pass
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
