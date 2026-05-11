# src/security/credential_schemas.py
"""Per-vendor credential schema loader and validator.

Schemas live in ``credential_schemas/<service_name>.json`` at the repo root
(sibling to ``compliance_templates/``). Each schema declares the required and
optional fields a credential payload must carry, the legal scope labels, the
default scope, a rotation-recommended cadence, an optional test-ping action,
and a docs URL surfaced via ``/credential schema <service>``.

Loading is process-cached at import to keep `store_credential()` cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("security.credential_schemas")

# Resolve repo root once: this file lives at <repo>/src/security/. The
# schemas directory is <repo>/credential_schemas/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMAS_DIR = _REPO_ROOT / "credential_schemas"

_REQUIRED_TOP_KEYS = (
    "service_name",
    "required_fields",
    "optional_fields",
    "scopes",
    "default_scope",
)


def _schemas_dir() -> Path:
    return _SCHEMAS_DIR


def _read_schema_file(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(
            f"failed to load credential schema {path.name}: {exc}",
            path=str(path),
            error=str(exc),
        )
        return None
    missing = [k for k in _REQUIRED_TOP_KEYS if k not in data]
    if missing:
        logger.error(
            f"credential schema {path.name} missing keys: {missing}",
            path=str(path),
            missing=missing,
        )
        return None
    return data


def _build_cache() -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not _SCHEMAS_DIR.is_dir():
        return cache
    for entry in sorted(_SCHEMAS_DIR.glob("*.json")):
        data = _read_schema_file(entry)
        if data is None:
            continue
        name = data.get("service_name") or entry.stem
        cache[name] = data
    return cache


# Process-level cache built lazily on first access so tests can monkeypatch
# `_SCHEMAS_DIR` before the first call.
_CACHE: dict[str, dict[str, Any]] | None = None


def _cache() -> dict[str, dict[str, Any]]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _build_cache()
    return _CACHE


def reset_cache() -> None:
    """Clear the loader cache (test helper)."""
    global _CACHE
    _CACHE = None


def known_services() -> list[str]:
    """Return sorted service names that have a schema file."""
    return sorted(_cache().keys())


def load_schema(service_name: str) -> dict[str, Any] | None:
    """Return the schema dict for *service_name* or ``None`` if absent."""
    return _cache().get(service_name)


def validate_payload(
    service_name: str,
    payload: dict[str, Any],
    scope: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate *payload* against the schema for *service_name*.

    Returns ``(ok, errors)``. When no schema is registered for the service,
    returns ``(True, [])`` so callers can opt-in to validation by simply
    shipping a schema file.

    Rules:
      * Every ``required_fields`` entry must be present and non-empty.
      * Any payload key not in ``required_fields`` ∪ ``optional_fields`` is
        flagged as unknown.
      * If *scope* is provided, it must be in ``schema.scopes``.
    """
    schema = load_schema(service_name)
    if schema is None:
        return True, []

    errors: list[str] = []
    required = list(schema.get("required_fields") or [])
    optional = list(schema.get("optional_fields") or [])
    allowed_scopes = list(schema.get("scopes") or [])

    for field in required:
        value = payload.get(field)
        if value is None or value == "":
            errors.append(f"missing required field: {field}")

    allowed_keys = set(required) | set(optional)
    for key in payload:
        if key not in allowed_keys:
            errors.append(f"unknown field: {key}")

    if scope is not None and allowed_scopes and scope not in allowed_scopes:
        errors.append(
            f"invalid scope '{scope}'; allowed: {', '.join(allowed_scopes)}"
        )

    return (not errors), errors


def describe_schema(service_name: str) -> str:
    """Render a human-readable schema summary for Telegram surfacing."""
    schema = load_schema(service_name)
    if schema is None:
        return f"No schema registered for `{service_name}`."
    lines = [
        f"*Schema: {schema['service_name']}*",
        f"Required: `{', '.join(schema.get('required_fields') or []) or '(none)'}`",
        f"Optional: `{', '.join(schema.get('optional_fields') or []) or '(none)'}`",
        f"Scopes: `{', '.join(schema.get('scopes') or [])}`",
        f"Default scope: `{schema.get('default_scope', 'read_write')}`",
        f"Rotate every: {schema.get('rotation_recommended_days', 90)}d",
    ]
    docs = schema.get("docs_url")
    if docs:
        lines.append(f"Docs: {docs}")
    return "\n".join(lines)
