"""Z6 T3D — pipe-separated ``real_tool_kind`` resolver.

A step's ``real_tool_kind`` may be a single vendor (``"vercel"``) or a
pipe-list of acceptable substitutes (``"vercel|railway|supabase"``,
``"datadog|sentry|new_relic"``). This module picks the first kind that has
**both** (a) an adapter registered in IntegrationRegistry AND (b) credentials
stored in the vault. Returns ``None`` when none qualify.

Caller (Z6 admission gate) decides what to do with ``None`` — typically a
``vendor_enroll`` founder_action listing the original candidate set.

Pure read-only; safe to call on every pump tick.
"""
from __future__ import annotations

from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("integrations.resolver")


def _split_kinds(real_tool_kind: str | list | None) -> list[str]:
    if not real_tool_kind:
        return []
    if isinstance(real_tool_kind, list):
        return [str(k).strip() for k in real_tool_kind if str(k).strip()]
    return [k.strip() for k in str(real_tool_kind).split("|") if k.strip()]


async def resolve_real_tool(real_tool_kind: str | list | None) -> Optional[str]:
    """Return the first kind with adapter + credentials, else None.

    Async because credential lookup hits the encrypted store.
    """
    kinds = _split_kinds(real_tool_kind)
    if not kinds:
        return None

    try:
        from src.integrations.registry import get_integration_registry
        registry = get_integration_registry()
    except Exception as exc:  # noqa: BLE001
        logger.debug("registry import failed", error=str(exc))
        return None

    # Defer the credential import; tests can patch it cleanly.
    try:
        from src.security.credential_store import get_credential
    except Exception as exc:  # noqa: BLE001
        logger.debug("credential_store import failed", error=str(exc))
        get_credential = None  # type: ignore[assignment]

    for kind in kinds:
        if registry.get(kind) is None:
            continue
        if get_credential is None:
            # No vault available — fall back to adapter-only match (best
            # we can do; admission gate still emits the credential_paste
            # founder_action downstream).
            return kind
        try:
            cred = await get_credential(kind)
        except Exception as exc:  # noqa: BLE001
            logger.debug("credential lookup raised", kind=kind, error=str(exc))
            cred = None
        if cred:
            return kind

    return None


__all__ = ["resolve_real_tool"]
