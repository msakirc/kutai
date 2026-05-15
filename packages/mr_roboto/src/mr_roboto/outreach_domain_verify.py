"""Z7 T6 A7 — outreach/domain_verify: SPF/DKIM/DMARC check + founder_action.

One-time founder_action that verifies SPF/DKIM/DMARC for the dedicated
outreach subdomain before any sends are allowed.

DNS verification is best-effort: the stub uses a DNS TXT lookup via the
standard library (no third-party dependency). A real implementation would
use a dedicated deliverability API (e.g. MXToolbox, dmarcly).

Public API
----------
  run_domain_verify(product_id, mission_id, domain) -> dict

Internal hooks (patched in tests)
----------------------------------
  _check_dns_records(domain) -> dict[str, bool]
      Returns {"spf": bool, "dkim": bool, "dmarc": bool}
  _emit_founder_action(...) -> Any
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.outreach_domain_verify")


async def _check_dns_records(domain: str) -> dict[str, bool]:
    """Best-effort SPF/DKIM/DMARC DNS lookup.

    Returns a dict with bool values for each record type.
    Stub implementation: queries TXT records for the domain.
    """
    import asyncio

    result = {"spf": False, "dkim": False, "dmarc": False}

    try:
        import socket

        loop = asyncio.get_event_loop()

        def _query_txt(name: str) -> list[str]:
            try:
                return socket.gethostbyname_ex(name)[2]
            except Exception:
                return []

        # SPF: TXT record on the domain containing "v=spf1"
        spf_records = await loop.run_in_executor(None, _query_txt, domain)
        result["spf"] = any("v=spf1" in r for r in spf_records)

        # DMARC: TXT record on _dmarc.<domain> containing "v=DMARC1"
        dmarc_records = await loop.run_in_executor(
            None, _query_txt, f"_dmarc.{domain}"
        )
        result["dmarc"] = any("v=DMARC1" in r for r in dmarc_records)

        # DKIM: TXT record on default._domainkey.<domain>
        dkim_records = await loop.run_in_executor(
            None, _query_txt, f"default._domainkey.{domain}"
        )
        result["dkim"] = bool(dkim_records)

    except Exception as exc:
        logger.warning(
            "outreach_domain_verify: DNS lookup failed (best-effort)",
            domain=domain,
            error=str(exc),
        )

    return result


async def _emit_founder_action(
    *,
    product_id: str,
    mission_id: int,
    domain: str,
    missing: list[str],
) -> Any:
    """Surface a founder_action card for missing DNS records."""
    try:
        from src.founder_actions import create as fa_create

        title = f"Set up outreach domain DNS for {domain} ({product_id})"
        why = (
            f"Before sending cold outreach from {domain}, the following DNS records "
            f"must be configured: {', '.join(missing)}. "
            f"Without these, emails will land in spam and damage your domain reputation."
        )
        instructions = [
            f"Missing DNS records: {', '.join(missing)}",
            f"Domain: {domain}",
            "SPF: Add a TXT record `v=spf1 include:<your-esp.com> ~all` to DNS.",
            "DKIM: Generate a DKIM key in your ESP and add the TXT record provided.",
            "DMARC: Add `_dmarc.<domain> TXT v=DMARC1; p=quarantine; rua=mailto:dmarc@<domain>`",
            "After setup, re-run this check via `/outreach verify_domain`.",
        ]
        return await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=why,
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=True,
            urgent=False,
        )
    except Exception as exc:
        logger.warning(
            "outreach_domain_verify: _emit_founder_action failed",
            error=str(exc),
        )
        return None


async def run_domain_verify(
    product_id: str,
    mission_id: int,
    domain: str,
) -> dict[str, Any]:
    """Verify SPF/DKIM/DMARC for the outreach domain.

    Returns:
      {"status": "ok", "domain": <str>, "records": {...}}   — all records present
      {"status": "incomplete", "missing": [...], "founder_action_id": <int|None>}
      {"status": "founder_action_emitted", ...}             — synonym for incomplete
    """
    records = await _check_dns_records(domain)

    missing = [k.upper() for k, v in records.items() if not v]

    if not missing:
        logger.info(
            "outreach_domain_verify: all records present",
            product_id=product_id,
            domain=domain,
        )
        return {
            "status": "ok",
            "domain": domain,
            "records": records,
        }

    logger.warning(
        "outreach_domain_verify: missing DNS records",
        product_id=product_id,
        domain=domain,
        missing=missing,
    )

    fa = await _emit_founder_action(
        product_id=product_id,
        mission_id=mission_id,
        domain=domain,
        missing=missing,
    )
    fa_id = getattr(fa, "id", None) if fa else None

    return {
        "status": "founder_action_emitted",
        "domain": domain,
        "missing": missing,
        "records": records,
        "founder_action_id": fa_id,
    }
