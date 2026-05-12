"""Z8 T5C — cve_scan mechanical executor.

Queries OSV.dev ``POST /v1/query`` for each ``(name, version)`` pair in
the supplied package list. No auth required.

Payload shape
-------------
```
{
    "action": "cve_scan",
    "ecosystem": "PyPI" | "npm" | "Debian" | ...,
    "packages": [{"name": "...", "version": "..."}, ...],
}
```

Returns ``{"ok": bool, "ecosystem": str, "queried": int,
"vulnerabilities": [...], "skipped": bool, "reason": str|None}``.

``vulnerabilities`` is a flat list of
``{"package", "version", "id", "summary", "severity"}`` entries.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.cve_scan")

_OSV_QUERY_URL = "https://api.osv.dev/v1/query"
_DEFAULT_TIMEOUT = 15.0


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    ecosystem = (payload.get("ecosystem") or "PyPI").strip()
    packages = payload.get("packages") or []

    if not isinstance(packages, list) or not packages:
        return {
            "ok": True,
            "ecosystem": ecosystem,
            "queried": 0,
            "vulnerabilities": [],
            "skipped": False,
            "reason": "no packages supplied",
        }

    try:
        import aiohttp  # type: ignore[import]
    except ImportError:
        return {
            "ok": False,
            "ecosystem": ecosystem,
            "queried": 0,
            "vulnerabilities": [],
            "skipped": True,
            "reason": "aiohttp not installed",
        }

    vulns: list[dict] = []
    queried = 0
    timeout = aiohttp.ClientTimeout(total=_DEFAULT_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for pkg in packages:
                if not isinstance(pkg, dict):
                    continue
                name = pkg.get("name")
                version = pkg.get("version")
                if not name:
                    continue
                queried += 1
                body = {
                    "package": {"name": name, "ecosystem": ecosystem},
                }
                if version:
                    body["version"] = version
                try:
                    async with session.post(_OSV_QUERY_URL, json=body) as resp:
                        if resp.status != 200:
                            logger.debug(
                                "osv non-200 for %s@%s: %s",
                                name, version, resp.status,
                            )
                            continue
                        data = await resp.json()
                except Exception as e:
                    logger.debug("osv query failed for %s: %s", name, e)
                    continue
                for v in (data.get("vulns") or []):
                    if not isinstance(v, dict):
                        continue
                    vulns.append({
                        "package": name,
                        "version": version,
                        "id": v.get("id"),
                        "summary": (v.get("summary") or "")[:200],
                        "severity": _max_severity(v.get("severity") or []),
                    })
    except Exception as e:
        return {
            "ok": False,
            "ecosystem": ecosystem,
            "queried": queried,
            "vulnerabilities": vulns,
            "skipped": False,
            "reason": f"osv session failed: {e}",
        }

    return {
        "ok": len(vulns) == 0,
        "ecosystem": ecosystem,
        "queried": queried,
        "vulnerabilities": vulns,
        "skipped": False,
        "reason": (
            None if not vulns
            else f"{len(vulns)} vulnerability/ies across {queried} package(s)"
        ),
    }


def _max_severity(severities: list) -> str | None:
    """OSV severity is a list of {type, score}. Pick the first CVSS_V3 score."""
    for s in severities:
        if not isinstance(s, dict):
            continue
        if (s.get("type") or "").startswith("CVSS"):
            return str(s.get("score") or "")
    return None
