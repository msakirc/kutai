"""Z8 T5B — dependency_scan mechanical executor.

Wraps ``pip-audit --format json`` (python) or ``npm audit --json`` (node)
as a subprocess. V1 only flags vulnerabilities — auto-merge of patch
versions is deferred to V2 with founder approval.

Returns ``{"ok": bool, "ecosystem": str, "vulnerabilities": [...],
"skipped": bool, "reason": str|None}``.

Skipped (not ok=False) when the tool isn't installed — caller decides
whether that's a hard failure or expected (CI without dev tools).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.dependency_scan")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    ecosystem = (payload.get("ecosystem") or "python").lower()
    workspace = payload.get("workspace_path") or "."
    if ecosystem == "python":
        return _scan_python(workspace, payload)
    if ecosystem == "node":
        return _scan_node(workspace, payload)
    return {
        "ok": False,
        "ecosystem": ecosystem,
        "vulnerabilities": [],
        "skipped": False,
        "reason": f"unsupported ecosystem: {ecosystem!r}",
    }


def _scan_python(workspace: str, payload: dict) -> dict:
    if shutil.which("pip-audit") is None:
        return {
            "ok": True,
            "ecosystem": "python",
            "vulnerabilities": [],
            "skipped": True,
            "reason": "pip-audit not installed",
        }
    try:
        proc = subprocess.run(
            ["pip-audit", "--format", "json"],
            cwd=workspace,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "ecosystem": "python",
            "vulnerabilities": [],
            "skipped": False,
            "reason": "pip-audit timed out",
        }
    except Exception as e:
        return {
            "ok": False,
            "ecosystem": "python",
            "vulnerabilities": [],
            "skipped": False,
            "reason": f"pip-audit invocation failed: {e}",
        }

    out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    if not out:
        return {
            "ok": proc.returncode == 0,
            "ecosystem": "python",
            "vulnerabilities": [],
            "skipped": False,
            "reason": (
                None if proc.returncode == 0
                else f"pip-audit empty stdout, rc={proc.returncode}"
            ),
        }

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "ecosystem": "python",
            "vulnerabilities": [],
            "skipped": False,
            "reason": f"pip-audit JSON parse error: {e}",
        }

    vulns = _flatten_python(parsed)
    return {
        "ok": len(vulns) == 0,
        "ecosystem": "python",
        "vulnerabilities": vulns,
        "skipped": False,
        "reason": None if not vulns else f"{len(vulns)} vulnerable package(s) found",
    }


def _flatten_python(parsed: Any) -> list[dict]:
    """pip-audit JSON has top-level ``dependencies`` list."""
    out: list[dict] = []
    deps = parsed.get("dependencies", []) if isinstance(parsed, dict) else []
    for d in deps:
        if not isinstance(d, dict):
            continue
        name = d.get("name")
        ver = d.get("version")
        for v in d.get("vulns", []) or []:
            if not isinstance(v, dict):
                continue
            out.append({
                "package": name,
                "installed_version": ver,
                "id": v.get("id"),
                "fix_versions": v.get("fix_versions", []),
                "description": v.get("description", ""),
            })
    return out


def _scan_node(workspace: str, payload: dict) -> dict:
    if shutil.which("npm") is None:
        return {
            "ok": True,
            "ecosystem": "node",
            "vulnerabilities": [],
            "skipped": True,
            "reason": "npm not installed",
        }
    try:
        proc = subprocess.run(
            ["npm", "audit", "--json"],
            cwd=workspace,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "ecosystem": "node",
            "vulnerabilities": [],
            "skipped": False,
            "reason": "npm audit timed out",
        }
    except Exception as e:
        return {
            "ok": False,
            "ecosystem": "node",
            "vulnerabilities": [],
            "skipped": False,
            "reason": f"npm audit invocation failed: {e}",
        }

    out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    if not out:
        return {
            "ok": True,
            "ecosystem": "node",
            "vulnerabilities": [],
            "skipped": False,
            "reason": None,
        }
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "ecosystem": "node",
            "vulnerabilities": [],
            "skipped": False,
            "reason": f"npm audit JSON parse error: {e}",
        }
    vulns = _flatten_node(parsed)
    return {
        "ok": len(vulns) == 0,
        "ecosystem": "node",
        "vulnerabilities": vulns,
        "skipped": False,
        "reason": None if not vulns else f"{len(vulns)} vulnerable package(s) found",
    }


def _flatten_node(parsed: Any) -> list[dict]:
    """npm audit v2 JSON has ``vulnerabilities`` keyed by package name."""
    out: list[dict] = []
    vulns_map = parsed.get("vulnerabilities", {}) if isinstance(parsed, dict) else {}
    if not isinstance(vulns_map, dict):
        return out
    for name, info in vulns_map.items():
        if not isinstance(info, dict):
            continue
        out.append({
            "package": name,
            "severity": info.get("severity"),
            "via": info.get("via", []),
            "fix_available": bool(info.get("fixAvailable")),
        })
    return out
