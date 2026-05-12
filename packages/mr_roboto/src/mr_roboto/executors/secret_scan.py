"""Z8 T5C — secret_scan mechanical executor.

V1 wraps ``gitleaks detect --no-banner --report-format json``. When
gitleaks is not installed, the executor returns ``skipped=True`` with a
clear reason — caller decides whether that's a hard failure or expected
in this environment.

Payload
-------
```
{
    "action": "secret_scan",
    "workspace_path": "/abs/path/to/repo",
}
```

Returns ``{"ok": bool, "findings": [...], "skipped": bool,
"reason": str|None}``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.secret_scan")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    workspace = payload.get("workspace_path") or "."

    if shutil.which("gitleaks") is None:
        return {
            "ok": True,
            "findings": [],
            "skipped": True,
            "reason": "gitleaks not installed",
        }

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "gitleaks.json"
        try:
            proc = subprocess.run(
                [
                    "gitleaks", "detect",
                    "--no-banner",
                    "--report-format", "json",
                    "--report-path", str(report_path),
                    "--source", str(workspace),
                ],
                capture_output=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "findings": [],
                "skipped": False,
                "reason": "gitleaks timed out",
            }
        except Exception as e:
            return {
                "ok": False,
                "findings": [],
                "skipped": False,
                "reason": f"gitleaks invocation failed: {e}",
            }

        # gitleaks exits non-zero when findings exist; that's expected.
        findings: list[dict] = []
        if report_path.exists():
            try:
                raw = report_path.read_text(encoding="utf-8", errors="replace").strip()
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        for f in parsed:
                            if not isinstance(f, dict):
                                continue
                            findings.append({
                                "rule": f.get("RuleID") or f.get("Description"),
                                "file": f.get("File"),
                                "line": f.get("StartLine"),
                                "commit": f.get("Commit"),
                            })
            except json.JSONDecodeError as e:
                return {
                    "ok": False,
                    "findings": [],
                    "skipped": False,
                    "reason": f"gitleaks report parse error: {e}",
                }

    return {
        "ok": len(findings) == 0,
        "findings": findings,
        "skipped": False,
        "reason": (
            None if not findings else f"{len(findings)} secret finding(s)"
        ),
    }
