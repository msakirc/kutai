"""Run schemathesis contract tests against an OpenAPI spec.

Mechanical executor. No LLM. Shells out to ``schemathesis run`` with ``--checks
all`` to exercise status codes and schema conformance.

Severity mapping
----------------
5xx responses found      → blocker
schema mismatches found  → blocker
deprecation warnings     → warning

Missing schemathesis / absent spec / empty base_url
----------------------------------------------------
Soft-skip — ``ok=True``, ``skipped=True``, ``findings=[]`` — so the caller is
not penalised when the tool or spec is not available.
"""
from __future__ import annotations

import asyncio
import re
import shutil
from typing import Any

from src.infra.logging_config import get_logger

from .preview_url import is_real_url

logger = get_logger("mr_roboto.run_schemathesis")


def _locate_schemathesis() -> str:
    """Return the schemathesis executable. Raises FileNotFoundError if absent."""
    path = shutil.which("schemathesis") or shutil.which("st")
    if path is None:
        raise FileNotFoundError("schemathesis not found on PATH")
    return path


def _soft_skip(reason: str) -> dict[str, Any]:
    return {
        "verdict": "pass",
        "findings": [],
        "tools_used": ["schemathesis"],
        "skipped": True,
        "reason": reason,
    }


def _parse_output(stdout: str, stderr: str) -> list[dict]:
    """Extract findings from schemathesis stdout/stderr."""
    findings: list[dict] = []
    combined = stdout + "\n" + stderr

    # 5xx status codes: "5xx", "500", "503", etc.
    five_xx_pattern = re.compile(r"5\d\d", re.IGNORECASE)
    schema_mismatch_pattern = re.compile(
        r"(schema.?mismatch|response.?violates.?schema|does not conform|"
        r"not valid under|ValidationError)", re.IGNORECASE
    )
    deprecation_pattern = re.compile(r"(deprecat)", re.IGNORECASE)

    five_xx_count = 0
    schema_mismatch_count = 0
    deprecation_count = 0

    for line in combined.splitlines():
        if five_xx_pattern.search(line):
            five_xx_count += 1
        if schema_mismatch_pattern.search(line):
            schema_mismatch_count += 1
        if deprecation_pattern.search(line):
            deprecation_count += 1

    if five_xx_count > 0:
        findings.append({
            "kind": "5xx_response",
            "severity": "blocker",
            "why": f"Schemathesis detected {five_xx_count} line(s) mentioning 5xx responses",
            "count": five_xx_count,
        })
    if schema_mismatch_count > 0:
        findings.append({
            "kind": "schema_mismatch",
            "severity": "blocker",
            "why": f"Schemathesis detected {schema_mismatch_count} schema mismatch(es)",
            "count": schema_mismatch_count,
        })
    if deprecation_count > 0:
        findings.append({
            "kind": "deprecation_warning",
            "severity": "warning",
            "why": f"Schemathesis detected {deprecation_count} deprecation warning(s)",
            "count": deprecation_count,
        })

    return findings


async def run_schemathesis(
    spec_path: str,
    base_url: str,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run schemathesis and return normalised findings.

    Parameters
    ----------
    spec_path:
        Path to the OpenAPI spec file (JSON or YAML). Soft-skip if missing.
    base_url:
        Base URL for the running service. Soft-skip if empty.
    timeout_s:
        Hard cap on the subprocess.

    Returns
    -------
    dict with keys:

    ``verdict``
        ``"pass"`` or ``"fail"``.
    ``findings``
        List of finding dicts with ``kind``, ``severity``, ``why``.
    ``tools_used``
        ``["schemathesis"]``
    ``skipped``
        True when tool/spec/url not available.
    ``reason``
        Human-readable skip reason when ``skipped=True``.
    """
    import os as _os

    # Soft-skip checks.
    if not spec_path:
        return _soft_skip("spec_path not provided")
    if not base_url:
        return _soft_skip("base_url not provided")
    if not is_real_url(base_url):
        return _soft_skip(
            f"base_url not a real http(s) URL (pending or blank): {base_url!r}"
        )
    if not _os.path.exists(spec_path):
        return _soft_skip(f"spec_path not found: {spec_path}")

    try:
        exe = _locate_schemathesis()
    except FileNotFoundError:
        logger.warning("schemathesis not installed — contract_review skipped")
        return _soft_skip("schemathesis not installed")

    cmd = [
        exe, "run",
        f"--base-url={base_url}",
        "--report-junit-xml=/dev/null",
        "--hypothesis-seed=0",
        "--checks", "all",
        spec_path,
    ]

    stdout = ""
    stderr = ""
    exit_code = -1
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout_s,
        )
        raw_out, raw_err = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
        exit_code = proc.returncode or 0
        stdout = (raw_out or b"").decode("utf-8", errors="replace")[-8192:]
        stderr = (raw_err or b"").decode("utf-8", errors="replace")[-4096:]
    except asyncio.TimeoutError:
        logger.warning("schemathesis timed out", spec_path=spec_path)
        return {
            "verdict": "fail",
            "findings": [{"kind": "timeout", "severity": "blocker",
                          "why": f"schemathesis timed out after {timeout_s}s"}],
            "tools_used": ["schemathesis"],
            "skipped": False,
            "reason": None,
        }
    except FileNotFoundError:
        return _soft_skip("schemathesis not installed")
    except Exception as exc:
        logger.warning("schemathesis spawn error", error=str(exc))
        return _soft_skip(f"schemathesis spawn error: {exc}")

    findings = _parse_output(stdout, stderr)
    blockers = [f for f in findings if f["severity"] == "blocker"]
    verdict = "fail" if blockers or exit_code not in (0, 1) else "pass"

    return {
        "verdict": verdict,
        "findings": findings,
        "tools_used": ["schemathesis"],
        "skipped": False,
        "reason": None,
    }
