"""Run k6 load test script and evaluate against thresholds.

Mechanical executor. No LLM. Shells out to ``k6 run --quiet
--summary-export=<tmpfile> <script_path>`` and reads the summary JSON.

Default thresholds
------------------
http_req_failed_rate   <= 0.01  (1% failure rate)
http_req_duration_p95_ms <= 1000 (95th-pct latency under 1s)

Threshold breach → finding with severity=blocker.

Missing k6 / absent script_path
---------------------------------
Soft-skip — ``verdict="pass"``, ``skipped=True``, ``findings=[]``.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.run_k6")

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "http_req_failed_rate": 0.01,
    "http_req_duration_p95_ms": 1000.0,
}


def _locate_k6() -> str:
    """Return the k6 executable path. Raises FileNotFoundError if absent."""
    path = shutil.which("k6")
    if path is None:
        raise FileNotFoundError("k6 not found on PATH")
    return path


def _soft_skip(reason: str) -> dict[str, Any]:
    return {
        "verdict": "pass",
        "findings": [],
        "tools_used": ["k6"],
        "skipped": True,
        "reason": reason,
    }


def _evaluate_thresholds(
    summary: dict,
    thresholds: dict[str, float],
) -> list[dict]:
    """Compare k6 summary metrics against thresholds; return findings."""
    findings: list[dict] = []
    metrics = summary.get("metrics") or {}

    # http_req_failed: rate
    req_failed = metrics.get("http_req_failed") or {}
    rate = req_failed.get("rate") or req_failed.get("values", {}).get("rate")
    if rate is not None:
        limit = thresholds.get("http_req_failed_rate", _DEFAULT_THRESHOLDS["http_req_failed_rate"])
        if float(rate) > limit:
            findings.append({
                "metric": "http_req_failed_rate",
                "value": round(float(rate), 4),
                "threshold": limit,
                "severity": "blocker",
                "why": (
                    f"k6 http_req_failed rate {rate:.4f} "
                    f"exceeds threshold {limit:.4f}"
                ),
            })

    # http_req_duration: p95 in ms
    req_duration = metrics.get("http_req_duration") or {}
    p95 = (
        req_duration.get("p(95)")
        or req_duration.get("values", {}).get("p(95)")
        or req_duration.get("values", {}).get("p95")
    )
    if p95 is not None:
        limit = thresholds.get("http_req_duration_p95_ms", _DEFAULT_THRESHOLDS["http_req_duration_p95_ms"])
        if float(p95) > limit:
            findings.append({
                "metric": "http_req_duration_p95_ms",
                "value": round(float(p95), 2),
                "threshold": limit,
                "severity": "blocker",
                "why": (
                    f"k6 http_req_duration p95 {p95:.2f}ms "
                    f"exceeds threshold {limit:.0f}ms"
                ),
            })

    return findings


async def run_k6(
    script_path: str,
    thresholds: dict | None = None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Run k6 and return normalised findings.

    Parameters
    ----------
    script_path:
        Path to the k6 JavaScript test script. Soft-skip if missing.
    thresholds:
        Override default thresholds. Keys: ``http_req_failed_rate``,
        ``http_req_duration_p95_ms``.
    timeout_s:
        Hard cap on the subprocess.

    Returns
    -------
    dict with keys:

    ``verdict``
        ``"pass"`` or ``"fail"``.
    ``findings``
        List of finding dicts: ``{metric, value, threshold, severity, why}``.
    ``tools_used``
        ``["k6"]``
    ``skipped``
        True when tool/script not available.
    ``reason``
        Human-readable skip reason when ``skipped=True``.
    """
    import os as _os

    if not script_path:
        return _soft_skip("script_path not provided")
    if not _os.path.exists(script_path):
        return _soft_skip(f"script_path not found: {script_path}")

    try:
        exe = _locate_k6()
    except FileNotFoundError:
        logger.warning("k6 not installed — run_k6 skipped")
        return _soft_skip("k6 not installed")

    effective_thresholds = dict(_DEFAULT_THRESHOLDS)
    if thresholds:
        effective_thresholds.update(thresholds)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        export_path = tf.name

    cmd = [exe, "run", "--quiet", f"--summary-export={export_path}", script_path]

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
        _, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        exit_code = proc.returncode or 0
    except asyncio.TimeoutError:
        logger.warning("k6 timed out", script_path=script_path)
        try:
            Path(export_path).unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "verdict": "fail",
            "findings": [{"kind": "timeout", "severity": "blocker",
                          "why": f"k6 timed out after {timeout_s}s"}],
            "tools_used": ["k6"],
            "skipped": False,
            "reason": None,
        }
    except FileNotFoundError:
        return _soft_skip("k6 not installed")
    except Exception as exc:
        logger.warning("k6 spawn error", error=str(exc))
        try:
            Path(export_path).unlink(missing_ok=True)
        except Exception:
            pass
        return _soft_skip(f"k6 spawn error: {exc}")

    findings: list[dict] = []
    try:
        raw_json = Path(export_path).read_text(encoding="utf-8")
        summary = json.loads(raw_json)
        findings = _evaluate_thresholds(summary, effective_thresholds)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("k6 summary JSON parse failed", error=str(exc))
        if exit_code != 0:
            findings.append({
                "kind": "run_error",
                "severity": "blocker",
                "why": f"k6 exited {exit_code} and summary parse failed: {exc}",
            })
    finally:
        try:
            Path(export_path).unlink(missing_ok=True)
        except Exception:
            pass

    blockers = [f for f in findings if f["severity"] == "blocker"]
    verdict = "fail" if blockers or exit_code not in (0,) else "pass"

    return {
        "verdict": verdict,
        "findings": findings,
        "tools_used": ["k6"],
        "skipped": False,
        "reason": None,
    }
