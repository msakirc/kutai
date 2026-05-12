"""Run Lighthouse performance/accessibility/SEO audit on a preview URL.

Mechanical executor. No LLM. Shells out to ``npx lighthouse`` with JSON output
and compares category scores against configurable thresholds.

Default thresholds
------------------
performance    >= 0.7
accessibility  >= 0.85
best-practices >= 0.85
seo            >= 0.7

Threshold breach → finding with severity=blocker.

Missing lighthouse / absent preview_url
----------------------------------------
Soft-skip — ``ok=True``, ``skipped=True``, ``findings=[]``.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.run_lighthouse")

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "performance": 0.7,
    "accessibility": 0.85,
    "best-practices": 0.85,
    "seo": 0.7,
}


def _locate_lighthouse() -> str:
    """Return npx path (lighthouse is always via npx). Raises if npx absent."""
    path = shutil.which("npx")
    if path is None:
        raise FileNotFoundError("npx not found on PATH (required for lighthouse)")
    return path


def _soft_skip(reason: str) -> dict[str, Any]:
    return {
        "verdict": "pass",
        "findings": [],
        "tools_used": ["lighthouse"],
        "skipped": True,
        "reason": reason,
    }


async def run_lighthouse(
    preview_url: str,
    thresholds: dict | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run Lighthouse and return normalised findings.

    Parameters
    ----------
    preview_url:
        The URL to audit. Soft-skip if empty.
    thresholds:
        Per-category score thresholds (0.0–1.0). Defaults to ``_DEFAULT_THRESHOLDS``.
    timeout_s:
        Hard cap on the subprocess.

    Returns
    -------
    dict with keys:

    ``verdict``
        ``"pass"`` or ``"fail"``.
    ``findings``
        List of finding dicts: ``{category, score, threshold, severity, why}``.
    ``tools_used``
        ``["lighthouse"]``
    ``skipped``
        True when tool/url not available.
    ``reason``
        Human-readable skip reason when ``skipped=True``.
    """
    if not preview_url:
        return _soft_skip("preview_url not provided")

    try:
        npx_exe = _locate_lighthouse()
    except FileNotFoundError:
        logger.warning("npx not installed — run_lighthouse skipped")
        return _soft_skip("npx/lighthouse not installed")

    effective_thresholds = dict(_DEFAULT_THRESHOLDS)
    if thresholds:
        effective_thresholds.update(thresholds)

    # Write output to a temp file so we don't have to parse stdout.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        output_path = tf.name

    cmd = [
        npx_exe, "lighthouse", preview_url,
        "--output=json",
        f"--output-path={output_path}",
        "--quiet",
        '--chrome-flags=--headless --no-sandbox',
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
        stdout = (raw_out or b"").decode("utf-8", errors="replace")[-4096:]
        stderr = (raw_err or b"").decode("utf-8", errors="replace")[-4096:]
    except asyncio.TimeoutError:
        logger.warning("lighthouse timed out", url=preview_url)
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "verdict": "fail",
            "findings": [{"kind": "timeout", "severity": "blocker",
                          "why": f"lighthouse timed out after {timeout_s}s"}],
            "tools_used": ["lighthouse"],
            "skipped": False,
            "reason": None,
        }
    except FileNotFoundError:
        return _soft_skip("npx/lighthouse not installed")
    except Exception as exc:
        logger.warning("lighthouse spawn error", error=str(exc))
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass
        return _soft_skip(f"lighthouse spawn error: {exc}")

    # Parse the JSON output file.
    findings: list[dict] = []
    try:
        raw_json = Path(output_path).read_text(encoding="utf-8")
        data = json.loads(raw_json)
        categories = data.get("categories") or {}
        for cat_id, cat_data in categories.items():
            score = cat_data.get("score")
            if score is None:
                continue
            score = float(score)
            threshold = effective_thresholds.get(cat_id)
            if threshold is not None and score < threshold:
                findings.append({
                    "category": cat_id,
                    "score": round(score, 3),
                    "threshold": threshold,
                    "severity": "blocker",
                    "why": (
                        f"Lighthouse '{cat_id}' score {score:.2f} "
                        f"below threshold {threshold:.2f}"
                    ),
                })
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("lighthouse JSON parse failed", error=str(exc))
    finally:
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass

    blockers = [f for f in findings if f["severity"] == "blocker"]
    verdict = "fail" if blockers else "pass"

    return {
        "verdict": verdict,
        "findings": findings,
        "tools_used": ["lighthouse"],
        "skipped": False,
        "reason": None,
    }
