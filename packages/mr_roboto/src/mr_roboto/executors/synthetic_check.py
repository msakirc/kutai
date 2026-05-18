"""Z8 T5F — synthetic_check mechanical executor.

Routes via ``mr_roboto.run`` when ``payload["action"] == "synthetic_check"``.

Two backends:

- **lighthouse** (``backend="lighthouse"``) — subprocess ``lighthouse
  <TARGET_URL> --output=json --quiet``. Parses ``audits`` for FCP/LCP/TBT
  (we synthesize p50/p95/p99 as identical because Lighthouse reports a
  single value per metric — the perf_baselines schema is shared with k6).
- **k6** (``backend="k6"``) — subprocess ``k6 run --summary-export=...``.
  Parses ``metrics.http_req_duration.values.{p(50),p(95),p(99)}``.

Both skip cleanly with ``skipped=true`` if the binary is absent. The
executor compares the new sample against the last green baseline in
``perf_baselines``; on regression it returns ``regression_detected=True``
so the workflow engine can offer the rollback verb via the on-call agent.

A successful (non-regression) run is recorded as the new baseline for
the next comparison.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger
from src.ops.perf_baselines import (
    has_regression,
    latest_green_baseline,
    record_baseline,
    regression_pct,
)

logger = get_logger("mr_roboto.synthetic_check")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    backend = (payload.get("backend") or "lighthouse").lower()
    target_url = payload.get("target_url") or ""
    mission_id = task.get("mission_id")
    release_tag = payload.get("release_tag") or "unknown"
    metric = payload.get("metric") or backend
    threshold_pct = float(payload.get("regression_threshold_pct") or 10.0)

    if not target_url:
        return {
            "ok": False,
            "backend": backend,
            "skipped": False,
            "reason": "missing target_url",
        }

    if backend == "lighthouse":
        sample = _run_lighthouse(target_url, payload)
    elif backend == "k6":
        sample = _run_k6(target_url, payload)
    else:
        return {
            "ok": False,
            "backend": backend,
            "skipped": False,
            "reason": f"unsupported backend: {backend!r}",
        }

    if sample.get("skipped"):
        return sample

    if not sample.get("ok"):
        return sample

    current = {"p50": sample.get("p50"), "p95": sample.get("p95"), "p99": sample.get("p99")}
    baseline = None
    if mission_id is not None:
        baseline = await latest_green_baseline(int(mission_id), metric)

    regressed = has_regression(baseline, current, threshold_pct=threshold_pct)

    delta = {
        "p50": regression_pct(baseline.p50 if baseline else None, current["p50"]),
        "p95": regression_pct(baseline.p95 if baseline else None, current["p95"]),
        "p99": regression_pct(baseline.p99 if baseline else None, current["p99"]),
    }

    if not regressed and mission_id is not None:
        await record_baseline(
            int(mission_id),
            release_tag,
            metric,
            p50=current["p50"],
            p95=current["p95"],
            p99=current["p99"],
        )

    return {
        "ok": not regressed,
        "backend": backend,
        "metric": metric,
        "target_url": target_url,
        "sample": current,
        "baseline": (
            {
                "p50": baseline.p50,
                "p95": baseline.p95,
                "p99": baseline.p99,
                "release_tag": baseline.release_tag,
            }
            if baseline
            else None
        ),
        "delta_pct": delta,
        "regression_detected": regressed,
        "threshold_pct": threshold_pct,
        "skipped": False,
    }


def _run_lighthouse(target_url: str, payload: dict) -> dict:
    if shutil.which("lighthouse") is None:
        return {
            "ok": False,
            "backend": "lighthouse",
            "skipped": True,
            "reason": "lighthouse not on PATH",
        }
    try:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "lh.json"
            subprocess.run(
                [
                    "lighthouse", target_url,
                    "--output=json",
                    "--output-path", str(out_path),
                    "--quiet",
                    "--chrome-flags=--headless",
                ],
                check=True,
                capture_output=True,
                timeout=180,
            )
            data = json.loads(out_path.read_text(encoding="utf-8"))
        audits = data.get("audits") or {}
        # Lighthouse reports a single number per audit; we put it in p50/p95/p99.
        # Use largest_contentful_paint as the canonical perf metric.
        lcp = (audits.get("largest-contentful-paint") or {}).get("numericValue")
        return {
            "ok": True,
            "backend": "lighthouse",
            "p50": float(lcp) if lcp is not None else None,
            "p95": float(lcp) if lcp is not None else None,
            "p99": float(lcp) if lcp is not None else None,
            "skipped": False,
        }
    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "backend": "lighthouse",
            "skipped": False,
            "reason": f"lighthouse failed: {e.stderr!r}",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "backend": "lighthouse",
            "skipped": False,
            "reason": f"lighthouse error: {e}",
        }


def _run_k6(target_url: str, payload: dict) -> dict:
    if shutil.which("k6") is None:
        return {
            "ok": False,
            "backend": "k6",
            "skipped": True,
            "reason": "k6 not on PATH",
        }
    script_path = payload.get("script_path")
    if not script_path or not Path(str(script_path)).is_file():
        return {
            "ok": False,
            "backend": "k6",
            "skipped": False,
            "reason": "missing k6 script_path",
        }
    try:
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "summary.json"
            subprocess.run(
                [
                    "k6", "run",
                    "--summary-export", str(summary),
                    "-e", f"TARGET_URL={target_url}",
                    "-e", f"VUS={payload.get('vus', 5)}",
                    "-e", f"DURATION={payload.get('duration', '30s')}",
                    str(script_path),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            data = json.loads(summary.read_text(encoding="utf-8"))
        m = (data.get("metrics") or {}).get("http_req_duration", {}).get("values", {})
        return {
            "ok": True,
            "backend": "k6",
            "p50": m.get("p(50)"),
            "p95": m.get("p(95)"),
            "p99": m.get("p(99)"),
            "skipped": False,
        }
    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "backend": "k6",
            "skipped": False,
            "reason": f"k6 failed: {e.stderr!r}",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "backend": "k6",
            "skipped": False,
            "reason": f"k6 error: {e}",
        }
