"""Z1 Tier 6B (P5) — prior_art_report.json post-hook.

Wired as ``post_hooks: ["prior_art_min_coverage"]`` on step ``1.0
prior_art_search``.

Rules:

1. ``attempted_solutions`` count >= 1 unless ``verdict ==
   "blue_ocean_validated"`` AND coverage >= (3 queries, 20 results).
2. Every ``attempted_solutions`` URL must resolve (non-empty + http/https
   scheme — actual reachability already swept inside
   ``vecihi.find_prior_art``; we accept ``status == "dead"`` as evidence
   the sweep ran). A URL of ``""`` or missing is a hard fail.
3. At least one ``key_lessons`` entry when ``attempted_solutions`` is
   non-empty.
4. Every ``status in ("dead", "dormant")`` entry must carry either a
   ``wayback_first_capture`` field OR an HN reference in ``sources``.
"""
from __future__ import annotations

import json
import os
from typing import Any


def _looks_like_url(u: Any) -> bool:
    if not isinstance(u, str):
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")


def _has_hn_reference(sources: list[Any]) -> bool:
    for s in sources or []:
        if isinstance(s, str) and "ycombinator" in s:
            return True
    return False


def _load_report(
    report: dict[str, Any] | None,
    report_path: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(report, dict):
        return report, None
    if report_path and os.path.isfile(report_path):
        try:
            with open(report_path, encoding="utf-8") as fh:
                return json.load(fh), None
        except Exception as e:
            return None, f"failed to read {report_path}: {e}"
    return None, "no report payload provided"


def prior_art_min_coverage(
    report: dict[str, Any] | None = None,
    report_path: str | None = None,
) -> dict[str, Any]:
    """Validate a prior_art_report against the four rules above.

    Returns ``{"ok": bool, "problems": [...], "verdict": str|None,
    "attempted": int}``.
    """
    payload, err = _load_report(report, report_path)
    if payload is None:
        return {"ok": False, "problems": [err or "missing report"]}

    problems: list[str] = []
    verdict = payload.get("verdict")
    summary = payload.get("search_summary") or {}
    queries = summary.get("queries_run") or []
    inspected = int(summary.get("total_results_inspected") or 0)
    attempted = payload.get("attempted_solutions") or []
    lessons = payload.get("key_lessons") or []

    # Rule 1 — attempted count
    if not attempted:
        if verdict == "blue_ocean_validated":
            if len(queries) < 3 or inspected < 20:
                problems.append(
                    f"blue_ocean claim requires >=3 queries (got {len(queries)}) "
                    f"and >=20 inspected (got {inspected})"
                )
        else:
            problems.append(
                f"verdict={verdict!r} but attempted_solutions is empty; "
                "broaden queries or set verdict=blue_ocean_validated with >=3 "
                "queries + >=20 inspected"
            )

    # Rule 2 — every URL non-empty + scheme valid
    for i, sol in enumerate(attempted):
        url = sol.get("url")
        if not _looks_like_url(url):
            problems.append(
                f"attempted_solutions[{i}].url missing/invalid: {url!r} "
                f"(name={sol.get('name')!r})"
            )

    # Rule 3 — at least one key_lesson when attempted is non-empty
    if attempted and not lessons:
        problems.append(
            "attempted_solutions non-empty but key_lessons is empty; "
            "extract at least one lesson"
        )

    # Rule 4 — dead/dormant entries need Wayback OR HN evidence
    suspicious: list[str] = []
    for sol in attempted:
        if sol.get("status") in ("dead", "dormant"):
            sources = sol.get("sources") or []
            wb = sol.get("wayback_first_capture")
            if not wb and not _has_hn_reference(sources):
                suspicious.append(sol.get("name") or "<unnamed>")
    if suspicious:
        problems.append(
            f"unverifiable dead/dormant solutions (no Wayback + no HN ref): "
            f"{suspicious}"
        )

    return {
        "ok": not problems,
        "problems": problems,
        "verdict": verdict,
        "attempted": len(attempted),
        "graveyard_count": payload.get("graveyard_count"),
    }
