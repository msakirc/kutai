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


def _resolve_report_path(report_path: str | None) -> str | None:
    """Resolve a workspace-relative report path against WORKSPACE_DIR.

    Mechanical executors run with cwd = repo root, but the post-hook payload
    carries a workspace-relative path (e.g.
    ``mission_79/.research/prior_art_report.json``). Without this join
    ``os.path.isfile`` is False and the report is reported missing even
    though it is present on disk (mission_79 #225583/#226311, 2026-05-30).
    """
    if not report_path or os.path.isabs(report_path) or os.path.isfile(report_path):
        return report_path
    try:
        from src.tools.workspace import WORKSPACE_DIR
        cand = os.path.join(WORKSPACE_DIR, report_path)
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    return report_path


def _load_report(
    report: dict[str, Any] | None,
    report_path: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    # A non-empty inline report wins; an empty/None inline report must NOT
    # shadow a good report_path (the producer's stored result is often null
    # when its write_file side-effect landed the file but its result did not).
    if isinstance(report, dict) and report:
        return report, None
    rp = _resolve_report_path(report_path)
    if rp and os.path.isfile(rp):
        try:
            with open(rp, encoding="utf-8") as fh:
                return json.load(fh), None
        except Exception as e:
            return None, f"failed to read {rp}: {e}"
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
    # search_summary is prose (str) in real reports — only an object form
    # carries queries_run / total_results_inspected. Guard so a string
    # summary doesn't AttributeError on .get (mission_79, 2026-05-30).
    summary = payload.get("search_summary")
    if not isinstance(summary, dict):
        summary = {}
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
