"""Z7 T5 B2 — changelog/draft mechanical executor.

Pulls the mission's git commits since the last changelog entry, maps
conventional-commit prefixes to Keep-A-Changelog buckets, runs the draft
through A5 brand_voice_lint + A6 copy_compliance (degrade gracefully),
writes a draft row (published=0) to changelog_entries, and surfaces a
founder_action "review + publish changelog entry?".

Public surface
--------------
  map_commits_to_kac_buckets(commits: list[str]) -> dict
      Tested independently; no I/O.

  run(payload: dict) -> dict
      mr_roboto executor entry point.
      payload keys:
        product_id   TEXT  (required)
        mission_id   int   (required)
        version      TEXT  (required, e.g. "1.2.0")
        since_entry_id int (optional — fetch commits since this entry's
                           released_at; if omitted, use last published entry)
        git_range    TEXT  (optional — explicit "sha1..sha2" passthrough)

  Returns:
    {"status": "ok", "entry_id": int, "lint_degraded": bool,
     "compliance_degraded": bool, "founder_action_id": int | None}
  or
    {"status": "error", "error": str}
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.changelog_draft")

# ---------------------------------------------------------------------------
# Conventional-commit prefix → KAC bucket mapping
# ---------------------------------------------------------------------------

# Order matters: most specific first.
_PREFIX_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^feat(?:\([^)]*\))?\s*!?:"), "added"),
    (re.compile(r"^fix(?:\([^)]*\))?\s*!?:"), "fixed"),
    (re.compile(r"^refactor(?:\([^)]*\))?\s*!?:"), "changed"),
    (re.compile(r"^perf(?:\([^)]*\))?\s*!?:"), "changed"),
    (re.compile(r"^style(?:\([^)]*\))?\s*!?:"), "changed"),
    (re.compile(r"^deprecate[ds]?(?:\([^)]*\))?\s*!?:"), "deprecated"),
    (re.compile(r"^remove[ds]?(?:\([^)]*\))?\s*!?:"), "removed"),
    (re.compile(r"^revert(?:\([^)]*\))?\s*!?:"), "changed"),
    # build/chore/ci/docs/test — ignored (internal, no user-facing impact)
]

_EMPTY_KAC: dict[str, list[str]] = {
    "added": [],
    "changed": [],
    "fixed": [],
    "deprecated": [],
    "removed": [],
}


def map_commits_to_kac_buckets(commits: list[str]) -> dict[str, list[str]]:
    """Map a list of conventional-commit message subjects to KAC buckets.

    Commits that do not match any mapped prefix (chore/docs/ci/build/test)
    are silently dropped — they represent internal work with no user impact.

    Args:
        commits: List of raw commit subject lines.

    Returns:
        Dict with keys: added, changed, fixed, deprecated, removed.
        Each value is a list of human-readable bullet strings.
    """
    result: dict[str, list[str]] = {k: [] for k in _EMPTY_KAC}

    for commit in commits:
        commit = commit.strip()
        if not commit:
            continue
        for pattern, bucket in _PREFIX_MAP:
            m = pattern.match(commit)
            if m:
                # Strip the prefix to get the human-readable description
                description = commit[m.end():].strip()
                if description:
                    result[bucket].append(description)
                break  # First match wins

    return result


# ---------------------------------------------------------------------------
# Git log helper
# ---------------------------------------------------------------------------

async def _git_log_since_last_entry(
    product_id: str,
    git_range: str | None,
    since_entry_id: int | None,
) -> list[str]:
    """Return commit subject lines in the relevant range.

    Production path: run ``git log --oneline <range>`` via subprocess.
    Falls back to empty list on any error (caller degrades gracefully).
    """
    import subprocess
    import asyncio

    try:
        if git_range:
            cmd = ["git", "log", "--format=%s", git_range]
        else:
            # Find released_at of the last published entry for this product.
            since_date: str | None = None
            if since_entry_id is not None:
                try:
                    from src.infra.db import get_db
                    db = await get_db()
                    cur = await db.execute(
                        "SELECT released_at FROM changelog_entries WHERE entry_id=?",
                        (since_entry_id,),
                    )
                    row = await cur.fetchone()
                    if row and row[0]:
                        since_date = row[0]
                except Exception:
                    pass
            else:
                try:
                    from src.infra.db import get_db
                    db = await get_db()
                    cur = await db.execute(
                        "SELECT released_at FROM changelog_entries "
                        "WHERE product_id=? AND published=1 "
                        "ORDER BY released_at DESC LIMIT 1",
                        (product_id,),
                    )
                    row = await cur.fetchone()
                    if row and row[0]:
                        since_date = row[0]
                except Exception:
                    pass

            if since_date:
                cmd = ["git", "log", "--format=%s", f"--since={since_date}"]
            else:
                # No prior entry — take the last 50 commits
                cmd = ["git", "log", "--format=%s", "-50"]

        loop = asyncio.get_event_loop()
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            ),
        )
        if proc.returncode != 0:
            return []
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return lines
    except Exception as exc:
        logger.warning("changelog_draft: git_log failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Lint helpers (best-effort — degrade gracefully)
# ---------------------------------------------------------------------------

async def _run_brand_voice_lint(body_md: str, task: dict) -> dict:
    """Run A5 brand_voice_lint on body_md. Raises ImportError if unavailable."""
    from general_beckman.posthook_handlers.brand_voice_lint import handle
    fake_task = dict(task)
    fake_task.setdefault("context", {})
    if isinstance(fake_task["context"], str):
        fake_task["context"] = json.loads(fake_task["context"] or "{}")
    fake_task["context"]["brand_voice_audience"] = "marketing"
    return await handle(fake_task, {"result": body_md})


async def _run_copy_compliance(body_md: str, task: dict) -> dict:
    """Run A6 copy_compliance_review on body_md. Raises ImportError if unavailable."""
    from general_beckman.posthook_handlers.copy_compliance_review import handle
    fake_task = dict(task)
    fake_task.setdefault("context", {})
    if isinstance(fake_task["context"], str):
        fake_task["context"] = json.loads(fake_task["context"] or "{}")
    fake_task["context"]["channel"] = "blog_post"
    return await handle(fake_task, {"result": body_md})


# ---------------------------------------------------------------------------
# Founder-action emitter
# ---------------------------------------------------------------------------

async def _emit_founder_action(
    *,
    mission_id: int,
    entry_id: int,
    version: str,
    product_id: str,
    lint_warnings: list[dict],
) -> Any:
    """Surface a founder_action 'review + publish changelog entry?'."""
    try:
        from src.founder_actions import create as fa_create
        instructions = [
            f"Review the draft changelog entry for version {version}.",
            "Edit prose as needed (the LLM draft is a starting point).",
            "Run changelog/publish when satisfied.",
        ]
        if lint_warnings:
            instructions.append(
                f"Note: {len(lint_warnings)} lint warning(s) found — review before publish."
            )
        return await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=f"Ready to publish changelog entry {version}? (product: {product_id})",
            why=(
                f"Mission completed with conventional commits mapped to Keep-A-Changelog "
                f"format. Entry id={entry_id} is in draft state."
            ),
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=False,
        )
    except Exception as exc:
        logger.warning("changelog_draft: _emit_founder_action failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """mr_roboto executor: changelog/draft.

    Writes a draft row to changelog_entries and surfaces a founder_action.
    """
    product_id: str = str(payload.get("product_id") or "")
    mission_id: int = int(payload.get("mission_id") or 0)
    version: str = str(payload.get("version") or "")
    git_range: str | None = payload.get("git_range")
    since_entry_id: int | None = payload.get("since_entry_id")

    if not product_id:
        return {"status": "error", "error": "product_id is required"}
    if not version:
        return {"status": "error", "error": "version is required"}

    lint_degraded = False
    compliance_degraded = False

    # 1. Fetch git commits
    commits = await _git_log_since_last_entry(product_id, git_range, since_entry_id)

    # 2. Map to KAC buckets
    buckets = map_commits_to_kac_buckets(commits)

    # 3. Compose body_md (simple Keep-A-Changelog format)
    body_parts: list[str] = [f"## [{version}]"]
    for bucket_name, items in buckets.items():
        if items:
            body_parts.append(f"\n### {bucket_name.capitalize()}")
            for item in items:
                body_parts.append(f"- {item}")
    body_md = "\n".join(body_parts) if len(body_parts) > 1 else f"## [{version}]\n\nNo user-facing changes."

    # 4. Brand voice lint (A5) — best-effort
    lint_violations: list[dict] = []
    fake_task: dict = {"id": None, "mission_id": mission_id, "context": {}}
    try:
        lint_result = await _run_brand_voice_lint(body_md, fake_task)
        lint_violations = lint_result.get("violations", [])
    except (ImportError, Exception) as exc:
        logger.info("changelog_draft: A5 brand_voice_lint unavailable: %s", exc)
        lint_degraded = True

    # 5. Copy compliance (A6) — best-effort
    try:
        await _run_copy_compliance(body_md, fake_task)
    except (ImportError, Exception) as exc:
        logger.info("changelog_draft: A6 copy_compliance unavailable: %s", exc)
        compliance_degraded = True

    # 6. Write draft row to changelog_entries
    try:
        from src.infra.db import get_db
        db = await get_db()
        from src.infra.times import db_now
        now_str = db_now()

        cur = await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published, "
            " released_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
            (
                product_id,
                version,
                f"Release {version}",
                body_md,
                json.dumps(buckets),
                json.dumps([item for bucket in buckets.values() for item in bucket]),
                json.dumps([mission_id] if mission_id else []),
                now_str,
                now_str,
                now_str,
            ),
        )
        await db.commit()
        entry_id = cur.lastrowid
    except Exception as exc:
        logger.error("changelog_draft: DB insert failed: %s", exc)
        return {"status": "error", "error": f"DB insert failed: {exc}"}

    # 7. Surface founder_action
    fa = await _emit_founder_action(
        mission_id=mission_id,
        entry_id=entry_id,
        version=version,
        product_id=product_id,
        lint_warnings=[v for v in lint_violations if v.get("severity") in ("warning", "blocker")],
    )
    fa_id = getattr(fa, "id", None) if fa else None

    logger.info(
        "changelog_draft: draft entry created",
        entry_id=entry_id,
        product_id=product_id,
        version=version,
        commits_mapped=sum(len(v) for v in buckets.values()),
        lint_degraded=lint_degraded,
        compliance_degraded=compliance_degraded,
    )

    return {
        "status": "ok",
        "entry_id": entry_id,
        "version": version,
        "product_id": product_id,
        "commits_mapped": sum(len(v) for v in buckets.values()),
        "kind_breakdown": buckets,
        "lint_degraded": lint_degraded,
        "compliance_degraded": compliance_degraded,
        "founder_action_id": fa_id,
    }
