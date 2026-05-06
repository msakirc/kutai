"""Workflow-step exemplar store.

Goal: when a workflow step (`[X.Y] step_name`) passes grading, capture the
*result text* and the *tool call sequence* under the exact key
``(workflow, step_id, agent_type)``. Future runs of the same step retrieve
the top-N exemplars and inject them as worked examples — no vector match,
no cross-step pollution.

This replaces the step-keyed `auto:` skill capture path that minted noise
from grader prose.

Schema (created lazily):
    workflow_exemplars
      id, workflow, step_id, agent_type,
      result, tool_recipe (JSON list of {tool, args}),
      quality_score, task_id, mission_id, created_at

Invariants:
  * Top ``MAX_PER_KEY`` exemplars per (workflow, step_id, agent_type) ranked
    by ``quality_score DESC, id DESC``. Older/lower entries pruned on insert.
  * Step ID is whatever sits inside the leading ``[...]`` of the title;
    sub-step variants like ``[8.F-00.feat.15]`` are kept as-is — same step
    on different features still aggregate.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from src.infra.db import get_db
from src.infra.logging_config import get_logger

logger = get_logger("memory.workflow_exemplars")

MAX_PER_KEY = 3
RESULT_MAX_CHARS = 8000
RECIPE_MAX_CALLS = 25

_STEP_ID_RE = re.compile(r"^\s*\[([^\]]+)\]")


def extract_step_id(title: str) -> Optional[str]:
    """Return the bracketed step id from a title, or None.

    Accepts any non-bracket payload — ``[7.6]``, ``[3.10a]``,
    ``[8.F-00.feat.15]``, ``[2.spike]``.
    """
    if not title:
        return None
    m = _STEP_ID_RE.match(title)
    return m.group(1) if m else None


async def _ensure_table() -> None:
    db = await get_db()
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS workflow_exemplars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow TEXT,
            step_id TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            result TEXT,
            tool_recipe TEXT DEFAULT '[]',
            quality_score REAL DEFAULT 0,
            task_id INTEGER,
            mission_id INTEGER,
            created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
        )
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_wfex_key "
        "ON workflow_exemplars(workflow, step_id, agent_type)"
    )


async def _tool_recipe_for_task(task_id: int) -> list[dict]:
    """Reconstruct the ordered successful tool calls for a task.

    Reads ``audit_log`` for ``tool_exec`` entries (full args up to 500 chars)
    and cross-checks against ``task_traces`` for success markers.
    Failed calls (output starting with ``❌``) are dropped.
    """
    if not task_id:
        return []
    db = await get_db()

    # Audit gives ordered tool, args. Trace gives success.
    rows = []
    async with db.execute(
        "SELECT target, details FROM audit_log "
        "WHERE task_id=? AND action='tool_exec' "
        "ORDER BY id ASC",
        (task_id,),
    ) as cur:
        async for r in cur:
            rows.append((r[0], r[1]))

    # success_set = ordinal indices of trace tool entries that started ✅
    success_indices: set[int] = set()
    async with db.execute(
        "SELECT trace FROM task_traces WHERE task_id=?",
        (task_id,),
    ) as cur:
        row = await cur.fetchone()
    if row:
        try:
            trace = json.loads(row[0])
            tool_idx = 0
            for entry in trace:
                if entry.get("type") == "tool":
                    out = entry.get("output", "") or ""
                    if out.lstrip().startswith("✅"):
                        success_indices.add(tool_idx)
                    tool_idx += 1
        except (json.JSONDecodeError, TypeError):
            pass

    recipe: list[dict] = []
    for i, (tool_name, details) in enumerate(rows):
        # If we have trace data, only keep successful calls. If we don't
        # (older tasks), keep everything — better to ship something.
        if success_indices and i not in success_indices:
            continue
        try:
            args = json.loads(details) if details and details.startswith("{") else {}
        except (json.JSONDecodeError, TypeError):
            # repr-style "{'key': 'val'}" — fall through with raw details
            args = {"_raw": details}
        recipe.append({"tool": tool_name, "args": args})
        if len(recipe) >= RECIPE_MAX_CALLS:
            break
    return recipe


async def capture_exemplar(
    *,
    workflow: Optional[str],
    step_id: str,
    agent_type: str,
    result: str,
    quality_score: float,
    task_id: int,
    mission_id: Optional[int] = None,
) -> bool:
    """Persist an exemplar and prune the key to the top MAX_PER_KEY by quality.

    Returns True if the exemplar was kept (made the top-N), False if pruned
    immediately (its quality lost to all incumbents).
    """
    if not step_id or not agent_type:
        return False
    await _ensure_table()
    db = await get_db()

    tool_recipe = await _tool_recipe_for_task(task_id)
    result_clipped = (result or "")[:RESULT_MAX_CHARS]

    await db.execute(
        "INSERT INTO workflow_exemplars "
        "(workflow, step_id, agent_type, result, tool_recipe, quality_score, task_id, mission_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            workflow or "",
            step_id,
            agent_type,
            result_clipped,
            json.dumps(tool_recipe, ensure_ascii=False),
            float(quality_score or 0),
            task_id,
            mission_id,
        ),
    )

    # Prune: keep top MAX_PER_KEY by quality_score DESC, id DESC.
    await db.execute(
        """
        DELETE FROM workflow_exemplars
        WHERE id IN (
            SELECT id FROM workflow_exemplars
            WHERE workflow=? AND step_id=? AND agent_type=?
            ORDER BY quality_score DESC, id DESC
            LIMIT -1 OFFSET ?
        )
        """,
        (workflow or "", step_id, agent_type, MAX_PER_KEY),
    )
    logger.info(
        "Workflow exemplar captured: wf=%s step=%s agent=%s task#%s q=%s "
        "recipe_calls=%d",
        workflow, step_id, agent_type, task_id, quality_score, len(tool_recipe),
    )
    return True


async def lookup_exemplars(
    *,
    workflow: Optional[str],
    step_id: str,
    agent_type: str,
) -> list[dict]:
    """Return up to MAX_PER_KEY exemplars for the exact key, best first."""
    if not step_id or not agent_type:
        return []
    await _ensure_table()
    db = await get_db()
    out: list[dict] = []
    async with db.execute(
        "SELECT result, tool_recipe, quality_score, task_id "
        "FROM workflow_exemplars "
        "WHERE workflow=? AND step_id=? AND agent_type=? "
        "ORDER BY quality_score DESC, id DESC LIMIT ?",
        (workflow or "", step_id, agent_type, MAX_PER_KEY),
    ) as cur:
        async for r in cur:
            try:
                recipe = json.loads(r[1]) if r[1] else []
            except (json.JSONDecodeError, TypeError):
                recipe = []
            out.append({
                "result": r[0] or "",
                "tool_recipe": recipe,
                "quality_score": r[2],
                "task_id": r[3],
            })
    return out


def format_exemplars_for_prompt(
    exemplars: list[dict],
    *,
    step_id: str,
    max_chars: int = 2400,
) -> str:
    """Render exemplars as a compact reference block for prompt injection.

    Shows the top exemplar's result (truncated) plus a one-liner tool recipe
    summarising what tool calls actually worked. Additional exemplars hint at
    diversity but only contribute their tool sequences (cheaper).
    """
    if not exemplars:
        return ""
    top = exemplars[0]
    result_snip = (top.get("result") or "").strip()
    if len(result_snip) > max_chars:
        result_snip = result_snip[:max_chars].rstrip() + "…"

    recipe_lines: list[str] = []
    for ex in exemplars:
        seq = []
        for call in (ex.get("tool_recipe") or [])[:8]:
            tool = call.get("tool", "?")
            args = call.get("args") or {}
            if isinstance(args, dict) and args:
                # Show first 2 keys for shape hint; values clipped.
                shown = ", ".join(
                    f"{k}={str(v)[:40]!r}" for k, v in list(args.items())[:2]
                )
                seq.append(f"{tool}({shown})")
            else:
                seq.append(f"{tool}()")
        if seq:
            recipe_lines.append(" → ".join(seq))

    parts = [
        f"## Reference: prior `[{step_id}]` output that passed grading",
        result_snip,
    ]
    if recipe_lines:
        parts.append("")
        parts.append("**Tool sequence(s) that worked:**")
        for line in recipe_lines[:MAX_PER_KEY]:
            parts.append(f"- {line}")
    return "\n".join(parts)
