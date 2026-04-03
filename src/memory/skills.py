# memory/skills.py
"""
Skill Library v2 — Vector-only matching, strategy accumulation, adaptive injection.

Replaces the old regex-based matching system. Skills are matched via ChromaDB
vector similarity, strategies accumulate with dedup and pruning, and injection
formatting adapts based on confidence and context budget.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from src.infra.logging_config import get_logger
from src.infra.db import (
    get_db,
    upsert_skill,
    get_all_skills,
    get_skill_by_name,
    increment_skill_injection,
    increment_skill_success,
)

logger = get_logger("memory.skills")

# ─── Constants / Thresholds ──────────────────────────────────────────────────

DEDUP_SIMILARITY_THRESHOLD = 0.85
MATCH_SIMILARITY_THRESHOLD = 0.6
HIGH_CONFIDENCE_THRESHOLD = 0.8
MIN_INJECTIONS_FOR_CONFIDENCE = 5
MAX_STRATEGIES_PER_SKILL = 5
TOOL_INJECTION_THRESHOLD = 0.7
TOOL_INJECTION_MIN_COUNT = 5


# ─── Helper Functions ────────────────────────────────────────────────────────

def _injection_success_rate(skill: dict) -> float:
    """Return injection_success / injection_count, capped at 0.5 if < MIN_INJECTIONS_FOR_CONFIDENCE."""
    count = skill.get("injection_count", 0)
    success = skill.get("injection_success", 0)
    if count == 0:
        return 0.5
    rate = success / count
    if count < MIN_INJECTIONS_FOR_CONFIDENCE:
        return min(rate, 0.5)
    return rate


def _best_strategy(skill: dict) -> dict | None:
    """Return highest-ranked strategy from JSON strategies list.

    Proven strategies (>= 5 injections) ranked by success rate.
    Unproven strategies ranked newest-first (last in list = newest).
    """
    strategies_raw = skill.get("strategies", "[]")
    if isinstance(strategies_raw, str):
        try:
            strategies = json.loads(strategies_raw)
        except (json.JSONDecodeError, TypeError):
            return None
    else:
        strategies = strategies_raw

    if not strategies:
        return None

    proven = []
    unproven = []
    for s in strategies:
        inj_count = s.get("injection_count", 0)
        if inj_count >= 5:
            inj_success = s.get("injection_success", 0)
            rate = inj_success / max(inj_count, 1)
            proven.append((rate, s))
        else:
            unproven.append(s)

    if proven:
        proven.sort(key=lambda x: x[0], reverse=True)
        return proven[0][1]

    # Unproven: newest-first (last in list)
    if unproven:
        return unproven[-1]

    return None


def _prune_strategies(strategies: list[dict]) -> list[dict]:
    """Enforce MAX_STRATEGIES_PER_SKILL. Never drop unproven. Drop worst proven first."""
    if len(strategies) <= MAX_STRATEGIES_PER_SKILL:
        return strategies

    proven = []
    unproven = []
    for s in strategies:
        if s.get("injection_count", 0) >= 5:
            rate = s.get("injection_success", 0) / max(s.get("injection_count", 1), 1)
            proven.append((rate, s))
        else:
            unproven.append(s)

    # Sort proven by rate ascending (worst first)
    proven.sort(key=lambda x: x[0])

    # Drop worst proven strategies until we fit
    while len(proven) + len(unproven) > MAX_STRATEGIES_PER_SKILL and proven:
        proven.pop(0)

    result = [s for _, s in proven] + unproven
    return result[:MAX_STRATEGIES_PER_SKILL]


# ─── Vector Operations ───────────────────────────────────────────────────────

async def _embed_skill(name: str, description: str) -> None:
    """Embed skill into ChromaDB semantic collection."""
    try:
        from src.memory.vector_store import embed_and_store
        embed_text = f"{name}: {description}"
        await embed_and_store(
            text=embed_text,
            metadata={"type": "skill", "skill_name": name},
            collection="semantic",
            doc_id=f"skill:{name}",
        )
    except Exception as exc:
        logger.debug(f"Skill embedding failed (non-critical): {exc}")


async def _find_duplicate_skill(description: str) -> dict | None:
    """Query ChromaDB for existing skill with similarity >= DEDUP_SIMILARITY_THRESHOLD.

    ChromaDB returns L2 distance; convert: similarity = max(0, 1.0 - distance).
    If match found, return the full skill dict from DB.
    """
    try:
        from src.memory.vector_store import query as vquery
        results = await vquery(
            text=description,
            collection="semantic",
            top_k=3,
            where={"type": "skill"},
        )
        for r in results:
            distance = r.get("distance", 1.0)
            similarity = max(0.0, 1.0 - distance)
            if similarity >= DEDUP_SIMILARITY_THRESHOLD:
                skill_name = r.get("metadata", {}).get("skill_name", "")
                if skill_name:
                    skill = await get_skill_by_name(skill_name)
                    if skill:
                        return skill
    except Exception as exc:
        logger.debug(f"Duplicate skill check failed: {exc}")
    return None


async def _vector_search_skills(task_text: str, top_k: int = 5) -> list[dict]:
    """Find skills by vector similarity >= MATCH_SIMILARITY_THRESHOLD.

    Returns list of {"skill_name": ..., "similarity": ...}.
    """
    try:
        from src.memory.vector_store import query as vquery
        results = await vquery(
            text=task_text,
            collection="semantic",
            top_k=top_k * 2,
            where={"type": "skill"},
        )
        matches = []
        for r in results:
            distance = r.get("distance", 1.0)
            similarity = max(0.0, 1.0 - distance)
            if similarity >= MATCH_SIMILARITY_THRESHOLD:
                skill_name = r.get("metadata", {}).get("skill_name", "")
                if skill_name:
                    matches.append({"skill_name": skill_name, "similarity": similarity})
        return matches[:top_k]
    except Exception as exc:
        logger.debug(f"Vector skill search failed: {exc}")
        return []


# ─── Capture ─────────────────────────────────────────────────────────────────

async def add_skill(
    name: str,
    description: str,
    strategy_summary: str = "",
    tool_template: str | None = None,
    tools_used: list[str] | None = None,
    avg_iterations: float = 0,
    source_grade: str = "great",
    source_task_id: int = 0,
) -> str | None:
    """Create or merge a skill. Returns skill name used (may differ if merged)."""
    strategy = {
        "summary": strategy_summary or description,
        "tool_template": tool_template or "",
        "tools_used": tools_used or [],
        "avg_iterations": avg_iterations,
        "source_grade": source_grade,
        "source_task_id": source_task_id,
        "injection_count": 0,
        "injection_success": 0,
        "created_at": time.time(),
    }

    # Check for duplicate
    duplicate = await _find_duplicate_skill(description)
    if duplicate:
        # Merge strategy into existing skill
        dup_name = duplicate["name"]
        strategies_raw = duplicate.get("strategies", "[]")
        if isinstance(strategies_raw, str):
            try:
                strategies = json.loads(strategies_raw)
            except (json.JSONDecodeError, TypeError):
                strategies = []
        else:
            strategies = strategies_raw or []

        strategies.append(strategy)
        strategies = _prune_strategies(strategies)

        await upsert_skill(
            name=dup_name,
            description=duplicate["description"],
            skill_type=duplicate.get("skill_type", "auto"),
            strategies=strategies,
        )
        logger.info(f"Skill merged into existing: {dup_name} (from {name})")
        return dup_name

    # Create new skill
    await upsert_skill(
        name=name,
        description=description,
        skill_type="auto",
        strategies=[strategy],
    )
    await _embed_skill(name, description)
    logger.info(f"Skill created: {name}")
    return name


# ─── Injection ───────────────────────────────────────────────────────────────

async def find_relevant_skills(task_text: str, limit: int = 5) -> list[dict]:
    """Find skills matching task via vector search, ranked by similarity + success rate."""
    try:
        vector_matches = await _vector_search_skills(task_text, top_k=limit * 2)
        if not vector_matches:
            return []

        results = []
        seen = set()
        for vm in vector_matches:
            skill_name = vm["skill_name"]
            if skill_name in seen:
                continue
            seen.add(skill_name)

            skill = await get_skill_by_name(skill_name)
            if not skill:
                continue

            similarity = vm["similarity"]
            rate = _injection_success_rate(skill)
            match_score = similarity * 0.5 + rate * 0.5

            skill["_match_score"] = match_score
            skill["_similarity"] = similarity
            results.append(skill)

        results.sort(key=lambda s: s["_match_score"], reverse=True)
        return results[:limit]

    except Exception as exc:
        logger.warning("find_relevant_skills failed: %s", exc)
        return []


def select_injection_depth(
    skills: list[dict], context_budget: int = 4096
) -> tuple[str, list[dict]]:
    """Select injection depth based on confidence and budget.

    Returns ("verbose", [top_skill]) if top skill is highly trusted,
    else ("compact", skills[:max]) where max depends on budget.
    """
    if not skills:
        return ("compact", [])

    top = skills[0]
    rate = _injection_success_rate(top)
    count = top.get("injection_count", 0)

    if rate >= HIGH_CONFIDENCE_THRESHOLD and count >= MIN_INJECTIONS_FOR_CONFIDENCE:
        return ("verbose", [top])

    # Compact mode: limit by budget
    if context_budget < 2048:
        max_skills = 1
    elif context_budget < 4096:
        max_skills = 2
    else:
        max_skills = 3

    return ("compact", skills[:max_skills])


def format_skill_verbose(skill: dict) -> str:
    """Full markdown block with situation, strategy, steps, track record."""
    name = skill.get("name", "unknown")
    desc = skill.get("description", "")
    count = skill.get("injection_count", 0)
    success = skill.get("injection_success", 0)
    rate = (success / count * 100) if count > 0 else 0

    best = _best_strategy(skill)
    summary = best.get("summary", desc) if best else desc
    tools = best.get("tools_used", []) if best else []
    tool_template = best.get("tool_template", "") if best else ""

    lines = [
        f"## Skill: {name}",
        f"**Situation:** {desc}",
        f"**Strategy:** {summary}",
    ]
    if tool_template:
        lines.append(f"**Steps:** `{tool_template}`")
    if tools:
        lines.append(f"**Tools:** {', '.join(tools)}")
    lines.append(f"**Track record:** {rate:.0f}% success ({success}/{count} injections)")
    return "\n".join(lines)


def format_skill_compact(skill: dict) -> str:
    """Single line: - {name}: {summary} (tools: {tools}, {rate}% success)"""
    name = skill.get("name", "unknown")
    count = skill.get("injection_count", 0)
    success = skill.get("injection_success", 0)
    rate = (success / count * 100) if count > 0 else 0

    best = _best_strategy(skill)
    summary = best.get("summary", skill.get("description", "")) if best else skill.get("description", "")
    tools = best.get("tools_used", []) if best else []
    tools_str = ", ".join(tools) if tools else "none"

    return f"- {name}: {summary} (tools: {tools_str}, {rate:.0f}% success)"


def format_skills_for_prompt(skills: list[dict], context_budget: int = 4096) -> str:
    """Format skills for agent prompt injection using adaptive depth."""
    if not skills:
        return ""

    depth, selected = select_injection_depth(skills, context_budget)

    if not selected:
        return ""

    lines = ["## Relevant Skills from Library\n"]
    if depth == "verbose":
        for s in selected:
            lines.append(format_skill_verbose(s))
            lines.append("")
    else:
        for s in selected:
            lines.append(format_skill_compact(s))
        lines.append("")

    return "\n".join(lines)


def get_tools_to_inject(skills: list[dict]) -> list[str]:
    """Return tool names from strategies of high-confidence skills."""
    tools = set()
    for skill in skills:
        rate = _injection_success_rate(skill)
        count = skill.get("injection_count", 0)
        if rate >= TOOL_INJECTION_THRESHOLD and count >= TOOL_INJECTION_MIN_COUNT:
            best = _best_strategy(skill)
            if best and best.get("tools_used"):
                tools.update(best["tools_used"])
    return list(tools)


# ─── Tracking ────────────────────────────────────────────────────────────────

async def record_injection(skill_names: list[str]) -> None:
    """Increment injection_count on skill AND on best strategy."""
    for name in skill_names:
        try:
            await increment_skill_injection(name)
            # Also increment on best strategy
            skill = await get_skill_by_name(name)
            if skill:
                best = _best_strategy(skill)
                if best:
                    best["injection_count"] = best.get("injection_count", 0) + 1
                    # Save updated strategies back
                    strategies_raw = skill.get("strategies", "[]")
                    if isinstance(strategies_raw, str):
                        strategies = json.loads(strategies_raw)
                    else:
                        strategies = strategies_raw or []
                    await upsert_skill(
                        name=name,
                        description=skill["description"],
                        skill_type=skill.get("skill_type", "auto"),
                        strategies=strategies,
                    )
        except Exception as exc:
            logger.debug(f"record_injection failed for {name}: {exc}")


async def record_injection_success(skill_names: list[str]) -> None:
    """Increment injection_success on skill AND on best strategy."""
    for name in skill_names:
        try:
            await increment_skill_success(name)
            # Also increment on best strategy
            skill = await get_skill_by_name(name)
            if skill:
                best = _best_strategy(skill)
                if best:
                    best["injection_success"] = best.get("injection_success", 0) + 1
                    strategies_raw = skill.get("strategies", "[]")
                    if isinstance(strategies_raw, str):
                        strategies = json.loads(strategies_raw)
                    else:
                        strategies = strategies_raw or []
                    await upsert_skill(
                        name=name,
                        description=skill["description"],
                        skill_type=skill.get("skill_type", "auto"),
                        strategies=strategies,
                    )
        except Exception as exc:
            logger.debug(f"record_injection_success failed for {name}: {exc}")


# ─── Admin ───────────────────────────────────────────────────────────────────

async def list_skills() -> list[dict]:
    """List all skills. Delegates to get_all_skills() from db.py."""
    return await get_all_skills()


# ─── Backward Compatibility ──────────────────────────────────────────────────

async def record_skill_outcome(name: str, success: bool) -> None:
    """DEPRECATED: backward-compatible stub. Use record_injection/record_injection_success."""
    if success:
        await record_injection_success([name])
    else:
        await record_injection([name])


async def _ensure_table(db=None) -> None:
    """DEPRECATED: no-op stub for backward compatibility. Schema is managed by init_db()."""
    pass
