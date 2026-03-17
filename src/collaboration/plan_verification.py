# collaboration/plan_verification.py
"""
Phase 13.2 — Plan Verification.

After planner creates subtasks, verify:
1. Agent type assignments are sensible
2. Dependency graph is acyclic
3. No duplicate subtasks
4. Estimated cost fits within budget
"""
from __future__ import annotations

import re
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("collaboration.plan_verification")

# ── Heuristic: keywords → expected agent_type mapping ────────────────────────

_TASK_AGENT_HINTS: dict[str, set[str]] = {
    "coder": {
        r"write code", r"implement", r"build.*api", r"create.*endpoint",
        r"fix.*bug", r"write.*test", r"unit test", r"integration test",
        r"refactor", r"write.*function", r"write.*class", r"\bcode\b",
        r"add.*endpoint", r"create.*service", r"add.*feature",
    },
    "pipeline": {
        r"complex", r"multi-file", r"full.*implementation",
        r"architect.*and.*implement",
    },
    "researcher": {
        r"research", r"\bfind\b", r"search", r"\bcompare\b", r"investigate",
        r"look up", r"analyze.*option", r"evaluate", r"study",
    },
    "writer": {
        r"document", r"readme", r"write.*doc", r"create.*doc",
        r"documentation", r"write.*guide", r"write.*report",
    },
    "reviewer": {
        r"review", r"audit", r"check.*quality", r"inspect",
    },
}

# Rough estimated cost per task by tier (USD)
_TIER_COST_ESTIMATE: dict[str, float] = {
    "cheap": 0.01,
    "medium": 0.05,
    "code": 0.05,
    "expensive": 0.20,
    "auto": 0.03,
}


# ── Public API ───────────────────────────────────────────────────────────────

def verify_plan(
    subtasks: list[dict],
    goal_budget: float = 10.0,
) -> list[str]:
    """Verify a plan's subtasks, returning a list of issue strings.

    Empty list = plan is OK.
    """
    issues: list[str] = []

    if not subtasks:
        return issues

    # 1. Cycle detection in dependency graph
    issues.extend(_check_cycles(subtasks))

    # 2. Agent type sanity check
    for i, st in enumerate(subtasks):
        text = (
            f"{st.get('title', '')} {st.get('description', '')}"
        ).lower()
        assigned = st.get("agent_type", "executor")
        suggested = _suggest_agent_types(text)
        if suggested and assigned not in suggested and assigned != "executor":
            issues.append(
                f"Subtask {i} '{st.get('title', '?')[:40]}': agent_type "
                f"'{assigned}' may be wrong — content suggests {sorted(suggested)}"
            )

    # 3. Duplicate titles
    titles = [st.get("title", "").strip().lower() for st in subtasks]
    seen: set[str] = set()
    for i, t in enumerate(titles):
        if t in seen and t:
            issues.append(f"Subtask {i} duplicate title: '{t[:40]}'")
        seen.add(t)

    # 4. Budget check
    total_cost = sum(
        _TIER_COST_ESTIMATE.get(st.get("tier", "auto"), 0.03)
        for st in subtasks
    )
    if total_cost > goal_budget:
        issues.append(
            f"Estimated plan cost ${total_cost:.2f} exceeds budget "
            f"${goal_budget:.2f}"
        )

    return issues


# ── Internal helpers ─────────────────────────────────────────────────────────

def _check_cycles(subtasks: list[dict]) -> list[str]:
    """Detect cycles in the dependency graph using Kahn's algorithm."""
    n = len(subtasks)
    if n == 0:
        return []

    # Build adjacency list: dep → dependents
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    in_degree = [0] * n

    for i, st in enumerate(subtasks):
        dep = st.get("depends_on_step")
        if dep is None:
            continue
        deps = [dep] if isinstance(dep, int) else dep
        for d in deps:
            if isinstance(d, int) and 0 <= d < n:
                adj[d].append(i)
                in_degree[i] += 1

    # Kahn's algorithm
    queue = [i for i in range(n) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited < n:
        return [
            f"Dependency cycle detected in plan "
            f"({n - visited} task(s) in cycle)"
        ]
    return []


def _suggest_agent_types(text: str) -> set[str] | None:
    """Suggest acceptable agent types from task text.

    Returns None when no strong signal is detected.
    """
    suggestions: set[str] = set()
    for agent, patterns in _TASK_AGENT_HINTS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                suggestions.add(agent)
                break
    return suggestions or None
