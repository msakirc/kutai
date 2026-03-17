"""Artifact store for workflow step inputs/outputs.

Artifacts are stored on the goal's blackboard under the "artifacts" key.
In-memory cache for fast access. Supports v2 context_strategy for
tiered artifact loading (primary > reference > full_only_if_needed).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Context budget tiers ────────────────────────────────────────────────────

CONTEXT_BUDGETS: dict[str, int] = {
    "primary": 8000,
    "reference": 3000,
    "full_only_if_needed": 1500,
    "default": 6000,
}

_TIER_ORDER = ["primary", "reference", "full_only_if_needed"]


# ── ArtifactStore ───────────────────────────────────────────────────────────

class ArtifactStore:
    """In-memory artifact cache with optional blackboard persistence."""

    def __init__(self, use_db: bool = True) -> None:
        self._cache: dict[str, dict[str, str]] = {}  # goal_id_str -> {name: value}
        self._use_db = use_db

    def _goal_key(self, goal_id: int | str) -> str:
        return str(goal_id)

    async def store(self, goal_id: int | str, name: str, value: str) -> None:
        """Store an artifact — DB first, then cache on success."""
        key = self._goal_key(goal_id)
        if self._use_db:
            from src.collaboration.blackboard import update_blackboard_entry
            await update_blackboard_entry(goal_id, "artifacts", name, value)
        # Cache only after successful DB write (or if use_db=False)
        if key not in self._cache:
            self._cache[key] = {}
        self._cache[key][name] = value

    async def retrieve(self, goal_id: int | str, name: str) -> Optional[str]:
        """Retrieve an artifact, cache-first with DB fallback."""
        key = self._goal_key(goal_id)

        # Cache hit
        if key in self._cache and name in self._cache[key]:
            return self._cache[key][name]

        # DB fallback
        if self._use_db:
            try:
                from src.collaboration.blackboard import read_blackboard
                artifacts = await read_blackboard(goal_id, "artifacts")
                if isinstance(artifacts, dict) and name in artifacts:
                    # Populate cache
                    if key not in self._cache:
                        self._cache[key] = {}
                    self._cache[key][name] = artifacts[name]
                    return artifacts[name]
            except Exception as exc:
                logger.debug(f"Artifact DB read failed: {exc}")

        return None

    async def has(self, goal_id: int | str, name: str) -> bool:
        """Check if an artifact exists."""
        return (await self.retrieve(goal_id, name)) is not None

    async def collect(
        self, goal_id: int | str, names: list[str]
    ) -> dict[str, Optional[str]]:
        """Batch retrieve multiple artifacts. Missing ones map to None."""
        result: dict[str, Optional[str]] = {}
        for n in names:
            result[n] = await self.retrieve(goal_id, n)
        return result

    async def list_artifacts(self, goal_id: int | str) -> list[str]:
        """List cached artifact names for a goal."""
        key = self._goal_key(goal_id)
        if key in self._cache:
            return list(self._cache[key].keys())
        return []

    async def flush_cache(self) -> None:
        """Best-effort flush of all cached artifacts to DB.

        Used during graceful shutdown.  Logs failures but does not raise.
        """
        if not self._use_db:
            return
        from src.collaboration.blackboard import update_blackboard_entry

        for goal_key, artifacts in self._cache.items():
            goal_id = int(goal_key)
            for name, value in artifacts.items():
                try:
                    await update_blackboard_entry(goal_id, "artifacts", name, value)
                except Exception as exc:
                    logger.warning(
                        "flush_cache: failed to persist artifact %s for goal %s: %s",
                        name, goal_key, exc,
                    )

    async def warm_cache(self, goal_id: int | str) -> None:
        """Load all artifacts for *goal_id* from the blackboard into cache.

        Used during workflow resume to restore in-memory state.
        """
        if not self._use_db:
            return
        from src.collaboration.blackboard import read_blackboard

        artifacts = await read_blackboard(goal_id, "artifacts")
        if isinstance(artifacts, dict) and artifacts:
            key = self._goal_key(goal_id)
            if key not in self._cache:
                self._cache[key] = {}
            self._cache[key].update(artifacts)


# ── Prompt formatting ───────────────────────────────────────────────────────

def _truncate(content: str, budget: int) -> str:
    """Truncate content to fit within a character budget."""
    if len(content) <= budget:
        return content
    return content[:budget - 3] + "..."


def format_artifacts_for_prompt(
    artifacts: dict[str, str],
    context_strategy: Optional[dict[str, list[str]]] = None,
    max_total: int = 20000,
) -> str:
    """Format artifacts for inclusion in a prompt.

    If context_strategy is provided, artifacts are loaded with tiered budgets:
      - primary: 8000 chars
      - reference: 3000 chars
      - full_only_if_needed: 1500 chars
      - uncategorized: 6000 chars (default)

    If no strategy, all artifacts get the default budget.
    Final output is truncated to max_total.
    """
    if not artifacts:
        return ""

    sections: list[str] = []

    if context_strategy:
        # Track which artifacts are assigned to a tier
        assigned: set[str] = set()
        for tier in _TIER_ORDER:
            names = context_strategy.get(tier, [])
            budget = CONTEXT_BUDGETS[tier]
            for name in names:
                if name in artifacts:
                    assigned.add(name)
                    content = _truncate(artifacts[name], budget)
                    sections.append(f"### {name}\n\n{content}")

        # Uncategorized artifacts get default budget
        default_budget = CONTEXT_BUDGETS["default"]
        for name, value in artifacts.items():
            if name not in assigned:
                content = _truncate(value, default_budget)
                sections.append(f"### {name}\n\n{content}")
    else:
        default_budget = CONTEXT_BUDGETS["default"]
        for name, value in artifacts.items():
            content = _truncate(value, default_budget)
            sections.append(f"### {name}\n\n{content}")

    result = "\n\n---\n\n".join(sections)

    if len(result) > max_total:
        result = result[:max_total]

    return result


# ── Phase summary helpers ────────────────────────────────────────────────────

async def get_phase_summaries(
    store: ArtifactStore,
    goal_id: int | str,
    up_to_phase: str,
) -> dict[str, str]:
    """Collect all ``phase_{N}_summary`` artifacts for phases before *up_to_phase*.

    Parameters
    ----------
    store:
        The :class:`ArtifactStore` to query.
    goal_id:
        Workflow goal identifier.
    up_to_phase:
        Phase identifier like ``"phase_3"``.  Summaries for all phases with
        a number strictly less than the parsed value are returned.

    Returns
    -------
    dict[str, str]
        Mapping of artifact name (e.g. ``"phase_0_summary"``) to its content.
        Empty dict when no summaries exist or the phase string cannot be parsed.
    """
    try:
        current_num = int(up_to_phase.split("_", 1)[1])
    except (IndexError, ValueError):
        return {}

    summaries: dict[str, str] = {}
    # Phases range from -1 up to current_num - 1
    for n in range(-1, current_num):
        artifact_name = f"phase_{n}_summary"
        content = await store.retrieve(goal_id, artifact_name)
        if content is not None:
            summaries[artifact_name] = content

    return summaries
