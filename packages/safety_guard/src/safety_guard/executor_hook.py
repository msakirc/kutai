"""Pre-action decision: Allow | WaitForFounder | Block.

Called by the workflow engine before any executor action runs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Union

from safety_guard.tags import Reversibility, resolve
from safety_guard.collision import (
    detect_force_push,
    detect_shared_history_rewrite,
    detect_shell_outside_workspace,
    detect_destructive_shared_db,
    detect_blocklist,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class WaitForFounder:
    reason: str


@dataclass(frozen=True)
class Block:
    reason: str


Decision = Union[Allow, WaitForFounder, Block]


def _allowlist_matches(cmd: str, patterns: list[str]) -> bool:
    return any(re.search(p, cmd) for p in patterns)


def pre_action(
    step: dict,
    action: dict,
    *,
    workspace_root: str,
    current_branch: str,
    founder_recently_active: bool = True,
    mission_allowlist: Optional[list[str]] = None,
    runtime_reversibility: Optional[Reversibility] = None,
) -> Decision:
    """Single decision point before executing a workflow step's action."""
    cmd = action.get("command", "") or ""
    mission_allowlist = mission_allowlist or []

    # 1. Hardcoded blocklist — always wins.
    if detect_blocklist(cmd):
        return Block(reason="blocklist")

    # 2. Collision guards — bypassable only by per-mission allowlist.
    for check_value, name in (
        (detect_force_push(cmd), "force_push"),
        (detect_shared_history_rewrite(cmd, current_branch), "shared_history_rewrite"),
        (detect_shell_outside_workspace(cmd, workspace_root), "shell_outside_workspace"),
        (detect_destructive_shared_db(cmd), "destructive_shared_db"),
    ):
        if check_value and not _allowlist_matches(cmd, mission_allowlist):
            return Block(reason=f"collision:{name}")

    # 3. Reversibility-driven flow.
    tag = resolve(step, runtime_reversibility)
    if tag is Reversibility.NONE:
        if step.get("locked", False) or not founder_recently_active:
            return WaitForFounder(reason="non_reversible_step")

    return Allow()
