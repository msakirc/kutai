"""Founder Actions — Z6 T1B real-world bridge queue.

A ``founder_action`` is a real-world handoff: the agent has prepared
everything it can autonomously, but the next step requires human
delegation (vendor enrollment, credential paste, cost ack, legal counsel,
KYC, free-text generic). The action card is rendered to the mission's
Telegram thread; the founder responds via inline buttons or
``/action_done``. Mission lifecycle (T1E) parks while any actions for the
mission are pending/in_progress.

This module is the dedicated repo for the ``founder_actions`` table. All
async, aiosqlite-backed, isolation-friendly. Status transitions are
guarded by ``_VALID_TRANSITIONS``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from src.infra.logging_config import get_logger

logger = get_logger("founder_actions")

# Canonical kind set. The repo accepts any string but unknown values warn.
FounderActionKind = Literal[
    "credential_paste",
    "vendor_enroll",
    "cost_ack",
    "legal_counsel",
    "kyc",
    "generic",
]
KNOWN_KINDS: frozenset[str] = frozenset(
    {"credential_paste", "vendor_enroll", "cost_ack",
     "legal_counsel", "kyc", "generic"}
)

# Status lifecycle. Terminal: done / cancelled. ``blocked`` is recoverable
# (founder said "I can't do this right now" — surfaces in /actions for retry).
_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"in_progress", "done", "blocked", "cancelled"}),
    "in_progress": frozenset({"done", "blocked", "cancelled", "pending"}),
    "blocked": frozenset({"pending", "in_progress", "cancelled"}),
    # Terminal — no outgoing edges. resolve() always passes 'done'.
    "done": frozenset(),
    "cancelled": frozenset(),
}


@dataclass
class FounderAction:
    """In-memory view of one founder_actions row."""
    id: int
    mission_id: int
    blocking_task_id: Optional[int]
    blocking_step_id: Optional[str]
    kind: str
    title: str
    why: str
    instructions: list[str]
    expected_output_kind: Optional[str]
    expected_output_schema: Optional[dict]
    cost_estimate_usd: Optional[float]
    reversibility: Optional[str]
    status: str
    response_payload: Optional[dict]
    created_at: str
    updated_at: str
    resolved_at: Optional[str]
    urgent: bool = False

    @classmethod
    def from_row(cls, row: Any) -> "FounderAction":
        # aiosqlite.Row supports dict-like access if row_factory is Row.
        # Fall back to tuple-positional only when needed (rare).
        try:
            d = dict(row)
        except (TypeError, ValueError):
            # Tuple path — order must match _SELECT_COLS.
            keys = [
                "id", "mission_id", "blocking_task_id", "blocking_step_id",
                "kind", "title", "why", "instructions_json",
                "expected_output_kind", "expected_output_schema_json",
                "cost_estimate_usd", "reversibility", "status",
                "response_payload_json", "created_at", "updated_at",
                "resolved_at",
            ]
            d = dict(zip(keys, row))
        return cls(
            id=int(d["id"]),
            mission_id=int(d["mission_id"]),
            blocking_task_id=(
                int(d["blocking_task_id"])
                if d.get("blocking_task_id") is not None else None
            ),
            blocking_step_id=d.get("blocking_step_id"),
            kind=d["kind"],
            title=d["title"],
            why=d["why"],
            instructions=_safe_json_list(d.get("instructions_json")),
            expected_output_kind=d.get("expected_output_kind"),
            expected_output_schema=_safe_json_dict(
                d.get("expected_output_schema_json")
            ),
            cost_estimate_usd=(
                float(d["cost_estimate_usd"])
                if d.get("cost_estimate_usd") is not None else None
            ),
            reversibility=d.get("reversibility"),
            status=d["status"],
            response_payload=_safe_json_dict(d.get("response_payload_json")),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            resolved_at=d.get("resolved_at"),
            urgent=bool(d.get("urgent") or 0),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "blocking_task_id": self.blocking_task_id,
            "blocking_step_id": self.blocking_step_id,
            "kind": self.kind,
            "title": self.title,
            "why": self.why,
            "instructions": list(self.instructions),
            "expected_output_kind": self.expected_output_kind,
            "expected_output_schema": (
                dict(self.expected_output_schema)
                if self.expected_output_schema else None
            ),
            "cost_estimate_usd": self.cost_estimate_usd,
            "reversibility": self.reversibility,
            "status": self.status,
            "response_payload": (
                dict(self.response_payload)
                if self.response_payload else None
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "urgent": bool(self.urgent),
        }


def _safe_json_list(raw: Any) -> list:
    if not raw:
        return []
    if isinstance(raw, list):
        return list(raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _safe_json_dict(raw: Any) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create(
    mission_id: int,
    kind: str,
    title: str,
    why: str,
    instructions: list[str],
    *,
    blocking_task_id: Optional[int] = None,
    blocking_step_id: Optional[str] = None,
    expected_output_kind: Optional[str] = None,
    expected_output_schema: Optional[dict] = None,
    cost_estimate_usd: Optional[float] = None,
    reversibility: Optional[str] = None,
    urgent: bool = False,
    notify_telegram: bool = True,
) -> FounderAction:
    """Create a new pending founder_action. Returns the persisted row.

    When ``notify_telegram=True`` and the mission has a ``telegram_thread_id``
    column, an inline card is posted to the mission thread (best-effort —
    failure is logged but does not raise).
    """
    if kind not in KNOWN_KINDS:
        logger.warning(
            "founder_action: unknown kind %r — accepted but verify schema", kind,
        )

    from src.infra.db import get_db
    now = _utc_now()
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO founder_actions "
        "(mission_id, blocking_task_id, blocking_step_id, kind, title, why, "
        " instructions_json, expected_output_kind, expected_output_schema_json,"
        " cost_estimate_usd, reversibility, urgent, status, "
        " created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
        (
            mission_id,
            blocking_task_id,
            blocking_step_id,
            kind,
            title,
            why,
            json.dumps(list(instructions or [])),
            expected_output_kind,
            json.dumps(expected_output_schema) if expected_output_schema else None,
            cost_estimate_usd,
            reversibility,
            1 if urgent else 0,
            now,
            now,
        ),
    )
    await db.commit()
    new_id = int(cursor.lastrowid or 0)
    logger.info(
        "founder_action created",
        action_id=new_id, mission_id=mission_id, kind=kind,
    )
    action = await get(new_id)
    assert action is not None
    # T1E: blocking_if_needed flips mission lifecycle.
    try:
        await block_mission_if_needed(mission_id)
    except Exception as e:  # noqa: BLE001
        logger.debug("block_mission_if_needed deferred", error=str(e))
    # Best-effort Telegram surface — render module imports avoided here
    # to keep this module dependency-free for unit tests.
    if notify_telegram:
        try:
            await _notify_telegram(action)
        except Exception as e:  # noqa: BLE001
            logger.debug("founder_action telegram notify failed", error=str(e))
    return action


_notifier = None  # set by src.app.telegram_bot at startup


def register_notifier(fn) -> None:
    """Register an async callable that surfaces a new founder_action to the
    user. fn signature: ``async fn(action: FounderAction) -> None``.

    Telegram bot startup calls this with the inline-card poster. Tests skip
    notification by passing ``notify_telegram=False`` on create().
    """
    global _notifier
    _notifier = fn


async def _notify_telegram(action: FounderAction) -> None:
    """Best-effort Telegram surface — invokes the registered notifier."""
    if _notifier is None:
        return
    try:
        await _notifier(action)
    except Exception as e:  # noqa: BLE001
        logger.debug("founder_action notifier failed", error=str(e))


async def get(action_id: int) -> Optional[FounderAction]:
    """Fetch one row by id or None."""
    from src.infra.db import get_db
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM founder_actions WHERE id = ?", (action_id,),
    )
    row = await cursor.fetchone()
    return FounderAction.from_row(row) if row else None


async def list_by_mission(
    mission_id: int,
    status_filter: Optional[list[str] | str] = None,
) -> list[FounderAction]:
    """Return founder_actions for a mission, ordered by created_at desc.

    ``status_filter`` may be a single status or list. None = all.
    """
    from src.infra.db import get_db
    db = await get_db()
    if status_filter:
        statuses = (
            [status_filter] if isinstance(status_filter, str)
            else list(status_filter)
        )
        placeholders = ",".join(["?"] * len(statuses))
        cursor = await db.execute(
            f"SELECT * FROM founder_actions "
            f"WHERE mission_id = ? AND status IN ({placeholders}) "
            f"ORDER BY created_at DESC",
            (mission_id, *statuses),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM founder_actions WHERE mission_id = ? "
            "ORDER BY created_at DESC",
            (mission_id,),
        )
    rows = await cursor.fetchall()
    return [FounderAction.from_row(r) for r in rows]


async def list_pending() -> list[FounderAction]:
    """All actions where status IN ('pending', 'in_progress'), across missions."""
    from src.infra.db import get_db
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM founder_actions "
        "WHERE status IN ('pending', 'in_progress') "
        "ORDER BY created_at ASC"
    )
    rows = await cursor.fetchall()
    return [FounderAction.from_row(r) for r in rows]


async def update_status(
    action_id: int,
    new_status: str,
    response_payload: Optional[dict] = None,
) -> FounderAction:
    """Transition an action's status. Raises ValueError on invalid transition.

    ``done`` and ``cancelled`` set ``resolved_at`` to now.
    """
    current = await get(action_id)
    if current is None:
        raise ValueError(f"founder_action #{action_id} not found")
    allowed = _VALID_TRANSITIONS.get(current.status, frozenset())
    if new_status not in allowed:
        raise ValueError(
            f"invalid transition {current.status!r} -> {new_status!r} "
            f"(allowed: {sorted(allowed)})"
        )

    from src.infra.db import get_db
    now = _utc_now()
    resolved_at = now if new_status in ("done", "cancelled") else None
    payload_json = (
        json.dumps(response_payload) if response_payload is not None else None
    )
    db = await get_db()
    # Preserve any prior response_payload when caller doesn't supply one
    # (e.g. a pending → in_progress flip that doesn't carry new data).
    if response_payload is None:
        await db.execute(
            "UPDATE founder_actions SET status = ?, updated_at = ?, "
            " resolved_at = COALESCE(?, resolved_at) WHERE id = ?",
            (new_status, now, resolved_at, action_id),
        )
    else:
        await db.execute(
            "UPDATE founder_actions SET status = ?, updated_at = ?, "
            " resolved_at = COALESCE(?, resolved_at), "
            " response_payload_json = ? WHERE id = ?",
            (new_status, now, resolved_at, payload_json, action_id),
        )
    await db.commit()
    logger.info(
        "founder_action status",
        action_id=action_id, old=current.status, new=new_status,
    )
    refreshed = await get(action_id)
    assert refreshed is not None
    # T1E hooks: mission may transition on every status flip.
    try:
        if new_status in ("done", "cancelled"):
            await unblock_mission_if_clear(refreshed.mission_id)
        else:
            await block_mission_if_needed(refreshed.mission_id)
    except Exception as e:  # noqa: BLE001
        logger.debug("mission lifecycle hook failed", error=str(e))
    return refreshed


async def resolve(
    action_id: int,
    response_payload: Optional[dict] = None,
) -> FounderAction:
    """Convenience: transition to ``done`` with optional payload."""
    return await update_status(action_id, "done", response_payload)


# ─── T1E: mission lifecycle coordination ───────────────────────────────────
# Schema-aware: if missions.lifecycle_state (Z0) exists at runtime we use
# that; otherwise we set missions.status to 'blocked_on_founder_action'.
# Cached after first probe — the check is one PRAGMA call.

_BLOCKED_LITERAL = "blocked_on_founder_action"
_lifecycle_column_cache: Optional[str] = None


async def _missions_lifecycle_column() -> str:
    """Return 'lifecycle_state' if the column exists, else 'status'."""
    global _lifecycle_column_cache
    if _lifecycle_column_cache is not None:
        return _lifecycle_column_cache
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute("PRAGMA table_info(missions)")
    cols = [row[1] for row in await cur.fetchall()]
    _lifecycle_column_cache = (
        "lifecycle_state" if "lifecycle_state" in cols else "status"
    )
    return _lifecycle_column_cache


def _reset_lifecycle_cache() -> None:
    """Test hook: force re-probe on next call. Used after fixture switches DBs."""
    global _lifecycle_column_cache
    _lifecycle_column_cache = None


async def block_mission_if_needed(mission_id: int) -> bool:
    """Flip mission to ``blocked_on_founder_action`` if any actions are
    pending/in_progress AND the mission is currently active.

    Returns True if the flip happened.
    """
    from src.infra.db import get_db, get_mission
    mission = await get_mission(mission_id)
    if not mission:
        return False
    col = await _missions_lifecycle_column()
    current = mission.get(col) or mission.get("status")
    if current != "active":
        return False
    db = await get_db()
    cur = await db.execute(
        "SELECT COUNT(*) FROM founder_actions WHERE mission_id = ? "
        "AND status IN ('pending', 'in_progress')",
        (mission_id,),
    )
    pending = int((await cur.fetchone())[0])
    if pending == 0:
        return False
    await db.execute(
        f"UPDATE missions SET {col} = ? WHERE id = ?",
        (_BLOCKED_LITERAL, mission_id),
    )
    await db.commit()
    logger.info(
        "mission blocked on founder_actions",
        mission_id=mission_id, pending=pending, column=col,
    )
    return True


async def unblock_mission_if_clear(mission_id: int) -> bool:
    """If mission is in ``blocked_on_founder_action`` AND no
    pending/in_progress actions remain, flip it back to ``active``.

    Returns True if the flip happened.
    """
    from src.infra.db import get_db, get_mission
    mission = await get_mission(mission_id)
    if not mission:
        return False
    col = await _missions_lifecycle_column()
    current = mission.get(col) or mission.get("status")
    if current != _BLOCKED_LITERAL:
        return False
    db = await get_db()
    cur = await db.execute(
        "SELECT COUNT(*) FROM founder_actions WHERE mission_id = ? "
        "AND status IN ('pending', 'in_progress')",
        (mission_id,),
    )
    pending = int((await cur.fetchone())[0])
    if pending > 0:
        return False
    await db.execute(
        f"UPDATE missions SET {col} = ? WHERE id = ?",
        ("active", mission_id),
    )
    # Also flip any tasks that beckman parked in
    # 'blocked_on_founder_action' for this mission back to pending so
    # the next pump tick re-evaluates them. Tasks that didn't go
    # through Z6 admission won't have that status, so this is a
    # narrow, idempotent UPDATE.
    await db.execute(
        "UPDATE tasks SET status = 'pending' "
        "WHERE mission_id = ? AND status = 'blocked_on_founder_action'",
        (mission_id,),
    )
    await db.commit()
    logger.info(
        "mission unblocked — no pending actions",
        mission_id=mission_id, column=col,
    )
    return True


async def sweep_unblock_all() -> int:
    """Orchestrator pump helper: scan blocked missions, unblock any that are
    clear. Returns the count of unblocks performed.
    """
    from src.infra.db import get_db
    col = await _missions_lifecycle_column()
    db = await get_db()
    cur = await db.execute(
        f"SELECT id FROM missions WHERE {col} = ?",
        (_BLOCKED_LITERAL,),
    )
    rows = await cur.fetchall()
    n = 0
    for row in rows:
        if await unblock_mission_if_clear(int(row[0])):
            n += 1
    return n


__all__ = [
    "FounderAction",
    "FounderActionKind",
    "KNOWN_KINDS",
    "create",
    "get",
    "list_by_mission",
    "list_pending",
    "update_status",
    "resolve",
    "block_mission_if_needed",
    "unblock_mission_if_clear",
    "sweep_unblock_all",
    "register_notifier",
]
