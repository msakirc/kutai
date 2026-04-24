"""In-flight registry — peer module shared by Beckman, Dispatcher, Orchestrator.

Holds the authoritative in-flight call list and pushes it to nerd_herd on every
mutation. Exists as its own module (rather than living inside llm_dispatcher)
so Beckman can call `reserve_task()` at admission time without importing
dispatcher logic — preserving the "no upstream→downstream import" invariant.

Three lifecycle states, all represented in a single dict keyed by ``task_id``
(plus a secondary dict for standalone calls with no task context):

1. **Admitted** — Beckman claimed the DB row and chose a model. Beckman calls
   ``reserve_task(task_id, pick)`` immediately after ``_claim_task`` so the
   very next admission tick sees the lane occupied.
2. **Calling** — Dispatcher ran ``begin_call``. Same ``task-{id}`` key, so the
   reserved entry is UPSERTED with the actual model (may differ from the
   admission-time pick on retry / Hoca re-select).
3. **Between calls (ReAct iteration gap)** — ``end_call`` preserves
   ``task-*``-prefixed entries so the slot stays held while the agent runs
   tools between LLM calls.

The slot is cleared only by ``release_task(task_id)``, which the orchestrator
calls in ``_dispatch``'s finally block when the task fully terminates.

Why a separate module: dispatcher is the sole producer of in-flight state,
but Beckman needs to write state (1) at admission time. Importing
llm_dispatcher from Beckman would create an upstream→downstream import
(Beckman runs BEFORE dispatcher per pump cycle). The peer module resolves
this: neither side is "upstream of the other", both import from a shared
data store. The registry remains the architectural single-producer (i.e.
only this module calls ``nerd_herd.push_in_flight``).
"""
from __future__ import annotations

import time
import uuid

from src.infra.logging_config import get_logger

logger = get_logger("core.in_flight")


# ─── State ────────────────────────────────────────────────────────────────
_task_slots: dict[int, "_InFlightEntry"] = {}
_call_entries: dict[str, "_InFlightEntry"] = {}


class _InFlightEntry:
    __slots__ = ("call_id", "task_id", "category", "model", "provider", "is_local", "started_at")

    def __init__(self, call_id, task_id, category, model, provider, is_local, started_at):
        self.call_id = call_id
        self.task_id = task_id
        self.category = category
        self.model = model
        self.provider = provider
        self.is_local = is_local
        self.started_at = started_at


# ─── Read-only accessor ───────────────────────────────────────────────────
def in_flight_snapshot() -> list:
    """Return a copy of the current in-flight list. For telemetry, not admission.

    Admission reads via ``nerd_herd.snapshot().in_flight_calls`` — same list,
    routed through the observability bus for layering consistency.
    """
    return list(_task_slots.values()) + list(_call_entries.values())


# ─── Mutations ────────────────────────────────────────────────────────────
async def _push() -> None:
    """Push full merged list to nerd_herd. Best-effort; swallows errors."""
    try:
        import nerd_herd
        from nerd_herd.types import InFlightCall
        payload = [
            InFlightCall(
                call_id=e.call_id,
                task_id=e.task_id,
                category=e.category,
                model=e.model,
                provider=e.provider,
                is_local=e.is_local,
                started_at=e.started_at,
            )
            for e in list(_task_slots.values()) + list(_call_entries.values())
        ]
        await nerd_herd.push_in_flight(payload)
    except Exception as exc:
        logger.debug("in_flight push failed: %s", exc)


async def reserve_task(task_id: int, pick) -> None:
    """Seed a per-task slot at admission time with the Beckman-chosen pick.

    Called by Beckman after ``_claim_task`` succeeds. Fills the gap between
    admission and the dispatcher's first ``begin_call`` (which can be many
    seconds when agent pre-work does RAG / chain-context / file-tree scans).
    Dispatcher's ``begin_call`` later UPSERTs the same slot with the actual
    call-time model (retry / re-select may change it).

    ``pick.model`` must have ``name``, ``provider``, ``is_local`` attributes.
    """
    model = pick.model
    _task_slots[task_id] = _InFlightEntry(
        call_id=f"task-{task_id}",
        task_id=task_id,
        category="main_work",  # provisional; upserted by begin_call
        model=getattr(model, "name", ""),
        provider=getattr(model, "provider", ""),
        is_local=bool(getattr(model, "is_local", False)),
        started_at=time.time(),
    )
    await _push()


async def begin_call(
    category: str,
    model_name: str,
    provider: str,
    is_local: bool,
    task_id: int | None,
) -> str:
    """Register an in-flight call. Returns call_id for end_call pairing.

    Task-associated calls UPSERT the per-task slot keyed ``task-{task_id}``.
    If ``reserve_task`` already seeded the slot, the model/provider fields
    are overwritten with current-call values — correct for mid-task model
    changes (retry / Hoca re-select).

    Standalone calls (task_id is None) get a fresh uuid entry. They live
    only for the duration of the single call and are removed by ``end_call``.
    """
    if task_id is not None:
        call_id = f"task-{task_id}"
        _task_slots[task_id] = _InFlightEntry(
            call_id=call_id,
            task_id=task_id,
            category=category,
            model=model_name,
            provider=provider,
            is_local=is_local,
            started_at=time.time(),
        )
    else:
        call_id = str(uuid.uuid4())
        _call_entries[call_id] = _InFlightEntry(
            call_id=call_id,
            task_id=None,
            category=category,
            model=model_name,
            provider=provider,
            is_local=is_local,
            started_at=time.time(),
        )
    await _push()
    return call_id


async def end_call(call_id: str) -> None:
    """Remove a standalone entry. Task-scoped slots are preserved until
    ``release_task`` fires — the slot survives ReAct iteration gaps where
    the agent runs tools between LLM calls.
    """
    if call_id.startswith("task-"):
        return
    if _call_entries.pop(call_id, None) is not None:
        await _push()


async def release_task(task_id: int) -> None:
    """Clear the per-task slot when the task fully terminates.

    Called by orchestrator's ``_dispatch`` finally block. Safe to call
    multiple times — no-op when the slot is already absent (e.g. mechanical
    tasks that never reached dispatcher).
    """
    if _task_slots.pop(task_id, None) is not None:
        await _push()
