"""Intake-todo generator — Tier 1 of Z1 (B1).

Replaces phase-0's interrogation-by-many-questions pattern with a single
generated todo list the founder confirms before charter generation begins.

Behaviour:
  1. Read the optional analyst draft (step ``0.0a.draft``) from
     ``mission_{mission_id}/.intake/intake_todo_draft.json`` — a list of
     ``{n, category, question}`` items where the analyst has reworded each
     canonical question for THIS product.
  2. Merge the draft onto the fixed 14-slot canonical skeleton: each slot
     uses the analyst wording only when its declared category matches the
     canonical category for that slot, else the canonical question. This
     guarantees coverage + ordering regardless of model quality — a weak
     model can specialise wording but can never drop a dimension, reorder,
     or hallucinate extra slots.
  3. Write ``mission_{mission_id}/.intake/intake_todo.md`` (always the 14
     canonical items, optionally specialised).
  4. Return ``status="needs_clarification"`` with ``keyboard_sent=True`` so
     :mod:`general_beckman.result_router` keeps the row as ``waiting_human``
     (per its 148-180 branch). Founder confirms via Telegram → step
     completes; founder edits → todo regenerates on the next pass.

This step makes NO LLM call itself — generation lives in the upstream
analyst step ``0.0a.draft`` (per ``feedback_no_direct_dispatcher``: a
mechanical must never call HaLLederiz Kadir / the dispatcher directly). If
no draft is present (offline / analyst skipped), the canonical wording is
used verbatim, so the action stays testable without any model loaded.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Default item count window — Paraflow's gate is ~10-15.
_DEFAULT_MIN_ITEMS = 10
_DEFAULT_MAX_ITEMS = 15

# Canonical sections the charter + PRD will need answered. These cover
# the categories that today's i2p phase-0 interrogation hits one-by-one.
_CANONICAL_TOPICS = (
    ("Audience", "Who is the primary user (one named persona) and one secondary?"),
    ("Audience", "Where do these users live online today (forums, apps, communities)?"),
    ("Problem", "What single sentence captures the felt pain you are solving?"),
    ("Problem", "How are users coping with this pain today?"),
    ("Outcome", "What does success look like for the user 30 days after first use?"),
    ("Scope", "What 3-5 solutions will the product own (each with boundaries)?"),
    ("Scope", "What categories of work will the product explicitly NOT do?"),
    ("Differentiation", "Name 2-3 named competitors and why a user would switch."),
    ("Channel", "How will the first 10 users discover the product?"),
    ("Constraints", "What is the cost ceiling and timeline for the MVP?"),
    ("Constraints", "Any compliance / regulatory constraints (GDPR, HIPAA, etc.)?"),
    ("Surfaces", "Which surfaces (mobile / web / desktop) ship first?"),
    ("Brand", "Name 5 brand keywords (one-liners) that anchor the product feel."),
    ("Risk", "Name 2-3 risks that would kill the product."),
    ("Confirmation", "Confirm this todo list — reply 'OK' to proceed or edit any item."),
)


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        logger.debug("intake_todo: read failed for %s: %s", path, exc)
        return ""


def _gather_inputs(payload: dict) -> dict[str, str]:
    """Pull text inputs from payload (inline first, then paths)."""
    bag: dict[str, str] = {}
    inline = payload.get("inputs") or {}
    if isinstance(inline, dict):
        for k, v in inline.items():
            if isinstance(v, str) and v.strip():
                bag[k] = v
    paths = payload.get("paths") or {}
    if isinstance(paths, dict):
        for k, p in paths.items():
            if k in bag:
                continue
            if isinstance(p, str) and p.strip():
                txt = _read_text(p)
                if txt:
                    bag[k] = txt
    return bag


def _build_intake(
    inputs: dict[str, str], draft: dict[int, str] | None = None
) -> str:
    """Build the intake todo over the fixed 14-slot canonical skeleton.

    ``draft`` (slot ``n`` → analyst-reworded question, from ``0.0a.draft``)
    specialises the wording per slot; any slot the analyst omitted or
    mis-categorised falls back to the canonical question, so coverage and
    ordering never regress whatever the model quality.
    """
    draft = draft or {}
    pitch = (inputs.get("founder_pitch") or "").strip()
    rp = (inputs.get("reverse_pitch") or "").strip()
    z0 = (inputs.get("z0_outputs") or "").strip()

    head_lines = ["# Intake Todo", ""]
    if pitch:
        first = pitch.splitlines()[0][:200]
        head_lines.append(f"_Pitch:_ {first}")
    if rp:
        # First non-empty line of the reverse pitch is usually the headline.
        rp_head = next(
            (ln.strip().lstrip("# ").strip() for ln in rp.splitlines() if ln.strip()),
            "",
        )
        if rp_head:
            head_lines.append(f"_Reverse pitch headline:_ {rp_head[:200]}")
    if z0:
        head_lines.append("_Z0 outputs ingested._")
    if draft:
        head_lines.append(
            f"_Specialised {len(draft)}/{len(_CANONICAL_TOPICS)} items to this product._"
        )
    head_lines.append("")
    head_lines.append(
        "Confirm or edit any item. Reply **OK** to proceed; reply with edits to "
        "regenerate."
    )
    head_lines.append("")
    head_lines.append("## Items")

    items: list[str] = []
    for idx, (cat, q) in enumerate(_CANONICAL_TOPICS, start=1):
        question = draft.get(idx, q)
        items.append(f"- [ ] **{idx}. {cat}** — {question}")
    body = head_lines + items
    return "\n".join(body) + "\n"


def _deterministic_builder(inputs: dict[str, str]) -> str:
    """Back-compat alias: canonical skeleton, no analyst specialisation."""
    return _build_intake(inputs, None)


_DRAFT_REL = os.path.join(".intake", "intake_todo_draft.json")


def _load_analyst_draft(workspace_root: str, mission_id: Any) -> dict[int, str]:
    """Read the upstream analyst draft (``0.0a.draft``) and return
    ``{slot_n: question}`` for slots whose declared category matches the
    canonical category for that slot.

    Lenient by design: a missing file, JSON error, bad slot index, or
    category mismatch is silently dropped so the caller fills that slot from
    the canonical wording. A weak model can specialise wording but cannot
    drop, reorder, or invent dimensions. NO LLM call lives here — generation
    is the analyst step's job (``feedback_no_direct_dispatcher``).
    """
    import json

    path = os.path.join(workspace_root, f"mission_{mission_id}", _DRAFT_REL)
    raw = _read_text(path)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("intake_todo: analyst draft at %s is not valid JSON", path)
        return {}
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return {}
    out: dict[int, str] = {}
    n_slots = len(_CANONICAL_TOPICS)
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            n = int(it.get("n"))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= n_slots:
            continue
        cat = str(it.get("category") or "").strip().lower()
        q = str(it.get("question") or "").strip()
        if q and cat == _CANONICAL_TOPICS[n - 1][0].lower():
            out[n] = q
    return out


def _resolve_workspace_dir(workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    # Lazy import to avoid circulars at module import time.
    from src.tools.workspace import WORKSPACE_DIR
    return WORKSPACE_DIR


async def generate_intake_todo(task: dict) -> dict[str, Any]:
    """Build + persist ``intake_todo.md`` and return the clarify shape.

    Expected ``payload`` keys:
        inputs (dict[str, str], optional):
            ``founder_pitch``, ``reverse_pitch``, ``z0_outputs``.
        paths (dict[str, str], optional):
            Same keys, but absolute paths to read on disk.
        workspace_path (str, optional):
            Override for the WORKSPACE_DIR root.

    The product-specific wording, when present, comes from the upstream
    analyst draft on disk (see :func:`_load_analyst_draft`), never an LLM
    call made here.

    Output ``intake_todo.md`` lives at
    ``{workspace_root}/mission_{mission_id}/.intake/intake_todo.md``
    (per v3 N3 — paths are relative-to-WORKSPACE_DIR with the
    ``mission_{mission_id}/...`` prefix).
    """
    payload = task.get("payload") or {}
    mission_id = task.get("mission_id")
    if mission_id is None:
        return {
            "status": "failed",
            "error": "generate_intake_todo: task missing mission_id",
        }

    inputs = _gather_inputs(payload)
    workspace_root = _resolve_workspace_dir(payload.get("workspace_path"))
    draft = _load_analyst_draft(workspace_root, mission_id)
    body = _build_intake(inputs, draft)

    intake_dir = os.path.join(
        workspace_root, f"mission_{mission_id}", ".intake"
    )
    try:
        os.makedirs(intake_dir, exist_ok=True)
    except OSError as exc:
        return {
            "status": "failed",
            "error": f"generate_intake_todo: mkdir failed for {intake_dir}: {exc}",
        }
    todo_path = os.path.join(intake_dir, "intake_todo.md")
    try:
        with open(todo_path, "w", encoding="utf-8") as fh:
            fh.write(body)
    except OSError as exc:
        return {
            "status": "failed",
            "error": f"generate_intake_todo: write failed for {todo_path}: {exc}",
        }

    relative_path = (
        f"mission_{mission_id}/.intake/intake_todo.md"
    )
    # Send inline file + [OK / Regenerate / Edit] keyboard via the same
    # artifact-confirm helper used by reverse_pitch_confirm. Founder
    # never has to open the file in a shell.
    keyboard_sent = False
    try:
        from src.app.telegram_bot import get_telegram
        from src.collaboration.blackboard import read_blackboard
        from dabidabi import get_db as _get_db
        chat_id = None
        try:
            arts = await read_blackboard(int(mission_id), "artifacts")
            if isinstance(arts, dict):
                chat_id = arts.get("chat_id")
        except Exception:
            chat_id = None
        if chat_id is None:
            try:
                import json as _json
                _db = await _get_db()
                _cur = await _db.execute(
                    "SELECT context FROM missions WHERE id = ?", (mission_id,),
                )
                _row = await _cur.fetchone()
                await _cur.close()
                if _row and _row[0]:
                    _mctx = _json.loads(_row[0])
                    if isinstance(_mctx, str):
                        _mctx = _json.loads(_mctx)
                    chat_id = (_mctx or {}).get("chat_id")
            except Exception:
                pass
        if chat_id is not None:
            tg = get_telegram()
            if tg is not None:
                await tg.send_artifact_confirm_keyboard(
                    chat_id=int(chat_id),
                    mission_id=int(mission_id),
                    task_id=int(task.get("id") or 0),
                    kind="intake_todo_confirm",
                    question="Intake todo draft below. Tap a button to confirm, regenerate, or edit.",
                    files=[(relative_path, body)],
                    regenerate_step_id="0.0a",
                )
                keyboard_sent = True
    except Exception as exc:
        logger.warning("intake_todo: artifact-confirm keyboard send failed: %s", exc)
    return {
        "status": "needs_clarification",
        "kind": "intake_todo",
        "todo_path": relative_path,
        "todo_path_abs": todo_path,
        "item_count": sum(1 for line in body.splitlines() if line.startswith("- [ ]")),
        "keyboard_sent": keyboard_sent,
        "prompt": (
            "I drafted an intake todo at "
            f"`{relative_path}`. Reply OK to proceed, or send edits to "
            "regenerate."
        ),
    }
