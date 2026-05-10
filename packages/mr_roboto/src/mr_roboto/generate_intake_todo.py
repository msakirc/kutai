"""Intake-todo generator — Tier 1 of Z1 (B1).

Replaces phase-0's interrogation-by-many-questions pattern with a single
generated todo list the founder confirms before charter generation begins.

Behaviour:
  1. Read founder pitch + Z0 outputs + ``reverse_pitch.md`` (from 0.0z)
     from ``payload`` (or from disk via ``payload['paths']``).
  2. Either invoke an analyst LLM (when wired) or fall back to a
     deterministic structural builder so the action is testable in CI
     without any model loaded.
  3. Write ``mission_{mission_id}/.intake/intake_todo.md`` (10-15 items).
  4. Return ``status="needs_clarification"`` with ``keyboard_sent=True``
     so :mod:`general_beckman.result_router` keeps the row as
     ``waiting_human`` (per its 148-180 branch). The founder confirms via
     Telegram → step completes; founder edits → todo regenerates on the
     next pass.

The LLM hook is opportunistic: when ``payload.get("use_llm")`` is true
AND ``hallederiz_kadir`` is importable, we delegate text-generation; the
caller controls cost. Default is the deterministic builder, which keeps
unit tests fast and offline.
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


def _deterministic_builder(inputs: dict[str, str]) -> str:
    """Build a baseline intake todo without any LLM call.

    The list is structural — every charter+PRD field downstream needs an
    answer for. The LLM-driven path (when wired) will reword each item
    against the actual founder pitch + reverse pitch; the deterministic
    path keeps the same 10-15 item shape so callers can rely on the
    contract in tests.
    """
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
    head_lines.append("")
    head_lines.append(
        "Confirm or edit any item. Reply **OK** to proceed; reply with edits to "
        "regenerate."
    )
    head_lines.append("")
    head_lines.append("## Items")

    items: list[str] = []
    for idx, (cat, q) in enumerate(_CANONICAL_TOPICS, start=1):
        items.append(f"- [ ] **{idx}. {cat}** — {q}")
    body = head_lines + items
    return "\n".join(body) + "\n"


def _llm_builder(inputs: dict[str, str]) -> str | None:
    """Optional: delegate generation to hallederiz_kadir analyst call.

    Returns ``None`` on any error so the caller falls back to the
    deterministic builder.
    """
    try:  # pragma: no cover - optional path
        from hallederiz_kadir import call as _hk_call  # type: ignore
    except Exception:
        return None
    prompt_parts = [
        "Generate a 10-15 item intake todo for an early-product founder. "
        "Each item is one bullet, prefixed by a category tag. Categories: "
        "Audience, Problem, Outcome, Scope, Differentiation, Channel, "
        "Constraints, Surfaces, Brand, Risk. End with a Confirmation item.",
        "",
        "Founder pitch:",
        inputs.get("founder_pitch", "(none)"),
        "",
        "Reverse pitch:",
        inputs.get("reverse_pitch", "(none)"),
        "",
        "Z0 outputs:",
        inputs.get("z0_outputs", "(none)"),
    ]
    try:
        resp = _hk_call(  # type: ignore[call-arg]
            messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
            max_tokens=2000,
            profile="overhead",
        )
    except Exception as exc:
        logger.warning("intake_todo: LLM path failed (%s) — using deterministic builder", exc)
        return None
    text = (resp or {}).get("text") if isinstance(resp, dict) else None
    if not text or not isinstance(text, str):
        return None
    return text


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
        use_llm (bool, optional):
            When True + hallederiz_kadir available, delegate text gen.

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
    body: str | None = None
    if payload.get("use_llm"):
        body = _llm_builder(inputs)
    if not body:
        body = _deterministic_builder(inputs)

    workspace_root = _resolve_workspace_dir(payload.get("workspace_path"))
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

    # Hand control back to the founder via clarify-shape so beckman keeps
    # the row in waiting_human (general_beckman.result_router lines 148-180).
    relative_path = (
        f"mission_{mission_id}/.intake/intake_todo.md"
    )
    return {
        "status": "needs_clarification",
        "kind": "intake_todo",
        "todo_path": relative_path,
        "todo_path_abs": todo_path,
        "item_count": sum(1 for line in body.splitlines() if line.startswith("- [ ]")),
        "keyboard_sent": True,
        "prompt": (
            "I drafted an intake todo at "
            f"`{relative_path}`. Reply OK to proceed, or send edits to "
            "regenerate."
        ),
    }
