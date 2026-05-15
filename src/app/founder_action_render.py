"""Z6 T1D — founder_action card rendering for Telegram.

Each founder_action surfaces in the mission thread as a card with:
  - Title, why, instructions, expected output kind hint
  - Inline buttons: ``In Progress``, ``Done``, ``Block``
  - Kind-specific cues (credential paste hint, cost confirm, etc.)

Z7 A0 addition — render_briefing(briefing_row, weekly_minutes_saved=None):
  Formats a mission_briefings row into a Telegram-ready Markdown string.
  Sections: phase summary / changed files / deferred / cost / recovered-failures.
  Includes a weekly founder_minutes_saved rollup when weekly_minutes_saved provided.
  Contains a clearly-marked placeholder where the attention-budget queue will be
  rendered (owned by a separate agent; returns empty string until wired).

Pure functions — no IO. The bot wiring layer picks up the text and posts it.
"""
from __future__ import annotations

from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


# Emoji per kind — quick visual scan in a busy mission thread.
_KIND_EMOJI = {
    "credential_paste": "🔑",
    "vendor_enroll": "🏢",
    "cost_ack": "💸",
    "legal_counsel": "⚖️",
    "kyc": "🪪",
    "generic": "📌",
}


def _kind_emoji(kind: str) -> str:
    return _KIND_EMOJI.get(kind, "📌")


def _credential_paste_hint(action: dict) -> str:
    """Build a copy-pasteable /credential add hint with schema fields."""
    schema = action.get("expected_output_schema") or {}
    fields = []
    if isinstance(schema, dict):
        required = schema.get("required_fields") or []
        if isinstance(required, list):
            fields = list(required)
    # Default to a single 'token' field if no schema declared.
    if not fields:
        fields = ["token"]
    placeholder = ", ".join(f'"{f}": "<value>"' for f in fields)
    # Derive service name from instructions if possible — leave a {service}
    # token so the founder can fill in. Best-effort heuristic: look for
    # words ending in "Connect/dashboard" in title.
    title = action.get("title") or ""
    service = "<service>"
    # Crude extraction: 'Paste Stripe credentials' -> 'stripe'
    parts = title.lower().split()
    if "paste" in parts:
        idx = parts.index("paste")
        if idx + 1 < len(parts):
            service = parts[idx + 1].rstrip(":")
    return f"`/credential add {service} {{{placeholder}}}`"


def render_action_card(action: dict) -> tuple[str, InlineKeyboardMarkup]:
    """Format a founder_action dict into (markdown_text, InlineKeyboardMarkup).

    ``action`` is the dict shape produced by ``FounderAction.to_dict``.
    """
    kind = action.get("kind", "generic")
    emoji = _kind_emoji(kind)
    aid = action.get("id")
    title = action.get("title") or "Founder action"
    why = action.get("why") or ""
    instructions = action.get("instructions") or []
    expected = action.get("expected_output_kind")
    cost = action.get("cost_estimate_usd")
    step_id = action.get("blocking_step_id")
    mission_id = action.get("mission_id")

    lines: list[str] = []
    lines.append(f"{emoji} *Founder action #{aid}* — {kind}")
    lines.append(f"*{title}*")
    if mission_id:
        suffix = f" • step `{step_id}`" if step_id else ""
        lines.append(f"_Mission {mission_id}{suffix}_")
    if why:
        lines.append("")
        lines.append(f"_Why:_ {why}")
    if instructions:
        lines.append("")
        lines.append("*What to do:*")
        for i, step in enumerate(instructions, 1):
            lines.append(f"{i}. {step}")
    if expected:
        lines.append("")
        lines.append(f"_Expected output:_ `{expected}`")

    # Kind-specific cues.
    if kind == "credential_paste":
        lines.append("")
        lines.append("Paste via:")
        lines.append(_credential_paste_hint(action))
    elif kind == "cost_ack":
        if cost is not None:
            lines.append("")
            lines.append(f"💰 *Estimated cost: ${float(cost):.2f}*")
            lines.append("Tap *Confirm* to authorize this spend.")
    elif kind == "vendor_enroll":
        lines.append("")
        lines.append(
            "After enrolling, paste back the resulting credential / "
            "order ID via the link button or `/action_done`."
        )

    text = "\n".join(lines)

    # Buttons — different for cost_ack (single confirm) vs the rest.
    if kind == "cost_ack":
        kb_rows = [[
            InlineKeyboardButton("✅ Confirm", callback_data=f"fa_done_{aid}"),
            InlineKeyboardButton("⛔ Block", callback_data=f"fa_block_{aid}"),
        ]]
    else:
        kb_rows = [
            [
                InlineKeyboardButton(
                    "▶️ In Progress", callback_data=f"fa_inprogress_{aid}",
                ),
                InlineKeyboardButton(
                    "✅ Done", callback_data=f"fa_done_{aid}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "⛔ Block", callback_data=f"fa_block_{aid}",
                ),
            ],
        ]
    return text, InlineKeyboardMarkup(kb_rows)


# ---------------------------------------------------------------------------
# Z7 A0 — Briefing card renderer
# ---------------------------------------------------------------------------

def _render_attention_section(product_id: str) -> str:
    """Placeholder: render the attention-budget queue for product_id.

    *** INSERTION POINT — attention-budget queue ***
    Another agent (Z7 attention-budget owner) will replace this stub with real
    logic that surfaces prioritised founder_actions from the attention queue.
    Until wired, returns empty string so render_briefing is unblocked.

    Import is defensive: if the attention module is absent (not yet shipped),
    this function silently returns "".
    """
    try:
        from src.app.attention_budget import render_queue  # type: ignore
        return render_queue(product_id)
    except (ImportError, AttributeError):
        return ""


def render_briefing(
    briefing_row: dict,
    *,
    weekly_minutes_saved: int | None = None,
) -> str:
    """Format a mission_briefings DB row into a Telegram-ready Markdown string.

    Args:
        briefing_row: dict with at minimum keys: kind, body_md,
            founder_minutes_saved_estimate, product_id.
        weekly_minutes_saved: if provided, appends a weekly ROI rollup section.

    Returns:
        Markdown text string. Never raises — falls back to a minimal string
        on any unexpected input so the bot send never crashes.
    """
    try:
        kind = briefing_row.get("kind", "briefing")
        product_id = str(briefing_row.get("product_id") or "")
        mission_id = briefing_row.get("mission_id")
        body_md = briefing_row.get("body_md") or "_(no content)_"
        minutes_saved = briefing_row.get("founder_minutes_saved_estimate")
        prepared_at = briefing_row.get("prepared_at") or ""

        # Header
        if kind == "completion":
            header = f"*Mission Completion Briefing*"
            if mission_id:
                header += f" — Mission #{mission_id}"
        else:
            header = "*Daily Briefing*"
        if prepared_at:
            # Trim to date+time without seconds for readability
            prepared_str = str(prepared_at)[:16]
            header += f"\n_{prepared_str}_"

        parts: list[str] = [header, "", body_md]

        # Founder minutes saved (this briefing)
        if minutes_saved is not None and int(minutes_saved) > 0:
            parts.append(
                f"\n*Time saved this mission:* {int(minutes_saved)} min "
                f"({int(minutes_saved) // 60}h {int(minutes_saved) % 60}m)"
            )

        # Weekly rollup
        if weekly_minutes_saved is not None:
            hours = weekly_minutes_saved // 60
            mins = weekly_minutes_saved % 60
            parts.append(
                f"*Weekly founder time saved:* {weekly_minutes_saved} min "
                f"({hours}h {mins}m)"
            )

        # *** ATTENTION BUDGET QUEUE — placeholder section ***
        # Owned by Z7 attention-budget agent. Returns "" until wired.
        attention_section = _render_attention_section(product_id)
        if attention_section:
            parts.append(attention_section)

        return "\n".join(parts)

    except Exception as exc:  # noqa: BLE001
        # Never let a rendering error crash the bot send path.
        return f"*Briefing* _(render error: {exc})_"
