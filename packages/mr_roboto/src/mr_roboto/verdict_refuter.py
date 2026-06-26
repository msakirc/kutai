"""Tier-2 adversarial refuter for reviewer verdict verification (2026-06-26).

When Tier-1 deterministic grounding cannot decide a blocking finding (no
checkable quote / section), the apply layer spawns ONE batched admitted LLM
child — an independent model asked to REFUTE each surviving finding. We keep
only the findings the refuter can actually support, fail-closed against a
refuter that confabulates its own supporting quote.

Mirrors the SP6 critic-gate split: a pure spec builder + a fail-closed parser
here; the admitted child + resume + verdict application live in general_beckman
(`posthook_continuations._verdict_verify_resume`). A mechanical never calls the
dispatcher — this module only builds/parses; it makes no LLM call.
"""
from __future__ import annotations

import json
import re
from typing import Any

_MAX_CONTENT_CHARS = 6000  # per-artifact excerpt cap in the refuter prompt

_PROMPT_HEADER = (
    "You are an adversarial fact-checker for a product-research reviewer. A "
    "reviewer flagged the findings below as reasons to REJECT (halt) the work. "
    "Reviewers frequently CONFABULATE — they invent quotes, echo rubric examples, "
    "and claim sections are missing that are actually present. Your job is to "
    "REFUTE each finding: decide whether it is genuinely TRUE of the artifact.\n\n"
    "For each finding:\n"
    "- If it claims something IS in the artifact (a quote, contradiction, or "
    "stated fact), answer SUPPORTED only if you can quote the exact offending "
    "text from the artifact. Put that verbatim text in `quote`.\n"
    "- If it claims something is MISSING/absent/empty, answer SUPPORTED only if, "
    "after looking, the named thing is genuinely absent (leave `quote` empty).\n"
    "- Otherwise answer UNSUPPORTED. DEFAULT TO UNSUPPORTED when you are unsure "
    "or cannot find concrete evidence. A wrongful SUPPORTED halts real work.\n\n"
    "Respond with a single JSON object and nothing else:\n"
    '{"verdicts": [{"index": <int>, "status": "SUPPORTED" | "UNSUPPORTED", '
    '"quote": "<verbatim evidence or empty>"}, ...]}\n\n'
)


def build_refuter_spec(candidates: list[dict]) -> dict:
    """Build the raw_dispatch OVERHEAD spec for the batched refuter child.

    ``candidates`` — list of ``{target_artifact, problem, content}``. The
    artifact content (capped) is injected so the refuter judges against the
    real text, not its training prior. Mirrors ``critic_gate._build_critic_spec``
    (single non-agentic overhead LLM call).
    """
    blocks: list[str] = []
    for i, c in enumerate(candidates):
        content = str(c.get("content") or "")
        if len(content) > _MAX_CONTENT_CHARS:
            content = content[:_MAX_CONTENT_CHARS] + "\n…[truncated]"
        blocks.append(
            f"### Finding index {i}\n"
            f"target_artifact: {c.get('target_artifact')}\n"
            f"finding: {c.get('problem')}\n\n"
            f"--- artifact `{c.get('target_artifact')}` ---\n{content}\n--- end artifact ---\n"
        )
    prompt = _PROMPT_HEADER + "\n".join(blocks)
    messages = [{"role": "user", "content": prompt}]
    return {
        "title": "verdict_refuter",
        "description": "Adversarial refutation of reviewer findings before a halt.",
        "agent_type": "critic",
        "kind": "overhead",
        "priority": 1,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "verdict_refuter",
            "agent_type": "critic",
            "difficulty": 3,
            "messages": messages,
            "failures": [],
            "estimated_output_tokens": 768,
            "prefer_speed": False,
        }},
    }


def parse_refuter_output(raw: str, num_candidates: int) -> dict[int, dict] | None:
    """Parse the refuter's batched verdicts into ``{index: {status, quote}}``.

    Returns ``None`` when the WHOLE output is unparseable (no JSON object / no
    ``verdicts`` array) — the caller then keeps ALL candidates (a refuter outage
    must never silently disable the halt). A parsed-but-partial output yields a
    dict; candidates absent from it are handled by ``refuter_keep(None, …)``.
    """
    text = (raw or "").strip()
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    obj_m = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_m:
        text = obj_m.group(0)
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(obj, dict):
        return None
    verdicts = obj.get("verdicts")
    if not isinstance(verdicts, list):
        return None
    out: dict[int, dict] = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        try:
            idx = int(v.get("index"))
        except (TypeError, ValueError):
            continue
        status = str(v.get("status") or "").strip().lower()
        if status not in {"supported", "unsupported"}:
            continue
        out[idx] = {"status": status, "quote": str(v.get("quote") or "")}
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def refuter_keep(entry: Any, content: str | None) -> bool:
    """Per-candidate keep decision (fail-closed against confabulated support).

      * no entry / empty entry (refuter silent on it) → KEEP (don't drop on silence)
      * UNSUPPORTED                                   → DROP
      * SUPPORTED, no quote (absence confirmation)    → KEEP
      * SUPPORTED, quote present in artifact           → KEEP (real evidence)
      * SUPPORTED, quote NOT in artifact (fabricated)  → DROP (fail-closed)
    """
    if not isinstance(entry, dict) or not entry:
        return True
    status = str(entry.get("status") or "").lower()
    if status == "unsupported":
        return False
    if status != "supported":
        return True  # unknown status → keep (don't drop on ambiguity)
    quote = str(entry.get("quote") or "").strip()
    if not quote:
        return True  # absence-claim confirmation — no quote to verify
    if not isinstance(content, str) or not content.strip():
        return True  # can't re-ground → keep
    return _norm(quote) in _norm(content)
