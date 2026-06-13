"""Critic gate (B4) — second-LLM veto on irreversible actions.

Z1 Tier 5C. Wrapped as a post-hook on `git_commit` and `notify_user`.

SP3b Task 8 — split into PRODUCER + mechanical CONFIRM gate
-----------------------------------------------------------
A mechanical must NEVER call the dispatcher directly (memory
``feedback_no_direct_dispatcher``). The gate is therefore split:

- :func:`produce_verdict` — the verdict PRODUCER. An admitted single-call
  unit: it builds the critic prompt and runs it through ``husam.run`` (the
  non-agentic single-call worker), which does select→execute→map. husam is
  the admitted worker path; the producer never imports/instantiates the
  dispatcher. The parsed verdict is persisted via :func:`_persist`.
- :func:`confirm_gate` — the mechanical CONFIRM gate. It makes NO LLM call
  and imports NO dispatcher. Given a persisted verdict (or the opt-out), it
  returns pass/block. Mechanical executors call this.

The public :func:`critic_gate` is kept as a thin orchestrator (opt-out →
``produce_verdict``); the inline ``git_commit``/``notify_user`` mechanical
sites still call it, but the LLM hop now travels through husam, not the
dispatcher.

Prompt: "About to perform <action_name> with payload <redacted_payload>.
Would this action: (a) break the locked spec, (b) cause founder fury,
(c) leak secrets/credentials/PII? Respond JSON: `{verdict: "pass"|"veto",
reasons: [str]}`." The router translates ``verdict == "veto"`` into
``Action(status="failed", ...)`` and rolls the irreversible side-effect back
(unstage commit, drop notification).

Opt-out
-------
Set ``KUTAI_CRITIC_GATE=off`` to bypass the gate (default: on). Default-pass
is also returned when the producer call itself raises — never block work on a
broken critic. Logs+critic_log row still records the bypass.

Schema
------
Idempotent CREATE of ``critic_log(id, mission_id, action_name, verdict,
reasons_json, redacted_payload_hash, created_at)`` is wired in
`src/infra/db.py::init_db`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.critic_gate")


# Patterns to redact before sending payload to the critic.
_SECRET_PATTERNS = [
    # api keys / tokens / bearer
    re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|bearer)\s*[=:]\s*['\"]?([^'\"\s,}]+)"),
    # AWS-style keys
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # Long hex strings (>=32 chars) — likely keys/hashes
    re.compile(r"\b[a-fA-F0-9]{32,}\b"),
    # Bearer tokens in headers
    re.compile(r"(?i)Authorization:\s*Bearer\s+[^\s]+"),
    # Email addresses (PII)
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Phone numbers (loose)
    re.compile(r"\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}"),
]

_REDACTED = "[REDACTED]"


def _redact(text: str) -> str:
    if not text:
        return text or ""
    out = text
    for pat in _SECRET_PATTERNS:
        # Replace whole match with REDACTED (or pattern-specific shape).
        out = pat.sub(lambda m: f"{m.group(1)}={_REDACTED}" if m.lastindex and m.group(1) else _REDACTED, out)
    return out


def _redact_payload(payload: Any) -> Any:
    """Recursively redact secrets from payload before sending to critic."""
    if isinstance(payload, str):
        return _redact(payload)
    if isinstance(payload, dict):
        return {k: _redact_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_redact_payload(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_redact_payload(v) for v in payload)
    return payload


def _hash_payload(payload: Any) -> str:
    try:
        s = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        s = str(payload)
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _opt_out() -> bool:
    return (os.environ.get("KUTAI_CRITIC_GATE") or "").strip().lower() in {"off", "0", "false", "no"}


_PROMPT_TEMPLATE = (
    "You are a release-critic. About to perform action `{action_name}` with "
    "the following (secret-redacted) payload:\n\n"
    "```json\n{payload_json}\n```\n\n"
    "Would this action: (a) break the locked product spec, (b) cause founder "
    "fury (loss of trust, embarrassment, contradicting prior decisions), or "
    "(c) leak secrets / credentials / PII?\n\n"
    "Respond with a single JSON object and nothing else:\n"
    '{{"verdict": "pass" | "veto", "reasons": ["short reason", ...]}}\n'
    "Default to 'pass' when no clear hazard is present. Use 'veto' only "
    "when at least one of (a)/(b)/(c) is concretely visible in the payload."
)


def _parse_verdict(raw: str) -> dict:
    """Best-effort JSON extraction. Defaults to pass on garbage."""
    if not raw:
        return {"verdict": "pass", "reasons": ["empty critic response"]}
    text = raw.strip()
    # Strip ```json fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # First {...} block
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        text = obj_match.group(0)
    try:
        obj = json.loads(text)
    except Exception:
        return {"verdict": "pass", "reasons": [f"unparseable critic response: {raw[:200]}"]}
    verdict = str(obj.get("verdict", "pass")).strip().lower()
    if verdict not in {"pass", "veto"}:
        verdict = "pass"
    reasons = obj.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r) for r in reasons]
    return {"verdict": verdict, "reasons": reasons}


async def _persist(
    mission_id: int | None,
    action_name: str,
    verdict: str,
    reasons: list[str],
    payload_hash: str,
) -> None:
    """Write a critic_log row. Best-effort; never raises."""
    try:
        from src.infra.db import get_db
    except Exception:
        return
    try:
        db = await get_db()
        await db.execute(
            """
            INSERT INTO critic_log
                (mission_id, action_name, verdict, reasons_json,
                 redacted_payload_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(mission_id) if mission_id is not None else 0,
                action_name,
                verdict,
                json.dumps(reasons, ensure_ascii=False),
                payload_hash,
            ),
        )
        await db.commit()
    except Exception as e:  # pragma: no cover
        logger.debug(f"critic_log persist skipped: {e}")


def _build_critic_spec(action_name: str, redacted: Any) -> dict:
    """Build the raw_dispatch OVERHEAD spec for ``husam.run``.

    Mirrors the ``constrained_emit`` / ``self_reflect`` post-hook child specs
    (apply.py ~1423) — a single non-agentic LLM call on the overhead lane. The
    critic prompt carries the SECRET-REDACTED payload only.
    """
    try:
        payload_json = json.dumps(redacted, indent=2, ensure_ascii=False, default=str)[:4000]
    except Exception:
        payload_json = str(redacted)[:4000]

    prompt = _PROMPT_TEMPLATE.format(
        action_name=action_name,
        payload_json=payload_json,
    )
    messages = [{"role": "user", "content": prompt}]
    return {
        "title": f"critic_gate:{action_name}",
        "description": "Release-critic veto check on an irreversible action.",
        "agent_type": "critic",
        "kind": "overhead",
        "priority": 1,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": f"critic_gate:{action_name}",
            "agent_type": "critic",
            "difficulty": 2,
            "messages": messages,
            "failures": [],
            "estimated_output_tokens": 512,
            "prefer_speed": True,
        }},
    }


async def produce_verdict(
    action_name: str,
    payload: Any,
    *,
    mission_id: int | None = None,
) -> dict:
    """Verdict PRODUCER — run the critic prompt through the admitted worker.

    Routes through ``husam.run`` (the non-agentic single-call worker) — NEVER
    ``LLMDispatcher().request(...)`` — so a mechanical never calls the
    dispatcher directly (``feedback_no_direct_dispatcher``). husam handles
    select→execute→map internally and returns the legacy response dict.

    Persists one ``critic_log`` row with the parsed verdict and returns::

        {"verdict": "pass" | "veto", "reasons": [...],
         "bypassed": False, "payload_hash": str, "raw": str}

    Never raises — on any producer error returns a default-pass with an
    explanatory reason (never block work on a broken critic).
    """
    redacted = _redact_payload(payload)
    payload_hash = _hash_payload(redacted)

    raw_text = ""
    try:
        import husam
        spec = _build_critic_spec(action_name, redacted)
        if mission_id is not None:
            spec["mission_id"] = mission_id
        resp = await husam.run(spec)
        # Response is the legacy dict with a `content` key.
        if isinstance(resp, dict):
            raw_text = str(resp.get("content") or resp.get("text") or "")
        else:
            raw_text = str(resp)
    except Exception as e:
        logger.warning(f"critic_gate producer call failed: {e}; default-passing")
        result = {
            "verdict": "pass",
            "reasons": [f"critic call failed: {e}"],
            "bypassed": False,
            "payload_hash": payload_hash,
        }
        await _persist(mission_id, action_name, "pass", result["reasons"], payload_hash)
        return result

    parsed = _parse_verdict(raw_text)
    result = {
        "verdict": parsed["verdict"],
        "reasons": parsed["reasons"],
        "bypassed": False,
        "payload_hash": payload_hash,
        "raw": raw_text[:400],
    }
    await _persist(mission_id, action_name, parsed["verdict"], parsed["reasons"], payload_hash)
    return result


async def confirm_gate(
    action_name: str,
    payload: Any,
    *,
    mission_id: int | None = None,
    persisted_verdict: dict | None = None,
) -> dict:
    """Mechanical CONFIRM gate — reads a persisted verdict, returns pass/block.

    Makes NO LLM call and imports NO dispatcher. A mechanical executor calls
    this with the verdict the PRODUCER persisted (``persisted_verdict``).

    Resolution order:
      1. ``KUTAI_CRITIC_GATE=off`` → pass + ``bypassed=True`` (records a row).
         This is the ONLY pass-without-verdict path.
      2. A ``persisted_verdict`` carrying ``verdict in {"pass","veto"}`` →
         honour it.
      3. No usable verdict (producer never ran / failed / garbage) →
         VETO (fail-CLOSED, SP6). A broken critic must block the irreversible
         action; ``KUTAI_CRITIC_GATE=off`` is the only bypass.

    Returns the same dict shape as :func:`produce_verdict`.
    """
    redacted = _redact_payload(payload)
    payload_hash = _hash_payload(redacted)

    if _opt_out():
        result = {
            "verdict": "pass",
            "reasons": ["KUTAI_CRITIC_GATE=off — gate bypassed"],
            "bypassed": True,
            "payload_hash": payload_hash,
        }
        await _persist(mission_id, action_name, "pass", result["reasons"], payload_hash)
        return result

    verdict = ""
    reasons: list[str] = []
    if isinstance(persisted_verdict, dict):
        verdict = str(persisted_verdict.get("verdict") or "").strip().lower()
        raw_reasons = persisted_verdict.get("reasons") or []
        if isinstance(raw_reasons, list):
            reasons = [str(r) for r in raw_reasons]
        elif raw_reasons:
            reasons = [str(raw_reasons)]

    if verdict not in {"pass", "veto"}:
        # SP6: gate is ENABLED (opt-out handled above) but no usable verdict
        # is present (producer never ran / failed / garbage). FAIL-CLOSED — a
        # broken critic must BLOCK the irreversible action, never wave it
        # through. KUTAI_CRITIC_GATE=off (rule 1) is the only pass-without-
        # verdict path.
        return {
            "verdict": "veto",
            "reasons": reasons or ["no critic verdict available — fail-closed"],
            "bypassed": False,
            "payload_hash": payload_hash,
        }

    return {
        "verdict": verdict,
        "reasons": reasons,
        "bypassed": False,
        "payload_hash": payload_hash,
    }


async def critic_gate(
    action_name: str,
    payload: Any,
    *,
    mission_id: int | None = None,
) -> dict:
    """Run the critic gate against an action+payload (thin orchestrator).

    Kept stable for the inline ``git_commit`` / ``notify_user`` mechanical
    sites and the standalone ``critic_gate`` action. The opt-out short-circuit
    lives here; otherwise it delegates to the PRODUCER (:func:`produce_verdict`),
    which routes the LLM hop through ``husam`` rather than the dispatcher.

    Returns dict::

        {
          "verdict": "pass" | "veto",
          "reasons": [str, ...],
          "bypassed": bool,         # True if KUTAI_CRITIC_GATE=off
          "payload_hash": str,
        }

    Never raises — on any internal error returns a default-pass with an
    explanatory reason. Persists one ``critic_log`` row per call (unless
    persistence itself fails, which is logged at debug).
    """
    if _opt_out():
        redacted = _redact_payload(payload)
        payload_hash = _hash_payload(redacted)
        result = {
            "verdict": "pass",
            "reasons": ["KUTAI_CRITIC_GATE=off — gate bypassed"],
            "bypassed": True,
            "payload_hash": payload_hash,
        }
        await _persist(mission_id, action_name, "pass", result["reasons"], payload_hash)
        return result

    return await produce_verdict(action_name, payload, mission_id=mission_id)
