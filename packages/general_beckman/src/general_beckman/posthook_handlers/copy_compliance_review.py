"""Z7 A6 — copy_compliance_review posthook handler.

Runs over externally-published text artifacts and checks:

  1. **Privacy-policy ↔ marketing-copy mismatch** (LLM semantic, OVERHEAD lane)
     Finds claims in the copy that contradict the generated privacy_policy.md.
     Structured output: ``{contradicts: yes|no|unclear, citation: str}``.
     Severity: **blocker**.

  2. **Outcome claims without disclosure** (regex)
     Catches quantified outcomes ("save 10 hours/week", "3× faster") without
     a nearby "results vary" / "results may vary" / "individual results" phrase.
     Severity: **warning**.

  3. **Trademark/superlative violations** (regex, jurisdiction-aware)
     Catches "best", "guaranteed", "#1", etc. in jurisdictions that require
     substantiation (US, EU, GB, AU; skip for unknown/permissive).
     Severity: **warning**.

  4. **Forward-looking statements without safe-harbor** (regex)
     Catches "will", "expect", "project", "anticipate" in forward-looking
     sentences. Degrade gracefully if jurisdiction unknown.
     Severity: **info**.

  5. **Channel-specific rules** (deterministic, ``ChannelRules``)
     Loads ``docs/templates/channel_rules/<channel>.md`` and enforces
     max-length, banned-words, required-disclosures, image-required.
     Severity: **warning** (channel policy, not legal).

All inputs are read from task context. Missing inputs degrade gracefully:
the check is skipped with an info-level note rather than crashing.

Handler contract
----------------
Return ``{"status": "ok"}`` on pass or ``{"status": "fail", "verdict": "fail",
"findings": [...], "blocker_count": N, "warning_count": N,
"info_count": N, "fix_suggestions": {check_id: alt_phrasing}}``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.copy_compliance_review")

# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------

SEV_BLOCKER = "blocker"
SEV_WARNING = "warning"
SEV_INFO = "info"

# ---------------------------------------------------------------------------
# Jurisdictions that require substantiation for superlatives / guarantees
# ---------------------------------------------------------------------------

_SUBSTANTIATION_REQUIRED_JURISDICTIONS: frozenset[str] = frozenset({
    "us", "eu", "gb", "uk", "au", "ca",
})

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Outcome claim: quantified statements like "save N hours" / "X% faster"
_OUTCOME_CLAIM_RE = re.compile(
    r"""
    (?:
        # N hours/minutes/days/weeks faster/better/cheaper/saved/more/less
        (?:\d+\s*(?:x|×|%|hours?|minutes?|days?|weeks?)\s+
           (?:faster|better|cheaper|saved?|gain(?:ed)?|more|less))
        # save(s) [you/users/teams/...] N hours/minutes/dollars
      | (?:save[sd]?\s+(?:\w+\s+)?\d+\s+(?:hours?|minutes?|days?|dollars?|\$))
        # 3x ROI / 3x faster / 3x growth
      | (?:\d+x\s+(?:roi|return|growth|faster|better))
        # reduces N% / improves N%
      | (?:(?:reduc|increas|improv)\w*\s+(?:\w+\s+)?\d+\s*%)
        # N% faster/more/better
      | (?:\d+\s*%\s+(?:faster|more|better|cheaper|less|increase|improvement))
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RESULTS_VARY_RE = re.compile(
    r"results?\s+(?:may\s+)?vary|individual\s+results?|past\s+performance",
    re.IGNORECASE,
)

# Superlative / guarantee patterns
_SUPERLATIVE_RE = re.compile(
    r"\b(?:best|#1|number\s+one|guaranteed|world[\s-]?class|industry[\s-]?leading"
    r"|most\s+\w+|top[\s-]rated|unmatched|unrivaled|unbeatable|perfect)\b",
    re.IGNORECASE,
)

# Forward-looking statement triggers
_FORWARD_LOOKING_RE = re.compile(
    r"\b(?:will\s+(?:be|become|achieve|deliver|enable|allow|provide|grow|increase)"
    r"|expect\s+to|project(?:ed|s)?\s+to|anticipate[sd]?\s+to|intend[sd]?\s+to"
    r"|plan[sd]?\s+to|forecast[sd]?)\b",
    re.IGNORECASE,
)

# Safe-harbor language
_SAFE_HARBOR_RE = re.compile(
    r"forward[\s-]looking|safe[\s-]harbor|actual\s+results?\s+may\s+differ"
    r"|uncertainties|risks?\s+and\s+uncertainties",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_text(path: str | None, workspace_path: str | None = None) -> str | None:
    """Resolve and read a text file. Returns None if not found."""
    if not path:
        return None
    candidates = [path]
    if workspace_path and not os.path.isabs(path):
        candidates.insert(0, os.path.join(workspace_path, path))
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return fh.read()
        except (OSError, FileNotFoundError):
            continue
    return None


def _load_privacy_policy(task: dict, ctx: dict) -> str | None:
    """Find privacy_policy.md from compliance_templates or workspace."""
    workspace_path: str = ctx.get("workspace_path") or ""
    # 1. Explicit override in context
    explicit = ctx.get("privacy_policy_path")
    if explicit:
        content = _load_text(explicit, workspace_path)
        if content:
            return content

    # 2. Look in workspace compliance_templates/ (Z6 renders here)
    if workspace_path:
        candidates = [
            os.path.join(workspace_path, "compliance_templates", "privacy_policy.md"),
            os.path.join(workspace_path, "privacy_policy.md"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        return fh.read()
                except OSError:
                    pass

    # 3. From mission_id → workspace
    mission_id = task.get("mission_id")
    if mission_id:
        try:
            from src.tools.workspace import get_mission_workspace
            ws = get_mission_workspace(int(mission_id))
            for rel in ("compliance_templates/privacy_policy.md", "privacy_policy.md"):
                p = os.path.join(ws, rel)
                if os.path.isfile(p):
                    with open(p, "r", encoding="utf-8") as fh:
                        return fh.read()
        except Exception:
            pass

    # 4. The workspace/mission_57 JSON format (legacy)
    # privacy_policy.md can be a JSON file with a "privacy_policy" key
    if workspace_path:
        json_pp = os.path.join(workspace_path, "privacy_policy.md")
        if os.path.isfile(json_pp):
            try:
                with open(json_pp, "r", encoding="utf-8") as fh:
                    raw = fh.read()
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and "privacy_policy" in parsed:
                        return parsed["privacy_policy"]
                except json.JSONDecodeError:
                    pass
                return raw
            except OSError:
                pass
    return None


def _load_copy_text(task: dict, ctx: dict, result: dict) -> str | None:
    """Extract the copy text from the artifact result or workspace file."""
    # From result dict (structured output)
    copy_text = result.get("copy_text") or result.get("text") or result.get("content")
    if isinstance(copy_text, str) and copy_text.strip():
        return copy_text.strip()

    # From context artifact path
    artifact_path = ctx.get("artifact_path") or ctx.get("copy_path")
    workspace_path = ctx.get("workspace_path") or ""
    if artifact_path:
        content = _load_text(artifact_path, workspace_path)
        if content:
            return content.strip()

    # From produces[0] in context
    produces = list(ctx.get("produces") or [])
    if produces:
        for p in produces:
            content = _load_text(p, workspace_path)
            if content:
                return content.strip()

    # From task title/description as last resort (for unit tests)
    desc = task.get("description") or ""
    if desc.strip():
        return desc.strip()
    return None


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _check_outcome_claims(copy_text: str) -> list[dict]:
    """Check for outcome claims without 'results vary' disclosure."""
    findings: list[dict] = []
    for m in _OUTCOME_CLAIM_RE.finditer(copy_text):
        # Check if 'results vary' appears anywhere in the text (document-level)
        if not _RESULTS_VARY_RE.search(copy_text):
            findings.append({
                "check": "outcome_claim_no_disclosure",
                "severity": SEV_WARNING,
                "excerpt": copy_text[max(0, m.start() - 20): m.end() + 20].strip(),
                "why": (
                    "Outcome claim found without required 'results vary' disclosure. "
                    "Add 'Results may vary.' near any quantified claims."
                ),
            })
            break  # one finding per document is sufficient
    return findings


def _check_superlatives(copy_text: str, jurisdiction: str) -> list[dict]:
    """Check for superlatives/guarantees in jurisdictions requiring substantiation."""
    findings: list[dict] = []
    jur = (jurisdiction or "").strip().lower()
    if jur and jur not in _SUBSTANTIATION_REQUIRED_JURISDICTIONS:
        return findings  # skip for permissive/unknown jurisdictions

    for m in _SUPERLATIVE_RE.finditer(copy_text):
        findings.append({
            "check": "superlative_without_substantiation",
            "severity": SEV_WARNING,
            "excerpt": copy_text[max(0, m.start() - 20): m.end() + 20].strip(),
            "why": (
                f"Superlative or guarantee '{m.group()}' used in jurisdiction "
                f"'{jurisdiction or 'default'}' which requires substantiation. "
                "Either remove, add a substantiation footnote, or use qualified language."
            ),
        })
    return findings


def _check_forward_looking(copy_text: str) -> list[dict]:
    """Check for forward-looking statements without safe-harbor language."""
    findings: list[dict] = []
    if not _FORWARD_LOOKING_RE.search(copy_text):
        return findings
    if _SAFE_HARBOR_RE.search(copy_text):
        return findings  # safe-harbor present — pass
    findings.append({
        "check": "forward_looking_no_safe_harbor",
        "severity": SEV_INFO,
        "excerpt": "",
        "why": (
            "Forward-looking statements detected without safe-harbor disclaimer. "
            "Consider adding: 'These are forward-looking statements. Actual results "
            "may differ materially.'"
        ),
    })
    return findings


def _check_channel_rules(
    copy_text: str,
    channel: str,
    artifact_metadata: dict,
    rules_dir: str | None = None,
) -> list[dict]:
    """Enforce channel-specific rules from docs/templates/channel_rules/."""
    findings: list[dict] = []
    if not channel:
        return findings

    from general_beckman.posthook_handlers.channel_rules_loader import load_channel_rules
    rules = load_channel_rules(channel, rules_dir=rules_dir)
    if rules is None:
        logger.debug("copy_compliance: no rules file for channel %r — skipping", channel)
        return [
            {
                "check": "channel_rules_missing",
                "severity": SEV_INFO,
                "excerpt": "",
                "why": (
                    f"No channel-rules file found for '{channel}'. "
                    "Create docs/templates/channel_rules/{channel}.md to enforce rules."
                ),
            }
        ]

    # Split copy_text into title/body if metadata provides it
    title = artifact_metadata.get("title") or ""
    body = artifact_metadata.get("body") or copy_text

    # Length checks
    if rules.max_title_chars and title and len(title) > rules.max_title_chars:
        findings.append({
            "check": "channel_max_title_chars",
            "severity": SEV_WARNING,
            "excerpt": title[:80],
            "why": (
                f"Title is {len(title)} chars (limit {rules.max_title_chars}) "
                f"for channel '{channel}'."
            ),
        })
    if rules.max_body_chars and body and len(body) > rules.max_body_chars:
        findings.append({
            "check": "channel_max_body_chars",
            "severity": SEV_WARNING,
            "excerpt": body[:80],
            "why": (
                f"Body is {len(body)} chars (limit {rules.max_body_chars}) "
                f"for channel '{channel}'."
            ),
        })
    if rules.max_total_chars and (len(title) + len(body)) > rules.max_total_chars:
        findings.append({
            "check": "channel_max_total_chars",
            "severity": SEV_WARNING,
            "excerpt": "",
            "why": (
                f"Combined length {len(title) + len(body)} chars exceeds "
                f"limit {rules.max_total_chars} for channel '{channel}'."
            ),
        })

    # Banned-word checks
    for pattern_str in rules.banned_words:
        if not pattern_str:
            continue
        if pattern_str.startswith("/") and len(pattern_str) > 2:
            # /regex/ pattern
            raw_pat = pattern_str[1:]
            end = raw_pat.rfind("/")
            if end > 0:
                raw_pat = raw_pat[:end]
            try:
                rx = re.compile(raw_pat, re.IGNORECASE)
                m = rx.search(copy_text)
                if m:
                    findings.append({
                        "check": "channel_banned_word",
                        "severity": SEV_WARNING,
                        "excerpt": copy_text[max(0, m.start() - 10): m.end() + 10].strip(),
                        "why": (
                            f"Banned pattern '{pattern_str}' matched in channel '{channel}'."
                        ),
                    })
            except re.error:
                pass
        else:
            # Literal string match (case-insensitive)
            if re.search(re.escape(pattern_str), copy_text, re.IGNORECASE):
                findings.append({
                    "check": "channel_banned_word",
                    "severity": SEV_WARNING,
                    "excerpt": pattern_str,
                    "why": (
                        f"Banned phrase '{pattern_str}' found in copy for channel '{channel}'."
                    ),
                })

    # Required disclosures
    for disc in rules.required_disclosures:
        label = disc.get("label") or ""
        pattern_str = disc.get("pattern") or label
        if not pattern_str:
            continue
        if pattern_str.startswith("/") and len(pattern_str) > 2:
            raw_pat = pattern_str[1:]
            end = raw_pat.rfind("/")
            if end > 0:
                raw_pat = raw_pat[:end]
            try:
                rx = re.compile(raw_pat, re.IGNORECASE)
                if not rx.search(copy_text):
                    findings.append({
                        "check": "channel_missing_disclosure",
                        "severity": SEV_WARNING,
                        "excerpt": "",
                        "why": (
                            f"Required disclosure '{label}' not found in copy "
                            f"for channel '{channel}'."
                        ),
                    })
            except re.error:
                pass
        else:
            if not re.search(re.escape(pattern_str), copy_text, re.IGNORECASE):
                findings.append({
                    "check": "channel_missing_disclosure",
                    "severity": SEV_WARNING,
                    "excerpt": "",
                    "why": (
                        f"Required disclosure '{label}' not found in copy "
                        f"for channel '{channel}'."
                    ),
                })

    # Image requirement
    if rules.image_required:
        image_url = artifact_metadata.get("image_url") or ""
        if not image_url:
            findings.append({
                "check": "channel_image_required",
                "severity": SEV_WARNING,
                "excerpt": "",
                "why": (
                    f"Channel '{channel}' requires an image but none was provided "
                    "in artifact metadata (image_url)."
                ),
            })

    return findings


async def _check_privacy_mismatch_llm(
    copy_text: str,
    privacy_policy: str,
    task: dict,
    ctx: dict,
) -> list[dict]:
    """Semantic check: does marketing copy contradict the privacy policy?

    Routes through ``husam.run`` with a ``raw_dispatch`` llm_call spec (OVERHEAD lane).
    Returns a list of findings (may be empty on pass or LLM failure).

    Structured output contract (returned by the LLM):
    ::

        {
          "contradicts": "yes" | "no" | "unclear",
          "citation": "<sentence from copy that contradicts the policy, or empty string>"
        }
    """
    findings: list[dict] = []
    task_id = task.get("id", "?")
    mission_id = task.get("mission_id")

    # Truncate to avoid blowing the context budget
    copy_excerpt = copy_text[:2000]
    policy_excerpt = privacy_policy[:3000]

    from prompt_foundry import build_messages
    _msgs = build_messages("copy_compliance", {
        "copy_excerpt": copy_excerpt,
        "policy_excerpt": policy_excerpt,
    })
    # Original sends a single user message (no system) — preserve that structure.
    user_msg = _msgs[1]

    spec = {
        "title": f"copy_compliance privacy check (source #{task_id})",
        "description": "Marketing-copy vs privacy-policy contradiction check.",
        "agent_type": "classifier",
        "kind": "overhead",
        "priority": 1,
        "context": {
            "llm_call": {
                "raw_dispatch": True,
                "call_category": "overhead",
                "task": "classifier",
                "agent_type": "classifier",
                "difficulty": 2,
                "messages": [user_msg],
                "failures": [],
                "estimated_input_tokens": 800,
                "estimated_output_tokens": 200,
            },
        },
    }
    if mission_id is not None:
        spec["mission_id"] = mission_id

    try:
        import husam
        resp = await husam.run(spec)
        text_out = resp.get("content", "") if isinstance(resp, dict) else str(resp)
        if isinstance(text_out, str):
            text_out = text_out.strip()
            text_out = re.sub(r"^```(?:json)?\s*", "", text_out)
            text_out = re.sub(r"\s*```$", "", text_out)
        parsed = json.loads(text_out) if isinstance(text_out, str) and text_out else {}
    except Exception as exc:
        logger.warning(
            "copy_compliance: LLM privacy-check failed — skipped",
            task_id=task_id, error=str(exc),
        )
        findings.append({
            "check": "privacy_policy_contradiction",
            "severity": SEV_INFO,
            "excerpt": "",
            "why": (
                f"Privacy↔copy LLM check skipped due to error: {str(exc)[:200]}. "
                "Manual review recommended."
            ),
        })
        return findings

    contradicts = str(parsed.get("contradicts") or "unclear").lower()
    citation = str(parsed.get("citation") or "")
    if contradicts == "yes":
        findings.append({
            "check": "privacy_policy_contradiction",
            "severity": SEV_BLOCKER,
            "excerpt": citation[:300],
            "why": (
                "Marketing copy contradicts the generated privacy policy. "
                "The copy makes a claim that is inconsistent with the data "
                "practices disclosed in the privacy policy."
            ),
            "fix_suggestion": (
                "Remove or requalify the contradicting claim. "
                "Alternatively, update the privacy policy if the claim "
                "reflects actual product behaviour."
            ),
        })
    elif contradicts == "unclear":
        findings.append({
            "check": "privacy_policy_contradiction",
            "severity": SEV_INFO,
            "excerpt": citation[:300],
            "why": (
                "Potential inconsistency between marketing copy and privacy policy — "
                "could not determine definitively. Manual review recommended."
            ),
        })
    return findings


# ---------------------------------------------------------------------------
# Fix suggestions for blockers
# ---------------------------------------------------------------------------

def _make_fix_suggestion(finding: dict) -> str | None:
    """Return a short alternative phrasing for blocker findings."""
    check = finding.get("check", "")
    if check == "privacy_policy_contradiction":
        return finding.get("fix_suggestion") or (
            "Revise the claim to align with the privacy policy, or add a "
            "qualifying statement (e.g., 'when you opt in to analytics')."
        )
    return None


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle(task: dict, result: dict) -> dict:
    """copy_compliance_review posthook handler.

    Reads copy text from the artifact result/context, then runs 5 checks.
    Returns ``{"status": "ok"}`` on pass or ``{"status": "fail", ...}`` on
    violations.
    """
    task_id = task.get("id", "?")
    mission_id = task.get("mission_id")

    # ── Parse context ──────────────────────────────────────────────────────
    raw_ctx = task.get("context") or {}
    if isinstance(raw_ctx, str):
        try:
            ctx: dict = json.loads(raw_ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    else:
        ctx = dict(raw_ctx)

    jurisdiction: str = str(ctx.get("jurisdiction") or "").strip().lower()
    channel: str = str(ctx.get("channel") or "").strip()
    artifact_metadata: dict = ctx.get("artifact_metadata") or {}
    workspace_path: str = ctx.get("workspace_path") or ""

    logger.debug(
        "copy_compliance_review: starting",
        task_id=task_id,
        mission_id=mission_id,
        channel=channel or "(none)",
        jurisdiction=jurisdiction or "(none)",
    )

    # ── Extract copy text ──────────────────────────────────────────────────
    copy_text = _load_copy_text(task, ctx, result)
    if not copy_text:
        logger.info(
            "copy_compliance_review: no copy text found — skipping",
            task_id=task_id,
        )
        return {
            "status": "ok",
            "verdict": "skip",
            "reason": "no copy text found in artifact result or context",
            "findings": [],
        }

    all_findings: list[dict] = []

    # ── Check 1: Privacy-policy ↔ copy (LLM, OVERHEAD) ────────────────────
    privacy_policy = _load_privacy_policy(task, ctx)
    if privacy_policy:
        pp_findings = await _check_privacy_mismatch_llm(
            copy_text=copy_text,
            privacy_policy=privacy_policy,
            task=task,
            ctx=ctx,
        )
        all_findings.extend(pp_findings)
    else:
        logger.debug(
            "copy_compliance_review: no privacy_policy found — skipping semantic check",
            task_id=task_id,
        )
        all_findings.append({
            "check": "privacy_policy_contradiction",
            "severity": SEV_INFO,
            "excerpt": "",
            "why": (
                "No privacy_policy.md found for this mission — semantic check skipped. "
                "Generate a privacy policy via Z6 compliance templates to enable this check."
            ),
        })

    # ── Check 2: Outcome claims without disclosure ─────────────────────────
    all_findings.extend(_check_outcome_claims(copy_text))

    # ── Check 3: Superlative/guarantee violations ──────────────────────────
    all_findings.extend(_check_superlatives(copy_text, jurisdiction))

    # ── Check 4: Forward-looking statements without safe harbor ────────────
    all_findings.extend(_check_forward_looking(copy_text))

    # ── Check 5: Channel-specific rules ───────────────────────────────────
    all_findings.extend(
        _check_channel_rules(copy_text, channel, artifact_metadata)
    )

    # ── Aggregate ──────────────────────────────────────────────────────────
    blocker_count = sum(1 for f in all_findings if f.get("severity") == SEV_BLOCKER)
    warning_count = sum(1 for f in all_findings if f.get("severity") == SEV_WARNING)
    info_count = sum(1 for f in all_findings if f.get("severity") == SEV_INFO)

    # Fix suggestions for blockers
    fix_suggestions: dict[str, str] = {}
    for f in all_findings:
        if f.get("severity") == SEV_BLOCKER:
            suggestion = _make_fix_suggestion(f)
            if suggestion:
                fix_suggestions[f.get("check", "unknown")] = suggestion

    logger.info(
        "copy_compliance_review: complete",
        task_id=task_id,
        blockers=blocker_count,
        warnings=warning_count,
        infos=info_count,
    )

    if blocker_count > 0:
        return {
            "status": "fail",
            "verdict": "fail",
            "findings": all_findings,
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "fix_suggestions": fix_suggestions,
        }

    return {
        "status": "ok",
        "verdict": "pass",
        "findings": all_findings,
        "blocker_count": 0,
        "warning_count": warning_count,
        "info_count": info_count,
        "fix_suggestions": fix_suggestions,
    }
