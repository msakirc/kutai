"""Z7 T3E — B6: crisis/draft_holding mr_roboto verb.

LLM-bound: reads the tier-specific playbook + event context, outputs
2+ holding-statement variants for founder selection.

The draft is returned as variants — it is NOT posted anywhere automatically.
Founder selects and edits via founder_action card.

Payload::

    {
        "product_id": "prod-abc",     # required
        "event_id":   42,             # required
        "tier":       2,              # 1-4, required
        "summary":    "...",          # optional; fetched from crisis_events if omitted
    }

Returns::

    {
        "status":   "ok",
        "variants": ["...", "...", ...],
        "tier":     2,
        "event_id": 42,
    }
"""
from __future__ import annotations

import json as _json
import os
import re

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.crisis_draft_holding")

# Matches "Variant A:", "Variant B:", "Variant 1:", etc. (case-insensitive).
# Used by the text-split heuristic to strip the label prefix and keep the content.
_VARIANT_PREFIX_RE = re.compile(r'^Variant\s+\w+\s*:\s*', re.IGNORECASE)


def _playbook_path(tier: int) -> str:
    """Return the absolute path to the crisis_comms_tier{N}.md playbook.

    Resolves relative to the project root using WORKSPACE_DIR or CWD fallback.
    """
    # Try WORKSPACE_DIR env var first (set in production)
    workspace_dir = os.environ.get("WORKSPACE_DIR", "")
    if workspace_dir:
        # WORKSPACE_DIR is the missions/ workspace subdir; go up one level to repo root
        repo_root = os.path.normpath(os.path.join(workspace_dir, ".."))
        candidate = os.path.join(repo_root, "playbooks", f"crisis_comms_tier{tier}.md")
        if os.path.isfile(candidate):
            return candidate

    # Fallback: traverse up from this file
    # packages/mr_roboto/src/mr_roboto/__file__ → up 4 dirs → repo root
    here = os.path.dirname(os.path.abspath(__file__))
    # mr_roboto/ → src/ → mr_roboto (pkg) → packages/ → repo_root
    repo_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
    candidate = os.path.join(repo_root, "playbooks", f"crisis_comms_tier{tier}.md")
    if os.path.isfile(candidate):
        return candidate

    # Last resort: current working directory (test / dev context)
    cwd_candidate = os.path.join(os.getcwd(), "playbooks", f"crisis_comms_tier{tier}.md")
    return cwd_candidate


def _read_playbook(tier: int) -> str:
    """Read the playbook for *tier*. Returns empty string on failure."""
    path = _playbook_path(tier)
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except Exception as exc:
        logger.warning(
            "crisis_draft_holding: could not read playbook",
            tier=tier,
            path=path,
            error=str(exc),
        )
        return ""


def parse_variants(raw_str: str) -> list[str]:
    """Parse the LLM holding-statement response into a list of variants."""
    raw_str = (raw_str or "").strip()
    m = re.search(r'\[.*?\]', raw_str, re.DOTALL)
    if m:
        try:
            parsed = _json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except Exception:
            pass
    variants = []
    for line in raw_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = _VARIANT_PREFIX_RE.sub("", line).strip()
        if len(line) > 20:
            variants.append(line)
    return variants[:2] if variants else []


def canned_variants(tier: int, product_id: str) -> list[str]:
    """Deterministic fallback when the LLM produced nothing."""
    tier_labels = {1: "this situation", 2: "an ongoing service issue",
                   3: "a security incident", 4: "a critical situation"}
    context = tier_labels.get(tier, "this issue")
    return [
        (f"We are aware of {context} affecting {product_id}. Our team is actively "
         "working on a resolution. We will provide an update shortly."),
        (f"We know about {context} and are on it. Thank you for your patience — "
         "more details coming soon."),
    ]


# ---------------------------------------------------------------------------
# Public run() — kept for backward compat; NOT on the live path after SP4b.
# Returns canned output only (no LLM call, no await_inline).
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """Execute crisis/draft_holding (canned fallback only; live path uses CPS producer).

    Returns holding-statement variants for founder selection.
    """
    product_id = payload.get("product_id") or ""
    event_id = payload.get("event_id")
    tier = int(payload.get("tier") or 1)

    if not event_id:
        return {"status": "error", "error": "event_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    variants = canned_variants(tier, product_id)

    logger.info(
        "crisis_draft_holding: canned variants ready",
        event_id=event_id,
        product_id=product_id,
        tier=tier,
        variant_count=len(variants),
    )

    return {
        "status": "ok",
        "variants": variants,
        "tier": tier,
        "event_id": event_id,
        "product_id": product_id,
    }
