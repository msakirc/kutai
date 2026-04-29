"""Label step — LLM taxonomy per ProductGroup."""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger
from src.workflows.shopping.pipeline_v2 import Candidate, ProductGroup, _strip_json_fences
from src.workflows.shopping.prompts_v2 import LABEL_PROMPT

logger = get_logger("workflows.shopping.labels")

VALID_PRODUCT_TYPES = {
    "authentic_product", "accessory", "replacement_part",
    "knockoff", "refurbished", "unknown",
}


async def _label_llm_call(prompt: str) -> dict:
    """Dispatch the label prompt. Split out so tests can patch this one function."""
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    dispatcher = get_dispatcher()
    return await dispatcher.request(
        category=CallCategory.MAIN_WORK,
        task="shopping_labeler",
        agent_type="shopping_pipeline_v2",
        difficulty=4,
        messages=[
            {"role": "system", "content": "You output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )


def _fallback_labels(groups: list[ProductGroup]) -> list[ProductGroup]:
    for g in groups:
        g.product_type = "authentic_product"
        g.base_model = g.representative_title
        g.variant = None
        g.authenticity_confidence = 0.8   # above 0.7 filter threshold
        g.matches_user_intent = True
    return groups


async def step_label(
    groups: list[ProductGroup],
    candidates: list[Candidate],
    query: str,
) -> list[ProductGroup]:
    """Label every group with taxonomy via one LLM call. Mutates groups in place."""
    if not groups:
        return groups

    view = []
    for i, g in enumerate(groups):
        member_cands = [candidates[m] for m in g.member_indices if 0 <= m < len(candidates)]
        category = next((c.category_path for c in member_cands if c.category_path), "")
        view.append({
            "group_id": i,
            "title": g.representative_title,
            "category_path": category,
            "member_count": len(g.member_indices),
        })

    prompt = LABEL_PROMPT.format(
        query=query,
        groups_json=json.dumps(view, ensure_ascii=False),
    )

    try:
        resp = await _label_llm_call(prompt)
    except Exception as exc:
        logger.warning("label LLM failed, using fallback: %s", exc)
        return _fallback_labels(groups)

    content = _strip_json_fences(str(resp.get("content", "")).strip())
    try:
        parsed = json.loads(content)
        entries = parsed.get("groups", [])
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("label LLM parse failed: %s", exc)
        return _fallback_labels(groups)

    by_id: dict[int, dict] = {
        e["group_id"]: e for e in entries
        if isinstance(e, dict) and "group_id" in e
    }
    for i, g in enumerate(groups):
        e = by_id.get(i)
        if not e:
            g.product_type = "authentic_product"
            g.matches_user_intent = True
            g.authenticity_confidence = 0.8   # above 0.7 filter threshold
            g.base_model = g.representative_title
            continue
        pt = str(e.get("product_type", "unknown"))
        g.product_type = pt if pt in VALID_PRODUCT_TYPES else "unknown"
        g.base_model = str(e.get("base_model", g.representative_title))
        variant = e.get("variant")
        g.variant = str(variant) if variant else None
        # LLM-emitted canonical line slug; sanitize to [a-z0-9-]
        import re as _re
        raw_lid = str(e.get("line_id", "")).lower().strip()
        g.line_id = _re.sub(r"[^a-z0-9-]+", "-", raw_lid).strip("-")[:80]
        try:
            g.authenticity_confidence = float(e.get("authenticity_confidence", 0.5))
        except (TypeError, ValueError):
            g.authenticity_confidence = 0.5
        g.matches_user_intent = bool(e.get("matches_user_intent", True))

    return groups
