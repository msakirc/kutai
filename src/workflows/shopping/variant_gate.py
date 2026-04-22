"""Pure-code filter + gate logic for variant disambiguation."""
from __future__ import annotations

from src.workflows.shopping.pipeline_v2 import ProductGroup

FILTER_AUTHENTICITY_MIN = 0.7
MAX_CLARIFY_OPTIONS = 5


def step_filter(groups: list[ProductGroup]) -> list[ProductGroup]:
    """Drop groups that aren't authentic, intent-matched products."""
    return [
        g for g in groups
        if g.product_type == "authentic_product"
        and g.matches_user_intent
        and g.authenticity_confidence >= FILTER_AUTHENTICITY_MIN
    ]


def step_variant_gate(
    survivors: list[ProductGroup],
    all_groups: list[ProductGroup],
) -> dict:
    """Decide how to proceed after filtering.

    Returns one of:
      {"kind": "chosen", "group": ProductGroup}
      {"kind": "clarify", "options": [{label, group_id, prominence}], "payloads": {gid: group}}
      {"kind": "escalation", "reason": "all_filtered"}
    """
    if not survivors:
        return {"kind": "escalation", "reason": "all_filtered"}

    by_variant: dict[tuple[str, str | None], list[ProductGroup]] = {}
    for g in survivors:
        by_variant.setdefault((g.base_model, g.variant), []).append(g)

    if len(by_variant) == 1:
        only_bucket = next(iter(by_variant.values()))
        chosen = max(only_bucket, key=lambda g: g.prominence)
        return {"kind": "chosen", "group": chosen}

    sorted_buckets = sorted(
        by_variant.items(),
        key=lambda kv: max(g.prominence for g in kv[1]),
        reverse=True,
    )
    gid_for = {id(g): i for i, g in enumerate(all_groups)}
    options: list[dict] = []
    payloads: dict[int, ProductGroup] = {}
    for _key, bucket in sorted_buckets[:MAX_CLARIFY_OPTIONS]:
        rep = max(bucket, key=lambda g: g.prominence)
        gid = gid_for.get(id(rep), -1)
        options.append({
            "label": rep.representative_title,
            "group_id": gid,
            "prominence": rep.prominence,
        })
        payloads[gid] = rep

    return {"kind": "clarify", "options": options, "payloads": payloads}
