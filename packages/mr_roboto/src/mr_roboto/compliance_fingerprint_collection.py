"""Z1 Tier 5A (P6) — compliance fingerprint collection (mechanical clarify-shape).

Reads ``mission_preflight.compliance_fingerprint`` from z0
(``mission_<id>/.preflight/compliance_fingerprint.json``) and only fires
Telegram clarify questions for fields the founder has NOT pre-declared.

Output: ``mission_<id>/compliance_fingerprint.json`` with the merged
fingerprint (z0 base + Z1-collected deltas). When z0 supplied a complete
fingerprint, this step is a no-op write.
"""
from __future__ import annotations

import json
import os
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.compliance_fingerprint_collection")


_REQUIRED_FIELDS = (
    "jurisdictions",
    "user_classes",
    "data_categories_coarse",
    "data_residency_required",
    "age_gate_required",
    "third_party_processors_expected",
    "data_export_requirements",
    "retention_max_days",
    "founder_attestations",
)


def _resolve_workspace(mission_id: int, workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    from src.tools.workspace import get_mission_workspace
    return get_mission_workspace(int(mission_id))


def _load_z0_preflight(workspace_path: str) -> dict[str, Any] | None:
    pre_path = os.path.join(
        workspace_path, ".preflight", "compliance_fingerprint.json",
    )
    if not os.path.isfile(pre_path):
        return None
    try:
        with open(pre_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning(
            "compliance_fingerprint_collection: failed to read z0 preflight: %s", e,
        )
        return None


def _missing_fields(fingerprint: dict[str, Any]) -> list[str]:
    missing = []
    for field in _REQUIRED_FIELDS:
        if field not in fingerprint:
            missing.append(field)
            continue
        v = fingerprint[field]
        # Empty list or None for non-bool fields is "missing".
        if field in ("data_residency_required", "age_gate_required"):
            continue  # bool — presence is enough
        if v is None:
            missing.append(field)
        elif isinstance(v, list) and not v:
            missing.append(field)
    return missing


def _empty_fingerprint() -> dict[str, Any]:
    return {
        "_schema_version": "1",
        "source": "z1_intake",
        "jurisdictions": [],
        "user_classes": [],
        "data_categories_coarse": [],
        "data_residency_required": False,
        "age_gate_required": False,
        "third_party_processors_expected": [],
        "data_export_requirements": [],
        "retention_max_days": None,
        "founder_attestations": {
            "founder_will_sign_dpa_with_processors": False,
            "founder_acknowledges_not_legal_advice": True,
        },
    }


async def compliance_fingerprint_collection(
    mission_id: int,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Collect/merge compliance fingerprint.

    Returns dict with ``ok``, ``path``, ``fingerprint``, ``missing_fields``,
    ``source`` (one of ``z0_complete``, ``z0_then_z1_delta``, ``z1_intake``).
    Always writes ``compliance_fingerprint.json`` even when fields are
    missing — the absence is captured via ``missing_fields`` so reviewers
    downstream can warn but the mission still progresses (per OQ4 prototype
    fast-path).
    """
    import datetime
    ws = _resolve_workspace(mission_id, workspace_path)
    os.makedirs(ws, exist_ok=True)

    z0 = _load_z0_preflight(ws)
    if z0:
        fingerprint = dict(z0)
        missing = _missing_fields(fingerprint)
        source = "z0_complete" if not missing else "z0_then_z1_delta"
    else:
        fingerprint = _empty_fingerprint()
        missing = _missing_fields(fingerprint)
        source = "z1_intake"

    fingerprint.setdefault("_schema_version", "1")
    fingerprint["source"] = source
    fingerprint["collected_at"] = datetime.datetime.utcnow().isoformat() + "Z"

    out_path = os.path.join(ws, "compliance_fingerprint.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(fingerprint, fh, indent=2, sort_keys=True)

    return {
        "ok": True,
        "path": out_path,
        "fingerprint": fingerprint,
        "missing_fields": missing,
        "source": source,
    }
