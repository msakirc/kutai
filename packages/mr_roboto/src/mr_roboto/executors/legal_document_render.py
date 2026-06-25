"""Z6 T4A — render legal documents from compliance_overlay.

Mechanical executor for step 12.1 `legal_documents`. Reads the
mission's ``compliance_overlay.json`` (produced by Z1 step 1.11a),
walks ``required_documents[]`` and renders Markdown drafts for the
three writer-facing docs (terms_of_service / privacy_policy /
cookie_policy) via :func:`src.tools.compliance_templates.compliance_template_render`.

Doc-type mapping (overlay -> output artifact):

- ``tos``               → terms_of_service
- ``terms_of_service``  → terms_of_service
- ``privacy_policy``    → privacy_policy
- ``cookie_banner``     → cookie_policy
- ``cookie_policy``     → cookie_policy

Rendered files land at
``mission_<id>/.compliance/legal/<output_artifact>.md`` and the same
relative path is registered as the named artifact via
``register_artifact``. Existing 1.11a renders (which already wrote
``mission_<id>/.compliance/legal/<doc>.md`` for the rendered overlay
templates) are overwritten with the canonical names so step 12.1b can
read them by output-artifact name.

Returns ``{ok, rendered: [paths], skipped: [...], errors: [...]}``.

Failure modes
-------------
- Overlay missing or unparseable → ``ok=False, reason="overlay_missing"``.
- ``required_documents[]`` empty → ``ok=True, skipped=["all"]``.
- A given doc_type not in the writer mapping → entry in ``skipped``.
- Template render returns ``ok=False`` for one doc → entry in ``errors``
  but other docs still attempted. ``ok`` is True if at least one
  writer-facing doc rendered.
"""
from __future__ import annotations

import json
import os
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.legal_document_render")


# overlay doc_type -> (output_artifact_name, template_doc_type)
# We accept both legacy names ("tos") and the canonical ones used in
# i2p_v3 ("terms_of_service"). Templates live under the names actually
# present on disk (tos.md.j2 for ToS, privacy_policy.md.j2, cookie_banner.md.j2).
_DOC_TYPE_TO_ARTIFACT: dict[str, tuple[str, str]] = {
    "tos": ("terms_of_service", "tos"),
    "terms_of_service": ("terms_of_service", "tos"),
    "privacy_policy": ("privacy_policy", "privacy_policy"),
    "cookie_banner": ("cookie_policy", "cookie_banner"),
    "cookie_policy": ("cookie_policy", "cookie_banner"),
}


def _resolve_workspace(mission_id: int, workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    from src.tools.workspace import get_mission_workspace
    return get_mission_workspace(int(mission_id))


def _load_overlay(
    workspace_path: str,
    overlay_obj: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if overlay_obj is not None:
        return overlay_obj
    candidates = [
        os.path.join(workspace_path, "compliance_overlay.json"),
        os.path.join(workspace_path, ".compliance", "overlay.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as e:
                logger.warning(
                    "legal_document_render: overlay read failed %s: %s", path, e,
                )
    return None


def _extract_fingerprint(overlay: dict[str, Any]) -> dict[str, Any]:
    """Build a render-time fingerprint from the overlay.

    The overlay is the post-1.11a artifact. We forward any obvious
    fingerprint-shaped fields and let the template's Jinja `default()`
    guards handle anything missing.
    """
    fp = dict(overlay.get("fingerprint") or {})
    # Common conveniences: pull jurisdictions / data categories from the
    # overlay top level when not nested under a fingerprint sub-object.
    if "jurisdictions" not in fp:
        fp["jurisdictions"] = list(overlay.get("jurisdictions") or [])
    if "data_categories" not in fp:
        fp["data_categories"] = list(overlay.get("data_categories") or [])
    if "data_categories_coarse" not in fp:
        fp["data_categories_coarse"] = list(
            overlay.get("data_categories_coarse")
            or fp.get("data_categories")
            or []
        )
    if "user_classes" not in fp:
        fp["user_classes"] = list(overlay.get("user_classes") or [])
    if "third_parties" not in fp:
        fp["third_parties"] = list(
            overlay.get("third_parties")
            or overlay.get("third_party_processors_expected")
            or []
        )
    if "third_party_processors_expected" not in fp:
        fp["third_party_processors_expected"] = list(
            fp.get("third_parties") or []
        )
    return fp


async def legal_document_render(
    mission_id: int,
    workspace_path: str | None = None,
    overlay_obj: dict[str, Any] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Render legal docs from compliance_overlay.required_documents[].

    See module docstring for return shape and failure modes.
    """
    ws = _resolve_workspace(int(mission_id), workspace_path)
    overlay = _load_overlay(ws, overlay_obj)
    if overlay is None:
        return {
            "ok": False,
            "reason": "overlay_missing",
            "rendered": [],
            "skipped": [],
            "errors": [],
        }

    required = list(overlay.get("required_documents") or [])
    if not required:
        return {
            "ok": True,
            "rendered": [],
            "skipped": ["all"],
            "errors": [],
            "reason": "no_required_documents",
        }

    fingerprint = _extract_fingerprint(overlay)

    legal_dir = os.path.join(ws, ".compliance", "legal")
    os.makedirs(legal_dir, exist_ok=True)

    rendered_paths: list[str] = []
    skipped: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    artifacts_to_register: dict[str, str] = {}

    from src.tools.compliance_templates import compliance_template_render

    seen_artifacts: set[str] = set()

    for entry in required:
        doc_type_raw = (entry.get("doc_type") or "").strip()
        if not doc_type_raw:
            continue
        mapping = _DOC_TYPE_TO_ARTIFACT.get(doc_type_raw)
        if mapping is None:
            skipped.append({
                "doc_type": doc_type_raw,
                "reason": "not_writer_facing",
            })
            continue
        artifact_name, template_doc_type = mapping
        if artifact_name in seen_artifacts:
            # Already rendered this artifact for an earlier required-docs row.
            continue

        # Optional per-doc overrides on the required_documents row.
        entry_lang = (entry.get("lang") or lang or "en").strip() or "en"
        entry_fp = dict(fingerprint)
        # Allow per-row jurisdiction override (overlay may declare per-doc).
        per_doc_juris = entry.get("jurisdiction")
        if per_doc_juris and "jurisdictions" not in entry.get("fingerprint", {}):
            entry_fp["jurisdictions"] = [per_doc_juris]

        res = compliance_template_render(
            fingerprint=entry_fp,
            doc_type=template_doc_type,
            lang=entry_lang,
        )
        if not res.get("ok"):
            errors.append({
                "doc_type": doc_type_raw,
                "artifact": artifact_name,
                "error": str(res.get("error") or "render_failed"),
            })
            continue

        out_path = os.path.join(legal_dir, f"{artifact_name}.md")
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(res.get("rendered") or "")
        except OSError as e:
            errors.append({
                "doc_type": doc_type_raw,
                "artifact": artifact_name,
                "error": f"write_failed: {e}",
            })
            continue

        rendered_paths.append(out_path)
        artifacts_to_register[artifact_name] = out_path
        seen_artifacts.add(artifact_name)

    # Best-effort: register each rendered file as a named artifact so 12.1b
    # can resolve it. Failure is logged but does not flip ok. Z6 T7D
    # swapped the legacy ``src.infra.db.register_artifact`` (which never
    # existed) for the dedicated helper, now ``mr_roboto.artifacts_register``.
    if artifacts_to_register:
        try:
            from mr_roboto.artifacts_register import register_artifact
        except ImportError:  # pragma: no cover
            register_artifact = None  # type: ignore[assignment]
        if register_artifact is not None:
            for name, path in artifacts_to_register.items():
                try:
                    await register_artifact(
                        mission_id=int(mission_id),
                        artifact_name=name,
                        artifact_path=path,
                        domain_keywords=["compliance", "legal", name],
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "legal_document_render: register_artifact %s failed: %s",
                        name, e,
                    )

    ok = bool(rendered_paths)
    return {
        "ok": ok,
        "rendered": rendered_paths,
        "artifacts": artifacts_to_register,
        "skipped": skipped,
        "errors": errors,
    }
