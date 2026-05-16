"""Manifest YAML parsing, validation, and canonical-name rules.

The recon found that on-disk SKILL.md files carry NO yalayut format — adapters
synthesize Manifest objects. parse_manifest_yaml here is for our OWN authored
seed manifests (packages/yalayut/seed/manifests/*.yaml).
"""
from __future__ import annotations

import yaml

from yalayut.contracts import Manifest

VALID_ARTIFACT_TYPES = {"skill", "api", "mcp"}
VALID_SKILL_KINDS = {
    "internal_hint", "prompt_skill", "shell_recipe", "procedure",
    "agent_config",
}


def parse_manifest_yaml(text: str) -> Manifest:
    """Parse a yalayut-format YAML manifest into a Manifest dataclass."""
    raw = yaml.safe_load(text) or {}
    return Manifest(
        name=raw.get("name", ""),
        name_original=raw.get("name_original", raw.get("name", "")),
        version=str(raw.get("version", "")),
        artifact_type=raw.get("artifact_type", ""),
        kind=raw.get("kind"),
        source=raw.get("source", ""),
        owner=raw.get("owner"),
        license=raw.get("license"),
        mechanizable=bool(raw.get("mechanizable", False)),
        model_hint=raw.get("model_hint"),
        applies_to=raw.get("applies_to", "execution"),
        intent_keywords=list(raw.get("intent_keywords", []) or []),
        inputs_schema=dict(raw.get("inputs_schema", {}) or {}),
        invocation=dict(raw.get("invocation", {}) or {}),
        artifacts=list(raw.get("artifacts", []) or []),
        disabled_imports_check=bool(raw.get("disabled_imports_check", True)),
        mcp=dict(raw.get("mcp", {}) or {}),
        auth_env=raw.get("auth_env"),
    )


def validate_manifest(m: Manifest) -> list[str]:
    """Return a list of human-readable validation errors. [] means valid."""
    errs: list[str] = []
    if not m.name:
        errs.append("missing required field: name")
    if not m.version:
        errs.append("missing required field: version")
    if not m.artifact_type:
        errs.append("missing required field: artifact_type")
    elif m.artifact_type not in VALID_ARTIFACT_TYPES:
        errs.append(
            f"invalid artifact_type {m.artifact_type!r}; "
            f"expected one of {sorted(VALID_ARTIFACT_TYPES)}"
        )
    if m.artifact_type == "skill" and m.kind and m.kind not in VALID_SKILL_KINDS:
        errs.append(
            f"invalid skill kind {m.kind!r}; "
            f"expected one of {sorted(VALID_SKILL_KINDS)}"
        )
    if m.applies_to not in {"execution", "grading"}:
        errs.append(f"invalid applies_to {m.applies_to!r}")
    return errs


def canonical_name(source_slug: str, original: str) -> str:
    """Build the canonical '<source-slug>-<original>' name with recon's
    failure-mode rules:
      - cookiecutter-* templates collapse to cc-* (drop both prefixes)
      - drop the source prefix when the original already starts with it
        (matlab/matlab-live-script -> matlab-live-script, not
        matlab-matlab-live-script)
    """
    original = original.strip().lower().replace(" ", "-")
    source_slug = source_slug.strip().lower()
    if original.startswith("cookiecutter-"):
        return "cc-" + original[len("cookiecutter-"):]
    if original.startswith(source_slug + "-") or original == source_slug:
        return original
    return f"{source_slug}-{original}"
