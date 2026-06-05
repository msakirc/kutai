"""Instruction<->schema field-drift lint (fix #4).

Fix #1 promotes ``artifact_schema`` to a deterministic gate at the grade
boundary. A *drifted* schema then becomes dangerous: when a step's instruction
tells the producer to emit fields its schema does not list, the gate and the
prose grader judge different contracts (#289737 step 2.10 — instruction named
6 fields, schema required 3). This module extracts the fields a step's
instruction *enumerates* and flags any that are absent from its schema.

The extractor keys off the low-noise signal only — a comma-separated run of
``snake_case`` tokens (a field enumeration), e.g. ``"a_field, b_field,
c_field"`` — not every passing prose mention. That deliberately misses prose
that names fields without enumerating them, in exchange for near-zero false
positives on input-artifact references and free text.
"""
from __future__ import annotations

import re

# A field-name-looking token: snake_case with at least one underscore. This is
# the low-noise signal we COLLECT — descriptive field names carry underscores,
# prose adjectives ("fast, cheap, reliable") do not, so an underscore filter
# keeps false positives near zero.
_SNAKE = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+"
# A run spans generic comma-separated word tokens so a mixed enumeration like
# "pricing_model, tiers, free_tier_strategy" is recognised as ONE list (a
# single-word item like ``tiers`` must not split the run); we then collect only
# the snake_case tokens from inside it.
_WORD = r"[a-z][a-z0-9_]*"
_RUN = re.compile(
    rf"{_WORD}(?:\s*,\s*(?:and\s+)?{_WORD})+"
)
_TOKEN = re.compile(_SNAKE)


def instruction_declared_fields(instruction: str) -> set[str]:
    """Return the set of fields the instruction *enumerates* (comma-run)."""
    if not instruction:
        return set()
    out: set[str] = set()
    for run in _RUN.findall(instruction):
        out.update(_TOKEN.findall(run))
    return out


def _collect_rule_field_names(rule: dict, names: set[str]) -> None:
    """Recurse a single dialect rule, collecting every declared field name.

    The schema_dialect (and the fix #1 gate) enforce nested fields, so the lint
    must descend into ``fields`` / ``items`` or it false-flags sub-field names.
    """
    if not isinstance(rule, dict):
        return
    names.update(rule.get("required_fields") or [])
    names.update(rule.get("item_fields") or [])
    fields = rule.get("fields")
    if isinstance(fields, dict):
        names.update(fields.keys())
        for sub in fields.values():
            _collect_rule_field_names(sub, names)
    items = rule.get("items")
    if isinstance(items, dict):
        _collect_rule_field_names(items, names)


def schema_field_names(schema: dict) -> set[str]:
    """Union of every field name an artifact_schema declares (recursive).

    Covers the legacy (``required_fields`` / ``item_fields``) and canonical
    (nested ``fields`` / ``items``) dialect forms across all artifact entries.
    """
    names: set[str] = set()
    if not isinstance(schema, dict):
        return names
    for rules in schema.values():
        if not isinstance(rules, dict):
            continue  # skip non-artifact entries (e.g. max_output_chars)
        _collect_rule_field_names(rules, names)
    return names


def lint_step_field_drift(step: dict) -> set[str]:
    """Return fields the step's instruction enumerates but its schema omits.

    Empty when the step has no ``artifact_schema`` (nothing to enforce) or when
    instruction and schema agree.
    """
    schema = step.get("artifact_schema")
    if not isinstance(schema, dict) or not schema:
        return set()
    declared = instruction_declared_fields(step.get("instruction", ""))
    # Upstream input artifacts and the step's own output artifact names are
    # references, not missing output fields — never count them as drift.
    artifact_refs: set[str] = set()
    for key in ("input_artifacts", "output_artifacts"):
        for a in step.get(key) or []:
            if isinstance(a, str):
                artifact_refs.add(a)
    return declared - schema_field_names(schema) - artifact_refs
