"""KutAI artifact_schema dialect — single source of truth.

Canonical shape (recursive):

    {
      "type": "object | array | string | number | boolean | markdown",
      "fields":   { "<name>": <rule>, ... },   # object
      "items":    <rule>,                       # array
      "required_sections": ["..."],             # markdown
      "min_items": int, "max_items": int,       # array
      "min_keys":  int,                         # variable-key object
      "min_length": int,                        # string / markdown
      "optional":  bool                         # default false (required)
    }

All keys in ``fields`` are required unless marked ``optional: true``. Empty
``fields`` map = "object with any keys; use ``min_keys`` if non-empty
required". No ``items`` on array = "any array contents".

This module is the dialect's home. Validator, translator, example
generator, and per-artifact checklist all consume the four helpers below
so a single change here propagates everywhere.

Public API:
    validate_value(rule, value, path)   -> error str or None
    translate_rule(rule)                 -> JSON Schema dict or None
    make_example(rule)                   -> python value (json-dumpable)
    iter_required_paths(rule, prefix)    -> yields (path, rule) leaves
"""
from __future__ import annotations

from typing import Any, Iterator, Optional


# ── Old-form normalization ──────────────────────────────────────────────
#
# Pre-E1 the dialect was flat: ``required_fields: [str]`` for objects and
# ``item_fields: [str]`` for arrays. Migrating every schema across the
# workflow tree at once is high-risk; instead, every helper below entrypoints
# through ``_normalize_rule`` which reshapes legacy form into canonical:
#
#   {required_fields: [a, b]}        ->  {fields: {a: {type: "string"},
#                                                  b: {type: "string"}}}
#   {item_fields:     [a, b]}        ->  {items:  {type: "object",
#                                                  fields: {a: ..., b: ...}}}
#
# Migrated schemas use the canonical form and pass through unchanged. The
# normalizer is idempotent: running it on a canonical schema is a no-op.

def _normalize_rule(rule: Any) -> Any:
    if not isinstance(rule, dict):
        return rule
    rtype = rule.get("type")

    if rtype == "object":
        # Already canonical?
        if "fields" in rule:
            normalized = dict(rule)
            normalized["fields"] = {
                k: _normalize_rule(v) for k, v in (rule.get("fields") or {}).items()
            }
            return normalized
        # Legacy shape with required_fields list of strings. Pre-E1 had no
        # per-field type info — defaulting to ``string`` was wrong (rejected
        # legitimate arrays / numbers). Use empty rule = presence-only check.
        # ``must_be_true`` is the legacy hook for verification flags
        # (``dependencies_installed``, ``health_check_passed``, etc.) — fields
        # listed here become ``{type: boolean, equals: true}`` instead of the
        # default presence-only check, so the validator rejects ``false``
        # self-reports without a full schema migration.
        legacy = rule.get("required_fields")
        if isinstance(legacy, list):
            must_true = set(rule.get("must_be_true") or [])
            normalized = {
                k: v for k, v in rule.items()
                if k not in ("required_fields", "must_be_true")
            }
            built: dict[str, dict] = {}
            for f in legacy:
                if not isinstance(f, str):
                    continue
                built[f] = (
                    {"type": "boolean", "equals": True}
                    if f in must_true else {}
                )
            normalized["fields"] = built
            return normalized
        return rule

    if rtype == "array":
        # Canonical?
        if "items" in rule:
            normalized = dict(rule)
            normalized["items"] = _normalize_rule(rule["items"])
            return normalized
        # Legacy item_fields → wrap as object items with presence-only sub
        # rules (no string default — see comment above).
        legacy = rule.get("item_fields")
        if isinstance(legacy, list) and legacy:
            normalized = {k: v for k, v in rule.items() if k != "item_fields"}
            normalized["items"] = {
                "type": "object",
                "fields": {
                    f: {} for f in legacy if isinstance(f, str)
                },
            }
            return normalized
        return rule

    return rule


# ── Empty-placeholder detection ─────────────────────────────────────────

def is_empty_required_value(val: Any) -> bool:
    """Reject placeholders that satisfy ``in`` but carry no real content.

    Catches the constrained-decoder pathology where every required field
    gets ``{}``/``[]``/``""`` to satisfy presence (mission 46 task 2964).
    """
    if val is None:
        return True
    if isinstance(val, str):
        return not val.strip() or val.strip() == "..."
    if isinstance(val, (dict, list)):
        return len(val) == 0
    return False


# ── Validation ──────────────────────────────────────────────────────────

def validate_value(rule: dict, value: Any, path: str = "") -> Optional[str]:
    """Validate ``value`` against dialect ``rule``. Returns error or None.

    Recurses through nested objects/arrays. Reports the failing path so
    retry feedback can pinpoint the breakage (e.g. ``info.title``,
    ``sprint_plans[0].tasks[2].task_id``).
    """
    if not isinstance(rule, dict):
        return f"{path or '<root>'}: rule is not a dict"
    rule = _normalize_rule(rule)
    rtype = rule.get("type")

    # Untyped rule = presence-only check (legacy fields without type info).
    # ``is_empty_required_value`` upstream catches empty placeholders; nothing
    # left to validate at this level.
    if rtype is None:
        return None

    if rtype == "object":
        if not isinstance(value, dict):
            return f"{path or '<root>'}: expected object, got {type(value).__name__}"
        min_keys = int(rule.get("min_keys") or 0)
        if min_keys and len(value) < min_keys:
            return (f"{path or '<root>'}: object has {len(value)} keys, "
                    f"need >= {min_keys}")
        fields = rule.get("fields") or {}
        for fname, frule in fields.items():
            if not isinstance(frule, dict):
                continue
            if frule.get("optional"):
                continue
            field_path = f"{path}.{fname}" if path else fname
            if fname not in value:
                return f"{field_path}: missing required field"
            fvalue = value[fname]
            if is_empty_required_value(fvalue):
                return f"{field_path}: empty placeholder value"
            sub = validate_value(frule, fvalue, field_path)
            if sub:
                return sub
        return None

    if rtype == "array":
        if not isinstance(value, list):
            return f"{path or '<root>'}: expected array, got {type(value).__name__}"
        min_items = int(rule.get("min_items") or 0)
        if len(value) < min_items:
            return (f"{path or '<root>'}: {len(value)} items, "
                    f"need >= {min_items}")
        max_items = rule.get("max_items")
        if isinstance(max_items, int) and len(value) > max_items:
            return (f"{path or '<root>'}: {len(value)} items, "
                    f"max {max_items}")
        items_rule = rule.get("items")
        if isinstance(items_rule, dict):
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                if is_empty_required_value(item):
                    return f"{item_path}: empty placeholder value"
                sub = validate_value(items_rule, item, item_path)
                if sub:
                    return sub
        return None

    if rtype == "string":
        if not isinstance(value, str):
            return f"{path or '<root>'}: expected string, got {type(value).__name__}"
        min_length = int(rule.get("min_length") or 1)
        if len(value.strip()) < min_length:
            return f"{path or '<root>'}: string too short (min {min_length})"
        return None

    if rtype == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{path or '<root>'}: expected number, got {type(value).__name__}"
        return None

    if rtype == "boolean":
        if not isinstance(value, bool):
            return f"{path or '<root>'}: expected boolean, got {type(value).__name__}"
        if "equals" in rule and value != rule["equals"]:
            # Didactic error: small models tend to flip the bool on retry
            # without actually doing the verification. Spell out the
            # requirement so the retry prompt steers toward real work.
            if rule["equals"] is True:
                return (
                    f"{path or '<root>'}: must be true — this is a "
                    f"VERIFICATION flag. Do NOT just flip the value. "
                    f"Actually run the check (start the server, curl the "
                    f"endpoint, run the test, etc.) and only emit true if "
                    f"the check succeeds. If you can't verify, the step "
                    f"genuinely failed; report the blocker, don't fake it."
                )
            return f"{path or '<root>'}: expected {rule['equals']!r}, got {value!r}"
        return None

    if rtype == "markdown":
        # Markdown validation lives in hooks.py — section-header matching
        # has its own logic. This dialect helper only handles structured
        # types. Caller must dispatch.
        return None

    return f"{path or '<root>'}: unknown type '{rtype}'"


# ── JSON Schema translation ─────────────────────────────────────────────

# Permissive leaf type for fields where the dialect has no further hint.
# Bare ``{}`` triggers OpenAI strict + llama.cpp into emitting empty
# objects; listing concrete types tells the decoder to sample real
# content. See json_schema_translator.py for the original incident.
_LEAF_VALUE_SCHEMA = {
    "type": ["string", "number", "boolean", "array", "object", "null"],
}


def translate_rule(rule: dict) -> Optional[dict]:
    """Translate a dialect rule into JSON Schema (for constrained decoding).

    Returns ``None`` for unconstrainable types (markdown, unknown).
    """
    if not isinstance(rule, dict):
        return None
    rule = _normalize_rule(rule)
    rtype = rule.get("type")

    if rtype == "object":
        # Variable-key object: ``fields`` key absent (vs present-but-empty,
        # which is "strict empty"). ``min_keys`` lives here as the
        # constraint hint for the decoder.
        if "fields" not in rule:
            return {"type": "object"}
        fields = rule.get("fields") or {}
        properties: dict[str, dict] = {}
        required: list[str] = []
        for fname, frule in fields.items():
            if not isinstance(frule, dict):
                continue
            sub = translate_rule(frule)
            properties[fname] = sub if sub is not None else dict(_LEAF_VALUE_SCHEMA)
            if not frule.get("optional"):
                required.append(fname)
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }

    if rtype == "array":
        out: dict[str, Any] = {"type": "array"}
        min_items = rule.get("min_items")
        if isinstance(min_items, int) and min_items > 0:
            out["minItems"] = min_items
        items_rule = rule.get("items")
        if isinstance(items_rule, dict):
            sub = translate_rule(items_rule)
            if sub is not None:
                out["items"] = sub
        return out

    if rtype == "string":
        return {"type": "string"}
    if rtype == "number":
        return {"type": "number"}
    if rtype == "boolean":
        # NOTE: ``equals`` is intentionally NOT translated to JSON Schema
        # ``const``. Forcing the token at decode time made small models
        # fabricate ``true`` for verification flags they never actually
        # checked (mission 57 task 4458 2026-04-30: agent emitted
        # ``health_check_verified: true`` with zero curl in audit_log).
        # ``equals`` stays a post-emit validator constraint only — the
        # model is free to emit ``false``, validator rejects it, retry
        # prompt feedback steers the agent toward real verification.
        return {"type": "boolean"}

    # markdown / unknown — not constrainable.
    return None


# ── Example generation ──────────────────────────────────────────────────

def make_example(rule: dict) -> Any:
    """Build a JSON-dumpable example value matching ``rule``.

    Includes all REQUIRED fields populated with type-appropriate
    placeholders. Skips optional fields (caller's example shouldn't
    suggest the model produce optional content unless it has reason to).
    """
    if not isinstance(rule, dict):
        return "..."
    rule = _normalize_rule(rule)
    rtype = rule.get("type")

    if rtype == "object":
        fields = rule.get("fields") or {}
        if not fields:
            min_keys = int(rule.get("min_keys") or 0)
            if min_keys:
                return {"<key>": "<value>"}
            return {}
        out: dict[str, Any] = {}
        for fname, frule in fields.items():
            if isinstance(frule, dict) and frule.get("optional"):
                continue
            out[fname] = make_example(frule) if isinstance(frule, dict) else "..."
        return out

    if rtype == "array":
        items_rule = rule.get("items")
        if isinstance(items_rule, dict):
            return [make_example(items_rule)]
        return ["..."]

    if rtype == "string":
        return "..."
    if rtype == "number":
        return 0
    if rtype == "boolean":
        if "equals" in rule and isinstance(rule["equals"], bool):
            return rule["equals"]
        return False
    if rtype == "markdown":
        return "..."
    return "..."


# ── Required-path enumeration ───────────────────────────────────────────

def iter_required_paths(rule: dict, prefix: str = "") -> Iterator[tuple[str, dict]]:
    """Yield ``(path, rule)`` for every required position in the tree.

    Used by the per-artifact retry checklist so the checklist text
    mirrors the validator's traversal. Arrays surface as
    ``<prefix>[]`` plus the items-rule's recursion (representative of
    every element).
    """
    if not isinstance(rule, dict):
        return
    rule = _normalize_rule(rule)
    rtype = rule.get("type")

    if rtype == "object":
        fields = rule.get("fields") or {}
        if not fields:
            yield (prefix or "<root>", rule)
            return
        for fname, frule in fields.items():
            if not isinstance(frule, dict) or frule.get("optional"):
                continue
            new_prefix = f"{prefix}.{fname}" if prefix else fname
            yield from iter_required_paths(frule, new_prefix)
        return

    if rtype == "array":
        items_rule = rule.get("items")
        new_prefix = f"{prefix}[]" if prefix else "[]"
        if isinstance(items_rule, dict):
            yield from iter_required_paths(items_rule, new_prefix)
        else:
            yield (new_prefix, rule)
        return

    yield (prefix or "<root>", rule)


# ── Field-presence walker (top-level only, for retry checklist) ────────

def render_checklist(rule: dict, value: Any, indent: str = "    ") -> list[str]:
    """Render a recursive ``[x]/[ ]`` checklist of required paths vs ``value``.

    Used by the per-artifact retry hint so the checklist text mirrors the
    validator's traversal: nested objects/arrays surface as
    ``info.title``, ``sprint_plans[0].tasks[0].task_id``, etc. Optional
    fields are skipped — checklist shows only what the validator demands.
    """
    if not isinstance(rule, dict):
        return []
    rule = _normalize_rule(rule)
    lines: list[str] = []
    _render_checklist_recursive(rule, value, indent, "", lines)
    return lines


def _render_checklist_recursive(
    rule: dict, value: Any, indent: str, prefix: str, lines: list[str]
) -> None:
    rtype = rule.get("type")

    if rtype == "object":
        fields = rule.get("fields") or {}
        if not fields:
            present = isinstance(value, dict) and len(value) > 0
            mark = "x" if present else " "
            lines.append(f"{indent}- [{mark}] {prefix or '<root>'} (object)")
            return
        for fname, frule in fields.items():
            if not isinstance(frule, dict) or frule.get("optional"):
                continue
            new_prefix = f"{prefix}.{fname}" if prefix else fname
            fvalue = value.get(fname) if isinstance(value, dict) else None
            present = (
                isinstance(value, dict)
                and fname in value
                and not is_empty_required_value(fvalue)
            )
            sub_type = frule.get("type")
            if sub_type in ("object", "array") and present:
                _render_checklist_recursive(frule, fvalue, indent, new_prefix, lines)
            else:
                mark = "x" if present else " "
                lines.append(f"{indent}- [{mark}] {new_prefix}")
        return

    if rtype == "array":
        min_items = int(rule.get("min_items") or 0)
        have = len(value) if isinstance(value, list) else 0
        cnt_mark = "x" if have >= min_items else " "
        path_label = prefix or "<root>"
        lines.append(
            f"{indent}- [{cnt_mark}] {path_label} (array, need >= {min_items}, have {have})"
        )
        items_rule = rule.get("items")
        if isinstance(items_rule, dict) and have:
            # Render against the FIRST item as representative; full per-item
            # checklist gets noisy on long lists. Validator catches per-item
            # issues with specific paths.
            _render_checklist_recursive(
                items_rule, value[0], indent + "  ", f"{prefix}[0]", lines
            )
        return


def check_presence_top_level(rule: dict, value: Any) -> dict[str, bool]:
    """Return ``{field_name: bool_present}`` for top-level required fields.

    Used by the retry checklist to render ``[x] info`` / ``[ ] paths``
    style status lines. Doesn't descend — full-tree traversal is the
    validator's job; the checklist is a high-level orientation.
    """
    if not isinstance(rule, dict):
        return {}
    rule = _normalize_rule(rule)
    rtype = rule.get("type")
    if rtype == "object":
        fields = rule.get("fields") or {}
        out: dict[str, bool] = {}
        if not isinstance(value, dict):
            return {fname: False for fname, frule in fields.items()
                    if not (isinstance(frule, dict) and frule.get("optional"))}
        for fname, frule in fields.items():
            if isinstance(frule, dict) and frule.get("optional"):
                continue
            present = (
                fname in value
                and not is_empty_required_value(value[fname])
            )
            out[fname] = present
        return out
    if rtype == "array":
        # Surface count vs min_items as the "presence" signal for arrays.
        min_items = int(rule.get("min_items") or 0)
        if not isinstance(value, list):
            return {f"items (min {min_items})": False}
        return {f"items (min {min_items})": len(value) >= min_items}
    return {}
