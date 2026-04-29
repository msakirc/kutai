"""Coverage for the canonical artifact_schema dialect (E1)."""
import json
import pytest

from src.workflows.engine.schema_dialect import (
    validate_value,
    translate_rule,
    make_example,
    iter_required_paths,
    render_checklist,
    is_empty_required_value,
    _normalize_rule,
)


# ── Old-form normalization ──────────────────────────────────────────────

class TestNormalize:
    def test_object_legacy_required_fields_to_canonical(self):
        # Legacy required_fields had no per-field type — auto-normalizer
        # leaves them untyped (presence-only) so legitimate non-string
        # values (arrays, numbers) aren't rejected.
        legacy = {"type": "object", "required_fields": ["a", "b"]}
        out = _normalize_rule(legacy)
        assert out == {
            "type": "object",
            "fields": {"a": {}, "b": {}},
        }

    def test_array_legacy_item_fields_to_canonical(self):
        legacy = {"type": "array", "min_items": 1, "item_fields": ["x", "y"]}
        out = _normalize_rule(legacy)
        assert out == {
            "type": "array",
            "min_items": 1,
            "items": {"type": "object", "fields": {"x": {}, "y": {}}},
        }

    def test_canonical_object_passthrough(self):
        canonical = {"type": "object", "fields": {"a": {"type": "string"}}}
        assert _normalize_rule(canonical) == canonical

    def test_object_no_fields_is_variable_key(self):
        # Canonical "variable-key object" lacks ``fields`` entirely.
        rule = {"type": "object", "min_keys": 1}
        assert _normalize_rule(rule) == rule


# ── Validation ──────────────────────────────────────────────────────────

class TestValidate:
    def test_object_passes_full(self):
        rule = {"type": "object", "fields": {"a": {"type": "string"}}}
        assert validate_value(rule, {"a": "hello"}) is None

    def test_object_missing_required(self):
        rule = {"type": "object", "fields": {
            "a": {"type": "string"}, "b": {"type": "string"}
        }}
        err = validate_value(rule, {"a": "x"})
        assert err and "b" in err

    def test_object_empty_placeholder_rejected(self):
        rule = {"type": "object", "fields": {"info": {"type": "object"}}}
        err = validate_value(rule, {"info": {}})
        assert err and "empty placeholder" in err

    def test_optional_field_skipped(self):
        rule = {"type": "object", "fields": {
            "a": {"type": "string"},
            "b": {"type": "string", "optional": True},
        }}
        assert validate_value(rule, {"a": "x"}) is None

    def test_array_min_items(self):
        rule = {"type": "array", "min_items": 2}
        err = validate_value(rule, ["one"])
        assert err and ">= 2" in err

    def test_nested_object_path_in_error(self):
        rule = {"type": "object", "fields": {
            "info": {"type": "object", "fields": {
                "title": {"type": "string"}
            }}
        }}
        err = validate_value(rule, {"info": {}})
        # info itself is an empty placeholder — error should reference info
        assert err and "info" in err

    def test_deeply_nested_array_of_objects(self):
        rule = {"type": "object", "fields": {
            "sprints": {"type": "array", "min_items": 1, "items": {
                "type": "object", "fields": {
                    "tasks": {"type": "array", "items": {
                        "type": "object", "fields": {
                            "task_id": {"type": "string"}
                        }
                    }}
                }
            }}
        }}
        err = validate_value(rule, {"sprints": [{"tasks": [{}]}]})
        assert err and "sprints[0].tasks[0]" in err


# ── Translation ─────────────────────────────────────────────────────────

class TestTranslate:
    def test_nested_object_descends(self):
        rule = {"type": "object", "fields": {
            "info": {"type": "object", "fields": {
                "title": {"type": "string"}
            }}
        }}
        out = translate_rule(rule)
        assert out["type"] == "object"
        assert out["properties"]["info"]["type"] == "object"
        assert out["properties"]["info"]["properties"]["title"] == {"type": "string"}
        assert "title" in out["properties"]["info"]["required"]

    def test_array_with_object_items(self):
        rule = {"type": "array", "min_items": 1, "items": {
            "type": "object", "fields": {"id": {"type": "string"}}
        }}
        out = translate_rule(rule)
        assert out == {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id"],
                "properties": {"id": {"type": "string"}},
            },
        }

    def test_legacy_form_translates(self):
        # Auto-normalize accepts legacy.
        rule = {"type": "object", "required_fields": ["a", "b"]}
        out = translate_rule(rule)
        assert set(out["required"]) == {"a", "b"}


# ── Example generation ──────────────────────────────────────────────────

class TestExample:
    def test_nested_example_emits_proper_structure(self):
        rule = {"type": "object", "fields": {
            "openapi": {"type": "string"},
            "info": {"type": "object", "fields": {
                "title": {"type": "string"}, "version": {"type": "string"}
            }},
        }}
        ex = make_example(rule)
        assert ex == {
            "openapi": "...",
            "info": {"title": "...", "version": "..."},
        }

    def test_array_of_objects_example(self):
        rule = {"type": "array", "min_items": 1, "items": {
            "type": "object", "fields": {"id": {"type": "string"}}
        }}
        assert make_example(rule) == [{"id": "..."}]

    def test_optional_skipped_in_example(self):
        rule = {"type": "object", "fields": {
            "req": {"type": "string"},
            "opt": {"type": "string", "optional": True},
        }}
        assert make_example(rule) == {"req": "..."}


# ── Checklist rendering ─────────────────────────────────────────────────

class TestChecklist:
    def test_top_level_only_when_value_missing(self):
        rule = {"type": "object", "fields": {
            "a": {"type": "string"}, "b": {"type": "string"}
        }}
        lines = render_checklist(rule, {})
        assert any("[ ] a" in l for l in lines)
        assert any("[ ] b" in l for l in lines)

    def test_descends_when_nested_value_present(self):
        rule = {"type": "object", "fields": {
            "info": {"type": "object", "fields": {
                "title": {"type": "string"}, "version": {"type": "string"}
            }},
        }}
        lines = render_checklist(rule, {"info": {"title": "X"}})
        assert any("[x] info.title" in l for l in lines)
        assert any("[ ] info.version" in l for l in lines)

    def test_empty_subobject_marks_parent_missing(self):
        rule = {"type": "object", "fields": {"info": {"type": "object", "fields": {
            "title": {"type": "string"}
        }}}}
        # Empty info dict should be flagged as missing at the parent level
        lines = render_checklist(rule, {"info": {}})
        assert any("[ ] info" in l for l in lines)


# ── Empty placeholder detection ─────────────────────────────────────────

class TestEmptyPlaceholder:
    @pytest.mark.parametrize("val", [None, "", "  ", "...", {}, []])
    def test_rejects(self, val):
        assert is_empty_required_value(val) is True

    @pytest.mark.parametrize("val", ["text", 0, False, {"k": "v"}, ["x"]])
    def test_accepts(self, val):
        assert is_empty_required_value(val) is False
