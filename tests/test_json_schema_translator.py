"""Tests for artifact_schema -> JSON Schema translation.

Covers the dialect cases observed in i2p_v3 (object with required_fields,
array with min_items + item_fields, markdown skip, multi-artifact wrap)
and the response_format payload shape that ``hallederiz_kadir.call``
expects.
"""
from __future__ import annotations

import pytest

from src.workflows.engine.json_schema_translator import (
    artifact_schema_to_json_schema,
    build_response_format,
)


class TestObjectTranslation:
    def test_required_fields_become_required_and_properties(self):
        sch = {
            "db_client": {
                "type": "object",
                "required_fields": ["client_path", "connection_verified"],
            }
        }
        out = artifact_schema_to_json_schema(sch)
        assert out["type"] == "object"
        assert out["additionalProperties"] is False
        assert sorted(out["required"]) == ["client_path", "connection_verified"]
        assert set(out["properties"].keys()) == {"client_path", "connection_verified"}

    def test_object_with_no_required_fields(self):
        # Empty required_fields still emits a constrainable schema —
        # additionalProperties:false alone is meaningful.
        sch = {"thing": {"type": "object", "required_fields": []}}
        out = artifact_schema_to_json_schema(sch)
        assert out == {
            "type": "object",
            "additionalProperties": False,
            "required": [],
            "properties": {},
        }


class TestArrayTranslation:
    def test_array_with_min_items_and_item_fields(self):
        sch = {
            "milestones": {
                "type": "array",
                "min_items": 4,
                "item_fields": ["id", "title", "due_date"],
            }
        }
        out = artifact_schema_to_json_schema(sch)
        assert out["type"] == "array"
        assert out["minItems"] == 4
        items = out["items"]
        assert items["type"] == "object"
        assert items["additionalProperties"] is False
        assert items["required"] == ["id", "title", "due_date"]

    def test_array_no_item_fields_emits_loose_array(self):
        sch = {"items": {"type": "array", "min_items": 1}}
        out = artifact_schema_to_json_schema(sch)
        assert out == {"type": "array", "minItems": 1}

    def test_array_no_min_items(self):
        sch = {"items": {"type": "array", "item_fields": ["x"]}}
        out = artifact_schema_to_json_schema(sch)
        assert "minItems" not in out
        assert out["items"]["required"] == ["x"]


class TestUnconstrainable:
    def test_markdown_returns_none(self):
        assert artifact_schema_to_json_schema({"doc": {"type": "markdown"}}) is None

    def test_string_returns_none(self):
        assert artifact_schema_to_json_schema({"x": {"type": "string"}}) is None

    def test_empty_returns_none(self):
        assert artifact_schema_to_json_schema({}) is None
        assert artifact_schema_to_json_schema(None) is None  # type: ignore[arg-type]

    def test_non_dict_returns_none(self):
        assert artifact_schema_to_json_schema("nope") is None  # type: ignore[arg-type]

    def test_config_only_returns_none(self):
        # max_output_chars is config, not an artifact rules dict.
        sch = {"max_output_chars": 20000}
        assert artifact_schema_to_json_schema(sch) is None


class TestMultiArtifactWrap:
    def test_two_artifacts_wrap_in_object(self):
        sch = {
            "primary": {"type": "object", "required_fields": ["a", "b"]},
            "secondary": {"type": "array", "min_items": 1},
        }
        out = artifact_schema_to_json_schema(sch)
        assert out["type"] == "object"
        assert sorted(out["required"]) == ["primary", "secondary"]
        assert out["properties"]["primary"]["type"] == "object"
        assert out["properties"]["secondary"]["type"] == "array"

    def test_config_entry_skipped_in_multi(self):
        sch = {
            "primary": {"type": "object", "required_fields": ["a"]},
            "secondary": {"type": "array", "min_items": 1},
            "max_output_chars": 5000,  # not an artifact
        }
        out = artifact_schema_to_json_schema(sch)
        assert "max_output_chars" not in out["required"]

    def test_one_constrainable_one_markdown_unwraps(self):
        # Markdown skipped; one constrainable left -> emit bare schema.
        sch = {
            "primary": {"type": "object", "required_fields": ["a"]},
            "doc": {"type": "markdown"},
        }
        out = artifact_schema_to_json_schema(sch)
        assert out["type"] == "object"
        assert out["required"] == ["a"]


class TestBuildResponseFormat:
    def test_shape_matches_openai_structured_outputs(self):
        sch = {"x": {"type": "object", "required_fields": ["a"]}}
        rf = build_response_format(sch, name="step_xyz")
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "step_xyz"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"]["type"] == "object"

    def test_returns_none_for_unconstrainable(self):
        assert build_response_format({"doc": {"type": "markdown"}}) is None
        assert build_response_format({}) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
