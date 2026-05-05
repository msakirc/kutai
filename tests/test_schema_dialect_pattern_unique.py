"""Coverage for new dialect rules: string `pattern`, string `equals`/`one_of`, array `unique_by`."""
from src.workflows.engine.schema_dialect import validate_value, translate_rule


# ── String equals / one_of ───────────────────────────────────────────────

class TestStringEquals:
    def test_single_value_pass(self):
        rule = {"type": "string", "equals": "pass"}
        assert validate_value(rule, "pass") is None

    def test_single_value_fail(self):
        rule = {"type": "string", "equals": "pass"}
        err = validate_value(rule, "fail")
        assert err is not None
        assert "VERDICT" in err
        assert "pass" in err

    def test_list_accepts_any_listed_value(self):
        rule = {"type": "string", "equals": ["pass", "needs_minor_fixes"]}
        assert validate_value(rule, "pass") is None
        assert validate_value(rule, "needs_minor_fixes") is None
        err = validate_value(rule, "changes_requested")
        assert err is not None

    def test_one_of_synonym(self):
        rule = {"type": "string", "one_of": ["yes", "no"]}
        assert validate_value(rule, "yes") is None
        err = validate_value(rule, "maybe")
        assert err is not None

    def test_equals_does_not_translate_to_json_schema_enum(self):
        # Same anti-fabrication rationale as boolean.equals — equals is a
        # post-emit validator gate, not a decode-time constraint.
        rule = {"type": "string", "equals": "pass"}
        out = translate_rule(rule)
        assert out == {"type": "string"}
        assert "enum" not in (out or {})

    def test_combined_with_pattern_both_apply(self):
        rule = {"type": "string", "pattern": r"^[a-z_]+$", "equals": ["pass", "fail"]}
        # Pattern matches AND equals matches → ok
        assert validate_value(rule, "pass") is None
        # Pattern matches but equals fails → reject
        err = validate_value(rule, "yes")
        assert err is not None and "VERDICT" in err

    def test_min_length_runs_before_equals(self):
        rule = {"type": "string", "min_length": 1, "equals": "pass"}
        # Empty fails min_length first
        assert validate_value(rule, "") is not None


# ── Existing rules below ─────────────────────────────────────────────────


# ── String pattern ───────────────────────────────────────────────────────

class TestStringPattern:
    def test_pattern_match_passes(self):
        rule = {"type": "string", "pattern": r"^F-\d{2,3}$"}
        assert validate_value(rule, "F-001") is None
        assert validate_value(rule, "F-99") is None

    def test_pattern_mismatch_fails(self):
        rule = {"type": "string", "pattern": r"^F-\d{2,3}$"}
        err = validate_value(rule, "feat_1")
        assert err is not None
        assert "pattern" in err

    def test_pattern_empty_string_caught_by_min_length_first(self):
        rule = {"type": "string", "pattern": r"^F-\d+$", "min_length": 1}
        # Empty string fails min_length before reaching pattern; either
        # error is acceptable as long as we reject.
        assert validate_value(rule, "") is not None

    def test_pattern_invalid_regex_surfaces_error(self):
        rule = {"type": "string", "pattern": "([unclosed"}
        err = validate_value(rule, "anything")
        assert err is not None
        assert "invalid pattern" in err

    def test_pattern_translates_to_json_schema(self):
        rule = {"type": "string", "pattern": r"^F-\d+$"}
        out = translate_rule(rule)
        assert out == {"type": "string", "pattern": r"^F-\d+$"}

    def test_string_without_pattern_unchanged(self):
        rule = {"type": "string"}
        assert translate_rule(rule) == {"type": "string"}


# ── Array unique_by ──────────────────────────────────────────────────────

class TestArrayUniqueBy:
    def test_unique_by_field_dedup_passes(self):
        rule = {
            "type": "array",
            "min_items": 2,
            "unique_by": "feature_id",
            "items": {
                "type": "object",
                "fields": {
                    "feature_id": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        }
        value = [
            {"feature_id": "F-001", "name": "a"},
            {"feature_id": "F-002", "name": "b"},
        ]
        assert validate_value(rule, value) is None

    def test_unique_by_field_duplicate_fails(self):
        rule = {
            "type": "array",
            "min_items": 2,
            "unique_by": "feature_id",
            "items": {
                "type": "object",
                "fields": {
                    "feature_id": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        }
        value = [
            {"feature_id": "F-001", "name": "a"},
            {"feature_id": "F-001", "name": "b-dup"},
        ]
        err = validate_value(rule, value)
        assert err is not None
        assert "duplicate" in err
        assert "F-001" in err

    def test_unique_by_dot_for_scalar_items(self):
        rule = {
            "type": "array",
            "min_items": 2,
            "unique_by": ".",
            "items": {"type": "string"},
        }
        assert validate_value(rule, ["a", "b"]) is None
        err = validate_value(rule, ["a", "a"])
        assert err is not None and "duplicate" in err

    def test_unique_by_missing_field_skips_silently(self):
        # An item without the unique-by field is not a uniqueness violation
        # (different concern handled by per-field presence checks). The
        # uniqueness check itself must not flag a None vs a present value.
        rule = {
            "type": "array",
            "unique_by": "feature_id",
            "items": {
                "type": "object",
                "fields": {
                    "feature_id": {"type": "string", "optional": True},
                    "name": {"type": "string"},
                },
            },
        }
        value = [
            {"name": "no-id"},
            {"feature_id": "F-001", "name": "has-id"},
        ]
        assert validate_value(rule, value) is None

    def test_unique_by_runs_before_per_item_validation(self):
        # min_items + unique_by both relevant; we want a clear unique-by error
        # rather than a generic per-item one.
        rule = {
            "type": "array",
            "unique_by": "feature_id",
            "items": {
                "type": "object",
                "fields": {"feature_id": {"type": "string"}},
            },
        }
        dup = [
            {"feature_id": "F-001"},
            {"feature_id": "F-001"},
        ]
        err = validate_value(rule, dup)
        assert err is not None and "duplicate" in err


# ── Real i2p_v3 [8.0] schema shape ───────────────────────────────────────

class TestImplementationBacklogSchema:
    """End-to-end check on the actual schema shape used in i2p_v3.json."""

    schema = {
        "type": "array",
        "min_items": 5,
        "unique_by": "feature_id",
        "items": {
            "type": "object",
            "fields": {
                "feature_id": {"type": "string", "pattern": r"^F-\d{2,3}$"},
                "feature_name": {"type": "string", "min_length": 3},
                "epic_id": {},
                "sprint_id": {},
                "status": {},
                "template_ref": {},
                "depends_on_features": {"type": "array", "optional": True},
            },
        },
    }

    def _row(self, fid: str, name: str = "Real Time Sync") -> dict:
        return {
            "feature_id": fid,
            "feature_name": name,
            "epic_id": "E-01",
            "sprint_id": "Sprint 1",
            "status": "pending",
            "template_ref": "feature_implementation_template",
        }

    def test_mission57_pattern_one_entry_fails_min_items(self):
        # Mission 57 emitted only F-00; new schema rejects.
        value = [self._row("F-00", "Project Infra")]
        err = validate_value(self.schema, value)
        assert err is not None
        assert "items" in err and "5" in err

    def test_duplicate_feature_id_fails(self):
        value = [self._row(f"F-{i:03d}") for i in range(1, 6)]
        value[3]["feature_id"] = value[2]["feature_id"]
        err = validate_value(self.schema, value)
        assert err is not None and "duplicate" in err

    def test_bad_feature_id_pattern_fails(self):
        value = [self._row(f"F-{i:03d}") for i in range(1, 6)]
        value[2]["feature_id"] = "feature-three"
        err = validate_value(self.schema, value)
        assert err is not None and "pattern" in err

    def test_short_feature_name_fails(self):
        value = [self._row(f"F-{i:03d}") for i in range(1, 6)]
        value[1]["feature_name"] = "x"
        err = validate_value(self.schema, value)
        assert err is not None and "feature_name" in err

    def test_clean_backlog_passes(self):
        names = [
            "User Management",
            "Real-time Sync",
            "Offline Mode",
            "Categories",
            "Search",
        ]
        value = [self._row(f"F-{i+1:03d}", names[i]) for i in range(5)]
        assert validate_value(self.schema, value) is None
