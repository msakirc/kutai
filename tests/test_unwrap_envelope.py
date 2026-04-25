"""Regressions for ``src.workflows.engine.hooks._unwrap_envelope`` truncation
recovery.

The older regex fallback required a downstream sentinel (``"memories"``,
``"subtasks"``, or a closing ``}``) to recover the ``result`` field from a
broken envelope. LLM outputs that ran out of budget mid-string had none of
those, so the raw envelope leaked through as the artifact (observed task
2890 user_stories, 4132-char triple-escaped truncated output).

The new ``_extract_json_string_field`` walks char-by-char, honors JSON
escape rules, and returns the partial body on truncation.
"""
from __future__ import annotations

import json

import pytest

from src.workflows.engine.hooks import (
    _unwrap_envelope,
    _extract_json_string_field,
    _unescape_json_string,
    canonicalize_for_retry,
)


class TestCanonicalizeForRetry:
    """Structural fix for Qwen JSON over-escape (deferred item 5).

    When ``_prev_output`` is fed back into a retry prompt, models
    re-wrap and re-escape inner quotes — every retry adds a layer.
    Canonicalization collapses N escape layers down to clean JSON
    so the next retry sees a parseable array/object.
    """

    def test_bare_array_stays_clean(self):
        arr = '[{"id":"x","name":"y"}]'
        assert json.loads(canonicalize_for_retry(arr)) == [
            {"id": "x", "name": "y"}
        ]

    def test_one_escape_layer_peeled(self):
        # The agent emitted a JSON-encoded string holding a JSON array.
        encoded = json.dumps('[{"id":"x"}]')
        out = canonicalize_for_retry(encoded)
        assert json.loads(out) == [{"id": "x"}]

    def test_two_escape_layers_peeled(self):
        # After two retries that compounded escapes.
        encoded = json.dumps(json.dumps('[{"id":"x"}]'))
        out = canonicalize_for_retry(encoded)
        assert json.loads(out) == [{"id": "x"}]

    def test_object_preserved(self):
        obj = json.dumps({"a": 1, "b": [1, 2, 3]})
        out = canonicalize_for_retry(obj)
        assert json.loads(out) == {"a": 1, "b": [1, 2, 3]}

    def test_plain_text_passthrough(self):
        assert canonicalize_for_retry("plain markdown text") == (
            "plain markdown text"
        )

    def test_broken_json_passthrough(self):
        assert canonicalize_for_retry("{partial broken") == "{partial broken"

    def test_empty_string(self):
        assert canonicalize_for_retry("") == ""

    def test_non_string_returned_as_is(self):
        assert canonicalize_for_retry(None) is None

    def test_max_depth_terminates(self):
        # Wrap a string that points back to itself in escape form. Should
        # not loop forever — bounded by max_depth.
        s = '"' + ('\\' * 30) + '"'
        out = canonicalize_for_retry(s, max_depth=3)
        # Either returns a leaf string or the original — never crashes.
        assert isinstance(out, str)


class TestExtractJsonStringField:
    def test_returns_none_when_key_absent(self):
        assert _extract_json_string_field('{"foo":1}', "result") is None

    def test_extracts_intact_value(self):
        body = _extract_json_string_field(
            '{"result":"hello world"}', "result"
        )
        assert body == "hello world"

    def test_recovers_truncated_string(self):
        body = _extract_json_string_field(
            '"result":"line1\\nline2 partial', "result"
        )
        # body is escaped — caller does the unescape pass
        assert body == "line1\\nline2 partial"

    def test_handles_escaped_quote_in_body(self):
        # An escaped quote must NOT terminate the extraction.
        body = _extract_json_string_field(
            '{"result":"he said \\"hi\\" to her"}', "result"
        )
        assert body == 'he said \\"hi\\" to her'

    def test_handles_dangling_backslash(self):
        # A trailing backslash with no successor should not crash.
        body = _extract_json_string_field('{"result":"abc\\', "result")
        assert body == "abc\\"


class TestUnescapeJsonString:
    def test_strict_path(self):
        assert _unescape_json_string("a\\nb") == "a\nb"

    def test_handles_triple_escape(self):
        # Qwen sometimes emits triple-backslashed quotes after retry layers
        # add escape passes. Strict json.loads bails — manual fallback runs.
        out = _unescape_json_string('a\\\\\\"b')
        # Manual fallback: \\\\ -> \\, then \\" -> "  (best-effort)
        assert '"' in out


class TestUnwrapEnvelope:
    def test_intact_final_answer(self):
        env = json.dumps({"action": "final_answer", "result": "hello world"})
        assert _unwrap_envelope(env) == "hello world"

    def test_intact_with_siblings_kept_working(self):
        env = json.dumps({
            "action": "final_answer",
            "result": "x",
            "memories": [],
        })
        assert _unwrap_envelope(env) == "x"

    def test_truncated_envelope_recovers_body(self):
        # No closing brace, no sibling key. Old regex would fail.
        broken = '{"action":"final_answer","result":"line1\\nline2\\nline3 partial'
        out = _unwrap_envelope(broken)
        assert "line1" in out
        assert "line2" in out
        assert "line3 partial" in out

    def test_truncated_write_file_recovers_content(self):
        broken = (
            '{"action":"tool_call","tool":"write_file",'
            '"args":{"content":"# Title\\n\\nbody text'
        )
        out = _unwrap_envelope(broken)
        assert "# Title" in out
        assert "body text" in out

    def test_strips_model_tokens(self):
        env = '<|function_call|>{"action":"final_answer","result":"x"}<|im_end|>'
        assert _unwrap_envelope(env) == "x"

    def test_strips_markdown_fences(self):
        env = '```json\n{"action":"final_answer","result":"x"}\n```'
        assert _unwrap_envelope(env) == "x"

    def test_passthrough_when_no_envelope(self):
        assert _unwrap_envelope("plain text") == "plain text"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
