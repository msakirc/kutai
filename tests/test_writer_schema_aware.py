"""Regressions for the schema-aware WriterAgent system prompt.

When ``artifact_schema.type == "markdown"``, the agent must switch from
the "write_file + summarize" pattern to "emit markdown content directly
in result". The old pattern reliably failed required-section validation
because ``result`` carried only a summary blurb while the actual content
lived in a file the validator never read (i2p_v3 steps 7.15, 11.4, 12.1
before this change).
"""
from __future__ import annotations

import json

import pytest

from src.agents.writer import (
    WriterAgent,
    _detect_markdown_schema,
    _FILE_WRITE_PROMPT,
    _INLINE_MARKDOWN_PROMPT,
)


class TestDetectMarkdownSchema:
    def test_markdown_schema_detected(self):
        task = {
            "context": json.dumps(
                {"artifact_schema": {"prd": {"type": "markdown"}}}
            )
        }
        assert _detect_markdown_schema(task)

    def test_object_schema_not_detected(self):
        task = {
            "context": json.dumps(
                {"artifact_schema": {"spec": {"type": "object"}}}
            )
        }
        assert not _detect_markdown_schema(task)

    def test_array_schema_not_detected(self):
        task = {
            "context": json.dumps(
                {"artifact_schema": {"items": {"type": "array"}}}
            )
        }
        assert not _detect_markdown_schema(task)

    def test_no_context_returns_false(self):
        assert not _detect_markdown_schema({})

    def test_empty_context_returns_false(self):
        assert not _detect_markdown_schema({"context": "{}"})

    def test_malformed_context_returns_false(self):
        assert not _detect_markdown_schema({"context": "broken json"})

    def test_already_parsed_dict_context(self):
        task = {
            "context": {
                "artifact_schema": {"x": {"type": "markdown"}}
            }
        }
        assert _detect_markdown_schema(task)

    def test_double_encoded_context(self):
        # Some legacy DB rows store context as a JSON-encoded string of
        # a JSON-encoded string. Both layers must unwind.
        inner = json.dumps({"artifact_schema": {"x": {"type": "markdown"}}})
        outer = json.dumps(inner)
        assert _detect_markdown_schema({"context": outer})

    def test_no_artifact_schema_key(self):
        task = {"context": json.dumps({"some_other_key": True})}
        assert not _detect_markdown_schema(task)


class TestWriterSystemPromptSwitching:
    def test_markdown_task_gets_inline_prompt(self):
        task = {
            "context": json.dumps(
                {"artifact_schema": {"prd": {"type": "markdown"}}}
            )
        }
        prompt = WriterAgent().get_system_prompt(task)
        assert prompt is _INLINE_MARKDOWN_PROMPT

    def test_object_task_gets_file_write_prompt(self):
        task = {
            "context": json.dumps(
                {"artifact_schema": {"spec": {"type": "object"}}}
            )
        }
        prompt = WriterAgent().get_system_prompt(task)
        assert prompt is _FILE_WRITE_PROMPT

    def test_no_schema_task_gets_file_write_prompt(self):
        prompt = WriterAgent().get_system_prompt({})
        assert prompt is _FILE_WRITE_PROMPT

    def test_inline_prompt_forbids_write_file(self):
        # The structural fix — inline mode must explicitly tell the model
        # not to call write_file, since the workflow engine persists the
        # result itself.
        assert "DO NOT call `write_file`" in _INLINE_MARKDOWN_PROMPT

    def test_inline_prompt_demands_full_content(self):
        # The whole point: result must carry the markdown, not a summary.
        assert "FULL markdown content" in _INLINE_MARKDOWN_PROMPT
        assert "not a summary" in _INLINE_MARKDOWN_PROMPT

    def test_file_write_prompt_keeps_summary_pattern(self):
        # Non-markdown-schema tasks (free-form documentation, code
        # comments, etc.) still use the summary-blurb result.
        assert "Wrote [filename]" in _FILE_WRITE_PROMPT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
