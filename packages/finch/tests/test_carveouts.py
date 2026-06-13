"""Task 9 carve-out tests — WriterProfile leaf subclass.

Verifies:
  (a) Writer plain (base) branch: get_system_prompt returns system_prompt
      for a plain task (no markdown schema).
  (b) Writer markdown branch: a task with artifact_schema.type == "markdown"
      and NO produces returns markdown_prompt (≠ system_prompt).
  (c) Char-exact match against the original src/agents/writer.py prompt
      strings (recovered from git HEAD before deletion).
  (d) WriterProfile is the concrete singleton type in the registry.
  (e) produces guard: non-empty produces forces the base branch even if
      artifact_schema.type == "markdown".
"""
from __future__ import annotations

import json

import pytest

from finch import get_profile
from finch.profile import WriterProfile, _detect_markdown_schema


# ── Originals, recovered from git show HEAD:src/agents/writer.py ──────────
# These are the verbatim concatenated string values from the deleted module.
# If either assertion fails the porting was lossy.
_ORIG_BASE_LEN = 1395
_ORIG_MARKDOWN_LEN = 1665

_ORIG_BASE_START = "You are a professional technical writer. You create clear, well-structured documentation and content."
_ORIG_MARKDOWN_START = "You are a professional technical writer producing a structured markdown artifact for a workflow step."

_ORIG_BASE_END = "IMPORTANT: Do NOT give a final_answer until you have actually saved the file(s) with `write_file`."
_ORIG_MARKDOWN_END = "Inner quotes inside the markdown must be escaped exactly once (`\\\"`). Do not double-escape — the workflow engine canonicalizes your output and an extra escape layer just makes retries harder."


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def writer() -> WriterProfile:
    p = get_profile("writer")
    assert isinstance(p, WriterProfile), f"expected WriterProfile singleton, got {type(p)}"
    return p  # type: ignore[return-value]


def _md_task(produces=None):
    """Build a task dict that triggers the markdown branch."""
    ctx: dict = {"artifact_schema": {"prd": {"type": "markdown"}}}
    if produces is not None:
        ctx["produces"] = produces
    return {"context": ctx}


def _plain_task():
    return {"title": "x"}


# ── Type check ─────────────────────────────────────────────────────────────

def test_writer_is_writer_profile_subclass(writer):
    assert isinstance(writer, WriterProfile)
    assert issubclass(WriterProfile, __import__("finch").Profile)


# ── Branch selection ───────────────────────────────────────────────────────

def test_plain_branch_returns_system_prompt(writer):
    """Plain task (no artifact_schema) → system_prompt (base branch)."""
    prompt = writer.get_system_prompt(_plain_task())
    assert prompt == writer.system_prompt
    assert len(prompt) > 20


def test_markdown_branch_returns_markdown_prompt(writer):
    """Task with artifact_schema.type=markdown → markdown_prompt."""
    prompt = writer.get_system_prompt(_md_task())
    assert prompt == writer.markdown_prompt
    assert prompt != writer.system_prompt


def test_markdown_with_nonempty_produces_returns_base(writer):
    """produces=[...] forces base branch even for markdown schema."""
    prompt = writer.get_system_prompt(_md_task(produces=["out/report.md"]))
    assert prompt == writer.system_prompt


def test_empty_produces_does_not_block_markdown_branch(writer):
    """produces=[] is treated as absent; markdown branch fires."""
    prompt = writer.get_system_prompt(_md_task(produces=[]))
    assert prompt == writer.markdown_prompt


# ── Char-exact verification ────────────────────────────────────────────────

def test_base_prompt_char_exact_length(writer):
    assert len(writer.system_prompt) == _ORIG_BASE_LEN, (
        f"system_prompt length {len(writer.system_prompt)} != original {_ORIG_BASE_LEN}"
    )


def test_base_prompt_starts_verbatim(writer):
    assert writer.system_prompt.startswith(_ORIG_BASE_START)


def test_base_prompt_ends_verbatim(writer):
    assert writer.system_prompt.endswith(_ORIG_BASE_END)


def test_markdown_prompt_char_exact_length(writer):
    assert len(writer.markdown_prompt) == _ORIG_MARKDOWN_LEN, (
        f"markdown_prompt length {len(writer.markdown_prompt)} != original {_ORIG_MARKDOWN_LEN}"
    )


def test_markdown_prompt_starts_verbatim(writer):
    assert writer.markdown_prompt.startswith(_ORIG_MARKDOWN_START)


def test_markdown_prompt_ends_verbatim(writer):
    assert writer.markdown_prompt.endswith(_ORIG_MARKDOWN_END)


# ── Key phrase checks ──────────────────────────────────────────────────────

def test_markdown_prompt_forbids_write_file(writer):
    assert "DO NOT call `write_file`" in writer.markdown_prompt


def test_markdown_prompt_demands_full_content(writer):
    assert "FULL markdown content" in writer.markdown_prompt
    assert "not a summary" in writer.markdown_prompt


def test_base_prompt_has_summary_pattern(writer):
    assert "Wrote [filename]" in writer.system_prompt


# ── _detect_markdown_schema unit tests ────────────────────────────────────

class TestDetectMarkdownSchema:
    def test_markdown_schema_detected(self):
        task = {"context": json.dumps({"artifact_schema": {"prd": {"type": "markdown"}}})}
        assert _detect_markdown_schema(task)

    def test_object_schema_not_detected(self):
        task = {"context": json.dumps({"artifact_schema": {"spec": {"type": "object"}}})}
        assert not _detect_markdown_schema(task)

    def test_no_context_returns_false(self):
        assert not _detect_markdown_schema({})

    def test_empty_context_returns_false(self):
        assert not _detect_markdown_schema({"context": "{}"})

    def test_malformed_context_returns_false(self):
        assert not _detect_markdown_schema({"context": "broken json"})

    def test_already_parsed_dict_context(self):
        task = {"context": {"artifact_schema": {"x": {"type": "markdown"}}}}
        assert _detect_markdown_schema(task)

    def test_double_encoded_context(self):
        inner = json.dumps({"artifact_schema": {"x": {"type": "markdown"}}})
        outer = json.dumps(inner)
        assert _detect_markdown_schema({"context": outer})

    def test_nonempty_produces_blocks_markdown(self):
        ctx = {"artifact_schema": {"prd": {"type": "markdown"}}, "produces": ["file.md"]}
        assert not _detect_markdown_schema({"context": ctx})
