"""Tests for B3 streaming post-processor guards."""
from __future__ import annotations

import pytest

from coulson.streaming_guards import (
    BrokenFenceGuard,
    HallucinatedImportGuard,
    MalformedJsonGuard,
    StreamingGuardPipeline,
    TypoGuard,
    GuardResult,
)


# ─── TypoGuard ───────────────────────────────────────────────────────────


def test_typo_guard_fixes_console_typo():
    g = TypoGuard(typo_map={"cosnole.log": "console.log"})
    out = ""
    for tok in ["cos", "nole", ".log('hi');"]:
        res = g.check(tok, out)
        if res.action == "pass":
            out += tok
        elif res.patched_token is not None:
            out += res.patched_token
    out += g.flush()
    assert "console.log" in out
    assert "cosnole" not in out


def test_typo_guard_no_typo_passes_through():
    g = TypoGuard(typo_map={"cosnole.log": "console.log"})
    out = ""
    for tok in ["console.log('ok');"]:
        res = g.check(tok, out)
        if res.action == "pass":
            out += tok
        elif res.patched_token is not None:
            out += res.patched_token
    out += g.flush()
    assert out == "console.log('ok');"


def test_typo_guard_split_across_tokens():
    """Typo split across token boundaries must still be caught."""
    g = TypoGuard(typo_map={"improt ": "import "})
    out = ""
    for tok in ["impr", "ot ", "os"]:
        res = g.check(tok, out)
        if res.action == "pass":
            out += tok
        elif res.patched_token is not None:
            out += res.patched_token
    out += g.flush()
    assert "import os" in out


# ─── MalformedJsonGuard ──────────────────────────────────────────────────


def test_malformed_json_guard_clean_block_passes():
    g = MalformedJsonGuard()
    text = '```json\n{"a": 1, "b": [2,3]}\n```'
    res = g.check(text, "")
    assert res.action == "pass"


def test_malformed_json_guard_unbalanced_close_warns():
    g = MalformedJsonGuard()
    # Open a json fence with unbalanced braces, then close
    text = '```json\n{"a": 1, "b": [2,3\n```'
    res = g.check(text, "")
    assert res.action == "warn"
    assert "unbalanced" in res.note


def test_malformed_json_guard_outside_fence_noop():
    g = MalformedJsonGuard()
    text = "this is just prose with } unbalanced characters"
    res = g.check(text, "")
    assert res.action == "pass"


# ─── BrokenFenceGuard ────────────────────────────────────────────────────


def test_broken_fence_guard_balanced_no_finalize():
    g = BrokenFenceGuard()
    g.check("here is code:\n```\nhello\n```\n", "")
    assert g.finalize() is None


def test_broken_fence_guard_unclosed_auto_closes():
    g = BrokenFenceGuard()
    g.check("here is code:\n```python\nhello\n", "")
    res = g.finalize()
    assert res is not None
    assert res.action == "fix"
    assert "```" in (res.patched_token or "")


# ─── HallucinatedImportGuard ─────────────────────────────────────────────


def test_hallucinated_import_guard_allowlisted_passes():
    g = HallucinatedImportGuard(allowlist={"os", "json"})
    res = g.check("import os\nimport json\n", "")
    assert res.action == "pass"


def test_hallucinated_import_guard_unknown_warns():
    g = HallucinatedImportGuard(allowlist={"os"})
    res = g.check("from totally_made_up_pkg import thing\n", "")
    assert res.action == "warn"
    assert "totally_made_up_pkg" in res.note


def test_hallucinated_import_guard_dedupes():
    """Same suspicious import seen twice should warn only once."""
    g = HallucinatedImportGuard(allowlist=set())
    r1 = g.check("import banana\n", "")
    r2 = g.check("import banana\n", "")
    assert r1.action == "warn"
    assert r2.action == "pass"  # already seen


def test_hallucinated_import_guard_relative_js_skipped():
    g = HallucinatedImportGuard(allowlist=set())
    res = g.check("import x from './local';\n", "")
    assert res.action == "pass"


# ─── Pipeline integration ────────────────────────────────────────────────


def test_pipeline_passes_clean_text_through():
    p = StreamingGuardPipeline()
    chunks = ["hello ", "world\n"]
    full = ""
    for c in chunks:
        out = p.process(c)
        full += out.text
    full += p.finalize().text
    assert "hello world" in full


def test_pipeline_fixes_typos_inline():
    p = StreamingGuardPipeline()
    chunks = ["cos", "nole", ".log('x');"]
    full = ""
    for c in chunks:
        out = p.process(c)
        full += out.text
    full += p.finalize().text
    assert "console.log" in full
    assert "cosnole" not in full


def test_pipeline_collects_warnings():
    captured: list[GuardResult] = []
    p = StreamingGuardPipeline(sink=captured.append)
    # Unbalanced JSON fence triggers MalformedJsonGuard warn
    p.process('```json\n{"a": 1, "b": [\n```')
    p.finalize()
    names = {r.guard_name for r in captured}
    assert "malformed_json" in names


def test_pipeline_finalize_auto_closes_unclosed_fence():
    p = StreamingGuardPipeline()
    p.process("here:\n```python\nfoo\n")
    tail = p.finalize()
    assert "```" in tail.text


def test_pipeline_opt_out_no_op(monkeypatch):
    monkeypatch.setenv("KUTAI_STREAMING_GUARDS", "off")
    p = StreamingGuardPipeline()
    out = p.process("cosnole.log('x');")
    # When disabled, no fixes happen
    assert out.text == "cosnole.log('x');"


def test_pipeline_halt_aborts_stream():
    """A guard returning halt must propagate to the pipeline outcome."""
    class HaltGuard:
        name = "halter"

        def reset(self):
            pass

        def check(self, token, accumulated):
            if "STOP" in token:
                return GuardResult(action="halt", note="stopword", guard_name=self.name)
            return GuardResult(action="pass", guard_name=self.name)

    p = StreamingGuardPipeline(guards=[HaltGuard()])
    out1 = p.process("hello ")
    out2 = p.process("STOP now")
    assert out1.halt is False
    assert out2.halt is True


def test_pipeline_reset_clears_state():
    p = StreamingGuardPipeline()
    p.process("```json\n{")
    p.reset()
    # After reset, the json guard should not still think we're inside a fence
    out = p.process("just text")
    assert out.text == "just text"
    # No warnings on plain text post-reset
    assert all(w.action != "warn" for w in out.warnings)
