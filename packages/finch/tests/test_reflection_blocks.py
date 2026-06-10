"""Phase 3 Task 12 Batch H — reflection block CONTENT lives in the leaf as data.

The TEXT moved here from coulson (HYBRID); coulson keeps composition. These
tests lock the data shape + the package-level exports the back-compat
re-exports depend on.
"""
import finch as pf
from finch.reflection_blocks import (
    REFLECT_SYSTEM_BASE,
    REFLECTION_BLOCKS,
    STACK_BLOCKS,
    LAYER_BLOCKS,
    _GENERIC_REFLECTION_BLOCK,
)


def test_exports_present_on_package():
    assert pf.REFLECT_SYSTEM_BASE is REFLECT_SYSTEM_BASE
    assert pf.REFLECTION_BLOCKS is REFLECTION_BLOCKS
    assert pf.STACK_BLOCKS is STACK_BLOCKS
    assert pf.LAYER_BLOCKS is LAYER_BLOCKS
    assert pf._GENERIC_REFLECTION_BLOCK is _GENERIC_REFLECTION_BLOCK


def test_reflection_blocks_shape():
    assert isinstance(REFLECTION_BLOCKS, dict)
    for k, v in REFLECTION_BLOCKS.items():
        assert isinstance(v, str) and v, f"REFLECTION_BLOCKS[{k!r}] empty"
    # Per-agent entries documented in CLAUDE.md / wired via enable_self_reflection.
    for required in ("coder", "implementer", "fixer", "integration_reviewer"):
        assert required in REFLECTION_BLOCKS


def test_stack_blocks_seeded():
    assert len(STACK_BLOCKS) >= 7
    for k, v in STACK_BLOCKS.items():
        assert isinstance(v, str) and len(v) > 50, k


def test_layer_blocks_named_keys():
    for k in ("domain", "adapter", "infra", "ui", "test", "unknown"):
        assert k in LAYER_BLOCKS
    assert LAYER_BLOCKS["test"] == ""
    assert LAYER_BLOCKS["unknown"] == ""


def test_reflect_system_base_is_verdict_reviewer():
    assert "reviewer" in REFLECT_SYSTEM_BASE
    assert '"verdict"' in REFLECT_SYSTEM_BASE
