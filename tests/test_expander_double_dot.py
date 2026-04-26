"""Regression for the double-dot step_id bug in
``src.workflows.engine.expander.expand_template``.

Mission 46 phase 8 had every task titled ``[8.<fid>..feat.X]`` — two
dots between fid and template_step_id. Caller in
``hooks._trigger_template_expansion`` passed prefix=``"8.{fid}."``
(trailing dot to keep the dot grammar visible at the call site); the
f-string in expander then added another dot, doubling. Fix: strip
trailing dot in expander before adding its own.
"""
from __future__ import annotations

import pytest

from src.workflows.engine.expander import expand_template


_TEMPLATE = {
    "context_artifacts": [],
    "steps": [
        {
            "template_step_id": "feat.5",
            "instruction": "do {feature_name}",
            "agent": "coder",
            "output_artifacts": [],
        },
    ],
}


def test_trailing_dot_prefix_no_double_dot():
    """Caller in hooks.py uses prefix='8.<fid>.' — must not double-dot."""
    out = expand_template(
        _TEMPLATE,
        params={"feature_id": "FEAT-1", "feature_name": "Auth"},
        prefix="8.FEAT-1.",
    )
    assert out[0]["id"] == "8.FEAT-1.feat.5"
    assert ".." not in out[0]["id"]


def test_clean_prefix_works():
    """Caller passing prefix without trailing dot also works."""
    out = expand_template(
        _TEMPLATE,
        params={"feature_id": "FEAT-1", "feature_name": "Auth"},
        prefix="8.FEAT-1",
    )
    assert out[0]["id"] == "8.FEAT-1.feat.5"


def test_empty_prefix_returns_template_id_only():
    out = expand_template(
        _TEMPLATE,
        params={"feature_id": "X", "feature_name": "Y"},
        prefix="",
    )
    assert out[0]["id"] == "feat.5"


def test_empty_fid_with_trailing_dot_prefix():
    """Edge: caller passes prefix='8..' from empty fid — should still
    not produce ``8...feat.5``. The rstrip strips ALL trailing dots."""
    out = expand_template(
        _TEMPLATE,
        params={"feature_id": "", "feature_name": "Y"},
        prefix="8..",
    )
    # rstrip('.') strips everything → step_id = ".feat.5"? No — the
    # outer if-branch only fires when prefix is truthy, and after
    # rstrip the prefix could be empty. With current implementation,
    # `prefix.rstrip('.')` on "8.." yields "8" — step_id = "8.feat.5".
    assert out[0]["id"] == "8.feat.5"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
