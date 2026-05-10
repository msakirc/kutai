"""Z10-T1B — verb reversibility registry tests."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mr_roboto.reversibility import (
    DEFAULT_REVERSIBILITY,
    VERB_REVERSIBILITY,
    get_reversibility,
)


def test_known_verb_full() -> None:
    assert get_reversibility("git_commit") == "full"


def test_known_verb_irreversible() -> None:
    assert get_reversibility("notify_user") == "irreversible"


def test_unknown_verb_falls_back_to_default() -> None:
    assert get_reversibility("publish_app") == DEFAULT_REVERSIBILITY == "partial"


def test_override_wins_over_registry() -> None:
    # registry says "partial" for run_cmd; caller knows their command is destructive.
    assert get_reversibility("run_cmd", override="irreversible") == "irreversible"
    # caller can also widen — declare a destructive verb as full for a safe sub-case.
    assert get_reversibility("notify_user", override="full") == "full"


def test_all_tags_are_valid() -> None:
    valid = {"full", "partial", "irreversible"}
    for verb, tag in VERB_REVERSIBILITY.items():
        assert tag in valid, f"verb {verb!r} has invalid tag {tag!r}"


def test_every_registry_verb_is_a_real_dispatcher_action() -> None:
    """Cross-check: every verb in VERB_REVERSIBILITY must appear as an
    `if action == "<verb>"` block in mr_roboto/__init__.py.

    Allowed exceptions: ``git_push`` (real verb but executed via git_ops,
    not as a top-level mr_roboto action — registered for shell-side
    tagging) and ``propose_spec_patch`` (kept as alias for compat).
    """
    init_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mr_roboto"
        / "__init__.py"
    )
    src = init_path.read_text(encoding="utf-8")
    pattern = re.compile(r'if action == "([a-z_][a-z0-9_]*)"')
    dispatched = set(pattern.findall(src))
    # ``propose_spec_patch_from_html_diff`` IS in dispatcher; the registry
    # uses the shorter alias too. ``git_push`` and ``propose_spec_patch``
    # are intentional non-dispatcher entries (see docstring).
    # ``sandbox_local_mode`` and ``broader_egress`` (Z10-T3B) are
    # caller-opened confirmation verbs (src/tools/shell.py opens them
    # directly via request_confirmation); they do not have dispatcher
    # blocks. They share the tag taxonomy so live with the registry.
    EXEMPT = {
        "git_push",
        "propose_spec_patch",
        "sandbox_local_mode",
        "broader_egress",
    }
    for verb in VERB_REVERSIBILITY:
        if verb in EXEMPT:
            continue
        assert verb in dispatched, (
            f"registry verb {verb!r} not found as a dispatcher action; "
            f"either rename the registry entry or add to EXEMPT"
        )


def test_every_dispatcher_action_is_in_registry() -> None:
    """Catch new actions added to dispatcher without a reversibility tag."""
    init_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mr_roboto"
        / "__init__.py"
    )
    src = init_path.read_text(encoding="utf-8")
    pattern = re.compile(r'if action == "([a-z_][a-z0-9_]*)"')
    dispatched = set(pattern.findall(src))
    missing = dispatched - set(VERB_REVERSIBILITY)
    assert not missing, (
        f"dispatcher actions missing from VERB_REVERSIBILITY: {sorted(missing)}"
    )
