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


@pytest.mark.parametrize(
    "verb,expected",
    [
        ("inject_north_star", "full"),
        ("emit_metric", "full"),
        ("record_hypothesis", "full"),
        ("record_verdict", "full"),
        ("suppress_hypothesis", "full"),
        ("assign_variant", "partial"),
        ("retire_variant", "partial"),
        ("score_backlog", "full"),
        ("score_sunset", "full"),
    ],
)
def test_z9_growth_verbs_registered(verb: str, expected: str) -> None:
    """Z9 T1C — the 9 growth verbs have explicit reversibility tags.

    assign_variant / retire_variant are ``partial`` because once a
    variant is shown to real users that exposure can't be unwound even
    though the DB row is deletable.
    """
    assert verb in VERB_REVERSIBILITY, f"{verb!r} missing from registry"
    assert get_reversibility(verb) == expected


def test_all_tags_are_valid() -> None:
    valid = {"full", "partial", "irreversible"}
    for verb, tag in VERB_REVERSIBILITY.items():
        assert tag in valid, f"verb {verb!r} has invalid tag {tag!r}"


def test_every_registry_verb_is_a_real_dispatcher_action() -> None:
    """Cross-check: every verb in VERB_REVERSIBILITY must be wired into the
    mr_roboto dispatcher.

    The dispatcher routes verbs in three shapes, all of which count:
      1. ``if action == "<verb>"``           — the common case
      2. ``if action in ("<verb>", ...)``    — cron + sub-verb groups
      3. ``action.startswith("<family>/")``  — slash-verb families
         (``reviews/poll/*``, ``mention_polls/*``, …) routed by prefix

    A verb is "wired" if its literal appears as a quoted string in
    ``__init__.py`` (covers shapes 1+2) or a slash ancestor is prefix-routed
    (shape 3).

    EXEMPT holds verbs intentionally NOT dispatched as top-level actions:
    they carry a reversibility tag for the shared taxonomy but reach
    execution another way.
    """
    init_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mr_roboto"
        / "__init__.py"
    )
    src = init_path.read_text(encoding="utf-8")

    def _is_dispatched(verb: str) -> bool:
        # Shapes 1+2 — verb literal appears in an `==` or `in (...)` check.
        if f'"{verb}"' in src or f"'{verb}'" in src:
            return True
        # Shape 3 — a slash ancestor is routed by a `startswith` prefix.
        if "/" in verb:
            parts = verb.split("/")
            for i in range(1, len(parts)):
                prefix = "/".join(parts[:i]) + "/"
                if (
                    f'startswith("{prefix}")' in src
                    or f"startswith('{prefix}')" in src
                ):
                    return True
        return False

    EXEMPT = {
        # git_ops shell-side tag only; propose_spec_patch is a compat alias.
        "git_push",
        "propose_spec_patch",
        # Z10-T3B caller-opened confirmation verbs — src/tools/shell.py opens
        # them directly via request_confirmation; no dispatcher block.
        "sandbox_local_mode",
        "broader_egress",
        # Z8 T4B on-call verbs — executed under the ``oncall_action``
        # gateway (whitelist + cooldown check), not as top-level actions.
        "restart_service",
        "rollback_to_last_green",
        "scale_up",
        "scale_down",
        "drain_traffic",
        "rotate_failed_key",
        "archive_flake_test",
        "escalate_to_founder",
        # Sub-mode of demo/distribute — flips an already-uploaded video to
        # public from inside the distribute handler, not a top-level action.
        "demo/distribute/flip_to_public",
        # Z9 growth verbs still pending a dispatcher block (later tiers).
        "emit_metric",
        "suppress_hypothesis",
    }
    for verb in VERB_REVERSIBILITY:
        if verb in EXEMPT:
            continue
        assert _is_dispatched(verb), (
            f"registry verb {verb!r} not wired into the dispatcher; "
            f"either fix the registry entry or add to EXEMPT"
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
