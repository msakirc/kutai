"""Grounding: a DIRECTORY produces slot (trailing slash) is satisfied by any
write under it — not by an impossible exact string match.

m90 5.20a/5.20b (.screens/) and 5.30a/5.30b (.web/) declare a whole DIRECTORY
of agent-authored files. The produces path is the bare directory
(``mission_90/.screens/``). Grounding matched it with ``norm in written`` —
an exact string compare that a per-file write (``mission_90/.screens/home.md``)
can never satisfy, so grounding failed even after the agent authored every file
(m90 task 567454). A trailing-slash slot must be satisfied when at least one
written path lives under that directory prefix.
"""
from __future__ import annotations

from coulson.grounding import match_produces_entry, unmatched_produces


def test_dir_slot_satisfied_by_file_written_under_it():
    written = {"mission_90/.screens/home.md", "mission_90/.screens/settings.md"}
    assert match_produces_entry("mission_90/.screens/", written) is True


def test_dir_slot_unmatched_when_no_file_under_it():
    written = {"mission_90/.charter/product_charter.md"}
    assert match_produces_entry("mission_90/.screens/", written) is False
    assert unmatched_produces(["mission_90/.screens/"], written) == [
        "mission_90/.screens/"
    ]


def test_dir_slot_fully_grounded_yields_empty_unmatched():
    written = {"mission_90/.web/index.html"}
    assert unmatched_produces(["mission_90/.web/"], written) == []


def test_dir_slot_not_satisfied_by_sibling_directory_prefix():
    """`.screens/` must not be satisfied by a write to `.screens2/…` — the
    prefix compare has to respect the directory boundary."""
    written = {"mission_90/.screens2/home.md"}
    assert match_produces_entry("mission_90/.screens/", written) is False


def test_exact_slot_satisfied_by_scaffold_prefixed_path():
    """A generic slot (prisma/schema.prisma) is satisfied by the coder's
    scaffold-prefixed write (backend/prisma/schema.prisma) — the file IS the
    declared artifact, just under the project subdir. m90 7.4 cycled forever
    because grounding required an exact root-relative match the coder never
    writes (it scaffolds under backend/ or frontend/)."""
    assert match_produces_entry("prisma/schema.prisma",
                                {"backend/prisma/schema.prisma"}) is True
    assert match_produces_entry("scripts/seed.ts",
                                {"frontend/src/scripts/seed.ts"}) is True


def test_exact_slot_boundary_safe_no_partial_segment_match():
    """Suffix match must respect the path-segment boundary: a slot is only
    satisfied when the '/'-prefixed slot is a real suffix, so 'xprisma/...'
    does NOT satisfy 'prisma/...'."""
    assert match_produces_entry("prisma/schema.prisma",
                                {"backend/xprisma/schema.prisma"}) is False
    assert match_produces_entry("schema.prisma",
                                {"backend/myschema.prisma"}) is False


def test_exact_slot_still_matches_root_relative_write():
    assert match_produces_entry("prisma/schema.prisma",
                                {"prisma/schema.prisma"}) is True


def test_literal_file_slot_still_requires_exact_match():
    """Regression guard: a non-directory literal must NOT gain prefix semantics —
    `foo/bar.md` is not satisfied by `foo/bar.md.bak`."""
    written = {"mission_90/notes.md.bak"}
    assert match_produces_entry("mission_90/notes.md", written) is False
