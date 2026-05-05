"""Coverage: glob patterns + any_of (nested list) entries in verify_artifacts.

Stack-variant scaffolds (i2p_v3 7.4 db_setup, 7.6 test_infra) declare paths
the agent might emit under any of several frameworks (alembic vs prisma
vs drizzle, pytest vs jest vs vitest). The verb expands `*`-bearing
strings via glob and treats nested lists as any_of slots — at least one
alternative must match.
"""
from __future__ import annotations

import pytest

from salako.verify_artifacts import verify_artifacts


# ── glob patterns ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_glob_matches_existing_files(tmp_path):
    (tmp_path / "migrations").mkdir()
    (tmp_path / "migrations" / "versions").mkdir()
    (tmp_path / "migrations" / "versions" / "001_initial.py").write_text("# x\n")
    (tmp_path / "migrations" / "versions" / "002_add_users.py").write_text("# y\n")

    res = await verify_artifacts(
        mission_id=None,
        paths=["migrations/**/*.py"],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is True
    assert res["missing"] == []
    assert len(res["verified"]) == 2


@pytest.mark.asyncio
async def test_glob_with_no_matches_reported_missing(tmp_path):
    (tmp_path / "migrations").mkdir()  # empty dir
    res = await verify_artifacts(
        mission_id=None,
        paths=["migrations/**/*.py"],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False
    assert "migrations/**/*.py" in res["missing"]


@pytest.mark.asyncio
async def test_glob_excludes_directories(tmp_path):
    """`**/*` would otherwise hit dirs as well as files."""
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "y").mkdir()  # no files at all
    res = await verify_artifacts(
        mission_id=None,
        paths=["**/*"],
        workspace_path=str(tmp_path),
    )
    # Only directories exist — no file matches.
    assert "**/*" in res["missing"]


@pytest.mark.asyncio
async def test_glob_absolute_pattern_rejected(tmp_path):
    res = await verify_artifacts(
        mission_id=None,
        paths=["/etc/**/*"],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False
    assert "/etc/**/*" in res["missing"]


# ── any_of nested-list entries ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_any_of_passes_when_one_alternative_present(tmp_path):
    (tmp_path / "prisma").mkdir()
    (tmp_path / "prisma" / "schema.prisma").write_text("model X {}\n")

    res = await verify_artifacts(
        mission_id=None,
        paths=[
            ["migrations/versions/initial.py", "prisma/schema.prisma", "drizzle.config.ts"],
        ],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is True
    assert res["missing"] == []
    # The matched alternative is recorded with an any_of[ ] prefix.
    assert any("prisma/schema.prisma" in v["path"] for v in res["verified"])


@pytest.mark.asyncio
async def test_any_of_fails_when_no_alternative_present(tmp_path):
    res = await verify_artifacts(
        mission_id=None,
        paths=[
            ["migrations/versions/initial.py", "prisma/schema.prisma", "drizzle.config.ts"],
        ],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False
    assert any("any_of" in m for m in res["missing"])


@pytest.mark.asyncio
async def test_any_of_with_glob_alternative(tmp_path):
    (tmp_path / "migrations").mkdir()
    (tmp_path / "migrations" / "versions").mkdir()
    (tmp_path / "migrations" / "versions" / "001_initial.py").write_text("# x\n")

    res = await verify_artifacts(
        mission_id=None,
        paths=[
            ["migrations/**/*.py", "prisma/schema.prisma"],
        ],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is True
    assert any("migrations" in v["path"] for v in res["verified"])


@pytest.mark.asyncio
async def test_any_of_first_match_wins(tmp_path):
    """If two alternatives both match, the first listed wins (deterministic)."""
    (tmp_path / "a.py").write_text("# a\n")
    (tmp_path / "b.py").write_text("# b\n")

    res = await verify_artifacts(
        mission_id=None,
        paths=[["a.py", "b.py"]],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is True
    assert len(res["verified"]) == 1
    assert "any_of[a.py]" in res["verified"][0]["path"]


@pytest.mark.asyncio
async def test_mixed_string_glob_and_any_of(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python\n")
    (tmp_path / "migrations").mkdir()
    (tmp_path / "migrations" / "001.py").write_text("# x\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "seed.ts").write_text("export {}\n")

    res = await verify_artifacts(
        mission_id=None,
        paths=[
            "Dockerfile",                               # literal
            "migrations/*.py",                          # glob
            ["scripts/seed.py", "scripts/seed.ts"],     # any_of
        ],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is True
    assert res["missing"] == []
    paths = {v["path"] for v in res["verified"]}
    assert "Dockerfile" in paths


@pytest.mark.asyncio
async def test_unsupported_path_entry_type_fails(tmp_path):
    res = await verify_artifacts(
        mission_id=None,
        paths=[42, "ok.txt"],  # type: ignore[list-item]
        workspace_path=str(tmp_path),
    )
    (tmp_path / "ok.txt").write_text("hi\n")
    res = await verify_artifacts(
        mission_id=None,
        paths=[42],  # type: ignore[list-item]
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False
    assert any("unsupported" in f["reason"] for f in res["failed"])
