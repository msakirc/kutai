"""Seed data + seed manifest tests."""
import pytest

from yalayut.manifest import parse_manifest_yaml, validate_manifest
from yalayut.seed.seed_data import (
    seed_owners, seed_sources, seed_disabled_imports, load_seed_manifests,
)

pytestmark = pytest.mark.asyncio


def test_exactly_20_seed_manifests():
    manifests = load_seed_manifests()
    assert len(manifests) == 20


def test_every_seed_manifest_is_valid():
    for fname, text in load_seed_manifests():
        m = parse_manifest_yaml(text)
        errs = validate_manifest(m)
        assert errs == [], f"{fname}: {errs}"


def test_shell_recipes_have_invocation_steps():
    for fname, text in load_seed_manifests():
        m = parse_manifest_yaml(text)
        if m.kind == "shell_recipe":
            assert m.mechanizable is True, fname
            assert m.invocation.get("steps"), fname


async def test_seed_owners_idempotent(yalayut_db):
    n1 = await seed_owners(yalayut_db)
    assert n1 == 7
    n2 = await seed_owners(yalayut_db)
    assert n2 == 0


async def test_seed_sources_all_cron_trusted(yalayut_db):
    await seed_sources(yalayut_db)
    cur = await yalayut_db.execute(
        "SELECT COUNT(*) c FROM yalayut_sources "
        "WHERE discovery_mode='cron' AND trusted=1"
    )
    assert (await cur.fetchone())["c"] == 4


async def test_seed_disabled_imports(yalayut_db):
    await seed_disabled_imports(yalayut_db)
    cur = await yalayut_db.execute(
        "SELECT artifact_name FROM yalayut_disabled_imports"
    )
    names = {r["artifact_name"] for r in await cur.fetchall()}
    assert "using-superpowers" in names
    assert "using-git-worktrees" in names
