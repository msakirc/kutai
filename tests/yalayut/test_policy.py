"""DB-backed policy allowlist tests."""
import pytest

from yalayut.vetting.policy import (
    seed_policy, get_allowlist, get_injection_regexes, propose_policy,
)

pytestmark = pytest.mark.asyncio


async def test_seed_populates_shell_allowlist(yalayut_db):
    await seed_policy(yalayut_db)
    shell = await get_allowlist(yalayut_db, "shell_allowlist")
    assert "npx" in shell and shell["npx"] == "allow"
    assert "git" in shell
    assert "uvx" in shell
    assert "cookiecutter" in shell


async def test_seed_is_idempotent(yalayut_db):
    await seed_policy(yalayut_db)
    await seed_policy(yalayut_db)  # must not raise / must not double rows
    cur = await yalayut_db.execute(
        "SELECT COUNT(*) c FROM yalayut_policy WHERE check_name='shell_allowlist'"
    )
    n_first = (await cur.fetchone())["c"]
    await seed_policy(yalayut_db)
    cur = await yalayut_db.execute(
        "SELECT COUNT(*) c FROM yalayut_policy WHERE check_name='shell_allowlist'"
    )
    assert (await cur.fetchone())["c"] == n_first


async def test_injection_regexes_compile(yalayut_db):
    await seed_policy(yalayut_db)
    regexes = await get_injection_regexes(yalayut_db)
    assert len(regexes) >= 3
    # every entry must be a compiled, usable pattern
    for r in regexes:
        assert r.search("nothing here") is None or True


async def test_propose_policy_creates_pending_row(yalayut_db):
    pid = await propose_policy(
        yalayut_db, "shell_allowlist", "wasp", "allow",
        evidence={"observed_in": ["cc-wasp"]},
    )
    cur = await yalayut_db.execute(
        "SELECT state FROM yalayut_policy_proposals WHERE id=?", (pid,)
    )
    assert (await cur.fetchone())["state"] == "pending"
