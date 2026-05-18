"""Source/owner trust tier-cap tests."""
import pytest

from yalayut.trust import (
    SOURCE_MAX, OWNER_MAX, source_max_tier, owner_max_tier,
)

pytestmark = pytest.mark.asyncio


def test_source_max_constants():
    assert SOURCE_MAX["trusted"] == 0
    assert SOURCE_MAX["review"] == 1
    assert SOURCE_MAX["untrusted"] == 2


async def test_source_max_tier_trusted(yalayut_db):
    await yalayut_db.execute(
        "INSERT INTO yalayut_sources (source_id, trusted) VALUES (?, 1)",
        ("github:anthropics/skills@/skills",),
    )
    t = await source_max_tier(yalayut_db, "github:anthropics/skills@/skills")
    assert t == 0


async def test_source_max_tier_unknown_source_is_untrusted(yalayut_db):
    t = await source_max_tier(yalayut_db, "github:nobody/repo")
    assert t == 2


async def test_owner_max_tier_from_trust_score(yalayut_db):
    await yalayut_db.execute(
        "INSERT INTO yalayut_owners (owner_id, trust_score) VALUES (?, ?)",
        ("anthropics", 0.95),
    )
    await yalayut_db.execute(
        "INSERT INTO yalayut_owners (owner_id, trust_score) VALUES (?, ?)",
        ("sketchy", 0.2),
    )
    assert await owner_max_tier(yalayut_db, "anthropics") == 0
    assert await owner_max_tier(yalayut_db, "sketchy") == 2
    # unknown owner -> no elevation, weakest cap
    assert await owner_max_tier(yalayut_db, "ghost") == 3
