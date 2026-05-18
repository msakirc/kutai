"""Index storage + read tests."""
import pytest

from yalayut.contracts import Manifest
from yalayut.index import store, read_all_enabled, get, embedding_to_blob, \
    blob_to_embedding

pytestmark = pytest.mark.asyncio


def _m(**o):
    base = dict(
        name="anthropics-pdf", name_original="pdf", version="1.0.0",
        artifact_type="skill", kind="prompt_skill",
        source="github:anthropics/skills@/skills", owner="anthropics",
        license="proprietary", intent_keywords=["pdf", "extract"],
    )
    base.update(o)
    return Manifest(**base)


async def test_blob_roundtrip():
    vec = [0.1, 0.2, -0.3]
    assert blob_to_embedding(embedding_to_blob(vec)) == pytest.approx(vec)


async def test_store_inserts_row(yalayut_db):
    aid = await store(
        yalayut_db, _m(), body="A skill about PDFs.", tier=0,
        audit={"source_max": 0, "check_maxes": {}},
        embedding=[0.1] * 768,
    )
    assert aid > 0
    row = await get(yalayut_db, aid)
    assert row.name == "anthropics-pdf"
    assert row.vet_tier == 0
    assert row.exposure_class == "inject"     # default for prompt_skill
    assert row.enabled is True


async def test_store_t3_is_disabled(yalayut_db):
    aid = await store(
        yalayut_db, _m(), body="x", tier=3, audit={}, embedding=[0.0] * 768,
    )
    row = await get(yalayut_db, aid)
    assert row.enabled is False
    assert row.exposure_class == "quarantine"


async def test_store_t2_quarantined_in_v1(yalayut_db):
    aid = await store(
        yalayut_db, _m(), body="x", tier=2, audit={}, embedding=[0.0] * 768,
    )
    row = await get(yalayut_db, aid)
    # v1: T2 quarantined-until-founder-promotes -> not enabled
    assert row.enabled is False


async def test_read_all_enabled_skips_disabled(yalayut_db):
    await store(yalayut_db, _m(version="1"), "x", 0, {}, [0.1] * 768)
    await store(yalayut_db, _m(version="2"), "x", 3, {}, [0.1] * 768)
    rows = await read_all_enabled(yalayut_db)
    assert len(rows) == 1
    assert rows[0].vet_tier == 0


async def test_store_upsert_on_conflict(yalayut_db):
    a1 = await store(yalayut_db, _m(), "x", 0, {}, [0.1] * 768)
    a2 = await store(yalayut_db, _m(), "y", 1, {}, [0.2] * 768)
    assert a1 == a2  # same (source,name,version) -> update in place
    row = await get(yalayut_db, a1)
    assert row.vet_tier == 1
