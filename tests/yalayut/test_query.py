"""query() vector-similarity tests."""
import pytest

from yalayut.contracts import Manifest, TaskContext, Artifact
from yalayut.index import store
from yalayut._query_engine import query_db

pytestmark = pytest.mark.asyncio


def _m(name, version="1.0.0"):
    return Manifest(
        name=name, name_original=name, version=version,
        artifact_type="skill", kind="prompt_skill",
        source="github:anthropics/skills@/skills", owner="anthropics",
        license="MIT",
    )


async def test_query_ranks_by_cosine(yalayut_db):
    # craft two artifacts with deliberately different embeddings
    await store(yalayut_db, _m("pdf-skill"), "pdf body", 0, {},
                embedding=[1.0, 0.0] + [0.0] * 766)
    await store(yalayut_db, _m("excel-skill"), "excel body", 0, {},
                embedding=[0.0, 1.0] + [0.0] * 766)
    ctx = TaskContext(title="convert pdf")
    results = await query_db(
        yalayut_db, ctx, query_embedding=[1.0, 0.0] + [0.0] * 766,
    )
    assert results[0].name == "pdf-skill"
    assert results[0].score > results[1].score
    assert all(isinstance(r, Artifact) for r in results)


async def test_query_skips_disabled(yalayut_db):
    await store(yalayut_db, _m("good"), "x", 0, {}, [1.0] + [0.0] * 767)
    await store(yalayut_db, _m("bad"), "x", 3, {}, [1.0] + [0.0] * 767)
    results = await query_db(
        yalayut_db, TaskContext(title="anything"),
        query_embedding=[1.0] + [0.0] * 767,
    )
    assert {r.name for r in results} == {"good"}


async def test_query_respects_top_k(yalayut_db):
    for i in range(10):
        await store(yalayut_db, _m(f"s{i}"), "x", 0, {},
                    [float(i)] + [0.0] * 767)
    results = await query_db(
        yalayut_db, TaskContext(title="x"),
        query_embedding=[5.0] + [0.0] * 767, top_k=3,
    )
    assert len(results) == 3


async def test_query_empty_index_returns_empty(yalayut_db):
    results = await query_db(
        yalayut_db, TaskContext(title="x"), query_embedding=[1.0] * 768,
    )
    assert results == []
