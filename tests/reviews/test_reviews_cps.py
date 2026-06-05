import json
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_enqueue_classify_builds_overhead_child_with_continuation():
    from src.reviews import producers

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return 4242

    fake_row = (7, "g2", "Ada", 1, "It crashes on save")

    class _Cur:
        async def fetchone(self):
            return fake_row

    class _DB:
        async def execute(self, *a, **k):
            return _Cur()

    async def fake_get_db():
        return _DB()

    with patch.object(producers, "enqueue", fake_enqueue), \
         patch("src.infra.db.get_db", fake_get_db):
        tid = await producers.enqueue_classify(review_id=7, product_id="prod-x")

    assert tid == 4242
    k = captured["kwargs"]
    assert k["on_complete"] == "reviews.classify.resume"
    assert k["on_error"] == "reviews.classify.resume_err"
    assert k["lane"] == "oneshot"
    st = k["cont_state"]
    assert st["review_id"] == 7 and st["rating"] == 1 and st["body_md"]
    llm = captured["spec"]["context"]["llm_call"]
    assert llm["raw_dispatch"] is True and llm["call_category"] == "overhead"
    assert "crashes" in llm["messages"][0]["content"]
