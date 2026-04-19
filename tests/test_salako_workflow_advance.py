import pytest
from salako.workflow_advance import run


@pytest.mark.asyncio
async def test_missing_payload_fails():
    r = await run({"id": 1, "payload": {}})
    assert r["status"] == "failed"


@pytest.mark.asyncio
async def test_payload_with_str_json():
    # String payload is valid JSON; shouldn't explode. The advance call
    # will likely fail with no mission in the test DB, but we're testing
    # the payload-parsing path only.
    import json
    r = await run({"id": 1, "payload": json.dumps({})})
    assert r["status"] == "failed"
