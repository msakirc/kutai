# packages/mr_roboto/tests/test_deploy_util.py
import pytest
from mr_roboto.deploy_util import poll_until

@pytest.mark.asyncio
async def test_poll_ready_on_third_call():
    seq = iter(["BUILDING", "BUILDING", "READY"])
    async def fetch(): return {"state": next(seq)}
    res = await poll_until(fetch, ready=lambda r: r["state"] == "READY",
                           fail=lambda r: r["state"] == "ERROR",
                           max_wait_s=5, base_delay_s=0)
    assert res["ok"] and res["result"]["state"] == "READY"

@pytest.mark.asyncio
async def test_poll_terminal_fail_aborts_early():
    async def fetch(): return {"state": "ERROR"}
    res = await poll_until(fetch, ready=lambda r: r["state"] == "READY",
                           fail=lambda r: r["state"] == "ERROR",
                           max_wait_s=5, base_delay_s=0)
    assert res["ok"] is False and res["reason"] == "terminal_fail"

@pytest.mark.asyncio
async def test_poll_timeout():
    async def fetch(): return {"state": "BUILDING"}
    res = await poll_until(fetch, ready=lambda r: r["state"] == "READY",
                           fail=lambda r: False, max_wait_s=0.1, base_delay_s=0.05)
    assert res["ok"] is False and res["reason"] == "timeout"
