# packages/mr_roboto/tests/test_deploy_staging.py
import pytest
from mr_roboto.executors import deploy_staging as ds

@pytest.mark.asyncio
async def test_rejects_non_render_backend_arch():
    task = {"payload": {"action": "deploy_staging", "backend_arch": "serverless_workers",
                        "repo": "https://github.com/kutay/habithub.git", "workspace": "/tmp/x"}}
    res = await ds.run(task)
    assert res["ok"] is False and res["reason"] == "serverless_not_yet_supported"

@pytest.mark.asyncio
async def test_missing_params_fail_fast():
    res = await ds.run({"payload": {"action": "deploy_staging"}})
    assert res["ok"] is False and "missing" in res["reason"].lower()

def test_is_mocked_detects_tag():
    assert ds._is_mocked({"status": "ok", "data": {}, "mocked": True}) is True
    assert ds._is_mocked({"status": "ok", "data": {}}) is False
