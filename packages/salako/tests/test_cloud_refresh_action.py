from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_handle_cloud_refresh_invokes_discovery_and_returns_summary():
    """salako.cloud_refresh.run() drives fatih_hoca discovery + bench refresh
    and returns a {providers_probed, providers_ok, models_registered} summary."""
    from salako.cloud_refresh import run

    fake_summary = {
        "providers_probed": 3,
        "providers_ok": 2,
        "models_registered": 27,
    }
    with patch("salako.cloud_refresh._refresh_cloud_subsystem", new=AsyncMock(return_value=fake_summary)) as m:
        result = await run({"id": 1, "payload": {"action": "cloud_refresh"}})
    m.assert_awaited_once()
    assert result == fake_summary


@pytest.mark.asyncio
async def test_run_dispatch_routes_cloud_refresh_branch():
    """salako.run() must route action='cloud_refresh' to the new handler."""
    from salako import run as salako_run

    with patch("salako.cloud_refresh.run", new=AsyncMock(return_value={"providers_probed": 0, "providers_ok": 0, "models_registered": 0})) as m:
        action = await salako_run({"id": 1, "payload": {"action": "cloud_refresh"}})
    m.assert_awaited_once()
    assert action.status == "completed"
    assert action.result == {"providers_probed": 0, "providers_ok": 0, "models_registered": 0}
