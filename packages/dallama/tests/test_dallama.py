"""Integration tests for the DaLLaMa main class."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from dallama import DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus, DaLLaMaLoadError

@pytest.fixture
def cfg():
    return DaLLaMaConfig(llama_server_path="/usr/bin/llama-server", port=8080, idle_timeout_seconds=60)

@pytest.fixture
def dallama(cfg):
    return DaLLaMa(cfg)

def test_initial_status(dallama):
    st = dallama.status
    assert st.model_name is None
    assert st.healthy is False
    assert st.busy is False
    assert st.measured_tps == 0.0
    assert st.context_length == 0

@pytest.mark.asyncio
async def test_infer_loads_model(dallama):
    config = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096)
    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True):
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config) as session:
                    assert session.url == "http://127.0.0.1:8080"
                    assert session.model_name == "test"
    assert dallama.status.model_name == "test"

@pytest.mark.asyncio
async def test_infer_same_model_no_swap(dallama):
    config = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096)
    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True) as mock_swap:
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config):
                    pass
                async with dallama.infer(config):
                    pass
    assert mock_swap.call_count == 1

@pytest.mark.asyncio
async def test_infer_different_model_triggers_swap(dallama):
    config1 = ServerConfig(model_path="/m/a.gguf", model_name="model-a", context_length=4096)
    config2 = ServerConfig(model_path="/m/b.gguf", model_name="model-b", context_length=8192)
    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True) as mock_swap:
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config1):
                    pass
                async with dallama.infer(config2):
                    pass
    assert mock_swap.call_count == 2

@pytest.mark.asyncio
async def test_infer_thinking_change_triggers_swap(dallama):
    cfg_off = ServerConfig(model_path="/m/a.gguf", model_name="model-a", context_length=4096, thinking=False)
    cfg_on = ServerConfig(model_path="/m/a.gguf", model_name="model-a", context_length=4096, thinking=True)
    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True) as mock_swap:
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(cfg_off):
                    pass
                async with dallama.infer(cfg_on):
                    pass
    assert mock_swap.call_count == 2

@pytest.mark.asyncio
async def test_infer_raises_on_load_failure(dallama):
    config = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096)
    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=False):
        with pytest.raises(DaLLaMaLoadError, match="test"):
            async with dallama.infer(config):
                pass

@pytest.mark.asyncio
async def test_infer_tracks_inflight(dallama):
    config = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096)
    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True):
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config):
                    assert dallama.status.busy is True
                assert dallama.status.busy is False

def test_keep_alive(dallama):
    dallama.keep_alive()

@pytest.mark.asyncio
async def test_start_stop(dallama):
    with patch.object(dallama._platform, "kill_orphans"):
        await dallama.start()
    with patch.object(dallama._server, "stop", new_callable=AsyncMock):
        await dallama.stop()
