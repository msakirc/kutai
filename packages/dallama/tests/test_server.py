"""Tests for ServerProcess — cmd building, start/stop, health."""
import os
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from dallama.config import DaLLaMaConfig, ServerConfig
from dallama.platform import PlatformHelper
from dallama.server import ServerProcess

@pytest.fixture
def dallama_cfg():
    return DaLLaMaConfig(llama_server_path="/usr/bin/llama-server", port=8080, host="127.0.0.1")

@pytest.fixture
def platform_helper():
    return PlatformHelper()

@pytest.fixture
def server(dallama_cfg, platform_helper):
    return ServerProcess(dallama_cfg, platform_helper)

def test_build_cmd_minimal(server):
    cfg = ServerConfig(model_path="/models/test.gguf", model_name="test", context_length=4096)
    cmd = server.build_cmd(cfg)
    assert "/usr/bin/llama-server" == cmd[0]
    assert "--model" in cmd
    assert "/models/test.gguf" in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "--ctx-size" in cmd
    assert "4096" in cmd
    assert "--metrics" in cmd
    assert "--jinja" in cmd
    assert "--reasoning" not in cmd

def test_build_cmd_thinking_on(server):
    cfg = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=8192, thinking=True)
    cmd = server.build_cmd(cfg)
    idx = cmd.index("--reasoning")
    assert cmd[idx + 1] == "on"

def test_build_cmd_thinking_off_no_flags(server):
    """When thinking=False, no reasoning flags at all."""
    cfg = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=8192, thinking=False)
    cmd = server.build_cmd(cfg)
    assert "--reasoning" not in cmd

def test_build_cmd_no_jinja_skips_reasoning(server):
    cfg = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096,
                       thinking=True, extra_flags=["--no-jinja"])
    cmd = server.build_cmd(cfg)
    assert "--reasoning" not in cmd
    assert "--jinja" not in cmd
    assert "--no-jinja" in cmd

def test_build_cmd_vision(server):
    cfg = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096,
                       vision_projector="/m/mmproj.gguf")
    cmd = server.build_cmd(cfg)
    assert "--mmproj" in cmd
    assert "/m/mmproj.gguf" in cmd

def test_build_cmd_extra_flags(server):
    cfg = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096,
                       extra_flags=["--chat-template", "chatml", "--override-kv", "key=val"])
    cmd = server.build_cmd(cfg)
    assert "--chat-template" in cmd
    assert "chatml" in cmd

def test_is_alive_no_process(server):
    assert server.is_alive() is False

@pytest.mark.asyncio
async def test_health_check_no_process(server):
    assert await server.health_check() is False

@pytest.mark.asyncio
async def test_stop_no_process(server):
    await server.stop()
