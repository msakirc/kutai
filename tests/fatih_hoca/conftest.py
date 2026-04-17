# tests/fatih_hoca/conftest.py
"""Shared fixtures for fatih_hoca tests: canned AA cache, fake Nerd Herd, registry."""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pytest

from fatih_hoca.registry import ModelInfo, ModelRegistry
from nerd_herd.types import SystemSnapshot, LocalModelState


@pytest.fixture
def canned_aa_cache(tmp_path: Path) -> Path:
    """Write a realistic AA bulk cache with the Qwen3 trio + a coder fine-tune + a dense small model."""
    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    payload = {
        "timestamp": time.time(),  # fresh
        "models": {
            # Base Qwen3 30B A3B MoE (thinking variant) — strong reasoning, weak coding
            "qwen3-30b-a3b-instruct::thinking": {
                "reasoning": 7.5, "analysis": 7.0, "domain_knowledge": 6.5,
                "code_generation": 5.0, "code_reasoning": 5.5,
                "instruction_adherence": 7.0,
            },
            # Dense 32B Qwen3 — balanced, mid coder
            "qwen3-32b-instruct": {
                "reasoning": 7.0, "analysis": 6.8, "domain_knowledge": 6.5,
                "code_generation": 6.5, "code_reasoning": 6.5,
                "instruction_adherence": 7.0,
            },
            # Qwen3 Coder 480B — strong coder (this is the ONLY AA coder variant; no 30B-coder exists)
            "qwen3-coder-480b-a35b-instruct": {
                "reasoning": 7.5, "analysis": 7.2, "domain_knowledge": 7.0,
                "code_generation": 9.0, "code_reasoning": 8.8,
                "instruction_adherence": 7.5,
            },
            # Llama 3.3 70B cloud
            "llama-3-3-70b-instruct": {
                "reasoning": 7.0, "code_generation": 6.0, "domain_knowledge": 7.5,
                "instruction_adherence": 7.2,
            },
        },
    }
    cache_path = cache_dir / "_bulk_artificial_analysis.json"
    cache_path.write_text(json.dumps(payload))
    return cache_dir


@dataclass
class FakeNerdHerd:
    """Minimal Nerd Herd stub that returns a controllable SystemSnapshot."""
    loaded_model: Optional[str] = None
    measured_tps: float = 0.0
    vram_available_mb: int = 24000

    def snapshot(self) -> SystemSnapshot:
        snap = SystemSnapshot(vram_available_mb=self.vram_available_mb)
        snap.local = LocalModelState(
            model_name=self.loaded_model,
            measured_tps=self.measured_tps,
        )
        return snap


@pytest.fixture
def fake_nerd_herd() -> FakeNerdHerd:
    return FakeNerdHerd()


@pytest.fixture
def registry_with_qwen_trio() -> ModelRegistry:
    """Registry seeded with three Qwen3 local models covering the disambiguation trap."""
    reg = ModelRegistry()
    reg._models["qwen3-30b-a3b"] = ModelInfo(
        name="qwen3-30b-a3b",
        location="local",
        provider="llama_cpp",
        litellm_name="openai/qwen3-30b-a3b",
        path="/fake/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf",
        total_params_b=30.0,
        active_params_b=3.0,
        family="qwen3",
        capabilities={"reasoning": 6.0, "code_generation": 5.0},
        thinking_model=True,
    )
    reg._models["qwen3-coder-30b"] = ModelInfo(
        name="qwen3-coder-30b",
        location="local",
        provider="llama_cpp",
        litellm_name="openai/qwen3-coder-30b",
        path="/fake/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        total_params_b=30.0,
        active_params_b=3.0,
        family="qwen3-coder",
        capabilities={"reasoning": 6.0, "code_generation": 7.5},
        specialty="coding",
    )
    reg._models["qwen3-32b"] = ModelInfo(
        name="qwen3-32b",
        location="local",
        provider="llama_cpp",
        litellm_name="openai/qwen3-32b",
        path="/fake/Qwen3-32B-Instruct-Q4_K_M.gguf",
        total_params_b=32.0,
        active_params_b=32.0,
        family="qwen3",
        capabilities={"reasoning": 6.5, "code_generation": 6.0},
    )
    return reg
