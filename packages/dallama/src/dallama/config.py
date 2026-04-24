"""Configuration dataclasses for DaLLaMa."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class DaLLaMaConfig:
    """Engine settings — configured once at startup."""
    llama_server_path: str = "llama-server"
    port: int = 8080
    host: str = "127.0.0.1"
    idle_timeout_seconds: float = 60.0
    circuit_breaker_threshold: int = 2
    circuit_breaker_cooldown_seconds: float = 300.0
    inference_drain_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 30.0
    health_fail_threshold: int = 3
    min_free_vram_mb: int = 4096
    on_ready: Callable[[str | None, str], None] | None = None
    get_vram_free_mb: Callable[[], int] | None = None

@dataclass
class ServerConfig:
    """Job description — what model to load and how."""
    model_path: str
    model_name: str
    context_length: int
    thinking: bool = False
    vision_projector: str = ""
    extra_flags: list[str] = field(default_factory=list)
    # Offload count to use only as a fallback if --fit-based load fails
    # with a CUDA OOM. First attempt uses llama.cpp's --fit (better in
    # the benchmark case, esp. for models that don't fully fit VRAM).
    # Set from fatih_hoca.registry.calculate_gpu_layers at load time.
    fallback_gpu_layers: int = 0
    # Estimated VRAM (MB) required to honour an explicit --n-gpu-layers
    # override (weights for the offloaded layers + KV cache + compute
    # buffer headroom).  Only consulted at swap-time when extra_flags
    # contains --n-gpu-layers.  If live free VRAM is below this value
    # the override is stripped and llama-server falls back to --fit.
    # Zero (default) disables the recheck — bare --fit loads are
    # unaffected since --fit already auto-sizes from live VRAM.
    required_vram_mb: int = 0

@dataclass
class ServerStatus:
    """What the dispatcher needs for routing decisions."""
    model_name: str | None
    healthy: bool
    busy: bool
    measured_tps: float
    context_length: int

@dataclass
class InferenceSession:
    """Context holder yielded by DaLLaMa.infer()."""
    url: str
    model_name: str

class DaLLaMaLoadError(RuntimeError):
    """Raised when DaLLaMa cannot load the requested model."""
    def __init__(self, model_name: str, reason: str = ""):
        msg = f"Failed to load model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.model_name = model_name
        self.reason = reason
