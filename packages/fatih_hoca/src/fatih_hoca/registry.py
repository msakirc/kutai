# registry.py
"""
Model Registry — ModelInfo dataclass, GGUF scanning, YAML parsing,
capability estimation, and the ModelRegistry catalog.

Extracted from src/models/model_registry.py into the fatih_hoca package.
GPU info comes through Nerd Herd snapshots; benchmark enrichment is a no-op
stub (to be wired later).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from fatih_hoca.capabilities import (
    ALL_CAPABILITIES,
    Cap,
    EXECUTION_DIMENSIONS,
    KNOWLEDGE_DIMENSIONS,
    REASONING_DIMENSIONS,
    TaskRequirements,
    rank_models_for_task,
    score_model_for_task,
)
from fatih_hoca.profiles import (
    CLOUD_PROFILES,
    FAMILY_PROFILES,
    detect_family,
    get_default_profile,
    get_quant_retention,
    interpolate_size_multiplier,
)

logger = logging.getLogger("fatih_hoca.registry")

# ─── ModelInfo ───────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Unified model descriptor — local, cloud, and ollama."""
    name: str
    location: str                           # "local" | "cloud" | "ollama"
    provider: str                           # "llama_cpp" | "gemini" | "anthropic" | "ollama" ...
    litellm_name: str
    capabilities: dict[str, float] = field(default_factory=dict)  # 14-dimension scores, 0.0-10.0
    context_length: int = 8192
    max_tokens: int = 4096
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    # Strict ``response_format: json_schema`` (token-level constrained
    # decoding). Strict superset of ``supports_json_mode``: every model
    # with json_schema also supports json_object, but not vice versa.
    # Local llama.cpp / Ollama / OpenAI gpt-4o / Gemini 1.5+ are TRUE;
    # Claude / Groq / Cerebras / Sambanova are FALSE (some have it on
    # specific newer models — verify per-model before enabling).
    supports_json_schema: bool = False
    thinking_model: bool = False
    has_vision: bool = False
    mmproj_path: Optional[str] = None  # path to vision projector GGUF

    # Cloud-specific
    tier: str = "free"
    rate_limit_rpm: int = 30
    rate_limit_tpm: int = 100000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    # Local-specific
    path: Optional[str] = None
    model_type: str = "dense"               # "dense" | "moe"
    total_params_b: float = 0.0
    active_params_b: float = 0.0
    quantization: str = ""
    gpu_layers: int = 0
    _gpu_layers_from_override: bool = False  # True if set via models.yaml
    total_layers: int = 0
    file_size_mb: float = 0.0
    tokens_per_second: float = 0.0          # measured generation tokens/sec
    pp_tps: float = 0.0                     # measured prompt-processing tokens/sec
    load_time_seconds: float = 30.0
    priority_class: str = "primary"
    specialty: str = ""
    family: str = ""                        # detected family key

    # Per-model sampling overrides (from models.yaml)
    # Keys are task types or "default"; values are dicts of sampling params.
    # Example: {"default": {"temperature": 0.3}, "coding": {"temperature": 0.1}}
    sampling_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    # Per-model llama-server extra flags (from models.yaml or family detection)
    # Example: ["--no-jinja", "--chat-template", "chatml"]
    extra_server_flags: list[str] = field(default_factory=list)

    # Score provenance (for auto-tuner blending)
    profile_scores: dict[str, float] = field(default_factory=dict)
    benchmark_scores: dict[str, float] = field(default_factory=dict)

    # Runtime state
    is_loaded: bool = False
    demoted: bool = False
    api_base: Optional[str] = None

    # Variant fields
    is_variant: bool = False
    base_model_name: str = ""
    variant_flags: set[str] = field(default_factory=set)

    def score_for(self, cap: str) -> float:
        """Get score for a single capability."""
        return self.capabilities.get(cap, 0.0)

    def best_score(self) -> float:
        return max(self.capabilities.values()) if self.capabilities else 0.0

    def estimated_cost(self, input_tokens: int, output_tokens: int) -> float:
        if self.location == "local":
            return 0.0
        return (
            (input_tokens / 1000) * self.cost_per_1k_input
            + (output_tokens / 1000) * self.cost_per_1k_output
        )

    def operational_dict(self) -> dict:
        """Return operational metadata for scoring functions."""
        return {
            "location": self.location,
            "provider": self.provider,
            "context_length": self.context_length,
            "max_tokens": self.max_tokens,
            "supports_function_calling": self.supports_function_calling,
            "supports_json_mode": self.supports_json_mode,
            "supports_json_schema": self.supports_json_schema,
            "thinking_model": self.thinking_model,
            "has_vision": self.has_vision,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "tokens_per_second": self.tokens_per_second,
            "tier": self.tier,
            "rate_limit_rpm": self.rate_limit_rpm,
            "model_type": self.model_type,
            "total_params_b": self.total_params_b,
            "active_params_b": self.active_params_b,
            "is_variant": getattr(self, "is_variant", False),
            "variant_flags": getattr(self, "variant_flags", set()),
        }

    @property
    def is_local(self) -> bool:
        return self.location in ("local", "ollama")

    @property
    def is_free(self) -> bool:
        return self.location in ("local", "ollama") or self.tier == "free"

    def effective_context_at_current_vram(
        self,
        available_vram_mb: int,
        available_ram_mb: int = 65536,
    ) -> int:
        """Return the max context Hoca should assume for this model given
        current VRAM (and RAM) headroom.

        Cloud/ollama models: returns the trained-native `context_length`.
        Local GGUF models: delegates to `calculate_dynamic_context` using
        the model's file_size/n_layers/gpu_layers against live memory.
        The returned value is capped at `context_length` (capability
        ceiling) and rounded to the calculator's 2048-block granularity.

        This is the context selector eligibility should filter on — static
        `context_length` is the trained ceiling, but a tight-VRAM runtime
        may only fit a smaller window; filtering on the static value lets
        a model win and then fail at load time.
        """
        if not self.is_local or self.location == "ollama":
            return self.context_length
        if self.file_size_mb <= 0 or self.total_layers <= 0:
            return self.context_length
        try:
            effective = calculate_dynamic_context(
                file_size_mb=self.file_size_mb,
                n_layers=self.total_layers,
                gpu_layers=self.gpu_layers,
                available_ram_mb=max(0, int(available_ram_mb)),
                available_vram_mb=max(0, int(available_vram_mb)),
                family_key=self.family or None,
            )
        except Exception:
            return self.context_length
        # Never exceed the trained-native ceiling.
        return max(0, min(int(effective), int(self.context_length)))


# ─── GGUF Metadata Reader ───────────────────────────────────────────────────

# Cache GGUF metadata to avoid re-reading large files on every startup.
# Key: (path, file_size, mtime) → metadata dict
_GGUF_CACHE_FILE = Path(os.getenv("MODEL_DIR", "")) / ".gguf_metadata_cache.json" \
    if os.getenv("MODEL_DIR") else None

_gguf_cache: dict[str, dict] = {}
_gguf_cache_loaded = False

def _load_gguf_cache():
    global _gguf_cache, _gguf_cache_loaded
    if _gguf_cache_loaded:
        return
    _gguf_cache_loaded = True
    if _GGUF_CACHE_FILE and _GGUF_CACHE_FILE.exists():
        try:
            import json as _json
            _gguf_cache = _json.loads(_GGUF_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _gguf_cache = {}

def _save_gguf_cache():
    if not _GGUF_CACHE_FILE:
        return
    try:
        import json as _json
        _GGUF_CACHE_FILE.write_text(
            _json.dumps(_gguf_cache, indent=1), encoding="utf-8"
        )
    except Exception:
        pass

def _gguf_cache_key(path: str) -> str:
    """Cache key based on path + size + mtime."""
    try:
        st = os.stat(path)
        return f"{path}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return ""


def read_gguf_metadata(path: str) -> dict:
    """Read key metadata from a GGUF file header. Uses disk cache."""
    _load_gguf_cache()

    cache_key = _gguf_cache_key(path)
    if cache_key and cache_key in _gguf_cache:
        logger.debug(f"GGUF cache hit: {Path(path).name}")
        return dict(_gguf_cache[cache_key])

    metadata = _read_gguf_metadata_uncached(path)

    if cache_key and metadata:
        _gguf_cache[cache_key] = metadata
        _save_gguf_cache()

    return metadata


def _read_gguf_metadata_uncached(path: str) -> dict:
    """Read key metadata from a GGUF file header (no cache)."""
    metadata = {}
    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)

        for f in reader.fields.values():
            name = f.name
            if name == "general.architecture":
                metadata["architecture"] = str(f.parts[-1][0], "utf-8") if f.parts else ""
            elif name == "general.name":
                metadata["model_name"] = str(f.parts[-1][0], "utf-8") if f.parts else ""
            elif name.endswith(".context_length"):
                metadata["context_length"] = int(f.parts[-1][0])
            elif name.endswith(".block_count"):
                metadata["n_layers"] = int(f.parts[-1][0])
            elif name.endswith(".expert_count"):
                metadata["expert_count"] = int(f.parts[-1][0])
            elif name.endswith(".expert_used_count"):
                metadata["expert_used_count"] = int(f.parts[-1][0])
            elif name == "general.file_type":
                metadata["file_type"] = int(f.parts[-1][0])
            elif "clip" in name.lower() or "vision" in name.lower():
                metadata["has_vision_metadata"] = True
            elif "mmproj" in name.lower() or "image" in name.lower():
                metadata["has_vision_metadata"] = True

        metadata["file_size_mb"] = os.path.getsize(path) / (1024 * 1024)

    except ImportError:
        logger.warning("gguf package not installed — falling back to filename estimation")
        metadata = _estimate_from_filename(path)
    except Exception as e:
        logger.warning(f"Failed to read GGUF metadata from {path}: {e}")
        metadata = _estimate_from_filename(path)

    return metadata


def _estimate_from_filename(path: str) -> dict:
    """Estimate model properties from filename when GGUF parsing fails."""
    name = Path(path).stem.lower()
    metadata = {}

    try:
        metadata["file_size_mb"] = os.path.getsize(path) / (1024 * 1024)
    except Exception:
        metadata["file_size_mb"] = 0

    # Parameter count: match patterns like "30b", "3.8b", "30B-A3B"
    param_match = re.search(r"(?<![a-z])(\d+\.?\d*)b(?![a-z])", name)
    if param_match:
        metadata["estimated_params_b"] = float(param_match.group(1))

    # MoE active params: "A3B" pattern
    moe_match = re.search(r"a(\d+\.?\d*)b", name)
    if moe_match:
        metadata["active_params_b"] = float(moe_match.group(1))
        metadata["is_moe"] = True

    # Quantization
    quant_match = re.search(
        r"(iq[1-4]_[a-z]+|q[2-8]_k_[a-z]+|q[2-8]_k|q[2-8]_0|q[2-8]|f16|f32|fp16|bf16)",
        name,
        re.IGNORECASE,
    )
    if quant_match:
        metadata["quantization"] = quant_match.group(1).upper()

    # Architecture hints from name
    for arch in [
        "qwen3", "qwen2.5", "qwen2", "llama", "phi4", "phi3", "phi",
        "gemma3", "gemma2", "gemma", "mistral", "mixtral", "deepseek",
        "codellama", "starcoder", "llava", "moondream", "internvl",
        "minicpm", "command-r", "yi", "internlm", "qwq",
        "gigachat", "apriel", "gpt-oss", "gemma-4", "gemma4",
    ]:
        if arch in name:
            metadata["architecture"] = arch
            break

    # Specialty hints
    if "coder" in name or "code" in name:
        metadata["specialty"] = "coding"
    if "vision" in name or "vl" in name or "llava" in name or "moondream" in name:
        metadata["specialty"] = "vision"

    return metadata


# ─── Capability Estimation Engine ────────────────────────────────────────────

def estimate_capabilities(
    family_key: str | None,
    total_params_b: float,
    active_params_b: float | None,
    quantization: str,
    is_moe: bool = False,
    has_vision_hint: bool = False,
) -> dict[str, float]:
    """
    Estimate 15-dimension capability scores from model metadata.

    Strategy:
    1. Look up family profile (or use default)
    2. Calculate size multiplier using effective params per dimension type
    3. Apply quantization retention factor
    4. Clamp all scores to [0.0, 10.0]
    """
    # Get family profile
    if family_key and family_key in FAMILY_PROFILES:
        profile = FAMILY_PROFILES[family_key]
    elif family_key is None:
        # Unknown family — stub: benchmark enrichment is a no-op for now
        profile = get_default_profile()
    else:
        profile = get_default_profile()

    base_caps = profile.base_capabilities
    anchor = profile.anchor_params_b
    quant_retention = get_quant_retention(quantization)

    result: dict[str, float] = {}

    for dim_name in ALL_CAPABILITIES:
        base_score = base_caps.get(dim_name, 0.0)

        # Skip vision if family doesn't support it (unless hint says otherwise)
        if dim_name == "vision" and base_score == 0.0 and not has_vision_hint:
            result[dim_name] = 0.0
            continue

        # Determine effective param count for this dimension type
        dim_cap = Cap(dim_name)
        if is_moe and active_params_b and active_params_b > 0:
            if dim_cap in KNOWLEDGE_DIMENSIONS:
                # Knowledge stored in all params
                effective_params = total_params_b
            elif dim_cap in REASONING_DIMENSIONS:
                # Reasoning: geometric mean of active and total
                effective_params = math.sqrt(active_params_b * total_params_b)
            else:
                # Execution: active params
                effective_params = active_params_b
        else:
            effective_params = total_params_b

        # Size scaling relative to anchor
        # We compute multiplier for both actual and anchor, then take ratio
        actual_mult = interpolate_size_multiplier(effective_params)
        anchor_mult = interpolate_size_multiplier(anchor)

        if anchor_mult > 0:
            size_ratio = actual_mult / anchor_mult
        else:
            size_ratio = 1.0

        # Apply scaling
        scaled = base_score * size_ratio * quant_retention

        # Clamp
        result[dim_name] = round(max(0.0, min(10.0, scaled)), 1)

    return result


# ─── Dynamic Context Calculator ────────────────────────────────────────────

def calculate_dynamic_context(
    file_size_mb: float,
    n_layers: int,
    gpu_layers: int,
    available_ram_mb: int,
    available_vram_mb: int,
    family_key: str | None = None,
    thinking: bool = False,
) -> int:
    """Calculate max context length based on available memory.

    KV cache memory grows linearly with context length. We estimate how
    much memory is left after loading model weights and pick the largest
    context that fits (capped at the family default).

    When ``thinking=True``, the reserve budget is bumped to account for the
    larger activation / scratch footprint observed when reasoning is on
    (Qwen3.5-9B-thinking: sched_reserve GGML_ASSERT at 24576 ctx with the
    default 1300 MB reserve; 8GB GPU, 2026-04-23).
    """
    if n_layers <= 0 or file_size_mb <= 0:
        return 8192  # safe minimum

    # Family default context cap
    max_ctx = 32768
    if family_key and family_key in FAMILY_PROFILES:
        max_ctx = FAMILY_PROFILES[family_key].context_default

    # Estimate memory used by model weights (split across GPU/CPU)
    weight_per_layer_mb = file_size_mb / n_layers
    vram_used_by_weights = gpu_layers * weight_per_layer_mb
    ram_used_by_weights = (n_layers - gpu_layers) * weight_per_layer_mb

    # Available memory after weights. Reserve covers:
    #   ~500 MB CUDA baseline (driver, output buffer)
    #   ~500 MB compute buffer (depends on batch size; 493 MB observed at b=512)
    #   ~300 MB warmup MUL_MAT transient peak
    # Total: 1300 MB. The old 500 MB margin caused OOM at warmup
    # (Qwen3.5-9B observed 2026-04-22: 5974/6206 MiB committed pre-warmup,
    # warmup pushed it over).
    #
    # Thinking mode carries additional activation scratch (~500 MB observed
    # for Qwen3.5-9B-thinking at b=512). Bump the reserve so the compute
    # buffer sched_reserve doesn't abort with GGML_ASSERT(mem_buffer).
    VRAM_RESERVE_MB = 1800 if thinking else 1300
    free_vram = max(0, available_vram_mb - vram_used_by_weights - VRAM_RESERVE_MB) * 0.85
    free_ram = max(0, available_ram_mb - ram_used_by_weights - 2000) * 0.70

    # KV cache cost. Old 0.5 MB/layer/1k underestimates grouped-query
    # attention models (Qwen3.5-9B: actual 32 MB/1k vs old estimate 18).
    # Use 1.0 MB/layer/1k as the default — covers GQA without over-
    # reserving for non-GQA architectures. Conservative since underestimate
    # → 32k picked → load OOM, while overestimate just shrinks context.
    KV_PER_LAYER_PER_1K_MB = 1.0
    if n_layers <= 0:
        return max_ctx

    # KV is partitioned by tier: GPU layers' KV consumes VRAM; CPU layers'
    # KV consumes RAM. Old code summed free_vram+free_ram, which let RAM
    # headroom mask insufficient VRAM and picked a 32k context that
    # OOMed at warmup. Bound by the tightest tier.
    kv_per_1k_gpu_mb = gpu_layers * KV_PER_LAYER_PER_1K_MB
    kv_per_1k_cpu_mb = max(0, n_layers - gpu_layers) * KV_PER_LAYER_PER_1K_MB

    if kv_per_1k_gpu_mb > 0:
        max_ctx_from_vram = int((free_vram / kv_per_1k_gpu_mb) * 1024)
    else:
        max_ctx_from_vram = 10**9
    if kv_per_1k_cpu_mb > 0:
        max_ctx_from_ram = int((free_ram / kv_per_1k_cpu_mb) * 1024)
    else:
        max_ctx_from_ram = 10**9
    max_ctx_from_memory = min(max_ctx_from_vram, max_ctx_from_ram)

    # Hard cap: 32K for all local models unless overridden in models.yaml.
    # llama-server pre-allocates KV cache for the full context window —
    # 131K eats all VRAM even for a 361-token prompt. No agent task needs
    # more than ~4K tokens, so 32K is generous with room for tool outputs.
    # Models with >50% GPU can go up to 32K; CPU-heavy models get 16K.
    gpu_ratio = gpu_layers / max(n_layers, 1)
    if gpu_ratio < 0.3:
        hard_cap = 8192
    elif gpu_ratio <= 0.5:
        hard_cap = 16384
    else:
        hard_cap = 32768
    max_ctx = min(max_ctx, hard_cap)

    # Round down to nearest 2048 and clamp
    result = (max_ctx_from_memory // 2048) * 2048
    return max(4096, min(result, max_ctx))


# ─── GPU Layer Calculator ───────────────────────────────────────────────────

def calculate_gpu_layers(
    file_size_mb: float,
    n_layers: int,
    available_vram_mb: int,
    context_length: int = 8192,
) -> int:
    """Calculate how many layers can fit in VRAM."""
    if n_layers <= 0 or file_size_mb <= 0 or available_vram_mb <= 0:
        return 0

    cuda_overhead_mb = 300
    kv_per_layer_mb = (context_length / 1024) * 0.5

    usable_vram = (available_vram_mb - cuda_overhead_mb) * 0.90
    if usable_vram <= 0:
        return 0

    weight_per_layer_mb = file_size_mb / n_layers
    cost_per_layer = weight_per_layer_mb + kv_per_layer_mb

    max_layers = int(usable_vram / cost_per_layer)
    return min(max_layers, n_layers)


# ─── Vision Detection ───────────────────────────────────────────────────────

def detect_vision_support(
    family_key: str | None,
    gguf_metadata: dict,
    file_path: str,
) -> bool:
    """
    Detect if a model supports vision/image input.
    Checks multiple signals.
    """
    # 1. Family profile says it has vision
    if family_key and family_key in FAMILY_PROFILES:
        if FAMILY_PROFILES[family_key].has_vision:
            return True

    # 2. GGUF metadata contains vision-related fields
    if gguf_metadata.get("has_vision_metadata", False):
        return True

    # 3. Filename hints
    fname = Path(file_path).stem.lower()
    vision_indicators = [
        "llava", "moondream", "internvl", "minicpm-v", "minicpm_v",
        "qwen-vl", "qwen2-vl", "cogvlm", "paligemma", "idefics",
        "bunny", "phi-3-vision", "phi-3.5-vision",
    ]
    for indicator in vision_indicators:
        if indicator in fname:
            return True

    # 4. Check for companion mmproj file
    mmproj = find_mmproj_path(file_path)
    if mmproj:
        return True

    return False


def find_mmproj_path(file_path: str) -> str | None:
    """Find a companion mmproj GGUF for the given model file.

    Returns the full path to the mmproj file, or None.
    Matches by checking that the mmproj filename shares ALL of the first two
    stem segments with the model file (e.g. Qwen3.5-9B matches
    Qwen3.5-9B-...-mmproj-F16.gguf but Qwen3-Coder does NOT match
    Qwen3.5-9B-mmproj).
    """
    model_dir = Path(file_path).parent
    stem = Path(file_path).stem.lower()
    # Use first two hyphen-separated segments as identity (e.g. "qwen3.5-9b")
    stem_parts = stem.split("-")[:2]
    for f in model_dir.iterdir():
        if f.suffix == ".gguf" and "mmproj" in f.stem.lower():
            mmproj_lower = f.stem.lower()
            if all(part in mmproj_lower for part in stem_parts):
                return str(f)
    return None


# ─── Function Calling Detection ──────────────────────────────────────────────

# Families known to have native tool-call chat templates
_TOOL_CALL_FAMILIES = {
    "qwen3", "qwen35", "qwen3_coder", "qwen25", "qwen25_coder", "qwen2",
    "llama33", "llama32", "llama31",
    "mistral", "mixtral",
    "phi4", "phi4_mini",
    "gemma3",
    "glm4", "glm4_flash",
    "deepseek_v3", "deepseek_r1",
    "command_r",
    "internlm",
    "apriel", "apriel_thinker",
    "gpt_oss",
    "gigachat",
    "gemma4",
}


def detect_function_calling(family_key: str | None, gguf_metadata: dict) -> bool:
    """Detect if a local model supports function calling format."""
    if family_key in _TOOL_CALL_FAMILIES:
        return True

    # Check GGUF chat template for tool indicators
    # (would require parsing the template string — future enhancement)
    return False


# ─── Thinking Model Detection ───────────────────────────────────────────────

_THINKING_FAMILIES = {"qwen3", "qwen35", "qwen3_coder", "qwq", "deepseek_r1", "glm4_flash", "apriel_thinker", "gpt_oss", "gemma4"}
_THINKING_NAME_PATTERNS = ["o1", "o3", "o4", "qwq", "deepseek-r1", "gemini-2.5", "glm"]


def detect_thinking_model(
    family_key: str | None,
    litellm_name: str = "",
) -> bool:
    """Detect if model supports native thinking/reasoning mode."""
    if family_key in _THINKING_FAMILIES:
        return True
    name_lower = litellm_name.lower()
    return any(p in name_lower for p in _THINKING_NAME_PATTERNS)


# ─── GGUF Directory Scanner ─────────────────────────────────────────────────

def scan_model_directory(model_dir: str | Path) -> list[dict]:
    """
    Scan a directory for .gguf files and extract metadata for each.
    Returns list of raw model info dicts ready for registration.

    Skips:
    - mmproj files (vision projectors, handled separately)
    - Files smaller than 50MB (probably not real models)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        logger.warning(f"Model directory does not exist: {model_dir}")
        return []

    results = []
    gguf_files = sorted(model_dir.glob("**/*.gguf"))  # recursive

    for fpath in gguf_files:
        fname = fpath.stem.lower()

        # Skip projector files
        if "mmproj" in fname or "projector" in fname:
            continue

        # Skip tiny files
        try:
            size_mb = fpath.stat().st_size / (1024 * 1024)
        except OSError:
            continue
        if size_mb < 50:
            continue

        logger.debug(f"Scanning: {fpath.name}")

        # Read GGUF metadata
        meta = read_gguf_metadata(str(fpath))

        # Merge filename-based estimates for anything GGUF didn't provide
        fname_meta = _estimate_from_filename(str(fpath))
        for k, v in fname_meta.items():
            if k not in meta or not meta[k]:
                meta[k] = v

        # Determine quantization
        quantization = meta.get("quantization", "")
        if not quantization:
            # Try to extract from filename more aggressively
            q_match = re.search(
                r"(IQ[1-4]_\w+|Q[2-8]_K_\w+|Q[2-8]_K|Q[2-8]_0|Q[2-8]|F16|F32|FP16|BF16)",
                fpath.stem,
                re.IGNORECASE,
            )
            quantization = q_match.group(1).upper() if q_match else "Q4_K_M"

        # Param estimation
        total_params = meta.get("estimated_params_b", 0)
        if not total_params and size_mb > 0:
            bpp_map = {"Q8": 1.0, "Q6": 0.75, "Q5": 0.625, "Q4": 0.5, "Q3": 0.4375, "Q2": 0.3125, "F16": 2.0, "IQ4": 0.5, "IQ3": 0.4, "IQ2": 0.3, "IQ1": 0.2}
            bpp = 0.5
            for prefix, val in bpp_map.items():
                if prefix in quantization.upper():
                    bpp = val
                    break
            total_params = (size_mb * 1024 * 1024) / (bpp * 1e9)

        # MoE detection
        expert_count = meta.get("expert_count", 0)
        is_moe = expert_count > 1
        active_experts = meta.get("expert_used_count", expert_count)
        active_params = meta.get("active_params_b", 0)
        if is_moe and not active_params and total_params:
            if expert_count > 0:
                active_params = total_params * (active_experts / expert_count) * 0.6

        # Family detection (from architecture, model_name, or filename)
        detect_name = " ".join(filter(None, [
            meta.get("architecture", ""),
            meta.get("model_name", ""),
            fpath.stem,
        ])).lower()

        # Normalize separators for matching
        detect_name = detect_name.replace("_", "-")
        family_key = detect_family(detect_name)

        # Vision support + mmproj path
        has_vision = detect_vision_support(family_key, meta, str(fpath))
        mmproj_path = find_mmproj_path(str(fpath)) if has_vision else None

        # Function calling
        func_calling = detect_function_calling(family_key, meta)

        # Thinking model
        thinking = detect_thinking_model(family_key)

        # Context length
        n_layers = meta.get("n_layers", 32)
        native_ctx = meta.get("context_length", 8192)
        if family_key and family_key in FAMILY_PROFILES:
            default_ctx = FAMILY_PROFILES[family_key].context_default
            native_ctx = max(native_ctx, default_ctx)

        # Specialty
        specialty = meta.get("specialty", "")
        if not specialty and family_key and family_key in FAMILY_PROFILES:
            specialty = FAMILY_PROFILES[family_key].specialty

        # Derive a clean name from filename
        model_name = fpath.stem  # e.g., "Qwen3-30B-A3B-Q4_K_M"

        results.append({
            "name": model_name,
            "path": str(fpath),
            "file_size_mb": size_mb,
            "total_params_b": total_params,
            "active_params_b": active_params,
            "is_moe": is_moe,
            "expert_count": expert_count,
            "quantization": quantization,
            "family_key": family_key,
            "n_layers": n_layers,
            "native_ctx": native_ctx,
            "has_vision": has_vision,
            "mmproj_path": mmproj_path,
            "function_calling": func_calling,
            "thinking": thinking,
            "specialty": specialty,
        })

    logger.info(f"Scanned {model_dir}: found {len(results)} GGUF model(s)")
    return results


# ─── Cloud Model Auto-Detection ─────────────────────────────────────────────

# Initial rate limit defaults per provider — used as seed values until
# runtime header discovery provides actual limits from API responses.
_FREE_TIER_DEFAULTS: dict[str, dict] = {
    "gemini":    {"rpm": 15, "tpm": 1000000, "tier": "free"},
    "groq":      {"rpm": 30, "tpm": 131072, "tier": "free"},
    "cerebras":  {"rpm": 30, "tpm": 131072, "tier": "free"},
    "sambanova": {"rpm": 20, "tpm": 100000, "tier": "free"},
    "openai":    {"rpm": 500, "tpm": 800000, "tier": "paid"},
    "anthropic": {"rpm": 50, "tpm": 80000, "tier": "paid"},
}

# Providers whose API supports strict ``response_format: json_schema``
# (token-level constrained decoding). Verified as of 2026-04 — refresh
# when adding new providers or when a provider rolls out the feature.
#
#   openai     — Structured Outputs GA Aug 2024; gpt-4o-2024-08-06+.
#   gemini     — responseSchema (Gemini 1.5+); litellm maps json_schema.
#   llama_cpp  — OpenAI-compat endpoint; also grammar+json_schema body.
#   ollama     — format=<schema> since 0.5.0 (Dec 2024); same engine.
#
# Excluded (json_object only or no support):
#   anthropic  — no response_format; structured outputs via tool-use.
#   groq       — json_object yes, json_schema partial per-model.
#   cerebras   — json_object yes, json_schema in rollout.
#   sambanova  — json_object only as of late 2025.
_PROVIDERS_WITH_JSON_SCHEMA: set[str] = {
    "openai",
    "gemini",
    "llama_cpp",
    "ollama",
}

KNOWN_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "mistral",
    "cohere",
    "deepseek",
    "perplexity",
    "xai",
    "groq",
    "openrouter",
    "together",
    "fireworks",
    "replicate",
    "vertex",
    "bedrock",
    "ollama",
    "lmstudio",
    "azure",
}

PROVIDER_PREFIXES = {
    "openai": (
        "gpt",
        "o1",
        "o3",
        "o4",
    ),
    "anthropic": (
        "claude",
    ),
    "google": (
        "gemini",
        "palm",
    ),
    "mistral": (
        "mistral",
        "mixtral",
        "codestral",
    ),
    "cohere": (
        "command",
        "aya",
    ),
    "deepseek": (
        "deepseek",
    ),
    "perplexity": (
        "sonar",
        "pplx",
    ),
    "xai": (
        "grok",
    ),
}


def detect_cloud_model(litellm_name: str, provider: str) -> dict:
    """Auto-detect cloud model properties from litellm name + knowledge base."""
    info: dict = {}

    # Try litellm's model database
    try:
        import litellm as _lt
        model_info = _lt.get_model_info(model=litellm_name)
        if model_info:
            info["context_length"] = (
                model_info.get("max_input_tokens")
                or model_info.get("max_tokens")
                or 128000
            )
            info["max_tokens"] = model_info.get("max_output_tokens", 4096)
            info["supports_function_calling"] = model_info.get("supports_function_calling", True)
            info["supports_json_mode"] = model_info.get("supports_response_format", True)

            input_cost = model_info.get("input_cost_per_token", 0)
            output_cost = model_info.get("output_cost_per_token", 0)
            if input_cost:
                info["cost_per_1k_input"] = input_cost * 1000
            if output_cost:
                info["cost_per_1k_output"] = output_cost * 1000
    except Exception as e:
        logger.debug(f"litellm model info lookup failed for {litellm_name}: {e}")

    # Provider defaults
    provider_defaults = _FREE_TIER_DEFAULTS.get(provider, {})
    info.setdefault("rate_limit_rpm", provider_defaults.get("rpm", 30))
    info.setdefault("rate_limit_tpm", provider_defaults.get("tpm", 100000))
    info.setdefault("tier", provider_defaults.get("tier", "paid"))
    info.setdefault("context_length", 128000)
    info.setdefault("max_tokens", 4096)
    info.setdefault("supports_function_calling", True)
    info.setdefault("supports_json_mode", True)
    # Constrained decoding (strict response_format: json_schema). Defaults
    # to provider-known capability — see _PROVIDERS_WITH_JSON_SCHEMA. Per-
    # model overrides via models.yaml ``supports_json_schema: bool``.
    info.setdefault(
        "supports_json_schema",
        provider in _PROVIDERS_WITH_JSON_SCHEMA,
    )

    # Match against cloud profiles
    name_lower = litellm_name.lower()
    # Strip provider prefix (e.g., "cerebras/llama-3.3-70b" → "llama-3.3-70b")
    name_no_prefix = litellm_name.split("/", 1)[-1].lower() if "/" in litellm_name else name_lower
    capabilities = None
    has_vision = False
    thinking_model = False

    matched_len = 0
    matched_profile = None
    for hint_key, profile_data in CLOUD_PROFILES.items():
        if hint_key.lower() in name_no_prefix and len(hint_key) > matched_len:
            matched_len = len(hint_key)
            matched_profile = profile_data

    if matched_profile:
        capabilities = dict(matched_profile["capabilities"])
        has_vision = matched_profile.get("has_vision", False)
        thinking_model = matched_profile.get("thinking_model", False)

    if capabilities is None:
        # Unknown cloud model — reasonable defaults
        capabilities = {cap: 6.0 for cap in ALL_CAPABILITIES}
        capabilities["vision"] = 0.0

    # Thinking model detection from name
    if not thinking_model:
        thinking_model = any(t in name_lower for t in _THINKING_NAME_PATTERNS)

    info["capabilities"] = capabilities
    info["has_vision"] = has_vision
    info["thinking_model"] = thinking_model

    return info


def register_cloud_from_discovered(
    registry: "ModelRegistry",
    provider: str,
    discovered,  # DiscoveredModel; loose annotation to avoid forward-ref headaches at module import time
) -> "ModelInfo | None":
    """Register a discovered cloud model into the registry.

    Merges adapter-scraped fields with detect_cloud_model() output:
        - context_length: scraped wins over litellm-db default
        - max_tokens: scraped wins
        - cost_per_1k_*: scraped wins (provider-data is authoritative)
        - sampling_defaults: scraped seeds sampling_overrides if absent
        - active=False: skip registration entirely
    Family is computed via cloud.family.normalize().
    Returns the registered ModelInfo, or None if skipped.
    """
    if not getattr(discovered, "active", True):
        return None

    from .cloud.family import normalize as _family_normalize

    detected = detect_cloud_model(discovered.litellm_name, provider)

    if discovered.context_length is not None:
        detected["context_length"] = discovered.context_length
    if discovered.max_output_tokens is not None:
        detected["max_tokens"] = discovered.max_output_tokens
    if discovered.cost_per_1k_input is not None:
        detected["cost_per_1k_input"] = discovered.cost_per_1k_input
    if discovered.cost_per_1k_output is not None:
        detected["cost_per_1k_output"] = discovered.cost_per_1k_output

    family = _family_normalize(provider, discovered.litellm_name)

    model = ModelInfo(
        name=discovered.litellm_name,
        location="cloud",
        provider=provider,
        litellm_name=discovered.litellm_name,
        capabilities=detected["capabilities"],
        context_length=detected["context_length"],
        max_tokens=detected["max_tokens"],
        supports_function_calling=detected.get("supports_function_calling", True),
        supports_json_mode=detected.get("supports_json_mode", True),
        supports_json_schema=detected.get("supports_json_schema", False),
        thinking_model=detected.get("thinking_model", False),
        has_vision=detected.get("has_vision", False),
        tier=detected.get("tier", "paid"),
        rate_limit_rpm=detected["rate_limit_rpm"],
        rate_limit_tpm=detected.get("rate_limit_tpm", 100000),
        cost_per_1k_input=detected.get("cost_per_1k_input", 0.0),
        cost_per_1k_output=detected.get("cost_per_1k_output", 0.0),
        family=family,
    )
    if discovered.sampling_defaults and not model.sampling_overrides:
        model.sampling_overrides = {"default": dict(discovered.sampling_defaults)}
    registry.register(model)
    return model


# ─── Provider Resolution ─────────────────────────────────────────────────────

def _resolve_provider(litellm_name: str) -> str | None:
    if not litellm_name:
        return None

    name = litellm_name.lower().strip()

    # Explicit provider format
    if "/" in name:
        provider = name.split("/", 1)[0]
        if provider in KNOWN_PROVIDERS:
            return provider
        return None

    # Infer from model name
    for provider, prefixes in PROVIDER_PREFIXES.items():
        if name.startswith(prefixes):
            return provider

    return None


# ─── Model Variant Registration ─────────────────────────────────────────────

def _apply_thinking_deltas(capabilities: dict[str, float]) -> dict[str, float]:
    """Adjust capability scores for thinking mode variant.

    Deltas derived from LM Arena thinking/non-thinking pairs (Qwen3-235B,
    DeepSeek-V3.1/V3.2).  Frontier models show small gains; local models
    further from the ceiling should benefit proportionally more, so we
    apply modest positive deltas.
    """
    caps = dict(capabilities)
    deltas = {
        "reasoning": 0.4,
        "planning": 0.4,
        "analysis": 0.5,
        "code_reasoning": 0.2,
        "prose_quality": 0.3,
        "instruction_adherence": 0.3,
        "context_utilization": 0.2,
    }
    for key, delta in deltas.items():
        if key in caps:
            caps[key] = round(max(0.0, min(10.0, caps[key] + delta)), 1)
    return caps


def _create_model_variants(
    base: ModelInfo,
    family_profile,
) -> list[ModelInfo]:
    """
    Create 1-4 ModelInfo entries from a base model + family capabilities.

    Returns:
      - Base entry (thinking_model=False, has_vision=False)
      - Thinking variant if family is thinking_capable
      - Vision variant if family has_vision and mmproj_path exists
      - Thinking+Vision variant if both apply
    """
    from dataclasses import replace as dc_replace

    thinking_capable = family_profile.thinking_capable if family_profile else False
    # Vision: if mmproj file exists, model supports vision regardless of family profile
    vision_capable = bool(base.mmproj_path)

    # Base entry: always strip thinking/vision flags
    base_entry = dc_replace(
        base,
        thinking_model=False,
        has_vision=False,
        is_variant=False,
        base_model_name="",
        variant_flags=set(),
    )
    variants = [base_entry]

    if thinking_capable:
        thinking_entry = dc_replace(
            base,
            name=f"{base.name}-thinking",
            thinking_model=True,
            has_vision=False,
            is_variant=True,
            base_model_name=base.name,
            variant_flags={"thinking"},
            capabilities=_apply_thinking_deltas(base.capabilities),
            litellm_name=f"openai/{base.name}-thinking",
        )
        variants.append(thinking_entry)

    if vision_capable:
        vision_entry = dc_replace(
            base,
            name=f"{base.name}-vision",
            thinking_model=False,
            has_vision=True,
            is_variant=True,
            base_model_name=base.name,
            variant_flags={"vision"},
            litellm_name=f"openai/{base.name}-vision",
        )
        variants.append(vision_entry)

    if thinking_capable and vision_capable:
        tv_entry = dc_replace(
            base,
            name=f"{base.name}-thinking-vision",
            thinking_model=True,
            has_vision=True,
            is_variant=True,
            base_model_name=base.name,
            variant_flags={"thinking", "vision"},
            capabilities=_apply_thinking_deltas(base.capabilities),
            litellm_name=f"openai/{base.name}-thinking-vision",
        )
        variants.append(tv_entry)

    return variants


# ─── Available VRAM Helper ───────────────────────────────────────────────────

def _get_available_vram() -> int:
    """Get available VRAM in MB via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0


# ─── ModelRegistry ───────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Catalog of known models. Supports loading from YAML and GGUF directory scans.

    Usage:
        reg = ModelRegistry()
        reg.load_yaml("models.yaml")
        reg.load_gguf_dir("/path/to/models")
        model = reg.get("Qwen3-30B-Q4_K_M")
    """

    _SPEED_CACHE_PATH = Path("data/model_speeds.json")
    _SPEED_SAVE_INTERVAL = 30  # seconds between disk writes

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self.personal_projects: list[str] = []
        self._raw_config: dict = {}
        self._lock = threading.RLock()
        self._speed_cache_dirty = False
        self._speed_cache_last_save: float = 0.0

    # ── Core catalog interface ────────────────────────────────────────────────

    def register(self, model: ModelInfo) -> None:
        """Add or replace a model in the catalog."""
        self._models[model.name] = model

    def get(self, name: str) -> ModelInfo | None:
        """Look up by registry name."""
        return self._models.get(name)

    def all_models(self) -> list[ModelInfo]:
        """Return all registered models."""
        return list(self._models.values())

    def by_litellm_name(self, litellm_name: str) -> ModelInfo | None:
        """Find a model by its litellm_name."""
        for m in self._models.values():
            if m.litellm_name == litellm_name:
                return m
        return None

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_yaml(self, path: str) -> list[ModelInfo]:
        """Load cloud (and YAML-declared local) models from a YAML catalog.

        Returns the list of newly loaded ModelInfo objects.
        """
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"YAML catalog not found: {path}")
            return []

        with open(config_path, "r", encoding="utf-8") as f:
            self._raw_config = yaml.safe_load(f) or {}

        self.personal_projects = self._raw_config.get("personal_projects", [])
        overrides = self._raw_config.get("overrides", {})
        loaded: list[ModelInfo] = []

        # Cloud models
        for name, cfg in self._raw_config.get("cloud", {}).items():
            litellm_name = cfg.get("litellm_name", "")
            if not litellm_name:
                continue

            config_provider = _resolve_provider(litellm_name)
            if not config_provider:
                continue

            detected = detect_cloud_model(litellm_name, config_provider)
            model_overrides = overrides.get(name, {})

            # Apply scalar overrides
            for key in ["rate_limit_rpm", "rate_limit_tpm", "context_length", "max_tokens"]:
                if key in model_overrides:
                    detected[key] = model_overrides[key]
            for key in ["rate_limit_rpm", "rate_limit_tpm", "context_length", "max_tokens"]:
                if key in cfg:
                    detected.setdefault(key, cfg[key])

            model = ModelInfo(
                name=name,
                location="cloud",
                provider=config_provider,
                litellm_name=litellm_name,
                capabilities=detected["capabilities"],
                context_length=detected["context_length"],
                max_tokens=detected["max_tokens"],
                supports_function_calling=detected.get("supports_function_calling", True),
                supports_json_mode=detected.get("supports_json_mode", True),
                supports_json_schema=detected.get("supports_json_schema", False),
                thinking_model=detected.get("thinking_model", False),
                has_vision=detected.get("has_vision", False),
                tier=detected.get("tier", "paid"),
                rate_limit_rpm=detected["rate_limit_rpm"],
                rate_limit_tpm=detected.get("rate_limit_tpm", 100000),
                cost_per_1k_input=detected.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=detected.get("cost_per_1k_output", 0.0),
            )
            self.register(model)
            loaded.append(model)

        # Apply YAML capability + sampling overrides
        for model_name, model_overrides in overrides.items():
            if model_name not in self._models:
                continue
            if "capabilities" in model_overrides:
                manual_caps = {}
                for cap_name, score in model_overrides["capabilities"].items():
                    self._models[model_name].capabilities[cap_name] = float(score)
                    manual_caps[cap_name] = float(score)
                self._models[model_name].benchmark_scores = manual_caps
            if "sampling" in model_overrides:
                self._models[model_name].sampling_overrides = {
                    k: {pk: float(pv) for pk, pv in v.items()}
                    for k, v in model_overrides["sampling"].items()
                }

        logger.info(f"load_yaml: loaded {len(loaded)} cloud models from {path}")
        return loaded

    def load_gguf_dir(self, model_dir: str) -> list[ModelInfo]:
        """Scan a directory for GGUF files and register a ModelInfo for each.

        Returns the list of newly registered ModelInfo objects (including variants).
        """
        overrides = self._raw_config.get("overrides", {})
        available_vram = _get_available_vram()
        scanned = scan_model_directory(model_dir)
        loaded: list[ModelInfo] = []

        for raw in scanned:
            name = raw["name"]
            model_overrides = overrides.get(name, {})

            # Skip if explicitly disabled
            if model_overrides.get("disabled", False):
                logger.debug(f"Skipping disabled model: {name}")
                continue

            # Estimate capabilities
            capabilities = estimate_capabilities(
                family_key=raw["family_key"],
                total_params_b=raw["total_params_b"],
                active_params_b=raw["active_params_b"] if raw["is_moe"] else None,
                quantization=raw["quantization"],
                is_moe=raw["is_moe"],
                has_vision_hint=raw["has_vision"],
            )

            # Context length + GPU layers (co-dependent — resolve iteratively).
            _gpu_layers_from_override = "gpu_layers" in model_overrides and model_overrides["gpu_layers"]
            if model_overrides.get("context_length"):
                context_length = model_overrides["context_length"]
            else:
                # Registry holds the model's CAPABILITY ceiling (trained
                # native context). Actual runtime allocation is computed
                # by local_model_manager at load time against LIVE VRAM
                # and the current prompt's required ctx. Baking the
                # startup-time dynamic calc into the registry made the
                # selector filter against a stale tiny value — a model
                # that can handle 131k trained got rejected at 9k needed
                # because startup VRAM happened to fit only 4k.
                context_length = int(raw.get("native_ctx") or 32768)

            # GPU-layer baseline at 8K context — just a scoring heuristic
            # for ranking. Actual offload happens at load time via
            # llama.cpp --fit. Don't feed the native_ctx here or the calc
            # concludes almost no GPU layers fit.
            gpu_layers = model_overrides.get("gpu_layers") or calculate_gpu_layers(
                file_size_mb=raw["file_size_mb"],
                n_layers=raw["n_layers"],
                available_vram_mb=available_vram,
                context_length=8192,
            )

            # Priority class
            specialty = raw["specialty"]
            total_params = raw["total_params_b"]
            if specialty == "coding":
                priority_class = "coding"
            elif specialty == "vision":
                priority_class = "vision"
            elif total_params < 5:
                priority_class = "fast"
            elif raw["thinking"]:
                priority_class = "reasoning"
            else:
                priority_class = "primary"

            # Max output tokens
            max_tokens = model_overrides.get(
                "max_tokens",
                min(context_length // 4, 16384),
            )

            # Determine per-model server flags
            extra_server_flags = list(model_overrides.get("extra_server_flags", []))
            fc_supported = raw["function_calling"]
            if not extra_server_flags:
                family = raw.get("family_key", "") or ""
                name_lower = name.lower()
                if raw["is_moe"] and family in ("qwen3", "qwen3_coder"):
                    extra_server_flags = ["--override-kv", "tokenizer.ggml.eos_token_id=int:151645"]
                if "apriel" in name_lower:
                    extra_server_flags = ["--no-jinja", "--chat-template", "chatml"]

            # --no-jinja means llama-server rejects tools param
            if "--no-jinja" in extra_server_flags:
                fc_supported = False

            model = ModelInfo(
                name=name,
                location="local",
                provider="llama_cpp",
                litellm_name=f"openai/{name}",
                capabilities=capabilities,
                context_length=context_length,
                max_tokens=max_tokens,
                supports_function_calling=fc_supported,
                supports_json_mode=True,
                # Local llama.cpp supports strict json_schema constrained
                # decoding via the OpenAI-compat endpoint. Always-on for
                # the local pool — the base flag is true regardless of
                # --no-jinja since json_schema lives on the response_format
                # path, not the chat-template-tools path.
                supports_json_schema=True,
                thinking_model=raw["thinking"],
                has_vision=raw["has_vision"],
                mmproj_path=raw.get("mmproj_path"),
                path=raw["path"],
                model_type="moe" if raw["is_moe"] else "dense",
                total_params_b=total_params,
                active_params_b=raw["active_params_b"],
                quantization=raw["quantization"],
                gpu_layers=gpu_layers,
                _gpu_layers_from_override=_gpu_layers_from_override,
                total_layers=raw["n_layers"],
                file_size_mb=raw["file_size_mb"],
                load_time_seconds=max(10, raw["file_size_mb"] / 500),
                priority_class=priority_class,
                specialty=specialty,
                family=raw["family_key"] or "unknown",
                extra_server_flags=extra_server_flags,
            )

            # Create variants (base, thinking, vision, thinking+vision)
            family_profile = FAMILY_PROFILES.get(raw["family_key"] or "")
            variants = _create_model_variants(model, family_profile)
            for variant in variants:
                self.register(variant)
                loaded.append(variant)

        logger.info(f"load_gguf_dir: registered {len(loaded)} model entries from {model_dir}")
        return loaded

    # ── Compatibility helpers (mirrors legacy ModelRegistry API) ─────────────

    def find_by_litellm_name(self, litellm_name: str) -> ModelInfo | None:
        return self.by_litellm_name(litellm_name)

    def local_models(self) -> list[ModelInfo]:
        return [m for m in self._models.values() if m.is_local]

    def cloud_models(self) -> list[ModelInfo]:
        return [m for m in self._models.values() if not m.is_local]

    def vision_models(self) -> list[ModelInfo]:
        return [m for m in self._models.values() if m.has_vision]

    def thinking_models(self) -> list[ModelInfo]:
        return [m for m in self._models.values() if m.thinking_model]

    def models_with_capability(
        self, capability: str, min_score: float = 1.0
    ) -> list[ModelInfo]:
        return [
            m for m in self._models.values()
            if m.capabilities.get(capability, 0) >= min_score
        ]

    def is_personal_project(self, project_name: str) -> bool:
        return any(p.lower() in project_name.lower() for p in self.personal_projects)

    # ── Speed cache ──────────────────────────────────────────────────────────

    def _load_speed_cache(self) -> None:
        """Load persisted speed measurements into current models."""
        try:
            if not self._SPEED_CACHE_PATH.exists():
                return
            with open(self._SPEED_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            restored = 0
            for name, entry in cache.items():
                model = self._models.get(name)
                if model is None:
                    continue
                tps = entry.get("tps", 0.0)
                demoted = entry.get("demoted", False)
                if tps > 0 and model.tokens_per_second <= 0:
                    model.tokens_per_second = tps
                    restored += 1
                if demoted and not model.demoted:
                    model.demoted = True
            if restored:
                logger.info(f"Restored speed data for {restored} models from cache")
        except Exception as e:
            logger.warning(f"Failed to load speed cache: {e}")

    def _save_speed_cache(self, force: bool = False) -> None:
        """Persist speed measurements to disk (debounced)."""
        now = time.time()
        if not force and (now - self._speed_cache_last_save) < self._SPEED_SAVE_INTERVAL:
            self._speed_cache_dirty = True
            return
        try:
            cache = {}
            for name, model in self._models.items():
                if model.tokens_per_second > 0 or model.demoted:
                    cache[name] = {
                        "tps": round(model.tokens_per_second, 1),
                        "demoted": model.demoted,
                        "updated": time.strftime("%Y-%m-%d"),
                    }
            if not cache:
                return
            self._SPEED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._SPEED_CACHE_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            tmp.replace(self._SPEED_CACHE_PATH)
            self._speed_cache_last_save = now
            self._speed_cache_dirty = False
        except Exception as e:
            logger.warning(f"Failed to save speed cache: {e}")

    def flush_speed_cache(self) -> None:
        """Force-write speed cache to disk. Call on shutdown."""
        if self._speed_cache_dirty:
            self._save_speed_cache(force=True)

    # ── Legacy API compatibility shims ───────────────────────────────────────
    # These preserve backward-compat with callers that use the old ModelRegistry
    # API (dict-style .models, is_demoted(), load(), reload()).

    @property
    def models(self) -> dict[str, "ModelInfo"]:
        """Legacy dict-style access — mirrors self._models."""
        return self._models

    @models.setter
    def models(self, value: dict[str, "ModelInfo"]) -> None:
        """Allow direct assignment for test fixtures and compat."""
        self._models = value

    def is_demoted(self, name: str) -> bool:
        """Check if a model is currently demoted (timed or permanent)."""
        with self._lock:
            m = self._models.get(name)
            if m is None:
                return False
            until = getattr(m, "_demoted_until", 0)
            if until:
                if time.time() < until:
                    return True
                else:
                    m._demoted_until = 0  # type: ignore[attr-defined]
            return bool(m.demoted)

    def demote(self, name: str, duration: float = 300.0) -> None:
        """Temporarily demote a model after a load failure."""
        with self._lock:
            m = self._models.get(name)
            if m is not None:
                m._demoted_until = time.time() + duration  # type: ignore[attr-defined]

    def get_overrides(self, model_name: str) -> dict:
        """Return user overrides from models.yaml for a specific model."""
        return self._raw_config.get("overrides", {}).get(model_name, {})

    def mark_loaded(self, name: str, api_base: str) -> None:
        """Mark a local model as loaded (only one at a time)."""
        with self._lock:
            for m in self._models.values():
                if m.location == "local":
                    m.is_loaded = False
                    m.api_base = None
            if name in self._models:
                self._models[name].is_loaded = True
                self._models[name].api_base = api_base

    def mark_unloaded(self, name: str) -> None:
        """Mark a model as unloaded."""
        with self._lock:
            if name in self._models:
                self._models[name].is_loaded = False
                self._models[name].api_base = None

    def demote_model(self, name: str, duration: int = 300) -> None:
        """Alias for demote() — backward compat with local_model_manager."""
        self.demote(name, float(duration))

    def load(self, config_path: "Path | str | None" = None) -> None:
        """Load the model catalog from models.yaml (legacy compat)."""
        _REGISTRY_PATH = Path(__file__).parent.parent.parent.parent / "src" / "models" / "models.yaml"
        resolved = Path(config_path) if config_path else _REGISTRY_PATH
        if resolved.exists():
            self.load_yaml(str(resolved))
        else:
            logger.warning(f"Registry config not found at {resolved}")

    def reload(self, config_path: "Path | str | None" = None) -> dict:
        """Hot reload — rescan and rebuild registry."""
        old_names = set(self._models.keys())
        self.load(config_path)
        new_names = set(self._models.keys())
        added = sorted(new_names - old_names)
        removed = sorted(old_names - new_names)
        return {"added": added, "removed": removed, "total": len(new_names)}

    def currently_loaded(self) -> "ModelInfo | None":
        """Return the currently loaded local model, if any."""
        for m in self._models.values():
            if m.is_local and m.is_loaded:
                return m
        return None
