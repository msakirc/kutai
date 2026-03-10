"""
Model Registry v2 — auto-scanning, 14-dimension capabilities, hot reload.

Loads models from:
  1. GGUF files in MODEL_DIR (from .env)
  2. Cloud providers (from models.yaml + API key detection)
  3. Ollama (auto-detected)
  4. Custom endpoints (from env vars)

Call reload() at any time to rescan without restarting.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from capabilities import (
    ALL_CAPABILITIES,
    Cap,
    EXECUTION_DIMENSIONS,
    KNOWLEDGE_DIMENSIONS,
    REASONING_DIMENSIONS,
    TaskRequirements,
    rank_models_for_task,
    score_model_for_task,
)
from model_profiles import (
    CLOUD_PROFILES,
    FAMILY_PROFILES,
    detect_family,
    get_default_profile,
    get_quant_retention,
    interpolate_size_multiplier,
)

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent / "models.yaml"


# ─── ModelInfo ───────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Unified model descriptor — local, cloud, and ollama."""
    name: str
    location: str                           # "local" | "cloud" | "ollama"
    provider: str                           # "llama_cpp" | "gemini" | "anthropic" | "ollama" ...
    litellm_name: str
    capabilities: dict[str, float]          # 14-dimension scores, 0.0-10.0
    context_length: int
    max_tokens: int
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    thinking_model: bool = False
    has_vision: bool = False

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
    total_layers: int = 0
    file_size_mb: float = 0.0
    tokens_per_second: float = 0.0          # measured at runtime
    load_time_seconds: float = 30.0
    priority_class: str = "primary"
    specialty: str = ""
    family: str = ""                        # detected family key

    # Runtime state
    is_loaded: bool = False
    api_base: Optional[str] = None

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
            "thinking_model": self.thinking_model,
            "has_vision": self.has_vision,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "tokens_per_second": self.tokens_per_second,
            "tier": self.tier,
            "rate_limit_rpm": self.rate_limit_rpm,
        }

    @property
    def is_local(self) -> bool:
        return self.location in ("local", "ollama")

    @property
    def is_free(self) -> bool:
        return self.location in ("local", "ollama") or self.tier == "free"


# ─── GGUF Metadata Reader ───────────────────────────────────────────────────

def read_gguf_metadata(path: str) -> dict:
    """Read key metadata from a GGUF file header."""
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
            # Vision detection: look for clip/vision projector metadata
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
    Estimate 14-dimension capability scores from model metadata.

    Strategy:
    1. Look up family profile (or use default)
    2. Calculate size multiplier using effective params per dimension type
    3. Apply quantization retention factor
    4. Clamp all scores to [0.0, 10.0]
    """
    # Get family profile
    if family_key and family_key in FAMILY_PROFILES:
        profile = FAMILY_PROFILES[family_key]
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

    cuda_overhead_mb = 500
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
    model_dir = Path(file_path).parent
    stem = Path(file_path).stem.lower()
    for f in model_dir.iterdir():
        if f.suffix == ".gguf" and "mmproj" in f.stem.lower():
            # Check if it's plausibly for this model
            if any(part in f.stem.lower() for part in stem.split("-")[:2]):
                return True

    return False


# ─── Function Calling Detection ──────────────────────────────────────────────

# Families known to have native tool-call chat templates
_TOOL_CALL_FAMILIES = {
    "qwen3", "qwen3_coder", "qwen25", "qwen25_coder", "qwen2",
    "llama33", "llama32", "llama31",
    "mistral", "mixtral",
    "phi4", "phi4_mini",
    "gemma3",
    "deepseek_v3", "deepseek_r1",
    "command_r",
    "internlm",
}


def detect_function_calling(family_key: str | None, gguf_metadata: dict) -> bool:
    """Detect if a local model supports function calling format."""
    if family_key in _TOOL_CALL_FAMILIES:
        return True

    # Check GGUF chat template for tool indicators
    # (would require parsing the template string — future enhancement)
    return False


# ─── Thinking Model Detection ───────────────────────────────────────────────

_THINKING_FAMILIES = {"qwen3", "qwen3_coder", "qwq", "deepseek_r1"}
_THINKING_NAME_PATTERNS = ["o1", "o3", "o4", "qwq", "deepseek-r1", "gemini-2.5"]


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
        detect_name = detect_name.replace("_", "-").replace(".", "-")
        family_key = detect_family(detect_name)

        # Vision support
        has_vision = detect_vision_support(family_key, meta, str(fpath))

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
            "function_calling": func_calling,
            "thinking": thinking,
            "specialty": specialty,
        })

    logger.info(f"Scanned {model_dir}: found {len(results)} GGUF model(s)")
    return results


# ─── Cloud Model Auto-Detection ─────────────────────────────────────────────

_FREE_TIER_DEFAULTS: dict[str, dict] = {
    "gemini":    {"rpm": 15, "tpm": 1000000, "tier": "free"},
    "groq":      {"rpm": 30, "tpm": 131072, "tier": "free"},
    "cerebras":  {"rpm": 30, "tpm": 131072, "tier": "free"},
    "sambanova": {"rpm": 20, "tpm": 100000, "tier": "free"},
    "openai":    {"rpm": 500, "tpm": 800000, "tier": "paid"},
    "anthropic": {"rpm": 50, "tpm": 80000, "tier": "paid"},
}

_PROVIDER_MAP = {
    "gemini": "gemini", "groq": "groq", "cerebras": "cerebras",
    "sambanova": "sambanova", "anthropic": "anthropic",
    "claude": "anthropic",
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

    # Match against cloud profiles
    name_lower = litellm_name.lower()
    capabilities = None
    has_vision = False
    thinking_model = False

    for hint_key, profile_data in CLOUD_PROFILES.items():
        if hint_key.lower() in name_lower:
            capabilities = dict(profile_data["capabilities"])
            has_vision = profile_data.get("has_vision", False)
            thinking_model = profile_data.get("thinking_model", False)
            break

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


# ─── The Registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Central model registry. Thread-safe for hot reload.

    Usage:
        registry = get_registry()              # singleton
        registry.reload()                      # rescan without restart
        best = registry.best_for_task("coder") # smart selection
    """

    def __init__(self):
        self.models: dict[str, ModelInfo] = {}
        self.personal_projects: list[str] = []
        self._raw_config: dict = {}
        self._lock = threading.RLock()
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self, config_path: Path | str | None = None) -> None:
        """Initial load. Calls _build_registry internally."""
        config_path = Path(config_path) if config_path else REGISTRY_PATH
        self._build_registry(config_path)

    def reload(self, config_path: Path | str | None = None) -> dict:
        """
        Hot reload — rescan model directory and rebuild registry.
        Returns summary dict: {"added": [...], "removed": [...], "total": N}

        Safe to call from any thread while the system is running.
        """
        config_path = Path(config_path) if config_path else REGISTRY_PATH

        old_names = set(self.models.keys())
        # Preserve runtime state from currently loaded models
        runtime_state = {}
        for name, m in self.models.items():
            if m.is_loaded:
                runtime_state[name] = {"api_base": m.api_base, "tps": m.tokens_per_second}

        self._build_registry(config_path)

        # Restore runtime state
        for name, state in runtime_state.items():
            if name in self.models:
                self.models[name].is_loaded = True
                self.models[name].api_base = state["api_base"]
                self.models[name].tokens_per_second = state["tps"]

        new_names = set(self.models.keys())
        added = sorted(new_names - old_names)
        removed = sorted(old_names - new_names)

        if added:
            logger.info(f"Registry reload: added {added}")
        if removed:
            logger.info(f"Registry reload: removed {removed}")

        summary = {"added": added, "removed": removed, "total": len(self.models)}
        logger.info(
            f"Registry reload complete: {summary['total']} models "
            f"(+{len(added)} -{len(removed)})"
        )
        return summary

    def _build_registry(self, config_path: Path) -> None:
        """Core registry building logic."""
        with self._lock:
            new_models: dict[str, ModelInfo] = {}

            # Load YAML config
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self._raw_config = yaml.safe_load(f) or {}
            else:
                logger.warning(f"Config not found at {config_path}, using defaults")
                self._raw_config = {}

            overrides = self._raw_config.get("overrides", {})
            self.personal_projects = self._raw_config.get("personal_projects", [])

            # ── 1. Scan local GGUF directory ──
            model_dir = os.getenv("MODEL_DIR", "")
            logger.info("model_dir")
            logger.info(model_dir)
            if model_dir and Path(model_dir).exists():
                new_models.update(self._load_local_models(model_dir, overrides))
            else:
                if model_dir:
                    logger.warning(f"MODEL_DIR={model_dir} does not exist")

            # ── 2. Load YAML-declared local models (legacy support) ──
            for name, cfg in self._raw_config.get("local", {}).items():
                if name not in new_models:
                    model_path = cfg.get("path", "")
                    if model_path and os.path.isfile(model_path):
                        loaded = self._load_single_local(
                            name, model_path, cfg, overrides.get(name, {})
                        )
                        if loaded:
                            new_models[name] = loaded

            # ── 3. Load cloud models ──
            new_models.update(self._load_cloud_models(overrides))

            # ── 4. Load Ollama models ──
            new_models.update(self._load_ollama_models(overrides))

            # ── 5. Load custom endpoints ──
            new_models.update(self._load_custom_endpoints())

            # ── Apply YAML capability overrides ──
            for model_name, model_overrides in overrides.items():
                if model_name in new_models and "capabilities" in model_overrides:
                    for cap_name, score in model_overrides["capabilities"].items():
                        new_models[model_name].capabilities[cap_name] = float(score)

            self.models = new_models

            # ── 6. Enrich with benchmark data (optional) ──
            settings = self._raw_config.get("settings", {})
            if settings.get("enrich_with_benchmarks", False):
                try:
                    from benchmark_fetcher import enrich_registry_with_benchmarks
                    cache_dir = settings.get("benchmark_cache_dir", ".benchmark_cache")
                    min_sources = settings.get("benchmark_min_sources", 2)
                    enrich_registry_with_benchmarks(
                        registry=self,
                        cache_dir=cache_dir,
                        override_existing=False,
                        min_confidence_sources=min_sources,
                    )
                except ImportError:
                    logger.debug("benchmark_fetcher not available, skipping enrichment")
                except Exception as e:
                    logger.warning(f"Benchmark enrichment failed (non-fatal): {e}")

            self._loaded = True

            # Summary
            n_local = sum(1 for m in new_models.values() if m.location == "local")
            n_cloud = sum(1 for m in new_models.values() if m.location == "cloud")
            n_ollama = sum(1 for m in new_models.values() if m.location == "ollama")
            logger.info(
                f"Registry built: {len(new_models)} models "
                f"({n_local} local, {n_cloud} cloud, {n_ollama} ollama)"
            )

    def _load_local_models(
        self, model_dir: str, overrides: dict
    ) -> dict[str, ModelInfo]:
        """Scan directory and build ModelInfo for each GGUF file."""
        result = {}

        # Get available VRAM
        available_vram = self._get_available_vram()

        scanned = scan_model_directory(model_dir)
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

            # Context length (override > native)
            context_length = model_overrides.get("context_length", raw["native_ctx"])

            # GPU layers
            gpu_layers = model_overrides.get("gpu_layers") or calculate_gpu_layers(
                file_size_mb=raw["file_size_mb"],
                n_layers=raw["n_layers"],
                available_vram_mb=available_vram,
                context_length=context_length,
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

            model = ModelInfo(
                name=name,
                location="local",
                provider="llama_cpp",
                litellm_name=f"openai/{name}",
                capabilities=capabilities,
                context_length=context_length,
                max_tokens=max_tokens,
                supports_function_calling=raw["function_calling"],
                supports_json_mode=True,
                thinking_model=raw["thinking"],
                has_vision=raw["has_vision"],
                path=raw["path"],
                model_type="moe" if raw["is_moe"] else "dense",
                total_params_b=total_params,
                active_params_b=raw["active_params_b"],
                quantization=raw["quantization"],
                gpu_layers=gpu_layers,
                total_layers=raw["n_layers"],
                file_size_mb=raw["file_size_mb"],
                load_time_seconds=max(10, raw["file_size_mb"] / 500),
                priority_class=priority_class,
                specialty=specialty,
                family=raw["family_key"] or "unknown",
            )
            result[name] = model

        moe_info = f"(MoE {raw['active_params_b']:.1f}B active)" if raw['is_moe'] else ""

        logger.info(
            f"  Local: {name} "
            f"| family={raw['family_key'] or '?'} "
            f"| {total_params:.1f}B"
            f"{moe_info} "
            f"| {raw['quantization']} "
            f"| {gpu_layers}/{raw['n_layers']} GPU layers "
            f"| ctx={context_length} "
            f"| best={model.best_score():.1f}"
            f"{'| 👁️ vision' if raw['has_vision'] else ''}"
            f"{'| 🧠 thinking' if raw['thinking'] else ''}"
        )

        return result

    def _load_single_local(
        self, name: str, path: str, cfg: dict, overrides: dict
    ) -> ModelInfo | None:
        """Load a single local model from explicit YAML config."""
        meta = read_gguf_metadata(path)
        fname_meta = _estimate_from_filename(path)
        for k, v in fname_meta.items():
            if k not in meta or not meta[k]:
                meta[k] = v

        quantization = meta.get("quantization", "Q4_K_M")
        total_params = meta.get("estimated_params_b", 0)
        expert_count = meta.get("expert_count", 0)
        is_moe = expert_count > 1
        active_params = meta.get("active_params_b", 0)

        detect_name = " ".join(filter(None, [
            meta.get("architecture", ""), meta.get("model_name", ""), name
        ])).lower().replace("_", "-").replace(".", "-")
        family_key = detect_family(detect_name)

        has_vision = detect_vision_support(family_key, meta, path)

        capabilities = estimate_capabilities(
            family_key=family_key,
            total_params_b=total_params,
            active_params_b=active_params if is_moe else None,
            quantization=quantization,
            is_moe=is_moe,
            has_vision_hint=has_vision,
        )

        n_layers = meta.get("n_layers", 32)
        native_ctx = meta.get("context_length", 8192)
        context_length = overrides.get("context_length", cfg.get("context_length", native_ctx))

        available_vram = self._get_available_vram()
        gpu_layers = overrides.get("gpu_layers") or calculate_gpu_layers(
            file_size_mb=meta.get("file_size_mb", 0),
            n_layers=n_layers,
            available_vram_mb=available_vram,
            context_length=context_length,
        )

        specialty = cfg.get("specialty", meta.get("specialty", ""))

        return ModelInfo(
            name=name,
            location="local",
            provider="llama_cpp",
            litellm_name=f"openai/{name}",
            capabilities=capabilities,
            context_length=context_length,
            max_tokens=min(context_length // 4, 8192),
            supports_function_calling=detect_function_calling(family_key, meta),
            supports_json_mode=True,
            thinking_model=detect_thinking_model(family_key),
            has_vision=has_vision,
            path=path,
            model_type="moe" if is_moe else "dense",
            total_params_b=total_params,
            active_params_b=active_params,
            quantization=quantization,
            gpu_layers=gpu_layers,
            total_layers=n_layers,
            file_size_mb=meta.get("file_size_mb", 0),
            load_time_seconds=max(10, meta.get("file_size_mb", 0) / 500),
            specialty=specialty,
            family=family_key or "unknown",
        )

    def _load_cloud_models(self, overrides: dict) -> dict[str, ModelInfo]:
        """Load cloud models from YAML config."""
        result = {}

        try:
            from config import AVAILABLE_KEYS
        except ImportError:
            logger.warning("config.py not found — skipping cloud models")
            return result

        for name, cfg in self._raw_config.get("cloud", {}).items():
            litellm_name = cfg.get("litellm_name", "")
            if not litellm_name:
                continue

            provider = litellm_name.split("/")[0] if "/" in litellm_name else "openai"
            config_provider = _PROVIDER_MAP.get(provider, provider)

            if not AVAILABLE_KEYS.get(config_provider, False):
                logger.debug(f"Cloud model '{name}': no API key for {config_provider}")
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
                thinking_model=detected.get("thinking_model", False),
                has_vision=detected.get("has_vision", False),
                tier=detected.get("tier", "paid"),
                rate_limit_rpm=detected["rate_limit_rpm"],
                rate_limit_tpm=detected.get("rate_limit_tpm", 100000),
                cost_per_1k_input=detected.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=detected.get("cost_per_1k_output", 0.0),
            )
            result[name] = model

            logger.info(
                f"  Cloud: {name} ({config_provider}) "
                f"| {detected.get('tier', '?')} "
                f"| ctx={detected['context_length']} "
                f"| best={model.best_score():.1f}"
                f"{'| 👁️' if model.has_vision else ''}"
                f"{'| 🧠' if model.thinking_model else ''}"
            )

        return result

    def _load_ollama_models(self, overrides: dict) -> dict[str, ModelInfo]:
        """Auto-detect and load Ollama models."""
        result = {}

        try:
            import httpx
            r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
            if r.status_code != 200:
                return result
            ollama_models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            # Try CLI fallback
            try:
                import subprocess
                proc = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True, text=True, timeout=5,
                )
                if proc.returncode != 0:
                    return result
                ollama_models = []
                for line in proc.stdout.strip().split("\n")[1:]:
                    if line.strip():
                        ollama_models.append(line.split()[0])
            except Exception:
                return result

        for ollama_name in ollama_models:
            # Derive a clean registry name
            clean = ollama_name.replace(":", "-").replace("/", "-")
            name = f"ollama-{clean}"
            model_overrides = overrides.get(name, {})

            if model_overrides.get("disabled", False):
                continue

            # Detect family from ollama name
            detect_str = ollama_name.lower().replace(":", "-").replace("_", "-")
            family_key = detect_family(detect_str)

            # Estimate params from name
            param_match = re.search(r"(\d+\.?\d*)b", detect_str)
            total_params = float(param_match.group(1)) if param_match else 7.0

            # Quantization from name
            quant_match = re.search(
                r"(q[2-8]_k_\w+|q[2-8]_\w+|q[2-8]|f16|fp16)",
                detect_str,
                re.IGNORECASE,
            )
            quantization = quant_match.group(1).upper() if quant_match else "Q4_K_M"

            # MoE detection
            moe_match = re.search(r"a(\d+\.?\d*)b", detect_str)
            is_moe = moe_match is not None
            active_params = float(moe_match.group(1)) if moe_match else 0

            has_vision = detect_vision_support(family_key, {}, detect_str)

            capabilities = estimate_capabilities(
                family_key=family_key,
                total_params_b=total_params,
                active_params_b=active_params if is_moe else None,
                quantization=quantization,
                is_moe=is_moe,
                has_vision_hint=has_vision,
            )

            # Context from family default
            context_length = 32768
            if family_key and family_key in FAMILY_PROFILES:
                context_length = FAMILY_PROFILES[family_key].context_default
            context_length = model_overrides.get("context_length", context_length)

            model = ModelInfo(
                name=name,
                location="ollama",
                provider="ollama",
                litellm_name=f"ollama/{ollama_name}",
                capabilities=capabilities,
                context_length=context_length,
                max_tokens=model_overrides.get("max_tokens", min(context_length // 4, 8192)),
                supports_function_calling=detect_function_calling(family_key, {}),
                supports_json_mode=True,
                thinking_model=detect_thinking_model(family_key),
                has_vision=has_vision,
                model_type="moe" if is_moe else "dense",
                total_params_b=total_params,
                active_params_b=active_params,
                quantization=quantization,
                specialty=FAMILY_PROFILES.get(family_key, get_default_profile()).specialty if family_key else "",
                family=family_key or "unknown",
                is_loaded=True,  # Ollama models are always "loaded"
                api_base="http://localhost:11434",
            )
            result[name] = model

            logger.info(
                f"  Ollama: {name} | family={family_key or '?'} "
                f"| {total_params:.0f}B | best={model.best_score():.1f}"
            )

        return result

    def _load_custom_endpoints(self) -> dict[str, ModelInfo]:
        """Load models from custom OpenAI-compatible endpoints."""
        result = {}

        for env_var, provider_type in [
            ("LLAMA_CPP_ENDPOINTS", "llamacpp"),
            ("CUSTOM_OPENAI_ENDPOINTS", "custom_openai"),
        ]:
            raw = os.getenv(env_var, "").strip()
            if not raw:
                continue

            for entry in raw.split(","):
                entry = entry.strip()
                if "=" not in entry:
                    continue
                ep_name, url = entry.split("=", 1)
                ep_name = ep_name.strip()
                url = url.strip().rstrip("/")
                if not ep_name or not url:
                    continue

                # Probe endpoint
                alive = False
                try:
                    import httpx
                    r = httpx.get(f"{url}/v1/models", timeout=5.0)
                    alive = r.status_code == 200
                except Exception:
                    pass

                if not alive:
                    logger.warning(f"Custom endpoint '{ep_name}' at {url} not responding")
                    continue

                # Detect family from name
                family_key = detect_family(ep_name.lower().replace("_", "-"))
                capabilities = estimate_capabilities(
                    family_key=family_key,
                    total_params_b=7.0,  # unknown, conservative
                    active_params_b=None,
                    quantization="Q4_K_M",
                )

                name = f"{provider_type}-{ep_name}"
                result[name] = ModelInfo(
                    name=name,
                    location="local",
                    provider=provider_type,
                    litellm_name=f"openai/{ep_name}",
                    capabilities=capabilities,
                    context_length=8192,
                    max_tokens=4096,
                    supports_function_calling=False,
                    supports_json_mode=True,
                    is_loaded=True,
                    api_base=url,
                    family=family_key or "unknown",
                )
                logger.info(f"  Custom: {name} at {url}")

        return result

    @staticmethod
    def _get_available_vram() -> int:
        """Get available VRAM in MB."""
        try:
            from gpu_monitor import get_gpu_monitor
            gpu_state = get_gpu_monitor().get_state()
            return gpu_state.gpu.vram_total_mb if gpu_state.gpu.available else 0
        except Exception:
            # Fallback: try nvidia-smi
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    return int(result.stdout.strip().split("\n")[0])
            except Exception:
                pass
            return 0

    # ── Query Methods ────────────────────────────────────────────────────────

    def get(self, name: str) -> ModelInfo | None:
        return self.models.get(name)

    def find_by_litellm_name(self, litellm_name: str) -> ModelInfo | None:
        for m in self.models.values():
            if m.litellm_name == litellm_name:
                return m
        return None

    def local_models(self) -> list[ModelInfo]:
        return [m for m in self.models.values() if m.is_local]

    def cloud_models(self) -> list[ModelInfo]:
        return [m for m in self.models.values() if not m.is_local]

    def models_with_capability(
        self, capability: str, min_score: float = 1.0
    ) -> list[ModelInfo]:
        return [
            m for m in self.models.values()
            if m.capabilities.get(capability, 0) >= min_score
        ]

    def vision_models(self) -> list[ModelInfo]:
        return [m for m in self.models.values() if m.has_vision]

    def thinking_models(self) -> list[ModelInfo]:
        return [m for m in self.models.values() if m.thinking_model]

    def best_for_task(
        self,
        task_name: str,
        *,
        min_context: int = 0,
        needs_vision: bool = False,
        needs_function_calling: bool = False,
        needs_thinking: bool = False,
        prefer_local: bool = False,
        prefer_fast: bool = False,
        max_cost_per_1k_output: float = float("inf"),
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find the best models for a task.
        Returns [(model_name, score), ...] sorted by score descending.
        """
        requirements = TaskRequirements(
            task_name=task_name,
            min_context=min_context,
            needs_function_calling=needs_function_calling,
            needs_vision=needs_vision,
            needs_thinking=needs_thinking,
            max_cost_per_1k_output=max_cost_per_1k_output,
            prefer_local=prefer_local,
            prefer_fast=prefer_fast,
        )

        model_data = {
            name: (m.capabilities, m.operational_dict())
            for name, m in self.models.items()
        }
        return rank_models_for_task(model_data, requirements, top_k=top_k)

    def best_single_for_task(self, task_name: str, **kwargs) -> ModelInfo | None:
        """Convenience: returns the single best model, or None."""
        ranked = self.best_for_task(task_name, **kwargs)
        if ranked:
            return self.models.get(ranked[0][0])
        return None

    # ── State Management ─────────────────────────────────────────────────────

    def mark_loaded(self, name: str, api_base: str) -> None:
        with self._lock:
            # Unload other local models (only one can be loaded at a time)
            for m in self.models.values():
                if m.location == "local":
                    m.is_loaded = False
                    m.api_base = None
            if name in self.models:
                self.models[name].is_loaded = True
                self.models[name].api_base = api_base

    def mark_unloaded(self, name: str) -> None:
        with self._lock:
            if name in self.models:
                self.models[name].is_loaded = False
                self.models[name].api_base = None

    def currently_loaded(self) -> ModelInfo | None:
        for m in self.models.values():
            if m.location == "local" and m.is_loaded:
                return m
        return None

    def is_personal_project(self, project_name: str) -> bool:
        return any(p.lower() in project_name.lower() for p in self.personal_projects)

    # ── Quality Feedback Loop ────────────────────────────────────────────────

    def update_quality_from_grading(
        self, model_name: str, capability: str, measured_quality: float,
    ) -> None:
        """
        Update a model's capability score from grading feedback.
        EMA: new = 0.7 * old + 0.3 * measured
        """
        with self._lock:
            model = self.models.get(model_name)
            if not model:
                return
            old_q = model.capabilities.get(capability, 5.0)
            new_q = round(0.7 * old_q + 0.3 * measured_quality, 1)
            new_q = max(0.0, min(10.0, new_q))
            if abs(new_q - old_q) > 0.05:
                model.capabilities[capability] = new_q
                logger.info(f"Quality updated: {model_name}/{capability}: {old_q} → {new_q}")

    def update_speed(self, model_name: str, tokens_per_second: float) -> None:
        """Update measured inference speed."""
        with self._lock:
            if model_name in self.models:
                self.models[model_name].tokens_per_second = tokens_per_second

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a formatted summary of all registered models."""
        print("=" * 80)
        print(f"  📦 Model Registry — {len(self.models)} models")
        print("=" * 80)

        for location in ["local", "ollama", "cloud"]:
            models = [m for m in self.models.values() if m.location == location]
            if not models:
                continue

            icon = {"local": "💻", "ollama": "🦙", "cloud": "☁️"}[location]
            print(f"\n  {icon} {location.upper()} ({len(models)} models)")
            print(f"  {'─' * 76}")

            for m in sorted(models, key=lambda x: x.best_score(), reverse=True):
                # Top 3 capabilities
                top_caps = sorted(
                    m.capabilities.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                top_str = ", ".join(f"{k}={v:.1f}" for k, v in top_caps if v > 0)

                flags = ""
                if m.has_vision:
                    flags += "👁️"
                if m.thinking_model:
                    flags += "🧠"
                if m.supports_function_calling:
                    flags += "🔧"

                print(
                    f"  {m.name:40s} "
                    f"| best={m.best_score():4.1f} "
                    f"| {top_str:45s} "
                    f"| {flags}"
                )

        # Show task routing preview
        print(f"\n  🎯 Task Routing Preview")
        print(f"  {'─' * 76}")
        for task in ["planner", "coder", "fixer", "reviewer", "writer", "executor"]:
            ranked = self.best_for_task(task, top_k=3)
            if ranked:
                models_str = " → ".join(f"{n}({s:.1f})" for n, s in ranked)
                print(f"  {task:20s}: {models_str}")

        print("=" * 80)


# ─── Singleton ───────────────────────────────────────────────────────────────

_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """Thread-safe singleton access."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
                _registry.load()
    return _registry


def reload_registry() -> dict:
    """
    Hot reload — call this when you download a new model.
    Returns {"added": [...], "removed": [...], "total": N}

    Usage:
        from model_registry import reload_registry
        result = reload_registry()
        print(f"Added: {result['added']}")
    """
    return get_registry().reload()
