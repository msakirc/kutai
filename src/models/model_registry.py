"""Model registry — shim re-exporting from fatih_hoca.registry."""
from fatih_hoca.registry import (  # noqa: F401
    ModelInfo,
    ModelRegistry,
    scan_model_directory,
    calculate_dynamic_context,
    calculate_gpu_layers,
    detect_vision_support,
    find_mmproj_path,
    detect_function_calling,
    detect_thinking_model,
    estimate_capabilities,
    read_gguf_metadata,
    KNOWN_PROVIDERS,
    PROVIDER_PREFIXES,
    _FREE_TIER_DEFAULTS,
    _TOOL_CALL_FAMILIES,
    _THINKING_FAMILIES,
    _apply_thinking_deltas,
    _create_model_variants,
    detect_cloud_model,
)

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        _registry.load()
    return _registry


def reload_registry() -> dict:
    """Hot reload — call this when you download a new model.
    Returns {"added": [...], "removed": [...], "total": N}
    """
    return get_registry().reload()
