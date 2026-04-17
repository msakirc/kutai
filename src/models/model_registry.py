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
    """Return the canonical model registry.

    Prefers fatih_hoca's registry (populated by fatih_hoca.init() with
    YAML + GGUF + variants).  Falls back to a local-only YAML registry
    if fatih_hoca hasn't been initialized yet.
    """
    global _registry
    try:
        import fatih_hoca
        fh_reg = fatih_hoca._registry
        if fh_reg is not None:
            _registry = fh_reg
            return fh_reg
    except Exception:
        pass
    if _registry is None:
        _registry = ModelRegistry()
        _registry.load()
    return _registry


def reload_registry() -> dict:
    """Hot reload — call this when you download a new model.
    Returns {"added": [...], "removed": [...], "total": N}
    """
    return get_registry().reload()
