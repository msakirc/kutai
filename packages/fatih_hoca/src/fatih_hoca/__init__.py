"""Fatih Hoca — model manager: scoring, selection, swap budget."""
from __future__ import annotations

from fatih_hoca.types import Pick, Failure, SwapBudget
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements, AGENT_REQUIREMENTS, CAPABILITY_TO_TASK
from fatih_hoca.capabilities import Cap, ALL_CAPABILITIES, TASK_PROFILES
from fatih_hoca.ranking import ScoredModel
from fatih_hoca.selector import Selector

__all__ = [
    "init", "select", "all_models",
    "Pick", "Failure", "ModelInfo", "ModelRequirements", "ScoredModel",
    "AGENT_REQUIREMENTS", "CAPABILITY_TO_TASK",
    "Cap", "ALL_CAPABILITIES", "TASK_PROFILES",
]

_selector: Selector | None = None
_registry: ModelRegistry | None = None


def init(
    models_dir: str | None = None,
    catalog_path: str | None = None,
    nerd_herd: object = None,
    available_providers: set[str] | None = None,
) -> list[str]:
    """
    Initialize the Fatih Hoca model registry and selector.

    Parameters
    ----------
    models_dir : str, optional
        Directory to scan for GGUF files (local models).
    catalog_path : str, optional
        Path to a YAML model catalog (cloud + YAML-declared local models).
    nerd_herd : object, optional
        Nerd Herd instance providing system snapshots. If None, a no-op
        stub is used (snapshot() returns an empty SystemSnapshot).
    available_providers : set[str], optional
        Set of cloud provider names that have API keys configured.
        Cloud models whose provider is not in this set are filtered out.
        If None, all cloud models are eligible (no API key check).

    Returns
    -------
    list[str]
        Names of all models registered.
    """
    global _selector, _registry

    if nerd_herd is None:
        from nerd_herd.types import SystemSnapshot

        class _NoopNerdHerd:
            def snapshot(self) -> SystemSnapshot:
                return SystemSnapshot()

        nerd_herd = _NoopNerdHerd()

    _registry = ModelRegistry()
    model_names: list[str] = []

    if catalog_path:
        models = _registry.load_yaml(catalog_path)
        model_names.extend(m.name for m in models)

    if models_dir:
        models = _registry.load_gguf_dir(models_dir)
        model_names.extend(m.name for m in models)

    # Load persisted speed measurements + demoted flags into model objects.
    # Without this, all models default to 10 tok/s → timeouts are too short.
    _registry._load_speed_cache()

    _selector = Selector(
        registry=_registry,
        nerd_herd=nerd_herd,
        available_providers=available_providers,
    )
    return model_names


def select(**kwargs) -> Pick | None:
    """
    Select the best model for a task.

    Keyword arguments are forwarded to Selector.select(). Returns a Pick
    with the chosen model and estimated min_time_seconds, or None if no
    eligible model was found or if init() has not been called.

    See Selector.select() for the full list of keyword arguments.
    """
    if _selector is None:
        return None
    return _selector.select(**kwargs)


def all_models() -> list[ModelInfo]:
    """
    Return all models currently in the registry.

    Returns an empty list if init() has not been called.
    """
    if _registry is None:
        return []
    return _registry.all_models()
