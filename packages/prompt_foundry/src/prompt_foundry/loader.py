"""Load profiles/*.yaml into Profile singletons. Parsed ONCE at import.

Hot-path get_profile() must never parse YAML per call — it reads the
pre-built registry dict.
"""
from __future__ import annotations
from pathlib import Path
import yaml

from .profile import Profile

_PROFILES_DIR = Path(__file__).parent / "profiles"


def _load_all() -> dict[str, Profile]:
    registry: dict[str, Profile] = {}
    for yml in sorted(_PROFILES_DIR.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        name = data["name"]
        registry[name] = Profile(**data)
    return registry


PROFILE_REGISTRY: dict[str, Profile] = _load_all()


def get_profile(name: str) -> Profile | None:
    """Return the cached singleton Profile for `name`, or None if absent."""
    return PROFILE_REGISTRY.get(name)
