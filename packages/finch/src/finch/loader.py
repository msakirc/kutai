"""Load profiles/*.yaml into Profile singletons. Parsed ONCE at import.

Hot-path get_profile() must never parse YAML per call — it reads the
pre-built registry dict.
"""
from __future__ import annotations
import dataclasses
from pathlib import Path
import yaml

from .profile import Profile, WriterProfile

_PROFILES_DIR = Path(__file__).parent / "profiles"

# Public (non-private) field names accepted by Profile(**…); any other YAML
# key is silently dropped so future-proof YAML doesn't break older code.
_PROFILE_FIELDS = {f.name for f in dataclasses.fields(Profile) if not f.name.startswith("_")}

# Map of profile name → subclass to instantiate instead of the base Profile.
_PROFILE_CLASSES: dict[str, type[Profile]] = {
    "writer": WriterProfile,
}


def _load_all() -> dict[str, Profile]:
    registry: dict[str, Profile] = {}
    for yml in sorted(_PROFILES_DIR.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        if "name" not in data:
            raise ValueError(f"{yml}: missing required 'name' key")
        filtered = {k: v for k, v in data.items() if k in _PROFILE_FIELDS}
        cls = _PROFILE_CLASSES.get(filtered.get("name", ""), Profile)
        try:
            registry[filtered["name"]] = cls(**filtered)
        except TypeError as exc:
            raise TypeError(f"{yml}: {exc}") from exc
    return registry


PROFILE_REGISTRY: dict[str, Profile] = _load_all()


def get_profile(name: str) -> Profile | None:
    """Return the cached singleton Profile for `name`, or None if absent."""
    return PROFILE_REGISTRY.get(name)
