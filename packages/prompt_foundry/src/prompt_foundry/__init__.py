"""Prompt Foundry — leaf package owning prompt/profile content + build API.

Depends on NOTHING in src/ or feature packages. Storage is injected via
set_store(); with no store, profiles fall back to in-package YAML seeds.
"""
from .profile import Profile
from .loader import PROFILE_REGISTRY, get_profile
from .store import PromptStore, set_store, get_active, record_quality
from .build import build_messages, register_rubric

__all__ = [
    "Profile", "PROFILE_REGISTRY", "get_profile",
    "PromptStore", "set_store", "get_active", "record_quality",
    "build_messages", "register_rubric",
]
