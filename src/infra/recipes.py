"""Shim — recipes relocated to ``yalayut`` (the vetted-reusable-block catalog
that already owns the ``run_recipe`` / cookiecutter discovery path).

Kept as a thin re-export because ``src`` core still imports it
(``src/ops/playbooks.py`` → ``_load_yaml``). New code should import from
``yalayut.recipes`` directly. The ``src``→``src`` cleanup of this shim is
out of scope.
"""
from __future__ import annotations

from yalayut.recipes import (  # noqa: F401
    Recipe,
    _load_yaml,
    get_pinned_recipes,
    instantiate_recipe,
    list_recipes,
    load_recipe,
    match_recipe,
    pin_recipe,
    pin_recipes_from_artifact,
)

__all__ = [
    "Recipe",
    "_load_yaml",
    "get_pinned_recipes",
    "instantiate_recipe",
    "list_recipes",
    "load_recipe",
    "match_recipe",
    "pin_recipe",
    "pin_recipes_from_artifact",
]
