"""A11.r1 — Configurable handler registry for oncall_agent.

Replaces the hardcoded whitelist in ``mr_roboto.executors.oncall_action``
with a pluggable registry that supports multiple domains.  Existing Z8 ops
handlers register via ``register_handler('ops', ...)`` and the dispatch
table is unchanged — all existing oncall tests continue to pass.

New domains plug in via the same API::

    from coulson.agent_handlers.registry import register_handler

    async def _my_handler(verb, params, mission_id):
        ...

    register_handler('mention', 'my_verb', _my_handler)

Public API
----------
register_handler(domain, name, fn)
    Register an async handler for (domain, name).  Idempotent: re-registering
    the same key silently replaces.

lookup_handler(domain, name) -> callable | None
    Return the handler or None if not found.

list_verbs(domain=None) -> frozenset[str]
    Return all registered verb names, optionally filtered by domain.

get_whitelist(domain) -> frozenset[str]
    Return the registered verb names for a given domain.

is_registered(domain, name) -> bool
    True if handler exists for (domain, name).
"""
from __future__ import annotations

from typing import Any, Callable, Coroutine

from src.infra.logging_config import get_logger

logger = get_logger("coulson.agent_handlers.registry")

# Registry: domain -> verb_name -> async callable
# Intentionally not frozen — modules register at import time.
_REGISTRY: dict[str, dict[str, Callable[..., Coroutine[Any, Any, dict]]]] = {}


def register_handler(
    domain: str,
    name: str,
    fn: Callable[..., Coroutine[Any, Any, dict]],
) -> None:
    """Register an async handler for (domain, name).

    Parameters
    ----------
    domain:
        Logical domain grouping (e.g. ``'ops'``, ``'mention'``).
    name:
        Verb name as it appears in the task payload (e.g.
        ``'restart_service'``, ``'mention_ingest'``).
    fn:
        Async callable with signature
        ``async (verb: str, params: dict, mission_id: int) -> dict``.
        Should return a dict with at least a ``'status'`` key.
    """
    if domain not in _REGISTRY:
        _REGISTRY[domain] = {}
    if name in _REGISTRY[domain]:
        logger.debug(
            "agent_handlers.registry: replacing existing handler",
            domain=domain,
            name=name,
        )
    _REGISTRY[domain][name] = fn
    logger.debug(
        "agent_handlers.registry: registered handler",
        domain=domain,
        name=name,
    )


def lookup_handler(
    domain: str,
    name: str,
) -> Callable[..., Coroutine[Any, Any, dict]] | None:
    """Return handler for (domain, name), or None."""
    return _REGISTRY.get(domain, {}).get(name)


def list_verbs(domain: str | None = None) -> frozenset[str]:
    """Return all registered verb names across all domains, or within *domain*."""
    if domain is not None:
        return frozenset(_REGISTRY.get(domain, {}).keys())
    all_verbs: set[str] = set()
    for verbs in _REGISTRY.values():
        all_verbs.update(verbs.keys())
    return frozenset(all_verbs)


def get_whitelist(domain: str) -> frozenset[str]:
    """Return verb names for *domain* (alias for list_verbs(domain))."""
    return list_verbs(domain)


def is_registered(domain: str, name: str) -> bool:
    """True if handler exists for (domain, name)."""
    return name in _REGISTRY.get(domain, {})


def _clear_registry() -> None:
    """Test helper — wipe registry state between tests."""
    _REGISTRY.clear()
