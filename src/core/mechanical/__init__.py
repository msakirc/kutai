"""Backwards-compatible shim — the real package is ``mr_roboto``.

Imports like ``from src.core.mechanical.workspace_snapshot import ...`` are
served by :mod:`src.core.mechanical.workspace_snapshot`, which aliases
``mr_roboto.workspace_snapshot`` into :data:`sys.modules`. Nothing new should be
added here; edit :mod:`mr_roboto` instead.
"""
