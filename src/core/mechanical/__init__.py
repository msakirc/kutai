"""Backwards-compatible shim — the real package is ``salako``.

Imports like ``from src.core.mechanical.workspace_snapshot import ...`` are
served by :mod:`src.core.mechanical.workspace_snapshot`, which aliases
``salako.workspace_snapshot`` into :data:`sys.modules`. Nothing new should be
added here; edit :mod:`salako` instead.
"""
