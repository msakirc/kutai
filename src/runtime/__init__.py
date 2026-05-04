"""Backward-compat shim — runtime relocated to packages/coulson/.

Phase B (2026-05-04): the agent runtime moved out of ``src/runtime/`` into
the standalone ``coulson`` package alongside ``general_beckman``,
``fatih_hoca``, ``dallama``, ``hallederiz_kadir``, etc.

Old call-sites that import ``from src.runtime import execute`` (or any
runtime submodule) keep working through this shim. New code should import
from ``coulson`` directly.

See:
    docs/superpowers/specs/2026-05-04-runtime-extraction-design.md
    docs/superpowers/plans/2026-05-04-runtime-extraction.md
"""
from coulson import execute  # noqa: F401

__all__ = ["execute"]
