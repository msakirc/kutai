"""husam (kumandan hüsamettin) — the non-agentic single-call worker.

Beckman admits a raw_dispatch task; the orchestrator pump dispatches it
here. husam selects a model (fatih_hoca), calls the dumb dispatcher
primitive (execute), maps the result. It runs no agent profile and parses
no ReAct action DSL. It imports nothing from coulson.
"""
from __future__ import annotations

from .worker import run

__all__ = ["run"]
