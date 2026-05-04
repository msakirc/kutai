"""Runtime — multi-call orchestration for LLM tasks.

Architecture spec: docs/superpowers/specs/2026-05-04-runtime-extraction-design.md
Plan: docs/superpowers/plans/2026-05-04-runtime-extraction.md

Currently transitional: BaseAgent delegates to runtime modules during Phase A.
Full execute() entry point lands at Phase A.10.
"""
