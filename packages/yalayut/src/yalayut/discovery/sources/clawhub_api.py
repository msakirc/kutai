"""Yalayut Phase 4 — ClawHub source adapter.

EXPLICIT STUB. ClawHub's catalog is not enumerable without a headless
browser (spec non-goal). ``discover()`` returns an empty list so the cron
loop can include this adapter without special-casing it. When ClawHub
exposes a public API, replace ``discover()``/``fetch()`` with a real
implementation — no other Phase 4 file needs to change.
"""
from __future__ import annotations

from pathlib import Path

from yazbunu import get_logger
from yalayut.contracts import ArtifactRef, SourceConfig

logger = get_logger("yalayut.adapter.clawhub")


class ClawHubAdapter:
    source_type: str = "clawhub_api"

    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]:
        logger.debug("clawhub_api adapter is a stub — discover() returns []")
        return []

    async def fetch(self, ref: ArtifactRef) -> Path:
        raise NotImplementedError("clawhub_api adapter is a stub")
