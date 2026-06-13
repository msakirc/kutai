"""Concrete PromptStore over the existing prompt_versions DB table.

DISPOSABLE SCAFFOLDING. This is the app-side adapter wired into finch
at startup, and the ONLY thing bridging the leaf to src DB. A future dedicated
DB-layer package will own all DB ops; when it lands, re-point THIS adapter at it
— the leaf and the PromptStore port never change. Do NOT add a foundry-owned
sqlite file; that would be thrown away by the DB package. Keep this adapter thin
and isolated to this one file.
"""
from __future__ import annotations
from typing import Optional

from src.memory import prompt_versions


class DbPromptStore:
    async def get_active(self, key: str) -> Optional[str]:
        return await prompt_versions.get_active_prompt(key)

    async def save_version(self, key: str, text: str, notes: str = "", activate: bool = False) -> int:
        return await prompt_versions.save_prompt_version(key, text, notes=notes, activate=activate)

    async def record_quality(self, key: str, score: float) -> None:
        await prompt_versions.record_prompt_quality(key, score)

    async def list_versions(self, key: str) -> list[dict]:
        return await prompt_versions.list_prompt_versions(key)
