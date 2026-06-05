from __future__ import annotations
from typing import Protocol
from ..types import ImageSpec


class ImageProvider(Protocol):
    name: str
    def available(self) -> bool: ...
    async def generate(self, spec: ImageSpec, *, base_url: str | None = None) -> tuple[bytes, dict]:
        """Return (image_bytes, meta). meta may carry {'seed_used': int}.

        Raise-tolerant contract: providers SHOULD return on error rather than
        raise, but the caller (paintress.generate) catches and maps to
        ImageResult.error as a backstop.
        """
        ...
