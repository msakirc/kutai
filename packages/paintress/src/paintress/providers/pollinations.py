from __future__ import annotations

import random
import urllib.parse

import httpx

from ..types import ImageSpec

_BASE = "https://image.pollinations.ai/prompt/"
_TIMEOUT = 90.0
_MAX_BYTES = 25 * 1024 * 1024  # 25 MB — cap against runaway provider responses


class PollinationsProvider:
    name = "pollinations"

    def available(self) -> bool:
        return True  # no key, public free service

    async def generate(self, spec: ImageSpec, *, base_url: str | None = None):
        seed = spec.seed if spec.seed is not None else random.randint(1, 2_000_000_000)
        # safe="" encodes "/" too — pollinations puts the prompt in the path.
        prompt = urllib.parse.quote(spec.prompt or "", safe="")
        params = {
            "width": spec.width, "height": spec.height,
            "seed": seed, "model": "flux", "nologo": "true",
        }
        url = f"{base_url or _BASE}{prompt}?" + urllib.parse.urlencode(params)
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True, max_redirects=5) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > _MAX_BYTES:
                        raise RuntimeError("response_too_large")
                return bytes(buf), {"seed_used": seed}
