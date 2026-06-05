from __future__ import annotations

import random
import urllib.parse

import httpx

from ..types import ImageSpec

_BASE = "https://image.pollinations.ai/prompt/"
_TIMEOUT = 90.0


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
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content, {"seed_used": seed}
