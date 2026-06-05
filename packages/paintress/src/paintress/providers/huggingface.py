from __future__ import annotations

import os
import httpx

from ..types import ImageSpec

_MODEL = "black-forest-labs/FLUX.1-schnell"
_URL = f"https://api-inference.huggingface.co/models/{_MODEL}"
_TIMEOUT = 120.0


class HuggingFaceProvider:
    name = "huggingface"

    def available(self) -> bool:
        return bool(os.getenv("HF_TOKEN"))

    async def generate(self, spec: ImageSpec, *, base_url: str | None = None):
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}", "Accept": "image/png"}
        payload = {"inputs": spec.prompt or ""}
        if spec.seed is not None:
            payload["parameters"] = {"seed": spec.seed}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(base_url or _URL, headers=headers, json=payload)
        if resp.status_code == 503:
            raise RuntimeError("model_loading: HF FLUX cold start, retry")
        if resp.status_code in (401, 403):
            raise RuntimeError(f"hf_auth:{resp.status_code}")
        resp.raise_for_status()
        return resp.content, {"seed_used": spec.seed}
