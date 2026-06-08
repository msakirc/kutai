"""LocalServerProvider — paintress provider that calls clair_obscur's local
backend (ComfyUI default, A1111 via env). Backend chosen via clair_obscur
config so paintress doesn't duplicate config logic."""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Tuple

import httpx

from ..types import ImageSpec

_PROMPT_POLL_INTERVAL = 1.0
_PROMPT_TIMEOUT = 180.0


class LocalServerProvider:
    name = "clair_obscur"

    def available(self) -> bool:
        """Mirrors clair_obscur.available(). False when no backend exe set.
        Always re-reads CLAIR_OBSCUR_EXE from env so tests (and runtime env
        changes) are reflected without stale-singleton interference."""
        exe = os.getenv("CLAIR_OBSCUR_EXE", "")
        if exe and os.path.exists(exe):
            return True
        # Fallback: delegate to clair_obscur singleton (may have a different
        # exe configured via non-env means — future-proof hook).
        try:
            from clair_obscur import available as _co_available
            return bool(_co_available())
        except Exception:
            return False

    async def generate(
        self, spec: ImageSpec, *, base_url: str | None = None
    ) -> Tuple[bytes, dict]:
        backend = (os.getenv("CLAIR_OBSCUR_BACKEND", "comfyui") or "comfyui").lower()
        if not base_url:
            try:
                from clair_obscur import base_url as _co_base_url
                base_url = _co_base_url()
            except Exception:
                base_url = "http://127.0.0.1:8188"
        if backend == "a1111":
            return await self._a1111(spec, base_url)
        return await self._comfyui(spec, base_url)

    async def _a1111(self, spec: ImageSpec, base_url: str) -> Tuple[bytes, dict]:
        url = f"{base_url.rstrip('/')}/sdapi/v1/txt2img"
        payload = {
            "prompt": spec.prompt or "",
            "negative_prompt": spec.negative_prompt or "",
            "width": int(spec.width), "height": int(spec.height),
            "steps": int(spec.steps) if spec.steps else 20,
            "seed": -1 if spec.seed is None else int(spec.seed),
        }
        async with httpx.AsyncClient(timeout=_PROMPT_TIMEOUT) as c:
            resp = await c.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        images = body.get("images") or []
        if not images:
            raise RuntimeError("a1111_no_image")
        data = base64.b64decode(images[0])
        info = body.get("info") or "{}"
        try:
            info_d = json.loads(info)
            seed_used = info_d.get("seed", spec.seed)
        except Exception:
            seed_used = spec.seed
        return data, {"seed_used": seed_used}

    async def _comfyui(self, spec: ImageSpec, base_url: str) -> Tuple[bytes, dict]:
        base = base_url.rstrip("/")
        workflow = self._build_comfyui_workflow(spec)
        async with httpx.AsyncClient(timeout=_PROMPT_TIMEOUT) as c:
            resp = await c.post(f"{base}/prompt", json={"prompt": workflow})
            resp.raise_for_status()
            prompt_id = (resp.json() or {}).get("prompt_id")
            if not prompt_id:
                raise RuntimeError("comfyui_no_prompt_id")
            deadline = asyncio.get_event_loop().time() + _PROMPT_TIMEOUT
            image_meta = None
            while asyncio.get_event_loop().time() < deadline:
                h = await c.get(f"{base}/history/{prompt_id}")
                h.raise_for_status()
                hist = h.json() or {}
                entry = hist.get(prompt_id)
                if entry and entry.get("outputs"):
                    for _id, out in entry["outputs"].items():
                        imgs = out.get("images") or []
                        if imgs:
                            image_meta = imgs[0]
                            break
                if image_meta is not None:
                    break
                await asyncio.sleep(_PROMPT_POLL_INTERVAL)
            if image_meta is None:
                raise RuntimeError("comfyui_timeout")
            params = {
                "filename": image_meta["filename"],
                "subfolder": image_meta.get("subfolder", ""),
                "type": image_meta.get("type", "output"),
            }
            v = await c.get(f"{base}/view", params=params)
            v.raise_for_status()
            return v.content, {"seed_used": spec.seed}

    def _build_comfyui_workflow(self, spec: ImageSpec) -> dict:
        seed = int(spec.seed) if spec.seed is not None else 0
        steps = int(spec.steps) if spec.steps else 20
        return {
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": steps, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0],
                "negative": ["7", 0], "latent_image": ["5", 0],
            }},
            "4": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": os.getenv("CLAIR_OBSCUR_MODEL",
                                                    "sdxl-turbo")}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {
                "width": int(spec.width), "height": int(spec.height),
                "batch_size": 1,
            }},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": spec.prompt or "", "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": spec.negative_prompt or "", "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "kutai", "images": ["8", 0]}},
        }
