"""Static image-provider catalog. Image providers are NOT in cloud /models
discovery (LLM-only), so they're registered here.

Plan 2 will append the local clair_obscur entry; Plan 1 ships cloud only.
"""
from __future__ import annotations

from .registry import ImageModelInfo


def image_catalog() -> list[ImageModelInfo]:
    return [
        ImageModelInfo(
            name="pollinations/flux", provider="pollinations", location="cloud",
            endpoint="https://image.pollinations.ai/prompt/",
            quality_rank=6.0, cost_per_image=0.0, vram_mb=0,
            supports_seed=True, tier="free",
        ),
        ImageModelInfo(
            name="huggingface/flux-schnell", provider="huggingface", location="cloud",
            endpoint="https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
            quality_rank=8.0, cost_per_image=0.0, vram_mb=0,
            supports_seed=True, tier="free",
        ),
    ]
