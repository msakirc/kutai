from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ImageSpec:
    prompt: str
    out_dir: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    seed: int | None = None
    quality_tier: str = "fast"
    filename_hint: str | None = None


@dataclass
class ImageResult:
    path: str | None = None
    provider: str = ""
    model: str = ""
    cost: float = 0.0
    latency: float = 0.0
    seed_used: int | None = None
    error: str | None = None
