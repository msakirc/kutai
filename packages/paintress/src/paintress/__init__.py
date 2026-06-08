"""paintress — image interaction caller. Given hoca's pick, calls the provider,
validates via renoir, writes the PNG, returns ImageResult. LLM-free."""
from __future__ import annotations

import os
import re
import time

from renoir import assess
from .types import ImageSpec, ImageResult
from .providers.pollinations import PollinationsProvider
from .providers.huggingface import HuggingFaceProvider
from .providers.local_server import LocalServerProvider

_PROVIDERS: dict = {
    "pollinations": PollinationsProvider(),
    "huggingface": HuggingFaceProvider(),
    "clair_obscur": LocalServerProvider(),
}


def _safe_filename(hint: str | None) -> str:
    base = (hint or "image").strip() or "image"
    base = re.sub(r"[^A-Za-z0-9_\-]", "_", base)[:48]
    return f"{base}_{int(time.time() * 1000) % 1_000_000}.png"


async def generate(pick, spec: ImageSpec) -> ImageResult:
    model = pick.model
    provider = getattr(model, "provider", "")
    prov = _PROVIDERS.get(provider)
    if prov is None or not prov.available():
        return ImageResult(provider=provider, model=getattr(model, "name", ""),
                           error=f"unknown_provider:{provider}")

    base_url = getattr(model, "api_base", None) or getattr(model, "endpoint", None)
    started = time.time()
    try:
        data, meta = await prov.generate(spec, base_url=base_url)
    except Exception as exc:
        return ImageResult(provider=provider, model=getattr(model, "name", ""),
                           error=f"provider_raised:{exc.__class__.__name__}:{exc}")

    verdict = assess(data)
    if not verdict.ok:
        return ImageResult(provider=provider, model=getattr(model, "name", ""),
                           error=f"quality_failure:{verdict.reason}")

    os.makedirs(spec.out_dir, exist_ok=True)
    path = os.path.join(spec.out_dir, _safe_filename(spec.filename_hint))
    # Belt-and-suspenders: verify the resolved path is still inside out_dir.
    real_out = os.path.realpath(spec.out_dir)
    real_path = os.path.realpath(path)
    if not (real_path.startswith(real_out + os.sep) or real_path == real_out):
        return ImageResult(provider=provider, model=getattr(model, "name", ""),
                           error="path_escape")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

    return ImageResult(
        path=path, provider=provider, model=getattr(model, "name", ""),
        cost=float(getattr(model, "cost_per_image", 0.0) or 0.0),
        latency=time.time() - started,
        seed_used=(meta or {}).get("seed_used"),
    )
