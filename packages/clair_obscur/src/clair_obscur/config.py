"""ClairObscur config — backend (comfyui|a1111), URL/port, model, weights dir.

Loaded from env so absent backend → clair_obscur.available() returns False
and hoca's image_select filters the local entry out (no crash)."""
from __future__ import annotations

import os
from dataclasses import dataclass


_VALID_BACKENDS = ("comfyui", "a1111")


@dataclass(frozen=True)
class ClairObscurConfig:
    backend: str            # "comfyui" | "a1111"
    host: str               # "127.0.0.1"
    port: int               # 8188 (comfyui) / 7860 (a1111) defaults
    base_url: str           # env CLAIR_OBSCUR_URL overrides host:port
    model: str              # SDXL / SD1.5 model filename or repo id
    weights_dir: str        # absolute path to backend's models directory
    exe_path: str           # absolute path to launcher
    idle_release_seconds: int = 60   # backstop after record_release_hint


def load_config() -> ClairObscurConfig:
    backend = os.getenv("CLAIR_OBSCUR_BACKEND", "comfyui").strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"CLAIR_OBSCUR_BACKEND={backend!r} not in {_VALID_BACKENDS}"
        )
    default_port = 8188 if backend == "comfyui" else 7860
    host = os.getenv("CLAIR_OBSCUR_HOST", "127.0.0.1")
    port = int(os.getenv("CLAIR_OBSCUR_PORT", str(default_port)))
    base_url = os.getenv("CLAIR_OBSCUR_URL", "").strip() or f"http://{host}:{port}"
    model = os.getenv("CLAIR_OBSCUR_MODEL", "sdxl-turbo")
    weights_dir = os.getenv("CLAIR_OBSCUR_WEIGHTS_DIR", "")
    exe_path = os.getenv("CLAIR_OBSCUR_EXE", "")
    idle = int(os.getenv("CLAIR_OBSCUR_IDLE_RELEASE_SECONDS", "60"))
    return ClairObscurConfig(
        backend=backend, host=host, port=port, base_url=base_url,
        model=model, weights_dir=weights_dir, exe_path=exe_path,
        idle_release_seconds=idle,
    )
