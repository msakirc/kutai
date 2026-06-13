"""ClairObscur config — backend (comfyui|a1111), URL/port, launcher exe, cwd.

Loaded from env so absent backend → clair_obscur.available() returns False
and hoca's image_select filters the local entry out (no crash).

The checkpoint/model knob is NOT here: paintress's local_server provider
reads ``CLAIR_OBSCUR_MODEL`` directly into the ComfyUI workflow's
``ckpt_name`` when it submits a job. clair_obscur only manages the server
PROCESS lifecycle — it never picks the model — so it carries no model/weights
field (those were dead config nobody read)."""
from __future__ import annotations

import os
from dataclasses import dataclass


_VALID_BACKENDS = ("comfyui", "a1111")


@dataclass(frozen=True)
class ClairObscurConfig:
    """working_dir (env CLAIR_OBSCUR_DIR) is the directory the backend is
    launched FROM (Popen cwd). REQUIRED for comfyui: the launch cmd is
    ``<python> -u main.py ...`` and main.py resolves against cwd — without
    CLAIR_OBSCUR_DIR the launch resolves main.py against the orchestrator
    cwd and crashes. For a1111 it defaults to the launcher's own directory.
    Note: fatih_hoca's image_select availability gate only checks
    CLAIR_OBSCUR_EXE exists — it does NOT validate working_dir."""
    backend: str            # "comfyui" | "a1111"
    host: str               # "127.0.0.1"
    port: int               # 8188 (comfyui) / 7860 (a1111) defaults
    base_url: str           # env CLAIR_OBSCUR_URL overrides host:port
    exe_path: str           # absolute path to launcher
    working_dir: str = ""   # env CLAIR_OBSCUR_DIR — backend checkout dir (Popen cwd)
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
    exe_path = os.getenv("CLAIR_OBSCUR_EXE", "")
    working_dir = os.getenv("CLAIR_OBSCUR_DIR", "")
    idle = int(os.getenv("CLAIR_OBSCUR_IDLE_RELEASE_SECONDS", "60"))
    return ClairObscurConfig(
        backend=backend, host=host, port=port, base_url=base_url,
        exe_path=exe_path, working_dir=working_dir,
        idle_release_seconds=idle,
    )
