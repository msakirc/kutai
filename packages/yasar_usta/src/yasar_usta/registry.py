"""Declarative registry loader: registry.yaml -> (HubConfig, [ProjectConfig])."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .config import GuardConfig, HubConfig, ProjectConfig, SidecarConfig


def _resolve(value, tokens: dict):
    """Substitute ${token} placeholders in strings (recursively in lists/dicts)."""
    if isinstance(value, str):
        for k, v in tokens.items():
            value = value.replace("${" + k + "}", v)
        return value
    if isinstance(value, list):
        return [_resolve(x, tokens) for x in value]
    if isinstance(value, dict):
        return {k: _resolve(v, tokens) for k, v in value.items()}
    return value


# Fields whose values are filesystem paths → normalize separators (Windows).
_PATH_FIELDS = ("cwd", "log_dir", "log_file", "heartbeat_file",
                "claude_signal_file", "claude_cmd")


def _norm(v):
    return str(Path(v)) if isinstance(v, str) and v else v


def _build_target(raw: dict, tokens: dict) -> GuardConfig:
    raw = _resolve(raw, tokens)
    if "id" not in raw or "command" not in raw:
        raise ValueError(f"target missing id/command: {raw!r}")
    for k in _PATH_FIELDS:
        if k in raw:
            raw[k] = _norm(raw[k])
    # command[0] and pid_file paths also normalized
    if isinstance(raw.get("command"), list):
        raw["command"] = [_norm(raw["command"][0])] + list(raw["command"][1:]) \
            if raw["command"] else raw["command"]
    sidecars = []
    for sc in raw.get("sidecars", []):
        if sc.get("pid_file"):
            sc["pid_file"] = _norm(sc["pid_file"])
        if isinstance(sc.get("command"), list) and sc["command"]:
            sc["command"] = [_norm(sc["command"][0])] + list(sc["command"][1:])
        sidecars.append(SidecarConfig(**sc))
    return GuardConfig(
        name=raw["id"],
        app_name=raw.get("app_name", raw["id"]),
        command=raw["command"],
        cwd=raw.get("cwd"),
        env=raw.get("env", {}),
        heartbeat_file=raw.get("heartbeat_file"),
        heartbeat_stale_seconds=raw.get("heartbeat_stale_seconds", 120),
        heartbeat_healthy_seconds=raw.get("heartbeat_healthy_seconds", 90),
        restart_exit_code=raw.get("restart_exit_code", 42),
        log_dir=raw.get("log_dir", "logs"),
        log_file=raw.get("log_file"),
        stop_timeout=raw.get("stop_timeout", 30),
        auto_restart=raw.get("auto_restart", True),
        backoff_steps=raw.get("backoff_steps", [5, 15, 60, 300]),
        claude_enabled=raw.get("claude_enabled", True),
        claude_cmd=raw.get("claude_cmd"),
        claude_name=raw.get("claude_name"),
        claude_signal_file=raw.get("claude_signal_file"),
        sidecars=sidecars,
        extra_processes=raw.get("extra_processes", []),
    )


def load_registry(path, project_root: str) -> tuple[HubConfig, list[ProjectConfig]]:
    """Parse registry.yaml. Fail fast on structural error (no partial start)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "projects" not in data:
        raise ValueError("registry.yaml must have a top-level 'projects' mapping")

    tokens = {"project_root": project_root}

    raw_hub = data.get("hub", {})
    hub = HubConfig(
        name=raw_hub.get("name", "Yaşar Usta"),
        telegram_token=os.getenv(raw_hub.get("telegram_token_env", ""), ""),
        telegram_chat_id=os.getenv(raw_hub.get("telegram_chat_id_env", ""), ""),
        log_dir=_resolve(raw_hub.get("log_dir", "logs"), tokens),
    )

    projects: list[ProjectConfig] = []
    for pid, raw_proj in data["projects"].items():
        if "targets" not in raw_proj or not raw_proj["targets"]:
            raise ValueError(f"project {pid!r} has no targets")
        targets = [_build_target(t, tokens) for t in raw_proj["targets"]]
        projects.append(ProjectConfig(
            id=pid,
            name=raw_proj.get("name", pid),
            targets=targets,
            hook_module=raw_proj.get("hook_module"),
        ))
    return hub, projects
