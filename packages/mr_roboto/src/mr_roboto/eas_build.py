"""Wrap ``eas build`` — Z5 T3 mobile adapter.

Mechanical executor. No LLM. Shells out (via :func:`run_cmd`) to Expo
Application Services (EAS), Expo's *cloud* build service.

Why cloud
---------
The host OS is Windows. iOS builds require macOS-local tooling (Xcode,
``xcrun``) that cannot run here, so for iOS, EAS Build is the **only**
viable path — not a convenience. Android can build locally (see
``android_build``) but EAS Build also covers Android, so a single
``platform=all`` cloud build is the simplest cross-platform path.

Invocation
----------
EAS ships as ``eas-cli``, typically installed globally but invokable via
``npx eas-cli`` when it is not. This adapter calls
``npx --yes eas-cli build``. If ``npx`` (Node) is absent the run
soft-skips for free via ``run_cmd``'s missing-exe handling.

Always non-interactive: ``--non-interactive`` plus an explicit
``--platform`` / ``--profile`` so the CLI never blocks on a prompt.

Result parsing
--------------
EAS prints a build id and a build-details URL. This adapter scrapes both
out of stdout/stderr into ``build_id`` / ``build_url`` so downstream
verbs (notably ``eas_submit``) can reference the build without re-parsing.

Reversibility: ``full`` — a cloud build produces a disposable, fully
reproducible artifact on Expo's servers. Nothing user-visible is
published; re-running yields an equivalent build.
"""

from __future__ import annotations

import re
from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.eas_build")

_ALLOWED_PLATFORMS = frozenset({"ios", "android", "all"})

_TAIL_CHARS = 6000
DEFAULT_TIMEOUT_S = 1800.0  # EAS cloud builds are slow

# A build id is a UUID; the build URL is an expo.dev/.../builds/<id> link.
_BUILD_URL_RE = re.compile(
    r"https?://(?:expo\.dev|[\w.-]*\.?expo\.dev)/[^\s\"'<>]*/builds/"
    r"([0-9a-fA-F-]{8,})",
)
_BUILD_ID_RE = re.compile(
    r"\bbuild\s*id[:\s]+([0-9a-fA-F]{8}-[0-9a-fA-F-]{20,})\b",
    re.IGNORECASE,
)
_BARE_UUID_RE = re.compile(
    r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b",
)


def _tail(text: str | None) -> str:
    text = text or ""
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


def _is_missing_exe(raw: dict[str, Any]) -> bool:
    err = (raw.get("error") or "").lower()
    return "executable not found" in err or "not found" in err


def _parse_build_artifacts(text: str) -> tuple[str | None, str | None]:
    """Scrape (build_id, build_url) out of EAS CLI output.

    Falls back through three patterns: an explicit ``builds/<id>`` URL, a
    ``Build id: <uuid>`` line, then any bare UUID. Returns ``(None, None)``
    when nothing matches.
    """
    build_url: str | None = None
    build_id: str | None = None

    url_match = _BUILD_URL_RE.search(text)
    if url_match:
        build_url = url_match.group(0)
        build_id = url_match.group(1)

    if build_id is None:
        id_match = _BUILD_ID_RE.search(text)
        if id_match:
            build_id = id_match.group(1)

    if build_id is None:
        uuid_match = _BARE_UUID_RE.search(text)
        if uuid_match:
            build_id = uuid_match.group(1)

    return build_id, build_url


async def eas_build(
    mission_id: int | None,
    platform: str = "all",
    profile: str = "production",
    workspace_path: str | None = None,
    non_interactive: bool = True,
    extra_args: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Trigger an EAS cloud build and return a structured result.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd`` for workspace resolution.
    platform:
        ``ios`` / ``android`` / ``all``.
    profile:
        EAS build profile name from ``eas.json`` (e.g. ``production``,
        ``preview``).
    workspace_path:
        Explicit workspace root (the Expo project containing ``eas.json``).
    non_interactive:
        When True (default) ``--non-interactive`` is passed so the CLI
        never blocks on a prompt.
    extra_args:
        Extra argv appended after the standard flags.
    timeout_s:
        Hard cap. Default 1800 s — cloud builds are slow.

    Returns
    -------
    dict with keys ``ok, skipped, platform, profile, build_id, build_url,
    exit, stdout_tail, stderr_tail, duration_s, error``.
    """
    plat = (platform or "all").strip().lower()
    if plat not in _ALLOWED_PLATFORMS:
        return {
            "ok": False,
            "skipped": False,
            "platform": plat,
            "profile": profile,
            "build_id": None,
            "build_url": None,
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": (
                f"unsupported eas build platform {plat!r}; "
                f"allowed: {sorted(_ALLOWED_PLATFORMS)}"
            ),
        }

    cmd = ["npx", "--yes", "eas-cli", "build", "--platform", plat, "--profile", profile]
    if non_interactive:
        cmd.append("--non-interactive")
    cmd.extend(list(extra_args or []))

    raw = await run_cmd(
        mission_id=mission_id,
        cmd=cmd,
        cwd=None,
        timeout_s=timeout_s,
        require_exit_zero=False,
        workspace_path=workspace_path,
        reversibility_intent="full",
    )

    if _is_missing_exe(raw):
        logger.warning("eas-cli / npx not installed — eas_build skipped", platform=plat)
        return {
            "ok": True,
            "skipped": True,
            "platform": plat,
            "profile": profile,
            "build_id": None,
            "build_url": None,
            "exit": int(raw.get("exit", -1)),
            "stdout_tail": _tail(raw.get("stdout_tail")),
            "stderr_tail": _tail(raw.get("stderr_tail")),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": None,
        }

    exit_code = int(raw.get("exit", -1))
    timed_out = bool(raw.get("timed_out"))
    stdout = raw.get("stdout_tail") or ""
    stderr = raw.get("stderr_tail") or ""
    build_id, build_url = _parse_build_artifacts(stdout + "\n" + stderr)

    if timed_out:
        return {
            "ok": False,
            "skipped": False,
            "platform": plat,
            "profile": profile,
            "build_id": build_id,
            "build_url": build_url,
            "exit": exit_code,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": f"eas build timed out after {timeout_s}s",
        }

    ok = exit_code == 0
    return {
        "ok": ok,
        "skipped": False,
        "platform": plat,
        "profile": profile,
        "build_id": build_id,
        "build_url": build_url,
        "exit": exit_code,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "duration_s": float(raw.get("duration_s", 0.0)),
        "error": None if ok else f"eas build exited {exit_code}",
    }
