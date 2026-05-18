"""Wrap ``eas submit`` — Z5 T3 mobile adapter.

Mechanical executor. No LLM. Shells out (via :func:`run_cmd`) to EAS
Submit, which uploads a finished EAS build to a store distribution track:
TestFlight for iOS, the Play internal track for Android.

Why this is irreversible
------------------------
Unlike ``eas_build`` (which produces a disposable cloud artifact),
``eas submit`` pushes a binary onto a real store track. Testers receive
it, Apple/Google ingest it, and the upload cannot be cleanly un-done — at
best a build is *expired* or *withdrawn*, never erased. The verb is
therefore tagged ``irreversible`` so the dispatcher gates it behind a
confirmation when one is required.

Invocation
----------
``npx --yes eas-cli submit --platform <p> --profile <profile>``
with either ``--id <build_id>`` (a specific build) or ``--latest`` (most
recent build for the platform). Always ``--non-interactive``. If ``npx``
(Node) is absent the run soft-skips for free.

Result parsing
--------------
The submission id / URL printed by EAS is scraped into
``submission_id`` / ``submission_url`` for downstream tracking.
"""

from __future__ import annotations

import re
from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.eas_submit")

_ALLOWED_PLATFORMS = frozenset({"ios", "android"})

_TAIL_CHARS = 6000
DEFAULT_TIMEOUT_S = 900.0  # uploads are slower than a check, faster than a build

_SUBMISSION_URL_RE = re.compile(
    r"https?://(?:expo\.dev|[\w.-]*\.?expo\.dev)/[^\s\"'<>]*/submissions/"
    r"([0-9a-fA-F-]{8,})",
)
_SUBMISSION_ID_RE = re.compile(
    r"\bsubmission\s*id[:\s]+([0-9a-fA-F]{8}-[0-9a-fA-F-]{20,})\b",
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


def _parse_submission_artifacts(text: str) -> tuple[str | None, str | None]:
    """Scrape (submission_id, submission_url) out of EAS CLI output."""
    submission_url: str | None = None
    submission_id: str | None = None

    url_match = _SUBMISSION_URL_RE.search(text)
    if url_match:
        submission_url = url_match.group(0)
        submission_id = url_match.group(1)

    if submission_id is None:
        id_match = _SUBMISSION_ID_RE.search(text)
        if id_match:
            submission_id = id_match.group(1)

    if submission_id is None:
        uuid_match = _BARE_UUID_RE.search(text)
        if uuid_match:
            submission_id = uuid_match.group(1)

    return submission_id, submission_url


async def eas_submit(
    mission_id: int | None,
    platform: str,
    workspace_path: str | None = None,
    build_id: str | None = None,
    latest: bool = False,
    profile: str = "production",
    non_interactive: bool = True,
    extra_args: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Upload an EAS build to a store track and return a structured result.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd`` for workspace resolution.
    platform:
        ``ios`` (→ TestFlight) or ``android`` (→ Play internal track).
        ``all`` is not accepted — submit targets one store at a time.
    workspace_path:
        Explicit workspace root (the Expo project containing ``eas.json``).
    build_id:
        A specific EAS build id to submit. Mutually exclusive with
        ``latest``; when both are absent ``latest`` is implied.
    latest:
        Submit the most recent build for the platform.
    profile:
        EAS submit profile from ``eas.json``.
    non_interactive:
        When True (default) ``--non-interactive`` is passed.
    extra_args:
        Extra argv appended after the standard flags.
    timeout_s:
        Hard cap. Default 900 s.

    Returns
    -------
    dict with keys ``ok, skipped, platform, profile, build_id,
    submission_id, submission_url, exit, stdout_tail, stderr_tail,
    duration_s, error``.
    """
    plat = (platform or "").strip().lower()
    if plat not in _ALLOWED_PLATFORMS:
        return {
            "ok": False,
            "skipped": False,
            "platform": plat,
            "profile": profile,
            "build_id": build_id,
            "submission_id": None,
            "submission_url": None,
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": (
                f"unsupported eas submit platform {plat!r}; "
                f"allowed: {sorted(_ALLOWED_PLATFORMS)}"
            ),
        }

    cmd = ["npx", "--yes", "eas-cli", "submit", "--platform", plat, "--profile", profile]
    # Build selection: explicit id wins; otherwise --latest (also the
    # implied default when neither is given).
    if build_id:
        cmd.extend(["--id", str(build_id)])
    else:
        cmd.append("--latest")
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
        reversibility_intent="irreversible",
    )

    if _is_missing_exe(raw):
        logger.warning("eas-cli / npx not installed — eas_submit skipped", platform=plat)
        return {
            "ok": True,
            "skipped": True,
            "platform": plat,
            "profile": profile,
            "build_id": build_id,
            "submission_id": None,
            "submission_url": None,
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
    submission_id, submission_url = _parse_submission_artifacts(stdout + "\n" + stderr)

    if timed_out:
        return {
            "ok": False,
            "skipped": False,
            "platform": plat,
            "profile": profile,
            "build_id": build_id,
            "submission_id": submission_id,
            "submission_url": submission_url,
            "exit": exit_code,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": f"eas submit timed out after {timeout_s}s",
        }

    ok = exit_code == 0
    return {
        "ok": ok,
        "skipped": False,
        "platform": plat,
        "profile": profile,
        "build_id": build_id,
        "submission_id": submission_id,
        "submission_url": submission_url,
        "exit": exit_code,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "duration_s": float(raw.get("duration_s", 0.0)),
        "error": None if ok else f"eas submit exited {exit_code}",
    }
