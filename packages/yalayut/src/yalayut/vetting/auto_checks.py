"""Gate-zero auto-checks — run on every artifact regardless of source trust.

Each check returns the MAX tier the artifact may reach (0 best, 3 worst).
run_all() returns {check_name: max_tier}; tier_classifier then mins them.
Mapping table is the spec's Tier classifier section.
"""
from __future__ import annotations

import re
from pathlib import Path

import aiosqlite

from yalayut.contracts import Manifest
from yalayut.vetting.policy import get_allowlist, get_injection_regexes

# body-size caps (bytes)
_SKILL_BODY_CAP = 50 * 1024
_HINT_BODY_CAP = 5 * 1024

# Windows-incompat patterns. Catastrophic ones (rm -rf /) cap at T3.
_WIN_BLOCK_T3 = [re.compile(p) for p in [
    r"\brm\s+-rf\s+/", r"\bmkfs\b", r":\(\)\s*\{\s*:\|:&\s*\}",
]]
_WIN_BLOCK_T2 = [re.compile(p) for p in [
    r"\bchmod\s+\+x", r"\bsudo\b", r"\bapt-get\b", r"\bbrew\s+install\b",
    r"\byum\s+install\b", r"\bln\s+-s\b", r"\.sh\b",
]]

# first-token shell extractor: lines inside ``` fences or after $ / >
_CMD_LINE = re.compile(r"^[\s$>]*([A-Za-z0-9_./-]+)", re.MULTILINE)

# crude network-endpoint detector
_URL = re.compile(r"https?://[^\s)\"']+")


def _first_tokens(text: str) -> list[str]:
    """Best-effort first-token-of-command extraction from fenced code."""
    toks: list[str] = []
    in_fence = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and not s.startswith("$"):
            continue
        m = _CMD_LINE.match(s)
        if m:
            tok = m.group(1)
            if tok and not tok.startswith("#"):
                toks.append(tok)
    return toks


async def run_all(
    db: aiosqlite.Connection, manifest: Manifest, body_path: Path
) -> dict[str, int]:
    """Run all 9 gate-zero checks. Returns {check_name: max_tier}."""
    body = ""
    if body_path and body_path.exists():
        body = body_path.read_text(encoding="utf-8", errors="replace")

    shell_allow = await get_allowlist(db, "shell_allowlist")
    injection_regexes = await get_injection_regexes(db)
    is_hint = manifest.kind == "internal_hint"

    out: dict[str, int] = {}

    # 1. schema_valid — manifest has required fields
    out["schema_valid"] = (
        0 if (manifest.name and manifest.version and manifest.artifact_type)
        else 3
    )

    # 2. body_size_ok
    cap = _HINT_BODY_CAP if is_hint else _SKILL_BODY_CAP
    out["body_size_ok"] = 0 if len(body.encode("utf-8")) <= cap else 2

    # 3. shell_allowlist — first token of every command
    shell_tier = 0
    for tok in _first_tokens(body):
        verdict = shell_allow.get(tok)
        if verdict == "deny":
            shell_tier = max(shell_tier, 3)
        elif verdict != "allow":
            shell_tier = max(shell_tier, 2)
    out["shell_allowlist"] = shell_tier

    # 4. network_scope — URLs only allowed in api artifacts
    has_url = bool(_URL.search(body))
    out["network_scope"] = (
        1 if (has_url and manifest.artifact_type != "api") else 0
    )

    # 5. mcp_pinned — sha256 / digest present for mcp artifacts
    if manifest.artifact_type == "mcp":
        pinned = bool(manifest.mcp.get("sha256") or manifest.mcp.get("digest"))
        out["mcp_pinned"] = 0 if pinned else 2
    else:
        out["mcp_pinned"] = 0

    # 6. injection_scan
    hit = any(rx.search(body) for rx in injection_regexes)
    out["injection_scan"] = 3 if hit else 0

    # 7. license_present
    out["license_present"] = 0 if manifest.license else 2

    # 8. diff_size — Phase 1 only ever does first-fetch; first import is T0.
    #    Re-fetch diff sizing is meaningful only once a v2 of an artifact
    #    exists (Phase 1 imports v1 of everything) — index.store passes
    #    prior_body_len=None for first import, so this is always 0 here.
    out["diff_size"] = 0

    # 9. windows_compat
    win = 0
    if any(rx.search(body) for rx in _WIN_BLOCK_T3):
        win = 3
    elif any(rx.search(body) for rx in _WIN_BLOCK_T2):
        win = 2
    out["windows_compat"] = win

    return out
