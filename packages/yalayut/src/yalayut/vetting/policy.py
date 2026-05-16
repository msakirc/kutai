"""DB-backed policy allowlists for the auto-checks.

No static YAML — yalayut_policy is the single source of truth, seeded by
seed_policy(). KutAI proposes additions via propose_policy() (rows in
yalayut_policy_proposals); founder approves via Telegram (Phase 3).
"""
from __future__ import annotations

import json
import re

import aiosqlite

# Seed allowlist: shell binaries known-safe as first token of a command.
_SEED_SHELL = [
    "npx", "npm", "pip", "uvx", "git", "cookiecutter", "node", "python",
    "uv", "pytest", "yarn", "pnpm",
]

# Seed domain allowlist for network_scope check (api artifacts only).
_SEED_DOMAINS = [
    "api.github.com", "raw.githubusercontent.com", "api.coingecko.com",
    "pypi.org", "registry.npmjs.org",
]

# Seed prompt-injection regexes (case-insensitive). Conservative starter set.
_SEED_INJECTION = [
    r"ignore (all |previous |above )?instructions",
    r"disregard (the |all )?(system|previous) prompt",
    r"you are now (a |an )?(developer|admin|root|dan)\b",
    r"reveal (your |the )?(system )?prompt",
    r"</?(system|assistant|user)>",
    r"exfiltrat",
]


async def seed_policy(db: aiosqlite.Connection) -> None:
    """Populate yalayut_policy with baseline allowlists. Idempotent — uses
    INSERT OR IGNORE keyed on UNIQUE(check_name, key)."""
    rows: list[tuple[str, str, str]] = []
    for b in _SEED_SHELL:
        rows.append(("shell_allowlist", b, "allow"))
    for d in _SEED_DOMAINS:
        rows.append(("domain_allowlist", d, "allow"))
    for i, pat in enumerate(_SEED_INJECTION):
        rows.append(("injection_regex", f"seed_{i}", pat))
    for check_name, key, value in rows:
        await db.execute(
            "INSERT OR IGNORE INTO yalayut_policy "
            "(check_name, key, value, added_by, added_at) "
            "VALUES (?, ?, ?, 'seed', strftime('%Y-%m-%d %H:%M:%S','now'))",
            (check_name, key, value),
        )
    await db.commit()


async def get_allowlist(
    db: aiosqlite.Connection, check_name: str
) -> dict[str, str]:
    """Return {key: value} for one check (e.g. shell_allowlist)."""
    cur = await db.execute(
        "SELECT key, value FROM yalayut_policy WHERE check_name = ?",
        (check_name,),
    )
    return {r["key"]: r["value"] for r in await cur.fetchall()}


async def get_injection_regexes(
    db: aiosqlite.Connection,
) -> list[re.Pattern]:
    """Return compiled injection regexes from policy."""
    cur = await db.execute(
        "SELECT value FROM yalayut_policy WHERE check_name = 'injection_regex'"
    )
    out: list[re.Pattern] = []
    for r in await cur.fetchall():
        try:
            out.append(re.compile(r["value"], re.IGNORECASE))
        except re.error:
            continue
    return out


async def propose_policy(
    db: aiosqlite.Connection,
    check_name: str,
    key: str,
    proposed_value: str,
    evidence: dict | None = None,
) -> int:
    """Record a policy-addition proposal for founder review. Returns row id."""
    cur = await db.execute(
        "INSERT INTO yalayut_policy_proposals "
        "(check_name, key, proposed_value, evidence_json, state, proposed_at) "
        "VALUES (?, ?, ?, ?, 'pending', strftime('%Y-%m-%d %H:%M:%S','now'))",
        (check_name, key, proposed_value, json.dumps(evidence or {})),
    )
    await db.commit()
    return cur.lastrowid
