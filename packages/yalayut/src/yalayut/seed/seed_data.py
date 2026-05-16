"""Seed data for the yalayut catalog — owners, sources, disabled imports,
and the loader that installs the 20 hand-authored seed manifests.

Run by migration.run_full_migration() and idempotent.
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite

_MANIFEST_DIR = Path(__file__).parent / "manifests"

# (owner_id, trust_score, allowed_artifact_types, notes)
SEED_OWNERS = [
    ("anthropics", 0.95, '["skill","api","mcp"]', "first-party Anthropic"),
    ("obra", 0.9, '["skill"]', "superpowers author, vetted"),
    ("wshobson", 0.85, '["skill"]', "agent-config library, vetted subset"),
    ("cookiecutter", 0.85, '["skill"]', "cookiecutter org templates"),
    ("audreyfeldroy", 0.85, '["skill"]', "cookiecutter-pypackage author"),
    ("drivendataorg", 0.85, '["skill"]', "cookiecutter-data-science"),
    ("matlab", 0.8, '["skill"]', "matlab-official skills"),
]

# (source_id, source_type, endpoint, discovery_mode, trusted, min_interval_s)
SEED_SOURCES = [
    ("github:anthropics/skills@/skills", "github_path",
     "https://github.com/anthropics/skills", "cron", 1, 86400),
    ("github:obra/superpowers@/skills", "github_path",
     "https://github.com/obra/superpowers", "cron", 1, 86400),
    ("github:wshobson/agents@/plugins", "github_path",
     "https://github.com/wshobson/agents", "cron", 1, 86400),
    ("github:matlab/skills@/skills", "github_path",
     "https://github.com/matlab/skills", "cron", 1, 86400),
]

# (source, artifact_name, reason)
SEED_DISABLED_IMPORTS = [
    ("github:obra/superpowers@/skills", "using-superpowers",
     "boilerplate; refers to skill subsystem we replace"),
    ("github:obra/superpowers@/skills", "using-git-worktrees",
     "conflicts with KutAI .claude/worktrees/agent-<id> convention"),
    ("github:punkpeye/awesome-mcp-servers", "mcp-browser-use",
     "duplicates_vecihi"),
    ("github:public-apis/public-apis", "cat-facts", "low-signal joke API"),
]


async def seed_owners(db: aiosqlite.Connection) -> int:
    """Insert seed owners. Idempotent."""
    n = 0
    for owner_id, score, types, notes in SEED_OWNERS:
        cur = await db.execute(
            "INSERT OR IGNORE INTO yalayut_owners "
            "(owner_id, trust_score, allowed_artifact_types, source_count, "
            " rolling_success_rate, notes) VALUES (?, ?, ?, 0, NULL, ?)",
            (owner_id, score, types, notes),
        )
        n += cur.rowcount or 0
    await db.commit()
    return n


async def seed_sources(db: aiosqlite.Connection) -> int:
    """Insert seed sources. Idempotent."""
    n = 0
    for sid, stype, endpoint, mode, trusted, interval in SEED_SOURCES:
        cur = await db.execute(
            "INSERT OR IGNORE INTO yalayut_sources "
            "(source_id, source_type, endpoint, discovery_mode, trusted, "
            " trust_score, min_interval_s) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, stype, endpoint, mode, trusted, 0.9, interval),
        )
        n += cur.rowcount or 0
    await db.commit()
    return n


async def seed_disabled_imports(db: aiosqlite.Connection) -> int:
    """Insert known-reject imports. Idempotent."""
    n = 0
    for source, name, reason in SEED_DISABLED_IMPORTS:
        cur = await db.execute(
            "INSERT OR IGNORE INTO yalayut_disabled_imports "
            "(source, artifact_name, reason, added_at) "
            "VALUES (?, ?, ?, strftime('%Y-%m-%d %H:%M:%S','now'))",
            (source, name, reason),
        )
        n += cur.rowcount or 0
    await db.commit()
    return n


def load_seed_manifests() -> list[tuple[str, str]]:
    """Return [(filename, yaml_text)] for every seed manifest on disk."""
    out: list[tuple[str, str]] = []
    for p in sorted(_MANIFEST_DIR.glob("*.yaml")):
        out.append((p.name, p.read_text(encoding="utf-8")))
    return out
