"""Z6 T7D — thin helper for registering rendered artifacts.

Several mechanical executors (T4A ``legal_document_render``, future
Z6 deployments) want to insert / refresh a row in
``mission_artifacts_index`` without depending on the higher-level
workflow engine. This module is that helper.

Schema lives in :mod:`src.infra.db` (table
``mission_artifacts_index``); this function only inserts.
"""
from __future__ import annotations

import json
from typing import Iterable


async def register_artifact(
    mission_id: int,
    artifact_name: str,
    artifact_path: str,
    *,
    schema_version: str | None = "1",
    domain_keywords: Iterable[str] | None = None,
    founder_id: str = "default",
) -> None:
    """Insert-or-replace a row in ``mission_artifacts_index``.

    ``UNIQUE(mission_id, artifact_name)`` makes this idempotent for the
    common "executor re-runs and rewrites the same artifact" case.
    Failure modes (DB unavailable, schema drift) raise — callers wrap
    in try/except when they want best-effort behaviour.
    """
    from .db import get_db
    db = await get_db()
    keywords = list(domain_keywords or [])
    await db.execute(
        "INSERT OR REPLACE INTO mission_artifacts_index "
        "(mission_id, artifact_name, artifact_path, schema_version, "
        " domain_keywords_json, founder_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            int(mission_id),
            artifact_name,
            artifact_path,
            schema_version,
            json.dumps(keywords),
            founder_id,
        ),
    )
    await db.commit()


__all__ = ["register_artifact"]
