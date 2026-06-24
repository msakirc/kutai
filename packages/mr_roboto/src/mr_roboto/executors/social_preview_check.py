"""Social-preview check executor.

Reads the ``seo_implementation`` artifact for the mission, pulls a list of
URLs from it, runs :func:`mr_roboto.parse_og_tags.parse_og_tags` on each, and
aggregates pass/fail.

This is the mechanical replacement for what step 13.11 social_preview_test
used to fabricate. Honest scope: we verify the served HTML contains the
required Open Graph tags + the og:image is reachable. We do NOT replace
platform-specific debuggers (Facebook / Twitter / LinkedIn) which crawl
with their own UA and apply their own size/length rules. A pass here is a
necessary, not sufficient, condition for a real platform to render the
preview correctly.
"""

from __future__ import annotations

import json
from typing import Any

from yazbunu import get_logger

from mr_roboto.parse_og_tags import parse_og_tags

logger = get_logger("mr_roboto.social_preview_check")


# Field names we'll try, in order, when pulling URLs out of the
# seo_implementation artifact. The artifact's exact shape isn't fixed in
# the workflow JSON; the SEO step just emits a dict and downstream consumes
# it. We probe a few common keys before giving up.
_URL_FIELD_CANDIDATES = (
    "social_preview_urls",
    "preview_urls",
    "pages_optimized",
    "urls",
    "pages",
    "test_urls",
)


async def _load_artifact(mission_id: int, name: str) -> dict | None:
    """Read an artifact via the cache-first store, falling back to blackboard.

    Mirrors the pattern used in mr_roboto.clarify so the same divergence cases
    (cache not populated yet / store instance split) are handled.
    """
    try:
        from src.workflows.engine.hooks import get_artifact_store
        store = get_artifact_store()
        raw = await store.retrieve(mission_id, name)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
    except Exception as exc:
        logger.warning("artifact-store retrieve failed: %s", exc)

    try:
        from src.collaboration.blackboard import read_blackboard
        artifacts = await read_blackboard(int(mission_id), "artifacts")
        if isinstance(artifacts, dict):
            raw2 = artifacts.get(name)
            if isinstance(raw2, dict):
                return raw2
            if isinstance(raw2, str) and raw2.strip():
                try:
                    return json.loads(raw2)
                except json.JSONDecodeError:
                    return None
    except Exception as exc:
        logger.warning("blackboard read failed: %s", exc)
    return None


def _extract_urls(artifact: dict) -> list[str]:
    for field in _URL_FIELD_CANDIDATES:
        v = artifact.get(field)
        if isinstance(v, list):
            urls = [str(u).strip() for u in v if isinstance(u, str) and u.strip()]
            if urls:
                return urls
        if isinstance(v, str) and v.strip():
            return [v.strip()]
    # Last-ditch: scan any dict value for a url-shaped string
    for v in artifact.values():
        if isinstance(v, str) and v.startswith(("http://", "https://")):
            return [v]
    return []


async def run(task: dict[str, Any]) -> dict[str, Any]:
    mission_id = task.get("mission_id")
    if mission_id is None:
        return {"ok": False, "error": "no mission_id", "results": []}

    payload = task.get("payload") or {}
    artifact_name = payload.get("artifact") or "seo_implementation"

    artifact = await _load_artifact(int(mission_id), artifact_name)
    if not isinstance(artifact, dict) or not artifact:
        return {
            "ok": False,
            "error": f"artifact {artifact_name!r} missing or empty",
            "results": [],
        }

    urls = _extract_urls(artifact)
    # Allow override / extension via payload.urls
    extra = payload.get("urls") or []
    if isinstance(extra, list):
        urls.extend(str(u).strip() for u in extra if isinstance(u, str) and u.strip())

    if not urls:
        return {
            "ok": False,
            "error": (
                f"no URLs found in {artifact_name!r}; tried fields "
                f"{list(_URL_FIELD_CANDIDATES)}"
            ),
            "results": [],
        }

    results: list[dict] = []
    for url in urls:
        r = await parse_og_tags(
            url,
            timeout_s=float(payload.get("timeout_s", 15.0)),
            check_image=bool(payload.get("check_image", True)),
            required=payload.get("required"),
        )
        results.append(r)

    pass_count = sum(1 for r in results if r.get("ok"))
    all_ok = pass_count == len(results)
    return {
        "ok": all_ok,
        "platforms_tested": ["og_tags", "twitter_card"],
        "pass_fail": "pass" if all_ok else "fail",
        "urls_tested": urls,
        "passed": pass_count,
        "total": len(results),
        "results": results,
    }
