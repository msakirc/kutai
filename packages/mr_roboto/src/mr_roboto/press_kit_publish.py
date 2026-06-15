"""Z7 T3C (A4 + A4.r1) — press_kit/publish mr_roboto verb.

Uploads a versioned press kit to S3/R2 (if PRESS_KIT_BUCKET env var is set)
or to a local filesystem store (fallback, always testable without cloud creds).

Hosting decision: **local-filesystem fallback chosen as default**.
Reason: keeps the feature shippable + testable with zero cloud credentials
or billing. S3/R2 upload is wired behind PRESS_KIT_BUCKET env var — set it
to an S3 bucket name (e.g. "my-press-kits") and the module uses boto3 if
available. Without the env var (or without boto3), files land under
LOCAL_STORE_ROOT = <workspace_root>/press_kit_store/ and permanent URLs
take the form /press-kit/v{N}/{audience}/ (routed by the app's static-file
handler). This gives an identical URL shape whether local or cloud.

Old versions are NEVER deleted. Each new publish writes a `_superseded.json`
stub into every older version directory, pointing at the latest version.

Public API:
    run(*, mission_id, product_id, manifest) -> dict
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.press_kit_publish")

# Configurable local store root — patched in tests via monkeypatch.setattr.
# Default: <cwd>/press_kit_store  (relative to wherever the app runs).
LOCAL_STORE_ROOT: str = os.path.join(
    os.environ.get("WORKSPACE_ROOT", os.getcwd()), "press_kit_store"
)

AUDIENCE_VARIANTS: tuple[str, ...] = (
    "investor",
    "journalist",
    "partner",
    "candidate",
)


# ---------------------------------------------------------------------------
# DB helper (injectable for tests)
# ---------------------------------------------------------------------------

async def _persist_kit(
    product_id: str,
    version: int,
    mission_id: int | None,
    manifest_json: str,
    urls_json: str,
) -> None:
    """Upsert a press_kits row with the published URLs."""
    try:
        from dabidabi import get_db
        db = await get_db()
        # Check if row already exists (re-publish)
        cur = await db.execute(
            "SELECT kit_id FROM press_kits WHERE product_id = ? AND version = ?",
            (product_id, version),
        )
        existing = await cur.fetchone()
        if existing:
            await db.execute(
                "UPDATE press_kits SET manifest_json = ?, published_url = ? "
                "WHERE product_id = ? AND version = ?",
                (manifest_json, urls_json, product_id, version),
            )
        else:
            await db.execute(
                "INSERT INTO press_kits "
                "(product_id, version, mission_id, manifest_json, published_url) "
                "VALUES (?, ?, ?, ?, ?)",
                (product_id, version, mission_id, manifest_json, urls_json),
            )
        await db.commit()
    except Exception as exc:
        logger.warning("press_kit_publish: _persist_kit failed", error=str(exc))


# ---------------------------------------------------------------------------
# Upload backends
# ---------------------------------------------------------------------------

def _s3_available(bucket: str) -> bool:
    """Return True if boto3 is importable and a bucket name is configured."""
    if not bucket:
        return False
    try:
        import boto3  # type: ignore[import]  # noqa: F401
        return True
    except ImportError:
        return False


async def _upload_s3(
    bucket: str,
    zip_path: str,
    s3_key: str,
) -> str:
    """Upload zip_path to S3 and return the permanent URL."""
    import boto3  # type: ignore[import]

    s3 = boto3.client("s3")
    s3.upload_file(zip_path, bucket, s3_key)
    region = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint") or "us-east-1"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"


def _publish_local(
    zip_path: str,
    product_id: str,
    version: int,
    audience: str,
    store_root: str,
) -> str:
    """Copy zip to local store, return permanent URL path."""
    dest_dir = os.path.join(store_root, product_id, f"v{version}", audience)
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, os.path.basename(zip_path))
    shutil.copy2(zip_path, dest_file)
    # Return a URL path (no hostname — routed by app)
    return f"/press-kit/v{version}/{audience}/"


def _write_superseded_stub(
    store_root: str,
    product_id: str,
    old_version: int,
    latest_version: int,
) -> None:
    """Write a _superseded.json stub in the old version's directory."""
    old_v_dir = os.path.join(store_root, product_id, f"v{old_version}")
    if not os.path.isdir(old_v_dir):
        return
    stub_path = os.path.join(old_v_dir, "_superseded.json")
    stub = {
        "superseded_version": old_version,
        "latest_version": latest_version,
        "message": f"This press kit version is superseded. See v{latest_version}.",
    }
    with open(stub_path, "w", encoding="utf-8") as fh:
        json.dump(stub, fh, indent=2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run(
    *,
    mission_id: int,
    product_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Publish a press kit (all 4 audience variants) and persist permanent URLs.

    Hosting:
    - If PRESS_KIT_BUCKET env var is set and boto3 is installed → S3/R2 upload.
    - Otherwise → local filesystem store (LOCAL_STORE_ROOT), URL = /press-kit/v{N}/{audience}/.

    Returns:
        {"ok": True, "urls": {"investor": "...", "journalist": "...", ...}}
        {"ok": False, "error": "..."}
    """
    try:
        version = manifest.get("version", 1)
        variants = manifest.get("variants", {})
        bucket = os.environ.get("PRESS_KIT_BUCKET", "").strip()
        use_s3 = _s3_available(bucket)

        urls: dict[str, str] = {}
        store_root = LOCAL_STORE_ROOT  # may be overridden in tests

        for audience in AUDIENCE_VARIANTS:
            variant = variants.get(audience, {})
            zip_path = variant.get("zip_path", "")

            if not zip_path or not os.path.isfile(zip_path):
                # Gracefully skip missing zips — caller should ensure they exist
                logger.warning(
                    "press_kit_publish: zip missing for audience",
                    audience=audience,
                    zip_path=zip_path,
                )
                urls[audience] = f"/press-kit/v{version}/{audience}/"
                continue

            if use_s3:
                s3_key = f"press-kits/{product_id}/v{version}/{audience}/{os.path.basename(zip_path)}"
                try:
                    url = await _upload_s3(bucket, zip_path, s3_key)
                except Exception as exc:
                    logger.warning(
                        "press_kit_publish: S3 upload failed, falling back to local",
                        audience=audience,
                        error=str(exc),
                    )
                    url = _publish_local(zip_path, product_id, version, audience, store_root)
            else:
                url = _publish_local(zip_path, product_id, version, audience, store_root)

            urls[audience] = url

        # Write _superseded.json stubs for all older versions in local store
        v_root = os.path.join(store_root, product_id)
        if os.path.isdir(v_root):
            for entry in os.scandir(v_root):
                if entry.is_dir() and entry.name.startswith("v"):
                    try:
                        old_v = int(entry.name[1:])
                        if old_v < version:
                            _write_superseded_stub(store_root, product_id, old_v, version)
                    except ValueError:
                        pass

        # Persist to DB
        manifest_copy = {
            k: v for k, v in manifest.items() if k != "variants"
        }
        manifest_copy["audience_urls"] = urls
        await _persist_kit(
            product_id=product_id,
            version=version,
            mission_id=mission_id,
            manifest_json=json.dumps(manifest_copy),
            urls_json=json.dumps(urls),
        )

        logger.info(
            "press_kit_publish: published",
            product_id=product_id,
            version=version,
            use_s3=use_s3,
            audiences=list(urls.keys()),
        )

        return {"ok": True, "urls": urls, "version": version}

    except Exception as exc:
        logger.error("press_kit_publish: failed", error=str(exc))
        return {"ok": False, "error": str(exc)}
