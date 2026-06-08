"""Z7 T3C (A4 + A4.r1) — press_kit/assemble mr_roboto verb.

Assembles a versioned press kit with FOUR audience variants per run:
  - investor  : financial-leaning (traction, metrics, market size, team)
  - journalist: news-leaning (what's new, why it matters, quotes, stats)
  - partner   : integration-leaning (API, tech stack, joint opportunity)
  - candidate : culture-leaning (mission, team, growth, benefits)

Each variant gets its own zip under:
  <workspace_path>/press_kit/v{N}/{audience}/press_kit_v{N}_{audience}.zip

Sources gathered:
  - one_pager.md   — pre-drafted per audience (passed in one_pagers dict)
  - founder_bio.md — from `founder_bio` param
  - fact_sheet.md  — from `fact_sheet_md` param
  - quotes.md      — approved quotes from DB + passed `quotes` param
  - mentions.md    — `past_mentions` param
  - logo.*         — copied from `logo_path` if provided
  - screenshots/   — copied from `screenshot_paths` if provided

After assembly, emits a founder_action (kind='generic') requesting sign-off
before publish. Publish is a separate verb (press_kit/publish) which uploads
and sets published_url.

Public API (SP4b Plan3 — CPS):
    assemble_from_drafts(*, mission_id, product_id, version, workspace_path,
        spec_text, one_pagers, logo_path="", screenshot_paths=(), founder_bio="",
        fact_sheet_md="", quotes=(), past_mentions=()) -> dict
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime
from typing import Any, Sequence

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.press_kit_assemble")

AUDIENCE_VARIANTS: tuple[str, ...] = (
    "investor",
    "journalist",
    "partner",
    "candidate",
)


# ---------------------------------------------------------------------------
# Deterministic one-pager fallback (no LLM)
# ---------------------------------------------------------------------------

def audience_stub(audience: str, spec_text: str) -> str:
    """Deterministic one-pager fallback when the LLM produced nothing."""
    return f"[Draft one-pager for {audience} — see spec]\n\n{(spec_text or '')[:200]}"


async def _get_latest_version(product_id: str) -> int:
    """Return the highest version number already stored for product_id, or 0."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT MAX(version) FROM press_kits WHERE product_id = ?",
            (product_id,),
        )
        row = await cur.fetchone()
        if row and row[0] is not None:
            return int(row[0])
    except Exception as exc:
        logger.warning(
            "press_kit_assemble: _get_latest_version failed", error=str(exc)
        )
    return 0


async def _emit_founder_action(
    *,
    mission_id: int,
    product_id: str,
    version: int,
    audience_urls: dict[str, str],
) -> Any:
    """Emit a founder_action requesting sign-off before publish."""
    try:
        from src.founder_actions import create as create_founder_action

        title = f"Press kit v{version} assembled for {product_id} — approve to publish"
        why = (
            f"KutAI assembled press kit v{version} with 4 audience variants "
            f"(investor / journalist / partner / candidate) for product '{product_id}'. "
            "Review each zip and approve to trigger press_kit/publish."
        )
        instructions = [
            f"Review each variant zip: {', '.join(audience_urls.keys())}.",
            "Confirm the one-pager, fact sheet, and quotes are accurate.",
            "Mark this action done to approve publish.",
            "Or edit the source files and re-run press_kit/assemble to regenerate.",
        ]
        return await create_founder_action(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=why,
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=False,
        )
    except Exception as exc:
        logger.warning(
            "press_kit_assemble: _emit_founder_action failed", error=str(exc)
        )
        return None


# ---------------------------------------------------------------------------
# Main entry point (CPS-ready — no await_inline)
# ---------------------------------------------------------------------------

async def assemble_from_drafts(
    *,
    mission_id: int,
    product_id: str,
    version: int,
    workspace_path: str,
    spec_text: str,
    one_pagers: dict,
    logo_path: str = "",
    screenshot_paths: Sequence[str] = (),
    founder_bio: str = "",
    fact_sheet_md: str = "",
    quotes: Sequence[str] = (),
    past_mentions: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the versioned kit from already-drafted one-pagers (one per audience).
    Body is the old run() loop with version as a param and no LLM call.

    Returns:
        {"ok": True, "manifest": {...}, "version": N}
        {"ok": False, "error": "..."}
    """
    try:
        # Root output directory for this kit version
        kit_root = os.path.join(
            workspace_path, "press_kit", f"v{version}"
        )
        os.makedirs(kit_root, exist_ok=True)

        # Compute spec hash for freshness tracking
        spec_hash = hashlib.sha256(spec_text.encode()).hexdigest()[:16]

        variants: dict[str, dict[str, Any]] = {}

        for audience in AUDIENCE_VARIANTS:
            aud_dir = os.path.join(kit_root, audience)
            os.makedirs(aud_dir, exist_ok=True)

            # Use pre-drafted one-pager or deterministic stub
            one_pager_text = one_pagers.get(audience) or audience_stub(audience, spec_text)

            # Write source files to the per-audience staging dir
            with open(os.path.join(aud_dir, "one_pager.md"), "w", encoding="utf-8") as fh:
                fh.write(one_pager_text)

            if founder_bio:
                with open(
                    os.path.join(aud_dir, "founder_bio.md"), "w", encoding="utf-8"
                ) as fh:
                    fh.write(founder_bio)

            if fact_sheet_md:
                with open(
                    os.path.join(aud_dir, "fact_sheet.md"), "w", encoding="utf-8"
                ) as fh:
                    fh.write(fact_sheet_md)

            # Different audience sections: investor/journalist get quotes;
            # partner gets tech spec hint; candidate gets culture note.
            section_extra = _audience_extra_section(audience, quotes, past_mentions)
            if section_extra:
                with open(
                    os.path.join(aud_dir, "extras.md"), "w", encoding="utf-8"
                ) as fh:
                    fh.write(section_extra)

            # Logo
            if logo_path and os.path.isfile(logo_path):
                ext = os.path.splitext(logo_path)[1]
                shutil.copy2(logo_path, os.path.join(aud_dir, f"logo{ext}"))

            # Screenshots (only investor / journalist / partner variants)
            if audience in ("investor", "journalist", "partner") and screenshot_paths:
                screens_dir = os.path.join(aud_dir, "screenshots")
                os.makedirs(screens_dir, exist_ok=True)
                for sp in screenshot_paths:
                    if os.path.isfile(sp):
                        shutil.copy2(sp, os.path.join(screens_dir, os.path.basename(sp)))

            # Build zip
            zip_name = f"press_kit_v{version}_{audience}.zip"
            zip_path = os.path.join(kit_root, zip_name)
            _build_zip(aud_dir, zip_path)

            variants[audience] = {
                "zip_path": zip_path,
                "staging_dir": aud_dir,
            }

        manifest: dict[str, Any] = {
            "product_id": product_id,
            "version": version,
            "mission_id": mission_id,
            "spec_hash": spec_hash,
            "variants": variants,
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Emit founder_action for pre-publish approval
        audience_labels = {a: f"v{version}/{a}/" for a in AUDIENCE_VARIANTS}
        await _emit_founder_action(
            mission_id=mission_id,
            product_id=product_id,
            version=version,
            audience_urls=audience_labels,
        )

        logger.info(
            "press_kit_assemble: assembled",
            product_id=product_id,
            version=version,
            audiences=list(AUDIENCE_VARIANTS),
        )

        return {"ok": True, "manifest": manifest, "version": version}

    except Exception as exc:
        logger.error("press_kit_assemble: failed", error=str(exc))
        return {"ok": False, "error": str(exc)}


async def run(
    *,
    mission_id: int,
    product_id: str,
    spec_text: str,
    workspace_path: str,
    logo_path: str = "",
    screenshot_paths: Sequence[str] = (),
    founder_bio: str = "",
    fact_sheet_md: str = "",
    quotes: Sequence[str] = (),
    past_mentions: Sequence[str] = (),
) -> dict[str, Any]:
    """Backward-compat thin wrapper: computes version then delegates (all stubs).
    NOT on the live CPS path — use enqueue_press_kit instead."""
    version = (await _get_latest_version(product_id)) + 1
    return await assemble_from_drafts(
        mission_id=mission_id,
        product_id=product_id,
        version=version,
        workspace_path=workspace_path,
        spec_text=spec_text,
        one_pagers={},
        logo_path=logo_path,
        screenshot_paths=screenshot_paths,
        founder_bio=founder_bio,
        fact_sheet_md=fact_sheet_md,
        quotes=quotes,
        past_mentions=past_mentions,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _audience_extra_section(
    audience: str,
    quotes: Sequence[str],
    past_mentions: Sequence[str],
) -> str:
    """Return audience-specific extra markdown section."""
    lines: list[str] = []

    if audience == "investor":
        if past_mentions:
            lines.append("## Press mentions")
            for m in past_mentions:
                lines.append(f"- {m}")
    elif audience == "journalist":
        if quotes:
            lines.append("## Quotes")
            for q in quotes:
                lines.append(f"> {q}")
        if past_mentions:
            lines.append("\n## Prior coverage")
            for m in past_mentions:
                lines.append(f"- {m}")
    elif audience == "partner":
        lines.append("## Integration opportunity")
        lines.append("Contact us to discuss a joint integration or co-marketing arrangement.")
    elif audience == "candidate":
        if quotes:
            lines.append("## What our team says")
            for q in quotes[:3]:
                lines.append(f"> {q}")

    return "\n".join(lines)


def _build_zip(source_dir: str, zip_path: str) -> None:
    """Create a zip of all files under source_dir, paths relative to source_dir."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(source_dir):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, source_dir)
                zf.write(full, rel)
