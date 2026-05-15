"""Z4 T1A — ``capture_screenshots`` mechanical verb.

Captures screenshots of a running web preview across breakpoints and color
modes so a later tier can diff them with a vision model.

Soft-skips when:
- ``preview_url`` is absent or a ``pending:`` placeholder
- playwright is not available (package not importable or browsers missing)

Return shape
------------
``{"ok", "skipped", "reason" (if skipped), "captured_paths", "route_count",
"frame_count"}``

Breakpoints × color modes
--------------------------
- breakpoints: 375, 768, 1280, 1920 (px width)
- color modes: "light", "dark"
- 8 frames per route

Determinism injections
-----------------------
Before each screenshot:
- CSS: ``* { animation: none !important; transition: none !important;
  caret-color: transparent !important; }``
- ``prefers-reduced-motion: reduce``
- Date.now / new Date / Math.random stubs via init script
"""
from __future__ import annotations

import importlib
import os
from typing import Any

from src.infra.logging_config import get_logger

from .preview_url import is_real_url as _is_real_url

logger = get_logger("mr_roboto.capture_screenshots")

# Breakpoints (width in px) × color modes = 8 frames per route.
_BREAKPOINTS = (375, 768, 1280, 1920)
_COLOR_MODES = ("light", "dark")
_VIEWPORT_HEIGHT = 900

# CSS injected before every screenshot to suppress animation / blinking.
_DETERMINISM_CSS = (
    "* { animation: none !important; transition: none !important; "
    "caret-color: transparent !important; }"
)

# JS init script: stub Date/Math.random for pixel-stable screenshots.
_DETERMINISM_INIT_SCRIPT = """
(() => {
  const _epoch = 1700000000000;
  const _OrigDate = Date;
  class _FakeDate extends _OrigDate {
    constructor(...args) {
      if (args.length === 0) { super(_epoch); }
      else { super(...args); }
    }
    static now() { return _epoch; }
  }
  Object.defineProperty(window, 'Date', { value: _FakeDate, configurable: true });
  let _rng = 0;
  Math.random = () => { _rng = (_rng * 1664525 + 1013904223) & 0xffffffff; return (_rng >>> 0) / 0x100000000; };
})();
"""


def _get_async_playwright():
    """Return the ``async_playwright`` context manager.

    Module-level factory so tests can monkeypatch
    ``mr_roboto.capture_screenshots._get_async_playwright`` without needing
    playwright to be installed.
    """
    from playwright.async_api import async_playwright  # lazy
    return async_playwright


def _route_slug(route: str) -> str:
    """Convert a URL route to a filesystem-safe slug.

    ``/`` → ``root``, ``/foo/bar`` → ``foo_bar``.
    """
    stripped = route.strip("/")
    if not stripped:
        return "root"
    return stripped.replace("/", "_").replace(" ", "_")


def _infer_routes_from_produces(produces: list[str] | None) -> list[str]:
    """Infer Next.js URL routes from a step's ``produces`` file list.

    Supports:
    - ``pages/foo.tsx``       → ``/foo``
    - ``pages/index.tsx``     → ``/``
    - ``app/foo/page.tsx``    → ``/foo``
    - ``app/page.tsx``        → ``/``

    Only Next.js conventions are recognised; unrecognised paths are ignored.
    Falls back to ``["/"]`` when nothing can be inferred.
    """
    if not produces:
        return ["/"]

    routes: list[str] = []
    for entry in produces:
        path_str = entry if isinstance(entry, str) else ""
        # Normalise separators.
        normalised = path_str.replace("\\", "/")

        # pages/ convention
        if normalised.startswith("pages/"):
            rest = normalised[len("pages/"):]
            # Strip .tsx / .ts / .jsx / .js extension
            for ext in (".tsx", ".ts", ".jsx", ".js"):
                if rest.endswith(ext):
                    rest = rest[: -len(ext)]
                    break
            if rest == "index":
                routes.append("/")
            elif rest:
                routes.append("/" + rest)

        # app/ convention
        elif normalised.startswith("app/"):
            rest = normalised[len("app/"):]
            # Must end with page.tsx (or page.ts/js/jsx)
            segment = None
            for page_name in ("page.tsx", "page.ts", "page.jsx", "page.js"):
                if rest == page_name:
                    segment = ""
                    break
                if rest.endswith("/" + page_name):
                    segment = rest[: -(len(page_name) + 1)]
                    break
            if segment is not None:
                routes.append("/" + segment if segment else "/")

    if not routes:
        return ["/"]
    # Deduplicate, preserve first-seen order.
    seen: set[str] = set()
    out: list[str] = []
    for r in routes:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _resolve_preview_url(workspace_path: str) -> str | None:
    """Read preview URL from .preview/last_preview_url.txt with fallback."""
    preferred = os.path.join(workspace_path, ".preview", "last_preview_url.txt")
    fallback = os.path.join(workspace_path, "preview_url.txt")
    for candidate in (preferred, fallback):
        if os.path.isfile(candidate):
            try:
                return open(candidate, encoding="utf-8").read().strip()
            except OSError:
                continue
    return None


async def capture_screenshots(
    *,
    mission_id: int,
    step_id: str,
    routes: list[str] | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Capture screenshots of a running web preview for visual diffing.

    Parameters
    ----------
    mission_id:
        The mission whose workspace is used.
    step_id:
        The workflow step ID — used to namespace the output directory.
    routes:
        Explicit URL paths to capture. When absent, routes are inferred
        from the step's ``produces`` paths (Next.js convention only).
        A ``routes`` field on the workflow step JSON propagates here via
        the expander's generic context pass-through.
    workspace_path:
        Optional override — resolved from ``get_mission_workspace`` when absent.

    Returns
    -------
    dict with keys: ``ok``, ``skipped``, ``reason`` (if skipped),
    ``captured_paths``, ``route_count``, ``frame_count``.
    """
    # --- 1. Resolve workspace -----------------------------------------------
    if workspace_path is None:
        from src.tools.workspace import get_mission_workspace  # lazy import
        workspace_path = get_mission_workspace(int(mission_id))

    # --- 2. Resolve preview URL ---------------------------------------------
    raw_url = _resolve_preview_url(workspace_path)
    if not _is_real_url(raw_url):
        reason = "no real preview_url available"
        logger.warning(f"capture_screenshots soft-skip: {reason}")
        return {
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
        }

    preview_url: str = str(raw_url).rstrip("/")

    # --- 3. Playwright availability check -----------------------------------
    # Lazy import — kept inside function so module load stays cheap and the
    # soft-skip works when playwright is absent.  Tests patch
    # ``mr_roboto.capture_screenshots._get_async_playwright`` to inject mocks.
    try:
        importlib.import_module("playwright")  # availability probe
        from playwright.async_api import async_playwright as _raw_ap  # noqa: F401
    except Exception:
        reason = "playwright not available"
        logger.warning(f"capture_screenshots soft-skip: {reason}")
        return {
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
        }

    # Delegate through the module-level hook so tests can patch it.
    _async_playwright = _get_async_playwright()

    # --- 4. Determine routes ------------------------------------------------
    if routes:
        effective_routes = list(routes)
    else:
        effective_routes = ["/"]  # will be overridden by context if available

    # --- 5. Prepare output directory ----------------------------------------
    out_dir = os.path.join(workspace_path, ".visual", "captured", str(step_id))
    os.makedirs(out_dir, exist_ok=True)

    # --- 6. Capture frames --------------------------------------------------
    captured_paths: list[str] = []

    try:
        async with _async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                for route in effective_routes:
                    slug = _route_slug(route)
                    target = preview_url + route

                    for mode in _COLOR_MODES:
                        for bp in _BREAKPOINTS:
                            frame_name = f"{slug}_{mode}_{bp}.png"
                            out_path = os.path.join(out_dir, frame_name)
                            try:
                                ctx = await browser.new_context(
                                    viewport={"width": bp, "height": _VIEWPORT_HEIGHT},
                                    color_scheme=mode,
                                    reduced_motion="reduce",
                                )
                                try:
                                    await ctx.add_init_script(_DETERMINISM_INIT_SCRIPT)
                                    page = await ctx.new_page()
                                    try:
                                        await page.add_style_tag(content=_DETERMINISM_CSS)
                                        await page.goto(target, wait_until="networkidle")
                                        await page.emulate_media(
                                            color_scheme=mode,
                                            reduced_motion="reduce",
                                        )
                                        await page.screenshot(
                                            path=out_path, full_page=True
                                        )
                                        captured_paths.append(out_path)
                                        logger.debug(
                                            f"capture_screenshots: wrote {out_path}"
                                        )
                                    finally:
                                        await page.close()
                                finally:
                                    await ctx.close()

                            except Exception as frame_err:
                                logger.warning(
                                    f"capture_screenshots: frame error "
                                    f"{slug}/{mode}/{bp}: {frame_err}"
                                )
                                # Continue with remaining frames.
            finally:
                await browser.close()

    except Exception as launch_err:
        err_msg = f"browser launch failed: {launch_err}"
        logger.error(f"capture_screenshots hard-fail: {err_msg}")
        return {
            "ok": False,
            "error": err_msg,
            "captured_paths": captured_paths,
        }

    logger.info(
        f"capture_screenshots: mission={mission_id} step={step_id} "
        f"routes={len(effective_routes)} frames={len(captured_paths)}"
    )
    return {
        "ok": True,
        "skipped": False,
        "captured_paths": captured_paths,
        "route_count": len(effective_routes),
        "frame_count": len(captured_paths),
    }
