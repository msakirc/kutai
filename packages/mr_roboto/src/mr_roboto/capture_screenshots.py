"""Z4 T1A — ``capture_screenshots`` mechanical verb.

Captures screenshots of a running web preview across breakpoints and color
modes so a later tier can diff them with a vision model.

Soft-skips when:
- ``preview_url`` is absent or a ``pending:`` placeholder
- playwright is not available (package not importable or browsers missing)

Return shape
------------
``{"ok", "skipped", "reason" (if skipped), "captured_paths", "route_count",
"frame_count", "capture_mode"}``

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

Z5 T4a — device capture mode
----------------------------
``capture_mode="device"`` captures *mobile-shaped* frames instead of desktop
breakpoints.  Three arms run in order, each appending to a per-arm detail map:

1. **Playwright device descriptors** (the v1 default for device mode):
   Playwright ships built-in presets (``p.devices["iPhone 14"]`` etc.) that
   carry viewport / deviceScaleFactor / userAgent / isMobile / hasTouch.  We
   render the **Expo Web export** preview against a small representative set
   (one iOS-shaped, one Android-shaped).  Runs headless Chromium on Windows.
2. **``adb exec-out screencap``** — when an Android device / emulator is
   attached (probed via ``adb devices``), capture a real Android screenshot.
   Soft-skips cleanly when ``adb`` is absent or no device is attached.
3. **``xcrun simctl io``** — real-device / simulator iOS capture.  Guarded:
   on a non-macOS host this soft-skips with a clear reason.  The code path is
   left intact so a future macOS runner activates it.

Frame filenames are device-namespaced: ``{route_slug}_{device_slug}_{mode}.png``
so device-mode frames never collide with viewport-mode frames and route to
device-mode baselines (resolution is by basename).
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from typing import Any

from src.infra.logging_config import get_logger

from .preview_url import is_real_url as _is_real_url

logger = get_logger("mr_roboto.capture_screenshots")

# Breakpoints (width in px) × color modes = 8 frames per route.
_BREAKPOINTS = (375, 768, 1280, 1920)
_COLOR_MODES = ("light", "dark")
_VIEWPORT_HEIGHT = 900

# --- Z5 T4a — device capture mode ------------------------------------------
#
# Representative device set for the Playwright-descriptor arm.  Kept small
# (one iOS-shaped, one Android-shaped) to bound cost — each device is one
# vision-diffable frame.  The names MUST match Playwright's built-in
# ``p.devices`` registry keys; the descriptor carries viewport,
# deviceScaleFactor, userAgent, isMobile and hasTouch.
_DEVICE_DESCRIPTORS = ("iPhone 14", "Pixel 7")


def _device_slug(device_name: str) -> str:
    """Convert a Playwright device name to a filesystem-safe slug.

    ``"iPhone 14"`` → ``iphone_14``; ``"Pixel 7"`` → ``pixel_7``.
    """
    return device_name.strip().lower().replace(" ", "_").replace("/", "_")

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
    produces: list[str] | None = None,
    workspace_path: str | None = None,
    components: list[dict] | None = None,
    capture_mode: str = "viewport",
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
    components:
        **T5B** — Optional list of component descriptors.  Each entry is a
        dict with ``{"name": str, "selector": str}``.  When provided, after
        each full-page screenshot the verb also crops the matching DOM element
        via ``page.locator(selector).screenshot()`` and saves it as
        ``{route_slug}_{component_name}_{mode}_{breakpoint}.png`` alongside
        the full-page frame.  When absent, behaviour is unchanged.
    capture_mode:
        Controls the screenshot backend.

        - ``"viewport"`` *(default)*: Playwright headless Chromium against the
          web preview, desktop breakpoints × color modes.  Byte-for-byte
          unchanged from Z4.
        - ``"device"`` *(Z5 T4a)*: mobile-shaped capture.  Runs three arms —
          Playwright device descriptors (Expo Web export), ``adb exec-out
          screencap`` (real Android), ``xcrun simctl io`` (iOS, macOS-only).
          See the module docstring for arm details.

    Returns
    -------
    dict with keys: ``ok``, ``skipped``, ``reason`` (if skipped),
    ``captured_paths``, ``route_count``, ``frame_count``, ``capture_mode``.
    For ``capture_mode="device"`` an additional ``device_detail`` map carries
    per-arm outcomes.
    """
    # Z5 T4a — device capture mode dispatches to a dedicated implementation.
    if capture_mode == "device":
        return await _capture_device(
            mission_id=mission_id,
            step_id=step_id,
            routes=routes,
            produces=produces,
            workspace_path=workspace_path,
        )
    if capture_mode != "viewport":
        # Unknown mode — soft-skip rather than hard-fail so callers can opt
        # into future backends without breaking.
        reason = f"unknown capture_mode {capture_mode!r}"
        logger.info("capture_screenshots soft-skip: %s", reason)
        return {
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
            "capture_mode": capture_mode,
        }
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
            "capture_mode": "viewport",
        }

    preview_url: str = str(raw_url).rstrip("/")

    # --- 3. Playwright availability check -----------------------------------
    # Lazy import — kept inside function so module load stays cheap and the
    # soft-skip works when playwright is absent.  Tests patch
    # ``mr_roboto.capture_screenshots._get_async_playwright`` to inject mocks.
    try:
        importlib.import_module("playwright")  # availability probe
    except Exception:
        reason = "playwright not available"
        logger.warning(f"capture_screenshots soft-skip: {reason}")
        return {
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
            "capture_mode": "viewport",
        }

    # Delegate through the module-level hook so tests can patch it.
    _async_playwright = _get_async_playwright()

    # --- 4. Determine routes ------------------------------------------------
    # Explicit `routes` (from the step's `routes` JSON field) wins; otherwise
    # infer Next.js routes from the step's `produces` paths.
    if routes:
        effective_routes = list(routes)
    else:
        effective_routes = _infer_routes_from_produces(produces)

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

                                        # T5B — optional per-component crops
                                        if components:
                                            for comp in components:
                                                comp_name = str(comp.get("name") or "")
                                                selector = str(comp.get("selector") or "")
                                                if not comp_name or not selector:
                                                    continue
                                                comp_frame = f"{slug}_{comp_name}_{mode}_{bp}.png"
                                                comp_path = os.path.join(out_dir, comp_frame)
                                                try:
                                                    locator = page.locator(selector)
                                                    await locator.screenshot(path=comp_path)
                                                    captured_paths.append(comp_path)
                                                    logger.debug(
                                                        f"capture_screenshots: component crop {comp_path}"
                                                    )
                                                except Exception as comp_err:
                                                    logger.warning(
                                                        f"capture_screenshots: component crop error "
                                                        f"{comp_name}/{slug}/{mode}/{bp}: {comp_err}"
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
            "capture_mode": "viewport",
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
        "capture_mode": "viewport",
    }


# ===========================================================================
# Z5 T4a — device capture mode
# ===========================================================================


async def _capture_device_playwright_arm(
    *,
    preview_url: str,
    effective_routes: list[str],
    out_dir: str,
) -> dict[str, Any]:
    """Arm 1 — Playwright built-in device descriptors against the Expo Web export.

    Runs headless Chromium on any host (including Windows).  For each route ×
    device in :data:`_DEVICE_DESCRIPTORS`, applies the device preset (viewport,
    deviceScaleFactor, userAgent, isMobile, hasTouch) and captures a frame
    namespaced ``{route_slug}_{device_slug}_{mode}.png``.

    Returns ``{"arm": "playwright_device", "ok", "skipped", "reason"?,
    "captured_paths"}``.
    """
    # Playwright availability probe (mirrors the viewport path).
    try:
        importlib.import_module("playwright")
    except Exception:
        reason = "playwright not available"
        logger.warning("capture_screenshots(device) playwright arm soft-skip: %s", reason)
        return {
            "arm": "playwright_device",
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
        }

    _async_playwright = _get_async_playwright()
    captured: list[str] = []
    # Expo Web export renders one color scheme; capture "light" by default to
    # keep the device-mode frame set small. The device descriptor governs the
    # mobile shape; mode stays a stable namespace token.
    mode = "light"

    try:
        async with _async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                for route in effective_routes:
                    slug = _route_slug(route)
                    target = preview_url + route
                    for device_name in _DEVICE_DESCRIPTORS:
                        dslug = _device_slug(device_name)
                        frame_name = f"{slug}_{dslug}_{mode}.png"
                        out_path = os.path.join(out_dir, frame_name)
                        # Resolve the device descriptor from pw.devices; skip
                        # gracefully if this Playwright build lacks the preset.
                        descriptor = None
                        try:
                            descriptor = pw.devices[device_name]
                        except Exception:
                            logger.warning(
                                "capture_screenshots(device): unknown device "
                                "descriptor %r — skipping",
                                device_name,
                            )
                            continue
                        try:
                            ctx = await browser.new_context(
                                **descriptor,
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
                                    captured.append(out_path)
                                    logger.debug(
                                        "capture_screenshots(device): wrote %s",
                                        out_path,
                                    )
                                finally:
                                    await page.close()
                            finally:
                                await ctx.close()
                        except Exception as frame_err:
                            logger.warning(
                                "capture_screenshots(device): frame error "
                                "%s/%s: %s",
                                slug, dslug, frame_err,
                            )
            finally:
                await browser.close()
    except Exception as launch_err:
        reason = f"browser launch failed: {launch_err}"
        logger.error("capture_screenshots(device) playwright arm: %s", reason)
        return {
            "arm": "playwright_device",
            "ok": False,
            "skipped": False,
            "reason": reason,
            "captured_paths": captured,
        }

    return {
        "arm": "playwright_device",
        "ok": True,
        "skipped": False,
        "captured_paths": captured,
    }


async def _capture_device_adb_arm(
    *,
    mission_id: int,
    workspace_path: str,
    out_dir: str,
) -> dict[str, Any]:
    """Arm 2 — ``adb exec-out screencap`` for an attached Android device.

    Probes ``adb devices`` first.  Soft-skips cleanly when ``adb`` is not on
    PATH or when no device / emulator is attached.  When a device is present,
    captures one PNG per attached serial, namespaced
    ``device_{serial_slug}_android.png``.

    Returns ``{"arm": "adb", "ok", "skipped", "reason"?, "captured_paths"}``.
    """
    from .run_cmd import run_cmd  # lazy — keeps module import cheap

    # --- Probe attached devices --------------------------------------------
    probe = await run_cmd(
        mission_id,
        ["adb", "devices"],
        workspace_path=workspace_path,
        timeout_s=15.0,
    )
    if not probe.get("ok"):
        # run_cmd reports executable-not-found via the `error` key.
        reason = probe.get("error") or "adb probe failed"
        if "not found" in str(reason).lower():
            reason = "adb not available"
        logger.info("capture_screenshots(device) adb arm soft-skip: %s", reason)
        return {
            "arm": "adb",
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
        }

    # Parse `adb devices` output. Format:
    #   List of devices attached
    #   emulator-5554\tdevice
    serials: list[str] = []
    for line in str(probe.get("stdout_tail") or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("list of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])

    if not serials:
        reason = "no Android device attached"
        logger.info("capture_screenshots(device) adb arm soft-skip: %s", reason)
        return {
            "arm": "adb",
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
        }

    # --- Capture one screenshot per attached device ------------------------
    captured: list[str] = []
    for serial in serials:
        sslug = serial.lower().replace(":", "_").replace(".", "_").replace("-", "_")
        frame_name = f"device_{sslug}_android.png"
        out_path = os.path.join(out_dir, frame_name)
        # `adb exec-out screencap -p` writes a raw PNG to stdout. run_cmd
        # tail-decodes stdout to text, which would corrupt binary — so we
        # use the device-side file route: screencap to /sdcard then pull.
        remote_tmp = "/sdcard/.kutay_visual_capture.png"
        shot = await run_cmd(
            mission_id,
            ["adb", "-s", serial, "shell", "screencap", "-p", remote_tmp],
            workspace_path=workspace_path,
            timeout_s=30.0,
        )
        if not shot.get("ok"):
            logger.warning(
                "capture_screenshots(device) adb screencap failed for %s: %s",
                serial, shot.get("error") or shot.get("stderr_tail"),
            )
            continue
        pull = await run_cmd(
            mission_id,
            ["adb", "-s", serial, "pull", remote_tmp, out_path],
            workspace_path=workspace_path,
            timeout_s=30.0,
        )
        # Best-effort cleanup of the on-device temp file.
        await run_cmd(
            mission_id,
            ["adb", "-s", serial, "shell", "rm", "-f", remote_tmp],
            workspace_path=workspace_path,
            timeout_s=15.0,
        )
        if pull.get("ok") and os.path.isfile(out_path):
            captured.append(out_path)
            logger.debug("capture_screenshots(device): adb wrote %s", out_path)
        else:
            logger.warning(
                "capture_screenshots(device) adb pull failed for %s: %s",
                serial, pull.get("error") or pull.get("stderr_tail"),
            )

    return {
        "arm": "adb",
        "ok": True,
        "skipped": False,
        "reason": None,
        "captured_paths": captured,
    }


async def _capture_device_simctl_arm(
    *,
    mission_id: int,
    workspace_path: str,
    out_dir: str,
) -> dict[str, Any]:
    """Arm 3 — ``xcrun simctl io`` for iOS Simulator capture.

    ``xcrun simctl`` is macOS-only.  On a non-macOS host (the KutAI host runs
    Windows) this soft-skips with a clear reason.  The full capture code path
    is left in place so a future macOS runner activates it automatically.

    Returns ``{"arm": "simctl", "ok", "skipped", "reason"?, "captured_paths"}``.
    """
    # Hard guard — xcrun simctl cannot run anywhere but macOS.
    if sys.platform != "darwin" or platform.system() != "Darwin":
        reason = "real-device iOS capture requires a macOS runner"
        logger.info("capture_screenshots(device) simctl arm soft-skip: %s", reason)
        return {
            "arm": "simctl",
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
        }

    # --- macOS path (dormant on Windows; activated on a macOS runner) ------
    from .run_cmd import run_cmd  # lazy

    captured: list[str] = []
    # Capture the currently-booted simulator. `booted` resolves to whatever
    # simulator the runner has launched.
    frame_name = "device_ios_simulator.png"
    out_path = os.path.join(out_dir, frame_name)
    shot = await run_cmd(
        mission_id,
        ["xcrun", "simctl", "io", "booted", "screenshot", out_path],
        workspace_path=workspace_path,
        timeout_s=30.0,
    )
    if not shot.get("ok"):
        reason = shot.get("error") or shot.get("stderr_tail") or "simctl screenshot failed"
        if "not found" in str(reason).lower():
            reason = "xcrun not available"
        logger.info("capture_screenshots(device) simctl arm soft-skip: %s", reason)
        return {
            "arm": "simctl",
            "ok": True,
            "skipped": True,
            "reason": str(reason),
            "captured_paths": [],
        }
    if os.path.isfile(out_path):
        captured.append(out_path)
        logger.debug("capture_screenshots(device): simctl wrote %s", out_path)

    return {
        "arm": "simctl",
        "ok": True,
        "skipped": False,
        "reason": None,
        "captured_paths": captured,
    }


async def _capture_device(
    *,
    mission_id: int,
    step_id: str,
    routes: list[str] | None,
    produces: list[str] | None,
    workspace_path: str | None,
) -> dict[str, Any]:
    """Z5 T4a — device capture mode.

    Runs three arms (Playwright device descriptors, adb, xcrun simctl), each
    independently soft-skipping.  Aggregates captured paths and returns the
    standard envelope plus ``capture_mode="device"`` and a ``device_detail``
    map of per-arm outcomes.

    The verb is ``ok`` whenever no arm hard-fails; ``skipped`` is True only
    when *every* arm produced zero frames (so the posthook does not gate on
    an empty capture set).
    """
    # --- Resolve workspace --------------------------------------------------
    if workspace_path is None:
        from src.tools.workspace import get_mission_workspace  # lazy
        workspace_path = get_mission_workspace(int(mission_id))

    # --- Resolve preview URL (shared with the Playwright arm) --------------
    raw_url = _resolve_preview_url(workspace_path)
    have_preview = _is_real_url(raw_url)
    preview_url = str(raw_url).rstrip("/") if have_preview else ""

    # --- Determine routes ---------------------------------------------------
    if routes:
        effective_routes = list(routes)
    else:
        effective_routes = _infer_routes_from_produces(produces)

    # --- Output directory ---------------------------------------------------
    out_dir = os.path.join(workspace_path, ".visual", "captured", str(step_id))
    os.makedirs(out_dir, exist_ok=True)

    arms: list[dict[str, Any]] = []

    # --- Arm 1: Playwright device descriptors ------------------------------
    if have_preview:
        arms.append(
            await _capture_device_playwright_arm(
                preview_url=preview_url,
                effective_routes=effective_routes,
                out_dir=out_dir,
            )
        )
    else:
        arms.append({
            "arm": "playwright_device",
            "ok": True,
            "skipped": True,
            "reason": "no real preview_url available",
            "captured_paths": [],
        })

    # --- Arm 2: adb (real Android) -----------------------------------------
    arms.append(
        await _capture_device_adb_arm(
            mission_id=mission_id,
            workspace_path=workspace_path,
            out_dir=out_dir,
        )
    )

    # --- Arm 3: xcrun simctl (iOS, macOS-only) -----------------------------
    arms.append(
        await _capture_device_simctl_arm(
            mission_id=mission_id,
            workspace_path=workspace_path,
            out_dir=out_dir,
        )
    )

    # --- Aggregate ----------------------------------------------------------
    captured_paths: list[str] = []
    for arm in arms:
        captured_paths.extend(arm.get("captured_paths") or [])

    device_detail = {
        arm["arm"]: {
            "ok": arm.get("ok", False),
            "skipped": arm.get("skipped", False),
            "reason": arm.get("reason"),
            "frame_count": len(arm.get("captured_paths") or []),
        }
        for arm in arms
    }

    hard_failed = [a for a in arms if not a.get("ok")]
    all_empty = not captured_paths

    if hard_failed:
        err_msg = "; ".join(
            f"{a['arm']}: {a.get('reason')}" for a in hard_failed
        )
        logger.error("capture_screenshots(device) hard-fail: %s", err_msg)
        return {
            "ok": False,
            "skipped": False,
            "error": err_msg,
            "captured_paths": captured_paths,
            "route_count": len(effective_routes),
            "frame_count": len(captured_paths),
            "capture_mode": "device",
            "device_detail": device_detail,
        }

    if all_empty:
        # Every arm soft-skipped — return a soft-skip envelope so the
        # downstream visual_review posthook does not gate on nothing.
        reasons = "; ".join(
            f"{a['arm']}: {a.get('reason')}"
            for a in arms
            if a.get("reason")
        )
        reason = reasons or "no device capture backend available"
        logger.info("capture_screenshots(device) soft-skip: %s", reason)
        return {
            "ok": True,
            "skipped": True,
            "reason": reason,
            "captured_paths": [],
            "route_count": len(effective_routes),
            "frame_count": 0,
            "capture_mode": "device",
            "device_detail": device_detail,
        }

    logger.info(
        "capture_screenshots(device): mission=%s step=%s routes=%s frames=%s",
        mission_id, step_id, len(effective_routes), len(captured_paths),
    )
    return {
        "ok": True,
        "skipped": False,
        "captured_paths": captured_paths,
        "route_count": len(effective_routes),
        "frame_count": len(captured_paths),
        "capture_mode": "device",
        "device_detail": device_detail,
    }
