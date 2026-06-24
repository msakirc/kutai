"""Generate a GitHub Actions mobile-CI workflow — Z5 T3b adapter.

Mechanical executor. No LLM. Writes ``.github/workflows/mobile.yml`` into
the mission workspace: the **free-first** mobile build path the founder
picked over EAS (see ``docs/i2p-evolution/05-build-mobile-track-v2.md``,
"Founder decisions (2026-05-16)").

What the generated workflow does
--------------------------------
- **iOS job** on a ``macos-latest`` runner: ``expo prebuild --platform
  ios`` materialises the native ``ios/`` Xcode project, then ``xcodebuild``
  (or Fastlane, when a ``Fastfile`` is present) compiles it. GitHub's free
  tier bills macOS minutes at a **10x multiplier**, so this job only runs
  when iOS is in the requested platform set.
- **Android job** on an ``ubuntu-latest`` runner: ``expo prebuild
  --platform android`` materialises ``android/``, then ``./gradlew
  assembleRelease`` produces the APK. Linux minutes are 1x — cheap.

Why a generator, not a static file
-----------------------------------
The workflow is keyed to the mission's ``bundle_id`` and chosen platform
set, and must land inside the *mission workspace* repo (the generated
project), not KutAI's own repo. A verb that templates + writes it keeps
the per-mission parameters honest.

Reversibility: ``full`` — writes a single local file under the workspace;
``git checkout`` / deleting the file fully reverses it. Nothing is pushed
or published.

Structured result
-----------------
``{ok, skipped, workflow_path, jobs:[...], platforms:[...], error}``.
``skipped`` is always False here (no external CLI to be missing) — it is
kept in the schema so callers can treat all Z5 mobile verbs uniformly.
"""

from __future__ import annotations

import os
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.gen_mobile_ci")

_ALLOWED_PLATFORMS = ("ios", "android")

# Relative path of the generated workflow inside the workspace.
WORKFLOW_REL_PATH = os.path.join(".github", "workflows", "mobile.yml")

_DEFAULT_NODE_VERSION = "20"


def _ios_job(bundle_id: str, scheme: str) -> str:
    """Return the YAML block for the macos-latest iOS build job."""
    return f"""  ios:
    name: iOS build (macOS runner)
    runs-on: macos-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "{_DEFAULT_NODE_VERSION}"
          cache: npm
      - name: Install dependencies
        run: npm ci
      - name: Cache CocoaPods
        uses: actions/cache@v4
        with:
          path: ios/Pods
          key: pods-${{{{ runner.os }}}}-${{{{ hashFiles('ios/Podfile.lock') }}}}
      - name: Expo prebuild (iOS)
        run: npx expo prebuild --platform ios --non-interactive
      - name: Build with Fastlane or xcodebuild
        env:
          BUNDLE_ID: "{bundle_id}"
          MATCH_PASSWORD: ${{{{ secrets.MATCH_PASSWORD }}}}
          APP_STORE_CONNECT_KEY: ${{{{ secrets.APP_STORE_CONNECT_KEY }}}}
        run: |
          if [ -f ios/fastlane/Fastfile ]; then
            cd ios && bundle exec fastlane build
          else
            xcodebuild -workspace ios/{scheme}.xcworkspace \\
              -scheme {scheme} \\
              -configuration Release \\
              -sdk iphoneos \\
              -derivedDataPath ios/build \\
              CODE_SIGNING_ALLOWED=NO
          fi
"""


def _android_job(bundle_id: str) -> str:
    """Return the YAML block for the ubuntu-latest Android build job."""
    return f"""  android:
    name: Android build (Linux runner)
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "{_DEFAULT_NODE_VERSION}"
          cache: npm
      - name: Setup JDK
        uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"
      - name: Install dependencies
        run: npm ci
      - name: Expo prebuild (Android)
        run: npx expo prebuild --platform android --non-interactive
      - name: Decode Android keystore
        env:
          ANDROID_KEYSTORE_BASE64: ${{{{ secrets.ANDROID_KEYSTORE_BASE64 }}}}
        run: |
          if [ -n "$ANDROID_KEYSTORE_BASE64" ]; then
            echo "$ANDROID_KEYSTORE_BASE64" | base64 --decode > android/app/release.keystore
          fi
      - name: Gradle assembleRelease
        env:
          APP_ID: "{bundle_id}"
        run: cd android && ./gradlew assembleRelease --no-daemon
"""


def _render_workflow(platforms: list[str], bundle_id: str, scheme: str) -> tuple[str, list[str]]:
    """Render the full workflow YAML text and the list of job ids."""
    jobs: list[str] = []
    blocks: list[str] = []
    if "ios" in platforms:
        jobs.append("ios")
        blocks.append(_ios_job(bundle_id, scheme))
    if "android" in platforms:
        jobs.append("android")
        blocks.append(_android_job(bundle_id))

    header = (
        "# Generated by mr_roboto.gen_mobile_ci (Z5 T3b) — free-first mobile CI.\n"
        "name: Mobile CI\n\n"
        "on:\n"
        "  push:\n"
        "    branches: [main]\n"
        "  pull_request:\n"
        "  workflow_dispatch:\n\n"
        "jobs:\n"
    )
    return header + "\n".join(blocks), jobs


async def gen_mobile_ci(
    mission_id: int | None,
    workspace_path: str | None = None,
    platforms: list[str] | None = None,
    bundle_id: str = "com.example.app",
    scheme: str | None = None,
) -> dict[str, Any]:
    """Generate ``.github/workflows/mobile.yml`` into the mission workspace.

    Parameters
    ----------
    mission_id:
        Used to resolve the workspace when ``workspace_path`` is absent.
    workspace_path:
        Explicit workspace root (the Expo project). The workflow lands at
        ``<workspace>/.github/workflows/mobile.yml``.
    platforms:
        Subset of ``["ios", "android"]``. Defaults to both. An iOS entry
        emits the macOS-runner job; android emits the Linux job.
    bundle_id:
        iOS bundle id / Android application id, threaded into env vars in
        the generated jobs.
    scheme:
        Xcode scheme name for ``xcodebuild``. Defaults to a sanitised
        last segment of ``bundle_id`` when not given.

    Returns
    -------
    dict with keys ``ok, skipped, workflow_path, jobs, platforms, error``.
    """
    plats_raw = platforms if platforms is not None else list(_ALLOWED_PLATFORMS)
    plats = [str(p).strip().lower() for p in plats_raw if str(p).strip()]
    bad = [p for p in plats if p not in _ALLOWED_PLATFORMS]
    if bad:
        return {
            "ok": False,
            "skipped": False,
            "workflow_path": None,
            "jobs": [],
            "platforms": plats,
            "error": (
                f"unsupported mobile platform(s) {bad!r}; "
                f"allowed: {list(_ALLOWED_PLATFORMS)}"
            ),
        }
    if not plats:
        return {
            "ok": False,
            "skipped": False,
            "workflow_path": None,
            "jobs": [],
            "platforms": [],
            "error": "no platforms requested — expected ios and/or android",
        }

    # Resolve workspace.
    ws = workspace_path
    if ws is None:
        if mission_id is None:
            return {
                "ok": False,
                "skipped": False,
                "workflow_path": None,
                "jobs": [],
                "platforms": plats,
                "error": "no mission_id and no workspace_path",
            }
        from src.tools.workspace import get_mission_workspace

        ws = get_mission_workspace(mission_id)

    scheme_name = scheme or (bundle_id.rsplit(".", 1)[-1] or "App")
    # Xcode schemes cannot contain spaces / dots — sanitise defensively.
    scheme_name = "".join(c for c in scheme_name if c.isalnum() or c in ("-", "_")) or "App"

    text, jobs = _render_workflow(plats, str(bundle_id), scheme_name)

    out_path = os.path.join(ws, WORKFLOW_REL_PATH)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    except OSError as exc:
        return {
            "ok": False,
            "skipped": False,
            "workflow_path": None,
            "jobs": jobs,
            "platforms": plats,
            "error": f"failed to write workflow: {exc}",
        }

    logger.info(
        "gen_mobile_ci wrote workflow",
        workflow_path=out_path,
        jobs=jobs,
        platforms=plats,
    )
    return {
        "ok": True,
        "skipped": False,
        "workflow_path": out_path,
        "jobs": jobs,
        "platforms": plats,
        "error": None,
    }
