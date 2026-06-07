"""Z7 T3B — A3 Demo pipeline tests.

Covers:
  1. demo/storyboard verb: LLM call → storyboard JSON with ordered scenes.
  2. demo/record verb: playwright --video on → raw scene MP4s (subprocess mocked).
  3. demo/edit verb: ffmpeg concat+trim → cuts/{30s,60s,3min}.mp4 (subprocess mocked).
  4. demo/caption verb: WebVTT from storyboard narrator_text (no speech-to-text).
  5. demo/accessibility_pass verb (A3.r1): alt text, audio-desc track, keyboard-nav variant.
  6. demo_artifact_check posthook: cut files exist, duration ±10%, captions present.
  7. demo_accessibility_check posthook (A3.r1): accessibility manifest complete.
  8. Graceful degradation when a stage's input is missing.
"""
from __future__ import annotations

import json
import os
import pytest

# ---------------------------------------------------------------------------
# Helper: build a minimal storyboard dict
# ---------------------------------------------------------------------------

SAMPLE_STORYBOARD = {
    "title": "Product Demo",
    "total_target_seconds": 90,
    "scenes": [
        {
            "id": "scene_1",
            "title": "Introduction",
            "target_seconds": 20,
            "viewport_state": "home_page",
            "narrator_text": "Welcome to our product. Here you will see the main features.",
        },
        {
            "id": "scene_2",
            "title": "Core Feature",
            "target_seconds": 40,
            "viewport_state": "feature_page",
            "narrator_text": "This is the core feature that saves you hours every day.",
        },
        {
            "id": "scene_3",
            "title": "Wrap-up",
            "target_seconds": 30,
            "viewport_state": "home_page",
            "narrator_text": "Sign up today and get started in minutes.",
        },
    ],
}


# ===========================================================================
# 1. demo/storyboard verb
# ===========================================================================

class TestDemoStoryboard:
    """Tests for the demo/storyboard mr_roboto verb.

    SP4b: the verb is now a MECHANICAL sink (no LLM). The LLM draft is the
    `13.demo_storyboard_draft` workflow step (agent:reviewer) whose output is
    materialized to ``<mission_workspace>/demo/storyboard_raw.json``; this verb
    reads that raw file, normalizes scenes, and writes ``demo/storyboard.json``.
    (Canonical sink contract also covered by
    ``packages/mr_roboto/tests/test_demo_storyboard_sink.py``.)
    """

    @pytest.mark.asyncio
    async def test_storyboard_reads_raw_normalizes_and_writes(self, tmp_path):
        """Sink reads the producer's raw file, normalizes scenes, writes JSON."""
        from mr_roboto.demo_storyboard import run as storyboard_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        (demo_dir / "storyboard_raw.json").write_text(
            json.dumps(SAMPLE_STORYBOARD), encoding="utf-8"
        )

        result = await storyboard_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            raw_filename="demo/storyboard_raw.json",
        )

        assert result["ok"] is True
        storyboard = result["storyboard"]
        assert "scenes" in storyboard
        assert len(storyboard["scenes"]) >= 1
        for scene in storyboard["scenes"]:
            assert "id" in scene
            assert "target_seconds" in scene
            assert "viewport_state" in scene
            assert "narrator_text" in scene

    @pytest.mark.asyncio
    async def test_storyboard_writes_json_file(self, tmp_path):
        """Sink persists the storyboard to workspace_path/demo/storyboard.json."""
        from mr_roboto.demo_storyboard import run as storyboard_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        (demo_dir / "storyboard_raw.json").write_text(
            json.dumps(SAMPLE_STORYBOARD), encoding="utf-8"
        )

        result = await storyboard_run(
            mission_id=42,
            workspace_path=str(tmp_path),
            raw_filename="demo/storyboard_raw.json",
        )

        assert result["ok"] is True
        storyboard_path = tmp_path / "demo" / "storyboard.json"
        assert storyboard_path.exists()
        loaded = json.loads(storyboard_path.read_text())
        assert "scenes" in loaded

    @pytest.mark.asyncio
    async def test_storyboard_missing_raw_file_returns_error(self, tmp_path):
        """Missing raw file → graceful ok=False, not an exception."""
        from mr_roboto.demo_storyboard import run as storyboard_run

        result = await storyboard_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            raw_filename="demo/storyboard_raw.json",
        )
        assert result["ok"] is False
        assert "raw" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_storyboard_makes_no_llm_call(self):
        """Sink must carry no LLM symbols (founder rule: mechanical = no LLM)."""
        import mr_roboto.demo_storyboard as mod

        assert not hasattr(mod, "_enqueue_storyboard_llm")
        assert not hasattr(mod, "_STORYBOARD_SYSTEM")


# ===========================================================================
# 2. demo/record verb
# ===========================================================================

class TestDemoRecord:
    """Tests for demo/record mr_roboto verb (playwright mocked)."""

    @pytest.mark.asyncio
    async def test_record_creates_scene_mp4s(self, tmp_path, monkeypatch):
        """Record verb calls playwright per scene and produces raw MP4s."""
        from mr_roboto.demo_record import run as record_run

        # Write storyboard to workspace
        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))

        # Mock subprocess to simulate playwright producing .webm files
        async def _mock_subprocess(cmd, timeout=300):
            # Simulate playwright: create a .webm in the --output directory
            if "playwright" in " ".join(cmd):
                # Find the --output argument to discover where to write
                output_dir = None
                for i, arg in enumerate(cmd):
                    if arg == "--output" and i + 1 < len(cmd):
                        output_dir = cmd[i + 1]
                        break
                if output_dir:
                    import os as _os
                    _os.makedirs(output_dir, exist_ok=True)
                    fake_webm = _os.path.join(output_dir, "recording.webm")
                    with open(fake_webm, "wb") as f:
                        f.write(b"\x00" * 1024)
                return (0, "ok", "")
            return (0, "", "")

        monkeypatch.setattr("mr_roboto.demo_record._run_subprocess", _mock_subprocess)

        # Also mock ffprobe for duration
        def _mock_duration(path):
            return 20.0

        monkeypatch.setattr("mr_roboto.demo_record._video_duration_seconds", _mock_duration)

        result = await record_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
        )

        assert result["ok"] is True
        assert "scene_recordings" in result
        assert len(result["scene_recordings"]) == len(SAMPLE_STORYBOARD["scenes"])

    @pytest.mark.asyncio
    async def test_record_missing_storyboard(self, tmp_path):
        """Record verb returns ok=False when storyboard.json is absent."""
        from mr_roboto.demo_record import run as record_run

        result = await record_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(tmp_path / "demo" / "storyboard.json"),
        )

        assert result["ok"] is False
        assert "storyboard" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_record_playwright_failure_graceful(self, tmp_path, monkeypatch):
        """When playwright fails for a scene, record verb returns ok=False with error."""
        from mr_roboto.demo_record import run as record_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))

        async def _mock_subprocess(cmd, timeout=300):
            if "playwright" in " ".join(cmd):
                return (1, "", "playwright: command not found")
            return (0, "", "")

        monkeypatch.setattr("mr_roboto.demo_record._run_subprocess", _mock_subprocess)

        result = await record_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
        )

        assert result["ok"] is False


# ===========================================================================
# 3. demo/edit verb
# ===========================================================================

class TestDemoEdit:
    """Tests for demo/edit mr_roboto verb (ffmpeg mocked)."""

    def _make_raw_recordings(self, demo_dir, scenes):
        """Create fake raw .webm files for each scene."""
        raw_dir = demo_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        recordings = []
        for scene in scenes:
            path = raw_dir / f"{scene['id']}.webm"
            path.write_bytes(b"\x00" * 2048)
            recordings.append({
                "scene_id": scene["id"],
                "path": str(path),
                "target_seconds": scene["target_seconds"],
            })
        return recordings

    @pytest.mark.asyncio
    async def test_edit_produces_three_cuts(self, tmp_path, monkeypatch):
        """Edit verb produces cuts/30s.mp4, cuts/60s.mp4, cuts/3min.mp4."""
        from mr_roboto.demo_edit import run as edit_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))
        recordings = self._make_raw_recordings(demo_dir, SAMPLE_STORYBOARD["scenes"])

        call_log = []

        async def _mock_subprocess(cmd, timeout=300):
            # Simulate ffmpeg: create the output file
            for i, arg in enumerate(cmd):
                if arg == "-y" and i + 1 < len(cmd):
                    continue
                if arg not in ("-y",) and arg.endswith(".mp4") and not arg.startswith("-"):
                    os.makedirs(os.path.dirname(arg), exist_ok=True)
                    with open(arg, "wb") as f:
                        f.write(b"\x00" * 1024)
            call_log.append(cmd)
            return (0, "", "")

        monkeypatch.setattr("mr_roboto.demo_edit._run_subprocess", _mock_subprocess)

        def _mock_duration(path):
            return 28.0

        monkeypatch.setattr("mr_roboto.demo_edit._video_duration_seconds", _mock_duration)

        result = await edit_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
            scene_recordings=recordings,
        )

        assert result["ok"] is True
        cuts = result["cuts"]
        assert set(cuts.keys()) == {"30s", "60s", "3min"}
        for cut_name, cut_path in cuts.items():
            assert os.path.exists(cut_path), f"cut {cut_name} not found at {cut_path}"

    @pytest.mark.asyncio
    async def test_edit_missing_scene_recordings(self, tmp_path, monkeypatch):
        """Edit verb returns ok=False when scene_recordings list is empty."""
        from mr_roboto.demo_edit import run as edit_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))

        result = await edit_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
            scene_recordings=[],
        )

        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_edit_ffmpeg_failure_returns_error(self, tmp_path, monkeypatch):
        """Edit verb returns ok=False when ffmpeg fails."""
        from mr_roboto.demo_edit import run as edit_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))
        recordings = self._make_raw_recordings(demo_dir, SAMPLE_STORYBOARD["scenes"])

        async def _mock_subprocess(cmd, timeout=300):
            return (1, "", "ffmpeg: No such file or directory")

        monkeypatch.setattr("mr_roboto.demo_edit._run_subprocess", _mock_subprocess)

        result = await edit_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
            scene_recordings=recordings,
        )

        assert result["ok"] is False


# ===========================================================================
# 4. demo/caption verb
# ===========================================================================

class TestDemoCaption:
    """Tests for demo/caption mr_roboto verb (script-driven WebVTT)."""

    @pytest.mark.asyncio
    async def test_caption_produces_webvtt(self, tmp_path):
        """Caption verb writes a valid WebVTT file from narrator_text."""
        from mr_roboto.demo_caption import run as caption_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))

        result = await caption_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
        )

        assert result["ok"] is True
        vtt_path = result["vtt_path"]
        assert vtt_path.endswith(".vtt")
        assert os.path.exists(vtt_path)

        content = open(vtt_path).read()
        assert content.startswith("WEBVTT")
        # Each scene's narrator text should appear
        for scene in SAMPLE_STORYBOARD["scenes"]:
            assert scene["narrator_text"] in content

    @pytest.mark.asyncio
    async def test_caption_timestamps_match_scene_durations(self, tmp_path):
        """Caption timestamps are computed from cumulative target_seconds."""
        from mr_roboto.demo_caption import run as caption_run, _seconds_to_vtt_time

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))

        result = await caption_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
        )

        assert result["ok"] is True
        content = open(result["vtt_path"]).read()
        # Scene 1 starts at 00:00:00.000, scene 2 starts at 00:00:20.000
        assert "00:00:00.000" in content
        assert "00:00:20.000" in content

    @pytest.mark.asyncio
    async def test_caption_missing_storyboard(self, tmp_path):
        """Caption verb returns ok=False when storyboard is absent."""
        from mr_roboto.demo_caption import run as caption_run

        result = await caption_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(tmp_path / "demo" / "storyboard.json"),
        )

        assert result["ok"] is False

    def test_seconds_to_vtt_time_format(self):
        """_seconds_to_vtt_time converts float seconds to HH:MM:SS.mmm."""
        from mr_roboto.demo_caption import _seconds_to_vtt_time

        assert _seconds_to_vtt_time(0.0) == "00:00:00.000"
        assert _seconds_to_vtt_time(65.5) == "00:01:05.500"
        assert _seconds_to_vtt_time(3661.123) == "01:01:01.123"


# ===========================================================================
# 5. demo/accessibility_pass verb (A3.r1)
# ===========================================================================

class TestDemoAccessibilityPass:
    """Tests for demo/accessibility_pass mr_roboto verb."""

    @pytest.mark.asyncio
    async def test_accessibility_pass_produces_manifest(self, tmp_path):
        """Accessibility pass writes a manifest JSON with alt text and audio-desc."""
        from mr_roboto.demo_accessibility_pass import run as a11y_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))

        # Create fake thumbnail files
        thumbnails_dir = demo_dir / "thumbnails"
        thumbnails_dir.mkdir()
        for scene in SAMPLE_STORYBOARD["scenes"]:
            thumb = thumbnails_dir / f"{scene['id']}.png"
            thumb.write_bytes(b"\x89PNG fake")

        result = await a11y_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
        )

        assert result["ok"] is True
        manifest_path = result["manifest_path"]
        assert os.path.exists(manifest_path)
        manifest = json.loads(open(manifest_path).read())

        assert "alt_texts" in manifest
        assert "audio_description_track" in manifest
        assert "keyboard_nav_variant" in manifest

    @pytest.mark.asyncio
    async def test_accessibility_pass_alt_text_per_scene(self, tmp_path):
        """Each scene gets an alt_text entry in the manifest."""
        from mr_roboto.demo_accessibility_pass import run as a11y_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))

        result = await a11y_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
        )

        assert result["ok"] is True
        manifest = json.loads(open(result["manifest_path"]).read())
        alt_texts = manifest["alt_texts"]
        assert len(alt_texts) == len(SAMPLE_STORYBOARD["scenes"])
        for entry in alt_texts:
            assert "scene_id" in entry
            assert "alt_text" in entry
            assert len(entry["alt_text"]) > 0

    @pytest.mark.asyncio
    async def test_accessibility_pass_audio_description_visual_scenes(self, tmp_path):
        """Visual-only scenes get an audio description entry."""
        storyboard_with_visual = {
            "title": "Visual Demo",
            "total_target_seconds": 60,
            "scenes": [
                {
                    "id": "scene_1",
                    "title": "Silent UI",
                    "target_seconds": 30,
                    "viewport_state": "dashboard",
                    "narrator_text": "",  # visual-only: no narrator
                    "visual_only": True,
                },
                {
                    "id": "scene_2",
                    "title": "Narrated",
                    "target_seconds": 30,
                    "viewport_state": "feature",
                    "narrator_text": "Here is the feature you've been waiting for.",
                },
            ],
        }
        from mr_roboto.demo_accessibility_pass import run as a11y_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(storyboard_with_visual))

        result = await a11y_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
        )

        assert result["ok"] is True
        manifest = json.loads(open(result["manifest_path"]).read())
        audio_desc = manifest["audio_description_track"]
        # scene_1 is visual_only → should have an audio description entry
        scene_ids_with_desc = {e["scene_id"] for e in audio_desc}
        assert "scene_1" in scene_ids_with_desc

    @pytest.mark.asyncio
    async def test_accessibility_pass_keyboard_nav_variant_present(self, tmp_path):
        """Manifest includes a keyboard_nav_variant section with scene steps."""
        from mr_roboto.demo_accessibility_pass import run as a11y_run

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        storyboard_path = demo_dir / "storyboard.json"
        storyboard_path.write_text(json.dumps(SAMPLE_STORYBOARD))

        result = await a11y_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(storyboard_path),
        )

        assert result["ok"] is True
        manifest = json.loads(open(result["manifest_path"]).read())
        knv = manifest["keyboard_nav_variant"]
        assert isinstance(knv, list)
        assert len(knv) > 0
        for step in knv:
            assert "scene_id" in step
            assert "step_description" in step

    @pytest.mark.asyncio
    async def test_accessibility_pass_missing_storyboard(self, tmp_path):
        """Accessibility pass returns ok=False when storyboard missing."""
        from mr_roboto.demo_accessibility_pass import run as a11y_run

        result = await a11y_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            storyboard_path=str(tmp_path / "demo" / "storyboard.json"),
        )

        assert result["ok"] is False


# ===========================================================================
# 6. demo_artifact_check posthook
# ===========================================================================

class TestDemoArtifactCheckPosthook:
    """Tests for demo_artifact_check posthook handler."""

    def _make_cut_files(self, demo_dir, *, write_captions=True):
        """Create fake cut MP4s and optional .vtt in demo_dir."""
        cuts_dir = demo_dir / "cuts"
        cuts_dir.mkdir(parents=True, exist_ok=True)
        cut_paths = {}
        for name, target_s in [("30s", 30), ("60s", 60), ("3min", 180)]:
            path = cuts_dir / f"{name}.mp4"
            path.write_bytes(b"\x00" * 2048)
            cut_paths[name] = str(path)
        if write_captions:
            vtt = demo_dir / "demo.vtt"
            vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:20.000\nHello world")
        return cut_paths

    @pytest.mark.asyncio
    async def test_artifact_check_passes_when_all_present(self, tmp_path, monkeypatch):
        """demo_artifact_check passes when all cuts + .vtt exist and durations ok."""
        from general_beckman.posthook_handlers.demo_artifact_check import handle

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        cuts = self._make_cut_files(demo_dir)

        # Return duration close to each file's target (within ±10%)
        _target_by_name = {"30s": 29.0, "60s": 58.0, "3min": 175.0}

        def _mock_duration(path):
            for label, dur in _target_by_name.items():
                if label in path:
                    return dur
            return 29.0

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.demo_artifact_check._video_duration_seconds",
            _mock_duration,
        )

        task = {
            "id": 1,
            "mission_id": 1,
            "context": {
                "workspace_path": str(tmp_path),
                "demo_cuts": cuts,
                "demo_vtt_path": str(demo_dir / "demo.vtt"),
                "demo_cut_targets": {"30s": 30, "60s": 60, "3min": 180},
            },
        }
        result = await handle(task, {})
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_artifact_check_fails_missing_cut(self, tmp_path, monkeypatch):
        """demo_artifact_check fails when a required cut is missing."""
        from general_beckman.posthook_handlers.demo_artifact_check import handle

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        cuts = self._make_cut_files(demo_dir)
        # Remove the 60s cut
        os.remove(cuts["60s"])

        def _mock_duration(path):
            return 29.0

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.demo_artifact_check._video_duration_seconds",
            _mock_duration,
        )

        task = {
            "id": 1,
            "mission_id": 1,
            "context": {
                "workspace_path": str(tmp_path),
                "demo_cuts": cuts,
                "demo_vtt_path": str(demo_dir / "demo.vtt"),
                "demo_cut_targets": {"30s": 30, "60s": 60, "3min": 180},
            },
        }
        result = await handle(task, {})
        assert result["status"] == "failed"
        assert "60s" in result.get("error", "") or "60s" in str(result.get("missing_cuts", []))

    @pytest.mark.asyncio
    async def test_artifact_check_fails_duration_out_of_tolerance(self, tmp_path, monkeypatch):
        """demo_artifact_check fails when duration is >10% off from target."""
        from general_beckman.posthook_handlers.demo_artifact_check import handle

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        cuts = self._make_cut_files(demo_dir)

        def _mock_duration(path):
            # 30s cut returns 15s — 50% off, fails ±10%
            if "30s" in path:
                return 15.0
            # other cuts return valid durations
            if "60s" in path:
                return 58.0
            if "3min" in path:
                return 175.0
            return 29.0

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.demo_artifact_check._video_duration_seconds",
            _mock_duration,
        )

        task = {
            "id": 1,
            "mission_id": 1,
            "context": {
                "workspace_path": str(tmp_path),
                "demo_cuts": cuts,
                "demo_vtt_path": str(demo_dir / "demo.vtt"),
                "demo_cut_targets": {"30s": 30, "60s": 60, "3min": 180},
            },
        }
        result = await handle(task, {})
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_artifact_check_fails_missing_captions(self, tmp_path, monkeypatch):
        """demo_artifact_check fails when .vtt captions file is missing."""
        from general_beckman.posthook_handlers.demo_artifact_check import handle

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        cuts = self._make_cut_files(demo_dir, write_captions=False)

        def _mock_duration(path):
            return 29.0

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.demo_artifact_check._video_duration_seconds",
            _mock_duration,
        )

        task = {
            "id": 1,
            "mission_id": 1,
            "context": {
                "workspace_path": str(tmp_path),
                "demo_cuts": cuts,
                "demo_vtt_path": str(demo_dir / "demo.vtt"),
                "demo_cut_targets": {"30s": 30, "60s": 60, "3min": 180},
            },
        }
        result = await handle(task, {})
        assert result["status"] == "failed"
        assert "caption" in result.get("error", "").lower() or "vtt" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_artifact_check_skip_when_no_context(self, tmp_path):
        """demo_artifact_check skips gracefully when context has no demo_cuts."""
        from general_beckman.posthook_handlers.demo_artifact_check import handle

        task = {"id": 1, "mission_id": 1, "context": {}}
        result = await handle(task, {})
        assert result["status"] == "skip"


# ===========================================================================
# 7. demo_accessibility_check posthook (A3.r1)
# ===========================================================================

class TestDemoAccessibilityCheckPosthook:
    """Tests for demo_accessibility_check posthook handler."""

    def _make_complete_manifest(self, demo_dir):
        """Write a complete accessibility manifest."""
        manifest = {
            "alt_texts": [
                {"scene_id": "scene_1", "alt_text": "Screenshot of the home page showing the main navigation."},
                {"scene_id": "scene_2", "alt_text": "The core feature dashboard with analytics charts."},
            ],
            "audio_description_track": [
                {"scene_id": "scene_1", "description": "The cursor moves to the signup button in the top right corner."},
            ],
            "keyboard_nav_variant": [
                {"scene_id": "scene_1", "step_description": "Tab to navigation, press Enter on Features"},
                {"scene_id": "scene_2", "step_description": "Tab to main content, Ctrl+Enter to activate"},
            ],
        }
        manifest_path = demo_dir / "accessibility_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return str(manifest_path)

    @pytest.mark.asyncio
    async def test_accessibility_check_passes_complete_manifest(self, tmp_path):
        """demo_accessibility_check passes when manifest has all required sections."""
        from general_beckman.posthook_handlers.demo_accessibility_check import handle

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        manifest_path = self._make_complete_manifest(demo_dir)

        task = {
            "id": 1,
            "mission_id": 1,
            "context": {
                "workspace_path": str(tmp_path),
                "demo_accessibility_manifest_path": manifest_path,
            },
        }
        result = await handle(task, {})
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_accessibility_check_fails_missing_manifest(self, tmp_path):
        """demo_accessibility_check fails when manifest file doesn't exist."""
        from general_beckman.posthook_handlers.demo_accessibility_check import handle

        task = {
            "id": 1,
            "mission_id": 1,
            "context": {
                "workspace_path": str(tmp_path),
                "demo_accessibility_manifest_path": str(tmp_path / "demo" / "accessibility_manifest.json"),
            },
        }
        result = await handle(task, {})
        assert result["status"] == "failed"
        assert "manifest" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_accessibility_check_fails_missing_alt_texts(self, tmp_path):
        """demo_accessibility_check fails when alt_texts is empty list."""
        from general_beckman.posthook_handlers.demo_accessibility_check import handle

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        manifest = {
            "alt_texts": [],  # empty!
            "audio_description_track": [],
            "keyboard_nav_variant": [{"scene_id": "s1", "step_description": "Tab to nav"}],
        }
        manifest_path = demo_dir / "accessibility_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        task = {
            "id": 1,
            "mission_id": 1,
            "context": {
                "workspace_path": str(tmp_path),
                "demo_accessibility_manifest_path": str(manifest_path),
            },
        }
        result = await handle(task, {})
        assert result["status"] == "failed"
        assert "alt_text" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_accessibility_check_fails_missing_keyboard_nav(self, tmp_path):
        """demo_accessibility_check fails when keyboard_nav_variant is absent."""
        from general_beckman.posthook_handlers.demo_accessibility_check import handle

        demo_dir = tmp_path / "demo"
        demo_dir.mkdir()
        manifest = {
            "alt_texts": [{"scene_id": "s1", "alt_text": "Home page"}],
            "audio_description_track": [],
            # keyboard_nav_variant missing!
        }
        manifest_path = demo_dir / "accessibility_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        task = {
            "id": 1,
            "mission_id": 1,
            "context": {
                "workspace_path": str(tmp_path),
                "demo_accessibility_manifest_path": str(manifest_path),
            },
        }
        result = await handle(task, {})
        assert result["status"] == "failed"
        assert "keyboard" in result.get("error", "").lower() or "nav" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_accessibility_check_skip_when_no_manifest_path(self, tmp_path):
        """demo_accessibility_check skips gracefully when context has no manifest path."""
        from general_beckman.posthook_handlers.demo_accessibility_check import handle

        task = {"id": 1, "mission_id": 1, "context": {}}
        result = await handle(task, {})
        assert result["status"] == "skip"


# ===========================================================================
# 8. Posthook registry — new kinds registered
# ===========================================================================

class TestDemoPosthookRegistry:
    """demo_artifact_check + demo_accessibility_check appear in POST_HOOK_REGISTRY."""

    def test_demo_artifact_check_registered(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        assert "demo_artifact_check" in POST_HOOK_REGISTRY

    def test_demo_accessibility_check_registered(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        assert "demo_accessibility_check" in POST_HOOK_REGISTRY

    def test_demo_artifact_check_is_blocker(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["demo_artifact_check"]
        assert spec.default_severity == "blocker"

    def test_demo_accessibility_check_is_blocker(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["demo_accessibility_check"]
        assert spec.default_severity == "blocker"


# ===========================================================================
# 9. Reversibility — new demo verbs classified
# ===========================================================================

class TestDemoVerbReversibility:
    """All demo/* verbs appear in VERB_REVERSIBILITY."""

    def test_demo_storyboard_full(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("demo/storyboard") == "full"

    def test_demo_record_full(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("demo/record") == "full"

    def test_demo_edit_full(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("demo/edit") == "full"

    def test_demo_caption_full(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("demo/caption") == "full"

    def test_demo_accessibility_pass_full(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("demo/accessibility_pass") == "full"
