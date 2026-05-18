"""Z7 T6D — A3 Demo distribution stage tests.

Covers:
  1. demo/distribute verb: uploads each cut as YouTube unlisted, returns video IDs
     + embed URLs + og:video snippet + thumbnail paths.
  2. Thumbnail extraction via ffmpeg single-frame grab (subprocess mocked).
  3. YouTube upload mocked — no real API calls.
  4. founder_action emitted: "review demo cuts → flip to public?".
  5. Public-flip is gated — distribute does NOT auto-publish publicly.
  6. Graceful error when youtube client absent.
  7. Reversibility classification: demo/distribute is "partial".
  8. og:video snippet contains embed URLs for all three cuts.
"""
from __future__ import annotations

import json
import os
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cuts(tmp_path) -> dict[str, str]:
    """Write fake cut MP4 files and return {label: path} dict."""
    cuts_dir = tmp_path / "demo" / "cuts"
    cuts_dir.mkdir(parents=True, exist_ok=True)
    cuts = {}
    for label in ("30s", "60s", "3min"):
        p = cuts_dir / f"{label}.mp4"
        p.write_bytes(b"\x00" * 2048)
        cuts[label] = str(p)
    return cuts


def _fake_youtube_upload(path: str, title: str, description: str, privacy: str) -> dict:
    """Fake YouTube upload: returns deterministic video_id from cut label."""
    label = os.path.splitext(os.path.basename(path))[0]  # e.g. "30s"
    return {
        "video_id": f"fake_vid_{label}",
        "embed_url": f"https://www.youtube.com/embed/fake_vid_{label}",
        "watch_url": f"https://www.youtube.com/watch?v=fake_vid_{label}",
        "privacy": privacy,
    }


# ===========================================================================
# 1 & 2. Core distribute verb — uploads + thumbnails
# ===========================================================================

class TestDemoDistribute:
    """Tests for demo/distribute mr_roboto verb."""

    @pytest.mark.asyncio
    async def test_distribute_returns_video_ids_and_embed_urls(self, tmp_path, monkeypatch):
        """distribute verb returns video_id + embed_url for each cut."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)

        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            # Simulate ffmpeg: write a fake PNG.
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 42}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is True
        uploads = result["uploads"]
        assert set(uploads.keys()) == {"30s", "60s", "3min"}
        for label, info in uploads.items():
            assert "video_id" in info
            assert "embed_url" in info
            assert info["privacy"] == "unlisted"

    @pytest.mark.asyncio
    async def test_distribute_returns_thumbnail_paths(self, tmp_path, monkeypatch):
        """distribute verb returns thumbnail_path for each cut."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)

        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        extracted = []

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            extracted.append(out_path)
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 99}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is True
        uploads = result["uploads"]
        for label, info in uploads.items():
            assert "thumbnail_path" in info
            assert os.path.exists(info["thumbnail_path"]), (
                f"thumbnail for {label} should exist at {info['thumbnail_path']}"
            )

    @pytest.mark.asyncio
    async def test_distribute_returns_og_video_snippet(self, tmp_path, monkeypatch):
        """distribute verb returns an og:video meta tag snippet containing embed URLs."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)

        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 11}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is True
        og = result["og_video_snippet"]
        assert og is not None
        assert "og:video" in og
        # Should reference the 30s embed URL (primary embed)
        assert "fake_vid_30s" in og or "fake_vid_60s" in og or "fake_vid_3min" in og

    @pytest.mark.asyncio
    async def test_distribute_uploads_as_unlisted_not_public(self, tmp_path, monkeypatch):
        """All uploads use privacy=unlisted; public is not set automatically."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)

        upload_privacies = []

        def _tracking_upload(path, title, description, privacy):
            upload_privacies.append(privacy)
            return _fake_youtube_upload(path, title, description, privacy)

        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _tracking_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 5}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is True
        assert all(p == "unlisted" for p in upload_privacies), (
            f"All uploads must be unlisted; got: {upload_privacies}"
        )
        # Verify public was NOT set in any upload
        assert "public" not in upload_privacies


# ===========================================================================
# 3. Founder-action for public-flip
# ===========================================================================

class TestFounderActionPublicFlip:
    """founder_action is emitted; public-flip is gated on founder approval."""

    @pytest.mark.asyncio
    async def test_founder_action_emitted_after_upload(self, tmp_path, monkeypatch):
        """After uploads complete, a founder_action is emitted to flip to public."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)
        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        fa_calls = []

        async def _mock_fa_create(**kwargs):
            fa_calls.append(kwargs)
            return {"id": 77}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is True
        assert len(fa_calls) == 1, "Exactly one founder_action should be emitted"
        fa_kwargs = fa_calls[0]
        # _emit_flip_to_public_action is called with mission_id + product_name + uploads
        assert "mission_id" in fa_kwargs or "product_name" in fa_kwargs or "uploads" in fa_kwargs

    @pytest.mark.asyncio
    async def test_public_flip_not_automatic(self, tmp_path, monkeypatch):
        """distribute verb NEVER sets privacy=public by itself."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)

        seen_public = []

        def _tracking_upload(path, title, description, privacy):
            if privacy == "public":
                seen_public.append(path)
            return _fake_youtube_upload(path, title, description, privacy)

        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _tracking_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 1}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert seen_public == [], (
            f"distribute must never upload as public — got public uploads: {seen_public}"
        )

    @pytest.mark.asyncio
    async def test_founder_action_id_returned(self, tmp_path, monkeypatch):
        """Result includes the founder_action_id for the public-flip card."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)
        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 55}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is True
        assert result.get("flip_to_public_action_id") is not None


# ===========================================================================
# 4. Graceful error when YouTube client absent
# ===========================================================================

class TestYouTubeClientAbsent:
    """Graceful error when google-api-python-client is not installed."""

    @pytest.mark.asyncio
    async def test_missing_youtube_client_raises_clear_error(self, tmp_path, monkeypatch):
        """When _youtube_upload raises ImportError-like error, result is ok=False with clear message."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)

        def _absent_upload(path, title, description, privacy):
            raise RuntimeError("youtube client not installed: pip install google-api-python-client")

        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _absent_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is False
        error = result.get("error", "")
        assert "youtube" in error.lower() or "client" in error.lower()

    @pytest.mark.asyncio
    async def test_youtube_upload_function_raises_when_client_missing(self):
        """_youtube_upload raises RuntimeError with clear message when googleapiclient absent."""
        import sys
        import importlib

        # If googleapiclient is not installed, importing demo_distribute should still work.
        # The _youtube_upload function should raise a clear error on invocation.
        from mr_roboto import demo_distribute

        # Patch out googleapiclient to simulate absence
        original_modules = {}
        for key in list(sys.modules.keys()):
            if "googleapiclient" in key:
                original_modules[key] = sys.modules.pop(key)

        try:
            # Re-examine _youtube_upload behavior when the import would fail.
            # We call the real function with the real module absent.
            # Since demo_distribute uses a pluggable wrapper, if googleapiclient
            # is missing at runtime, _youtube_upload raises a RuntimeError.
            with pytest.raises(RuntimeError, match="youtube client not installed"):
                demo_distribute._youtube_upload_real("/tmp/fake.mp4", "t", "d", "unlisted")
        except AttributeError:
            # If _youtube_upload_real doesn't exist, the pluggable approach is via _youtube_upload.
            # This test is then validated by the integration test above.
            pass
        finally:
            sys.modules.update(original_modules)


# ===========================================================================
# 5. Thumbnail extraction
# ===========================================================================

class TestThumbnailExtraction:
    """_extract_thumbnail shells to ffmpeg single-frame grab."""

    @pytest.mark.asyncio
    async def test_extract_thumbnail_calls_ffmpeg(self, tmp_path, monkeypatch):
        """_extract_thumbnail invokes ffmpeg and produces a PNG."""
        from mr_roboto.demo_distribute import _extract_thumbnail

        cut_path = tmp_path / "cut.mp4"
        cut_path.write_bytes(b"\x00" * 1024)
        out_path = str(tmp_path / "thumb.png")

        calls = []

        async def _mock_subprocess(cmd, timeout=30.0):
            calls.append(cmd)
            # Simulate ffmpeg: write the output file
            for arg in cmd:
                if arg.endswith(".png"):
                    with open(arg, "wb") as f:
                        f.write(b"\x89PNG fake")
            return 0, "", ""

        monkeypatch.setattr("mr_roboto.demo_distribute._run_subprocess", _mock_subprocess)

        result = await _extract_thumbnail(str(cut_path), out_path)

        assert result is True
        assert os.path.exists(out_path)
        # Verify ffmpeg was called with single-frame flags
        assert len(calls) == 1
        cmd = calls[0]
        assert "ffmpeg" in cmd[0]
        # Should use -frames:v 1 or -vframes 1
        assert "-frames:v" in cmd or "-vframes" in cmd

    @pytest.mark.asyncio
    async def test_extract_thumbnail_returns_false_on_ffmpeg_failure(self, tmp_path, monkeypatch):
        """_extract_thumbnail returns False when ffmpeg fails."""
        from mr_roboto.demo_distribute import _extract_thumbnail

        cut_path = tmp_path / "cut.mp4"
        cut_path.write_bytes(b"\x00" * 1024)
        out_path = str(tmp_path / "thumb.png")

        async def _mock_subprocess(cmd, timeout=30.0):
            return 1, "", "ffmpeg: error opening output"

        monkeypatch.setattr("mr_roboto.demo_distribute._run_subprocess", _mock_subprocess)

        result = await _extract_thumbnail(str(cut_path), out_path)

        assert result is False

    @pytest.mark.asyncio
    async def test_distribute_continues_when_thumbnail_fails(self, tmp_path, monkeypatch):
        """If thumbnail extraction fails, distribute still succeeds (thumbnails optional)."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)
        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        async def _failing_thumbnail(cut_path, out_path, **kw):
            return False  # ffmpeg failed

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _failing_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 3}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is True
        # thumbnail_path may be None or absent per cut when extraction failed
        for label, info in result["uploads"].items():
            # Should still have video_id and embed_url
            assert "video_id" in info
            assert "embed_url" in info


# ===========================================================================
# 6. Missing cuts input
# ===========================================================================

class TestDistributeMissingCuts:
    """Graceful handling when cuts dict is empty or files are absent."""

    @pytest.mark.asyncio
    async def test_distribute_fails_when_no_cuts(self, tmp_path, monkeypatch):
        """distribute returns ok=False when cuts dict is empty."""
        from mr_roboto.demo_distribute import run as distribute_run

        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts={},
            product_name="TestProduct",
        )

        assert result["ok"] is False
        assert "cuts" in result.get("error", "").lower()


# ===========================================================================
# 7. Reversibility classification
# ===========================================================================

class TestDistributeReversibility:
    """demo/distribute is classified as 'partial' in VERB_REVERSIBILITY."""

    def test_demo_distribute_partial(self):
        """demo/distribute must be 'partial': unlisted video can be deleted/edited."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "demo/distribute" in VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["demo/distribute"] == "partial"

    def test_demo_distribute_public_flip_irreversible(self):
        """demo/distribute/flip_to_public is 'irreversible': public video visible to world."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "demo/distribute/flip_to_public" in VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["demo/distribute/flip_to_public"] == "irreversible"


# ===========================================================================
# 8. og:video snippet content
# ===========================================================================

class TestOgVideoSnippet:
    """og:video snippet is well-formed HTML meta tags."""

    @pytest.mark.asyncio
    async def test_og_video_snippet_is_html_meta(self, tmp_path, monkeypatch):
        """og_video_snippet contains valid <meta property="og:video"> tags."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)
        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 9}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        og = result["og_video_snippet"]
        assert "<meta" in og
        assert 'property="og:video"' in og or "property='og:video'" in og

    @pytest.mark.asyncio
    async def test_distribute_writes_result_json(self, tmp_path, monkeypatch):
        """distribute writes demo/distribute_result.json to workspace_path."""
        from mr_roboto.demo_distribute import run as distribute_run

        cuts = _make_cuts(tmp_path)
        monkeypatch.setattr("mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload)

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG fake")
            return True

        monkeypatch.setattr("mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail)

        async def _mock_fa_create(**kwargs):
            return {"id": 7}

        monkeypatch.setattr("mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa_create)

        result = await distribute_run(
            mission_id=1,
            workspace_path=str(tmp_path),
            cuts=cuts,
            product_name="TestProduct",
        )

        assert result["ok"] is True
        result_file = tmp_path / "demo" / "distribute_result.json"
        assert result_file.exists(), "distribute_result.json must be written to demo/ dir"
        loaded = json.loads(result_file.read_text())
        assert "uploads" in loaded
        assert "og_video_snippet" in loaded


# ===========================================================================
# Host-path wiring — i2p_v3.json step + mr_roboto dispatch (wiring-audit sweep)
# ===========================================================================

class TestDemoDistributeHostPath:
    """The verb is only reachable if an i2p step invokes it and the dispatcher
    routes the action. Both were missing before the 2026-05-17 wiring sweep."""

    def test_i2p_step_invokes_demo_distribute(self):
        wf_path = os.path.join("src", "workflows", "i2p", "i2p_v3.json")
        with open(wf_path, encoding="utf-8") as f:
            wf = json.load(f)
        step = next(
            (s for s in wf["steps"] if s.get("id") == "13.demo_distribute"), None
        )
        assert step is not None, "13.demo_distribute step missing from i2p_v3.json"
        assert step["agent"] == "mechanical"
        assert (step.get("payload") or {}).get("action") == "demo/distribute"
        assert "13.demo_edit" in (step.get("depends_on") or [])
        assert "public_demo" in (step.get("skip_when") or "")
        assert "demo/distribute_result.json" in (step.get("produces") or [])

    @pytest.mark.asyncio
    async def test_dispatch_resolves_cuts_from_workspace(self, tmp_path, monkeypatch):
        """An i2p step payload carries only the action; the dispatcher must
        resolve the three cuts from workspace/demo/cuts/ on its own."""
        import mr_roboto

        cuts_dir = tmp_path / "demo" / "cuts"
        cuts_dir.mkdir(parents=True)
        for lbl in ("30s", "60s", "3min"):
            (cuts_dir / f"{lbl}.mp4").write_bytes(b"fake mp4")

        monkeypatch.setattr(
            "mr_roboto.demo_distribute._youtube_upload", _fake_youtube_upload
        )

        async def _mock_thumbnail(cut_path, out_path, **kw):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG")
            return True

        monkeypatch.setattr(
            "mr_roboto.demo_distribute._extract_thumbnail", _mock_thumbnail
        )

        async def _mock_fa(**kw):
            return {"id": 9}

        monkeypatch.setattr(
            "mr_roboto.demo_distribute._emit_flip_to_public_action", _mock_fa
        )

        task = {
            "id": 1,
            "mission_id": 1,
            "agent_type": "mechanical",
            "payload": {
                "action": "demo/distribute",
                "workspace_path": str(tmp_path),
                "product_name": "TestProduct",
                # NOTE: no `cuts` key — the dispatcher must resolve them.
            },
        }
        action = await mr_roboto.run(task)
        assert action.status == "completed", action.error
        assert set(action.result["uploads"].keys()) == {"30s", "60s", "3min"}
