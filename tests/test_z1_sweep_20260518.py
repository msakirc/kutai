"""Host-path tests for Z1 P2 orphan-verb fixes from the 2026-05-18 sweep.

Z1 P2 — `derive_token_tag_signature` and `kill_preview_url` both had a
registered mr_roboto dispatch branch but no production caller (no i2p
step, no cron, no command). Two new i2p_v3 steps wire them:

  - 5.0.tag_signature — sibling of 5.0.verify (taste_extraction_verify),
    derives the paraflow tag_signature consumed by 5.30a/b HTML
    generation.
  - 15.14z_kill_preview_url — mission-end cleanup, depends on
    15.14b_deliverable_bundle so the preview URL tunnel survives long
    enough for the deliverable bundle and any out-of-band founder
    inspection.

The third Z1 orphan (`propose_spec_patch_from_html_diff`, P3) is left
for a follow-up — it needs a new Telegram inline-button callback.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
I2P_V3 = REPO_ROOT / "src/workflows/i2p/i2p_v3.json"


def _load_steps() -> dict:
    return {
        s["id"]: s
        for s in json.loads(I2P_V3.read_text(encoding="utf-8"))["steps"]
    }


class TestZ1P2TagSignatureStepWired(unittest.TestCase):
    """5.0.tag_signature must dispatch derive_token_tag_signature."""

    def test_step_exists(self):
        self.assertIn("5.0.tag_signature", _load_steps())

    def test_step_is_mechanical(self):
        s = _load_steps()["5.0.tag_signature"]
        self.assertEqual(s["agent"], "mechanical")
        self.assertEqual(
            s["payload"]["action"], "derive_token_tag_signature",
        )

    def test_step_depends_on_taste_verifier(self):
        s = _load_steps()["5.0.tag_signature"]
        self.assertIn("5.0.verify", s.get("depends_on", []))


class TestZ1P2KillPreviewUrlStepWired(unittest.TestCase):
    """15.14z_kill_preview_url must dispatch kill_preview_url at mission end."""

    def test_step_exists(self):
        self.assertIn("15.14z_kill_preview_url", _load_steps())

    def test_step_is_mechanical(self):
        s = _load_steps()["15.14z_kill_preview_url"]
        self.assertEqual(s["agent"], "mechanical")
        self.assertEqual(s["payload"]["action"], "kill_preview_url")

    def test_step_runs_after_deliverable_bundle(self):
        """Cleanup must NOT run before the bundle posts — the founder may
        click the preview link from the bundle message."""
        s = _load_steps()["15.14z_kill_preview_url"]
        self.assertIn(
            "15.14b_deliverable_bundle", s.get("depends_on", []),
        )

    def test_step_is_irreversible(self):
        """Closing a tunnel is not undoable — the tag must reflect that
        so the Z10 confirm policy can gate it."""
        s = _load_steps()["15.14z_kill_preview_url"]
        self.assertEqual(s["reversibility"], "irreversible")


class TestZ1P2BothVerbsHaveDispatchBranch(unittest.TestCase):
    """Sanity: the i2p step has somewhere to dispatch to."""

    def test_derive_token_tag_signature_dispatch_present(self):
        src = (
            REPO_ROOT
            / "packages/mr_roboto/src/mr_roboto/__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn('action == "derive_token_tag_signature"', src)

    def test_kill_preview_url_dispatch_present(self):
        src = (
            REPO_ROOT
            / "packages/mr_roboto/src/mr_roboto/__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn('action == "kill_preview_url"', src)


if __name__ == "__main__":
    unittest.main()
