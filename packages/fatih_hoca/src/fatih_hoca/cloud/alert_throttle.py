"""Per-provider alert cooldown state.

Persists last-alert timestamp + last-known-state per provider in a JSON
file. Alert decision rule:

    should_alert returns True iff
        - no prior alert for this provider, OR
        - last alert was >= 24h ago, OR
        - state transitioned since last alert
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ALERT_COOLDOWN_SECONDS = 24 * 3600


class AlertThrottle:
    def __init__(self, state_path: Path):
        self._path = Path(state_path)
        self._state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._state = json.loads(self._path.read_text())
            except Exception:
                self._state = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2))

    def should_alert(self, provider: str, current_state: str) -> bool:
        entry = self._state.get(provider)
        now = time.time()
        if entry is None:
            self._state[provider] = {"last_state": current_state, "last_alert_unix": now}
            self._save()
            return True
        prior_state = entry.get("last_state")
        last_alert = float(entry.get("last_alert_unix", 0.0))
        is_transition = prior_state != current_state
        cooldown_passed = (now - last_alert) >= ALERT_COOLDOWN_SECONDS
        if is_transition or cooldown_passed:
            entry["last_state"] = current_state
            entry["last_alert_unix"] = now
            self._save()
            return True
        # Still record state for accuracy, but don't trip the alert.
        entry["last_state"] = current_state
        self._save()
        return False
