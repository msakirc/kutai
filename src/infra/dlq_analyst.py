"""DLQ Analyst — deterministic failure pattern detection.

Hooks into quarantine_task() to detect cross-task failure patterns.
Groups recent DLQ entries by error category, model, tool, and mission.
Sends Telegram alerts with inline action buttons when patterns are detected.
No LLM calls — pure Python pattern matching.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("infra.dlq_analyst")

# Alert thresholds
PATTERN_THRESHOLD = 3       # min matches to trigger alert
DEDUP_SECONDS = 3600        # 1 hour dedup per pattern key


class DLQAnalyst:
    """Detects failure patterns across DLQ entries and formats alerts."""

    WINDOW_HOURS = 3  # sliding window for pattern detection

    def __init__(self):
        self._last_alert: dict[str, float] = {}  # pattern_key -> timestamp

    def detect_patterns(self, entries: list[dict]) -> list[dict]:
        """Group DLQ entries and return patterns that exceed threshold.

        Args:
            entries: list of dead_letter_tasks rows (dicts) from the last WINDOW_HOURS.

        Returns:
            List of pattern dicts with keys: group_key, count, entries
        """
        groups: dict[str, list[dict]] = defaultdict(list)

        for entry in entries:
            # Group by error category
            cat = entry.get("error_category", "unknown")
            groups[f"category:{cat}"].append(entry)

            # Group by mission
            mid = entry.get("mission_id")
            if mid:
                groups[f"mission:{mid}"].append(entry)

        # Filter to patterns exceeding threshold
        patterns = []
        for key, items in groups.items():
            if len(items) >= PATTERN_THRESHOLD:
                patterns.append({
                    "group_key": key,
                    "count": len(items),
                    "entries": items,
                })

        return patterns

    def is_deduped(self, pattern_key: str) -> bool:
        """Check if this pattern was already alerted within DEDUP_SECONDS."""
        last = self._last_alert.get(pattern_key)
        if last is None:
            return False
        return (time.time() - last) < DEDUP_SECONDS

    def record_alert(self, pattern_key: str) -> None:
        """Record that an alert was sent for this pattern."""
        self._last_alert[pattern_key] = time.time()

    def format_alert(self, pattern: dict) -> str:
        """Format a pattern into a Telegram alert message."""
        key = pattern["group_key"]
        count = pattern["count"]
        entries = pattern["entries"]
        diagnostic = pattern.get("diagnostic", "")

        # Build header
        group_type, group_value = key.split(":", 1)
        if group_type == "category":
            header = f"DLQ Pattern: {count} tasks failed with {group_value}"
        elif group_type == "mission":
            header = f"DLQ Pattern: {count} tasks from mission #{group_value} failed"
        else:
            header = f"DLQ Pattern: {count} tasks matched {key}"

        # Build task list (max 5)
        lines = [header, ""]
        for entry in entries[:5]:
            tid = entry.get("task_id", "?")
            agent = entry.get("original_agent", "?")
            error = entry.get("error", "")[:80]
            lines.append(f"- Task #{tid} ({agent}): {error}")

        if len(entries) > 5:
            lines.append(f"  ... and {len(entries) - 5} more")

        # Add diagnostic if available
        if diagnostic:
            lines.append("")
            lines.append(f"Diagnostic: {diagnostic}")

        return "\n".join(lines)
