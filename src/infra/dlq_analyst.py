"""DLQ Analyst — deterministic failure pattern detection.

Hooks into quarantine_task() to detect cross-task failure patterns.
Groups recent DLQ entries by error category, model, tool, and mission.
Sends Telegram alerts with inline action buttons when patterns are detected.
No LLM calls — pure Python pattern matching.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Optional

import aiohttp

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

    async def run_diagnostic(self, pattern_key: str, entries: list[dict]) -> str:
        """Run a quick diagnostic check based on the failure pattern.

        Returns a human-readable diagnostic string, or empty string if no check applies.
        """
        group_type, group_value = pattern_key.split(":", 1)

        if group_type == "category":
            if group_value == "timeout":
                return await self._check_llama_health()
            if group_value == "network_error":
                return await self._check_connectivity()
            if group_value == "rate_limit":
                return "Rate limiting detected — external API throttling"

        # Check if failures reference the same model
        model_counts = self._extract_model_mentions(entries)
        if model_counts:
            top_model, count = max(model_counts.items(), key=lambda x: x[1])
            if count >= PATTERN_THRESHOLD:
                return f"Model {top_model}: failed {count} times — may be misconfigured"

        return ""

    async def _check_llama_health(self) -> str:
        """Ping llama-server /health endpoint."""
        try:
            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://127.0.0.1:8080/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    elapsed = round(time.time() - start, 1)
                    if resp.status == 200:
                        if elapsed > 2.0:
                            return f"llama-server responding but slow ({elapsed}s)"
                        return f"llama-server healthy ({elapsed}s) — timeouts likely from long generation"
                    return f"llama-server unhealthy (HTTP {resp.status})"
        except Exception:
            return "llama-server not responding — may be down or restarting"

    async def _check_connectivity(self) -> str:
        """Basic internet connectivity check."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.google.com", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return "Internet reachable — issue may be with specific API"
                    return f"Connectivity check returned HTTP {resp.status}"
        except Exception:
            return "Network connectivity issue — internet unreachable"

    @staticmethod
    def _extract_model_mentions(entries: list[dict]) -> dict[str, int]:
        """Extract model names mentioned in error text."""
        model_counts: dict[str, int] = defaultdict(int)
        pattern = re.compile(r"model=([^\s,)]+)")
        for entry in entries:
            error = entry.get("error", "")
            for match in pattern.finditer(error):
                model_counts[match.group(1)] += 1
        return dict(model_counts)
