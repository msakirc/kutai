"""JSONL formatter — the schema contract for all yazbunu logs."""

import json
import logging
from datetime import datetime, timezone

# Context fields that get promoted to top-level JSON keys
CONTEXT_FIELDS = frozenset({
    "task", "mission", "agent", "model", "duration_ms",
})


class YazFormatter(logging.Formatter):
    """
    Outputs one JSON object per line.

    Required: ts, level, src, msg
    WARNING+: fn, ln (auto-populated from LogRecord)
    Optional: any extra context fields set on the record
    """

    def format(self, record: logging.LogRecord) -> str:
        doc: dict = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "src": record.name,
            "msg": record.getMessage(),
        }

        # fn/ln only on WARNING+
        if record.levelno >= logging.WARNING:
            doc["fn"] = record.funcName
            doc["ln"] = record.lineno

        # Known context fields
        for field in CONTEXT_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                doc[field] = val

        # Arbitrary extra context (set via _ContextLogger)
        for key, val in getattr(record, "_yaz_extra", {}).items():
            if key not in doc:
                doc[key] = val

        # Exception
        if record.exc_info and record.exc_info[0] is not None:
            doc["exc"] = self.formatException(record.exc_info)

        return json.dumps(doc, ensure_ascii=False)
