"""Test _format_log_entries for /logs command."""
import json
import types
import unittest

# Import _format_log_entries without importing the full telegram_bot module.
# telegram_bot.py has heavy deps (python-telegram-bot) that may not be installed.
import importlib.util
import ast

_spec = importlib.util.find_spec("src.app.telegram_bot")
with open(_spec.origin, encoding="utf-8") as _f:
    _source = _f.read()

_tree = ast.parse(_source)
_mod = types.ModuleType("_extracted")
_mod.__builtins__ = __builtins__
for _node in ast.iter_child_nodes(_tree):
    if isinstance(_node, ast.FunctionDef) and _node.name == "_format_log_entries":
        _func_source = ast.get_source_segment(_source, _node)
        exec(compile(_func_source, "<test>", "exec"), _mod.__dict__)
        break

fmt = _mod._format_log_entries


class TestFormatLogEntries(unittest.TestCase):

    def test_format_last_n_lines(self):
        """Should return last N lines, most recent last."""
        lines = [
            json.dumps({"timestamp": "2026-03-31T10:00:00", "level": "INFO", "component": "core", "message": "started"}),
            json.dumps({"timestamp": "2026-03-31T10:00:01", "level": "ERROR", "component": "agent", "message": "failed"}),
            json.dumps({"timestamp": "2026-03-31T10:00:02", "level": "INFO", "component": "core", "message": "recovered"}),
        ]

        result = fmt(lines, n=2)
        self.assertIn("failed", result)        # ERROR entry
        self.assertIn("recovered", result)     # last INFO entry
        self.assertNotIn("started", result)    # first entry should be excluded

    def test_format_empty_log(self):
        result = fmt([], n=10)
        self.assertIn("No log entries", result)

    def test_format_handles_malformed_json(self):
        lines = ["not valid json", '{"timestamp":"T12:00:00","level":"INFO","component":"x","message":"ok"}']
        result = fmt(lines, n=5)
        self.assertIn("not valid json", result)
        self.assertIn("ok", result)


if __name__ == "__main__":
    unittest.main()
