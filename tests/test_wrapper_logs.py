"""Verify the wrapper handles /logs as a wrapper command."""
import unittest


class TestWrapperLogsCommand(unittest.TestCase):

    def test_wrapper_recognizes_logs_command(self):
        with open("kutai_wrapper.py", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("/logs", text)
        self.assertIn('"/logs"', text)

    def test_wrapper_reads_orchestrator_jsonl(self):
        with open("kutai_wrapper.py", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("orchestrator.jsonl", text)


if __name__ == "__main__":
    unittest.main()
