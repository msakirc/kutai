"""Tests for quality gates (Gap 4)."""

import asyncio
import json
import unittest

from src.workflows.engine.quality_gates import (
    get_gate, evaluate_gate, format_gate_result,
    _parse_test_pass_rate, _parse_coverage,
    _check_security_scan, _check_all_tests_pass,
)
from src.workflows.engine.artifacts import ArtifactStore


class TestGetGate(unittest.TestCase):

    def test_returns_gate_for_phase_9(self):
        gate = get_gate("phase_9")
        self.assertIsNotNone(gate)
        self.assertIn("test_pass_rate", gate)

    def test_returns_gate_for_phase_10(self):
        gate = get_gate("phase_10")
        self.assertIn("security_scan_clean", gate)

    def test_returns_gate_for_phase_13(self):
        gate = get_gate("phase_13")
        self.assertIn("all_tests_pass", gate)
        self.assertIn("human_approval", gate)

    def test_returns_none_for_ungated_phase(self):
        self.assertIsNone(get_gate("phase_1"))


class TestParseTestPassRate(unittest.TestCase):

    def test_json_pass_rate(self):
        self.assertAlmostEqual(_parse_test_pass_rate(json.dumps({"pass_rate": 0.98})), 0.98)

    def test_json_passed_failed(self):
        self.assertAlmostEqual(_parse_test_pass_rate(json.dumps({"passed": 95, "failed": 5})), 0.95)

    def test_plain_text(self):
        self.assertAlmostEqual(_parse_test_pass_rate("Results: 48 passed, 2 failed"), 0.96)

    def test_empty_returns_none(self):
        self.assertIsNone(_parse_test_pass_rate(""))
        self.assertIsNone(_parse_test_pass_rate(None))

    def test_unparseable_returns_none(self):
        self.assertIsNone(_parse_test_pass_rate("no data here"))


class TestParseCoverage(unittest.TestCase):

    def test_json_coverage_decimal(self):
        self.assertAlmostEqual(_parse_coverage(json.dumps({"coverage": 0.75})), 0.75)

    def test_json_coverage_percentage(self):
        self.assertAlmostEqual(_parse_coverage(json.dumps({"coverage": 82})), 0.82)

    def test_plain_text_percentage(self):
        self.assertAlmostEqual(_parse_coverage("Total coverage: 65.5%"), 0.655)

    def test_empty_returns_none(self):
        self.assertIsNone(_parse_coverage(""))


class TestCheckSecurityScan(unittest.TestCase):

    def test_json_clean(self):
        self.assertTrue(_check_security_scan(json.dumps({"clean": True})))

    def test_json_issues_empty(self):
        self.assertTrue(_check_security_scan(json.dumps({"issues": []})))

    def test_json_issues_present(self):
        self.assertFalse(_check_security_scan(json.dumps({"issues": ["XSS"]})))

    def test_plain_text_clean(self):
        self.assertTrue(_check_security_scan("Security scan: clean, no issues found"))

    def test_plain_text_failure(self):
        self.assertFalse(_check_security_scan("Found 3 vulnerabilities"))

    def test_empty_returns_none(self):
        self.assertIsNone(_check_security_scan(""))


class TestCheckAllTestsPass(unittest.TestCase):

    def test_all_pass(self):
        self.assertTrue(_check_all_tests_pass(json.dumps({"passed": 100, "failed": 0})))

    def test_some_fail(self):
        self.assertFalse(_check_all_tests_pass(json.dumps({"passed": 98, "failed": 2})))


class TestEvaluateGate(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.store = ArtifactStore(use_db=False)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_no_gate_passes(self):
        passed, details = self._run(evaluate_gate(1, "phase_1", self.store))
        self.assertTrue(passed)
        self.assertEqual(details, {})

    def test_phase_9_passes_with_good_artifacts(self):
        self._run(self.store.store(1, "test_results", json.dumps({"passed": 100, "failed": 2})))
        self._run(self.store.store(1, "coverage_report", json.dumps({"coverage": 0.75})))
        passed, details = self._run(evaluate_gate(1, "phase_9", self.store))
        self.assertTrue(passed)

    def test_phase_9_fails_low_pass_rate(self):
        self._run(self.store.store(1, "test_results", json.dumps({"passed": 80, "failed": 20})))
        self._run(self.store.store(1, "coverage_report", json.dumps({"coverage": 0.75})))
        passed, details = self._run(evaluate_gate(1, "phase_9", self.store))
        self.assertFalse(passed)

    def test_phase_9_fails_missing_artifact(self):
        passed, details = self._run(evaluate_gate(1, "phase_9", self.store))
        self.assertFalse(passed)
        self.assertIn("not found", details["test_pass_rate"]["message"])

    def test_phase_10_passes_clean_scan(self):
        self._run(self.store.store(1, "security_scan_results", json.dumps({"clean": True})))
        passed, _ = self._run(evaluate_gate(1, "phase_10", self.store))
        self.assertTrue(passed)

    def test_phase_10_fails_dirty_scan(self):
        self._run(self.store.store(1, "security_scan_results", json.dumps({"issues": ["SQL injection"]})))
        passed, _ = self._run(evaluate_gate(1, "phase_10", self.store))
        self.assertFalse(passed)

    def test_phase_13_requires_human_approval(self):
        self._run(self.store.store(1, "test_results", json.dumps({"passed": 100, "failed": 0})))
        passed, details = self._run(evaluate_gate(1, "phase_13", self.store))
        self.assertFalse(passed)
        self.assertIn("human_approval", details)

    def test_phase_13_passes_all_criteria(self):
        self._run(self.store.store(1, "test_results", json.dumps({"passed": 100, "failed": 0})))
        self._run(self.store.store(1, "phase_13_human_approval", "approved"))
        passed, _ = self._run(evaluate_gate(1, "phase_13", self.store))
        self.assertTrue(passed)


class TestFormatGateResult(unittest.TestCase):

    def test_passed_format(self):
        result = format_gate_result("phase_9", True, {
            "test_pass_rate": {"passed": True, "message": "98% pass rate"}})
        self.assertIn("PASSED", result)

    def test_failed_format(self):
        result = format_gate_result("phase_9", False, {
            "test_pass_rate": {"passed": False, "message": "80% pass rate"}})
        self.assertIn("FAILED", result)

    def test_no_criteria_format(self):
        result = format_gate_result("phase_1", True, {})
        self.assertIn("auto-passed", result)


if __name__ == "__main__":
    unittest.main()
