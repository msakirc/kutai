"""Quality gates for workflow phase transitions.

Gates define requirements that must be met before the next phase can start.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from src.infra.logging_config import get_logger
from .artifacts import ArtifactStore

logger = get_logger("workflows.engine.quality_gates")

# ── Gate definitions ──────────────────────────────────────────────────────────

PHASE_GATES: dict[str, dict[str, Any]] = {
    "phase_9": {  # Comprehensive Testing
        "test_pass_rate": 0.95,
        "coverage_minimum": 0.60,
    },
    "phase_10": {  # Security Hardening
        "security_scan_clean": True,
    },
    "phase_13": {  # Pre-Launch
        "all_tests_pass": True,
        "human_approval": True,
    },
}


def get_gate(phase_id: str) -> dict | None:
    """Return the gate definition for *phase_id*, or None if no gate."""
    return PHASE_GATES.get(phase_id)


# ── Criterion evaluators ─────────────────────────────────────────────────────

def _parse_test_pass_rate(artifact_value: str) -> Optional[float]:
    """Extract pass rate from test results artifact.

    Supports formats like:
      - JSON with "passed" / "failed" or "pass_rate" keys
      - Plain text with "X passed, Y failed"
    """
    if not artifact_value:
        return None

    # Try JSON first
    try:
        data = json.loads(artifact_value)
        if isinstance(data, dict):
            if "pass_rate" in data:
                return float(data["pass_rate"])
            passed = data.get("passed", 0)
            failed = data.get("failed", 0)
            total = passed + failed
            if total > 0:
                return passed / total
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try plain-text pattern: "X passed, Y failed"
    m = re.search(r"(\d+)\s*passed.*?(\d+)\s*failed", artifact_value, re.IGNORECASE)
    if m:
        passed = int(m.group(1))
        failed = int(m.group(2))
        total = passed + failed
        if total > 0:
            return passed / total

    return None


def _parse_coverage(artifact_value: str) -> Optional[float]:
    """Extract coverage percentage from a coverage report artifact.

    Supports:
      - JSON with "coverage" key (0-1 float or 0-100 int)
      - Plain text with "coverage: XX%" or "XX% coverage"
    """
    if not artifact_value:
        return None

    # Try JSON
    try:
        data = json.loads(artifact_value)
        if isinstance(data, dict) and "coverage" in data:
            val = float(data["coverage"])
            # Normalise percentages > 1 to 0-1 range
            if val > 1:
                val = val / 100.0
            return val
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Plain-text patterns
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", artifact_value)
    if m:
        return float(m.group(1)) / 100.0

    return None


def _check_security_scan(artifact_value: str) -> Optional[bool]:
    """Return True if the security scan is clean."""
    if not artifact_value:
        return None

    # Try JSON
    try:
        data = json.loads(artifact_value)
        if isinstance(data, dict):
            if "clean" in data:
                return bool(data["clean"])
            if "issues" in data:
                issues = data["issues"]
                if isinstance(issues, list):
                    return len(issues) == 0
                return not bool(issues)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Plain text
    lower = artifact_value.lower()
    if "clean" in lower or "no issues" in lower or "passed" in lower:
        return True
    if "fail" in lower or "issue" in lower or "vulnerability" in lower:
        return False

    return None


def _check_all_tests_pass(artifact_value: str) -> Optional[bool]:
    """Return True if all tests passed (zero failures)."""
    rate = _parse_test_pass_rate(artifact_value)
    if rate is not None:
        return rate >= 1.0
    return None


# ── Main evaluation ──────────────────────────────────────────────────────────

async def evaluate_json_gate(
    mission_id: int,
    gate_def: dict,
    artifact_store: ArtifactStore,
) -> tuple[bool, dict]:
    """Evaluate a quality gate defined in workflow JSON.

    gate_def format:
    {
        "criteria": [
            {"name": "test_pass_rate", "artifact": "test_results", "check": "pass_rate >= 0.95", "message": "..."},
        ],
        "on_failure": "needs_clarification"
    }
    """
    criteria = gate_def.get("criteria", [])
    if not criteria:
        return True, {}

    details: dict[str, dict[str, Any]] = {}
    all_passed = True

    for criterion in criteria:
        name = criterion.get("name", "unknown")
        artifact_name = criterion.get("artifact", "")
        check_expr = criterion.get("check", "")
        message = criterion.get("message", name)

        artifact_value = await artifact_store.retrieve(mission_id, artifact_name)
        if artifact_value is None:
            details[name] = {
                "passed": False,
                "value": None,
                "message": f"Artifact '{artifact_name}' not found",
            }
            all_passed = False
            continue

        # Parse the check expression
        passed = _evaluate_check_expression(check_expr, artifact_value)
        details[name] = {
            "passed": passed,
            "value": artifact_value[:100] if isinstance(artifact_value, str) else artifact_value,
            "message": message if passed else f"FAILED: {message}",
        }
        if not passed:
            all_passed = False

    return all_passed, details


def _evaluate_check_expression(check_expr: str, artifact_value: str) -> bool:
    """Evaluate a simple check expression against an artifact value.

    Supports: 'field >= N', 'field == true/false', 'field == "value"'
    """
    if not check_expr:
        return True

    # Try to parse artifact as JSON
    data = {}
    try:
        parsed = json.loads(artifact_value)
        if isinstance(parsed, dict):
            data = parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Parse expression: "field >= value" or "field == value"
    import re as _re
    m = _re.match(r'(\w+)\s*(>=|<=|==|!=|>|<)\s*(.+)', check_expr.strip())
    if not m:
        return True  # Can't parse, assume passed

    field, op, expected_str = m.group(1), m.group(2), m.group(3).strip()

    # Get actual value
    actual = data.get(field)
    if actual is None:
        # Try parsing from text using existing helpers
        if 'pass_rate' in field:
            actual = _parse_test_pass_rate(artifact_value)
        elif 'coverage' in field:
            actual = _parse_coverage(artifact_value)
        elif 'clean' in field or 'scan' in field:
            actual = _check_security_scan(artifact_value)
        elif 'approv' in field:
            actual = artifact_value.strip().lower() == 'approved'

    if actual is None:
        return False

    # Parse expected value
    if expected_str.lower() in ('true', 'false'):
        expected: Any = expected_str.lower() == 'true'
        actual = bool(actual)
    else:
        try:
            expected = float(expected_str.strip('"\''))
            actual = float(actual)
        except (ValueError, TypeError):
            expected = expected_str.strip('"\'')
            actual = str(actual)

    # Compare
    if op == '>=': return actual >= expected
    if op == '<=': return actual <= expected
    if op == '==': return actual == expected
    if op == '!=': return actual != expected
    if op == '>': return actual > expected
    if op == '<': return actual < expected
    return True


async def evaluate_gate(
    mission_id: int,
    phase_id: str,
    artifact_store: ArtifactStore,
    gate_def: Optional[dict] = None,
) -> tuple[bool, dict]:
    """Evaluate the quality gate for *phase_id*.

    Returns
    -------
    (passed, details)
        *passed* is True when every criterion is satisfied (or no gate exists).
        *details* maps each criterion name to a dict with ``passed``, ``value``,
        and ``message`` keys.
    """
    # If a JSON gate definition is provided (v3), use it
    if gate_def is not None:
        return await evaluate_json_gate(mission_id, gate_def, artifact_store)

    # Fall back to hardcoded gates (v1/v2 compatibility)
    gate = get_gate(phase_id)
    if gate is None:
        return True, {}

    details: dict[str, dict[str, Any]] = {}
    all_passed = True

    # Extract the phase number for artifact name lookups
    phase_num = phase_id.replace("phase_", "")

    for criterion, threshold in gate.items():
        if criterion == "test_pass_rate":
            artifact = (
                await artifact_store.retrieve(mission_id, "test_results")
                or await artifact_store.retrieve(mission_id, f"phase_{phase_num}_test_results")
            )
            if artifact is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": "Artifact 'test_results' not found",
                }
                all_passed = False
                continue

            rate = _parse_test_pass_rate(artifact)
            if rate is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": "Could not parse test pass rate from artifact",
                }
                all_passed = False
                continue

            passed = rate >= threshold
            details[criterion] = {
                "passed": passed,
                "value": rate,
                "message": f"Test pass rate: {rate:.1%} (threshold: {threshold:.0%})",
            }
            if not passed:
                all_passed = False

        elif criterion == "coverage_minimum":
            artifact = await artifact_store.retrieve(mission_id, "coverage_report")
            if artifact is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": "Artifact 'coverage_report' not found",
                }
                all_passed = False
                continue

            coverage = _parse_coverage(artifact)
            if coverage is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": "Could not parse coverage from artifact",
                }
                all_passed = False
                continue

            passed = coverage >= threshold
            details[criterion] = {
                "passed": passed,
                "value": coverage,
                "message": f"Coverage: {coverage:.1%} (threshold: {threshold:.0%})",
            }
            if not passed:
                all_passed = False

        elif criterion == "security_scan_clean":
            artifact = await artifact_store.retrieve(mission_id, "security_scan_results")
            if artifact is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": "Artifact 'security_scan_results' not found",
                }
                all_passed = False
                continue

            clean = _check_security_scan(artifact)
            if clean is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": "Could not determine security scan status from artifact",
                }
                all_passed = False
                continue

            details[criterion] = {
                "passed": clean,
                "value": clean,
                "message": "Security scan: clean" if clean else "Security scan: issues found",
            }
            if not clean:
                all_passed = False

        elif criterion == "all_tests_pass":
            artifact = await artifact_store.retrieve(mission_id, "test_results")
            if artifact is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": "Artifact 'test_results' not found",
                }
                all_passed = False
                continue

            all_pass = _check_all_tests_pass(artifact)
            if all_pass is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": "Could not parse test results from artifact",
                }
                all_passed = False
                continue

            details[criterion] = {
                "passed": all_pass,
                "value": all_pass,
                "message": "All tests passed" if all_pass else "Some tests failed",
            }
            if not all_pass:
                all_passed = False

        elif criterion == "human_approval":
            artifact = await artifact_store.retrieve(
                mission_id, f"phase_{phase_num}_human_approval"
            )
            if artifact is None:
                details[criterion] = {
                    "passed": False,
                    "value": None,
                    "message": f"Artifact 'phase_{phase_num}_human_approval' not found",
                }
                all_passed = False
                continue

            approved = artifact.strip().lower() == "approved"
            details[criterion] = {
                "passed": approved,
                "value": artifact.strip(),
                "message": "Human approval: approved" if approved else f"Human approval: {artifact.strip()}",
            }
            if not approved:
                all_passed = False

        else:
            logger.warning(f"Unknown gate criterion: {criterion}")
            details[criterion] = {
                "passed": False,
                "value": None,
                "message": f"Unknown criterion '{criterion}'",
            }
            all_passed = False

    return all_passed, details


# ── Formatting ───────────────────────────────────────────────────────────────

def format_gate_result(phase_id: str, passed: bool, details: dict) -> str:
    """Format a human-readable gate result suitable for Telegram notification."""
    status = "PASSED" if passed else "FAILED"
    lines = [f"Quality Gate {phase_id}: {status}"]

    if not details:
        lines.append("  No criteria (gate auto-passed)")
        return "\n".join(lines)

    for criterion, info in details.items():
        icon = "\u2705" if info.get("passed") else "\u274c"
        message = info.get("message", criterion)
        lines.append(f"  {icon} {message}")

    return "\n".join(lines)
