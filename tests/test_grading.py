import pytest
from src.core.grading import parse_grade_response, GradeResult


class TestGradeResult:
    def test_passed_has_no_score_field(self):
        """GradeResult must not have a score field."""
        result = GradeResult(passed=True)
        assert not hasattr(result, "score")

    def test_passed_with_skill_fields(self):
        result = GradeResult(
            passed=True,
            situation="Price comparison across Turkish stores",
            strategy="Search each store separately then compare",
            tools=["smart_search", "web_search"],
        )
        assert result.passed is True
        assert result.situation == "Price comparison across Turkish stores"
        assert result.tools == ["smart_search", "web_search"]

    def test_default_skill_fields_empty(self):
        result = GradeResult(passed=False)
        assert result.situation == ""
        assert result.strategy == ""
        assert result.tools == []


class TestParseGradeResponse:
    def test_full_output_with_skill_fields(self):
        raw = (
            "RELEVANT: YES\n"
            "COMPLETE: YES\n"
            "VERDICT: PASS\n"
            "SITUATION: Price comparison across Turkish stores\n"
            "STRATEGY: Search each store separately then compare\n"
            "TOOLS: smart_search, web_search"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.relevant is True
        assert result.complete is True
        assert result.situation == "Price comparison across Turkish stores"
        assert result.strategy == "Search each store separately then compare"
        assert result.tools == ["smart_search", "web_search"]

    def test_verdict_pass_without_skill_fields(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS"
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.situation == ""
        assert result.strategy == ""
        assert result.tools == []

    def test_verdict_fail(self):
        raw = "RELEVANT: YES\nCOMPLETE: NO\nVERDICT: FAIL"
        result = parse_grade_response(raw)
        assert result.passed is False
        assert result.complete is False

    def test_all_yes_no(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: YES"
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_verdict_no(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: NO"
        result = parse_grade_response(raw)
        assert result.passed is False

    def test_derive_from_relevant_complete_when_no_verdict(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES"
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_derive_fail_when_relevant_no(self):
        raw = "RELEVANT: NO\nCOMPLETE: YES"
        result = parse_grade_response(raw)
        assert result.passed is False

    def test_bare_pass_fallback(self):
        raw = "I think this response is good and should be accepted. PASS"
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.relevant is None
        assert result.complete is None

    def test_bare_fail_fallback(self):
        raw = "The response does not address the task at all. FAIL"
        result = parse_grade_response(raw)
        assert result.passed is False

    def test_bare_pass_case_insensitive(self):
        raw = "Overall this is a pass from me."
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="grader incapable"):
            parse_grade_response("Here is my analysis of the task response quality metrics")

    def test_case_insensitive_fields(self):
        raw = "relevant: yes\ncomplete: Yes\nverdict: PASS"
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_with_reasoning_noise(self):
        raw = (
            "The response looks good.\n"
            "RELEVANT: YES\n"
            "I think it is complete.\n"
            "COMPLETE: YES\n"
            "VERDICT: PASS\n"
            "SITUATION: Weather lookup for Istanbul\n"
            "STRATEGY: Used weather API directly\n"
            "TOOLS: api_call"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.situation == "Weather lookup for Istanbul"
        assert result.tools == ["api_call"]

    def test_partial_skill_fields(self):
        raw = (
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
            "SITUATION: Currency conversion task"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.situation == "Currency conversion task"
        assert result.strategy == ""
        assert result.tools == []

    def test_tools_with_spaces(self):
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: test\n"
            "STRATEGY: test\n"
            "TOOLS: smart_search , web_search , api_call"
        )
        result = parse_grade_response(raw)
        assert result.tools == ["smart_search", "web_search", "api_call"]


class TestPreferenceField:
    def test_preference_parsed(self):
        raw = (
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
            "SITUATION: test\nSTRATEGY: test\nTOOLS: api_call\n"
            "PREFERENCE: User prefers Turkish responses\n"
            "INSIGHT: NONE"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.preference == "User prefers Turkish responses"

    def test_preference_none_becomes_empty(self):
        raw = (
            "VERDICT: PASS\n"
            "PREFERENCE: NONE"
        )
        result = parse_grade_response(raw)
        assert result.preference == ""

    def test_preference_none_lowercase(self):
        raw = (
            "VERDICT: PASS\n"
            "PREFERENCE: none"
        )
        result = parse_grade_response(raw)
        assert result.preference == ""

    def test_missing_preference_stays_empty(self):
        raw = "VERDICT: PASS"
        result = parse_grade_response(raw)
        assert result.preference == ""

    def test_default_preference_empty(self):
        result = GradeResult(passed=True)
        assert result.preference == ""
