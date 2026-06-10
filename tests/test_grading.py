import pytest
from src.core.grading import parse_grade_response, GradeResult


class TestGraderSemanticOnly:
    """Grader judges semantic quality only. Structural completeness
    (required fields/sections present + non-empty) is verified
    deterministically by the schema gate BEFORE the grader runs, so the
    grader must never fail an output for field/section drift — the exact
    mechanism behind the instruction>schema `COMPLETE:NO` DLQ class.
    """

    def test_system_prompt_declares_structure_verified_upstream(self):
        # The grading system prompt migrated from the GRADING_SYSTEM constant
        # to the Foundry "grading" rubric (build_messages). Intent unchanged:
        # the system message must declare structure is verified upstream and
        # forbid failing for field/section drift.
        from prompt_foundry import build_messages
        msgs = build_messages("grading", {
            "title": "t", "description": "d", "response": "r",
        })
        s = msgs[0]["content"].lower()
        # Structure is verified deterministically before the grader.
        assert "deterministic" in s
        # Must forbid failing for field/section drift.
        assert "missing, extra, or renamed" in s
        assert "never fail" in s

    def test_complete_field_is_semantic_not_presence(self):
        # COMPLETE-field semantics now live in the Foundry "grading" rubric
        # user_template (was GRADING_PROMPT). Intent unchanged.
        from prompt_foundry import build_messages
        msgs = build_messages("grading", {
            "title": "t", "description": "d", "response": "r",
        })
        p = msgs[1]["content"].lower()
        # COMPLETE is redefined to content adequacy, not field presence.
        assert "complete:" in p
        assert "content" in p
        # Explicitly tells the grader not to judge presence / not to
        # penalise instruction-named fields absent from the output.
        assert "do not judge" in p or "not judge" in p
        assert "absent from the output" in p


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

    def test_well_formed_pass_does_not_trigger_bare_cascade(self):
        """WELL_FORMED: PASS must not be mistaken for a bare PASS verdict."""
        raw = "WELL_FORMED: PASS\nCOHERENT: PASS"
        with pytest.raises(ValueError, match="grader incapable"):
            parse_grade_response(raw)

    def test_well_formed_fail_does_not_trigger_bare_cascade(self):
        """WELL_FORMED: FAIL must not be mistaken for a bare FAIL verdict."""
        raw = "WELL_FORMED: FAIL\nCOHERENT: FAIL"
        with pytest.raises(ValueError, match="grader incapable"):
            parse_grade_response(raw)

    def test_well_formed_overrides_verdict_pass(self):
        raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\nWELL_FORMED: FAIL\nCOHERENT: PASS"
        result = parse_grade_response(raw)
        assert result.passed is False  # WELL_FORMED: FAIL overrides VERDICT: PASS
        assert result.well_formed is False

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


class TestThinkingPreambleStrip:
    """Thinking-model preambles must not hide a valid structured grade."""

    def test_strip_think_xml_block(self):
        raw = (
            "<think>Let me analyze this task. The result looks good.</think>\n"
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
            "WELL_FORMED: PASS\nCOHERENT: PASS\n"
            "SITUATION: test\nSTRATEGY: test\nTOOLS: api_call"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.relevant is True

    def test_strip_thinking_process_preamble(self):
        raw = (
            "Thinking Process:\n"
            "1. Analyze the request: the user wants X.\n"
            "2. Evaluate: the result covers X correctly.\n\n"
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
            "WELL_FORMED: PASS\nCOHERENT: PASS\n"
            "SITUATION: test\nSTRATEGY: test\nTOOLS: api_call"
        )
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_strip_markdown_thinking_header(self):
        raw = (
            "## Thinking\n"
            "The task asks for X. The response provides X. Looks complete.\n\n"
            "VERDICT: PASS\nWELL_FORMED: PASS\nCOHERENT: PASS"
        )
        result = parse_grade_response(raw)
        assert result.passed is True

    def test_pure_thinking_no_verdict_still_raises(self):
        """If the model spilled ONLY reasoning and no structured fields, fail loudly."""
        raw = (
            "Thinking Process:\n"
            "1. I am analyzing the request carefully.\n"
            "2. Considering all angles.\n"
            "3. Here are my thoughts on..."
        )
        with pytest.raises(ValueError):
            parse_grade_response(raw)

    def test_strip_numbered_analyze_preamble(self):
        """Task #2384 regression: Qwen3.5-A3B style numbered bullet analysis
        before structured fields must not block parsing."""
        raw = (
            "1.  **Analyze the Request:**\n"
            "    *   Task: Evaluate a specific task result based on provided criteria.\n"
            "    *   Input Task: `[0.1] raw_idea_intake` - Extract raw idea.\n\n"
            "2.  **Evaluate the Result against Task Requirements:**\n"
            "    *   Sections present: Original Statement, Mentioned Features\n\n"
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
            "WELL_FORMED: PASS\nCOHERENT: PASS\n"
            "SITUATION: test\nSTRATEGY: test\nTOOLS: none"
        )
        result = parse_grade_response(raw)
        assert result.passed is True
        assert result.relevant is True

    def test_tail_region_ignores_echoed_task_description(self):
        """Echoed `Task:` / `Description:` in preamble must not collide with
        field regex. Parser focuses on last contiguous KEY: run."""
        raw = (
            "Task: Compare products. Description: three items to compare.\n"
            "Looking at the result, I see that the agent completed.\n\n"
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: FAIL\n"
            "WELL_FORMED: PASS\nCOHERENT: PASS"
        )
        result = parse_grade_response(raw)
        assert result.passed is False  # VERDICT: FAIL wins
        assert result.relevant is True

    def test_last_match_wins_when_key_echoed_in_reasoning(self):
        """Model may quote the prompt (`VERDICT: PASS or FAIL`) inside
        reasoning. The last line with an actual value must win."""
        raw = (
            "The instructions said to reply with VERDICT: PASS or FAIL.\n"
            "I will evaluate now.\n\n"
            "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: FAIL\n"
            "WELL_FORMED: PASS\nCOHERENT: PASS"
        )
        result = parse_grade_response(raw)
        assert result.passed is False


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

    def test_preference_na_becomes_empty(self):
        raw = "VERDICT: PASS\nPREFERENCE: N/A"
        result = parse_grade_response(raw)
        assert result.preference == ""

    def test_preference_no_prefix_becomes_empty(self):
        raw = "VERDICT: PASS\nPREFERENCE: No preference observed"
        result = parse_grade_response(raw)
        assert result.preference == ""

    def test_insight_not_applicable_becomes_empty(self):
        raw = "VERDICT: PASS\nINSIGHT: Not applicable."
        result = parse_grade_response(raw)
        assert result.insight == ""


class TestInsightField:
    def test_insight_parsed(self):
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: test\nSTRATEGY: test\nTOOLS: api_call\n"
            "PREFERENCE: NONE\n"
            "INSIGHT: Turkish e-commerce sites require User-Agent header"
        )
        result = parse_grade_response(raw)
        assert result.insight == "Turkish e-commerce sites require User-Agent header"

    def test_insight_none_becomes_empty(self):
        raw = "VERDICT: PASS\nINSIGHT: NONE"
        result = parse_grade_response(raw)
        assert result.insight == ""

    def test_missing_insight_stays_empty(self):
        raw = "VERDICT: PASS"
        result = parse_grade_response(raw)
        assert result.insight == ""

    def test_default_insight_empty(self):
        result = GradeResult(passed=True)
        assert result.insight == ""


class TestMultilineParsing:
    def test_tools_spanning_two_lines(self):
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: Multi-store price check\n"
            "STRATEGY: Sequential scraping\n"
            "TOOLS: smart_search, web_search,\n"
            "  api_call, scraper\n"
            "PREFERENCE: NONE\n"
            "INSIGHT: NONE"
        )
        result = parse_grade_response(raw)
        assert "smart_search" in result.tools
        assert "api_call" in result.tools
        assert "scraper" in result.tools

    def test_strategy_wrapping(self):
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: Complex research\n"
            "STRATEGY: First searched each store,\n"
            "  then compared prices across all\n"
            "TOOLS: smart_search\n"
            "PREFERENCE: NONE"
        )
        result = parse_grade_response(raw)
        assert "First searched each store" in result.strategy
        assert "compared prices" in result.strategy

    def test_single_line_still_works(self):
        """Regression: single-line values must still parse correctly."""
        raw = (
            "VERDICT: PASS\n"
            "SITUATION: Weather lookup\n"
            "STRATEGY: Used API\n"
            "TOOLS: api_call"
        )
        result = parse_grade_response(raw)
        assert result.situation == "Weather lookup"
        assert result.strategy == "Used API"
        assert result.tools == ["api_call"]

    def test_last_field_captures_to_end(self):
        """Last field in output has no next KEY: to stop at."""
        raw = (
            "VERDICT: PASS\n"
            "INSIGHT: Turkish sites need UA header\n"
            "  and proper Accept-Language"
        )
        result = parse_grade_response(raw)
        assert "Turkish sites need UA header" in result.insight
        assert "Accept-Language" in result.insight


class TestBuildGradingSpecAutoFail:
    """SP3: the auto-fail-on-trivial-output short-circuit moved from the deleted
    grade_task() into build_grading_spec(). When the source output is empty or
    too short, the builder returns a GradeResult(passed=False) auto-fail verdict
    instead of a Beckman spec, so the caller applies the fail without enqueueing
    a reviewer child."""

    def test_empty_result_auto_fails(self):
        from src.core.grading import build_grading_spec
        source = {"id": 1, "title": "Test", "description": "Test", "result": ""}
        result = build_grading_spec(source, exclusions=[])
        assert isinstance(result, GradeResult)
        assert result.passed is False
        assert "auto-fail" in result.raw

    def test_short_result_auto_fails(self):
        from src.core.grading import build_grading_spec
        source = {"id": 1, "title": "Test", "description": "Test", "result": "ok"}
        result = build_grading_spec(source, exclusions=[])
        assert isinstance(result, GradeResult)
        assert result.passed is False
        assert "auto-fail" in result.raw
