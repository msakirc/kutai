import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.core.grading import parse_grade_response, GradeResult, apply_grade_result


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


class TestApplyGradeResultPass:
    """Test apply_grade_result PASS path."""

    @pytest.mark.asyncio
    @patch("src.core.state_machine.transition_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.infra.db.record_model_call", new_callable=AsyncMock)
    @patch("src.memory.preferences.store_preference", new_callable=AsyncMock)
    @patch("src.memory.episodic.store_insight", new_callable=AsyncMock)
    async def test_pass_with_rich_verdict(
        self, mock_insight, mock_pref, mock_record, mock_skill, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 42, "title": "Compare laptop prices",
            "agent_type": "shopping_advisor", "iterations": 3,
            "context": '{"generating_model": "test-model", "tools_used_names": ["smart_search", "web_search"], "chat_id": "12345"}',
        }
        verdict = GradeResult(
            passed=True,
            situation="Price comparison across Turkish stores",
            strategy="Search each store separately then compare",
            tools=["smart_search", "web_search"],
            preference="User prefers Turkish responses",
            insight="Trendyol requires User-Agent header",
        )

        await apply_grade_result(42, verdict)

        mock_trans.assert_called_once()
        mock_skill.assert_called_once()
        mock_pref.assert_called_once()
        mock_insight.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.core.state_machine.transition_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.infra.db.record_model_call", new_callable=AsyncMock)
    @patch("src.memory.preferences.store_preference", new_callable=AsyncMock)
    @patch("src.memory.episodic.store_insight", new_callable=AsyncMock)
    async def test_pass_empty_verdict_uses_mechanical_fallback(
        self, mock_insight, mock_pref, mock_record, mock_skill, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 43, "title": "Check weather",
            "agent_type": "executor", "iterations": 2,
            "context": '{"generating_model": "test-model", "tools_used_names": ["api_call"]}',
        }
        verdict = GradeResult(passed=True)

        await apply_grade_result(43, verdict)

        mock_trans.assert_called_once()
        mock_skill.assert_called_once()
        # Mechanical fallback: description should contain task title and agent type
        call_kwargs = mock_skill.call_args
        assert "Check weather" in call_kwargs.kwargs.get("description", call_kwargs[1].get("description", ""))
        mock_pref.assert_not_called()
        mock_insight.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.core.state_machine.transition_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.infra.db.record_model_call", new_callable=AsyncMock)
    async def test_pass_low_iterations_skips_skill(
        self, mock_record, mock_skill, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 44, "title": "Simple lookup",
            "agent_type": "executor", "iterations": 1,
            "context": '{"generating_model": "test-model", "tools_used_names": ["api_call"]}',
        }
        verdict = GradeResult(passed=True, situation="Simple API call")

        await apply_grade_result(44, verdict)

        mock_trans.assert_called_once()
        mock_skill.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.core.state_machine.transition_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_task", new_callable=AsyncMock)
    @patch("src.memory.skills.add_skill", new_callable=AsyncMock)
    @patch("src.infra.db.record_model_call", new_callable=AsyncMock)
    async def test_pass_no_tools_skips_skill(
        self, mock_record, mock_skill, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 45, "title": "Think about it",
            "agent_type": "executor", "iterations": 5,
            "context": '{"generating_model": "test-model"}',
        }
        verdict = GradeResult(passed=True, situation="Deep thinking task")

        await apply_grade_result(45, verdict)

        mock_trans.assert_called_once()
        mock_skill.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.core.state_machine.transition_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_task", new_callable=AsyncMock)
    @patch("src.infra.db.record_model_call", new_callable=AsyncMock)
    async def test_pass_task_not_found(
        self, mock_record, mock_get, mock_trans
    ):
        mock_get.return_value = None

        await apply_grade_result(999, GradeResult(passed=True))

        mock_trans.assert_not_called()


class TestApplyGradeResultFail:
    """Test apply_grade_result FAIL path."""

    @pytest.mark.asyncio
    @patch("src.core.state_machine.transition_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_task", new_callable=AsyncMock)
    @patch("src.core.retry.RetryContext.from_task")
    async def test_fail_with_retries_remaining(
        self, mock_from_task, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 50, "title": "Failed task",
            "agent_type": "coder", "worker_attempts": 1, "max_worker_attempts": 6,
            "context": '{"generating_model": "test-model"}',
        }
        mock_retry_ctx = MagicMock()
        mock_retry_ctx.record_failure.return_value = MagicMock(action="delayed", delay_seconds=30)
        mock_retry_ctx.to_context_patch.return_value = {"failed_models": ["test-model"]}
        mock_retry_ctx.to_db_fields.return_value = {"worker_attempts": 2}
        mock_retry_ctx.grade_attempts = 0
        mock_retry_ctx.next_retry_at = None
        mock_from_task.return_value = mock_retry_ctx
        verdict = GradeResult(passed=False)

        await apply_grade_result(50, verdict)

        mock_retry_ctx.record_failure.assert_called_once_with("quality", model="test-model")
        # Should transition to pending for retry
        mock_trans.assert_called_once()
        args, kwargs = mock_trans.call_args
        assert args == (50, "pending")

    @pytest.mark.asyncio
    @patch("src.core.state_machine.transition_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_task", new_callable=AsyncMock)
    @patch("src.core.retry.RetryContext.from_task")
    @patch("src.infra.dead_letter.quarantine_task", new_callable=AsyncMock)
    async def test_fail_terminal_quarantines(
        self, mock_quarantine, mock_from_task, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 51, "title": "Hopeless task",
            "agent_type": "coder", "worker_attempts": 5, "max_worker_attempts": 6,
            "context": '{"generating_model": "test-model"}',
        }
        mock_retry_ctx = MagicMock()
        mock_retry_ctx.record_failure.return_value = MagicMock(action="terminal")
        mock_retry_ctx.to_context_patch.return_value = {"failed_models": ["test-model"]}
        mock_retry_ctx.to_db_fields.return_value = {"worker_attempts": 6}
        mock_retry_ctx.worker_attempts = 6
        mock_from_task.return_value = mock_retry_ctx
        verdict = GradeResult(passed=False)

        await apply_grade_result(51, verdict)

        # Should transition to failed
        mock_trans.assert_called_once()
        args, kwargs = mock_trans.call_args
        assert args == (51, "failed")
        # Should quarantine to DLQ
        mock_quarantine.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.core.state_machine.transition_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_task", new_callable=AsyncMock)
    @patch("src.core.retry.RetryContext.from_task")
    async def test_fail_immediate_retry(
        self, mock_from_task, mock_get, mock_trans
    ):
        mock_get.return_value = {
            "id": 52, "title": "Retry immediately",
            "agent_type": "executor", "worker_attempts": 0, "max_worker_attempts": 6,
            "context": '{"generating_model": "test-model"}',
        }
        mock_retry_ctx = MagicMock()
        mock_retry_ctx.record_failure.return_value = MagicMock(action="immediate", delay_seconds=0)
        mock_retry_ctx.to_context_patch.return_value = {}
        mock_retry_ctx.to_db_fields.return_value = {"worker_attempts": 1}
        mock_retry_ctx.grade_attempts = 0
        mock_retry_ctx.next_retry_at = None
        mock_from_task.return_value = mock_retry_ctx
        verdict = GradeResult(passed=False)

        await apply_grade_result(52, verdict)

        mock_trans.assert_called_once()
        args, kwargs = mock_trans.call_args
        assert args == (52, "pending")


class TestGradeTaskAutoFail:
    @pytest.mark.asyncio
    async def test_empty_result_auto_fails(self):
        from src.core.grading import grade_task
        task = {"title": "Test", "description": "Test", "result": "", "context": "{}"}
        result = await grade_task(task, "test-model")
        assert result.passed is False
        assert "auto-fail" in result.raw

    @pytest.mark.asyncio
    async def test_short_result_auto_fails(self):
        from src.core.grading import grade_task
        task = {"title": "Test", "description": "Test", "result": "ok", "context": "{}"}
        result = await grade_task(task, "test-model")
        assert result.passed is False
        assert "auto-fail" in result.raw
