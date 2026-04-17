import pytest
from src.core.result_router import route_result, Complete, SpawnSubtasks, RequestClarification, RequestReview, Exhausted, Failed


def test_complete_result_produces_complete_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "complete", "result": "done", "iterations": 3}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], Complete)
    assert actions[0].result == "done"


def test_subtasks_result_produces_spawn_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "subtasks", "subtasks": [{"title": "s1"}, {"title": "s2"}]}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], SpawnSubtasks)
    assert len(actions[0].subtasks) == 2


def test_clarification_result_produces_clarification_action():
    task = {"id": 1, "title": "t", "chat_id": 42}
    agent_result = {"status": "clarification", "question": "which one?"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], RequestClarification)
    assert actions[0].question == "which one?"


def test_review_result_produces_review_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "review", "summary": "please review"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], RequestReview)


def test_exhausted_result_produces_exhausted_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "exhausted", "error": "max iterations"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], Exhausted)


def test_failed_result_produces_failed_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "failed", "error": "llm timeout"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], Failed)
    assert actions[0].error == "llm timeout"


def test_unknown_status_treated_as_failed():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "???"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], Failed)


def test_none_result_treated_as_failed():
    task = {"id": 1, "title": "t"}
    actions = route_result(task, None)
    assert len(actions) == 1
    assert isinstance(actions[0], Failed)
