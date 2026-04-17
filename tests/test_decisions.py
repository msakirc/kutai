from src.core.decisions import Dispatch, NotifyUser, GateDecision, Allow, Block, Cancel


def test_dispatch_carries_task_and_executor():
    d = Dispatch(task_id=42, executor="llm", payload={"agent_type": "executor"})
    assert d.task_id == 42
    assert d.executor == "llm"
    assert d.payload["agent_type"] == "executor"


def test_notify_user_carries_chat_and_text():
    n = NotifyUser(chat_id=1001, text="done")
    assert n.chat_id == 1001
    assert n.text == "done"


def test_gate_decision_allow_has_no_reason():
    g: GateDecision = Allow()
    assert isinstance(g, Allow)


def test_gate_decision_block_carries_reason():
    g: GateDecision = Block(reason="awaiting_approval")
    assert isinstance(g, Block)
    assert g.reason == "awaiting_approval"


def test_gate_decision_cancel_carries_reason():
    g: GateDecision = Cancel(reason="risk_rejected")
    assert isinstance(g, Cancel)
    assert g.reason == "risk_rejected"
