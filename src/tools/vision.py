# vision.py — image analysis via vision-capable models

import base64, os, time as _time, uuid as _uuid
from src.infra.logging_config import get_logger
logger = get_logger("tools.vision")

async def analyze_image(filepath: str, question: str = "Describe what you see in this image.") -> str:
    """Analyze an image file using a vision-capable model."""
    if not os.path.exists(filepath):
        return f"Error: file not found: {filepath}"
    try:
        with open(filepath, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(filepath)[1].lower().lstrip(".")
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                      "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")

        logger.info("analyzing image", filepath=filepath)

        import general_beckman
        from src.core.llm_dispatcher import _task_result_to_request_response

        messages = [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
        ]}]

        # Resolve parent_id from the orchestrator's per-task ContextVar.
        # Vision tool is always invoked from within an agent's tool execution,
        # so current_task_id is set by the orchestrator at dispatch time.
        _parent_id = None
        try:
            from src.core.heartbeat import current_task_id as _ctid
            _parent_id = _ctid.get()
        except Exception:
            pass

        _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
        spec = {
            "title": f"vision:analyze:{_suffix}",
            "description": "Vision analysis of image file",
            "agent_type": "visual_reviewer",
            "kind": "tool_call",
            "priority": 5,
            "context": {
                "llm_call": {
                    "raw_dispatch": True,
                    "call_category": "main_work",
                    "task": "visual_reviewer",
                    "agent_type": "visual_reviewer",
                    "difficulty": 4,
                    "messages": messages,
                    "failures": [],
                    "estimated_input_tokens": 1500,
                    "estimated_output_tokens": 500,
                    "needs_vision": True,
                },
            },
        }
        task_result = await general_beckman.enqueue(
            spec,
            parent_id=_parent_id,
            await_inline=True,
        )

        if task_result.status == "failed":
            logger.warning("vision enqueue failed", error=task_result.error)
            return f"Error: vision call failed ({task_result.error})"

        result = _task_result_to_request_response(task_result)
        analysis = result.get("content", "")
        from dogru_mu_samet import assess as cq_assess
        _vis_cq = cq_assess(analysis)
        if _vis_cq.is_degenerate:
            logger.warning("vision analysis degenerate", summary=_vis_cq.summary)
            return f"Error: vision analysis produced degenerate output ({_vis_cq.summary})"
        return analysis
    except Exception as e:
        logger.error("vision analysis failed", filepath=filepath, error=str(e))
        return f"Error analyzing image: {e}"
