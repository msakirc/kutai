# vision.py — image analysis via vision-capable models

import base64, os, time as _time, uuid as _uuid
from src.infra.logging_config import get_logger
logger = get_logger("tools.vision")


def _encode_image(filepath: str) -> tuple[str, str]:
    """Return (media_type, base64_data) for a single image file."""
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                  "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return media_type, data


async def analyze_image(filepaths: "list[str] | str", question: str = "Describe what you see in this image.") -> str:
    """Analyze one or more image files using a vision-capable model.

    Parameters
    ----------
    filepaths:
        A single file path string (backward-compatible) or a list of paths.
        When given a list, all images are included in a single vision call
        with one text block followed by one image block per image.
    question:
        The question / prompt to send alongside the image(s).
    """
    # Normalise to list
    if isinstance(filepaths, str):
        path_list = [filepaths]
    else:
        path_list = list(filepaths)

    # Validate all paths exist
    for filepath in path_list:
        if not os.path.exists(filepath):
            return f"Error: file not found: {filepath}"

    try:
        logger.info("analyzing image(s)", count=len(path_list), first=path_list[0])

        import general_beckman
        from src.core.llm_dispatcher import _task_result_to_request_response

        # Build content: one text block then one image block per file
        content: list[dict] = [{"type": "text", "text": question}]
        for filepath in path_list:
            media_type, data = _encode_image(filepath)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })

        messages = [{"role": "user", "content": content}]

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
        logger.error("vision analysis failed", filepaths=path_list, error=str(e))
        return f"Error analyzing image: {e}"
