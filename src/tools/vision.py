# vision.py — image analysis via vision-capable models

import base64, os
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

        from src.models.model_registry import get_registry
        # Pick first vision-capable model available
        registry = get_registry()
        vision_model = next(
            (m for m in registry.models.values() if m.has_vision and not m.demoted),
            None
        )
        if not vision_model:
            return "Error: no vision-capable model available"

        logger.info("analyzing image", filepath=filepath, model=vision_model.name)

        # Route through dispatcher as MAIN_WORK — vision is real task work,
        # not overhead. Using model_override to pin the vision-capable model.
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        from src.core.router import ModelRequirements

        reqs = ModelRequirements(
            task="vision",
            difficulty=4,
            priority=5,
            estimated_input_tokens=1500,  # image tokens
            estimated_output_tokens=500,
            model_override=vision_model.litellm_name,
        )
        messages = [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
        ]}]

        result = await get_dispatcher().request(
            category=CallCategory.MAIN_WORK,
            reqs=reqs,
            messages=messages,
        )
        analysis = result.get("content", "")
        from content_quality import assess as cq_assess
        _vis_cq = cq_assess(analysis)
        if _vis_cq.is_degenerate:
            logger.warning("vision analysis degenerate", summary=_vis_cq.summary)
            return f"Error: vision analysis produced degenerate output ({_vis_cq.summary})"
        return analysis
    except Exception as e:
        logger.error("vision analysis failed", filepath=filepath, error=str(e))
        return f"Error analyzing image: {e}"
