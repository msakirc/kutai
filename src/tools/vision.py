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

        logger.info("analyzing image", filepath=filepath)

        # Route through dispatcher as MAIN_WORK — Fatih Hoca picks the
        # best vision-capable model via needs_vision=True.
        from src.core.llm_dispatcher import get_dispatcher, CallCategory

        messages = [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
        ]}]

        result = await get_dispatcher().request(
            category=CallCategory.MAIN_WORK,
            task="visual_reviewer",
            difficulty=4,
            messages=messages,
            priority=5,
            estimated_input_tokens=1500,
            estimated_output_tokens=500,
            needs_vision=True,
        )
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
