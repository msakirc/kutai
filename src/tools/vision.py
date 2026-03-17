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

        import litellm
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
        response = await litellm.acompletion(
            model=vision_model.litellm_name,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
            ]}],
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("vision analysis failed", filepath=filepath, error=str(e))
        return f"Error analyzing image: {e}"
