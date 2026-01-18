"""Calendar image generation route handler."""
from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, SessionToken, get_db
from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.chat import _get_model_pricing
from any_llm.gateway.routes.image import (
    _add_user_spend,
    _coerce_usage_metadata,
    _log_image_usage,
    _set_usage_cost,
)
from any_llm.gateway.routes.utils import (
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)
from any_llm.gateway.routes.webtoon.genai_helper import (
    create_genai_client,
    ensure_genai_available,
)

from .schema import (
    DEFAULT_MODEL,
    IMAGE_SIZE,
    AspectRatioType,
    GenerateImageRequest,
    GenerateImageResponse,
    LayoutType,
)

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

router = APIRouter(prefix="/v1/calendar", tags=["calendar"])


def get_layout_config(layout: LayoutType) -> tuple[AspectRatioType, str]:
    """Get aspect ratio and description for a given layout."""
    if layout == "wide_16_9":
        return "16:9", "2:1 ultra-wide panorama"
    if layout == "wide_16_9_half_left":
        return "1:1", "8:9 portrait-oriented square"
    return "16:9", "16:9 landscape"


def build_image_prompt(prompt: str, ratio_desc: str) -> str:
    """Build the final image generation prompt."""
    return f"""Best quality, masterpiece, ultra-high resolution, 8k, extremely detailed, cinematic lighting, photorealistic or high-end illustration suitable for a calendar background.
Subject: {prompt}.
Composition: Optimized for {ratio_desc} layout, no text, clean focus."""


@router.post("/image", response_model=GenerateImageResponse)
async def generate_calendar_image(
    request: GenerateImageRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> JSONResponse | GenerateImageResponse:
    """Generate a calendar background image."""
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=request.user,
        missing_master_detail="When using master key, 'user' field is required in request body",
    )
    validate_user_credit(db, user_id)

    prompt = request.prompt.strip()
    layout = request.layout

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt is required",
        )

    aspect_ratio, ratio_desc = get_layout_config(layout)

    logger.info(
        "calendar.image request layout=%s aspectRatio=%s user=%s model=%s promptLength=%d prompt=%s",
        layout,
        aspect_ratio,
        user_id,
        DEFAULT_MODEL,
        len(prompt),
        prompt[:100] + "..." if len(prompt) > 100 else prompt,
    )

    ensure_genai_available()
    assert genai is not None

    client = create_genai_client(config)
    model_input = DEFAULT_MODEL
    provider_name = "gemini"
    model_key, _ = _get_model_pricing(db, provider_name, model_input)

    final_prompt = build_image_prompt(prompt, ratio_desc)

    try:
        response = client.models.generate_content(
            model=model_input,
            contents={"parts": [{"text": final_prompt}]},
            config={
                "image_config": {
                    "aspect_ratio": aspect_ratio,
                    "image_size": IMAGE_SIZE,
                },
            },
        )

        usage_info = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        usage_for_charge = _coerce_usage_metadata(usage_info) or usage_info
        usage_log_id = _log_image_usage(
            db=db,
            api_key_obj=api_key,
            model=model_input,
            provider=provider_name,
            endpoint="/v1/calendar/image",
            user_id=user_id,
            usage=usage_for_charge,
        )

        if usage_for_charge:
            cost = charge_usage_cost(
                db,
                user_id=user_id,
                usage=usage_for_charge,
                model_key=model_key,
                usage_id=usage_log_id,
            )
            _set_usage_cost(db, usage_log_id, cost)
            _add_user_spend(db, user_id, cost)

        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []

            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data:
                    data = getattr(inline_data, "data", None)
                    mime_type = getattr(inline_data, "mime_type", "image/png")

                    if data:
                        if isinstance(data, bytes):
                            b64_data = base64.b64encode(data).decode("utf-8")
                        else:
                            b64_data = data

                        logger.info(
                            "calendar.image success layout=%s user=%s mimeType=%s imageSize=%d",
                            layout,
                            user_id,
                            mime_type,
                            len(b64_data),
                        )
                        return GenerateImageResponse(
                            image=f"data:{mime_type};base64,{b64_data}"
                        )

        logger.error(
            "calendar.image no image data layout=%s user=%s",
            layout,
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No image data found in response",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "calendar.image failed layout=%s user=%s error=%s",
            layout,
            user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate image",
        ) from exc
