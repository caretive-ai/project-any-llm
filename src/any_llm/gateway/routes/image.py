import base64
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, SessionToken, get_db
from any_llm.gateway.routes.utils import resolve_target_user, validate_user_credit
from any_llm.gateway.log_config import logger

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


router = APIRouter(prefix="/v1/generate", tags=["generate"])


class GenerateImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    model: str | None = None
    aspect_ratio: str | None = None
    image_size: str | None = None


class GenerateImageResponse(BaseModel):
    mimeType: str
    base64: str
    texts: list[str] = Field(default_factory=list)
    thoughts: list[str] = Field(default_factory=list)


def _get_gemini_api_key(config: GatewayConfig) -> str:
    provider_cfg = config.providers.get("gemini", {})
    api_key = provider_cfg.get("api_key")
    if not api_key or not isinstance(api_key, str):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini provider api_key is not configured on the gateway",
        )
    return api_key


@router.post("/image", response_model=GenerateImageResponse)
async def generate_image(
    request: GenerateImageRequest,
    auth_result: Annotated[
        tuple[APIKey | None, bool, str | None, SessionToken | None],
        Depends(verify_jwt_or_api_key_or_master),
    ],
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GenerateImageResponse:
    """Generate a single image and return it as base64."""
    _user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, use chat endpoints to specify 'user' or use an access token",
    )
    validate_user_credit(db, _user_id)

    logger.info("Generating image for user %s", request)

    if genai is None:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai dependency is not installed",
        )

    api_key = _get_gemini_api_key(config)
    model_id = request.model or "gemini-3-pro-image-preview"

    try:
        client = genai.Client(api_key=api_key)
        aspect_ratio = request.aspect_ratio or "16:9"
        image_size = request.image_size or "4K"
        image_config = genai.types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        )

        config_kwargs: dict[str, object] = {
            "response_modalities": ["Text", "Image"],
            "image_config": image_config,
        }

        config_kwargs["tools"] = [{"google_search": {}}]
        config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
            include_thoughts=True
        )
        resp = client.models.generate_content(
            model=model_id,
            contents=[request.prompt],
            config=genai.types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Image generation failed: {e!s}",
        ) from e

    parts = getattr(resp, "parts", None) or []
    texts: list[str] = []
    thoughts: list[str] = []
    image_bytes: bytes | None = None
    mime_type: str = "image/png"
    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value:
            if getattr(part, "thought", False):
                thoughts.append(text_value)
            else:
                texts.append(text_value)

        inline_data = getattr(part, "inline_data", None)
        data = getattr(inline_data, "data", None) if inline_data is not None else None
        candidate_mime_type = getattr(inline_data, "mime_type", None) if inline_data is not None else None
        logger.info(
            "image part summary: has_text=%s thought=%s has_inline=%s mime_type=%s data_len=%s",
            bool(text_value),
            bool(getattr(part, "thought", False)),
            bool(inline_data),
            candidate_mime_type,
            len(data) if isinstance(data, (bytes, bytearray)) else None,
        )
        if not data:
            continue
        if isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, bytes):
            continue
        if isinstance(candidate_mime_type, str) and candidate_mime_type.startswith("image/"):
            mime_type = candidate_mime_type
        image_bytes = data
        break

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Image generation returned no image parts",
        )

    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return GenerateImageResponse(mimeType=mime_type, base64=base64_str, texts=texts, thoughts=thoughts)
