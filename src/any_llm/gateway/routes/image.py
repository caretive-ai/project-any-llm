import base64
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
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
    stream: bool = False


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
) -> GenerateImageResponse | StreamingResponse:
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
            "candidate_count": 1,
        }

        # config_kwargs["tools"] = [{"google_search": {}}]
        config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
            include_thoughts=True
        )
        content_config = genai.types.GenerateContentConfig(**config_kwargs)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Image generation failed: {e!s}",
        ) from e

    def _sanitize_for_logging(value: str | None) -> str | None:
        if not value:
            return None
        return value.replace("[", "\\[").replace("]", "\\]")

    def _iter_parts(chunk) -> list[object]:
        parts = getattr(chunk, "parts", None)
        if parts:
            return parts
        candidates = getattr(chunk, "candidates", None)
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                return parts
        return []

    def _format_sse_event(payload: dict[str, object]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    if request.stream:
        try:
            logger.info("Using generate_content_stream")
            stream = client.models.generate_content_stream(
                model=model_id,
                contents=[request.prompt],
                config=content_config,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Image generation failed: {e!s}",
            ) from e

        def event_stream(client_ref=client, stream_ref=stream):
            try:
                chunk_index = 0
                for chunk in stream_ref:
                    chunk_index += 1
                    parts = _iter_parts(chunk)
                    logger.info(
                        "stream chunk %d received with %d parts (prompt_len=%d)",
                        chunk_index,
                        len(parts),
                        len(request.prompt),
                    )
                    chunk_usage = getattr(chunk, "usage_metadata", None)
                    logger.info("chunk_usage: %s", chunk_usage)
                    if chunk_usage:
                        logger.info(
                            "stream chunk %d usage: %s",
                            chunk_index,
                            json.dumps(
                                {
                                    "prompt_tokens": getattr(chunk_usage, "prompt_token_count", None),
                                    "completion_tokens": getattr(chunk_usage, "candidates_token_count", None),
                                    "thought_tokens": getattr(chunk_usage, "thoughts_token_count", None),
                                    "cached_tokens": getattr(chunk_usage, "cached_content_token_count", None),
                                    "total_tokens": getattr(chunk_usage, "total_token_count", None),
                                }
                            ),
                        )
                    for part in parts:
                        text_value = getattr(part, "text", None)
                        text_snippet = _sanitize_for_logging(text_value)[:120] if isinstance(text_value, str) else None
                        if isinstance(text_value, str) and text_value:
                            logger.info(
                                "stream part chunk=%d thought=%s text_snippet=%s",
                                chunk_index,
                                bool(getattr(part, "thought", False)),
                                text_snippet,
                            )
                            if getattr(part, "thought", False):
                                yield _format_sse_event(
                                    {"type": "thought", "content": text_value}
                                )
                            else:
                                yield _format_sse_event(
                                    {"type": "text", "content": text_value}
                                )

                        inline_data = getattr(part, "inline_data", None)
                        data = (
                            getattr(inline_data, "data", None)
                            if inline_data is not None
                            else None
                        )
                        candidate_mime_type = (
                            getattr(inline_data, "mime_type", None)
                            if inline_data is not None
                            else None
                        )
                        data_length = len(data) if isinstance(data, (bytes, bytearray)) else None
                        logger.info(
                            "stream inline chunk=%d mime=%s data_len=%s",
                            chunk_index,
                            candidate_mime_type,
                            data_length,
                        )
                        if not data:
                            continue
                        if isinstance(data, bytearray):
                            data = bytes(data)
                        if not isinstance(data, bytes):
                            continue
                        if (
                            isinstance(candidate_mime_type, str)
                            and candidate_mime_type.startswith("image/")
                        ):
                            mime_type = candidate_mime_type
                        else:
                            mime_type = "image/png"
                        base64_str = base64.b64encode(data).decode("utf-8")
                        yield _format_sse_event(
                            {
                                "type": "image",
                                "mimeType": mime_type,
                                "base64": base64_str,
                            }
                        )
                logger.info("image stream completed after %d chunk(s)", chunk_index)
                yield _format_sse_event({"type": "done"})
            except Exception as e:
                logger.error("image stream failed: %s", str(e))
                yield _format_sse_event({"type": "error", "message": str(e)})
            finally:
                try:
                    client_ref.close()
                except Exception:
                    pass

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=[request.prompt],
            config=content_config,
        )
        usage_info = getattr(resp, "usage", None)
        if usage_info:
            logger.info(
                "image request usage: %s",
                json.dumps(
                    {
                        "prompt_tokens": getattr(usage_info, "prompt_tokens", None),
                        "completion_tokens": getattr(usage_info, "completion_tokens", None),
                        "total_tokens": getattr(usage_info, "total_tokens", None),
                    }
                ),
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
