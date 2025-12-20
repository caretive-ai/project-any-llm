import base64
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, SessionToken, UsageLog, User, get_db
from any_llm.gateway.routes.utils import (
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)
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


def _build_model_key(provider: str | None, model: str) -> str:
    if provider:
        return f"{provider}:{model}"
    return model


def _log_image_usage(
    db: Session,
    api_key_obj: APIKey | None,
    model: str,
    provider: str | None,
    endpoint: str,
    user_id: str | None,
    usage: Any | None,
    error: str | None = None,
) -> str | None:
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_obj.id if api_key_obj else None,
        user_id=user_id,
        timestamp=datetime.now(UTC).replace(tzinfo=None),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
    )

    if usage:
        usage_log.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        thought_tokens = getattr(usage, "thought_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        if completion_tokens or thought_tokens:
            usage_log.completion_tokens = completion_tokens + thought_tokens
        usage_log.total_tokens = getattr(usage, "total_tokens", 0) or 0
        usage_log.cached_tokens = 0

    db.add(usage_log)
    try:
        db.commit()
        return usage_log.id
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to log image usage: %s", str(exc))
        db.rollback()
        return None


def _set_usage_cost(db: Session, usage_log_id: str | None, cost: float) -> None:
    if not usage_log_id:
        return
    try:
        db.query(UsageLog).filter(UsageLog.id == usage_log_id).update({"cost": cost})
        db.commit()
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to update image usage cost: %s", str(exc))


def _add_user_spend(db: Session, user_id: str | None, amount: float) -> None:
    if not user_id or amount <= 0:
        return
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if user:
            user.spend = float(user.spend) + amount
            db.add(user)
            db.commit()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to update user spend for image usage: %s", str(exc))


def _coerce_usage_metadata(usage: Any | None) -> Any | None:
    if not usage:
        return None
    if getattr(usage, "prompt_tokens", None) is not None:
        return usage

    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage, "total_token_count", 0) or 0
    thought_tokens = getattr(usage, "thoughts_token_count", 0) or 0

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return usage

    total_with_thoughts = total_tokens
    if total_with_thoughts is None:
        total_with_thoughts = (prompt_tokens or 0) + (completion_tokens or 0) + (thought_tokens or 0)
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_with_thoughts or None,
        thought_tokens=thought_tokens,
    )


def _get_usage_numeric(metadata: Any, *attrs: str) -> int | None:
    for attr in attrs:
        value = getattr(metadata, attr, None)
        if isinstance(value, (int, float)):
            return int(value)
    return None


class UsageAccumulator:
    def __init__(self) -> None:
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None
        self.thought_tokens: int = 0
        self._last_thought_tokens: int = 0

    def record(self, metadata: Any | None) -> None:
        if not metadata:
            return

        prompt = _get_usage_numeric(metadata, "prompt_token_count", "prompt_tokens")
        if prompt is not None:
            self.prompt_tokens = max(self.prompt_tokens or 0, prompt)

        completion = _get_usage_numeric(metadata, "candidates_token_count", "completion_tokens")
        if completion is not None:
            self.completion_tokens = max(self.completion_tokens or 0, completion)

        total = _get_usage_numeric(metadata, "total_token_count", "total_tokens")
        if total is not None:
            self.total_tokens = max(self.total_tokens or 0, total)

        thought = _get_usage_numeric(metadata, "thoughts_token_count")
        if thought is not None:
            delta = max(0, thought - self._last_thought_tokens)
            self.thought_tokens += delta
            self._last_thought_tokens = max(self._last_thought_tokens, thought)

    def finalize(self) -> Any | None:
        completion_with_thought = (self.completion_tokens or 0) + self.thought_tokens
        total_with_thought = self.total_tokens
        if total_with_thought is None:
            total_with_thought = (self.prompt_tokens or 0) + completion_with_thought

        if self.prompt_tokens is None and completion_with_thought == 0 and total_with_thought is None:
            return None

        return SimpleNamespace(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=completion_with_thought or None,
            total_tokens=total_with_thought or None,
        )


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
    api_key_obj = auth_result[0]
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
    provider_name = "gemini"
    model_key = _build_model_key(provider_name, model_id)

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
            usage_accumulator = UsageAccumulator()
            stream_error: str | None = None
            usage_log_id: str | None = None
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
                    chunk_usage = getattr(chunk, "usage_metadata", None) or getattr(chunk, "usage", None)
                    logger.info("chunk_usage: %s", chunk_usage)
                    if chunk_usage:
                        logger.info(
                            "stream chunk %d usage: %s",
                            chunk_index,
                            json.dumps(
                                {
                                    "prompt_tokens": getattr(chunk_usage, "prompt_token_count", None)
                                    or getattr(chunk_usage, "prompt_tokens", None),
                                    "completion_tokens": getattr(chunk_usage, "candidates_token_count", None)
                                    or getattr(chunk_usage, "completion_tokens", None),
                                    "thought_tokens": getattr(chunk_usage, "thoughts_token_count", None),
                                    "total_tokens": getattr(chunk_usage, "total_token_count", None)
                                    or getattr(chunk_usage, "total_tokens", None),
                                }
                            ),
                        )
                        usage_accumulator.record(chunk_usage)
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
                stream_error = str(e)
                logger.error("image stream failed: %s", stream_error)
                yield _format_sse_event({"type": "error", "message": stream_error})
            finally:
                usage_info_stream = usage_accumulator.finalize()
                usage_log_id = _log_image_usage(
                    db=db,
                    api_key_obj=api_key_obj,
                    model=model_id,
                    provider=provider_name,
                    endpoint="/v1/generate/image",
                    user_id=_user_id,
                    usage=usage_info_stream,
                    error=stream_error,
                )
                if usage_info_stream:
                    cost = charge_usage_cost(
                        db,
                        user_id=_user_id,
                        usage=usage_info_stream,
                        model_key=model_key,
                        usage_id=usage_log_id,
                    )
                    _set_usage_cost(db, usage_log_id, cost)
                    _add_user_spend(db, _user_id, cost)
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
        usage_info = getattr(resp, "usage_metadata", None)
        if usage_info:
            logger.info(
                "image request usage: %s",
                json.dumps(
                    {
                        "prompt_tokens": getattr(usage_info, "prompt_token_count", None),
                        "completion_tokens": getattr(usage_info, "candidates_token_count", None),
                        "thought_tokens": getattr(usage_info, "thoughts_token_count", None),
                        "total_tokens": getattr(usage_info, "total_token_count", None),
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
    usage_info = getattr(resp, "usage", None)
    usage_for_charge = _coerce_usage_metadata(usage_info) or usage_info
    usage_log_id = _log_image_usage(
        db=db,
        api_key_obj=api_key_obj,
        model=model_id,
        provider=provider_name,
        endpoint="/v1/generate/image",
        user_id=_user_id,
        usage=usage_for_charge,
    )
    if usage_for_charge:
        cost = charge_usage_cost(
            db,
            user_id=_user_id,
            usage=usage_for_charge,
            model_key=model_key,
            usage_id=usage_log_id,
        )
        _set_usage_cost(db, usage_log_id, cost)
        _add_user_spend(db, _user_id, cost)
    return GenerateImageResponse(mimeType=mime_type, base64=base64_str, texts=texts, thoughts=thoughts)
