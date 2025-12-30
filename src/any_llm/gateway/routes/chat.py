import base64
import json
import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from any_llm import AnyLLM, LLMProvider, acompletion
from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.auth.vertex_auth import setup_vertex_environment
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, ModelPricing, SessionToken, UsageLog, User, get_db
from any_llm.gateway.log_config import logger
# for caret
from any_llm.gateway.routes.utils import (
    _get_cached_prompt_tokens,
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage

router = APIRouter(prefix="/v1/chat", tags=["chat"])


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[dict[str, Any]]
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


_IMAGE_DUMP_DIR_ENV = "CARET_IMAGE_DUMP_DIR"
_IMAGE_EXTENSION_BY_MIME = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/svg+xml": "svg",
    "image/heic": "heic",
    "image/heif": "heif",
    "image/avif": "avif",
}


def _image_extension(mime_type: str) -> str:
    normalized = mime_type.strip().lower()
    if normalized in _IMAGE_EXTENSION_BY_MIME:
        return _IMAGE_EXTENSION_BY_MIME[normalized]
    if "/" in normalized:
        extension = normalized.split("/", 1)[1]
        return extension.replace("+xml", "").replace("svg", "svg")
    return "bin"


def _get_image_dump_dir() -> Path | None:
    directory = os.getenv(_IMAGE_DUMP_DIR_ENV)
    if not directory:
        return None
    path = Path(directory).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Failed to create image dump directory %s: %s", path, str(exc))
        return None
    return path


def _dump_image(image_bytes: bytes, mime_type: str, dump_dir: Path, request_id: str, index: int) -> None:
    extension = _image_extension(mime_type)
    filename = f"{request_id}_{index:02d}.{extension}"
    file_path = dump_dir / filename
    try:
        file_path.write_bytes(image_bytes)
        logger.info("Saved uploaded image to %s", file_path)
    except Exception as exc:
        logger.warning("Failed to save uploaded image to %s: %s", file_path, str(exc))


def _normalize_data_url(value: str) -> tuple[str, str, bytes]:
    if not value.startswith("data:"):
        raise ValueError("image_url.url must be a data URL or a remote URL")

    header, base64_payload = value.split(",", 1) if "," in value else ("", "")
    if not header or ";base64" not in header:
        raise ValueError("image_url.url must be a base64-encoded data URL")

    mime_type = header[5:].split(";", 1)[0]
    if not mime_type.startswith("image/"):
        raise ValueError("image_url.url must be an image data URL")

    sanitized_payload = "".join(base64_payload.split())
    if not sanitized_payload:
        raise ValueError("image_url.url must contain base64 data")

    try:
        image_bytes = base64.b64decode(sanitized_payload, validate=True)
    except Exception as exc:
        raise ValueError("image_url.url contains invalid base64 data") from exc

    return f"{header},{sanitized_payload}", mime_type, image_bytes


def _normalize_images_in_messages(
    messages: list[dict[str, Any]],
    request_id: str,
) -> tuple[list[dict[str, Any]], int]:
    normalized_messages: list[dict[str, Any]] = []
    image_count = 0
    dump_dir = _get_image_dump_dir()

    for message in messages:
        normalized_message = dict(message)
        content = message.get("content")

        if isinstance(content, list):
            normalized_content = []
            for block in content:
                if block.get("type") == "image_url":
                    image_url = block.get("image_url")
                    if not isinstance(image_url, dict):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="image_url must be an object with a url field",
                        )

                    url = image_url.get("url")
                    if not isinstance(url, str) or not url:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="image_url.url must be a non-empty string",
                        )

                    if url.startswith("data:"):
                        try:
                            normalized_url, mime_type, image_bytes = _normalize_data_url(url)
                        except ValueError as exc:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=str(exc),
                            ) from exc
                        url = normalized_url
                        image_count += 1
                        if dump_dir is not None:
                            _dump_image(image_bytes, mime_type, dump_dir, request_id, image_count)
                    else:
                        image_count += 1

                    normalized_content.append({**block, "image_url": {**image_url, "url": url}})
                else:
                    normalized_content.append(block)

            normalized_message["content"] = normalized_content

        normalized_messages.append(normalized_message)

    return normalized_messages, image_count


def _get_provider_credentials(
    config: GatewayConfig,
    provider: LLMProvider,
) -> dict[str, Any]:
    """Get provider credentials from config.

    Args:
        config: Gateway configuration
        provider: Provider name

    Returns:
        Dictionary of provider credentials

    Raises:
        HTTPException: If credentials not found

    """
    credentials: dict[str, Any] = {}
    if provider.value in config.providers:
        provider_config = config.providers[provider.value]

        if provider == LLMProvider.VERTEXAI:
            vertex_creds = provider_config.get("credentials")
            vertex_project = provider_config.get("project")
            vertex_location = provider_config.get("location")

            setup_vertex_environment(
                credentials=vertex_creds,
                project=vertex_project,
                location=vertex_location,
            )
        else:
            credentials = provider_config

    return credentials


def _build_model_key(provider: str | LLMProvider | None, model: str) -> str:
    provider_value = provider.value if isinstance(provider, LLMProvider) else provider
    return f"{provider_value}:{model}" if provider_value else model


def _get_model_pricing(
    db: Session,
    provider: str | LLMProvider | None,
    model: str,
) -> tuple[str, ModelPricing | None]:
    """Resolve model key and fetch pricing once for reuse."""
    model_key = _build_model_key(provider, model)
    pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()
    return model_key, pricing


def _calculate_usage_cost(usage_data: CompletionUsage, pricing: ModelPricing) -> float:
    total_prompt_tokens = usage_data.prompt_tokens or 0
    cached_prompt_tokens = _get_cached_prompt_tokens(usage_data) or 0
    cached_prompt_tokens = min(max(cached_prompt_tokens, 0), total_prompt_tokens)
    standard_prompt_tokens = total_prompt_tokens - cached_prompt_tokens
    output_tokens = usage_data.completion_tokens or 0
    cached_price = pricing.cached_price_per_million
    if cached_price is None:
        cached_price = pricing.input_price_per_million

    return (
        (standard_prompt_tokens / 1_000_000) * pricing.input_price_per_million
        + (cached_prompt_tokens / 1_000_000) * cached_price
        + (output_tokens / 1_000_000) * pricing.output_price_per_million
    )


def _maybe_attach_cost_to_usage(
    usage: CompletionUsage | None,
    pricing: ModelPricing | None,
) -> CompletionUsage | None:
    """Fill missing usage.cost when pricing is available and tokens are present."""
    if not usage or not pricing:
        return usage

    if getattr(usage, "cost", None) is not None:
        return usage

    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    if prompt_tokens == 0 and completion_tokens == 0:
        return usage

    cost = _calculate_usage_cost(usage, pricing)
    try:
        usage.cost = cost  # type: ignore[attr-defined]
        return usage
    except Exception:
        try:
            return usage.model_copy(update={"cost": cost})
        except Exception:
            logger.debug("Failed to attach cost to usage data; continuing without cost.")
            return usage


async def _log_usage(
    db: Session,
    api_key_obj: APIKey | None,
    model: str,
    provider: str | LLMProvider | None,
    endpoint: str,
    user_id: str | None = None,
    response: ChatCompletion | AsyncIterator[ChatCompletionChunk] | None = None,
    usage_override: CompletionUsage | None = None,
    error: str | None = None,
    model_key: str | None = None,
    model_pricing: ModelPricing | None = None,
) -> str | None:
    """Log API usage to database and update user spend.

    Args:
        db: Database session
        api_key_obj: API key object (None if using master key)
        model: Model name
        provider: Provider name
        endpoint: Endpoint path
        user_id: User identifier for tracking
        response: Response object (if successful)
        usage_override: Usage data for streaming requests
        model_key: Precomputed model key for pricing lookup
        model_pricing: Pre-fetched pricing to avoid repeated DB queries
        error: Error message (if failed)

    """
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

    usage_data = usage_override
    if not usage_data and response and isinstance(response, ChatCompletion) and response.usage:
        usage_data = response.usage

    if usage_data:
        usage_log.prompt_tokens = usage_data.prompt_tokens
        usage_log.completion_tokens = usage_data.completion_tokens
        usage_log.total_tokens = usage_data.total_tokens
        usage_log.cached_tokens = getattr(usage_data, "cached_tokens", 0) or 0

        resolved_model_key = model_key or _build_model_key(provider, model)
        pricing = model_pricing
        if pricing is None:
            _, pricing = _get_model_pricing(db, provider, model)

        if pricing:
            cost = _calculate_usage_cost(usage_data, pricing)
            usage_log.cost = cost

            if user_id:
                user = db.query(User).filter(User.user_id == user_id).first()
                if user:
                    user.spend = float(user.spend) + cost
        else:
            logger.info(f"No pricing configured for model '{resolved_model_key}'. Usage will be tracked without cost.")

    db.add(usage_log)
    try:
        db.commit()
        return usage_log.id
    except Exception as e:
        logger.error(f"Failed to log usage to database: {e}")
        db.rollback()
        return None


@router.post("/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    Handles reasoning content from any-llm providers.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        request.user,
        missing_master_detail="When using master key, 'user' field is required in request body",
    )
    # TODO: caret
    # _ = await validate_user_budget(db, user_id)
    # for caret
    validate_user_credit(db, user_id)

    provider, model = AnyLLM.split_model_provider(request.model)
    model_key, model_pricing = _get_model_pricing(db, provider, model)
    credentials = _get_provider_credentials(config, provider)

    dump_request_id = f"chat_{uuid.uuid4().hex}"
    normalized_messages, image_count = _normalize_images_in_messages(request.messages, dump_request_id)
    completion_kwargs = request.model_dump()
    completion_kwargs["messages"] = normalized_messages
    completion_kwargs.update(credentials)

    logger.info(
        "chat completion request model=%s stream=%s messages=%d images=%d",
        request.model,
        request.stream,
        len(request.messages),
        image_count,
    )

    try:
        if request.stream:

            async def generate() -> AsyncIterator[str]:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                cached_tokens = 0
                cached_tokens_seen = False
                saw_finish_reason = False

                try:
                    stream: AsyncIterator[ChatCompletionChunk] = await acompletion(**completion_kwargs)  # type: ignore[assignment]
                    async for chunk in stream:
                        if chunk.usage:
                            chunk.usage = _maybe_attach_cost_to_usage(chunk.usage, model_pricing)

                        logger.info("Chunk: %s", chunk)
                        if chunk.usage:
                            # Prompt tokens should be constant, take first non-zero value
                            if chunk.usage.prompt_tokens and not prompt_tokens:
                                prompt_tokens = chunk.usage.prompt_tokens
                            if chunk.usage.completion_tokens:
                                completion_tokens = max(completion_tokens, chunk.usage.completion_tokens)
                            if chunk.usage.total_tokens:
                                total_tokens = max(total_tokens, chunk.usage.total_tokens)
                            cached_tokens_value = _get_cached_prompt_tokens(chunk.usage)
                            if cached_tokens_value is not None:
                                cached_tokens_seen = True
                                cached_tokens = max(cached_tokens, cached_tokens_value or 0)

                        if chunk.choices and any(choice.finish_reason for choice in chunk.choices):
                            saw_finish_reason = True

                        yield f"data: {chunk.model_dump_json()}\n\n"
                        if saw_finish_reason:
                            break
                    yield "data: [DONE]\n\n"

                    # Log aggregated usage
                    if prompt_tokens or completion_tokens or total_tokens or cached_tokens_seen:
                        usage_data = CompletionUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                        )
                        if cached_tokens_seen:
                            try:
                                usage_data.cached_tokens = cached_tokens  # type: ignore[attr-defined]
                            except Exception:
                                try:
                                    usage_data = usage_data.model_copy(update={"cached_tokens": cached_tokens})
                                except Exception:
                                    pass
                        # for caret
                        logger.info("Usage data: %s", json.dumps(usage_data.model_dump()))
                        usage_log_id = await _log_usage(
                            db=db,
                            api_key_obj=api_key,
                            model=model,
                            provider=provider,
                            endpoint="/v1/chat/completions",
                            user_id=user_id,
                            usage_override=usage_data,
                            model_key=model_key,
                            model_pricing=model_pricing,
                        )
                        charge_usage_cost(
                            db,
                            user_id=user_id,
                            usage=usage_data,
                            model_key=model_key,
                            usage_id=usage_log_id,
                        )
                    else:
                        # This should never happen.
                        logger.warning(f"No usage data received from streaming response for model {model}")
                except Exception as e:
                    await _log_usage(
                        db=db,
                        api_key_obj=api_key,
                        model=model,
                        provider=provider,
                        endpoint="/v1/chat/completions",
                        user_id=user_id,
                        model_key=model_key,
                        model_pricing=model_pricing,
                        error=str(e),
                    )
                    raise

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response: ChatCompletion = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        # for caret
        usage_log_id = await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            user_id=user_id,
            response=response,
            model_key=model_key,
            model_pricing=model_pricing,
        )
        charge_usage_cost(
            db,
            user_id=user_id,
            usage=response.usage,
            model_key=model_key,
            usage_id=usage_log_id,
        )

    except Exception as e:
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            user_id=user_id,
            model_key=model_key,
            model_pricing=model_pricing,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calling provider: {e!s}",
        ) from e
    return response
