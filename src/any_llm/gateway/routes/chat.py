import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
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
from any_llm.gateway.db import APIKey, ModelPricing, SessionToken, UsageLog, User, get_db, CreditBalance
from any_llm.gateway.log_config import logger
# for caret
from any_llm.gateway.routes.utils import (
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
    prompt_tokens = usage_data.prompt_tokens or 0
    completion_tokens = usage_data.completion_tokens or 0
    return (prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
        completion_tokens / 1_000_000
    ) * pricing.output_price_per_million


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
            logger.warning(f"No pricing configured for model '{resolved_model_key}'. Usage will be tracked without cost.")

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

    completion_kwargs = request.model_dump()
    completion_kwargs.update(credentials)

    try:
        if request.stream:

            async def generate() -> AsyncIterator[str]:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

                try:
                    stream: AsyncIterator[ChatCompletionChunk] = await acompletion(**completion_kwargs)  # type: ignore[assignment]
                    async for chunk in stream:
                        if chunk.usage:
                            chunk.usage = _maybe_attach_cost_to_usage(chunk.usage, model_pricing)
                            # Prompt tokens should be constant, take first non-zero value
                            if chunk.usage.prompt_tokens and not prompt_tokens:
                                prompt_tokens = chunk.usage.prompt_tokens
                            if chunk.usage.completion_tokens:
                                completion_tokens = max(completion_tokens, chunk.usage.completion_tokens)
                            if chunk.usage.total_tokens:
                                total_tokens = max(total_tokens, chunk.usage.total_tokens)

                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                    # Log aggregated usage
                    if prompt_tokens or completion_tokens or total_tokens:
                        usage_data = CompletionUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                        )
                        # for caret
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

            return StreamingResponse(generate(), media_type="text/event-stream")

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
