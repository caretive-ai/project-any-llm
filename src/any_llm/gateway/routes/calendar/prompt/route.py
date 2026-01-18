"""Calendar prompt generation route handler."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
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
    get_response_text,
)

from .schema import DEFAULT_MODEL, GeneratePromptRequest, GeneratePromptResponse

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

router = APIRouter(prefix="/v1/calendar", tags=["calendar"])


def build_prompt(month: int) -> str:
    """Build the prompt for generating calendar image suggestions."""
    is_odd = month % 2 != 0
    position = "right side" if is_odd else "left side"

    return f"""Generate 3 distinct image generation prompts for a 2026 calendar background for the month of {month}.

1. "default": A beautiful and atmospheric landscape of South Korea that matches the season of month {month}. This prompt must be written in KOREAN.
2. "anime_female": An anime style digital illustration of 1 beautiful girl standing on the {position} of the frame. The atmosphere and clothing should match month {month}. Must start with "digital illustration anime style". This prompt must be written in ENGLISH.
3. "anime_male": An anime style digital illustration of 1 handsome boy standing on the {position} of the frame. The atmosphere and clothing should match month {month}. Must start with "digital illustration anime style". This prompt must be written in ENGLISH.

Return the result as a JSON object with keys: "default", "anime_female", and "anime_male"."""


def parse_prompt_response(text: str | None) -> GeneratePromptResponse | None:
    """Parse the response text into GeneratePromptResponse."""
    if not text:
        return None
    try:
        data = json.loads(text)
        return GeneratePromptResponse(
            default=data.get("default", ""),
            anime_female=data.get("anime_female", ""),
            anime_male=data.get("anime_male", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse prompt response: %s", exc)
        return None


def build_fallback_prompts(month: int) -> GeneratePromptResponse:
    """Build fallback prompts when generation fails."""
    is_odd = month % 2 != 0
    position = "right side of the frame" if is_odd else "left side of the frame"

    season_map = {
        1: "겨울",
        2: "겨울",
        3: "봄",
        4: "봄",
        5: "봄",
        6: "여름",
        7: "여름",
        8: "여름",
        9: "가을",
        10: "가을",
        11: "가을",
        12: "겨울",
    }
    season = season_map.get(month, "봄")

    return GeneratePromptResponse(
        default=f"한국의 아름다운 {season} 풍경, {month}월의 계절감이 느껴지는 자연 배경",
        anime_female=f"digital illustration anime style, 1 beautiful girl standing on the {position}, {season} atmosphere, looking at viewer, high quality",
        anime_male=f"digital illustration anime style, 1 handsome boy standing on the {position}, {season} atmosphere, looking at viewer, high quality",
    )


@router.post("/prompt", response_model=GeneratePromptResponse)
async def generate_calendar_prompt(
    request: GeneratePromptRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> JSONResponse | GeneratePromptResponse:
    """Generate calendar image prompt suggestions for a given month."""
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=request.user,
        missing_master_detail="When using master key, 'user' field is required in request body",
    )
    validate_user_credit(db, user_id)

    month = request.month
    fallback = build_fallback_prompts(month)

    logger.info(
        "calendar.prompt request month=%s user=%s model=%s",
        month,
        user_id,
        DEFAULT_MODEL,
    )

    ensure_genai_available()
    assert genai is not None

    client = create_genai_client(config)
    model_input = DEFAULT_MODEL
    provider_name = "gemini"
    model_key, _ = _get_model_pricing(db, provider_name, model_input)

    prompt = build_prompt(month)

    try:
        response = client.models.generate_content(
            model=model_input,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "default": {"type": "STRING"},
                        "anime_female": {"type": "STRING"},
                        "anime_male": {"type": "STRING"},
                    },
                    "required": ["default", "anime_female", "anime_male"],
                },
            ),
        )

        usage_info = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        usage_for_charge = _coerce_usage_metadata(usage_info) or usage_info
        usage_log_id = _log_image_usage(
            db=db,
            api_key_obj=api_key,
            model=model_input,
            provider=provider_name,
            endpoint="/v1/calendar/prompt",
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

        text = get_response_text(response)
        parsed = parse_prompt_response(text)

        if not parsed:
            logger.error("calendar.prompt invalid response schema month=%s user=%s", month, user_id)
            return fallback

        logger.info(
            "calendar.prompt success month=%s user=%s defaultLength=%d",
            month,
            user_id,
            len(parsed.default),
        )
        return parsed

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "calendar.prompt failed month=%s user=%s error=%s",
            month,
            user_id,
            exc,
        )
        return fallback
