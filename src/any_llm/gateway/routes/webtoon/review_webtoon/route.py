from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
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

from ..genai_helper import (
    create_genai_client,
    generate_multimodal_content,
    generate_text_content,
    get_response_text,
)
from .parser import clean_text, parse_json
from .prompt import build_panel_context, build_prompt
from .schema import DEFAULT_MODEL, PanelReviewEntry, ReviewWebtoonRequest, ReviewWebtoonResponse

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


def _extract_images_for_review(request: ReviewWebtoonRequest) -> list[dict[str, str]]:
    """Extract images from request for multimodal review."""
    images: list[dict[str, str]] = []

    # Extract panel images
    if request.panels:
        for panel in request.panels:
            if panel.imageData and panel.imageMimeType:
                images.append({
                    "data": panel.imageData,
                    "mime_type": panel.imageMimeType,
                })

    # Extract character images
    if request.characters:
        for char in request.characters:
            if char.imageData and char.imageMimeType:
                images.append({
                    "data": char.imageData,
                    "mime_type": char.imageMimeType,
                })

    return images


def _count_image_bytes(images: list[dict[str, str]]) -> int:
    """Calculate approximate size of images in bytes."""
    total = 0
    for img in images:
        if img.get("data"):
            # Base64 is ~4/3 of original size
            total += len(img["data"]) * 3 // 4
    return total


def _build_log_metadata(request: ReviewWebtoonRequest, images: list[dict[str, str]]) -> dict[str, Any]:
    """Build metadata for logging (without actual image data)."""
    panels = request.panels or []
    characters = request.characters or []

    panel_info = []
    for p in panels:
        panel_info.append({
            "panel": p.panel or p.panelNumber,
            "hasScene": bool(p.scene),
            "hasSpeaker": bool(p.speaker),
            "hasDialogue": bool(p.dialogue),
            "hasMetadata": bool(p.metadata),
            "hasImage": bool(p.imageData),
            "imageMimeType": p.imageMimeType if p.imageData else None,
        })

    character_info = []
    for c in characters:
        character_info.append({
            "name": c.name,
            "hasDescription": bool(c.description),
            "hasMetadata": bool(c.metadata),
            "hasImage": bool(c.imageData),
            "imageMimeType": c.imageMimeType if c.imageData else None,
        })

    return {
        "topic": request.topic,
        "genre": request.genre,
        "style": request.style,
        "era": request.era,
        "season": request.season,
        "panelCount": len(panels),
        "characterCount": len(characters),
        "totalImageCount": len(images),
        "panelImageCount": sum(1 for p in panels if p.imageData),
        "characterImageCount": sum(1 for c in characters if c.imageData),
        "totalImageBytes": _count_image_bytes(images),
        "hasScriptSummary": bool(request.scriptSummary),
        "panels": panel_info,
        "characters": character_info,
    }


@router.post("/review-webtoon", response_model=ReviewWebtoonResponse)
async def review_webtoon(
    request: ReviewWebtoonRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> ReviewWebtoonResponse:
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    panels = request.panels or []
    characters = request.characters or []

    # Build panel context for prompt
    panel_context_list = []
    for panel in panels:
        index = panel.panel or panel.panelNumber or ""
        scene = panel.scene or ""
        speaker = panel.speaker or ""
        dialogue = panel.dialogue or ""
        metadata_info = f"\n   [이미지 메타: {panel.metadata}]" if panel.metadata else ""
        has_image_marker = " [이미지 첨부됨]" if panel.imageData else ""
        panel_context_list.append(
            f"패널 {index}:{has_image_marker}\n   장면: {scene}\n   화자: {speaker}\n   대사: {dialogue}{metadata_info}"
        )

    # Extract images for multimodal
    images = _extract_images_for_review(request)
    has_panel_images = any(p.imageData for p in panels)
    has_character_images = any(c.imageData for c in characters)

    panel_context = build_panel_context([ctx for ctx in panel_context_list if ctx.strip()])
    prompt_system, prompt_user = build_prompt(
        topic=request.topic or "",
        genre=request.genre or "",
        style=request.style or "",
        era=request.era,
        season=request.season,
        characters=request.characters,
        script_summary=request.scriptSummary or "",
        panel_context=panel_context,
        has_panel_images=has_panel_images,
        has_character_images=has_character_images,
    )

    model_input = DEFAULT_MODEL
    provider_name = "gemini"
    model_key, _ = _get_model_pricing(db, provider_name, model_input)

    client = create_genai_client(config)

    try:
        # Build log metadata
        log_metadata = _build_log_metadata(request, images)

        logger.info(
            "webtoon.review-webtoon request topic=%s genre=%s style=%s era=%s season=%s "
            "panelCount=%s characterCount=%s totalImageCount=%s panelImageCount=%s "
            "characterImageCount=%s totalImageBytes=%s hasScriptSummary=%s",
            log_metadata["topic"],
            log_metadata["genre"],
            log_metadata["style"],
            log_metadata["era"],
            log_metadata["season"],
            log_metadata["panelCount"],
            log_metadata["characterCount"],
            log_metadata["totalImageCount"],
            log_metadata["panelImageCount"],
            log_metadata["characterImageCount"],
            log_metadata["totalImageBytes"],
            log_metadata["hasScriptSummary"],
        )

        # Log panel details
        for panel_log in log_metadata["panels"]:
            logger.info(
                "webtoon.review-webtoon panel=%s hasScene=%s hasSpeaker=%s hasDialogue=%s "
                "hasMetadata=%s hasImage=%s imageMimeType=%s",
                panel_log["panel"],
                panel_log["hasScene"],
                panel_log["hasSpeaker"],
                panel_log["hasDialogue"],
                panel_log["hasMetadata"],
                panel_log["hasImage"],
                panel_log["imageMimeType"],
            )

        # Log character details
        for char_log in log_metadata["characters"]:
            logger.info(
                "webtoon.review-webtoon character=%s hasDescription=%s hasMetadata=%s "
                "hasImage=%s imageMimeType=%s",
                char_log["name"],
                char_log["hasDescription"],
                char_log["hasMetadata"],
                char_log["hasImage"],
                char_log["imageMimeType"],
            )

        # Choose generation method based on images
        if images:
            logger.info(
                "webtoon.review-webtoon using multimodal generation with %d images",
                len(images),
            )
            response = generate_multimodal_content(
                client,
                model_input,
                prompt_system,
                prompt_user,
                images,
            )
        else:
            logger.info("webtoon.review-webtoon using text-only generation")
            response = generate_text_content(
                client,
                model_input,
                prompt_system,
                prompt_user,
            )

        usage_info = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        usage_for_charge = _coerce_usage_metadata(usage_info) or usage_info

        # Log usage info
        if usage_for_charge:
            logger.info(
                "webtoon.review-webtoon usage promptTokens=%s candidatesTokens=%s totalTokens=%s",
                getattr(usage_for_charge, "prompt_token_count", None),
                getattr(usage_for_charge, "candidates_token_count", None),
                getattr(usage_for_charge, "total_token_count", None),
            )

        usage_log_id = _log_image_usage(
            db=db,
            api_key_obj=api_key,
            model=model_input,
            provider=provider_name,
            endpoint="/v1/webtoon/review-webtoon",
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
        if not text:
            logger.error("webtoon.review-webtoon empty response from AI")
            raise HTTPException(status_code=502, detail="Empty response from AI")

        cleaned = clean_text(text)
        parsed = parse_json(cleaned)
        if not parsed:
            logger.error("webtoon.review-webtoon invalid JSON response: %s", cleaned[:500] if cleaned else "empty")
            raise HTTPException(status_code=502, detail="Invalid response structure")

        try:
            result = ReviewWebtoonResponse.model_validate(parsed)
            logger.info(
                "webtoon.review-webtoon success headline=%s overallScore=%s strengthsCount=%s improvementsCount=%s",
                result.headline[:30] if result.headline else None,
                result.overallScore.overall if result.overallScore else None,
                len(result.strengths) if result.strengths else 0,
                len(result.improvements) if result.improvements else 0,
            )
            return result
        except Exception as exc:
            logger.warning("webtoon.review-webtoon validation failed: %s, parsed=%s", exc, str(parsed)[:500])
            raise HTTPException(status_code=502, detail="Invalid response structure")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("webtoon.review-webtoon failed: %s", exc)
        raise HTTPException(status_code=502, detail="Review generation failed") from exc
