from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import hashlib
from typing import Any, AsyncGenerator, Callable, Coroutine

from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
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
    _create_inline_part,
    _get_gemini_api_key,
    _log_image_usage,
    _set_usage_cost,
)
from any_llm.gateway.routes.utils import (
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)

from .parser import parse_json
from .prompt import resolve_era_label, resolve_season_label
from .schema import (
    AnalysisLevelType,
    AspectRatioType,
    PanelImageResponse,
    PanelRequest,
    ResolutionType,
)

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])

IMAGE_CACHE: dict[str, tuple[PanelImageResponse, float]] = {}
MAX_CACHE_ENTRIES = 50
CACHE_TTL_SECONDS = 5 * 60

DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_RESOLUTION: ResolutionType = "1K"
DEFAULT_ASPECT_RATIO: AspectRatioType = "1:1"
PROMPT_VERSION = "scene-dialogue-v1"

RESOLUTION_LONG_EDGE: dict[str, int] = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}


def _parse_aspect_ratio(value: AspectRatioType) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        return (1, 1)
    try:
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            return (1, 1)
        return (width, height)
    except ValueError:
        return (1, 1)


def _resolve_target_size(
    resolution: ResolutionType, aspect_ratio: AspectRatioType
) -> tuple[int, int]:
    long_edge = RESOLUTION_LONG_EDGE.get(resolution, RESOLUTION_LONG_EDGE["1K"])
    ratio_w, ratio_h = _parse_aspect_ratio(aspect_ratio)
    if ratio_w >= ratio_h:
        width = long_edge
        height = max(1, round((long_edge * ratio_h) / ratio_w))
    else:
        width = max(1, round((long_edge * ratio_w) / ratio_h))
        height = long_edge
    return (width, height)


def _normalize_image(
    image_bytes: bytes,
    resolution: ResolutionType,
    aspect_ratio: AspectRatioType,
) -> tuple[bytes, str]:
    """Normalize image to target resolution and convert to WebP (lossless for webtoon quality)."""
    try:
        width, height = _resolve_target_size(resolution, aspect_ratio)
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGBA")
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            # WebP lossless: 화질 100% 유지, PNG 대비 20-30% 용량 절감
            img.save(output, format="WEBP", lossless=True)
            return output.getvalue(), "image/webp"
    except Exception as exc:
        logger.warning("Failed to normalize panel image size: %s", exc)
        return image_bytes, "image/png"

STYLE_PROMPTS: dict[str, dict[str, str]] = {
    "webtoon": {
        "id": "webtoon",
        "name": "웹툰 스타일",
        "systemPrompt": "당신은 한국 웹툰 전문 이미지 생성 AI입니다. 네이버 웹툰, 카카오웹툰 스타일의 깔끔한 선화와 디지털 채색을 사용합니다.",
        "imagePrompt": "Korean webtoon style, clean line art, digital coloring, vibrant colors, modern illustration, professional manhwa art, detailed character design, smooth shading, clear outlines, anatomically correct hands and limbs",
        "negativePrompt": "messy lines, sketch style, rough draft, watercolor, oil painting, realistic photo, 3D render, blurry, low quality, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "manga": {
        "id": "manga",
        "name": "만화 스타일",
        "systemPrompt": "당신은 일본 만화 전문 이미지 생성 AI입니다. 세밀한 스크린톤, 다양한 선의 강약, 감정 표현이 풍부한 일본 망가 스타일을 사용합니다.",
        "imagePrompt": "Japanese manga style, detailed screen tones, expressive line work, dynamic composition, black and white ink art, shounen/shoujo manga aesthetic, detailed backgrounds, varied line weights, anatomically correct hands and limbs",
        "negativePrompt": "colored, full color, digital painting, western comic style, realistic, photo, 3D, blurry, low detail, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "cartoon": {
        "id": "cartoon",
        "name": "카툰 스타일",
        "systemPrompt": "당신은 귀여운 카툰 캐릭터 전문 이미지 생성 AI입니다. 단순하고 둥근 형태, 밝고 선명한 색상, 친근하고 귀여운 느낌의 캐릭터를 만듭니다.",
        "imagePrompt": "Cute cartoon style, simple rounded shapes, bright vibrant colors, friendly character design, kawaii aesthetic, chibi proportions, clean flat colors, playful illustration, anatomically correct hands and limbs",
        "negativePrompt": "realistic, detailed, complex, dark, gritty, serious, photo-realistic, 3D render, sketch, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "illustration": {
        "id": "illustration",
        "name": "일러스트",
        "systemPrompt": "당신은 감성적인 일러스트 전문 이미지 생성 AI입니다. 디테일한 표현, 부드러운 빛과 그림자, 감정이 느껴지는 색감을 사용합니다.",
        "imagePrompt": "Detailed digital illustration, soft lighting, emotional atmosphere, painterly style, artistic composition, rich textures, sophisticated color palette, professional book illustration quality, anatomically correct hands and limbs",
        "negativePrompt": "simple, cartoon, chibi, low detail, flat colors, sketch, rough, unfinished, photo, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "realistic": {
        "id": "realistic",
        "name": "실사 스타일",
        "systemPrompt": "당신은 실사 스타일 이미지 생성 AI입니다. 사진처럼 사실적인 질감, 자연스러운 조명, 현실감 있는 표현을 사용합니다.",
        "imagePrompt": "Photorealistic style, realistic textures, natural lighting, high detail photography, professional photo quality, cinematic composition, realistic shadows and highlights, anatomically correct hands and limbs",
        "negativePrompt": "cartoon, anime, illustration, drawn, painted, sketch, abstract, stylized, flat, low quality, blurry, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "3d": {
        "id": "3d",
        "name": "3D 렌더링",
        "systemPrompt": "당신은 3D 렌더링 전문 이미지 생성 AI입니다. Pixar, Disney 스타일의 입체적이고 부드러운 3D 캐릭터와 환경을 만듭니다.",
        "imagePrompt": "3D rendered style, Pixar quality, smooth 3D models, professional rendering, volumetric lighting, detailed textures, Disney/Pixar aesthetic, clean 3D animation style, anatomically correct hands and limbs",
        "negativePrompt": "2D, flat, hand-drawn, sketch, photo, realistic, anime style, low poly, draft quality, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
}

SCENE_ELEMENT_KEYS = ["subject", "action", "setting", "composition", "lighting", "style"]
SCENE_ELEMENT_LABELS = {
    "subject": "Subject(주제)",
    "action": "Action(동작)",
    "setting": "Setting(환경)",
    "composition": "Composition(구성/카메라)",
    "lighting": "Lighting(조명)",
    "style": "Style(스타일)",
}

CARICATURE_PANEL_STYLE_GUIDE = """
- Professional 2D vector-like caricature style.
- Bold, clean outlines with simplified shapes.
- Soft cel-shading and smooth gradients; avoid painterly textures.
- Make the head and facial features noticeably larger than the torso; simplify limbs.
- Exaggerate facial proportions while keeping a friendly, charismatic expression.
- Maintain a consistent caricature identity across panels.
""".strip()

ERA_GUARDRAILS: dict[str, dict[str, list[str]]] = {
    "joseon": {
        "must": [
            "traditional wooden or stone architecture",
            "historical cooking tools (iron pot, brazier, earthenware)",
        ],
        "avoid": ["electric appliances", "cars", "glass skyscrapers", "smartphones", "modern signage"],
    },
    "nineties": {
        "must": ["CRT TV or bulky monitor", "landline phone or pager", "retro kitchen appliances"],
        "avoid": ["smartphones", "tablets", "flat-screen TVs", "wireless earbuds", "ultra-minimal interiors"],
    },
    "seventies-eighties": {
        "must": ["simple vintage interior", "analog appliances", "enamel pots or coal briquette tools"],
        "avoid": ["smartphones", "laptops", "flat-screen TVs", "induction stoves", "modern smart devices"],
    },
    "future": {
        "must": ["futuristic devices or interfaces", "sleek high-tech materials"],
        "avoid": ["purely antique tools", "old CRT TVs", "hand-cranked devices"],
    },
}


def _get_style_prompt(style_id: str) -> dict[str, str]:
    return STYLE_PROMPTS.get(
        style_id,
        {
            "id": "webtoon",
            "name": "웹툰 스타일",
            "systemPrompt": "당신은 웹툰 이미지 생성 AI입니다.",
            "imagePrompt": "Korean webtoon style, clean digital art",
            "negativePrompt": "low quality, blurry",
        },
    )


def _build_image_prompt(style_id: str, scene_description: str, character_descriptions: list[str]) -> str:
    style = _get_style_prompt(style_id)
    character_info = (
        f"\n\nCharacters in scene: {', '.join(character_descriptions)}" if character_descriptions else ""
    )
    combined_negative = f"{style['negativePrompt']}, {ANATOMICAL_NEGATIVE_PROMPT}"
    return (
        f"{style['imagePrompt']}\n\n"
        f"Scene: {scene_description}{character_info}\n\n"
        f"Style requirements: {style['systemPrompt']}\n\n"
        f"Negative prompt: {combined_negative}"
    )


def _normalize_scene_elements(value: dict[str, str] | None) -> dict[str, str]:
    def normalize_field(raw: str | None) -> str:
        if raw is None or raw == "null":
            return ""
        return raw

    value = value or {}
    return {
        "subject": normalize_field(value.get("subject")),
        "action": normalize_field(value.get("action")),
        "setting": normalize_field(value.get("setting")),
        "composition": normalize_field(value.get("composition")),
        "lighting": normalize_field(value.get("lighting")),
        "style": normalize_field(value.get("style")),
    }


def _has_scene_elements(elements: dict[str, str] | None) -> bool:
    if not elements:
        return False
    return any(elements.get(key, "").strip() for key in SCENE_ELEMENT_KEYS)


def _build_scene_summary(elements: dict[str, str] | None, fallback: str) -> str:
    if not elements:
        return fallback.strip()
    parts = [elements.get(key, "").strip() for key in SCENE_ELEMENT_KEYS]
    summary = " ".join([part for part in parts if part]).strip()
    return summary or fallback.strip()


def _format_scene_elements(elements: dict[str, str]) -> str:
    lines: list[str] = []
    for key in SCENE_ELEMENT_KEYS:
        value = elements.get(key, "").strip()
        if not value:
            continue
        label = SCENE_ELEMENT_LABELS.get(key, key)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _split_dialogue_lines(value: str | None) -> list[str]:
    if not value:
        return []
    return [line.strip() for line in re.split(r"\r?\n", value) if line.strip()]


def _coerce_string_array(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;]\s*", value) if item.strip()]
    return []


def _extract_json_from_text(text: str) -> Any | None:
    trimmed = text.strip()
    if not trimmed:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, re.IGNORECASE)
    candidate = (fenced.group(1) if fenced else trimmed).strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _parse_character_sheet_metadata(raw: Any) -> dict[str, Any] | None:
    if not raw:
        return None
    payload = _extract_json_from_text(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        return None
    return {
        "summary": payload.get("summary"),
        "persona": payload.get("persona"),
        "outfit": _coerce_string_array(payload.get("outfit")),
        "colors": _coerce_string_array(payload.get("colors")),
        "accessories": _coerce_string_array(payload.get("accessories")),
        "hair": payload.get("hair"),
        "face": payload.get("face"),
        "body": payload.get("body"),
        "props": _coerce_string_array(payload.get("props")),
        "shoes": _coerce_string_array(payload.get("shoes")),
        "notes": _coerce_string_array(payload.get("notes")),
    }


def _parse_panel_metadata(raw: str | None) -> dict[str, Any] | None:
    """Parse panel metadata with 4 key elements: characters, camera, environment, continuity."""
    if not raw:
        return None
    payload = _extract_json_from_text(raw)
    if not isinstance(payload, dict):
        return None

    # Parse new 4-element structure
    characters = []
    for item in payload.get("characters") or []:
        if not isinstance(item, dict):
            continue
        characters.append(
            {
                "name": item.get("name"),
                # New position/spatial elements
                "position": item.get("position"),  # left|center|right
                "facing": item.get("facing"),  # left|right|camera
                "expression": item.get("expression"),
                # Legacy elements for backward compatibility
                "outfit": item.get("outfit"),
                "accessories": _coerce_string_array(item.get("accessories")),
                "hair": item.get("hair"),
                "props": _coerce_string_array(item.get("props")),
                "pose": item.get("pose"),
                "notes": item.get("notes"),
            }
        )

    # Parse camera info
    camera_raw = payload.get("camera") or {}
    camera = {
        "shot_type": camera_raw.get("shot_type"),  # close-up|medium|wide
        "angle": camera_raw.get("angle"),  # eye-level|low|high
    } if isinstance(camera_raw, dict) else None

    # Parse environment info
    env_raw = payload.get("environment") or {}
    environment = {
        "location": env_raw.get("location"),
        "time_of_day": env_raw.get("time_of_day"),  # morning|afternoon|evening|night
        "weather": env_raw.get("weather"),  # sunny|cloudy|rainy|snowy
        "lighting": env_raw.get("lighting"),
    } if isinstance(env_raw, dict) else None

    # Parse continuity info
    cont_raw = payload.get("continuity") or {}
    continuity = {
        "key_objects": _coerce_string_array(cont_raw.get("key_objects")),
        "spatial_notes": _coerce_string_array(cont_raw.get("spatial_notes")),
    } if isinstance(cont_raw, dict) else None

    return {
        "summary": payload.get("summary"),
        "characters": characters or None,
        "camera": camera,
        "environment": environment,
        "continuity": continuity,
        # Legacy fields for backward compatibility
        "background": payload.get("background"),
        "lighting": payload.get("lighting"),
        "changes": _coerce_string_array(payload.get("changes")),
        "notes": _coerce_string_array(payload.get("notes")),
    }


def _parse_reference_metadata(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    payload = _extract_json_from_text(raw)
    if not isinstance(payload, dict):
        return None
    characters = []
    for item in payload.get("characters") or []:
        if not isinstance(item, dict):
            continue
        characters.append(
            {
                "name": item.get("name"),
                "outfit": item.get("outfit"),
                "accessories": _coerce_string_array(item.get("accessories")),
                "hair": item.get("hair"),
                "props": _coerce_string_array(item.get("props")),
                "pose": item.get("pose"),
                "notes": item.get("notes"),
            }
        )
    return {
        "summary": payload.get("summary"),
        "characters": characters or None,
        "background": payload.get("background"),
        "lighting": payload.get("lighting"),
        "notes": _coerce_string_array(payload.get("notes")),
    }


def _format_character_sheet_metadata(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    if metadata.get("summary"):
        parts.append(f"summary: {metadata['summary']}")
    if metadata.get("persona"):
        parts.append(f"persona: {metadata['persona']}")
    if metadata.get("outfit"):
        parts.append(f"outfit: {', '.join(metadata['outfit'])}")
    if metadata.get("colors"):
        parts.append(f"colors: {', '.join(metadata['colors'])}")
    if metadata.get("accessories"):
        parts.append(f"accessories: {', '.join(metadata['accessories'])}")
    if metadata.get("hair"):
        parts.append(f"hair: {metadata['hair']}")
    if metadata.get("face"):
        parts.append(f"face: {metadata['face']}")
    if metadata.get("body"):
        parts.append(f"body: {metadata['body']}")
    if metadata.get("props"):
        parts.append(f"props: {', '.join(metadata['props'])}")
    if metadata.get("shoes"):
        parts.append(f"shoes: {', '.join(metadata['shoes'])}")
    if metadata.get("notes"):
        parts.append(f"notes: {', '.join(metadata['notes'])}")
    return "; ".join([part for part in parts if part])


def _extract_key_visual_elements(metadata: dict[str, Any]) -> dict[str, str]:
    """Extract key visual elements that MUST remain consistent across panels."""
    return {
        "hair": metadata.get("hair") or "",
        "face": metadata.get("face") or "",
        "outfit": ", ".join(metadata.get("outfit") or []) if isinstance(metadata.get("outfit"), list) else (metadata.get("outfit") or ""),
        "colors": ", ".join(metadata.get("colors") or []) if isinstance(metadata.get("colors"), list) else (metadata.get("colors") or ""),
        "accessories": ", ".join(metadata.get("accessories") or []) if isinstance(metadata.get("accessories"), list) else (metadata.get("accessories") or ""),
    }


def _format_visual_identity_lock(name: str, elements: dict[str, str]) -> str:
    """Format key visual elements as a strict identity lock for a character."""
    lines: list[str] = []
    if elements.get("hair"):
        lines.append(f"  - HAIR (IMMUTABLE): {elements['hair']}")
    if elements.get("face"):
        lines.append(f"  - FACE (IMMUTABLE): {elements['face']}")
    if elements.get("outfit"):
        lines.append(f"  - OUTFIT (IMMUTABLE): {elements['outfit']}")
    if elements.get("colors"):
        lines.append(f"  - COLORS (IMMUTABLE): {elements['colors']}")
    if elements.get("accessories"):
        lines.append(f"  - ACCESSORIES (IMMUTABLE): {elements['accessories']}")
    if not lines:
        return ""
    return f"[{name}]\n" + "\n".join(lines)


def _format_panel_metadata(metadata: dict[str, Any]) -> str:
    """Format panel metadata with 4 key elements for continuity in subsequent panels."""
    parts: list[str] = []
    if metadata.get("summary"):
        parts.append(f"summary: {metadata['summary']}")

    # Format characters with position/spatial info (4-element structure)
    characters = metadata.get("characters") or []
    if characters:
        character_lines = []
        for character in characters:
            name = character.get("name") or ""
            detail_parts: list[str] = []
            # New position/spatial elements (priority for continuity)
            if character.get("position"):
                detail_parts.append(f"position: {character['position']}")
            if character.get("facing"):
                detail_parts.append(f"facing: {character['facing']}")
            if character.get("expression"):
                detail_parts.append(f"expression: {character['expression']}")
            # Legacy elements
            if character.get("outfit"):
                detail_parts.append(f"outfit: {character['outfit']}")
            if character.get("accessories"):
                detail_parts.append(f"accessories: {', '.join(character['accessories'])}")
            if character.get("hair"):
                detail_parts.append(f"hair: {character['hair']}")
            if character.get("props"):
                detail_parts.append(f"props: {', '.join(character['props'])}")
            if character.get("pose"):
                detail_parts.append(f"pose: {character['pose']}")
            if character.get("notes"):
                detail_parts.append(f"notes: {character['notes']}")
            if detail_parts:
                character_lines.append(f"{name} ({', '.join(detail_parts)})")
            else:
                character_lines.append(name)
        if character_lines:
            parts.append(f"characters: {' | '.join(character_lines)}")

    # Format camera info (new 4-element structure)
    camera = metadata.get("camera")
    if camera and isinstance(camera, dict):
        camera_parts = []
        if camera.get("shot_type"):
            camera_parts.append(f"shot: {camera['shot_type']}")
        if camera.get("angle"):
            camera_parts.append(f"angle: {camera['angle']}")
        if camera_parts:
            parts.append(f"camera: {', '.join(camera_parts)}")

    # Format environment info (new 4-element structure)
    environment = metadata.get("environment")
    if environment and isinstance(environment, dict):
        env_parts = []
        if environment.get("location"):
            env_parts.append(f"location: {environment['location']}")
        if environment.get("time_of_day"):
            env_parts.append(f"time: {environment['time_of_day']}")
        if environment.get("weather"):
            env_parts.append(f"weather: {environment['weather']}")
        if environment.get("lighting"):
            env_parts.append(f"lighting: {environment['lighting']}")
        if env_parts:
            parts.append(f"environment: {', '.join(env_parts)}")

    # Format continuity info (new 4-element structure)
    continuity = metadata.get("continuity")
    if continuity and isinstance(continuity, dict):
        if continuity.get("key_objects"):
            parts.append(f"key_objects: {', '.join(continuity['key_objects'])}")
        if continuity.get("spatial_notes"):
            parts.append(f"spatial: {', '.join(continuity['spatial_notes'])}")

    # Legacy fields for backward compatibility
    if metadata.get("background"):
        parts.append(f"background: {metadata['background']}")
    if metadata.get("lighting") and not (environment and environment.get("lighting")):
        parts.append(f"lighting: {metadata['lighting']}")
    if metadata.get("changes"):
        parts.append(f"changes: {', '.join(metadata['changes'])}")
    if metadata.get("notes"):
        parts.append(f"notes: {', '.join(metadata['notes'])}")

    return "; ".join([part for part in parts if part])


def _format_reference_metadata(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    if metadata.get("summary"):
        parts.append(f"summary: {metadata['summary']}")
    characters = metadata.get("characters") or []
    if characters:
        character_lines = []
        for idx, character in enumerate(characters):
            name = (character.get("name") or "").strip() or f"Unknown {idx + 1}"
            detail_parts: list[str] = []
            if character.get("outfit"):
                detail_parts.append(f"outfit {character['outfit']}")
            if character.get("accessories"):
                detail_parts.append(f"accessories {', '.join(character['accessories'])}")
            if character.get("hair"):
                detail_parts.append(f"hair {character['hair']}")
            if character.get("props"):
                detail_parts.append(f"props {', '.join(character['props'])}")
            if character.get("pose"):
                detail_parts.append(f"pose {character['pose']}")
            if character.get("notes"):
                detail_parts.append(f"notes {character['notes']}")
            if detail_parts:
                character_lines.append(f"{name} ({', '.join(detail_parts)})")
            else:
                character_lines.append(name)
        if character_lines:
            parts.append(f"characters: {' | '.join(character_lines)}")
    if metadata.get("background"):
        parts.append(f"background: {metadata['background']}")
    if metadata.get("lighting"):
        parts.append(f"lighting: {metadata['lighting']}")
    if metadata.get("notes"):
        parts.append(f"notes: {', '.join(metadata['notes'])}")
    return "; ".join([part for part in parts if part])


def _build_reference_metadata_prompt() -> str:
    return """
Return JSON ONLY (no markdown). Values must be in Korean.

Schema:
{
  "summary": "Common summary of the reference images",
  "characters": [
    {
      "name": "Character name or Unknown",
      "outfit": "Outfit description",
      "accessories": ["Accessories"],
      "hair": "Hair style/color",
      "props": ["Props/held items"],
      "pose": "Pose/posture",
      "notes": "Details to preserve"
    }
  ],
  "background": "Background/location",
  "lighting": "Lighting/time of day",
  "notes": ["Must-keep continuity elements"]
}

Rules:
- Only describe observable facts (no guesses).
- If multiple reference images exist, focus on shared details to preserve.
- Use empty string or empty array if unknown.
""".strip()


def _build_era_guardrails(era_id: str | None) -> str:
    if not era_id:
        return ""
    guardrails = ERA_GUARDRAILS.get(era_id)
    if not guardrails:
        return ""
    lines = ["## Era Guardrails", "- The era must be visually unambiguous in the background and props."]
    if guardrails.get("must"):
        lines.append(f"- Must include cues like: {', '.join(guardrails['must'])}")
    if guardrails.get("avoid"):
        lines.append(f"- Must avoid anachronisms such as: {', '.join(guardrails['avoid'])}")
    return "\n".join(lines)


ANATOMICAL_CONSTRAINTS = """
ANATOMICAL ACCURACY - ABSOLUTE REQUIREMENTS:
- Each human character must have exactly: 2 arms, 2 hands with 5 fingers each, 2 legs, 1 head.
- Never generate extra limbs, duplicate body parts, or fused appendages.
- Hands must be clearly separated and anatomically correct.
- If a character's hands are not visible in the scene, do not force them into frame.
- Avoid complex overlapping poses that may cause limb confusion.
""".strip()

ANATOMICAL_NEGATIVE_PROMPT = (
    "extra limbs, extra hands, extra arms, extra fingers, extra legs, "
    "mutated hands, fused fingers, too many fingers, malformed limbs, "
    "missing fingers, deformed hands, anatomical errors, body horror, "
    "duplicate body parts, conjoined limbs, twisted anatomy, "
    "unnatural pose, impossible anatomy, broken wrists"
)


def _build_image_system_instruction(
    *,
    character_generation_mode: str | None,
    has_references: bool,
    has_character_images: bool,
    has_background_references: bool,
) -> str:
    base = (
        "CRITICAL RULES - MUST FOLLOW: "
        "1) NEVER generate ANY text, letters, numbers, symbols, captions, titles, labels, speech balloons, or written content in the image. "
        "The image must be completely text-free. This is an absolute requirement with no exceptions. "
        "2) NEVER create panel divisions, split screens, comic panels, grid layouts, or multiple frames. "
        "Generate exactly ONE single continuous scene without any borders, dividers, or panel separations. "
        "3) ANATOMICAL ACCURACY: Each character must have exactly 2 arms, 2 hands (5 fingers each), 2 legs, and 1 head. "
        "Never generate extra limbs, duplicate body parts, mutated hands, or fused fingers. "
        "4) CHARACTER VISUAL IDENTITY - ABSOLUTE REQUIREMENT: "
        "The character sheet images provided are the ONLY source of truth for character appearance. "
        "You MUST match EXACTLY: hair color, hairstyle, face shape, outfit, clothing colors, and accessories. "
        "These visual elements are IMMUTABLE and must remain identical across all panels. "
        "Do NOT modify, reinterpret, or 'improve' any visual aspect of the character. "
        "Copy the exact appearance from the character sheet image - treat it as a strict visual reference. "
        "If there is ANY conflict between the scene description and the character sheet image, "
        "ALWAYS prioritize the character sheet image for appearance (hair, face, outfit, colors, accessories). "
        "Render only one instance per named character and never duplicate the speaker. "
        "5) METADATA OUTPUT - REQUIRED: After generating the image, you MUST also output a JSON block describing the generated scene. "
        "This metadata will be used for continuity in subsequent panels. Output format: "
        '```json\n{"characters":[{"name":"Name","position":"left|center|right","facing":"left|right|camera","expression":"expression"}],'
        '"camera":{"shot_type":"close-up|medium|wide","angle":"eye-level|low|high"},'
        '"environment":{"location":"specific location","time_of_day":"morning|afternoon|evening|night","weather":"sunny|cloudy|rainy|snowy","lighting":"description"},'
        '"continuity":{"key_objects":["object: position"],"spatial_notes":["important spatial info"]}}\n```'
    )
    if character_generation_mode == "caricature":
        base += " Keep the output clearly stylized and avoid photorealistic rendering or likeness to real people."
    if has_references:
        base += " Reference images are PRIMARY for character appearance - match them exactly."
    if not has_references and has_character_images:
        base += " Character sheet images are PRIMARY for character appearance - match them exactly. Study the image carefully and replicate the exact visual details."
    if has_background_references:
        base += " Background reference images are environment-only; ignore any people within them and do not let them override character appearance."
    return base


def _extract_text_from_parts(parts: list[Any]) -> str:
    fragments: list[str] = []
    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            fragments.append(text_value.strip())
    return "\n".join(fragments).strip()


def _get_response_parts(response: Any) -> list[Any]:
    parts = getattr(response, "parts", None) or []
    if parts:
        return parts
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        return getattr(content, "parts", None) or []
    return []


def _strip_data_url(value: str) -> tuple[str | None, str | None]:
    if not value.startswith("data:") or "," not in value:
        return None, None
    header, payload = value.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] if header.startswith("data:") else None
    return mime_type, payload


def _build_character_image_parts(character_images: list[str] | None) -> list[Any]:
    parts: list[Any] = []
    if not character_images:
        return parts
    for image_url in character_images:
        if not image_url:
            continue
        mime_type, payload = _strip_data_url(image_url)
        if not payload:
            continue
        try:
            image_bytes = base64.b64decode("".join(payload.split()), validate=True)
        except Exception:
            logger.warning("Invalid character sheet image payload, skipping")
            continue
        if not image_bytes:
            continue
        parts.append(_create_inline_part(image_bytes, mime_type or "image/png"))
    return parts


def _analyze_reference_images(
    client: Any,
    references: list[Any],
) -> str:
    if not references:
        return ""
    assert genai is not None
    parts = [genai.types.Part.from_text(text=_build_reference_metadata_prompt())]
    parts.extend(_build_reference_parts(references))
    contents = [genai.types.Content(role="user", parts=parts)]
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
    except Exception as exc:
        logger.warning("Failed to analyze reference images: %s", exc)
        return ""
    text = _extract_text_from_parts(_get_response_parts(response))
    if not text.strip():
        return ""
    parsed = _parse_reference_metadata(text)
    if parsed:
        return _format_reference_metadata(parsed)
    return text.strip()


def _analyze_era_consistency(
    client: Any,
    image_base64: str,
    mime_type: str,
    era_id: str | None,
    era_label: str | None,
    season_label: str | None,
) -> dict[str, Any]:
    if not era_label or not image_base64:
        return {"consistent": True, "issues": [], "fix": ""}
    assert genai is not None
    guardrails = ERA_GUARDRAILS.get(era_id or "")
    must_list = guardrails.get("must", []) if guardrails else []
    avoid_list = guardrails.get("avoid", []) if guardrails else []
    prompt = f"""You are a strict visual era consistency inspector.
Check whether the image matches the specified era and report any obvious anachronisms.

Era: {era_label}
Season: {season_label or "Any"}

Must include cues: {", ".join(must_list) if must_list else "None required"}
Must avoid: {", ".join(avoid_list) if avoid_list else "None specified"}

Rules:
- If any clear anachronism is visible, set consistent=false.
- If cues are missing but no anachronisms, you may set consistent=false if the era is not visually clear.
- Keep the response concise.

Return JSON only in this format:
{{
  "consistent": true,
  "issues": ["short issue list"],
  "fix": "short correction directive for regeneration"
}}"""
    try:
        image_bytes = base64.b64decode(image_base64)
        contents = [
            genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part.from_text(text=prompt),
                    _create_inline_part(image_bytes, mime_type),
                ],
            )
        ]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        text = _extract_text_from_parts(_get_response_parts(response))
        parsed = _extract_json_from_text(text)
        if isinstance(parsed, dict):
            return {
                "consistent": parsed.get("consistent", True),
                "issues": parsed.get("issues", []),
                "fix": parsed.get("fix", ""),
            }
    except Exception as exc:
        logger.warning("Era consistency check failed: %s", exc)
    return {"consistent": True, "issues": [], "fix": ""}


def _analyze_background_consistency(
    client: Any,
    image_base64: str,
    mime_type: str,
    background_base64: str,
    background_mime_type: str | None,
    scene: str | None,
    era_label: str | None,
    season_label: str | None,
) -> dict[str, Any]:
    if not background_base64 or not image_base64:
        return {"consistent": True, "issues": [], "fix": ""}
    assert genai is not None
    prompt = f"""You are a strict background continuity inspector.
Image A is the background reference (environment baseline).
Image B is the generated panel.

Goal: confirm both images depict the same location and background elements.
Allow camera angle changes and lighting shifts, but preserve major layout, architecture, and props.

Scene: {scene or "N/A"}
Era: {era_label or "Any"}
Season: {season_label or "Any"}

Rules:
- If the location looks different or major background cues are missing, set consistent=false.
- Ignore people or character appearance; focus only on environment/props.
- Keep the response concise.

Return JSON only in this format:
{{
  "consistent": true,
  "issues": ["short issue list"],
  "fix": "short correction directive for regeneration"
}}"""
    try:
        bg_bytes = base64.b64decode(background_base64)
        img_bytes = base64.b64decode(image_base64)
        contents = [
            genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part.from_text(text=prompt),
                    _create_inline_part(bg_bytes, background_mime_type or "image/png"),
                    _create_inline_part(img_bytes, mime_type),
                ],
            )
        ]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        text = _extract_text_from_parts(_get_response_parts(response))
        parsed = _extract_json_from_text(text)
        if isinstance(parsed, dict):
            return {
                "consistent": parsed.get("consistent", True),
                "issues": parsed.get("issues", []),
                "fix": parsed.get("fix", ""),
            }
    except Exception as exc:
        logger.warning("Background consistency check failed: %s", exc)
    return {"consistent": True, "issues": [], "fix": ""}


def _generate_panel_metadata(
    client: Any,
    panel_description: str,
    scene_elements_text: str,
    dialogue_lines: list[str],
    characters: list[str],
    character_descriptions: list[str] | None,
    character_sheet_metadata_lines: str,
    continuity_prompt: str,
) -> str:
    """Generate panel metadata with 4 key elements for visual continuity."""
    assert genai is not None
    prompt = f"""Return JSON ONLY (no markdown). Values must be in Korean.

Schema (4 Key Elements for Visual Continuity):
{{
  "summary": "One-sentence panel summary",
  "characters": [
    {{
      "name": "Character name",
      "position": "left|center|right (character's position in frame)",
      "facing": "left|right|camera (direction character is facing)",
      "expression": "Current facial expression",
      "outfit": "Outfit description",
      "accessories": ["Accessories"],
      "hair": "Hair style/color",
      "props": ["Props/held items"],
      "pose": "Pose/posture",
      "notes": "Details that must be preserved"
    }}
  ],
  "camera": {{
    "shot_type": "close-up|medium|wide",
    "angle": "eye-level|low|high"
  }},
  "environment": {{
    "location": "Specific location description",
    "time_of_day": "morning|afternoon|evening|night",
    "weather": "sunny|cloudy|rainy|snowy",
    "lighting": "Lighting description"
  }},
  "continuity": {{
    "key_objects": ["object: position (important objects and their positions)"],
    "spatial_notes": ["Important spatial relationships between elements"]
  }},
  "background": "Background/location (legacy)",
  "lighting": "Lighting/time of day (legacy)",
  "changes": ["Elements changed from the previous panel"],
  "notes": ["Must-keep continuity elements"]
}}

Rules:
- CRITICAL: Position and facing direction must be specified for EACH character.
- This metadata will be used to maintain visual consistency in subsequent panels.
- Only describe observable facts (no guesses).
- Use empty string or empty array if unknown.

Scene Description: {panel_description}
{f"Scene Elements:\\n{scene_elements_text}" if scene_elements_text else ""}
{f"Dialogue Cues:\\n" + "\\n".join([f"- {line}" for line in dialogue_lines]) if dialogue_lines else ""}
Characters: {", ".join(characters)}
{f"Character Details:\\n" + "\\n".join(character_descriptions) if character_descriptions else ""}
{f"Character Sheet Metadata:\\n{character_sheet_metadata_lines}" if character_sheet_metadata_lines else ""}
{continuity_prompt or ""}"""
    try:
        contents = [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=prompt)])]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        text = _extract_text_from_parts(_get_response_parts(response))
        parsed = _parse_panel_metadata(text)
        if parsed:
            return json.dumps(parsed, ensure_ascii=False)
        if text.strip():
            return json.dumps(
                {
                    "summary": text.strip(),
                    "characters": [],
                    "camera": None,
                    "environment": None,
                    "continuity": None,
                    "background": "",
                    "lighting": "",
                    "changes": [],
                    "notes": [],
                },
                ensure_ascii=False,
            )
    except Exception as exc:
        logger.warning("Failed to generate metadata summary: %s", exc)
    return ""


def _translate_prompt(client: Any, prompt: str) -> str:
    assert genai is not None
    if not prompt.strip():
        return prompt
    translation_prompt = (
        "You are a professional translator. Translate the following prompt into fluent English suitable for "
        "image generation without adding extra content.\n\n"
        f"{prompt}"
    )
    contents = [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=translation_prompt)])]
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
    except Exception as exc:
        logger.warning("Prompt translation failed: %s", exc)
        return prompt
    translated = _extract_text_from_parts(_get_response_parts(response))
    return translated.strip() or prompt


def _simplify_scene_to_keyframe(
    client: Any,
    scene_description: str,
    characters: list[str],
    dialogue: str | None,
) -> str:
    """
    Simplify a scene description to a single keyframe moment.
    Converts multi-action sequences into a single decisive moment with explicit hand positions.
    """
    assert genai is not None
    if not scene_description.strip():
        return scene_description

    character_list = ", ".join(characters) if characters else "the character"
    dialogue_context = f"\nDialogue context: {dialogue}" if dialogue else ""

    simplification_prompt = f"""You are an expert at converting complex scene descriptions into single-frame image generation prompts.

TASK: Convert the following scene description into a SINGLE DECISIVE MOMENT that can be captured in one static image.

RULES:
1. Choose only ONE moment from any action sequence (prefer the most visually interesting or emotionally significant moment)
2. Explicitly specify the position of BOTH hands for each character
3. Remove temporal words like "then", "after", "while", "as", "suddenly", "moment when"
4. Remove action transitions like "reaching for", "about to", "starting to" - show the completed state
5. If the scene describes "A하다가 B하는" (doing A then B), choose ONLY B (the final state)
6. Keep all other scene details (setting, lighting, mood, camera angle)

CHARACTER(S): {character_list}
{dialogue_context}

ORIGINAL SCENE:
{scene_description}

OUTPUT FORMAT:
Return ONLY the simplified scene description in the same language as the original. Include explicit hand positions for each character.

Example transformations:
- "진동에 놀라 휴대폰을 꺼내 확인하는" → "오른손에 휴대폰을 들고 화면을 바라보는, 왼손은 책상 위에 놓여있다"
- "커피를 마시다가 노트북을 보는" → "왼손에 커피잔을 들고 노트북 화면을 응시하는, 오른손은 키보드 위에 있다"
- "문을 열며 들어오는" → "열린 문 앞에 서있는, 오른손은 문손잡이를 잡고 있고 왼손은 가방을 들고 있다"

SIMPLIFIED SCENE:"""

    contents = [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=simplification_prompt)])]
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
    except Exception as exc:
        logger.warning("Scene simplification failed: %s", exc)
        return scene_description

    simplified = _extract_text_from_parts(_get_response_parts(response))
    return simplified.strip() or scene_description


def build_cache_key(
    body: PanelRequest,
    aspect_ratio: AspectRatioType,
    resolution: ResolutionType,
    analysis_level: AnalysisLevelType,
) -> str:
    hash_builder = hashlib.sha256()
    scene_elements_value = _normalize_scene_elements(body.sceneElements) if body.sceneElements is not None else None
    core_payload = {
        "scene": body.scene,
        "dialogue": body.dialogue or "",
        "characters": body.characters,
        "style": body.style,
        "panelNumber": body.panelNumber,
        "era": body.era or None,
        "season": body.season or None,
        "sceneElements": scene_elements_value,
        "characterDescriptions": body.characterDescriptions or [],
        "styleDoc": body.styleDoc or "",
        "previousPanels": body.previousPanels or [],
        "characterSheetMetadata": [
            entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
            for entry in body.characterSheetMetadata or []
        ],
        "characterGenerationMode": body.characterGenerationMode or None,
        "characterCaricatureStrengths": body.characterCaricatureStrengths or [],
        "revisionNote": body.revisionNote or "",
        "aspectRatio": aspect_ratio,
        "resolution": resolution,
        "analysisLevel": analysis_level,
        "promptVersion": PROMPT_VERSION,
    }
    hash_builder.update(json.dumps(core_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for image_url in body.characterImages or []:
        hash_builder.update(b"|character-image|")
        hash_builder.update((image_url or "").encode("utf-8"))
    for ref in body.references or []:
        hash_builder.update(b"|reference|")
        hash_builder.update((ref.base64 or "").encode("utf-8"))
        hash_builder.update((ref.mimeType or "").encode("utf-8"))
        hash_builder.update((ref.purpose or "").encode("utf-8"))
    return hash_builder.hexdigest()


def get_cached_panel_image(key: str) -> PanelImageResponse | None:
    entry = IMAGE_CACHE.get(key)
    if not entry:
        return None
    payload, expires = entry
    if expires < time.time():
        del IMAGE_CACHE[key]
        return None
    return payload


def set_cached_panel_image(key: str, payload: PanelImageResponse):
    IMAGE_CACHE[key] = (payload, time.time() + CACHE_TTL_SECONDS)
    if len(IMAGE_CACHE) > MAX_CACHE_ENTRIES:
        oldest_key = next(iter(IMAGE_CACHE))
        del IMAGE_CACHE[oldest_key]


def finalize_response(
    payload_text: str,
    inline_image_base64: str,
    mime_type: str,
    aspect_ratio: AspectRatioType,
    resolution: ResolutionType,
    panel_number: int,
) -> PanelImageResponse:
    return PanelImageResponse(
        success=True,
        imageUrl=f"data:{mime_type};base64,{inline_image_base64}",
        imageBase64=inline_image_base64,
        mimeType=mime_type,
        metadata=payload_text,
        text=payload_text,
        aspectRatio=aspect_ratio,
        resolution=resolution,
        model="gemini-3-pro-image-preview",
        panelNumber=panel_number,
    )


def _ensure_genai_available() -> None:
    if genai is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai dependency is not installed",
        )


def _parse_reference_payload(reference: str) -> tuple[str | None, str]:
    if reference.startswith("data:") and "," in reference:
        header, payload = reference.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] if header.startswith("data:") else None
        return mime_type, payload
    return None, reference


def _build_reference_parts(references: list[Any] | None) -> list[Any]:
    parts: list[Any] = []
    if not references:
        return parts
    for ref in references:
        payload_raw = getattr(ref, "base64", "") or ""
        payload_raw = payload_raw.strip()
        if not payload_raw:
            continue
        inferred_mime, payload = _parse_reference_payload(payload_raw)
        mime_type = getattr(ref, "mimeType", None) or inferred_mime or "image/png"
        if not mime_type.startswith("image/"):
            mime_type = "image/png"
        try:
            image_bytes = base64.b64decode("".join(payload.split()), validate=True)
        except Exception:
            logger.warning("Invalid reference image payload, skipping")
            continue
        if not image_bytes:
            continue
        parts.append(_create_inline_part(image_bytes, mime_type))
    return parts


def _extract_image_and_text(parts: list[Any]) -> tuple[bytes | None, str, str]:
    texts: list[str] = []
    image_bytes: bytes | None = None
    mime_type = "image/png"

    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            texts.append(text_value.strip())

        inline_data = getattr(part, "inline_data", None)
        data = getattr(inline_data, "data", None) if inline_data is not None else None
        candidate_mime = getattr(inline_data, "mime_type", None) if inline_data is not None else None
        if not data:
            continue
        if isinstance(data, bytearray):
            payload = bytes(data)
        elif isinstance(data, bytes):
            payload = data
        else:
            continue
        if isinstance(candidate_mime, str) and candidate_mime.startswith("image/"):
            mime_type = candidate_mime
        image_bytes = payload

    return image_bytes, mime_type, "\n".join(texts).strip()


ProgressCallback = Callable[[str, str], Coroutine[Any, Any, None]]


async def _noop_progress(stage: str, message: str) -> None:
    pass


async def create_panel_image_response(
    payload: PanelRequest,
    user_id: str,
    api_key: APIKey | None,
    db: Session,
    config: GatewayConfig,
    cache_key: str,
    aspect_ratio: AspectRatioType,
    resolution: ResolutionType,
    analysis: AnalysisLevelType,
    on_progress: ProgressCallback | None = None,
) -> PanelImageResponse:
    report_progress = on_progress or _noop_progress

    provider_name = "gemini"
    model_input = DEFAULT_MODEL
    model_key, _ = _get_model_pricing(db, provider_name, model_input)

    _ensure_genai_available()
    assert genai is not None
    api_key_value = _get_gemini_api_key(config)
    client = genai.Client(api_key=api_key_value)

    await report_progress("prepare", "요청 준비 중")

    normalized_scene_elements = _normalize_scene_elements(payload.sceneElements)
    scene_value = payload.scene or ""
    scene_summary = scene_value.strip()
    if not scene_summary and _has_scene_elements(normalized_scene_elements):
        scene_summary = _build_scene_summary(normalized_scene_elements, scene_value)
    panel_description = scene_summary or scene_value
    dialogue_lines = _split_dialogue_lines(payload.dialogue)

    # Simplify scene to single keyframe to avoid multi-action rendering issues
    if analysis == "full" and panel_description.strip():
        await report_progress("keyframe", "장면 키프레임 추출 중")
        panel_description = _simplify_scene_to_keyframe(
            client,
            panel_description,
            payload.characters,
            payload.dialogue,
        )
        logger.info("Simplified scene to keyframe: %s", panel_description[:100])

    style_prompt = _get_style_prompt(payload.style)
    is_caricature = payload.characterGenerationMode == "caricature"
    character_descriptions = (
        payload.characterDescriptions
        if payload.characterDescriptions is not None
        else [f"Character: {name}" for name in payload.characters]
    )
    character_details_text = "\n".join(payload.characterDescriptions or [])
    full_image_prompt = "" if is_caricature else _build_image_prompt(
        payload.style,
        panel_description,
        character_descriptions,
    )
    role = "You are a caricature illustration expert." if is_caricature else f"You are a {style_prompt['name']} image generation expert."
    style_guide = f"{style_prompt['systemPrompt']}\n{style_prompt['imagePrompt']}"

    scene_elements_text = (
        _format_scene_elements(normalized_scene_elements) if _has_scene_elements(normalized_scene_elements) else ""
    )
    scene_elements_block = (
        "## Scene Elements (Structured)\n"
        f"{scene_elements_text}\n"
        "Use these elements to refine composition, lighting, and mood. If conflicts arise, follow the Scene Description and Dialogue Cues."
        if scene_elements_text
        else ""
    )
    dialogue_cue_block = (
        "## Dialogue Cues (Must Match Visual Details)\n"
        + "\n".join([f"- {line}" for line in dialogue_lines])
        + "\nIf the dialogue mentions props, device states, or actions (e.g., phone is off), make them visible and consistent in the scene."
        if dialogue_lines
        else ""
    )
    scene_authority_block = (
        "## Scene Authority\n"
        "- Scene Description and Dialogue Cues are authoritative for actions, props, device states, and environment.\n"
        "- Character sheets and reference images are authoritative only for character appearance (face, body, outfit).\n"
        "- If there is a conflict, keep appearance consistent but follow Scene/Dialogue for props and actions."
    )
    output_format = f"PNG ({aspect_ratio}), {resolution} quality, panel {payload.panelNumber} of 4"

    doc_instruction = ""
    if not is_caricature and payload.styleDoc and payload.styleDoc.strip():
        doc_instruction = f"Reference these style notes:\n{payload.styleDoc}\n"

    era_label = resolve_era_label(payload.era)
    season_label = resolve_season_label(payload.season)
    world_setting_block = ""
    if era_label or season_label:
        world_setting_block = (
            "## World Setting\n"
            f"{f'Era: {era_label}' if era_label else ''}\n"
            f"{f'Season: {season_label}' if season_label else ''}\n"
            "- If an era is provided, make it clearly visible through setting, props, architecture, and technology.\n"
            "- Reinterpret modern elements into era-appropriate equivalents when needed.\n"
            "- Season should color the mood, lighting, and palette without overriding the era.\n"
            "- If character sheets define outfits, keep them and emphasize the era via environment/props.\n"
            "- If the art style implies a different era, keep the rendering style but prioritize the chosen era.\n"
            "- Avoid anachronisms that break the chosen era.\n"
            "- Do not override explicit scene requirements or character sheet facts."
        ).strip()

    revision_focus = ""
    if payload.revisionNote and payload.revisionNote.strip():
        revision_focus = (
            "## Revision Focus\n"
            f"{payload.revisionNote}\n"
            "Apply the revision while preserving core composition, continuity, and character identity."
        )

    character_consistency_lines = "\n".join(
        [f"- Same character as {name}" for name in payload.characters if name]
    )
    character_sheet_metadata_lines_list: list[str] = []
    visual_identity_lock_lines: list[str] = []
    for entry in payload.characterSheetMetadata or []:
        name = entry.name.strip() if entry.name else ""
        if not name:
            continue
        parsed_metadata = _parse_character_sheet_metadata(entry.metadata)
        formatted = _format_character_sheet_metadata(parsed_metadata) if parsed_metadata else ""
        if not formatted and entry.metadata:
            formatted = str(entry.metadata).strip()
        if not formatted:
            continue
        character_sheet_metadata_lines_list.append(f"- {name}: {formatted}")
        # Extract key visual elements for identity lock
        if parsed_metadata:
            key_elements = _extract_key_visual_elements(parsed_metadata)
            identity_lock = _format_visual_identity_lock(name, key_elements)
            if identity_lock:
                visual_identity_lock_lines.append(identity_lock)
    character_sheet_metadata_lines = "\n".join(character_sheet_metadata_lines_list)
    visual_identity_lock_text = "\n\n".join(visual_identity_lock_lines)
    previous_panel_entries = payload.previousPanels or []
    continuity_notes: list[str] = []
    spatial_continuity_notes: list[str] = []  # For 4-element position/spatial info
    for previous in previous_panel_entries:
        scene_text = previous.get("scene") or ""
        dialogue_text = previous.get("dialogue") or ""
        metadata_text = previous.get("metadata") or ""
        formatted_metadata = ""
        if metadata_text:
            parsed_metadata = _parse_panel_metadata(metadata_text)
            if parsed_metadata:
                formatted_metadata = _format_panel_metadata(parsed_metadata) or metadata_text
                # Extract 4-element spatial/position info for strict continuity
                characters_meta = parsed_metadata.get("characters") or []
                for char in characters_meta:
                    char_name = char.get("name") or ""
                    position = char.get("position") or ""
                    facing = char.get("facing") or ""
                    if char_name and (position or facing):
                        spatial_continuity_notes.append(
                            f"- {char_name}: position={position}, facing={facing}"
                        )
                camera_meta = parsed_metadata.get("camera")
                if camera_meta:
                    shot = camera_meta.get("shot_type") or ""
                    angle = camera_meta.get("angle") or ""
                    if shot or angle:
                        spatial_continuity_notes.append(f"- Camera: shot={shot}, angle={angle}")
                env_meta = parsed_metadata.get("environment")
                if env_meta:
                    location = env_meta.get("location") or ""
                    if location:
                        spatial_continuity_notes.append(f"- Environment: {location}")
                cont_meta = parsed_metadata.get("continuity")
                if cont_meta:
                    key_objects = cont_meta.get("key_objects") or []
                    if key_objects:
                        spatial_continuity_notes.append(f"- Key objects: {', '.join(key_objects)}")
            else:
                formatted_metadata = metadata_text
        previous_dialogue = f" Dialogue - {dialogue_text}." if dialogue_text else ""
        continuity_notes.append(
            f"Panel {previous.get('panel')}: Scene description - {scene_text}.{previous_dialogue}"
            + (f" Metadata: {formatted_metadata}." if formatted_metadata else "")
        )
    continuity_prompt = ""
    if continuity_notes:
        spatial_block = ""
        if spatial_continuity_notes:
            spatial_block = (
                "\n\n## ⚠️ SPATIAL CONTINUITY (CRITICAL) ⚠️\n"
                "Maintain these positions and spatial relationships from the previous panel:\n"
                + "\n".join(spatial_continuity_notes)
                + "\n- Characters should stay in the SAME relative positions unless the scene explicitly describes movement."
                "\n- If a character was on the LEFT, keep them on the LEFT. If on the RIGHT, keep them on the RIGHT."
                "\n- Camera angle and shot type should remain consistent unless a scene transition occurs."
            )
        continuity_prompt = (
            "## Continuity Notes\n"
            + "\n".join(continuity_notes)
            + spatial_block
            + "\nAlways reuse the same outfit, accessories, props, and limb placements described above unless a scene explicitly calls for a change. "
            "Keep the background environment, lighting, and time-of-day consistent unless the script explicitly changes the setting. "
            "When a wardrobe, prop, or location change occurs, explain the reason while keeping facial features, hair color, and proportions consistent."
        )

    attachment_continuity_block = (
        "## Attachment Continuity Lock\n"
        "- Scope: all attached, worn, or held items (clothing layers, shoes, hats, glasses, jewelry, hair accessories, bags, belts, watches, patches, logos, patterns, handheld props, attached gadgets).\n"
        "- Rule: preserve base color, material, shape, size, and placement across panels. Do not recolor, retexture, or swap items.\n"
        "- Lighting may vary with the scene, but the item identity must remain the same.\n"
        "- Exception: if the scene explicitly changes an item, describe the change and keep all other items identical."
    )

    anatomical_accuracy_block = (
        "## Anatomical Accuracy Requirements\n"
        "- CRITICAL: Each human character must have exactly 2 arms, 2 hands (5 fingers each), 2 legs, and 1 head.\n"
        "- Never generate extra limbs, duplicate body parts, mutated hands, or fused fingers.\n"
        "- If hands are performing complex actions, simplify to a single clear gesture.\n"
        "- Avoid overlapping or intertwined limbs that may cause anatomical confusion.\n"
        "- When hands are not essential to the scene, keep them in natural resting positions or out of frame.\n"
        "- Prioritize anatomical correctness over action complexity."
    )

    pose_simplification_block = (
        "## Pose Simplification Guidelines\n"
        "- Convert complex multi-action descriptions into a single primary action.\n"
        "- If the scene describes 'doing A while doing B', focus on the most important action.\n"
        "- Specify hand positions explicitly: 'hands at sides', 'hands in pockets', 'holding [object] with both hands'.\n"
        "- Avoid foreshortening and extreme angles that may distort limb proportions.\n"
        "- Use simple, clear silhouettes that read well at a glance."
    )
    character_sheet_metadata_block = (
        "## Character Sheet Metadata (Authoritative)\n"
        f"{character_sheet_metadata_lines}\n"
        "Always follow these details exactly unless the scene explicitly requires a change.\n"
        "Reference the character sheet for exact body proportions - the character's limb count and body structure must match the reference exactly."
        if character_sheet_metadata_lines
        else ""
    )

    # Visual Identity Lock - HIGHEST PRIORITY for character appearance consistency
    visual_identity_lock_block = (
        "## ⚠️ VISUAL IDENTITY LOCK (HIGHEST PRIORITY) ⚠️\n"
        "The following visual elements are IMMUTABLE and MUST be copied exactly from the character sheet images.\n"
        "DO NOT modify, reinterpret, or change these elements under any circumstances:\n\n"
        f"{visual_identity_lock_text}\n\n"
        "ENFORCEMENT RULES:\n"
        "- Hair color and style: Copy EXACTLY from the character sheet image. No variations allowed.\n"
        "- Face shape and features: Match the reference image precisely.\n"
        "- Outfit and clothing: Replicate the exact design, colors, and patterns.\n"
        "- Accessories: Include ALL accessories shown in the character sheet.\n"
        "- If you cannot see a detail clearly in the character sheet, use the metadata description.\n"
        "- Scene descriptions may change poses and actions, but NEVER change appearance elements listed above."
        if visual_identity_lock_text
        else ""
    )

    character_anchoring_block = (
        "## Character Anchoring (Single Instance Rule)\n"
        + "\n".join([f"- Render [{name}] as a single, complete figure. Do not show {name} multiple times or from multiple angles." for name in payload.characters if name])
        + "\n- Each named character appears exactly once per panel.\n"
        "- Do not clone, mirror, or duplicate any character.\n"
        "- Background figures must be visually distinct from named characters."
        if payload.characters
        else ""
    )

    caricature_strength_lines = "\n".join([entry.strip() for entry in payload.characterCaricatureStrengths or [] if entry.strip()])
    caricature_strength_block = (
        "## Caricature Strength (Per Character)\n"
        f"{caricature_strength_lines}\n"
        "Use these levels to keep caricature exaggeration consistent across panels."
        if caricature_strength_lines
        else ""
    )

    primary_references = [ref for ref in payload.references or [] if ref.purpose not in ("background", "previous_panel")]
    background_references = [ref for ref in payload.references or [] if ref.purpose == "background"]
    previous_panel_references = [ref for ref in payload.references or [] if ref.purpose == "previous_panel"]
    has_primary_references = bool(primary_references)
    has_background_references = bool(background_references)
    has_previous_panel_reference = bool(previous_panel_references)
    has_character_images = bool(payload.characterImages)

    caricature_style_override = (
        "## Caricature Mode Override\n"
        "- Keep the overall look clearly caricatured and cartoon-like.\n"
        "- Use simplified shapes, bold outlines, and soft cel-shading.\n"
        "- Emphasize 2-3 distinctive facial features so the caricature is obvious but friendly.\n"
        "- Keep the head noticeably larger than the body and simplify limb details.\n"
        "- If the style guide conflicts with caricature exaggeration, prioritize caricature exaggeration."
        if is_caricature
        else ""
    )
    caricature_guardrails = (
        "- The output must be clearly stylized and cartoon-like.\n"
        "- Do not recreate or resemble real people, celebrities, or public figures.\n"
        "- Avoid photorealistic textures, skin detail, and lighting.\n"
        "- Exaggerate features moderately to emphasize a caricature feel while keeping it friendly."
        if is_caricature
        else ""
    )

    consistency_prompt = (
        "Character Consistency:\n"
        f"- {character_consistency_lines or 'Keep the same character.'}\n"
        + (
            "- Use the character sheet metadata below as the primary source of truth for outfits, accessories, hair, and props.\n"
            if character_sheet_metadata_lines
            else ""
        )
        + "- Keep clothing, accessories, hairstyles, and jewelry consistent with the character sheet.\n"
        "- Match the face shape, hairstyle, outfit, and props from the front-facing character sheet.\n"
        "- Preserve clothing/accessory colors, materials, logos, and placement unless the scene explicitly changes them.\n"
        "- Do not change body shape, skin tone, or hair color.\n"
        "- Maintain the same appearance so the character does not look like a different person.\n"
        "- Render only one instance per named character. Never duplicate or clone the speaker.\n"
        "- Props and device states must match the Scene Description and Dialogue Cues."
    )

    priority_rule = ""
    if has_primary_references:
        priority_rule = (
            "Priority: Reference images are primary for character appearance only. Do not let references override Scene Description "
            "or Dialogue Cues about actions, props, device states, or environment."
        )
    elif has_character_images:
        priority_rule = (
            "Priority: Character sheet images are primary for character appearance only. Do not let them override Scene Description "
            "or Dialogue Cues about actions, props, device states, or environment."
        )

    era_id = payload.era if payload.era and payload.era != "any" else None
    era_guardrails_block = _build_era_guardrails(era_id)

    should_run_analysis = analysis == "full"
    should_enforce_era = should_run_analysis and bool(era_label)
    should_enforce_background = should_run_analysis and has_background_references

    # Run metadata and reference analysis in parallel
    async def run_metadata_generation() -> str:
        if not should_run_analysis:
            return ""
        await report_progress("metadata", "패널 메타데이터 분석 중")
        return _generate_panel_metadata(
            client,
            panel_description,
            scene_elements_text,
            dialogue_lines,
            payload.characters,
            payload.characterDescriptions,
            character_sheet_metadata_lines,
            continuity_prompt,
        )

    async def run_reference_analysis() -> str:
        if not should_run_analysis or not has_primary_references:
            return ""
        await report_progress("reference-metadata", "참고 이미지 분석 중")
        return _analyze_reference_images(client, primary_references)

    metadata_summary, reference_metadata_text = await asyncio.gather(
        run_metadata_generation(),
        run_reference_analysis(),
    )
    reference_metadata_block = (
        "## Reference Image Metadata (Preserve)\n"
        f"{reference_metadata_text}\n"
        "Always preserve these visual details unless the scene explicitly changes them."
        if reference_metadata_text
        else ""
    )
    background_reference_block = (
        "## Background Reference (Environment Only)\n"
        "- Background reference images are provided to keep the location and props consistent.\n"
        "- Use them only for environment layout, architecture, and major props.\n"
        "- Do not copy any people, clothing, or character-specific details from them.\n"
        "- Keep the location consistent unless the scene explicitly changes it."
        if has_background_references
        else ""
    )

    # Previous Panel Reference - for layout and spatial consistency
    previous_panel_reference_block = (
        "## ⚠️ PREVIOUS PANEL REFERENCE (LAYOUT/COMPOSITION ONLY) ⚠️\n"
        "A previous panel image is provided to maintain visual continuity.\n"
        "USE THIS FOR:\n"
        "- Character POSITIONS (left/right placement, relative distances)\n"
        "- Camera ANGLE and PERSPECTIVE (maintain similar viewpoint)\n"
        "- Background LAYOUT (furniture, objects, architecture placement)\n"
        "- Overall COMPOSITION and FRAMING\n\n"
        "DO NOT USE THIS FOR:\n"
        "- Character APPEARANCE (hair, face, outfit) - use CHARACTER SHEET instead\n"
        "- Exact poses - follow the new scene description\n\n"
        "RULE: If characters were on the left/right in the previous panel, keep them in similar positions "
        "unless the scene explicitly describes movement or repositioning."
        if has_previous_panel_reference
        else ""
    )

    system_instruction = _build_image_system_instruction(
        character_generation_mode=payload.characterGenerationMode,
        has_references=has_primary_references,
        has_character_images=has_character_images,
        has_background_references=has_background_references,
    )

    image_config = genai.types.ImageConfig(
        aspect_ratio=aspect_ratio,
        image_size=resolution,
    )
    content_config = genai.types.GenerateContentConfig(
        response_modalities=["Text", "Image"],
        image_config=image_config,
        candidate_count=1,
    )

    def build_panel_prompt_with_corrections(
        era_correction: str = "", background_correction: str = ""
    ) -> str:
        era_correction_block = (
            f"## Era Correction\n{era_correction}\nThis correction is mandatory and must override conflicting details.\n"
            if era_correction
            else ""
        )
        background_correction_block = (
            f"## Background Correction\n{background_correction}\nThis correction is mandatory and must override conflicting details.\n"
            if background_correction
            else ""
        )
        base_text_prompt = (
            f"# Role\n{role}\n\n"
            "# Instruction\n"
            "Reference the provided **character sheet images** absolutely and depict a scene that matches the **scene description**. "
            "The character sheet images are your PRIMARY VISUAL REFERENCE - copy the appearance EXACTLY.\n"
            + (f"\n{visual_identity_lock_block}\n\n" if visual_identity_lock_block else "")
            + "---\n\n"
            "## Key Style Guide\n"
            f"{CARICATURE_PANEL_STYLE_GUIDE if is_caricature else style_guide}\n\n"
            "---\n\n"
            f"{world_setting_block}\n"
            + (f"\n{era_guardrails_block}\n" if era_guardrails_block else "")
            + era_correction_block
            + background_correction_block
            + "## Scene Information\n"
            f"Scene Description: {panel_description}\n"
            f"{scene_elements_block}\n"
            f"{dialogue_cue_block}\n"
            + (
                f"Character Details:\n{character_details_text}\n"
                if payload.characterDescriptions is not None
                else f"Characters: {', '.join(payload.characters)}\n"
            )
            + (f"\n{caricature_style_override}\n" if caricature_style_override else "")
            + (f"\n{caricature_strength_block}\n" if caricature_strength_block else "")
            + (f"\n{character_sheet_metadata_block}\n" if character_sheet_metadata_block else "")
            + (f"\n{reference_metadata_block}\n" if reference_metadata_block else "")
            + (f"\n{background_reference_block}\n" if background_reference_block else "")
            + (f"\n{previous_panel_reference_block}\n" if previous_panel_reference_block else "")
            + (f"\n{continuity_prompt}\n" if continuity_prompt else "")
            + f"\n{anatomical_accuracy_block}\n"
            + (f"\n{character_anchoring_block}\n" if character_anchoring_block else "")
            + (f"\n{revision_focus}\n" if revision_focus else "")
            + "\n## Output Format\n"
            f"{output_format}\n"
            "Additional Rule: Do not include speech balloons."
        )
        return (
            f"{full_image_prompt}\n\n{base_text_prompt}\n\n"
            f"Panel {payload.panelNumber}: {panel_description}\n"
            f"References: {', '.join(payload.characters)}\n"
            + (f"Character Details:\n{character_details_text}\n" if payload.characterDescriptions is not None else "")
            + doc_instruction
        ).strip()

    max_attempts = 1
    final_image_bytes: bytes | None = None
    final_mime_type = "image/png"
    final_text = ""
    era_correction = ""
    background_correction = ""

    for attempt in range(max_attempts):
        attempt_label = f" ({attempt + 1}/{max_attempts})" if should_enforce_era or should_enforce_background else ""

        await report_progress("translate", f"프롬프트 번역 중{attempt_label}")
        panel_prompt = build_panel_prompt_with_corrections(era_correction, background_correction)
        if should_run_analysis:
            panel_prompt = _translate_prompt(client, panel_prompt)

        prompt_with_system = f"{system_instruction}\n\n{panel_prompt}".strip()
        parts = [genai.types.Part.from_text(text=prompt_with_system)]
        parts.extend(_build_character_image_parts(payload.characterImages))
        parts.extend(_build_reference_parts(primary_references))
        parts.extend(_build_reference_parts(background_references))
        parts.extend(_build_reference_parts(previous_panel_references))
        contents: Any = [genai.types.Content(role="user", parts=parts)]

        await report_progress("generate", f"이미지 생성 중{attempt_label}")
        try:
            response = client.models.generate_content(
                model=model_input,
                contents=contents,
                config=content_config,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Panel image generation failed: %s", exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Panel image generation failed") from exc

        usage_info = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        usage_for_charge = _coerce_usage_metadata(usage_info) or usage_info
        usage_log_id = _log_image_usage(
            db=db,
            api_key_obj=api_key,
            model=model_input,
            provider=provider_name,
            endpoint="/v1/webtoon/generate-panel-image",
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

        response_parts = getattr(response, "parts", None) or []
        if not response_parts:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                content = getattr(candidates[0], "content", None)
                response_parts = getattr(content, "parts", None) or []

        image_bytes, mime, text = _extract_image_and_text(response_parts)

        if not image_bytes:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No image returned from model")

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        needs_retry = False

        # Era consistency check
        if should_enforce_era:
            await report_progress("era-check", f"시대 일치 확인 중{attempt_label}")
            era_check = _analyze_era_consistency(
                client,
                image_base64,
                mime,
                era_id,
                era_label,
                season_label,
            )
            if not era_check["consistent"] and attempt < max_attempts - 1:
                issues = era_check.get("issues", [])
                fallback_fix = (
                    f"Remove/avoid these anachronisms: {', '.join(issues)}."
                    if issues
                    else "Strengthen era-specific background and props while removing modern elements."
                )
                era_correction = (era_check.get("fix") or "").strip() or fallback_fix
                await report_progress("era-retry", "시대 불일치 감지, 재생성")
                needs_retry = True

        # Background consistency check
        if should_enforce_background:
            await report_progress("background-check", f"배경 일치 확인 중{attempt_label}")
            base_background = background_references[0] if background_references else None
            if base_background and getattr(base_background, "base64", None):
                background_check = _analyze_background_consistency(
                    client,
                    image_base64,
                    mime,
                    base_background.base64,
                    getattr(base_background, "mimeType", None),
                    panel_description,
                    era_label,
                    season_label,
                )
                if not background_check["consistent"] and attempt < max_attempts - 1:
                    issues = background_check.get("issues", [])
                    fallback_fix = (
                        f"Keep these background elements consistent: {', '.join(issues)}."
                        if issues
                        else "Match the background layout, architecture, and key props from the reference image."
                    )
                    background_correction = (background_check.get("fix") or "").strip() or fallback_fix
                    await report_progress("background-retry", "배경 불일치 감지, 재생성")
                    needs_retry = True

        if needs_retry and attempt < max_attempts - 1:
            continue

        final_image_bytes = image_bytes
        final_mime_type = mime
        final_text = text
        break

    if not final_image_bytes:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No image returned")

    # Normalize image size
    await report_progress("normalize", "이미지 규격화 중")
    normalized_bytes, normalized_mime = _normalize_image(final_image_bytes, resolution, aspect_ratio)

    await report_progress("complete", "완료")

    # Prefer image model's actual metadata (final_text) over pre-generated metadata (metadata_summary)
    # because final_text reflects what was actually generated (positions, camera angles, etc.)
    parsed = parse_json(final_text)
    if isinstance(parsed, dict) and parsed.get("characters"):
        # Image model returned valid 4-element metadata - use it
        metadata_text = json.dumps(parsed, ensure_ascii=False)
        logger.info("Using image model's metadata with %d characters", len(parsed.get("characters", [])))
    elif final_text.strip():
        # Image model returned some text but not valid JSON - try to use it
        metadata_text = final_text.strip()
    elif metadata_summary.strip():
        # Fall back to pre-generated metadata
        metadata_text = metadata_summary.strip()
    else:
        metadata_text = ""

    inline_data = base64.b64encode(normalized_bytes).decode("utf-8")
    result = finalize_response(
        payload_text=metadata_text,
        inline_image_base64=inline_data,
        mime_type=normalized_mime,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        panel_number=payload.panelNumber,
    )
    set_cached_panel_image(cache_key, result)
    return result


async def _generate_sse_stream(
    payload: PanelRequest,
    user_id: str,
    api_key: APIKey | None,
    db: Session,
    config: GatewayConfig,
    cache_key: str,
    aspect_ratio: AspectRatioType,
    resolution: ResolutionType,
    analysis: AnalysisLevelType,
) -> AsyncGenerator[str, None]:
    progress_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    result_holder: list[PanelImageResponse | Exception] = []

    async def on_progress(stage: str, message: str) -> None:
        try:
            progress_queue.put_nowait((stage, message))
            await asyncio.sleep(0)  # Yield to event loop to allow SSE to be sent
        except Exception:
            pass

    async def run_generation() -> None:
        try:
            result = await create_panel_image_response(
                payload,
                user_id,
                api_key,
                db,
                config,
                cache_key,
                aspect_ratio,
                resolution,
                analysis,
                on_progress=on_progress,
            )
            result_holder.append(result)
        except Exception as exc:
            result_holder.append(exc)
        finally:
            await progress_queue.put(None)

    generation_task = asyncio.create_task(run_generation())

    try:
        while True:
            item = await progress_queue.get()
            if item is None:
                break
            stage, message = item
            event_data = json.dumps({"stage": stage, "message": message}, ensure_ascii=False)
            yield f"event: status\ndata: {event_data}\n\n"

        await generation_task

        if result_holder:
            result_or_error = result_holder[0]
            if isinstance(result_or_error, Exception):
                error_msg = str(result_or_error)
                error_data = json.dumps({"message": error_msg}, ensure_ascii=False)
                yield f"event: error\ndata: {error_data}\n\n"
            else:
                result_data = result_or_error.model_dump() if hasattr(result_or_error, "model_dump") else result_or_error.dict()
                yield f"event: result\ndata: {json.dumps(result_data, ensure_ascii=False)}\n\n"

        yield f"event: done\ndata: {json.dumps({'ok': True})}\n\n"
    except asyncio.CancelledError:
        generation_task.cancel()
        raise


@router.post("/generate-panel-image", response_model=PanelImageResponse)
async def generate_panel_image(
    request: Request,
    payload: PanelRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
):
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    aspect_ratio: AspectRatioType = payload.aspectRatio or DEFAULT_ASPECT_RATIO
    resolution: ResolutionType = payload.resolution or DEFAULT_RESOLUTION
    analysis: AnalysisLevelType = payload.analysisLevel or "fast"
    cache_key = build_cache_key(payload, aspect_ratio, resolution, analysis)

    logger.info(
        "webtoon.generate-panel-image request panel=%s style=%s aspectRatio=%s resolution=%s analysisLevel=%s scene=%s dialogue=%s era=%s season=%s",
        payload.panelNumber,
        payload.style,
        aspect_ratio,
        resolution,
        analysis,
        payload.scene,
        payload.dialogue,
        payload.era,
        payload.season,
    )

    # Log additional metadata (excluding image data)
    logger.info(
        "webtoon.generate-panel-image metadata characters=%s characterDescriptions=%s characterImagesCount=%s referencesCount=%s revisionNote=%s",
        payload.characters,
        payload.characterDescriptions,
        len(payload.characterImages) if payload.characterImages else 0,
        len(payload.references) if payload.references else 0,
        payload.revisionNote,
    )

    if payload.characterSheetMetadata:
        sheet_metadata_summary = [
            {"name": entry.name, "hasMetadata": bool(entry.metadata)}
            for entry in payload.characterSheetMetadata
        ]
        logger.info(
            "webtoon.generate-panel-image characterSheetMetadata=%s",
            sheet_metadata_summary,
        )

    if payload.previousPanels:
        previous_panels_summary = [
            {
                "panel": p.get("panel"),
                "hasScene": bool(p.get("scene")),
                "hasDialogue": bool(p.get("dialogue")),
                "hasMetadata": bool(p.get("metadata")),
            }
            for p in payload.previousPanels
        ]
        logger.info(
            "webtoon.generate-panel-image previousPanels=%s",
            previous_panels_summary,
        )

    if payload.references:
        references_summary = [
            {"purpose": ref.purpose, "hasMimeType": bool(ref.mimeType)}
            for ref in payload.references
        ]
        logger.info(
            "webtoon.generate-panel-image references=%s",
            references_summary,
        )

    # Check for SSE streaming request
    accept_header = request.headers.get("accept", "")
    wants_stream = "text/event-stream" in accept_header

    cached = get_cached_panel_image(cache_key)
    if cached:
        if wants_stream:
            async def cached_stream() -> AsyncGenerator[str, None]:
                yield f"event: status\ndata: {json.dumps({'stage': 'cache', 'message': 'cache hit'}, ensure_ascii=False)}\n\n"
                result_data = cached.model_dump() if hasattr(cached, "model_dump") else cached.dict()
                yield f"event: result\ndata: {json.dumps(result_data, ensure_ascii=False)}\n\n"
                yield f"event: done\ndata: {json.dumps({'ok': True})}\n\n"
            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                },
            )
        return cached

    if wants_stream:
        return StreamingResponse(
            _generate_sse_stream(
                payload,
                user_id,
                api_key,
                db,
                config,
                cache_key,
                aspect_ratio,
                resolution,
                analysis,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
            },
        )

    return await create_panel_image_response(
        payload,
        user_id,
        api_key,
        db,
        config,
        cache_key,
        aspect_ratio,
        resolution,
        analysis,
    )
