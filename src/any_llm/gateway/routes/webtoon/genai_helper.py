"""Common helper functions for Google GenAI integration in webtoon routes."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.image import _get_gemini_api_key

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


def ensure_genai_available() -> None:
    """Raise HTTPException if google-genai is not installed."""
    if genai is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai dependency is not installed",
        )


def create_genai_client(config: GatewayConfig) -> Any:
    """Create and return a genai Client instance."""
    ensure_genai_available()
    assert genai is not None
    api_key_value = _get_gemini_api_key(config)
    return genai.Client(api_key=api_key_value)


def extract_text_from_candidate(candidate: Any) -> str:
    """Extract text from a genai response candidate."""
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    fragments: list[str] = []
    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            fragments.append(text_value.strip())
    return "\n".join(fragments).strip()


def get_response_text(response: Any) -> str | None:
    """Extract text from genai response."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return extract_text_from_candidate(candidates[0])


def build_text_content_config(
    system_instruction: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Build GenerateContentConfig for text-only responses."""
    assert genai is not None
    config_kwargs: dict[str, Any] = {
        "response_modalities": ["Text"],
        "candidate_count": 1,
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    return genai.types.GenerateContentConfig(**config_kwargs)


def build_user_contents(prompt: str) -> list[Any]:
    """Build contents list for user prompt."""
    assert genai is not None
    return [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=prompt)])]


def generate_text_content(
    client: Any,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    temperature: float | None = None,
) -> Any:
    """Generate text content using genai client."""
    assert genai is not None
    content_config = build_text_content_config(system_prompt, temperature)
    contents = build_user_contents(user_prompt)
    logger.info("generate_text_content: model=%s, temperature=%s", model, temperature)
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=content_config,
    )


def build_multimodal_contents(prompt: str, images: list[dict[str, str]]) -> list[Any]:
    """Build contents list for multimodal prompt with images.

    Args:
        prompt: Text prompt
        images: List of dicts with 'data' (base64) and 'mime_type' keys
    """
    assert genai is not None
    import base64

    parts: list[Any] = []

    # Add images first
    for img in images:
        if img.get("data") and img.get("mime_type"):
            try:
                image_bytes = base64.b64decode(img["data"])
                parts.append(
                    genai.types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=img["mime_type"],
                    )
                )
            except Exception as e:
                logger.warning("Failed to decode image: %s", e)
                continue

    # Add text prompt
    parts.append(genai.types.Part.from_text(text=prompt))

    return [genai.types.Content(role="user", parts=parts)]


def generate_multimodal_content(
    client: Any,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    images: list[dict[str, str]],
    temperature: float | None = None,
) -> Any:
    """Generate content with images using genai client.

    Args:
        client: genai Client instance
        model: Model name
        system_prompt: System instruction
        user_prompt: User prompt text
        images: List of dicts with 'data' (base64) and 'mime_type' keys
        temperature: Generation temperature
    """
    assert genai is not None
    content_config = build_text_content_config(system_prompt, temperature)
    contents = build_multimodal_contents(user_prompt, images)
    logger.info(
        "generate_multimodal_content: model=%s, temperature=%s, imageCount=%d",
        model,
        temperature,
        len(images),
    )
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=content_config,
    )
