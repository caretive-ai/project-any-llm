"""Schema definitions for calendar image generation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LayoutType = Literal["standard_4_3", "wide_16_9", "wide_16_9_half_left"]
AspectRatioType = Literal["1:1", "16:9"]


class GenerateImageRequest(BaseModel):
    """Request schema for calendar image generation."""

    prompt: str = Field(..., min_length=1, description="Image generation prompt")
    layout: LayoutType = Field(default="standard_4_3", description="Calendar layout type")
    user: str | None = Field(default=None, description="User ID for master key usage")


class GenerateImageResponse(BaseModel):
    """Response schema for calendar image generation."""

    image: str = Field(..., description="Base64 encoded image data URI")


DEFAULT_MODEL = "gemini-3-pro-image-preview"
IMAGE_SIZE = "4K"
