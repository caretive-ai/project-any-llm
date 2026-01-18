"""Schema definitions for calendar prompt generation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GeneratePromptRequest(BaseModel):
    """Request schema for calendar prompt generation."""

    month: int = Field(..., ge=1, le=12, description="Month number (1-12)")
    user: str | None = Field(default=None, description="User ID for master key usage")


class PromptSuggestions(BaseModel):
    """Generated prompt suggestions."""

    default: str = Field(..., description="Default Korean seasonal landscape prompt")
    anime_female: str = Field(..., description="Anime female character prompt in English")
    anime_male: str = Field(..., description="Anime male character prompt in English")


class GeneratePromptResponse(BaseModel):
    """Response schema for calendar prompt generation."""

    default: str
    anime_female: str
    anime_male: str


DEFAULT_MODEL = "gemini-3-flash-preview"
