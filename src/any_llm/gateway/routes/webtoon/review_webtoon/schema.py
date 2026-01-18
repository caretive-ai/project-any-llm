from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel


class CharacterEntry(BaseModel):
    name: str
    description: str | None = None
    imageData: str | None = None  # base64 encoded image
    imageMimeType: str | None = None
    metadata: str | None = None  # AI-generated character metadata


class PanelReviewEntry(BaseModel):
    panel: int | None = None
    panelNumber: int | None = None
    scene: str | None = None
    speaker: str | None = None
    dialogue: str | None = None
    metadata: str | None = None
    imageData: str | None = None  # base64 encoded panel image
    imageMimeType: str | None = None


EraType = Literal["any", "modern", "nineties", "seventies-eighties", "joseon", "future"]
SeasonType = Literal["any", "spring", "summer", "autumn", "winter"]


class ReviewWebtoonRequest(BaseModel):
    topic: str | None = None
    genre: str | None = None
    style: str | None = None
    era: EraType | None = None
    season: SeasonType | None = None
    characters: List[CharacterEntry] | None = None
    scriptSummary: str | None = None
    panels: List[PanelReviewEntry] | None = None


class ReviewNextIdea(BaseModel):
    title: str
    topic: str
    genre: str
    style: str
    hook: str


class VisualAnalysis(BaseModel):
    artStyleConsistency: str
    colorPalette: str
    composition: str
    characterConsistency: str


class PanelFeedback(BaseModel):
    panel: int
    strength: str
    suggestion: str


class CharacterAnalysis(BaseModel):
    name: str
    appeal: str
    consistency: str


class StoryStructure(BaseModel):
    opening: str
    development: str
    climax: str
    resolution: str


class OverallScore(BaseModel):
    story: int
    visual: int
    character: int
    overall: int


class ReviewWebtoonResponse(BaseModel):
    headline: str
    summary: str
    strengths: List[str]
    improvements: List[str]
    visualAnalysis: VisualAnalysis | None = None
    panelFeedback: List[PanelFeedback] | None = None
    characterAnalysis: List[CharacterAnalysis] | None = None
    storyStructure: StoryStructure | None = None
    overallScore: OverallScore | None = None
    encouragement: str
    nextIdeas: List[ReviewNextIdea]


DEFAULT_MODEL = "gemini-2.5-flash"
