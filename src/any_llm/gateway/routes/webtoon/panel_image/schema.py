from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ReferenceEntry(BaseModel):
    base64: str
    mimeType: str | None = None
    purpose: Literal["background", "character", "style", "previous_panel"] | None = None


class CharacterSheetMetadataEntry(BaseModel):
    name: str
    metadata: str | dict[str, Any] | None = None


class PanelMetadataEntryCharacter(BaseModel):
    name: str
    outfit: str | None = None
    accessories: list[str] = []
    hair: str | None = None
    props: list[str] = []
    pose: str | None = None
    notes: str | None = None


class PanelMetadataEntry(BaseModel):
    summary: str | None = None
    characters: list[PanelMetadataEntryCharacter] | None = None
    background: str | None = None
    lighting: str | None = None
    changes: list[str] = []
    notes: list[str] = []


AspectRatioType = Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
ResolutionType = Literal["1K", "2K", "4K"]
AnalysisLevelType = Literal["fast", "full"]


class PanelRequest(BaseModel):
    scene: str
    dialogue: str | None = None
    characters: list[str]
    style: str
    panelNumber: int
    era: str | None = None
    season: str | None = None
    characterDescriptions: list[str] | None = None
    characterImages: list[str] | None = None
    styleDoc: str | None = None
    sceneElements: dict[str, str] | None = None
    previousPanels: list[dict[str, Any]] | None = None
    characterSheetMetadata: list[CharacterSheetMetadataEntry] | None = None
    characterGenerationMode: Literal["ai", "caricature"] | None = None
    characterCaricatureStrengths: list[str] | None = None
    resolution: ResolutionType | None = None
    aspectRatio: AspectRatioType | None = None
    revisionNote: str | None = None
    references: list[ReferenceEntry] | None = None
    analysisLevel: AnalysisLevelType | None = None


class PanelImageResponse(BaseModel):
    success: Literal[True]
    imageUrl: str
    imageBase64: str
    mimeType: str
    metadata: str
    text: str
    aspectRatio: AspectRatioType
    resolution: ResolutionType
    model: str
    panelNumber: int


class StatusUpdate(BaseModel):
    stage: str
    message: str
