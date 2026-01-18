from __future__ import annotations

from typing import Literal, Sequence

from .schema import Language, SceneElements

ERA_IDS = ["any", "modern", "nineties", "seventies-eighties", "joseon", "future"]
SEASON_IDS = ["any", "spring", "summer", "autumn", "winter"]

LANGUAGE_LABELS: dict[Language, str] = {
    "ko": "한국어",
    "zh": "中文",
    "ja": "日本語",
}

def resolve_era_label(value: str | None) -> str | None:
    if not value:
        return None
    era = value if value in ERA_IDS else "any"
    if era == "any":
        return None
    return {
        "modern": "Modern",
        "nineties": "1990s",
        "seventies-eighties": "1970s-80s",
        "joseon": "Joseon / Traditional",
        "future": "Future / Virtual",
    }[era]

def resolve_season_label(value: str | None) -> str | None:
    if not value:
        return None
    season = value if value in SEASON_IDS else "any"
    if season == "any":
        return None
    return {
        "spring": "Spring",
        "summer": "Summer",
        "autumn": "Autumn",
        "winter": "Winter",
    }[season]

def build_world_setting_block(era_label: str | None, season_label: str | None) -> str:
    if not era_label and not season_label:
        return ""
    lines = ["Era/season guidance:"]
    if era_label:
        lines.append(f"- Era: {era_label}")
    if season_label:
        lines.append(f"- Season: {season_label}")
    lines.extend(
        [
            "- If an era is specified, subtly reflect it in tone and vocabulary.",
            "- Use season only as a mood hint.",
        ]
    )
    return "\n".join(lines)

def format_scene_elements(elements: SceneElements | None) -> str:
    if not elements:
        return ""
    return "\n".join(
        [
            "- Subject: " + elements.subject,
            "- Action: " + elements.action,
            "- Setting: " + elements.setting,
            "- Composition: " + elements.composition,
            "- Lighting: " + elements.lighting,
            "- Style: " + elements.style,
        ]
    )

def build_prompt(
    speakers: Sequence[str],
    language: Language,
    panelNumber: int | None,
    topic: str | None,
    genre: str | None,
    style: str | None,
    scene: str | None,
    scene_elements: SceneElements | None,
    world_setting_block: str,
    character_mode: Literal["ai", "caricature"],
) -> str:
    language_label = LANGUAGE_LABELS[language]
    elements_text = format_scene_elements(scene_elements)
    caricature_line = (
        "- In caricature mode, keep dialogue short and colloquial.\n" if character_mode == "caricature" else ""
    )
    multi_speaker = len(speakers) >= 2
    conversation_rules = (
        "- IMPORTANT: Create a back-and-forth conversation where speakers alternate turns (티키타카 style).\n"
        "- Each speaker must respond to or react to what the other said.\n"
        "- Speakers should interact naturally, not deliver separate monologues."
        if multi_speaker
        else "- The single speaker may have 2-3 lines expressing their thoughts or narration."
    )
    lines = [
        "You are a writer creating webtoon dialogue.",
        "",
        "Rules:",
        "- Output must be a single JSON object.",
        '- Use only the key "dialogueLines".',
        f"- Write naturally in {language_label}.",
        "- Produce 2–4 short dialogue lines.",
        "- Each line should be one sentence.",
        "- Use only the provided speaker list.",
        conversation_rules,
        "- No action tags like '(sigh)'.",
        "- Avoid overly explanatory lines; reveal emotion and relationships.",
        caricature_line.strip(),
        "",
        "Context:",
        f"Panel number: {panelNumber if panelNumber is not None else '미정'}",
        f"Topic: {topic or '미정'}",
        f"Genre: {genre or '미정'}",
        f"Style: {style or '미정'}",
        f"Scene description: {scene or '미정'}",
    ]
    if elements_text:
        lines.extend(["", "Six elements:", elements_text])
    if world_setting_block:
        lines.extend(["", world_setting_block])
    lines.extend(["", "Speaker list: " + ", ".join(speakers), "", "Output format:", "{", '  "dialogueLines": [', '    { "speaker": "Speaker", "text": "Dialogue" }', "  ]", "}"])
    return "\n".join(part for part in lines if part is not None)
