from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import CharacterEntry

ERA_LABELS: dict[str, str] = {
    "modern": "현대",
    "nineties": "1990년대",
    "seventies-eighties": "1970-80년대",
    "joseon": "조선시대/전통",
    "future": "미래/가상",
}

SEASON_LABELS: dict[str, str] = {
    "spring": "봄",
    "summer": "여름",
    "autumn": "가을",
    "winter": "겨울",
}


def build_panel_context(panels: list[str]) -> str:
    if not panels:
        return "없음"
    return "\n".join(panels)


def build_character_context(characters: list["CharacterEntry"] | None) -> str:
    if not characters:
        return "없음"
    lines = []
    for char in characters:
        desc = f" - {char.description}" if char.description else ""
        metadata = f"\n  [캐릭터 메타데이터: {char.metadata}]" if char.metadata else ""
        has_image = " [이미지 첨부됨]" if char.imageData else ""
        lines.append(f"• {char.name}{desc}{has_image}{metadata}")
    return "\n".join(lines)


def build_world_setting(era: str | None, season: str | None) -> str:
    parts = []
    if era and era != "any":
        parts.append(f"시대: {ERA_LABELS.get(era, era)}")
    if season and season != "any":
        parts.append(f"계절: {SEASON_LABELS.get(season, season)}")
    return ", ".join(parts) if parts else "없음"


def build_prompt(
    topic: str,
    genre: str,
    style: str,
    era: str | None,
    season: str | None,
    characters: list["CharacterEntry"] | None,
    script_summary: str,
    panel_context: str,
    has_panel_images: bool = False,
    has_character_images: bool = False,
) -> tuple[str, str]:
    character_context = build_character_context(characters)
    world_setting = build_world_setting(era, season)
    panel_count = len(panel_context.split("\n")) if panel_context and panel_context != "없음" else 0

    image_analysis_section = ""
    if has_panel_images or has_character_images:
        image_analysis_section = """

## Visual Analysis Guidelines (IMPORTANT - Images are provided)
You have access to the actual panel images and/or character sheet images. Analyze them carefully:

### Panel Image Analysis:
- Composition: Rule of thirds, focal points, visual flow between panels
- Color palette: Harmony, mood consistency, contrast usage
- Art style consistency: Line weight, shading technique, detail level
- Character rendering: Proportions, expressions, pose dynamics
- Background integration: Environment detail, depth perception
- Text/dialogue placement: Readability, visual balance

### Character Consistency Analysis:
- Compare character appearances across panels
- Check facial features, hairstyle, clothing consistency
- Evaluate body proportions and posture consistency
- Note any intentional style variations vs inconsistencies

### Visual Storytelling:
- Does the visual flow guide the reader naturally?
- Are emotional beats supported by visual composition?
- Is the pacing enhanced by panel sizes and layouts?
"""

    system_prompt = f"""You are a senior webtoon editor at a major Korean webtoon platform (Naver Webtoon, Kakao Page level) with 15+ years of experience.

## Your Expertise:
- Visual storytelling and panel composition
- Character design and consistency analysis
- Emotional pacing and narrative flow
- Korean webtoon market trends and reader preferences
- Art style evaluation across genres (romance, action, comedy, drama, horror, fantasy)

## Review Philosophy:
1. **Specific & Actionable**: Reference exact panels (e.g., "패널 2에서...")
2. **Encouraging but Honest**: Celebrate uniqueness while providing growth paths
3. **Creator-Intent Focused**: Understand what the creator tried to achieve
4. **Market-Aware**: Consider what makes content shareable and engaging

## Evaluation Criteria (Score 1-10 for each):

### 1. Story Flow (스토리 흐름) - 기승전결
- Opening hook: Does it grab attention in the first panel?
- Development: Is information revealed at the right pace?
- Climax: Is there a clear emotional peak?
- Resolution: Does it leave a satisfying or intriguing impression?

### 2. Character Appeal (캐릭터 매력)
- Visual distinctiveness: Can characters be recognized at a glance?
- Personality expression: Do expressions and poses convey personality?
- Relatability: Can readers connect emotionally?
- Consistency: Do characters look the same across panels?

### 3. Dialogue Quality (대사 품질)
- Natural flow: Does it sound like real conversation?
- Character voice: Does each character have a distinct way of speaking?
- Emotional resonance: Do key lines hit emotionally?
- Economy: Is every word necessary?

### 4. Visual Direction (시각 연출)
- Panel composition: Are shots visually interesting?
- Emotional emphasis: Do visuals amplify the story beats?
- Readability: Is the flow intuitive?
- Style consistency: Is the art style maintained throughout?

### 5. World Building (세계관 구축)
- Setting integration: Does the environment feel lived-in?
- Era/season reflection: Is the time period/season clear?
- Atmospheric detail: Do backgrounds enhance mood?
{image_analysis_section}
Return JSON only. No markdown code blocks."""

    user_prompt = f"""## Webtoon Information

**Basic Info:**
- Topic: {topic or "Not specified"}
- Genre: {genre or "Not specified"}
- Art Style: {style or "Not specified"}
- World Setting: {world_setting}
- Panel Count: {panel_count} panels
- Has Panel Images: {"Yes - PLEASE ANALYZE THE IMAGES" if has_panel_images else "No"}
- Has Character Images: {"Yes - PLEASE ANALYZE CHARACTER SHEETS" if has_character_images else "No"}

**Characters:**
{character_context}

**Script Summary:**
{script_summary or "None provided"}

**Panel Details:**
{panel_context}

---

## Your Task

{"IMPORTANT: Panel images and/or character images are attached. Please analyze them visually for your review." if has_panel_images or has_character_images else "Note: No images provided, review based on text descriptions only."}

Provide a comprehensive review in the following JSON structure. ALL text must be in Korean:

{{
  "headline": "작품의 핵심 매력을 담은 칭찬형 헤드라인 (15자 이내)",

  "summary": "작품 전체를 아우르는 따뜻한 2-3문장 요약. 스토리, 캐릭터, 분위기를 언급.",

  "strengths": [
    "구체적인 강점 1 (어떤 패널/장면을 언급하며)",
    "구체적인 강점 2",
    "구체적인 강점 3",
    "구체적인 강점 4"
  ],

  "improvements": [
    "구체적인 개선점 1 (어떻게 개선할 수 있는지 포함)",
    "구체적인 개선점 2",
    "구체적인 개선점 3"
  ],

  "visualAnalysis": {{
    "artStyleConsistency": "아트 스타일 일관성에 대한 상세 분석 (2-3문장)",
    "colorPalette": "색감과 톤에 대한 분석 (2-3문장)",
    "composition": "패널 구도와 시각적 흐름 분석 (2-3문장)",
    "characterConsistency": "캐릭터 외형 일관성 분석 (2-3문장)"
  }},

  "panelFeedback": [
    {{"panel": 1, "strength": "이 패널의 강점", "suggestion": "더 좋아질 수 있는 포인트"}},
    {{"panel": 2, "strength": "이 패널의 강점", "suggestion": "더 좋아질 수 있는 포인트"}},
    {{"panel": 3, "strength": "이 패널의 강점", "suggestion": "더 좋아질 수 있는 포인트"}},
    {{"panel": 4, "strength": "이 패널의 강점", "suggestion": "더 좋아질 수 있는 포인트"}}
  ],

  "characterAnalysis": [
    {{"name": "캐릭터이름", "appeal": "이 캐릭터의 매력 포인트", "consistency": "외형/성격 일관성 평가"}}
  ],

  "storyStructure": {{
    "opening": "도입부 평가 (첫 패널의 훅, 관심 유발 여부)",
    "development": "전개부 평가 (정보 전달 속도, 긴장감 구축)",
    "climax": "절정부 평가 (감정적 정점, 임팩트)",
    "resolution": "결말부 평가 (여운, 다음 이야기 기대감)"
  }},

  "overallScore": {{
    "story": 8,
    "visual": 7,
    "character": 8,
    "overall": 8
  }},

  "encouragement": "작가님의 다음 작품을 기대하게 만드는 따뜻하고 구체적인 응원 메시지 (3-4문장)",

  "nextIdeas": [
    {{
      "title": "추천 작품 제목",
      "topic": "주제/로그라인",
      "genre": "추천 장르",
      "style": "추천 스타일",
      "hook": "이 작품과 연결되는 추천 이유"
    }},
    {{
      "title": "또 다른 추천 작품 제목",
      "topic": "주제/로그라인",
      "genre": "추천 장르",
      "style": "추천 스타일",
      "hook": "이 작품과 연결되는 추천 이유"
    }}
  ]
}}

Remember:
- ALL values must be in Korean
- Be specific - reference exact panels and moments
- Scores should reflect honest assessment (not all 10s)
- Panel feedback should cover all {panel_count} panels
- Character analysis should cover all named characters"""

    return system_prompt, user_prompt
