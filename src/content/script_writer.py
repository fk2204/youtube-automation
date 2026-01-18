"""
Script Writer Module - Multi-Backend Support

Supports multiple AI providers:
- FREE: Ollama (local), Groq, Google Gemini
- PAID: Claude (Anthropic), OpenAI

Usage:
    # FREE - Using Ollama (local, unlimited)
    writer = ScriptWriter(provider="ollama", model="llama3.2:1b")

    # FREE - Using Groq (cloud, 30 req/min free)
    writer = ScriptWriter(provider="groq", api_key="your-key")

    # PAID - Using Claude (best quality)
    writer = ScriptWriter(provider="claude", api_key="your-key")

    script = writer.generate_script(
        topic="How to build a REST API with Python",
        style="educational",
        duration_minutes=10
    )
"""

import os
import json
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ============================================================
# AI Provider Backends
# ============================================================

class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        pass


class OllamaProvider(AIProvider):
    """
    FREE - Run LLMs locally with Ollama.

    Setup:
    1. Install Ollama: https://ollama.ai/download
    2. Run: ollama pull llama3.2:1b
    3. Start: ollama serve
    """

    def __init__(self, model: str = "llama3.2:1b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        logger.info(f"Ollama provider: {model} at {base_url}")

    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]


class GroqProvider(AIProvider):
    """
    FREE TIER - Groq cloud with Llama 3.3 (30 req/min free).

    Setup:
    1. Get API key: https://console.groq.com/
    2. Set GROQ_API_KEY environment variable
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY or pass api_key.")
        logger.info(f"Groq provider: {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError))
    )
    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class GeminiProvider(AIProvider):
    """
    FREE TIER - Google Gemini (15 req/min free).

    Setup:
    1. Get API key: https://aistudio.google.com/apikey
    2. Set GOOGLE_API_KEY environment variable
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key.")
        logger.info(f"Gemini provider: {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError))
    )
    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens}
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


class ClaudeProvider(AIProvider):
    """
    PAID - Claude API (best quality).

    Setup:
    1. Get API key: https://console.anthropic.com/
    2. Set ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        try:
            from anthropic import Anthropic, APIConnectionError, APITimeoutError, RateLimitError
            self._api_connection_error = APIConnectionError
            self._api_timeout_error = APITimeoutError
            self._rate_limit_error = RateLimitError
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key.")
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Claude provider: {model}")

    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        return self._generate_with_retry(prompt, max_tokens)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _generate_with_retry(self, prompt: str, max_tokens: int) -> str:
        """Generate with retry logic for network errors (not auth errors)."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except self._api_connection_error as e:
            # Re-raise as ConnectionError for retry
            raise ConnectionError(str(e)) from e
        except self._api_timeout_error as e:
            # Re-raise as TimeoutError for retry
            raise TimeoutError(str(e)) from e
        except self._rate_limit_error as e:
            # Re-raise as ConnectionError for retry (rate limits are temporary)
            raise ConnectionError(str(e)) from e


class OpenAIProvider(AIProvider):
    """
    PAID - OpenAI GPT-4.

    Setup:
    1. Get API key: https://platform.openai.com/
    2. Set OPENAI_API_KEY environment variable
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
        logger.info(f"OpenAI provider: {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError))
    )
    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# Provider factory
def get_provider(
    provider: str = "ollama",
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> AIProvider:
    """
    Get an AI provider instance.

    Args:
        provider: Provider name (ollama, groq, gemini, claude, openai)
        api_key: API key (not needed for ollama)
        model: Model name override
    """
    providers = {
        "ollama": (OllamaProvider, "llama3.2:1b"),
        "groq": (GroqProvider, "llama-3.3-70b-versatile"),
        "gemini": (GeminiProvider, "gemini-1.5-flash"),
        "claude": (ClaudeProvider, "claude-sonnet-4-20250514"),
        "openai": (OpenAIProvider, "gpt-4o"),
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Use: {list(providers.keys())}")

    provider_class, default_model = providers[provider]
    model = model or default_model

    if provider == "ollama":
        return provider_class(model=model)
    else:
        return provider_class(api_key=api_key, model=model)


@dataclass
class ScriptSection:
    """Represents a section of the video script."""
    timestamp: str          # e.g., "00:00-00:15"
    section_type: str       # hook, intro, content, outro
    title: str              # Section title
    narration: str          # What to say
    screen_action: str      # What to show on screen
    keywords: List[str]     # Keywords for stock footage search
    duration_seconds: int   # Duration of this section


@dataclass
class VideoScript:
    """Complete video script with all sections."""
    title: str
    description: str
    tags: List[str]
    sections: List[ScriptSection]
    total_duration: int     # Total duration in seconds
    thumbnail_idea: str     # Suggestion for thumbnail


class ScriptWriter:
    """
    YouTube script generator with multi-provider support.

    FREE options:
    - Ollama (local, unlimited) - Best for development
    - Groq (cloud, 30 req/min) - Fast, good quality
    - Gemini (cloud, 15 req/min) - Google's free tier

    PAID options (better quality):
    - Claude (Anthropic) - Best reasoning
    - OpenAI (GPT-4) - Most popular
    """

    SCRIPT_PROMPT_TEMPLATE = """You are an expert YouTube scriptwriter creating VIRAL faceless videos.

Write a complete {duration}-minute YouTube script for:

**Topic:** {topic}
**Style:** {style}
**Target Audience:** {audience}

## VIRAL VIDEO STRUCTURE (IMPORTANT):

1. **HOOK (0-15 seconds):** Pattern interrupt - shocking fact, question, or bold claim
2. **PROBLEM (15-45 seconds):** Identify pain point, create curiosity
3. **PROMISE (45-60 seconds):** Tease what they'll learn
4. **MAIN CONTENT (3-5 minutes):** 5-7 key points with stories/examples
5. **PAYOFF (30 seconds):** Deliver the key insight
6. **CTA (15 seconds):** Subscribe, comment, like

## ENGAGEMENT TECHNIQUES:
- Use "you" frequently to connect with viewer
- Add cliffhangers between sections ("But here's what most people miss...")
- Include specific numbers and statistics
- Tell micro-stories within points
- Use rhetorical questions
- Create curiosity loops

## Output JSON Format:
```json
{{
    "title": "VIRAL title with numbers/power words (under 60 chars)",
    "description": "SEO description with timestamps (200 words)",
    "tags": ["10-15 relevant tags"],
    "thumbnail_idea": "Eye-catching thumbnail concept",
    "sections": [
        {{
            "timestamp": "00:00-00:15",
            "section_type": "hook",
            "title": "Section Title",
            "narration": "Spoken text (50-100 words per section)",
            "screen_action": "Visual description for B-roll",
            "keywords": ["3-5 keywords for stock footage"],
            "duration_seconds": 15
        }}
    ]
}}
```

## CRITICAL REQUIREMENTS:
- Generate 8-12 sections for {duration} minute video
- Each section needs KEYWORDS for stock footage matching
- Narration should total {word_count} words (~{duration} minutes when spoken)
- Make it feel like a documentary, not a lecture
- Include emotional hooks and power words

Write the COMPLETE viral script now:"""

    # Longer detailed prompt for better videos
    VIRAL_SCRIPT_PROMPT = """You are a viral content strategist creating faceless YouTube videos that get millions of views.

Create a {duration}-MINUTE script about: {topic}

## NICHE-SPECIFIC GUIDELINES:
{niche_guide}

## VIRAL FORMULA:
The first 15 seconds determine if someone watches. Use pattern interrupts:
- Start mid-thought: "...and that's exactly why most people fail"
- Bold claim: "This one strategy made me $X in Y days"
- Question: "What if everything you knew about X was wrong?"

## STORY STRUCTURE:
Every point should follow: HOOK → TENSION → RESOLUTION
- Hook: Grab attention with a surprising fact
- Tension: Build curiosity, create stakes
- Resolution: Deliver the insight

## SECTIONS REQUIRED:
1. HOOK (15s) - Pattern interrupt, shocking opener
2. CONTEXT (30s) - Set up the problem/topic
3. POINT 1 (45-60s) - First major insight with story
4. POINT 2 (45-60s) - Second insight with example
5. POINT 3 (45-60s) - Third insight with proof
6. POINT 4 (45-60s) - Fourth insight (optional for longer videos)
7. POINT 5 (45-60s) - Fifth insight (optional)
8. TWIST (30s) - Unexpected perspective or revelation
9. CONCLUSION (30s) - Tie it all together
10. CTA (15s) - Call to action

## OUTPUT FORMAT:
Return ONLY valid JSON:
{{
    "title": "VIRAL title - numbers, power words, curiosity gap",
    "description": "YouTube description with timestamps and keywords",
    "tags": ["list", "of", "15", "SEO", "tags"],
    "thumbnail_idea": "High-contrast, faces/emotions, bold text overlay",
    "sections": [
        {{
            "timestamp": "00:00-00:15",
            "section_type": "hook",
            "title": "The Hidden Truth",
            "narration": "Full spoken text for this section...",
            "screen_action": "Dramatic footage of...",
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "duration_seconds": 15
        }}
    ]
}}

Generate the full {duration}-minute script now:"""

    def __init__(
        self,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the script writer.

        Args:
            provider: AI provider (ollama, groq, gemini, claude, openai)
            api_key: API key (not needed for ollama)
            model: Model override (uses provider default if not specified)

        Examples:
            # FREE - Local Ollama
            writer = ScriptWriter(provider="ollama")

            # FREE - Groq cloud
            writer = ScriptWriter(provider="groq", api_key="your-key")

            # PAID - Claude (best quality)
            writer = ScriptWriter(provider="claude", api_key="your-key")
        """
        self.provider_name = provider
        self.ai = get_provider(provider=provider, api_key=api_key, model=model)
        logger.info(f"ScriptWriter initialized with provider: {provider}")

    # Niche-specific content guidelines
    NICHE_GUIDES = {
        "finance": """
- Use specific dollar amounts and percentages
- Reference real investment strategies
- Mention common money mistakes people make
- Include actionable steps viewers can take today
- Tone: Authoritative but approachable""",
        "psychology": """
- Reference psychological studies and experiments
- Use terms like "cognitive bias", "subconscious", "manipulation"
- Include real-world examples of psychological principles
- Make viewers feel like insiders learning secrets
- Tone: Mysterious, intriguing, slightly dark""",
        "storytelling": """
- Build suspense and tension throughout
- Use vivid sensory details
- Include unexpected twists
- Create emotional connection to characters
- Tone: Dramatic, immersive, documentary-style""",
        "default": """
- Use clear, engaging language
- Include specific examples
- Build curiosity throughout
- End with a satisfying payoff
- Tone: Professional but conversational"""
    }

    def generate_script(
        self,
        topic: str,
        duration_minutes: int = 5,
        style: str = "educational",
        audience: str = "general",
        niche: str = "default"
    ) -> VideoScript:
        """
        Generate a complete YouTube video script.

        Args:
            topic: The video topic
            duration_minutes: Target video duration (5-6 minutes recommended)
            style: Script style (educational, documentary, storytelling)
            audience: Target audience
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            VideoScript object with all sections
        """
        logger.info(f"Generating {duration_minutes}-min script for: {topic}")

        # Calculate target word count (150 words per minute for natural speech)
        word_count = duration_minutes * 150

        # Get niche-specific guidance
        niche_guide = self.NICHE_GUIDES.get(niche, self.NICHE_GUIDES["default"])

        prompt = self.SCRIPT_PROMPT_TEMPLATE.format(
            topic=topic,
            duration=duration_minutes,
            style=style,
            audience=audience,
            word_count=word_count,
            niche_guide=niche_guide
        )

        try:
            # Use the AI provider to generate content
            content = self.ai.generate(prompt, max_tokens=6000)

            # Parse JSON from response
            script_data = self._parse_json_response(content)

            # Convert to VideoScript object
            video_script = self._create_video_script(script_data)

            logger.success(f"Script generated: {len(video_script.sections)} sections")
            return video_script

        except (requests.RequestException, ConnectionError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Complex script failed, trying simple format: {e}")

            # Fallback to simpler prompt for smaller models
            return self._generate_simple_script(topic, duration_minutes, niche)

    def _generate_simple_script(
        self,
        topic: str,
        duration_minutes: int,
        niche: str
    ) -> VideoScript:
        """Generate a simpler script format for smaller LLMs."""
        logger.info(f"Using simple script generator for: {topic}")

        # Simple prompt that smaller models can handle
        prompt = f"""Write a {duration_minutes}-minute YouTube video script about: {topic}

Write 5-8 paragraphs of narration text. Each paragraph should be about 50-80 words.

Include:
1. An attention-grabbing opening
2. 3-5 main points with examples
3. A conclusion with call to action

Just write the narration text, no JSON or formatting needed. Write naturally as if speaking to the viewer."""

        try:
            content = self.ai.generate(prompt, max_tokens=3000)

            # Create sections from paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]

            if not paragraphs:
                paragraphs = [content]

            sections = []
            time_offset = 0

            # Section types based on position
            section_types = ['hook', 'intro'] + ['content'] * (len(paragraphs) - 3) + ['conclusion', 'outro']

            for i, para in enumerate(paragraphs[:10]):  # Max 10 sections
                # Estimate duration (150 words per minute)
                word_count = len(para.split())
                duration = max(15, int(word_count / 2.5))  # ~150 wpm

                section_type = section_types[i] if i < len(section_types) else 'content'

                # Extract keywords from paragraph
                words = para.lower().split()
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but',
                             'in', 'on', 'at', 'to', 'for', 'of', 'with', 'you', 'your', 'this',
                             'that', 'it', 'they', 'we', 'i', 'my', 'be', 'have', 'has', 'can'}
                keywords = list(set([w for w in words if len(w) > 4 and w not in stop_words]))[:5]

                # Add niche-related keywords
                niche_keywords = {
                    'finance': ['money', 'wealth', 'invest', 'income', 'financial'],
                    'psychology': ['mind', 'brain', 'behavior', 'psychology', 'mental'],
                    'storytelling': ['mystery', 'story', 'crime', 'investigation', 'truth']
                }
                keywords.extend(niche_keywords.get(niche, ['documentary', 'educational'])[:2])

                start_time = time_offset
                end_time = time_offset + duration
                time_offset = end_time

                sections.append(ScriptSection(
                    timestamp=f"{start_time//60:02d}:{start_time%60:02d}-{end_time//60:02d}:{end_time%60:02d}",
                    section_type=section_type,
                    title=f"Section {i+1}",
                    narration=para,
                    screen_action=f"B-roll footage related to: {', '.join(keywords[:3])}",
                    keywords=keywords,
                    duration_seconds=duration
                ))

            total_duration = sum(s.duration_seconds for s in sections)

            # Generate simple title
            title = topic[:55] + "..." if len(topic) > 55 else topic

            return VideoScript(
                title=title,
                description=f"In this video, we explore {topic}. Subscribe for more content!",
                tags=topic.lower().split()[:10] + [niche, 'educational', 'tutorial'],
                sections=sections,
                total_duration=total_duration,
                thumbnail_idea=f"Bold text with '{topic[:20]}' overlay"
            )

        except (requests.RequestException, ConnectionError, TimeoutError, KeyError, ValueError) as e:
            logger.error(f"Simple script generation also failed: {e}")
            raise

    def _fix_json(self, json_str: str) -> str:
        """Fix common JSON issues from LLM outputs."""
        import re
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        return json_str

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from AI response with error handling."""
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Extract JSON from content
        json_str = content

        # Try to extract JSON from markdown code block
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            json_str = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            json_str = content[start:end].strip()
        else:
            # Try to find JSON object in text
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]

        # Try parsing with fixes
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Apply fixes and try again
            fixed = self._fix_json(json_str)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not parse JSON from response")

    def _create_video_script(self, data: Dict[str, Any]) -> VideoScript:
        """Create VideoScript object from parsed data."""
        sections = []

        for section_data in data.get("sections", []):
            # Extract keywords or generate from narration
            keywords = section_data.get("keywords", [])
            if not keywords and section_data.get("narration"):
                # Auto-generate keywords from narration
                narration = section_data.get("narration", "")
                words = narration.lower().split()
                # Filter common words and get unique keywords
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                             'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                             'you', 'your', 'this', 'that', 'it', 'they', 'we', 'i', 'my'}
                keywords = [w for w in words if len(w) > 4 and w not in stop_words][:5]

            section = ScriptSection(
                timestamp=section_data.get("timestamp", "00:00-00:00"),
                section_type=section_data.get("section_type", "content"),
                title=section_data.get("title", ""),
                narration=section_data.get("narration", ""),
                screen_action=section_data.get("screen_action", ""),
                keywords=keywords,
                duration_seconds=section_data.get("duration_seconds", 30)
            )
            sections.append(section)

        total_duration = sum(s.duration_seconds for s in sections)

        return VideoScript(
            title=data.get("title", "Untitled Video"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            sections=sections,
            total_duration=total_duration,
            thumbnail_idea=data.get("thumbnail_idea", "")
        )

    def generate_title_variations(self, topic: str, count: int = 5) -> List[str]:
        """Generate multiple title options for A/B testing."""
        logger.info(f"Generating {count} title variations for: {topic}")

        prompt = f"""Generate {count} different YouTube video title options for this topic:

Topic: {topic}

Requirements:
- Each title should be under 60 characters
- Use power words and numbers where appropriate
- Make them click-worthy but not clickbait
- Include relevant keywords for SEO

Return as a JSON array of strings:
["Title 1", "Title 2", ...]"""

        content = self.ai.generate(prompt, max_tokens=500)
        try:
            # Parse the array
            if "[" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                return json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Title variations JSON parsing failed: {e}")

        return [f"Tutorial: {topic}"]  # Fallback

    def improve_script(
        self,
        script: VideoScript,
        feedback: str
    ) -> VideoScript:
        """Improve an existing script based on feedback."""
        logger.info("Improving script based on feedback...")

        # Convert script to JSON for the prompt
        script_json = json.dumps({
            "title": script.title,
            "sections": [
                {
                    "timestamp": s.timestamp,
                    "section_type": s.section_type,
                    "narration": s.narration,
                    "screen_action": s.screen_action
                }
                for s in script.sections
            ]
        }, indent=2)

        prompt = f"""Here's an existing YouTube script that needs improvement:

```json
{script_json}
```

Feedback to incorporate:
{feedback}

Please provide an improved version of the script in the same JSON format, addressing the feedback while keeping the overall structure.
"""

        content = self.ai.generate(prompt, max_tokens=4096)
        improved_data = self._parse_json_response(content)

        # Preserve original metadata, update sections
        improved_data["title"] = improved_data.get("title", script.title)
        improved_data["description"] = script.description
        improved_data["tags"] = script.tags
        improved_data["thumbnail_idea"] = script.thumbnail_idea

        return self._create_video_script(improved_data)

    def get_full_narration(self, script: VideoScript) -> str:
        """Extract just the narration text for TTS."""
        narration_parts = []

        for section in script.sections:
            if section.narration:
                narration_parts.append(section.narration)

        return "\n\n".join(narration_parts)


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Initialize writer
        writer = ScriptWriter()

        # Generate a script
        script = writer.generate_script(
            topic="How to Build a REST API with Python and FastAPI",
            duration_minutes=10,
            style="educational",
            audience="beginners"
        )

        print(f"\n{'='*60}")
        print(f"Title: {script.title}")
        print(f"{'='*60}")
        print(f"\nDescription:\n{script.description}")
        print(f"\nTags: {', '.join(script.tags)}")
        print(f"\nThumbnail Idea: {script.thumbnail_idea}")
        print(f"\nTotal Duration: {script.total_duration} seconds")

        print(f"\n{'='*60}")
        print("SCRIPT SECTIONS:")
        print(f"{'='*60}\n")

        for i, section in enumerate(script.sections, 1):
            print(f"[{section.timestamp}] {section.section_type.upper()}")
            print(f"  Narration: {section.narration[:100]}...")
            print(f"  Screen: {section.screen_action}")
            print()

        # Get full narration for TTS
        narration = writer.get_full_narration(script)
        print(f"\n{'='*60}")
        print("FULL NARRATION (for TTS):")
        print(f"{'='*60}\n")
        print(narration)

    asyncio.run(main())
