"""
Script Writer Module - Multi-Backend Support

Supports multiple AI providers:
- FREE: Ollama (local), Groq, Google Gemini
- PAID: Claude (Anthropic), OpenAI

Usage:
    # FREE - Using Ollama (local, unlimited)
    writer = ScriptWriter(provider="ollama", model="llama3.2")

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
    2. Run: ollama pull llama3.2
    3. Start: ollama serve
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
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
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key.")
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Claude provider: {model}")

    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


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
        "ollama": (OllamaProvider, "llama3.2"),
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
    narration: str          # What to say
    screen_action: str      # What to show on screen
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

    SCRIPT_PROMPT_TEMPLATE = """You are an expert YouTube scriptwriter specializing in educational tutorial content.

Write a complete YouTube tutorial script for the following topic:

**Topic:** {topic}
**Target Duration:** {duration} minutes
**Style:** {style}
**Target Audience:** {audience}

## Requirements:

1. **HOOK (0-10 seconds):** Start with an attention-grabbing statement or question
2. **INTRO (10-30 seconds):** Explain what viewers will learn
3. **MAIN CONTENT:** Step-by-step tutorial with clear explanations
4. **OUTRO (last 20 seconds):** Call to action, subscribe reminder

## Output Format:

Return a JSON object with this exact structure:
```json
{{
    "title": "SEO-optimized video title (under 60 chars)",
    "description": "YouTube description with timestamps and keywords (200-300 words)",
    "tags": ["tag1", "tag2", "tag3", "..."],
    "thumbnail_idea": "Brief description of an eye-catching thumbnail",
    "sections": [
        {{
            "timestamp": "00:00-00:10",
            "section_type": "hook",
            "narration": "What to say (spoken text)",
            "screen_action": "What to show on screen",
            "duration_seconds": 10
        }}
    ]
}}
```

## Guidelines:
- Use conversational, engaging language
- Include specific examples and code snippets where relevant
- Break complex concepts into simple steps
- Add personality but stay professional
- Mention subscribing and liking naturally
- Include timestamps in the description

Write the complete script now:"""

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

    def generate_script(
        self,
        topic: str,
        duration_minutes: int = 10,
        style: str = "educational",
        audience: str = "beginners"
    ) -> VideoScript:
        """
        Generate a complete YouTube video script.

        Args:
            topic: The tutorial topic
            duration_minutes: Target video duration in minutes
            style: Script style (educational, casual, professional)
            audience: Target audience (beginners, intermediate, advanced)

        Returns:
            VideoScript object with all sections
        """
        logger.info(f"Generating script for: {topic}")

        prompt = self.SCRIPT_PROMPT_TEMPLATE.format(
            topic=topic,
            duration=duration_minutes,
            style=style,
            audience=audience
        )

        try:
            # Use the AI provider to generate content
            content = self.ai.generate(prompt, max_tokens=4096)

            # Parse JSON from response
            script_data = self._parse_json_response(content)

            # Convert to VideoScript object
            video_script = self._create_video_script(script_data)

            logger.success(f"Script generated: {len(video_script.sections)} sections")
            return video_script

        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            raise

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from Claude's response."""
        # Try to find JSON in the response
        try:
            # First, try direct JSON parsing
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            json_str = content[start:end].strip()
            return json.loads(json_str)

        if "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            json_str = content[start:end].strip()
            return json.loads(json_str)

        # Try to find JSON object in text
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            return json.loads(json_str)

        raise ValueError("Could not parse JSON from response")

    def _create_video_script(self, data: Dict[str, Any]) -> VideoScript:
        """Create VideoScript object from parsed data."""
        sections = []

        for section_data in data.get("sections", []):
            section = ScriptSection(
                timestamp=section_data.get("timestamp", "00:00-00:00"),
                section_type=section_data.get("section_type", "content"),
                narration=section_data.get("narration", ""),
                screen_action=section_data.get("screen_action", ""),
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
        except:
            pass

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
