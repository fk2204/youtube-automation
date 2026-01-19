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
import re
import json
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import best practices module for validation and compliance
try:
    from src.utils.best_practices import (
        validate_title,
        validate_hook,
        get_best_practices,
        suggest_improvements,
        pre_publish_checklist,
        get_niche_metrics,
        ValidationResult,
        PrePublishChecklist,
    )
    BEST_PRACTICES_AVAILABLE = True
except ImportError:
    BEST_PRACTICES_AVAILABLE = False
    logger.warning("best_practices module not available - validation features disabled")


# ============================================================
# Title Optimization Constants (for YouTube CTR improvement)
# Based on competitor analysis of top faceless channels (Jan 2026)
# ============================================================

# Power words that increase click-through rates
# Updated with psychology triggers: loss aversion, curiosity gap, social proof
POWER_WORDS = [
    # Authority/Trust triggers
    "Ultimate", "Proven", "Expert", "Complete", "Definitive", "Essential",
    # Curiosity/Intrigue triggers
    "Secret", "Hidden", "Shocking", "Unbelievable", "Surprising", "Actually",
    "Finally", "Revealed", "Truth", "Nobody", "Untold", "Dark",
    # Urgency/FOMO triggers
    "Critical", "Instant", "Guaranteed", "Revolutionary", "Breakthrough",
    # Emotional triggers
    "Powerful", "Incredible", "Amazing", "Massive", "Terrifying", "Genius"
]

# ============================================================
# VIRAL TITLE TEMPLATES (From Competitor Analysis - Jan 2026)
# These patterns consistently achieve high CTR across niches
# ============================================================

VIRAL_TITLE_TEMPLATES = {
    # FINANCE NICHE (money_blueprints) - Channels: The Swedish Investor, Practical Wisdom
    "finance": [
        "How {company} Makes Money (Business Explained)",
        "Why {stock} Will {multiplier}x in {year}",
        "{number} Passive Income Ideas That Actually Work in {year}",
        "The Truth About {financial_trend}",
        "{number} Money Mistakes Costing You ${amount}/Year",
        "I Analyzed {number} {investment_type} - Here's What I Found",
        "How to Turn ${small_amount} Into ${large_amount} ({timeframe})",
        "Why {percentage}% of Investors Lose Money (And How to Win)",
        "The ${amount} Side Hustle Nobody Talks About",
        "{company} Stock: Buy, Sell, or Hold in {year}?",
        "Warren Buffett's {number} Rules of Investing",
        "The Hidden Truth About {financial_product}",
    ],

    # PSYCHOLOGY NICHE (mind_unlocked) - Channels: Psych2Go, Brainy Dose, The Infographics Show
    "psychology": [
        "{number} Signs of {personality_type}",
        "Why Your Brain {brain_action}",
        "Dark Psychology Tricks {group} Uses Against You",
        "The Science Behind {behavior}",
        "{number} Psychological Tricks That Work on Everyone",
        "Why {percentage}% of People Fall for This Manipulation",
        "The {cognitive_bias} That's Ruining Your Life",
        "What Your {trait} Says About You (Psychology)",
        "{number} Body Language Signs Someone Is {emotion}",
        "How to Read Anyone in {number} Seconds",
        "The Psychology of {topic}: Why You {action}",
        "{number} Things Only {personality_type} Will Understand",
    ],

    # STORYTELLING NICHE (untold_stories) - Channels: JCS, Truly Criminal, Lazy Masquerade
    "storytelling": [
        "The Untold Story of {subject}",
        "How {company} Went From $0 to ${valuation}",
        "The Rise and Fall of {brand}",
        "What Happened to {forgotten_entity}?",
        "The {person} Who {achievement}",
        "Why {company} Is Secretly {adjective}",
        "The Dark Side of {successful_entity}",
        "{company}'s Biggest Mistake Ever",
        "The True Story Behind {famous_thing}",
        "Inside {company}'s Secret {noun}",
        "The {year} {event} That Changed Everything",
        "Why Everyone Was Wrong About {subject}",
    ],
}

# ============================================================
# HOOK FORMULAS (Based on retention data from top channels)
# First 5 seconds determine 80% of retention success
# ============================================================

HOOK_FORMULAS = {
    # Universal high-performing hooks
    "universal": [
        # Pattern Interrupt (23% higher retention)
        "Stop. What you're about to see changes everything...",
        "Forget everything you've been told about {topic}...",
        "This is the one thing nobody tells you about {topic}...",

        # Loss Aversion (strongest trigger - "must click")
        "You're losing ${amount} every year without knowing it...",
        "Right now, {percentage}% of your {resource} is being wasted...",
        "This mistake is costing you {consequence}...",

        # Curiosity Gap (32% increase in watch time)
        "What if I told you {surprising_claim}?",
        "In the next {duration}, I'll show you {promise}...",
        "There's a reason {authority} doesn't want you to know this...",

        # Stats Shock (high credibility)
        "Only {percentage}% of people know this. Here's why it matters...",
        "{number} out of {total} people fail at this. Here's how to be different...",
        "In {year}, {shocking_stat}. And it's getting worse...",

        # Story Lead (immediate tension)
        "In {year}, someone discovered something that changed everything...",
        "What happened next shocked everyone...",
        "Nobody believed it would work. They were wrong...",
    ],

    # Finance-specific hooks
    "finance": [
        "If you invested ${amount} in {investment} {timeframe} ago, you'd have ${result} today...",
        "The difference between 7% and 9% returns? ${amount} over 30 years...",
        "Wall Street doesn't want you to know this, but...",
        "{percentage}% of your paycheck is disappearing, and here's where it goes...",
        "In {year}, a man turned ${small_amount} into ${large_amount} using this exact method...",
    ],

    # Psychology-specific hooks
    "psychology": [
        "Your brain is lying to you right now, and you don't even know it...",
        "What I'm about to show you is used by FBI interrogators...",
        "Scientists discovered something terrifying about the human mind...",
        "In the next {duration}, you'll be able to read anyone's thoughts...",
        "They've been using this against you since you were {age} years old...",
    ],

    # Storytelling-specific hooks
    "storytelling": [
        "The door slammed. He had exactly {seconds} seconds to decide...",
        "Nobody knows why he did it. But what happened next shocked the world...",
        "To this day, no one can explain what really happened...",
        "He had everything. By morning, he had nothing...",
        "What I'm about to tell you is the true story they tried to bury...",
    ],
}

# Convert number words to digits (digits perform better in titles)
NUMBER_WORDS_TO_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20"
}

# Topics that benefit from having the current year in the title
YEAR_RELEVANT_TOPICS = [
    "best", "top", "guide", "tutorial", "how to", "tips", "tricks",
    "strategy", "strategies", "review", "comparison", "vs", "trends",
    "update", "new", "latest", "current", "modern", "today"
]


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
class ChapterMarker:
    """Represents a chapter marker for YouTube chapters."""
    timestamp_seconds: int  # Timestamp in seconds
    title: str              # Chapter title

    def to_timestamp_string(self) -> str:
        """Convert to YouTube timestamp format (MM:SS or HH:MM:SS)."""
        hours = self.timestamp_seconds // 3600
        minutes = (self.timestamp_seconds % 3600) // 60
        seconds = self.timestamp_seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class RetentionPoint:
    """Represents a predicted retention spike/dip in the video."""
    timestamp_seconds: int      # When this occurs
    retention_type: str         # "hook", "micro_payoff", "open_loop", "cliffhanger", "cta"
    description: str            # What happens at this point
    expected_impact: str        # "high", "medium", "low" - expected retention impact


@dataclass
class VideoScript:
    """Complete video script with all sections."""
    title: str
    description: str
    tags: List[str]
    sections: List[ScriptSection]
    total_duration: int         # Total duration in seconds
    thumbnail_idea: str         # Suggestion for thumbnail
    # New YouTube best practices fields
    hook_text: str = ""                     # First 5 seconds hook text
    chapter_markers: List[ChapterMarker] = None  # YouTube chapter markers
    estimated_retention_points: List[RetentionPoint] = None  # Retention predictions
    is_short: bool = False                  # Whether this is a YouTube Short

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.chapter_markers is None:
            self.chapter_markers = []
        if self.estimated_retention_points is None:
            self.estimated_retention_points = []
        # Auto-extract hook text from first section if not provided
        if not self.hook_text and self.sections:
            first_section = self.sections[0]
            if first_section.narration:
                # Get first ~15 seconds worth of text (approximately 40 words)
                words = first_section.narration.split()
                self.hook_text = " ".join(words[:40])

    def get_chapters_string(self) -> str:
        """Get YouTube-formatted chapters string for description."""
        if not self.chapter_markers:
            return ""
        lines = [f"{cm.to_timestamp_string()} {cm.title}" for cm in self.chapter_markers]
        return "\n".join(lines)

    def get_retention_summary(self) -> str:
        """Get a summary of retention optimization points."""
        if not self.estimated_retention_points:
            return "No retention points analyzed"

        summary_lines = []
        for rp in self.estimated_retention_points:
            timestamp = f"{rp.timestamp_seconds // 60}:{rp.timestamp_seconds % 60:02d}"
            summary_lines.append(f"[{timestamp}] {rp.retention_type.upper()}: {rp.description} ({rp.expected_impact} impact)")
        return "\n".join(summary_lines)


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

    SCRIPT_PROMPT_TEMPLATE = """You are an expert YouTube scriptwriter creating VIRAL faceless videos optimized for maximum retention.

Write a complete {duration}-minute YouTube script for:

**Topic:** {topic}
**Style:** {style}
**Target Audience:** {audience}

## YOUTUBE BEST PRACTICES - RETENTION OPTIMIZATION:

### 1. STRONG HOOK (First 5 Seconds) - CRITICAL FOR RETENTION:
Choose ONE proven hook formula and execute it in the FIRST 5 SECONDS:

**Pattern Interrupt Hooks:**
- **"The Shocking Truth"** - Counterintuitive fact that challenges beliefs
  "Everything you've been told about [topic] is completely wrong..."
- **"Story Lead"** - Compelling micro-story that creates immediate tension
  "In 2019, a man lost everything... but what happened next changed the industry forever."
- **"Question Stack"** - 3 rapid curiosity questions in first 5 seconds
  "What if I told you...? Would you believe...? And what would you do if...?"
- **"Stats Shock"** - Lead with a surprising, specific statistic
  "97.3% of people who try this method fail - here's why you won't."
- **"Bold Statement"** - Make a provocative claim that demands attention
  "This single technique is worth more than a college degree..."

### 2. MICRO-PAYOFFS (Every 30-60 Seconds):
Deliver small value moments throughout to maintain retention:
- Quick insight: "Here's a secret most people miss..."
- Surprising fact: "What you might not know is..."
- Actionable tip: "The first thing you can do RIGHT NOW is..."
- Story beat: "And that's when everything changed..."
Mark these in the script as [MICRO-PAYOFF] for tracking.

### 3. OPEN LOOPS (Minimum 3 Per Video):
Create curiosity gaps that keep viewers watching:
- "I'll reveal the most important tip at the end - trust me, it's worth waiting for..."
- "But there's something even more shocking coming up in point three..."
- "The real secret? I'll tell you after we cover the basics..."
- "What happened next will surprise you - but first..."
Mark these as [OPEN-LOOP] and [LOOP-CLOSED] when resolved.

### 4. CHAPTERS/CLEAR STRUCTURE:
Create distinct sections for YouTube chapters:
- Use clear transitions between topics
- Each chapter should have its own mini-hook
- Chapter titles should be descriptive and keyword-rich
- First chapter timestamp MUST be 00:00

### 5. CALL-TO-ACTION PLACEMENT:
Strategic CTA positioning for maximum effect:
- **SOFT CTA at 30%**: "If you're finding this valuable, hit subscribe..."
- **ENGAGEMENT CTA at 50%**: "Comment below with your experience..."
- **FINAL CTA at 95%**: "Like and subscribe for more content like this..."
NEVER put CTA in first 30 seconds (kills retention).

## VIRAL VIDEO STRUCTURE (OPTIMIZED):

1. **HOOK (0-5 seconds):** Pattern interrupt - grab attention IMMEDIATELY
2. **CONTEXT (5-15 seconds):** Set up why this matters, plant FIRST open loop
3. **PROBLEM (15-45 seconds):** Identify pain point, create urgency
4. **PROMISE (45-60 seconds):** Tease the payoff - "By the end of this video, you'll know exactly..."
5. **MAIN CONTENT ({duration_content_mins} minutes):** 5-7 key points with MICRO-PAYOFF every 30-60 seconds
6. **TWIST/INSIGHT (30 seconds):** Deliver unexpected value, close open loops
7. **PAYOFF (30 seconds):** Deliver the promised key insight
8. **CTA (15 seconds):** Subscribe, comment with specific question, like

## ENGAGEMENT TECHNIQUES (MANDATORY):

### Micro-Cliffhangers (every 60-90 seconds):
- "But here's where it gets interesting..."
- "What happened next shocked everyone..."
- "And this is where most people make the fatal mistake..."
- "Wait - there's a catch..."

### Direct Address - Use "you" at least 3 times per minute:
- "You might be thinking..." / "When you try this..." / "Here's what you need to understand..."
- Address the viewer directly to create personal connection
- "This is where YOUR strategy changes..."

### Rhetorical Questions (every 30-45 seconds):
- "But what does this really mean for you?"
- "Have you ever wondered why...?"
- "Sound familiar?"
- "So why don't more people do this?"

### Specific Numbers (NEVER use vague words):
- Say "73% of people" not "most people"
- Say "$2,847 per month" not "thousands of dollars"
- Say "in exactly 17 days" not "in a few weeks"
- Say "4.7 times more effective" not "much more effective"

## NICHE-SPECIFIC GUIDELINES:
{niche_guide}

## Output JSON Format:
```json
{{
    "title": "VIRAL title with numbers/power words (under 60 chars)",
    "description": "SEO description with timestamps (200 words)",
    "tags": ["10-15 relevant tags"],
    "thumbnail_idea": "Eye-catching thumbnail concept",
    "hook_text": "Exact first 5 seconds of narration - the pattern interrupt hook",
    "chapter_markers": [
        {{"timestamp_seconds": 0, "title": "The Hook"}},
        {{"timestamp_seconds": 60, "title": "Chapter Title"}}
    ],
    "retention_points": [
        {{"timestamp_seconds": 30, "type": "micro_payoff", "description": "Quick insight about X"}},
        {{"timestamp_seconds": 45, "type": "open_loop", "description": "Tease about upcoming reveal"}},
        {{"timestamp_seconds": 180, "type": "cliffhanger", "description": "Transition tension builder"}}
    ],
    "sections": [
        {{
            "timestamp": "00:00-00:05",
            "section_type": "hook",
            "title": "The Hook",
            "narration": "Exact hook text (pattern interrupt, 5 seconds max)",
            "screen_action": "Visual description for B-roll",
            "keywords": ["3-5 keywords for stock footage"],
            "duration_seconds": 5
        }},
        {{
            "timestamp": "00:05-00:15",
            "section_type": "context",
            "title": "Why This Matters",
            "narration": "Context with first open loop planted",
            "screen_action": "Visual description",
            "keywords": ["keywords"],
            "duration_seconds": 10
        }}
    ]
}}
```

## CRITICAL REQUIREMENTS (Report 5 Best Practices):

### WORD COUNT (MANDATORY):
- **Target: 800-1500 words total** (Report 5 optimal range)
- This equals approximately {duration} minutes at 150 words/minute
- Scripts outside this range perform worse on retention

### STRUCTURE REQUIREMENTS:
- Generate 10-15 sections for {duration} minute video
- FIRST SECTION must be exactly 5 seconds - the HOOK (15-20 words max)
- Each section needs KEYWORDS for stock footage matching
- Make it feel like a documentary, not a lecture

### HOOK REQUIREMENT (First 5 Seconds):
- The opening hook MUST grab attention in EXACTLY 5 seconds
- Use a pattern interrupt from the formulas above
- This is the #1 factor for retention - get it right

### CTA PLACEMENT (STRATEGIC):
- **SOFT CTA at 30%** ({cta_30_percent}s): "If you're finding this valuable, hit subscribe..."
- **ENGAGEMENT CTA at 50%** ({cta_50_percent}s): "Comment below with your experience..."
- **FINAL CTA at 95%** ({cta_95_percent}s): "Like and subscribe for more content like this..."
- NEVER put CTA in first 30 seconds (kills retention)

### ENGAGEMENT ELEMENTS:
- Use at least 3 OPEN LOOPS throughout the script (mark with [OPEN-LOOP])
- Include a MICRO-PAYOFF every 30-60 seconds (mark key moments)
- Add a micro-cliffhanger every 60-90 seconds
- Add a rhetorical question every 30-45 seconds
- Include emotional hooks and power words

### METADATA:
- Include chapter_markers array for YouTube chapters feature
- Include retention_points array with types: "hook", "micro_payoff", "open_loop", "cliffhanger", "cta"

Write the COMPLETE viral script now (800-1500 words):"""

    # Longer detailed prompt for better videos
    VIRAL_SCRIPT_PROMPT = """You are a viral content strategist creating faceless YouTube videos that get millions of views.

Create a {duration}-MINUTE script about: {topic}

## PROVEN HOOK FORMULAS (Use one for your opening):

1. **"The Shocking Truth"** - Start with a counterintuitive fact
   "Everything you've been told about [topic] is completely wrong. And by the end of this video, you'll see exactly why."

2. **"Story Lead"** - Open with a compelling micro-story
   "In 2019, a man with nothing but $47 in his bank account discovered something that would change everything..."

3. **"Question Stack"** - 3 rapid curiosity questions
   "What if I told you there's a method that 97% of experts hide? Would you believe it takes just 12 minutes a day? And what if it could transform your entire approach to [topic]?"

4. **"Stats Shock"** - Lead with a surprising statistic
   "Only 2.3% of people who try [topic] ever succeed. But here's the crazy part - the ones who do all share one thing in common."

## NICHE-SPECIFIC GUIDELINES:
{niche_guide}

## ENGAGEMENT TECHNIQUES (MANDATORY):

### Open Loops - Use at least 2 throughout the script:
- "I'll reveal the most critical piece at the end - and trust me, it's worth waiting for..."
- "But there's something even more shocking I need to show you first..."
- "The third point changes everything - but you need context from the first two..."

### Micro-Cliffhangers (insert every 60-90 seconds):
- "But here's where it gets really interesting..."
- "What happened next shocked even the experts..."
- "And this is the exact moment when everything changed..."
- "Most people stop here. But what comes next is what separates the winners..."

### Direct Address - Use "you" frequently:
- "You're probably thinking..." / "When you apply this..." / "Here's what you need to understand..."
- "This affects you directly because..." / "Your results depend on..."

### Rhetorical Questions (every 30-45 seconds):
- "But why does this actually matter?"
- "Have you ever stopped to think about...?"
- "Sound familiar? That's exactly what I thought too."
- "So what makes this different from everything else you've tried?"

### Specific Numbers (NEVER be vague):
- "73.2% of people" NOT "most people"
- "$4,829 per month" NOT "thousands of dollars"
- "in exactly 23 days" NOT "in a few weeks"
- "3.7x more effective" NOT "much more effective"
- "studied 847 cases" NOT "studied many cases"

## STORY STRUCTURE:
Every point should follow: HOOK → TENSION → RESOLUTION
- Hook: Grab attention with a surprising fact or question
- Tension: Build curiosity, create stakes, what's at risk
- Resolution: Deliver the insight with specific, actionable detail

## SECTIONS REQUIRED:
1. HOOK (15s) - Use one of the proven hook formulas above
2. CONTEXT (30s) - Set up the problem, add an open loop
3. POINT 1 (45-60s) - First major insight with story + cliffhanger
4. POINT 2 (45-60s) - Second insight with example + rhetorical question
5. POINT 3 (45-60s) - Third insight with proof + cliffhanger
6. POINT 4 (45-60s) - Fourth insight (optional) + open loop callback
7. POINT 5 (45-60s) - Fifth insight (optional)
8. TWIST (30s) - Unexpected perspective, deliver the promised insight
9. CONCLUSION (30s) - Tie it all together, create urgency
10. CTA (15s) - Subscribe, comment with specific question, like

## OUTPUT FORMAT:
Return ONLY valid JSON:
{{
    "title": "VIRAL title - specific numbers, power words, curiosity gap",
    "description": "YouTube description with timestamps and keywords",
    "tags": ["list", "of", "15", "SEO", "tags"],
    "thumbnail_idea": "High-contrast, faces/emotions, bold text overlay with specific number",
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
        provider: str = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the script writer.

        Args:
            provider: AI provider (ollama, groq, gemini, claude, openai). Defaults to AI_PROVIDER env var or "ollama".
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
        # Use environment variable if provider not specified
        if provider is None:
            provider = os.getenv("AI_PROVIDER", "ollama")
        self.provider_name = provider
        self.ai = get_provider(provider=provider, api_key=api_key, model=model)
        logger.info(f"ScriptWriter initialized with provider: {provider}")

    # Niche-specific content guidelines
    # Updated with Competitor Analysis (January 2026)
    # Sources: The Swedish Investor, Practical Wisdom, Psych2Go, Brainy Dose, JCS, Truly Criminal
    NICHE_GUIDES = {
        "finance": """
FINANCE NICHE (money_blueprints) - AUTHORITATIVE, DATA-DRIVEN APPROACH
CPM Range: $10-22 (one of highest paying niches)
Top Competitors: The Swedish Investor (1M+ subs), Practical Wisdom (10M views on best videos)

### WINNING CONTENT TYPES (Report 5):
1. **Stock Analysis with Charts/Timelines** - Visual evidence builds trust
2. **Passive Income Explainers** - High search volume, evergreen content
3. **Business Documentaries** - "Documentary-business hybrids" perform exceptionally
4. **Budgeting Tutorials** - Actionable, saves viewers money
5. **Side Hustle Guides** - High engagement, shareable

### VIRAL TOPIC TEMPLATES:
- "How [Company] Makes Money" (e.g., "How Netflix Makes Money")
- "Why [Stock] Will 10x" (e.g., "Why NVIDIA Will 10x in 2026")
- "5 Passive Income Ideas That Actually Work"
- "The Truth About [Financial Trend]" (e.g., "The Truth About AI Stocks")
- "[X] Passive Income Streams I Use to Make $[Amount]/Month"
- "I Analyzed 100 [Stocks/Investments] - Here's What I Found"

### FINANCE HOOK FORMULAS (First 5 Seconds):
Choose the MOST authoritative hook style:

1. **"Data Bomb"** - Lead with shocking, specific financial data:
   "In 2024, 73.2% of investors lost money doing this one thing..."
   "$847 billion was lost last year because of this single mistake..."

2. **"Money Math"** - Show surprising calculations:
   "If you invested $500/month starting at age 25, you'd have $2.4 million by 60..."
   "The difference between 7% and 9% returns? $1.2 million over 30 years..."

3. **"Insider Secret"** - Financial industry revelation:
   "Wall Street doesn't want you to know this, but..."
   "The strategy hedge funds use that banks will never tell you..."

4. **"Loss Aversion"** - Highlight what they're losing:
   "Right now, you're losing $347 every month without realizing it..."
   "This hidden fee is costing the average investor $23,000 over their lifetime..."

### Specific Numbers (ALWAYS include):
- Exact dollar amounts: "$4,273/month" not "thousands per month"
- Precise percentages: "23.7% ROI" not "high returns"
- Time frames: "in 47 days" not "in a few weeks"
- Comparison ratios: "3.2x faster growth" not "much faster"

### ROI and Growth Metrics:
- Always mention potential ROI with realistic ranges
- Include compound growth examples: "If you invested $500/month for 7 years..."
- Reference inflation-adjusted returns when relevant
- Compare to traditional savings: "vs. the 0.5% your bank gives you"

### Actionable Steps (MANDATORY):
- End each major point with "Here's what you can do TODAY..."
- Provide specific platforms, tools, or accounts to open
- Include exact steps: "Step 1: Open a brokerage account. Step 2: Set up automatic $200 transfers..."
- Give specific allocation percentages: "Put 60% in index funds, 30% in bonds, 10% in high-risk"

### Money Mistakes to Reference:
- "The average person loses $847/year to this one mistake..."
- "92% of people do this wrong - and it costs them $12,000 over a decade"
- Use loss aversion: "You're currently losing $X by not doing Y"

### GROWTH TACTICS (Report 5):
- Use visual evidence (charts, citations, timelines)
- Reference credible sources for trust
- Create "documentary-business hybrids"
- Target: 100k subs achievable in 8 months with consistency

### MONETIZATION BEYOND ADS:
- Affiliate tools (brokerage links, apps)
- Patreon/membership for premium analysis
- Course/ebook upsells

### Tone: Authoritative expert who's sharing insider knowledge, slightly urgent""",

        "psychology": """
PSYCHOLOGY NICHE (mind_unlocked) - CURIOSITY-DRIVEN, "WHAT IF" APPROACH
Why It Works: One of the easiest and most profitable faceless niches
Top Competitors: Psych2Go (12.7M subs, 18M views on best videos), Brainy Dose (10M+ views), The Infographics Show (15M subs)

### WINNING CONTENT TYPES (Report 5):
1. **Brain Facts** - "Why you remember embarrassing moments" (high curiosity)
2. **Cognitive Biases** - "How colors affect your decisions" (practical value)
3. **Marketing Psychology** - "Psychology tricks used in marketing" (mass appeal)
4. **Dark Psychology** - Manipulation awareness content (high engagement)
5. **Behavior Science** - "Why humans do X" format (evergreen)

### VIRAL TOPIC TEMPLATES:
- "5 Signs of [Personality Type]" (e.g., "5 Signs of a Narcissist")
- "Why Your Brain [Does X]" (e.g., "Why Your Brain Hates Change")
- "Dark Psychology Tricks [Group] Uses" (e.g., "Dark Psychology Tricks Salespeople Use")
- "The Science Behind [Behavior]" (e.g., "The Science Behind Procrastination")
- "[X] Psychological Tricks That Work on Everyone"
- "Why [Percentage]% of People Fall for This Manipulation"
- "The [Cognitive Bias] That's Ruining Your Life"

### PSYCHOLOGY HOOK FORMULAS (First 5 Seconds):
Choose the MOST curiosity-driven hook style:

1. **"What If" Question** - Open with mind-bending possibility:
   "What if everything you believe about yourself is a lie your brain tells you?"
   "What if there was a technique that could read anyone's mind in 3 seconds?"

2. **"Dark Pattern Reveal"** - Expose hidden manipulation:
   "Your brain is being hacked right now, and you don't even know it..."
   "They've been using this against you since you were 5 years old..."

3. **"Forbidden Knowledge"** - Tease secret psychological insight:
   "This is the technique the FBI uses to detect liars..."
   "What I'm about to show you is used by CIA interrogators..."

4. **"Mind Hack"** - Promise psychological superpower:
   "In the next 30 seconds, I'll give you the ability to influence anyone..."
   "By the end of this video, you'll know exactly what people are thinking..."

5. **"Study Shock"** - Lead with counterintuitive research:
   "A Stanford study proved that 92% of people are wrong about this..."
   "Scientists discovered something terrifying about the human brain..."

### Study References (ALWAYS include):
- Name specific studies: "In a 2019 Stanford study with 847 participants..."
- Reference famous experiments: "Just like Milgram's obedience experiments showed..."
- Include researcher names: "Dr. Robert Cialdini's research proves..."
- Cite statistics: "78% of participants in the study exhibited..."

### "Dark Psychology" Intrigue Elements:
- Use phrases like: "What I'm about to reveal is used by elite negotiators..."
- Reference manipulation awareness: "Once you see this pattern, you can't unsee it"
- Forbidden knowledge angle: "This technique is so powerful it's banned in some contexts..."
- Self-defense framing: "Knowing this protects you from being manipulated"

### Psychological Terms to Weave In:
- Cognitive biases: anchoring, confirmation bias, availability heuristic
- Subconscious triggers: priming, framing effects, social proof
- Influence techniques: reciprocity, scarcity, authority, commitment
- Brain science: amygdala hijack, dopamine loops, pattern recognition

### Real-World Examples:
- "This is exactly how casinos keep you playing..."
- "Advertisers have used this against you for decades..."
- "Every successful salesperson knows this trick..."
- "Politicians exploit this bias during every election..."

### GROWTH TACTICS (Report 5):
- Keep scripts snappy (800-1500 words)
- Pair with background footage or animated text
- Research credible sources (journals, studies)
- Use pattern interrupts every 2-3 seconds

### Tone: Mysterious insider revealing secrets, slightly dark and intriguing, make viewers feel like they're learning forbidden knowledge""",

        "storytelling": """
STORYTELLING NICHE (untold_stories) - DRAMATIC TENSION, CLIFFHANGERS
Why It Works: Business stories attract curious viewers, high watch time
Top Competitors: JCS Criminal Psychology (5M+ subs), Truly Criminal (30+ min episodes), Mr. Nightmare (6M subs), Lazy Masquerade (1.7M subs)

### WINNING CONTENT TYPES (Report 5):
1. **Company Rise/Fall Documentaries** - "How [Company] Went From $0 to $Billions"
2. **Historical Events with Modern Lessons** - "What [Event] Teaches Us Today"
3. **Unsolved Mysteries** - High curiosity, excellent retention
4. **"How [Person] Built [Empire]"** - Founder stories, inspirational
5. **True Crime** (careful with monetization) - High engagement but CPM risk

### VIRAL TOPIC TEMPLATES:
- "The Untold Story of [Company/Person]" (e.g., "The Untold Story of Enron")
- "How [Company] Went From $0 to $Billions" (e.g., "How Amazon Went From Garage to Trillion")
- "The Rise and Fall of [Brand]" (e.g., "The Rise and Fall of Blockbuster")
- "What Happened to [Forgotten Company]" (e.g., "What Happened to MySpace")
- "The [Person] Who [Incredible Achievement]" (e.g., "The Man Who Fooled Wall Street")
- "Why [Company] Is Secretly [Adjective]" (e.g., "Why Google Is Secretly Terrifying")
- "The Dark Side of [Successful Company/Person]"

### STORYTELLING HOOK FORMULAS (First 5 Seconds):
Choose the MOST dramatic hook style:

1. **"In Media Res"** - Start in the middle of action:
   "The door slammed. He had exactly 30 seconds to make a choice that would change everything..."
   "She looked down at her hands. They were covered in blood. And she had no memory of the last 3 hours..."

2. **"Dramatic Tension"** - Create immediate stakes:
   "In 2019, a man made a decision that would cost him everything he loved..."
   "What happened in that room would haunt them for the rest of their lives..."

3. **"Mystery Question"** - Open with an unanswered question:
   "Nobody knows why he did it. But what happened next shocked the entire world..."
   "To this day, no one can explain what really happened that night..."

4. **"Countdown"** - Create time pressure:
   "He had exactly 4 minutes. Four minutes to save everything..."
   "The clock showed 11:47 PM. By midnight, three people would be dead..."

5. **"Cliffhanger Tease"** - Promise dramatic revelation:
   "What I'm about to tell you is the true story that inspired [famous movie]..."
   "The ending of this story will change how you see everything..."

### Suspense Building Techniques:
- Plant questions early: "But there was something he didn't know..."
- Use time pressure: "She had exactly 47 minutes to make a decision that would change everything..."
- False resolutions: "He thought it was over. He was wrong."
- Escalation: "And then things got worse..."

### Sensory Details (MANDATORY in every scene):
- Visual: "The dim fluorescent light flickered above the empty hallway..."
- Sound: "The only sound was the rhythmic drip of water echoing..."
- Physical: "His hands trembled as he reached for the envelope..."
- Emotional: "A cold knot formed in her stomach..."

### Mini-Cliffhangers (every 45-60 seconds):
- "What happened next would haunt him forever..."
- "She had no idea what was waiting for her..."
- "But the real story was just beginning..."
- "And that's when everything changed..."

### Character Connection:
- Give specific details: "Maria, a 34-year-old nurse from Ohio with two kids..."
- Show vulnerability: "He had failed at this exact thing three times before..."
- Internal dialogue: "She thought to herself: 'This is it. There's no going back.'"
- Relatable stakes: "Everything he'd worked for was about to disappear..."

### Twist Techniques:
- Perspective shift: "But what nobody knew was that the real villain was..."
- Time reveal: "What seemed like coincidence was actually planned 10 years in advance..."
- Hidden information: "There was one detail she left out of her confession..."

### GROWTH TACTICS (Report 5):
- Turn companies into mini-documentaries
- Use storytelling techniques (hooks, tension, resolution)
- Background footage + voiceover + animations
- Research history, find the narrative arc

### Tone: Documentary-style narrator, dramatic and immersive, treat the viewer as someone watching a thriller unfold""",

        "default": """
DEFAULT NICHE - ENGAGEMENT REQUIREMENTS:

### Specificity (ALWAYS be concrete):
- Use exact numbers: "147 people tested this" not "many people"
- Give time frames: "takes 23 minutes" not "takes some time"
- Provide examples: "like Netflix did in 2015" not "like some companies"

### Engagement Hooks:
- Curiosity gaps: "There's one thing that makes all the difference..."
- Contrarian angles: "Everything you've heard about this is wrong..."
- Stakes: "Getting this wrong costs most people years of wasted effort..."

### Structure Each Point:
- Open with surprising fact or question
- Build tension with what's at stake
- Resolve with actionable insight
- Transition with cliffhanger to next point

### Call to Action Elements:
- Specific comment prompt: "Tell me in the comments: which of these 3 did you not know?"
- Subscribe hook: "We post 3 videos like this every week..."
- Engagement driver: "If this changed how you think, smash that like button"

### Tone: Professional but conversational, like a knowledgeable friend sharing valuable insights"""
    }

    # ============================================================
    # YouTube Shorts-Specific Prompts
    # ============================================================

    SHORTS_PROMPT_TEMPLATE = """You are an expert YouTube Shorts scriptwriter creating VIRAL 30-45 second vertical videos.

**Topic:** {topic}
**Style:** {style}
**Niche:** {niche}

## YOUTUBE SHORTS BEST PRACTICES (CRITICAL):

### 1. HOOK (First 1-2 Seconds) - MOST IMPORTANT:
The hook MUST grab attention INSTANTLY. Use one of these:
- **Visual Interrupt**: Start mid-action or with something unexpected
- **Bold Claim**: "This one trick tripled my income..."
- **Direct Challenge**: "You've been doing this WRONG your entire life..."
- **Curiosity Gap**: "Nobody talks about this, but..."
- **Numbers Hook**: "In exactly 17 seconds, I'll show you..."

### 2. OPTIMAL LENGTH: 20-45 Seconds
- Sweet spot is 30-35 seconds for maximum retention
- Shorter = easier to watch multiple times = more loops = algorithm boost
- Never exceed 58 seconds (60 sec is the max)

### 3. PATTERN INTERRUPTS (Every 2-3 Seconds):
Keep viewers engaged by changing something every 2-3 seconds:
- Change in vocal tone/energy
- New visual element mentioned
- Question posed
- Surprising fact
- Sound effect cue (mark with [SFX])
- Quick list items

### 4. LOOP-FRIENDLY ENDING:
End in a way that flows BACK to the beginning:
- "And that's exactly why..." (connects to opening claim)
- "Try it and comment below" (viewer wants to rewatch to remember)
- "Wait, did I mention..." (creates desire to rewatch)
- End mid-thought to encourage loop
- The last word/phrase should connect to the first

### 5. VISUAL CUES FOR SHORTS:
- Text overlay suggestions: [TEXT: "key phrase"]
- Visual transitions: [CUT], [ZOOM], [PAN]
- B-roll markers: [BROLL: description]
- On-screen graphics: [GRAPHIC: chart/arrow/emoji]

## NICHE-SPECIFIC SHORTS GUIDANCE:
{niche_guide}

## OUTPUT FORMAT (JSON):
```json
{{
    "title": "Catchy title under 40 chars with hook",
    "hook_text": "Exact words for first 1-2 seconds - the pattern interrupt",
    "description": "Brief description with hashtags",
    "hashtags": ["#short1", "#short2", "#short3"],
    "sections": [
        {{
            "timestamp": "00:00-00:03",
            "section_type": "hook",
            "narration": "Spoken text for this moment",
            "visual_cue": "[TEXT: Key Point] + [BROLL: relevant imagery]",
            "pattern_interrupt_note": "What changes here to keep attention"
        }}
    ],
    "loop_ending_note": "How ending connects back to beginning",
    "estimated_watch_time_seconds": 32
}}
```

## CRITICAL SHORT-FORM RULES:
1. Every sentence should be 5-10 words MAX
2. Use active voice ("Do this" not "This should be done")
3. Include at least ONE specific number
4. End every 2-3 seconds with a micro-payoff or tease
5. Total word count: {word_count} words (for ~{duration} seconds)
6. Make it re-watchable (hidden detail, fast info, satisfying ending)

Generate the Shorts script now:"""

    # Shorts-specific niche guides (condensed for short format)
    SHORTS_NICHE_GUIDES = {
        "finance": """
FINANCE SHORTS - HIGH CTR TACTICS:
- Lead with shocking money stat: "$2,847 in 3 days..."
- Show the math visually: [GRAPHIC: calculation appearing]
- Specific actionable tip they can do TODAY
- End with "Save this for later" or controversial claim
- Pattern interrupt: Calculator sound [SFX], money counter [BROLL]""",

        "psychology": """
PSYCHOLOGY SHORTS - VIRAL HOOKS:
- Start with "Your brain is lying to you..."
- Reference famous experiments in 5 words
- Use "Manipulation technique #X" format
- Dark pattern reveal: "They don't want you to know..."
- Pattern interrupt: Brain imagery [BROLL], dramatic pause""",

        "storytelling": """
STORYTELLING SHORTS - TENSION MAXIMIZING:
- Open mid-crisis: "He had 30 seconds to decide..."
- Fast cuts between tension moments
- Leave on cliffhanger OR satisfying twist
- Use countdown pressure: "In 5 seconds..."
- Pattern interrupt: Sound effects, dramatic zooms""",

        "default": """
GENERAL SHORTS - ENGAGEMENT TACTICS:
- Bold claim in first sentence
- List format works: "3 things you need to know..."
- End with question to drive comments
- Pattern interrupt: Visual changes, tone shifts
- Make them want to watch again"""
    }

    def generate_short_script(
        self,
        topic: str,
        duration_seconds: int = 30,
        style: str = "educational",
        niche: str = "default"
    ) -> VideoScript:
        """
        Generate a YouTube Shorts script (vertical, 20-60 seconds).

        Args:
            topic: The video topic
            duration_seconds: Target duration (20-45 seconds recommended)
            style: Script style (educational, entertaining, storytelling)
            niche: Content niche (finance, psychology, storytelling, default)

        Returns:
            VideoScript object optimized for Shorts format
        """
        logger.info(f"Generating {duration_seconds}s Short script for: {topic}")

        # Clamp duration to valid Shorts range
        duration_seconds = max(15, min(58, duration_seconds))

        # Calculate word count (approximately 2.5 words per second for fast-paced Shorts)
        word_count = int(duration_seconds * 2.5)

        # Get niche-specific guidance
        niche_guide = self.SHORTS_NICHE_GUIDES.get(niche, self.SHORTS_NICHE_GUIDES["default"])

        prompt = self.SHORTS_PROMPT_TEMPLATE.format(
            topic=topic,
            style=style,
            niche=niche,
            niche_guide=niche_guide,
            word_count=word_count,
            duration=duration_seconds
        )

        try:
            content = self.ai.generate(prompt, max_tokens=2000)
            script_data = self._parse_json_response(content)
            video_script = self._create_short_script(script_data, duration_seconds)

            if len(video_script.sections) == 0:
                logger.warning("Short script generated with 0 sections, retrying with simple format")
                return self._generate_simple_short_script(topic, duration_seconds, niche)

            logger.success(f"Short script generated: {len(video_script.sections)} sections, {video_script.total_duration}s")
            return video_script

        except (requests.RequestException, ConnectionError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Complex short script failed, trying simple format: {e}")
            return self._generate_simple_short_script(topic, duration_seconds, niche)

    def _create_short_script(self, data: Dict[str, Any], target_duration: int) -> VideoScript:
        """Create VideoScript object from parsed Shorts data."""
        sections = []

        for section_data in data.get("sections", []):
            # Calculate duration from timestamp if available
            timestamp = section_data.get("timestamp", "00:00-00:03")
            try:
                parts = timestamp.split("-")
                start_parts = parts[0].split(":")
                end_parts = parts[1].split(":")
                start_sec = int(start_parts[0]) * 60 + int(start_parts[1])
                end_sec = int(end_parts[0]) * 60 + int(end_parts[1])
                duration = end_sec - start_sec
            except (IndexError, ValueError):
                duration = 3  # Default 3 seconds per section for Shorts

            # Build screen_action from visual_cue
            visual_cue = section_data.get("visual_cue", "")
            pattern_note = section_data.get("pattern_interrupt_note", "")
            screen_action = f"{visual_cue} | Pattern: {pattern_note}" if pattern_note else visual_cue

            # Extract keywords for B-roll
            narration = section_data.get("narration", "")
            words = narration.lower().split()
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but',
                         'in', 'on', 'at', 'to', 'for', 'of', 'with', 'you', 'your', 'this',
                         'that', 'it', 'they', 'we', 'i', 'my', 'be', 'have', 'has', 'can'}
            keywords = [w for w in words if len(w) > 4 and w not in stop_words][:3]

            section = ScriptSection(
                timestamp=timestamp,
                section_type=section_data.get("section_type", "content"),
                title=section_data.get("title", ""),
                narration=narration,
                screen_action=screen_action,
                keywords=keywords,
                duration_seconds=duration
            )
            sections.append(section)

        total_duration = sum(s.duration_seconds for s in sections)

        # Extract hook text
        hook_text = data.get("hook_text", "")
        if not hook_text and sections:
            hook_text = sections[0].narration

        # Create retention points for Shorts
        retention_points = []
        current_time = 0
        for i, section in enumerate(sections):
            if section.section_type == "hook":
                retention_points.append(RetentionPoint(
                    timestamp_seconds=current_time,
                    retention_type="hook",
                    description=f"Opening hook: {section.narration[:50]}...",
                    expected_impact="high"
                ))
            current_time += section.duration_seconds

        # Add loop ending as retention point
        if data.get("loop_ending_note"):
            retention_points.append(RetentionPoint(
                timestamp_seconds=total_duration - 2,
                retention_type="open_loop",
                description=f"Loop ending: {data['loop_ending_note']}",
                expected_impact="high"
            ))

        return VideoScript(
            title=data.get("title", topic[:40]),
            description=data.get("description", ""),
            tags=data.get("hashtags", []),
            sections=sections,
            total_duration=total_duration,
            thumbnail_idea="Vertical format, bold text overlay, expressive face",
            hook_text=hook_text,
            chapter_markers=[],  # Shorts don't have chapters
            estimated_retention_points=retention_points,
            is_short=True
        )

    def _generate_simple_short_script(
        self,
        topic: str,
        duration_seconds: int,
        niche: str
    ) -> VideoScript:
        """Generate a simpler Shorts script for smaller LLMs."""
        logger.info(f"Using simple Short script generator for: {topic}")

        niche_hints = {
            "finance": "Include a specific dollar amount or percentage. End with 'Save this.'",
            "psychology": "Start with 'Your brain...' or 'They don't want you to know...'",
            "storytelling": "Open mid-action. End on tension or twist.",
            "default": "Bold opening claim. Specific number. Question to end."
        }
        niche_hint = niche_hints.get(niche, niche_hints["default"])

        word_count = int(duration_seconds * 2.5)

        prompt = f"""Write a {duration_seconds}-second YouTube Shorts script about: {topic}

Write exactly {word_count} words total, split into 3-4 short segments.

RULES:
1. FIRST 2-3 WORDS must grab attention (pattern interrupt)
2. Every sentence is MAX 8 words
3. Include ONE specific number
4. End in a way that loops back to the beginning
5. {niche_hint}

Example format:
[0-3s] Hook sentence here.
[3-10s] Quick point one. Point two.
[10-20s] Main insight with specific detail.
[20-{duration_seconds}s] Ending that loops. Try this now.

Write the script now:"""

        try:
            content = self.ai.generate(prompt, max_tokens=500)

            # Parse the simple format
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            sections = []
            current_time = 0

            for line in lines:
                # Try to extract timestamp from line
                if line.startswith('['):
                    try:
                        bracket_end = line.index(']')
                        timestamp_part = line[1:bracket_end]
                        text = line[bracket_end + 1:].strip()

                        # Parse timestamp
                        if '-' in timestamp_part:
                            times = timestamp_part.replace('s', '').split('-')
                            start = int(times[0])
                            end = int(times[1])
                            duration = end - start
                        else:
                            duration = 5
                    except (ValueError, IndexError):
                        text = line
                        duration = 5
                else:
                    text = line
                    duration = 5

                if text and len(text) > 5:
                    section_type = "hook" if current_time == 0 else "content"
                    sections.append(ScriptSection(
                        timestamp=f"00:{current_time:02d}-00:{current_time + duration:02d}",
                        section_type=section_type,
                        title=f"Segment {len(sections) + 1}",
                        narration=text,
                        screen_action="[Dynamic visual, text overlay]",
                        keywords=[topic.lower().split()[0], niche] if topic else [niche],
                        duration_seconds=duration
                    ))
                    current_time += duration

            if not sections:
                # Fallback: create single section
                sections.append(ScriptSection(
                    timestamp="00:00-00:30",
                    section_type="content",
                    title="Main",
                    narration=content[:200],
                    screen_action="[Engaging visuals]",
                    keywords=[niche, "shorts"],
                    duration_seconds=30
                ))

            total_duration = sum(s.duration_seconds for s in sections)

            return VideoScript(
                title=topic[:40] if len(topic) > 40 else topic,
                description=f"#shorts #{niche} #viral",
                tags=["shorts", niche, topic.lower().split()[0] if topic else "video"],
                sections=sections,
                total_duration=total_duration,
                thumbnail_idea="Bold text, vertical format",
                hook_text=sections[0].narration if sections else "",
                chapter_markers=[],
                estimated_retention_points=[
                    RetentionPoint(0, "hook", "Opening hook", "high"),
                    RetentionPoint(total_duration - 3, "open_loop", "Loop ending", "high")
                ],
                is_short=True
            )

        except (requests.RequestException, ConnectionError, TimeoutError, KeyError, ValueError) as e:
            logger.error(f"Simple short script generation failed: {e}")
            raise

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

        # Calculate target word count (Report 5: 800-1500 words optimal)
        # Use 150 words per minute for natural speech, but enforce 800-1500 range
        word_count = duration_minutes * 150
        word_count = max(800, min(1500, word_count))  # Enforce Report 5 optimal range

        # Calculate content duration (total minus intro/outro overhead)
        duration_content_mins = max(1, duration_minutes - 2)  # Reserve ~2 min for hook/context/cta

        # Calculate CTA placement timestamps (Report 5: 30%, 50%, 95%)
        total_seconds = duration_minutes * 60
        cta_30_percent = int(total_seconds * 0.30)
        cta_50_percent = int(total_seconds * 0.50)
        cta_95_percent = int(total_seconds * 0.95)

        # Get niche-specific guidance
        niche_guide = self.NICHE_GUIDES.get(niche, self.NICHE_GUIDES["default"])

        prompt = self.SCRIPT_PROMPT_TEMPLATE.format(
            topic=topic,
            duration=duration_minutes,
            duration_content_mins=duration_content_mins,
            style=style,
            audience=audience,
            word_count=word_count,
            niche_guide=niche_guide,
            cta_30_percent=cta_30_percent,
            cta_50_percent=cta_50_percent,
            cta_95_percent=cta_95_percent
        )

        try:
            # Use the AI provider to generate content
            content = self.ai.generate(prompt, max_tokens=6000)

            # Parse JSON from response
            script_data = self._parse_json_response(content)

            # Convert to VideoScript object
            video_script = self._create_video_script(script_data)

            # BUG FIX #2: Validate that script has at least 1 section
            if len(video_script.sections) == 0:
                logger.warning("Script generated with 0 sections, retrying with simple format")
                return self._generate_simple_script(topic, duration_minutes, niche)

            # Validate against best practices and log results
            video_script = self._validate_and_enhance_script(video_script, niche)

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

        # Get niche-specific guidance for the simple prompt
        niche_hints = {
            "finance": "Include specific dollar amounts (like $2,847), exact percentages (like 23.7% ROI), and actionable steps the viewer can take TODAY.",
            "psychology": "Reference psychological studies, use terms like 'cognitive bias' and 'subconscious triggers', and make it feel like revealing forbidden knowledge.",
            "storytelling": "Use vivid sensory details, build suspense with mini-cliffhangers, and make the viewer feel like they're watching a thriller unfold.",
            "default": "Use specific numbers instead of vague words, ask rhetorical questions, and address the viewer directly with 'you'."
        }
        niche_hint = niche_hints.get(niche, niche_hints["default"])

        # Enhanced simple prompt with engagement techniques
        prompt = f"""Write a {duration_minutes}-minute YouTube video script about: {topic}

Write 5-8 paragraphs of narration text. Each paragraph should be about 50-80 words.

CRITICAL ENGAGEMENT RULES:
1. HOOK: Start with ONE of these proven formulas:
   - "Shocking Truth": Start with a counterintuitive fact ("Everything you've been told about {topic} is wrong...")
   - "Story Lead": Open with a micro-story ("In 2019, someone discovered something that changed everything...")
   - "Question Stack": Ask 3 rapid questions ("What if...? Would you believe...? And what if...?")
   - "Stats Shock": Lead with a surprising statistic ("Only 2.3% of people who try this succeed...")

2. OPEN LOOPS: Include at least one phrase like "I'll reveal the most important tip at the end..." or "But there's something even more shocking coming up..."

3. CLIFFHANGERS: Between major points, add phrases like "But here's where it gets interesting..." or "What most people don't realize is..."

4. DIRECT ADDRESS: Use "you" at least 3 times per paragraph

5. RHETORICAL QUESTIONS: Ask questions like "Sound familiar?" or "Have you ever wondered why...?" every 30-45 seconds

6. SPECIFIC NUMBERS: Say "73% of people" NOT "most people". Say "$2,847" NOT "thousands of dollars"

7. NICHE SPECIFICS: {niche_hint}

Write the narration text now. No JSON or formatting needed. Write naturally as if speaking directly to the viewer."""

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

            # BUG FIX #2: Validate that we have at least 1 section
            if len(sections) == 0:
                logger.error("Simple script generation produced 0 sections")
                raise ValueError("Failed to generate any script sections")

            total_duration = sum(s.duration_seconds for s in sections)

            # Generate simple title
            title = topic[:55] + "..." if len(topic) > 55 else topic

            # Extract hook text from first section
            hook_text = ""
            if sections:
                first_words = sections[0].narration.split()[:15]
                hook_text = " ".join(first_words)

            # Generate chapter markers from sections
            chapter_markers = []
            current_time = 0
            for section in sections:
                if section.section_type in ["hook", "intro", "content", "conclusion"]:
                    chapter_markers.append(ChapterMarker(
                        timestamp_seconds=current_time,
                        title=section.title or section.section_type.capitalize()
                    ))
                current_time += section.duration_seconds

            # Generate retention points
            retention_points = []
            current_time = 0
            for i, section in enumerate(sections):
                if section.section_type == "hook":
                    retention_points.append(RetentionPoint(
                        timestamp_seconds=current_time,
                        retention_type="hook",
                        description=f"Opening hook",
                        expected_impact="high"
                    ))
                elif i > 0 and current_time % 45 < section.duration_seconds:
                    retention_points.append(RetentionPoint(
                        timestamp_seconds=current_time,
                        retention_type="micro_payoff",
                        description=f"Value point: {section.title}",
                        expected_impact="medium"
                    ))
                current_time += section.duration_seconds

            return VideoScript(
                title=title,
                description=f"In this video, we explore {topic}. Subscribe for more content!",
                tags=topic.lower().split()[:10] + [niche, 'educational', 'tutorial'],
                sections=sections,
                total_duration=total_duration,
                thumbnail_idea=f"Bold text with '{topic[:20]}' overlay",
                hook_text=hook_text,
                chapter_markers=chapter_markers,
                estimated_retention_points=retention_points,
                is_short=False
            )

        except (requests.RequestException, ConnectionError, TimeoutError, KeyError, ValueError) as e:
            logger.error(f"Simple script generation also failed: {e}")
            raise

    def _fix_json(self, json_str: str) -> str:
        """Fix common JSON issues from LLM outputs."""
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)

        # Fix single quotes used instead of double quotes for keys
        json_str = re.sub(r'(?<=[{,\s])(\w+)(?=\s*:)', r'""', json_str)

        # Remove any duplicate double quotes that may have been introduced
        json_str = re.sub(r'""(\w+)""', r'""', json_str)

        # Try to fix common truncation - add missing closing brackets
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')

        if open_braces > 0:
            json_str = json_str.rstrip() + '}' * open_braces
        if open_brackets > 0:
            json_str = json_str.rstrip() + ']' * open_brackets

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

        # Extract hook text from data or from first section
        hook_text = data.get("hook_text", "")
        if not hook_text and sections:
            first_section = sections[0]
            if first_section.narration:
                # Get first ~5 seconds worth (~12 words)
                words = first_section.narration.split()
                hook_text = " ".join(words[:15])

        # Parse chapter markers from data
        chapter_markers = []
        for cm_data in data.get("chapter_markers", []):
            if isinstance(cm_data, dict):
                chapter_markers.append(ChapterMarker(
                    timestamp_seconds=cm_data.get("timestamp_seconds", 0),
                    title=cm_data.get("title", "Chapter")
                ))

        # Auto-generate chapter markers from sections if not provided
        if not chapter_markers and sections:
            current_time = 0
            for section in sections:
                if section.section_type in ["hook", "intro", "content", "conclusion", "cta"]:
                    chapter_markers.append(ChapterMarker(
                        timestamp_seconds=current_time,
                        title=section.title or section.section_type.capitalize()
                    ))
                current_time += section.duration_seconds

        # Parse retention points from data
        retention_points = []
        for rp_data in data.get("retention_points", []):
            if isinstance(rp_data, dict):
                retention_points.append(RetentionPoint(
                    timestamp_seconds=rp_data.get("timestamp_seconds", 0),
                    retention_type=rp_data.get("type", "micro_payoff"),
                    description=rp_data.get("description", ""),
                    expected_impact=rp_data.get("expected_impact", "medium")
                ))

        # Auto-generate retention points if not provided
        if not retention_points and sections:
            current_time = 0
            for i, section in enumerate(sections):
                # Add hook point
                if section.section_type == "hook":
                    retention_points.append(RetentionPoint(
                        timestamp_seconds=current_time,
                        retention_type="hook",
                        description=f"Opening hook: {section.narration[:50]}...",
                        expected_impact="high"
                    ))
                # Add micro-payoff every 30-60 seconds
                elif current_time > 0 and current_time % 45 < section.duration_seconds:
                    retention_points.append(RetentionPoint(
                        timestamp_seconds=current_time,
                        retention_type="micro_payoff",
                        description=f"Value delivery: {section.title}",
                        expected_impact="medium"
                    ))
                # Add CTA points
                if section.section_type == "cta":
                    retention_points.append(RetentionPoint(
                        timestamp_seconds=current_time,
                        retention_type="cta",
                        description="Call to action",
                        expected_impact="medium"
                    ))
                current_time += section.duration_seconds

        return VideoScript(
            title=data.get("title", "Untitled Video"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            sections=sections,
            total_duration=total_duration,
            thumbnail_idea=data.get("thumbnail_idea", ""),
            hook_text=hook_text,
            chapter_markers=chapter_markers,
            estimated_retention_points=retention_points,
            is_short=False
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

    def optimize_title(self, title: str, primary_keyword: str = "") -> str:
        """
        Optimize a title for YouTube CTR and SEO.

        Best practices applied:
        - Primary keyword at the beginning (if provided)
        - Convert number words to digits ("Seven Tips" -> "7 Tips")
        - Add current year when topic is relevant
        - Add power words when title lacks impact
        - Keep titles within 50-60 character optimal range

        Args:
            title: The original title
            primary_keyword: Optional keyword to put at the start

        Returns:
            Optimized title string
        """
        optimized = title.strip()
        current_year = str(datetime.now().year)

        # Convert number words to digits (digits get higher CTR)
        title_lower = optimized.lower()
        for word, digit in NUMBER_WORDS_TO_DIGITS.items():
            # Match whole words only
            pattern = r'\b' + word + r'\b'
            if re.search(pattern, title_lower, re.IGNORECASE):
                optimized = re.sub(pattern, digit, optimized, flags=re.IGNORECASE)

        # Add current year if topic is relevant and year not already present
        if current_year not in optimized:
            topic_lower = optimized.lower()
            if any(topic in topic_lower for topic in YEAR_RELEVANT_TOPICS):
                # Add year at the end in brackets if there's room
                if len(optimized) <= 50:
                    optimized = f"{optimized} [{current_year}]"

        # Put primary keyword at the beginning if provided
        if primary_keyword and not optimized.lower().startswith(primary_keyword.lower()):
            # Check if keyword is in the title already
            if primary_keyword.lower() in optimized.lower():
                # Remove it and prepend
                pattern = re.compile(re.escape(primary_keyword), re.IGNORECASE)
                optimized = pattern.sub('', optimized).strip()
                optimized = re.sub(r'\s+', ' ', optimized)  # Clean up double spaces
                optimized = re.sub(r'^[:\-–—]\s*', '', optimized)  # Clean up leading punctuation
            optimized = f"{primary_keyword}: {optimized}"

        # Add a power word if title seems weak (no power words and no numbers)
        has_power_word = any(pw.lower() in optimized.lower() for pw in POWER_WORDS)
        has_number = any(c.isdigit() for c in optimized)
        if not has_power_word and not has_number and len(optimized) < 45:
            # Add "Ultimate" at the start
            optimized = f"The Ultimate {optimized}"

        # Truncate intelligently to 60 characters (don't cut mid-word)
        if len(optimized) > 60:
            optimized = optimized[:57].rsplit(' ', 1)[0] + "..."

        logger.debug(f"Title optimized: '{title}' -> '{optimized}'")
        return optimized

    def generate_timestamps(self, script: VideoScript) -> str:
        """
        Generate YouTube chapters/timestamps from script sections.

        YouTube requires:
        - First timestamp must be 00:00
        - At least 3 timestamps
        - Each chapter at least 10 seconds

        Args:
            script: VideoScript object with sections

        Returns:
            Formatted timestamp string for video description
        """
        if not script.sections:
            return ""

        timestamps = []
        current_time = 0

        # Map section types to friendly labels
        section_labels = {
            "hook": "Hook",
            "intro": "Introduction",
            "introduction": "Introduction",
            "problem": "The Problem",
            "promise": "What You'll Learn",
            "content": "Main Content",
            "point": "Key Point",
            "example": "Example",
            "story": "Story",
            "payoff": "Key Insight",
            "cta": "Final Thoughts",
            "outro": "Outro",
            "conclusion": "Conclusion"
        }

        for i, section in enumerate(script.sections):
            minutes = current_time // 60
            seconds = current_time % 60

            # Get a friendly label
            section_type = section.section_type.lower()
            label = section_labels.get(section_type, section.title or f"Part {i+1}")

            # Use section title if it's more descriptive
            if section.title and len(section.title) > 3:
                label = section.title

            timestamps.append(f"{minutes:02d}:{seconds:02d} - {label}")
            current_time += section.duration_seconds

        return "\n".join(timestamps)

    def generate_optimized_description(
        self,
        script: VideoScript,
        primary_keyword: str = "",
        niche: str = "default"
    ) -> str:
        """
        Generate an SEO-optimized YouTube description.

        Best practices applied:
        - Primary keyword in first 200 characters
        - Auto-generated timestamps/chapters
        - 3 relevant hashtags at the end
        - Clear call-to-action

        Args:
            script: VideoScript object
            primary_keyword: Main keyword for SEO
            niche: Content niche for relevant hashtags

        Returns:
            Optimized description string
        """
        # Start with a keyword-rich hook (first 200 chars are critical)
        keyword = primary_keyword or script.title.split(':')[0].strip()
        hook = f"Discover {keyword} in this comprehensive guide. "

        # Add the original description if it exists
        if script.description:
            hook += script.description[:150]

        # Ensure we have content in first 200 chars
        if len(hook) < 100:
            hook += f" Learn everything you need to know about {keyword}."

        # Generate timestamps
        timestamps = self.generate_timestamps(script)

        # Build the description
        parts = [hook.strip()]

        # Add timestamps section
        if timestamps and len(script.sections) >= 3:
            parts.append("\n\n📋 TIMESTAMPS:")
            parts.append(timestamps)

        # Add call-to-action
        parts.append("\n\n🔔 Don't forget to LIKE, COMMENT, and SUBSCRIBE for more content!")
        parts.append("💬 Tell me in the comments: What topic should I cover next?")

        # Add niche-specific hashtags (YouTube shows up to 3 above title)
        niche_hashtags = {
            "finance": ["#Finance", "#MoneyTips", "#WealthBuilding"],
            "psychology": ["#Psychology", "#MindHacks", "#SelfImprovement"],
            "storytelling": ["#TrueStory", "#Documentary", "#Storytelling"],
            "programming": ["#Programming", "#Coding", "#Tech"],
            "default": ["#Tutorial", "#HowTo", "#Education"]
        }

        hashtags = niche_hashtags.get(niche, niche_hashtags["default"])
        parts.append("\n\n" + " ".join(hashtags))

        description = "\n".join(parts)

        # YouTube description limit is 5000 characters
        if len(description) > 5000:
            description = description[:4997] + "..."

        logger.debug(f"Generated optimized description ({len(description)} chars)")
        return description

    def get_full_narration(
        self,
        script: VideoScript,
        clean: bool = True,
        validate: bool = True,
        niche: str = "default"
    ) -> str:
        """
        Extract the narration text for TTS, with optional cleaning and validation.

        Args:
            script: VideoScript object with sections
            clean: If True, clean the script (remove timestamps, formatting, etc.)
            validate: If True, validate and log any issues found
            niche: Content niche for validation context

        Returns:
            Narration text ready for TTS
        """
        narration_parts = []

        for section in script.sections:
            if section.narration:
                narration_parts.append(section.narration)

        full_narration = "\n\n".join(narration_parts)

        # Clean and validate if requested
        if clean or validate:
            try:
                from src.content.script_validator import ScriptValidator, ValidationResult
                validator = ScriptValidator()

                if clean:
                    original_length = len(full_narration)
                    full_narration = validator.clean_script(full_narration)
                    cleaned_length = len(full_narration)
                    if original_length != cleaned_length:
                        logger.info(f"Script cleaned: {original_length} -> {cleaned_length} chars "
                                   f"({original_length - cleaned_length} chars removed)")

                if validate:
                    result = validator.validate_script(
                        full_narration,
                        niche=niche,
                        is_short=script.is_short
                    )
                    if not result.is_valid:
                        logger.warning(f"Script validation failed (score: {result.score}/100)")
                        for issue in result.issues:
                            logger.warning(f"  - {issue}")
                    elif result.warnings:
                        logger.info(f"Script validation passed (score: {result.score}/100) "
                                   f"with {len(result.warnings)} warnings")
                    else:
                        logger.success(f"Script validation passed (score: {result.score}/100)")

            except ImportError as e:
                logger.debug(f"Script validator not available: {e}")
            except Exception as e:
                logger.warning(f"Script validation error: {e}")

        return full_narration

    def validate_script(
        self,
        script: VideoScript,
        niche: str = "default"
    ) -> "ValidationResult":
        """
        Validate a script and return detailed results.

        Args:
            script: VideoScript object to validate
            niche: Content niche for context

        Returns:
            ValidationResult with score, issues, warnings, and suggestions
        """
        from src.content.script_validator import ScriptValidator

        # Get full narration
        narration_parts = []
        for section in script.sections:
            if section.narration:
                narration_parts.append(section.narration)
        full_narration = "\n\n".join(narration_parts)

        validator = ScriptValidator()
        return validator.validate_script(
            full_narration,
            niche=niche,
            is_short=script.is_short
        )

    def clean_script_narration(
        self,
        script: VideoScript,
        improve: bool = False,
        niche: str = "default"
    ) -> VideoScript:
        """
        Clean all narration in a script, optionally improving it.

        Args:
            script: VideoScript to clean
            improve: If True, also improve the script (break sentences, etc.)
            niche: Content niche for improvement context

        Returns:
            New VideoScript with cleaned narration
        """
        from src.content.script_validator import ScriptValidator
        from copy import deepcopy

        validator = ScriptValidator()
        cleaned_script = deepcopy(script)

        for section in cleaned_script.sections:
            if section.narration:
                if improve:
                    section.narration = validator.improve_script(section.narration, niche=niche)
                else:
                    section.narration = validator.clean_script(section.narration)

        # Update hook text if it exists
        if cleaned_script.hook_text:
            if improve:
                cleaned_script.hook_text = validator.improve_script(cleaned_script.hook_text, niche=niche)
            else:
                cleaned_script.hook_text = validator.clean_script(cleaned_script.hook_text)

        logger.info(f"Script narration {'improved' if improve else 'cleaned'}")
        return cleaned_script

    def generate_viral_title(self, topic: str, niche: str = "default") -> str:
        """
        Generate a viral title using competitor-proven templates.

        Args:
            topic: The main topic/subject of the video
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            A viral title string filled with the topic
        """
        import random

        templates = VIRAL_TITLE_TEMPLATES.get(niche, VIRAL_TITLE_TEMPLATES.get("finance", []))
        if not templates:
            return topic

        # Select a template that can work with the topic
        template = random.choice(templates)

        # Replace common placeholders with the topic
        title = template
        placeholders = ["{subject}", "{topic}", "{company}", "{brand}", "{famous_thing}"]
        for placeholder in placeholders:
            if placeholder in title:
                title = title.replace(placeholder, topic, 1)
                break

        # If no placeholder was replaced, prepend the topic
        if title == template:
            title = f"{topic}: {template}"

        return self.optimize_title(title)

    def get_hook_for_niche(self, topic: str, niche: str = "default") -> str:
        """
        Get a proven hook formula for a specific niche.

        Args:
            topic: The video topic to insert into the hook
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            A hook string with the topic inserted
        """
        import random

        # Get niche-specific hooks, fallback to universal
        hooks = HOOK_FORMULAS.get(niche, []) + HOOK_FORMULAS.get("universal", [])
        if not hooks:
            return f"What if I told you everything you know about {topic} is wrong?"

        hook = random.choice(hooks)

        # Replace placeholders
        hook = hook.replace("{topic}", topic)
        hook = hook.replace("{duration}", "10 minutes")
        hook = hook.replace("{percentage}", str(random.randint(73, 97)))
        hook = hook.replace("{amount}", str(random.choice([100, 500, 1000, 5000])))
        hook = hook.replace("{year}", str(datetime.now().year))

        return hook

    @staticmethod
    def get_viral_title_templates(niche: str) -> List[str]:
        """
        Get all viral title templates for a niche.

        Args:
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            List of title template strings
        """
        return VIRAL_TITLE_TEMPLATES.get(niche, VIRAL_TITLE_TEMPLATES.get("finance", []))

    @staticmethod
    def get_hook_formulas(niche: str) -> List[str]:
        """
        Get all hook formulas for a niche.

        Args:
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            List of hook formula strings
        """
        niche_hooks = HOOK_FORMULAS.get(niche, [])
        universal_hooks = HOOK_FORMULAS.get("universal", [])
        return niche_hooks + universal_hooks

    def generate_pattern_interrupts(self, count: int = 5) -> List[str]:
        """
        Generate pattern interrupt phrases to use throughout script.

        Pattern interrupts keep viewers engaged by breaking expectations.
        Use these every 60-90 seconds to maintain retention.

        Args:
            count: Number of pattern interrupts to generate

        Returns:
            List of pattern interrupt phrases
        """
        interrupts = [
            "But here's where it gets really interesting...",
            "Wait - there's something most people miss here...",
            "Now, this next part is crucial...",
            "But there's a catch nobody talks about...",
            "Here's the part that changed everything for me...",
            "Stop. What I'm about to say changes the whole picture...",
            "And this is where most people go wrong...",
            "Pay attention to this next part - it's the key...",
            "What happened next shocked even the experts...",
            "But wait - it gets even more interesting...",
            "Here's what nobody tells you about this...",
            "And that's when everything changed...",
            "But there's one thing missing from this story...",
            "This is where things get complicated...",
            "Most people skip this part - don't make that mistake..."
        ]

        import random
        return random.sample(interrupts, min(count, len(interrupts)))

    def generate_first_30_seconds(self, topic: str, niche: str = "default") -> Dict[str, str]:
        """
        Generate optimized first 30 seconds of script for maximum retention.

        Structure:
        - 0-5s: Hook (pattern interrupt)
        - 5-15s: Context (why this matters)
        - 15-30s: Promise (what they'll learn) + First open loop

        Args:
            topic: The video topic
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            Dict with 'hook', 'context', 'promise' keys
        """
        import random

        # Generate hook using proven formula
        hook = self.get_hook_for_niche(topic, niche)

        # Generate context based on niche
        context_templates = {
            "finance": [
                f"And what I'm about to show you about {topic} could change your financial future forever.",
                f"Most people never learn this about {topic} - and it costs them thousands.",
                f"The wealthy have known this secret about {topic} for decades."
            ],
            "psychology": [
                f"Your brain processes {topic} in ways you've never imagined.",
                f"What science has discovered about {topic} will make you rethink everything.",
                f"Psychologists have studied {topic} for years - here's what they found."
            ],
            "storytelling": [
                f"The true story of {topic} is more shocking than fiction.",
                f"What really happened with {topic} has never been fully told - until now.",
                f"The events surrounding {topic} changed everything."
            ],
            "default": [
                f"What I'm about to show you about {topic} took me years to discover.",
                f"By the end of this video, {topic} will never look the same to you.",
                f"The truth about {topic} is more surprising than you think."
            ]
        }
        context = random.choice(context_templates.get(niche, context_templates["default"]))

        # Generate promise with open loop
        promise_templates = {
            "finance": [
                f"In the next few minutes, I'll reveal exactly how to use {topic} to build real wealth. But first, you need to understand something most people miss...",
                f"I'm going to show you the exact strategy successful investors use with {topic}. And at the end, I'll share the one thing that makes all the difference...",
            ],
            "psychology": [
                f"By the end of this video, you'll know exactly how {topic} works - and how to use it. But the third point I'm about to share will surprise you the most...",
                f"I'll reveal the science behind {topic} and give you practical techniques. What I share at the end is what truly changed my understanding...",
            ],
            "storytelling": [
                f"I'm going to tell you the complete story of {topic} - including the parts they tried to hide. The ending will shock you...",
                f"What you're about to learn about {topic} has never been fully revealed - until now. Wait until you hear what happened next...",
            ],
            "default": [
                f"In the next few minutes, you'll learn everything you need to know about {topic}. But the most important insight comes at the end...",
                f"I'll break down {topic} step by step. And what I reveal near the end might just change everything...",
            ]
        }
        promise = random.choice(promise_templates.get(niche, promise_templates["default"]))

        return {
            "hook": hook,
            "context": context,
            "promise": promise,
            "full_30_seconds": f"{hook}\n\n{context}\n\n{promise}"
        }

    def generate_chapter_suggestions(self, sections: List[ScriptSection]) -> List[ChapterMarker]:
        """
        Generate optimal YouTube chapter markers from script sections.

        Best practices:
        - First chapter must be 00:00
        - Minimum 3 chapters
        - Each chapter at least 10 seconds
        - Descriptive, keyword-rich titles

        Args:
            sections: List of script sections

        Returns:
            List of ChapterMarker objects
        """
        if not sections or len(sections) < 3:
            return []

        chapters = []
        current_time = 0

        # Ensure first chapter is at 00:00
        first_section = sections[0]
        chapters.append(ChapterMarker(
            timestamp_seconds=0,
            title=first_section.title or "Introduction"
        ))
        current_time += first_section.duration_seconds

        # Add chapters for significant sections
        for section in sections[1:]:
            # Skip very short sections (< 10s)
            if section.duration_seconds < 10:
                current_time += section.duration_seconds
                continue

            # Create chapter for content sections
            if section.section_type in ["content", "point", "story", "example", "conclusion"]:
                chapters.append(ChapterMarker(
                    timestamp_seconds=current_time,
                    title=section.title or section.section_type.capitalize()
                ))

            current_time += section.duration_seconds

        return chapters

    def _validate_and_enhance_script(
        self,
        script: VideoScript,
        niche: str
    ) -> VideoScript:
        """
        Validate script against best practices and log results.

        This method checks the script title, hook, and content against
        the best practices from COMPETITOR_ANALYSIS.md and logs any
        issues or improvement suggestions.

        Args:
            script: VideoScript object to validate
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            The original script (validation is logged, not modified)
        """
        if not BEST_PRACTICES_AVAILABLE:
            logger.debug("Best practices validation skipped (module not available)")
            return script

        logger.info("Validating script against best practices...")

        # Validate title
        title_result = validate_title(script.title, niche)
        if title_result.is_valid:
            logger.success(f"Title validation: {title_result}")
        else:
            logger.warning(f"Title validation: {title_result}")
            for suggestion in title_result.suggestions[:3]:
                logger.info(f"  Suggestion: {suggestion}")

        # Validate hook
        if script.hook_text:
            hook_result = validate_hook(script.hook_text, niche)
            if hook_result.is_valid:
                logger.success(f"Hook validation: {hook_result}")
            else:
                logger.warning(f"Hook validation: {hook_result}")
                for suggestion in hook_result.suggestions[:3]:
                    logger.info(f"  Suggestion: {suggestion}")

        # Get best practices for context
        practices = get_best_practices(niche)
        metrics = practices.get("metrics", {})

        # Validate video length
        duration_minutes = script.total_duration / 60
        optimal_range = metrics.get("optimal_video_length", (8, 15))
        if optimal_range[0] <= duration_minutes <= optimal_range[1]:
            logger.success(f"Video length: {duration_minutes:.1f} min (optimal: {optimal_range[0]}-{optimal_range[1]} min)")
        else:
            logger.warning(f"Video length: {duration_minutes:.1f} min (optimal: {optimal_range[0]}-{optimal_range[1]} min)")

        # Log overall content suggestions
        content_data = {
            "title": script.title,
            "hook": script.hook_text,
            "duration": duration_minutes,
            "tags": script.tags,
            "description": script.description,
        }
        suggestions = suggest_improvements(content_data, niche)
        if suggestions:
            logger.info(f"Improvement suggestions ({len(suggestions)} total):")
            for suggestion in suggestions[:5]:
                logger.info(f"  - {suggestion}")

        return script

    def run_pre_publish_checklist(
        self,
        script: VideoScript,
        niche: str
    ) -> 'PrePublishChecklist':
        """
        Run the pre-publish validation checklist on a script.

        This validates:
        - Title matches viral patterns for niche
        - Hook is in first 5 seconds and engaging
        - Video length is optimal for niche
        - Description follows SEO patterns
        - Tags are appropriate
        - Chapter markers present
        - Script has proper retention structure

        Args:
            script: VideoScript object to validate
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            PrePublishChecklist object with validation results

        Example:
            writer = ScriptWriter(provider="groq")
            script = writer.generate_script("How AI Works", niche="psychology")
            checklist = writer.run_pre_publish_checklist(script, "psychology")
            print(f"Ready to publish: {checklist.ready_to_publish}")
            for item in checklist.items:
                status = "[PASS]" if item.passed else "[FAIL]"
                print(f"{status} {item.name}: {item.details}")
        """
        if not BEST_PRACTICES_AVAILABLE:
            logger.warning("Pre-publish checklist not available (best_practices module not found)")
            # Return a mock checklist indicating unavailable
            from dataclasses import dataclass, field
            from typing import List

            @dataclass
            class MockChecklistItem:
                name: str
                passed: bool
                details: str
                priority: str = "high"

            @dataclass
            class MockChecklist:
                items: List[MockChecklistItem] = field(default_factory=list)
                overall_score: float = 0.0
                ready_to_publish: bool = False
                critical_issues: List[str] = field(default_factory=list)

                def __str__(self) -> str:
                    return "[NOT AVAILABLE] Best practices module not installed"

            return MockChecklist(
                items=[MockChecklistItem("Module Check", False, "best_practices module not found")],
                critical_issues=["best_practices module not found - install to enable validation"]
            )

        checklist = pre_publish_checklist(script, niche)

        # Log the results
        logger.info(f"Pre-publish checklist: {checklist}")
        for item in checklist.items:
            status = "PASS" if item.passed else "FAIL"
            log_fn = logger.success if item.passed else logger.warning
            log_fn(f"  [{status}] {item.name}: {item.details}")

        if checklist.critical_issues:
            logger.error("Critical issues found:")
            for issue in checklist.critical_issues:
                logger.error(f"  - {issue}")

        if checklist.ready_to_publish:
            logger.success("Script is READY TO PUBLISH")
        else:
            logger.warning("Script NEEDS IMPROVEMENTS before publishing")

        return checklist

    def get_niche_best_practices(self, niche: str) -> Dict[str, Any]:
        """
        Get all best practices for a specific niche.

        Returns metrics, viral patterns, hook formulas, SEO patterns,
        power words, and retention best practices from competitor analysis.

        Args:
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            Dictionary with all best practices for the niche

        Example:
            writer = ScriptWriter()
            practices = writer.get_niche_best_practices("finance")
            print(f"CPM Range: ${practices['metrics']['cpm_range'][0]}-${practices['metrics']['cpm_range'][1]}")
            print(f"Optimal length: {practices['metrics']['optimal_video_length'][0]}-{practices['metrics']['optimal_video_length'][1]} min")
        """
        if not BEST_PRACTICES_AVAILABLE:
            logger.warning("Best practices not available (module not found)")
            return {
                "niche": niche,
                "error": "best_practices module not found",
                "metrics": {},
            }

        return get_best_practices(niche)


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
