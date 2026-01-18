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

## PROVEN HOOK FORMULAS (Choose one for the opening):

1. **"The Shocking Truth"** - Start with a counterintuitive fact that challenges common beliefs
   Example: "Everything you've been told about [topic] is completely wrong..."

2. **"Story Lead"** - Open with a compelling micro-story (15-20 seconds)
   Example: "In 2019, a man lost everything... but what happened next changed the industry forever."

3. **"Question Stack"** - 3 rapid curiosity questions in the first 10 seconds
   Example: "What if I told you...? Would you believe...? And what would you do if...?"

4. **"Stats Shock"** - Lead with a surprising, specific statistic
   Example: "97.3% of people who try this method fail - here's why you won't."

## VIRAL VIDEO STRUCTURE (IMPORTANT):

1. **HOOK (0-15 seconds):** Use one of the proven hook formulas above
2. **PROBLEM (15-45 seconds):** Identify pain point, create curiosity
3. **PROMISE (45-60 seconds):** Tease what they'll learn - "By the end of this video, you'll know exactly..."
4. **MAIN CONTENT (3-5 minutes):** 5-7 key points with stories/examples
5. **PAYOFF (30 seconds):** Deliver the key insight
6. **CTA (15 seconds):** Subscribe, comment, like

## ENGAGEMENT TECHNIQUES (MANDATORY):

### Open Loops - Plant these throughout the script:
- "I'll reveal the most important tip at the end - trust me, it's worth waiting for..."
- "But there's something even more shocking coming up..."
- "The third point is where everything clicks - keep watching..."

### Micro-Cliffhangers (every 60-90 seconds):
- "But here's where it gets interesting..."
- "What happened next shocked everyone..."
- "And this is where most people make the fatal mistake..."

### Direct Address - Use "you" at least 3 times per minute:
- "You might be thinking..." / "When you try this..." / "Here's what you need to understand..."
- Address the viewer directly to create personal connection

### Rhetorical Questions (every 30-45 seconds):
- "But what does this really mean for you?"
- "Have you ever wondered why...?"
- "Sound familiar?"

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
- Use at least 2 open loops throughout the script
- Include a micro-cliffhanger every 60-90 seconds
- Add a rhetorical question every 30-45 seconds

Write the COMPLETE viral script now:"""

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
    NICHE_GUIDES = {
        "finance": """
FINANCE NICHE - CRITICAL REQUIREMENTS:

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

### Tone: Authoritative expert who's sharing insider knowledge, slightly urgent""",

        "psychology": """
PSYCHOLOGY NICHE - CRITICAL REQUIREMENTS:

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

### Tone: Mysterious insider revealing secrets, slightly dark and intriguing, make viewers feel like they're learning forbidden knowledge""",

        "storytelling": """
STORYTELLING NICHE - CRITICAL REQUIREMENTS:

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
