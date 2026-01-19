"""
Efficient Prompt Templates for YouTube Automation
Minimized, cached prompts that reduce token usage by 50%+.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from functools import lru_cache
import hashlib
import json


@dataclass
class PromptTemplate:
    """A reusable prompt template."""
    name: str
    system_prompt: str
    user_template: str
    output_format: str = "json"
    max_tokens: int = 1000
    temperature: float = 0.7

    def render(self, **kwargs) -> str:
        """Render the user template with variables."""
        return self.user_template.format(**kwargs)

    @property
    def cache_key(self) -> str:
        """Generate cache key for this template."""
        content = f"{self.name}:{self.system_prompt}:{self.user_template}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# ============================================================
# MINIMIZED SYSTEM PROMPTS (Token-efficient)
# ============================================================

SYSTEM_PROMPTS = {
    "script_writer": "Write engaging YouTube scripts. Output JSON only.",

    "hook_generator": "Create viral hooks. Output JSON array of hooks.",

    "title_optimizer": "Optimize titles for CTR. Output JSON with variants.",

    "seo_analyzer": "Analyze keywords. Output JSON with metrics.",

    "research": "Research topics. Output JSON with scored ideas.",

    "thumbnail": "Describe thumbnails. Output JSON with elements.",

    "retention": "Add retention elements. Output enhanced script JSON.",

    "quality_check": "Validate content. Output JSON checklist.",
}


# ============================================================
# SCRIPT GENERATION TEMPLATES
# ============================================================

SCRIPT_TEMPLATES = {
    "finance": PromptTemplate(
        name="finance_script",
        system_prompt=SYSTEM_PROMPTS["script_writer"],
        user_template="""Topic: {topic}
Niche: Finance
Length: {duration_minutes} min
Voice: Authority

Output JSON:
{{"title":"","hook":"","sections":[{{"heading":"","content":"","duration_seconds":0}}],"cta":"","outro":""}}""",
        max_tokens=2000
    ),

    "psychology": PromptTemplate(
        name="psychology_script",
        system_prompt=SYSTEM_PROMPTS["script_writer"],
        user_template="""Topic: {topic}
Niche: Psychology
Length: {duration_minutes} min
Voice: Calm, intriguing

Output JSON:
{{"title":"","hook":"","sections":[{{"heading":"","content":"","duration_seconds":0}}],"cta":"","outro":""}}""",
        max_tokens=2000
    ),

    "storytelling": PromptTemplate(
        name="storytelling_script",
        system_prompt=SYSTEM_PROMPTS["script_writer"],
        user_template="""Topic: {topic}
Niche: Storytelling
Length: {duration_minutes} min
Voice: Dramatic, engaging

Output JSON:
{{"title":"","hook":"","sections":[{{"heading":"","content":"","duration_seconds":0}}],"cta":"","outro":""}}""",
        max_tokens=3000
    ),

    "short": PromptTemplate(
        name="short_script",
        system_prompt="Write 30-60 second YouTube Short scripts. Hook in 1 second. Output JSON.",
        user_template="""Topic: {topic}
Niche: {niche}
Max: 60 seconds

Output JSON:
{{"hook":"","main_points":["","",""],"punchline":"","cta":""}}""",
        max_tokens=500
    ),
}


# ============================================================
# HOOK GENERATION TEMPLATES
# ============================================================

HOOK_TEMPLATES = {
    "curiosity_gap": PromptTemplate(
        name="curiosity_hook",
        system_prompt=SYSTEM_PROMPTS["hook_generator"],
        user_template="""Topic: {topic}
Style: Curiosity gap (create unanswered question)

Output 5 hooks as JSON array: ["hook1","hook2"...]""",
        max_tokens=300
    ),

    "controversy": PromptTemplate(
        name="controversy_hook",
        system_prompt=SYSTEM_PROMPTS["hook_generator"],
        user_template="""Topic: {topic}
Style: Controversial/polarizing (challenge beliefs)

Output 5 hooks as JSON array: ["hook1","hook2"...]""",
        max_tokens=300
    ),

    "number_proof": PromptTemplate(
        name="number_hook",
        system_prompt=SYSTEM_PROMPTS["hook_generator"],
        user_template="""Topic: {topic}
Style: Number/statistic proof (specific data)

Output 5 hooks as JSON array: ["hook1","hook2"...]""",
        max_tokens=300
    ),

    "story_tease": PromptTemplate(
        name="story_hook",
        system_prompt=SYSTEM_PROMPTS["hook_generator"],
        user_template="""Topic: {topic}
Style: Story tease (hint at dramatic story)

Output 5 hooks as JSON array: ["hook1","hook2"...]""",
        max_tokens=300
    ),
}


# ============================================================
# SEO OPTIMIZATION TEMPLATES
# ============================================================

SEO_TEMPLATES = {
    "title_variants": PromptTemplate(
        name="title_variants",
        system_prompt=SYSTEM_PROMPTS["title_optimizer"],
        user_template="""Original title: {title}
Niche: {niche}

Generate 5 A/B test variants optimized for CTR.
Output JSON: {{"variants":[{{"title":"","predicted_ctr":0.0}}]}}""",
        max_tokens=400
    ),

    "description": PromptTemplate(
        name="seo_description",
        system_prompt="Write SEO YouTube descriptions. Include timestamps, keywords, links. Output text only.",
        user_template="""Title: {title}
Sections: {sections}
Keywords: {keywords}

Write 2000-char description with:
- Hook paragraph
- Timestamps
- Keywords naturally integrated
- Subscribe CTA
- Social links placeholder""",
        max_tokens=600
    ),

    "tags": PromptTemplate(
        name="seo_tags",
        system_prompt="Generate YouTube tags. Mix broad and specific. Output JSON array.",
        user_template="""Title: {title}
Niche: {niche}

Output 30 tags as JSON: ["tag1","tag2"...]
Include: primary topic, synonyms, related topics, trending variations""",
        max_tokens=300
    ),
}


# ============================================================
# QUALITY CHECK TEMPLATES
# ============================================================

QUALITY_TEMPLATES = {
    "pre_publish": PromptTemplate(
        name="pre_publish_check",
        system_prompt=SYSTEM_PROMPTS["quality_check"],
        user_template="""Script: {script_preview}
Title: {title}
Niche: {niche}

Check:
1. Hook strength (1-10)
2. Open loops present (bool)
3. CTAs placed correctly (bool)
4. Retention elements (count)
5. Policy compliance (bool)

Output JSON: {{"score":0,"issues":[],"suggestions":[]}}""",
        max_tokens=400
    ),

    "title_check": PromptTemplate(
        name="title_check",
        system_prompt=SYSTEM_PROMPTS["quality_check"],
        user_template="""Title: {title}
Niche: {niche}

Evaluate: curiosity, clarity, keyword presence, length, emotional trigger

Output JSON: {{"score":0,"issues":[],"improved_title":""}}""",
        max_tokens=200
    ),
}


# ============================================================
# RESEARCH TEMPLATES
# ============================================================

RESEARCH_TEMPLATES = {
    "topic_ideas": PromptTemplate(
        name="topic_research",
        system_prompt=SYSTEM_PROMPTS["research"],
        user_template="""Niche: {niche}
Recent trends: {trends}

Generate {count} video ideas.
Output JSON: {{"ideas":[{{"title":"","score":0,"reasoning":""}}]}}""",
        max_tokens=800
    ),

    "competitor_analysis": PromptTemplate(
        name="competitor_analysis",
        system_prompt="Analyze competitor content. Output JSON insights.",
        user_template="""Competitor titles: {titles}
Niche: {niche}

Identify patterns:
1. Title formulas
2. Topics that work
3. Gaps to exploit

Output JSON: {{"formulas":[],"top_topics":[],"gaps":[]}}""",
        max_tokens=500
    ),
}


# ============================================================
# PROMPT CACHE
# ============================================================

class PromptCache:
    """
    Cache for rendered prompts and responses.
    Reduces redundant API calls.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}

    def get_key(self, template_name: str, **kwargs) -> str:
        """Generate cache key from template and params."""
        sorted_kwargs = json.dumps(kwargs, sort_keys=True)
        content = f"{template_name}:{sorted_kwargs}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get cached response."""
        return self._cache.get(key)

    def set(self, key: str, value: Any):
        """Cache a response."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entries (simple FIFO)
            oldest = list(self._cache.keys())[:100]
            for k in oldest:
                del self._cache[k]
        self._cache[key] = value

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global cache instance
_prompt_cache = PromptCache()


def get_prompt_cache() -> PromptCache:
    """Get the global prompt cache."""
    return _prompt_cache


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

@lru_cache(maxsize=100)
def get_script_template(niche: str) -> PromptTemplate:
    """Get script template for a niche."""
    return SCRIPT_TEMPLATES.get(niche, SCRIPT_TEMPLATES["finance"])


@lru_cache(maxsize=50)
def get_hook_template(style: str) -> PromptTemplate:
    """Get hook template by style."""
    return HOOK_TEMPLATES.get(style, HOOK_TEMPLATES["curiosity_gap"])


def get_all_templates() -> Dict[str, Dict[str, PromptTemplate]]:
    """Get all available templates."""
    return {
        "scripts": SCRIPT_TEMPLATES,
        "hooks": HOOK_TEMPLATES,
        "seo": SEO_TEMPLATES,
        "quality": QUALITY_TEMPLATES,
        "research": RESEARCH_TEMPLATES,
    }


def estimate_tokens(template: PromptTemplate, **kwargs) -> int:
    """Estimate tokens for a rendered prompt."""
    rendered = template.render(**kwargs)
    # Rough estimate: 4 chars per token
    return (len(template.system_prompt) + len(rendered)) // 4


# ============================================================
# FEW-SHOT EXAMPLES (Stored once, reused)
# ============================================================

FEW_SHOT_EXAMPLES = {
    "good_hook": [
        "99% of investors make this ONE mistake...",
        "I analyzed 1,000 millionaires. Here's what they have in common.",
        "Wall Street doesn't want you to know this.",
        "The psychology trick that makes you spend more.",
        "This man predicted every crash. Here's his next prediction.",
    ],

    "good_title": [
        "How I Made $10,000/Month with ONE Strategy",
        "5 Money Mistakes That Cost Me $50,000",
        "The Psychology of Why You Can't Save Money",
        "This Company Will 10x (Here's Why)",
        "The Untold Story of the $50 Billion Scam",
    ],

    "open_loops": [
        "But here's where it gets interesting...",
        "And the third reason? It might surprise you.",
        "But wait, there's a catch...",
        "I'll reveal the secret in just a moment.",
        "The answer shocked even me.",
    ],
}


if __name__ == "__main__":
    # Demo usage
    template = get_script_template("finance")
    rendered = template.render(topic="How to invest $1000", duration_minutes=10)
    print(f"Template: {template.name}")
    print(f"Estimated tokens: {estimate_tokens(template, topic='How to invest $1000', duration_minutes=10)}")
    print(f"\nRendered prompt:\n{rendered}")
