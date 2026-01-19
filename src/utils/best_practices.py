"""
Best Practices Module - Competitor Analysis Integration

Loads best practices from competitor analysis and provides validation functions
for video content creation. Based on COMPETITOR_ANALYSIS.md (January 2026).

Usage:
    from src.utils.best_practices import (
        validate_title,
        validate_hook,
        get_best_practices,
        suggest_improvements,
        pre_publish_checklist
    )

    # Validate a title
    result = validate_title("5 Money Mistakes Costing You $1000/Year", "finance")

    # Get all best practices for a niche
    practices = get_best_practices("psychology")

    # Run pre-publish checklist
    checklist = pre_publish_checklist(script, "storytelling")
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


# ============================================================
# KEY METRICS FROM COMPETITOR ANALYSIS (January 2026)
# ============================================================

NICHE_METRICS = {
    "finance": {
        "cpm_range": (10, 22),  # $10-22 CPM
        "optimal_video_length": (8, 15),  # 8-15 minutes
        "sweet_spot_minutes": 10,
        "posting_frequency": "2-3 videos per week",
        "best_days": ["Monday", "Wednesday", "Friday"],
        "best_times": ["15:00", "19:00", "21:00"],  # UTC equivalent of 3-5 PM EST
        "top_competitors": [
            {"name": "The Swedish Investor", "subs": "1M+", "best_views": "5.2M"},
            {"name": "Practical Wisdom", "subs": "500K+", "best_views": "10M"},
            {"name": "Two Cents", "subs": "1.5M+", "best_views": "3M+"},
        ],
    },
    "psychology": {
        "cpm_range": (3, 6),  # $3-6 CPM
        "optimal_video_length": (8, 12),  # 8-12 minutes
        "sweet_spot_minutes": 10,
        "posting_frequency": "3-4 videos per week",
        "best_days": ["Tuesday", "Thursday", "Saturday"],
        "best_times": ["16:00", "19:30", "21:30"],  # Psychology performs well on weekends
        "top_competitors": [
            {"name": "Psych2Go", "subs": "12.7M", "best_views": "18M"},
            {"name": "Brainy Dose", "subs": "2M+", "best_views": "10M"},
            {"name": "The Infographics Show", "subs": "15M", "best_views": "20M+"},
        ],
    },
    "storytelling": {
        "cpm_range": (4, 15),  # $4-15 CPM (documentary-style gets 30-50% higher)
        "optimal_video_length": (12, 30),  # 12-30 minutes
        "sweet_spot_minutes": 15,
        "posting_frequency": "Daily or every other day",
        "best_days": ["Every day"],
        "best_times": ["17:00", "20:00", "22:00"],  # 5-8 PM EST evening viewing
        "top_competitors": [
            {"name": "JCS - Criminal Psychology", "subs": "5M+", "best_views": "20M+"},
            {"name": "Truly Criminal", "subs": "500K+", "best_views": "2M+"},
            {"name": "Lazy Masquerade", "subs": "1.7M", "best_views": "5M+"},
            {"name": "Mr. Nightmare", "subs": "6M", "best_views": "15M+"},
        ],
    },
}


# ============================================================
# VIRAL TITLE PATTERNS FROM COMPETITOR ANALYSIS
# ============================================================

VIRAL_TITLE_PATTERNS = {
    "finance": [
        r"How .+ Makes Money",  # Business model explainers
        r"Why .+ Will \d+x",  # Stock analysis
        r"\d+ Passive Income",  # Passive income lists
        r"\d+ Money Mistakes",  # Loss aversion (highest CTR)
        r"The Truth About",  # Documentary-business hybrids
        r"I Analyzed \d+",  # Data-driven analysis
        r"\$[\d,]+ .+ (Month|Year|Week)",  # Dollar amount specificity
        r"Warren Buffett",  # Authority reference
        r"(Buy|Sell|Hold)",  # Stock decisions
    ],
    "psychology": [
        r"\d+ Signs of",  # Personality types
        r"Why Your Brain",  # Brain facts
        r"Dark Psychology",  # Dark psychology (highest engagement)
        r"The .+ That's Ruining",  # Cognitive biases
        r"\d+ Body Language",  # Body language
        r"How to Read",  # Mind reading appeal
        r"Tricks .+ Uses",  # Manipulation awareness
        r"\d+ Psychological",  # Psychology lists
        r"(Narcissist|Manipulation|Toxic)",  # High-engagement keywords
    ],
    "storytelling": [
        r"The Untold Story",  # Company documentaries
        r"Rise and Fall",  # Rise and fall narratives
        r"What Happened to",  # Mystery/tension
        r"The Dark Side",  # Dark side exposes
        r"From \$0 to",  # Rags to riches
        r"Biggest Mistake",  # Failure stories
        r"True Story",  # True stories
        r"Secret .+",  # Secret revelations
        r"Nobody Knows",  # Mystery hooks
    ],
}


# ============================================================
# HOOK FORMULAS FROM COMPETITOR ANALYSIS
# ============================================================

HOOK_FORMULAS = {
    "finance": [
        "If you invested ${amount} in {investment} {timeframe} ago, you'd have ${result} today...",
        "The difference between 7% and 9% returns? ${amount} over 30 years...",
        "Wall Street doesn't want you to know this, but...",
        "{percentage}% of your paycheck is disappearing, and here's where it goes...",
        "In {year}, a man turned ${small_amount} into ${large_amount} using this exact method...",
    ],
    "psychology": [
        "Your brain is lying to you right now, and you don't even know it...",
        "What I'm about to show you is used by FBI interrogators...",
        "Scientists discovered something terrifying about the human mind...",
        "In the next {duration}, you'll be able to read anyone's thoughts...",
        "They've been using this against you since you were {age} years old...",
    ],
    "storytelling": [
        "The door slammed. He had exactly {seconds} seconds to decide...",
        "Nobody knows why he did it. But what happened next shocked the world...",
        "To this day, no one can explain what really happened...",
        "He had everything. By morning, he had nothing...",
        "What I'm about to tell you is the true story they tried to bury...",
    ],
    "universal": [
        "Stop. What you're about to see changes everything...",
        "Forget everything you've been told about {topic}...",
        "This is the one thing nobody tells you about {topic}...",
        "You're losing ${amount} every year without knowing it...",
        "Only {percentage}% of people know this. Here's why it matters...",
    ],
}


# ============================================================
# POWER WORDS FOR VIRAL CONTENT
# ============================================================

POWER_WORDS = {
    "authority": ["Ultimate", "Proven", "Expert", "Complete", "Definitive", "Essential"],
    "curiosity": ["Secret", "Hidden", "Shocking", "Revealed", "Truth", "Nobody", "Untold", "Dark"],
    "urgency": ["Critical", "Instant", "Guaranteed", "Revolutionary", "Breakthrough"],
    "emotional": ["Powerful", "Incredible", "Amazing", "Massive", "Terrifying", "Genius"],
}


# ============================================================
# DESCRIPTION AND TAG PATTERNS
# ============================================================

SEO_PATTERNS = {
    "finance": {
        "required_tags": ["money", "finance", "investing", "passive income", "wealth building", "stock market", "personal finance"],
        "description_rules": [
            "Primary keyword in first 200 characters",
            "Include timestamps/chapters (3+ chapters minimum)",
            "Reference credible sources (Warren Buffett, academic studies)",
        ],
    },
    "psychology": {
        "required_tags": ["psychology", "dark psychology", "manipulation", "narcissist", "body language", "cognitive bias", "self improvement", "mindset"],
        "description_rules": [
            "Curiosity-building first line",
            "Reference specific studies (Stanford, Milgram, Cialdini)",
            "Include chapter markers for list content",
        ],
    },
    "storytelling": {
        "required_tags": ["true crime", "documentary", "mystery", "unsolved", "scandal", "business", "rise and fall", "untold story"],
        "description_rules": [
            "Dramatic opening line",
            "Timeline/chronological markers",
            "Sensory details in description",
        ],
    },
}


# ============================================================
# RETENTION BEST PRACTICES
# ============================================================

RETENTION_BEST_PRACTICES = {
    "first_30_seconds": {
        "0-5s": {"element": "Pattern interrupt/bold claim", "purpose": "Grab attention (critical for 70%+ retention)"},
        "5-15s": {"element": "Context + first open loop", "purpose": "Set stakes, plant curiosity"},
        "15-30s": {"element": "First micro-payoff", "purpose": "Deliver value, prevent drop-off"},
    },
    "engagement_techniques": {
        "open_loops": {"frequency": "Minimum 3 per video", "impact": "32% increase in watch time"},
        "micro_cliffhangers": {"frequency": "Every 45-60 seconds", "impact": "Maintains tension"},
        "direct_address": {"frequency": "Use 'you' at least 3 times per minute", "impact": "Personal connection"},
        "rhetorical_questions": {"frequency": "Every 30-45 seconds", "impact": "Active engagement"},
        "specific_numbers": {"rule": "Always use exact figures", "impact": "Credibility boost"},
    },
    "cta_placement": {
        "never": {"timing": "First 30 seconds", "reason": "Kills retention"},
        "soft": {"timing": "30% mark", "example": "If you're finding this valuable, hit subscribe..."},
        "engagement": {"timing": "50% mark", "example": "Comment below with your experience..."},
        "final": {"timing": "95% mark", "example": "Like and subscribe for more..."},
    },
}


# ============================================================
# IMPACT TITLE FORMULA
# ============================================================

IMPACT_FORMULA = {
    "I": "Immediate Hook (First 3-5 words)",
    "M": "Measurable Outcome (Numbers/results)",
    "P": "Personal or Proof Element (Credibility)",
    "A": "Audience Clarification (Who it's for)",
    "C": "Curiosity or Controversy (Intrigue)",
    "T": "Timeframe (When/urgency)",
}


# ============================================================
# VALIDATION RESULT DATACLASSES
# ============================================================

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.is_valid else "NEEDS IMPROVEMENT"
        return f"[{status}] Score: {self.score:.1%} | Passed: {len(self.passed_checks)} | Failed: {len(self.failed_checks)}"


@dataclass
class ChecklistItem:
    """Single item in the pre-publish checklist."""
    name: str
    passed: bool
    details: str
    priority: str = "high"  # high, medium, low


@dataclass
class PrePublishChecklist:
    """Complete pre-publish checklist result."""
    items: List[ChecklistItem]
    overall_score: float
    ready_to_publish: bool
    critical_issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        passed = sum(1 for item in self.items if item.passed)
        total = len(self.items)
        status = "READY" if self.ready_to_publish else "NOT READY"
        return f"[{status}] Score: {self.overall_score:.1%} | Passed: {passed}/{total}"


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def validate_title(title: str, niche: str) -> ValidationResult:
    """
    Validate a video title against viral patterns for a specific niche.

    Args:
        title: The video title to validate
        niche: The content niche (finance, psychology, storytelling)

    Returns:
        ValidationResult with score and suggestions
    """
    passed_checks = []
    failed_checks = []
    suggestions = []

    niche = niche.lower()
    if niche not in VIRAL_TITLE_PATTERNS:
        niche = "finance"  # Default fallback

    # Check 1: Length (under 60 characters is optimal)
    if len(title) <= 60:
        passed_checks.append("Title length is optimal (under 60 chars)")
    else:
        failed_checks.append(f"Title is too long ({len(title)} chars, should be under 60)")
        suggestions.append("Shorten title to under 60 characters for better display")

    # Check 2: Contains numbers (specific numbers perform better)
    if re.search(r'\d+', title):
        passed_checks.append("Contains specific numbers")
    else:
        failed_checks.append("Missing specific numbers")
        suggestions.append("Add specific numbers (e.g., '5 Ways', '$1,000', '10x')")

    # Check 3: Contains power words
    all_power_words = []
    for category_words in POWER_WORDS.values():
        all_power_words.extend(category_words)

    found_power_words = [w for w in all_power_words if w.lower() in title.lower()]
    if found_power_words:
        passed_checks.append(f"Contains power words: {', '.join(found_power_words)}")
    else:
        failed_checks.append("Missing power words")
        suggestions.append(f"Add power words like: {', '.join(POWER_WORDS['curiosity'][:3])}")

    # Check 4: Matches viral patterns for niche
    patterns = VIRAL_TITLE_PATTERNS.get(niche, [])
    matched_patterns = [p for p in patterns if re.search(p, title, re.IGNORECASE)]
    if matched_patterns:
        passed_checks.append(f"Matches {len(matched_patterns)} viral pattern(s)")
    else:
        failed_checks.append("Doesn't match known viral patterns")
        suggestions.append(f"Consider using patterns like: 'How X Makes Money', '{len(patterns)} Signs of Y'")

    # Check 5: Dollar amounts (for finance)
    if niche == "finance":
        if re.search(r'\$[\d,]+', title):
            passed_checks.append("Contains specific dollar amount (finance best practice)")
        else:
            failed_checks.append("Missing dollar amounts (finance titles should include $)")
            suggestions.append("Add specific dollar amounts: '$1,000', '$5,000/month'")

    # Check 6: No all caps (looks spammy)
    if title != title.upper():
        passed_checks.append("Not all caps (good)")
    else:
        failed_checks.append("All caps looks spammy")
        suggestions.append("Use title case instead of all caps")

    # Check 7: Year relevance (for evergreen topics)
    current_year = "2026"
    year_relevant_keywords = ["best", "top", "guide", "tips", "strategy", "review", "trends"]
    has_year_keyword = any(kw in title.lower() for kw in year_relevant_keywords)
    if has_year_keyword and current_year in title:
        passed_checks.append(f"Includes current year ({current_year})")
    elif has_year_keyword and current_year not in title:
        failed_checks.append("Missing current year for time-sensitive topic")
        suggestions.append(f"Add '{current_year}' to title for time-sensitive content")

    # Calculate score
    total_checks = len(passed_checks) + len(failed_checks)
    score = len(passed_checks) / total_checks if total_checks > 0 else 0.0
    is_valid = score >= 0.6  # 60% threshold

    return ValidationResult(
        is_valid=is_valid,
        score=score,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        suggestions=suggestions
    )


def validate_hook(hook: str, niche: str) -> ValidationResult:
    """
    Validate a video hook against best practices for engagement.

    Args:
        hook: The first 5 seconds of narration
        niche: The content niche (finance, psychology, storytelling)

    Returns:
        ValidationResult with score and suggestions
    """
    passed_checks = []
    failed_checks = []
    suggestions = []

    niche = niche.lower()

    # Check 1: Hook length (should be ~15-25 words for 5 seconds)
    word_count = len(hook.split())
    if 10 <= word_count <= 30:
        passed_checks.append(f"Hook length is good ({word_count} words)")
    elif word_count < 10:
        failed_checks.append(f"Hook is too short ({word_count} words)")
        suggestions.append("Expand hook to 15-25 words for full 5-second impact")
    else:
        failed_checks.append(f"Hook is too long ({word_count} words)")
        suggestions.append("Trim hook to 15-25 words (5 seconds max)")

    # Check 2: Pattern interrupt (starts with attention-grabbing word)
    interrupt_starters = ["stop", "wait", "listen", "what if", "imagine", "forget", "here's", "the"]
    starts_with_interrupt = any(hook.lower().startswith(starter) for starter in interrupt_starters)
    if starts_with_interrupt:
        passed_checks.append("Starts with pattern interrupt")
    else:
        failed_checks.append("Missing pattern interrupt at start")
        suggestions.append("Start with: 'What if...', 'Stop.', 'Here's the thing...'")

    # Check 3: Contains curiosity gap
    curiosity_phrases = [
        "you don't", "nobody", "secret", "truth", "hidden", "actually",
        "what if", "imagine", "shocking", "terrifying", "discovered"
    ]
    has_curiosity = any(phrase in hook.lower() for phrase in curiosity_phrases)
    if has_curiosity:
        passed_checks.append("Contains curiosity-building language")
    else:
        failed_checks.append("Missing curiosity gap")
        suggestions.append("Add curiosity: 'What you don't know...', 'The hidden truth...'")

    # Check 4: Specific numbers (where applicable)
    if re.search(r'\d+', hook) or re.search(r'\$[\d,]+', hook):
        passed_checks.append("Contains specific numbers/amounts")
    else:
        failed_checks.append("Missing specific numbers")
        suggestions.append("Add specific figures: '$1,000', '97%', '5 years'")

    # Check 5: Niche-specific elements
    niche_hooks = HOOK_FORMULAS.get(niche, [])
    universal_hooks = HOOK_FORMULAS.get("universal", [])

    # Check for niche-specific keywords
    niche_keywords = {
        "finance": ["invest", "money", "dollar", "$", "wall street", "return", "income"],
        "psychology": ["brain", "mind", "psychology", "think", "fbi", "scientist", "study"],
        "storytelling": ["story", "happened", "door", "truth", "buried", "shocked"],
    }

    keywords = niche_keywords.get(niche, [])
    has_niche_keywords = any(kw in hook.lower() for kw in keywords)
    if has_niche_keywords:
        passed_checks.append(f"Contains {niche}-specific language")
    else:
        failed_checks.append(f"Missing {niche}-specific hook elements")
        suggestions.append(f"Consider niche hooks: {niche_hooks[0] if niche_hooks else universal_hooks[0]}")

    # Check 6: Direct address (uses "you")
    if "you" in hook.lower():
        passed_checks.append("Uses direct address ('you')")
    else:
        failed_checks.append("Missing direct address")
        suggestions.append("Add 'you' to make it personal: 'You're losing...'")

    # Check 7: Ends with open loop or tension
    tension_enders = ["...", "but", "and", "however", "yet"]
    has_tension = any(hook.rstrip().endswith(ender) or ender in hook.lower()[-20:] for ender in tension_enders)
    if has_tension:
        passed_checks.append("Creates tension/open loop")
    else:
        failed_checks.append("Missing tension or open loop")
        suggestions.append("End with '...' or transition word to create anticipation")

    # Calculate score
    total_checks = len(passed_checks) + len(failed_checks)
    score = len(passed_checks) / total_checks if total_checks > 0 else 0.0
    is_valid = score >= 0.5  # 50% threshold

    return ValidationResult(
        is_valid=is_valid,
        score=score,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        suggestions=suggestions
    )


def get_best_practices(niche: str) -> Dict[str, Any]:
    """
    Get all best practices for a specific niche.

    Args:
        niche: The content niche (finance, psychology, storytelling)

    Returns:
        Dictionary containing all best practices for the niche
    """
    niche = niche.lower()

    # Normalize niche aliases
    niche_aliases = {
        "money_blueprints": "finance",
        "money": "finance",
        "investing": "finance",
        "mind_unlocked": "psychology",
        "mind": "psychology",
        "untold_stories": "storytelling",
        "stories": "storytelling",
        "documentary": "storytelling",
    }
    niche = niche_aliases.get(niche, niche)

    if niche not in NICHE_METRICS:
        logger.warning(f"Unknown niche '{niche}', defaulting to finance")
        niche = "finance"

    return {
        "niche": niche,
        "metrics": NICHE_METRICS.get(niche, {}),
        "viral_title_patterns": VIRAL_TITLE_PATTERNS.get(niche, []),
        "hook_formulas": HOOK_FORMULAS.get(niche, []) + HOOK_FORMULAS.get("universal", []),
        "seo_patterns": SEO_PATTERNS.get(niche, {}),
        "power_words": POWER_WORDS,
        "retention_best_practices": RETENTION_BEST_PRACTICES,
        "impact_formula": IMPACT_FORMULA,
    }


def suggest_improvements(content: Dict[str, Any], niche: str) -> List[str]:
    """
    Suggest improvements for video content based on best practices.

    Args:
        content: Dictionary with keys like 'title', 'hook', 'description', 'tags', 'duration'
        niche: The content niche (finance, psychology, storytelling)

    Returns:
        List of improvement suggestions
    """
    suggestions = []
    niche = niche.lower()

    practices = get_best_practices(niche)
    metrics = practices.get("metrics", {})

    # Check title
    if "title" in content:
        title_result = validate_title(content["title"], niche)
        suggestions.extend(title_result.suggestions)

    # Check hook
    if "hook" in content:
        hook_result = validate_hook(content["hook"], niche)
        suggestions.extend(hook_result.suggestions)

    # Check video length
    if "duration" in content:
        duration_minutes = content["duration"]
        if isinstance(duration_minutes, (int, float)):
            optimal_range = metrics.get("optimal_video_length", (8, 15))
            if duration_minutes < optimal_range[0]:
                suggestions.append(
                    f"Video is too short ({duration_minutes} min). "
                    f"Optimal for {niche}: {optimal_range[0]}-{optimal_range[1]} minutes"
                )
            elif duration_minutes > optimal_range[1]:
                sweet_spot = metrics.get("sweet_spot_minutes", 10)
                suggestions.append(
                    f"Video may be too long ({duration_minutes} min). "
                    f"Sweet spot for {niche}: {sweet_spot} minutes"
                )

    # Check tags
    if "tags" in content:
        required_tags = practices.get("seo_patterns", {}).get("required_tags", [])
        current_tags = [t.lower() for t in content["tags"]]
        missing_tags = [t for t in required_tags if t.lower() not in current_tags]
        if missing_tags:
            suggestions.append(f"Consider adding tags: {', '.join(missing_tags[:5])}")

    # Check description
    if "description" in content:
        desc = content["description"]
        # Check for timestamps/chapters
        if not re.search(r'\d{1,2}:\d{2}', desc):
            suggestions.append("Add chapter timestamps to description (e.g., '00:00 Introduction')")
        # Check length
        if len(desc) < 200:
            suggestions.append("Expand description to at least 200 characters for SEO")

    # Remove duplicates while preserving order
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)

    return unique_suggestions


def pre_publish_checklist(
    script: Any,  # VideoScript or dict
    niche: str
) -> PrePublishChecklist:
    """
    Run a complete pre-publish validation checklist.

    Args:
        script: VideoScript object or dictionary with video data
        niche: The content niche (finance, psychology, storytelling)

    Returns:
        PrePublishChecklist with all validation results
    """
    items = []
    critical_issues = []
    niche = niche.lower()

    practices = get_best_practices(niche)
    metrics = practices.get("metrics", {})

    # Convert script to dict if needed
    if hasattr(script, "__dict__"):
        data = {
            "title": getattr(script, "title", ""),
            "description": getattr(script, "description", ""),
            "tags": getattr(script, "tags", []),
            "hook_text": getattr(script, "hook_text", ""),
            "total_duration": getattr(script, "total_duration", 0),
            "sections": getattr(script, "sections", []),
            "chapter_markers": getattr(script, "chapter_markers", []),
        }
    else:
        data = script

    # 1. Title matches viral patterns
    title = data.get("title", "")
    title_result = validate_title(title, niche)
    items.append(ChecklistItem(
        name="Title matches viral patterns",
        passed=title_result.is_valid,
        details=f"Score: {title_result.score:.0%} - {', '.join(title_result.suggestions[:2]) if title_result.suggestions else 'Good'}",
        priority="high"
    ))
    if not title_result.is_valid:
        critical_issues.append(f"Title needs improvement: {title_result.suggestions[0] if title_result.suggestions else 'Check patterns'}")

    # 2. Hook is in first 5 seconds
    hook = data.get("hook_text", "")
    if not hook and data.get("sections"):
        # Try to get from first section
        first_section = data["sections"][0]
        if hasattr(first_section, "narration"):
            hook = first_section.narration
        elif isinstance(first_section, dict):
            hook = first_section.get("narration", "")

    hook_result = validate_hook(hook, niche) if hook else ValidationResult(
        is_valid=False, score=0.0, failed_checks=["No hook found"]
    )
    items.append(ChecklistItem(
        name="Hook is engaging (first 5 seconds)",
        passed=hook_result.is_valid,
        details=f"Score: {hook_result.score:.0%} - {', '.join(hook_result.suggestions[:2]) if hook_result.suggestions else 'Good'}",
        priority="high"
    ))
    if not hook_result.is_valid:
        critical_issues.append("Hook needs to be more engaging")

    # 3. Video length is optimal
    duration_seconds = data.get("total_duration", 0)
    duration_minutes = duration_seconds / 60 if duration_seconds else 0
    optimal_range = metrics.get("optimal_video_length", (8, 15))
    duration_ok = optimal_range[0] <= duration_minutes <= optimal_range[1]
    items.append(ChecklistItem(
        name="Video length is optimal",
        passed=duration_ok,
        details=f"{duration_minutes:.1f} min (optimal: {optimal_range[0]}-{optimal_range[1]} min)",
        priority="medium"
    ))

    # 4. Description follows SEO patterns
    description = data.get("description", "")
    has_timestamps = bool(re.search(r'\d{1,2}:\d{2}', description))
    desc_length_ok = len(description) >= 200
    seo_ok = has_timestamps and desc_length_ok
    items.append(ChecklistItem(
        name="Description follows SEO patterns",
        passed=seo_ok,
        details=f"Timestamps: {'Yes' if has_timestamps else 'No'}, Length: {len(description)} chars",
        priority="medium"
    ))
    if not seo_ok:
        critical_issues.append("Description missing timestamps or too short")

    # 5. Tags are appropriate
    tags = data.get("tags", [])
    required_tags = practices.get("seo_patterns", {}).get("required_tags", [])
    current_tags_lower = [t.lower() for t in tags]
    matching_tags = sum(1 for t in required_tags if t.lower() in current_tags_lower)
    tags_ok = len(tags) >= 10 and matching_tags >= 3
    items.append(ChecklistItem(
        name="Tags are appropriate",
        passed=tags_ok,
        details=f"{len(tags)} tags, {matching_tags}/{len(required_tags)} required tags present",
        priority="medium"
    ))

    # 6. Chapter markers present
    chapters = data.get("chapter_markers", [])
    chapters_ok = len(chapters) >= 3
    items.append(ChecklistItem(
        name="Chapter markers present (3+ required)",
        passed=chapters_ok,
        details=f"{len(chapters)} chapters",
        priority="low"
    ))

    # 7. Sections cover retention points
    sections = data.get("sections", [])
    has_hook_section = any(
        (hasattr(s, "section_type") and s.section_type == "hook") or
        (isinstance(s, dict) and s.get("section_type") == "hook")
        for s in sections
    )
    has_cta_section = any(
        (hasattr(s, "section_type") and s.section_type in ["cta", "outro"]) or
        (isinstance(s, dict) and s.get("section_type") in ["cta", "outro"])
        for s in sections
    )
    retention_ok = has_hook_section and has_cta_section
    items.append(ChecklistItem(
        name="Script has proper retention structure",
        passed=retention_ok,
        details=f"Hook: {'Yes' if has_hook_section else 'No'}, CTA: {'Yes' if has_cta_section else 'No'}",
        priority="high"
    ))

    # Calculate overall score
    total_weight = sum(3 if i.priority == "high" else 2 if i.priority == "medium" else 1 for i in items)
    passed_weight = sum(
        (3 if i.priority == "high" else 2 if i.priority == "medium" else 1)
        for i in items if i.passed
    )
    overall_score = passed_weight / total_weight if total_weight > 0 else 0.0

    # Ready to publish if no critical issues and score >= 70%
    ready_to_publish = len(critical_issues) == 0 and overall_score >= 0.7

    return PrePublishChecklist(
        items=items,
        overall_score=overall_score,
        ready_to_publish=ready_to_publish,
        critical_issues=critical_issues
    )


def get_niche_metrics(niche: str) -> Dict[str, Any]:
    """
    Get key metrics for a specific niche.

    Args:
        niche: The content niche (finance, psychology, storytelling)

    Returns:
        Dictionary with CPM, video length, posting times, etc.
    """
    niche = niche.lower()

    # Normalize niche aliases
    niche_aliases = {
        "money_blueprints": "finance",
        "mind_unlocked": "psychology",
        "untold_stories": "storytelling",
    }
    niche = niche_aliases.get(niche, niche)

    return NICHE_METRICS.get(niche, NICHE_METRICS["finance"])


def get_hook_for_niche(topic: str, niche: str) -> str:
    """
    Get a hook template for a specific niche and topic.

    Args:
        topic: The video topic
        niche: The content niche

    Returns:
        Hook template string
    """
    niche = niche.lower()
    hooks = HOOK_FORMULAS.get(niche, []) + HOOK_FORMULAS.get("universal", [])

    if not hooks:
        return f"What you're about to learn about {topic} will change everything..."

    # Return first hook as template (can be randomized in production)
    hook = hooks[0]
    # Replace placeholder with topic if present
    hook = hook.replace("{topic}", topic)

    return hook


def get_viral_title_templates(niche: str) -> List[str]:
    """
    Get viral title templates for a niche.

    Args:
        niche: The content niche

    Returns:
        List of title template strings
    """
    niche = niche.lower()
    return VIRAL_TITLE_PATTERNS.get(niche, VIRAL_TITLE_PATTERNS["finance"])


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("BEST PRACTICES MODULE TEST")
    print("=" * 60)

    # Test title validation
    print("\n1. Title Validation Test:")
    test_title = "5 Money Mistakes Costing You $1000/Year"
    result = validate_title(test_title, "finance")
    print(f"   Title: {test_title}")
    print(f"   Result: {result}")
    for check in result.passed_checks:
        print(f"   [PASS] {check}")
    for check in result.failed_checks:
        print(f"   [FAIL] {check}")

    # Test hook validation
    print("\n2. Hook Validation Test:")
    test_hook = "Wall Street doesn't want you to know this, but you're losing $347 every month..."
    result = validate_hook(test_hook, "finance")
    print(f"   Hook: {test_hook}")
    print(f"   Result: {result}")

    # Test get best practices
    print("\n3. Get Best Practices Test:")
    practices = get_best_practices("psychology")
    print(f"   Niche: psychology")
    print(f"   CPM Range: ${practices['metrics']['cpm_range'][0]}-${practices['metrics']['cpm_range'][1]}")
    print(f"   Optimal Length: {practices['metrics']['optimal_video_length'][0]}-{practices['metrics']['optimal_video_length'][1]} min")
    print(f"   Hook Formulas: {len(practices['hook_formulas'])} available")

    # Test suggest improvements
    print("\n4. Suggest Improvements Test:")
    content = {
        "title": "Psychology Tips",
        "hook": "Hello everyone, today we'll talk about psychology.",
        "duration": 5,
        "tags": ["tips"],
    }
    suggestions = suggest_improvements(content, "psychology")
    print(f"   Content: {content}")
    print(f"   Suggestions:")
    for s in suggestions[:5]:
        print(f"   - {s}")

    # Test pre-publish checklist
    print("\n5. Pre-Publish Checklist Test:")
    test_script = {
        "title": "5 Signs of a Narcissist That Will Shock You",
        "description": "Learn the hidden signs of narcissism. 00:00 Introduction 01:30 Sign 1 03:00 Sign 2",
        "tags": ["psychology", "narcissist", "dark psychology", "manipulation", "mental health",
                "self improvement", "mindset", "personality", "body language", "cognitive bias"],
        "hook_text": "Your brain is lying to you right now, and you don't even know it...",
        "total_duration": 600,  # 10 minutes
        "sections": [{"section_type": "hook", "narration": "Your brain is lying..."},
                     {"section_type": "outro", "narration": "Subscribe now!"}],
        "chapter_markers": [{"timestamp_seconds": 0}, {"timestamp_seconds": 90}, {"timestamp_seconds": 180}],
    }
    checklist = pre_publish_checklist(test_script, "psychology")
    print(f"   Result: {checklist}")
    print(f"   Ready to Publish: {checklist.ready_to_publish}")
    for item in checklist.items:
        status = "[PASS]" if item.passed else "[FAIL]"
        print(f"   {status} {item.name}: {item.details}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
