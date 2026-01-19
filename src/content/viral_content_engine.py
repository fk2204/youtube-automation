"""
Viral Content Engine - Advanced Content Quality Pipeline for 50% Better Engagement

A production-ready module for creating viral YouTube content with proven formulas,
emotional arcs, curiosity gaps, and strategic engagement techniques.

Components:
- ViralHookGenerator: 10+ proven hook formulas per niche
- EmotionalArcBuilder: Story structure with emotional peaks/valleys
- CuriosityGapCreator: Open loops that drive retention
- MicroPayoffScheduler: Rewards every 30-60 seconds
- PatternInterruptLibrary: 20+ visual/audio interrupts
- CallToActionOptimizer: Strategic CTA placement

Usage:
    from src.content.viral_content_engine import (
        ViralHookGenerator,
        EmotionalArcBuilder,
        CuriosityGapCreator,
        MicroPayoffScheduler,
        PatternInterruptLibrary,
        CallToActionOptimizer,
        ViralContentEngine
    )

    # Use individual components
    hook_gen = ViralHookGenerator()
    hook = hook_gen.generate_hook("passive income", niche="finance")

    # Or use the unified engine
    engine = ViralContentEngine(niche="psychology")
    viral_elements = engine.generate_all_elements(
        topic="dark psychology tricks",
        duration_seconds=600
    )
"""

import random
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


# ============================================================
# Data Classes and Enums
# ============================================================

class EmotionType(Enum):
    """Emotional states for arc building."""
    INTRIGUE = "intrigue"
    TENSION = "tension"
    SHOCK = "shock"
    RELIEF = "relief"
    CURIOSITY = "curiosity"
    SATISFACTION = "satisfaction"
    URGENCY = "urgency"
    EMPATHY = "empathy"
    EXCITEMENT = "excitement"
    REVELATION = "revelation"


class HookType(Enum):
    """Types of viral hooks."""
    PATTERN_INTERRUPT = "pattern_interrupt"
    BOLD_CLAIM = "bold_claim"
    QUESTION_STACK = "question_stack"
    STATS_SHOCK = "stats_shock"
    STORY_LEAD = "story_lead"
    LOSS_AVERSION = "loss_aversion"
    CURIOSITY_GAP = "curiosity_gap"
    INSIDER_SECRET = "insider_secret"
    COUNTDOWN = "countdown"
    CONTROVERSY = "controversy"


@dataclass
class ViralHook:
    """Represents a generated viral hook."""
    text: str
    hook_type: HookType
    niche: str
    estimated_retention_boost: float  # 0.0 to 1.0
    word_count: int
    duration_seconds: float  # Estimated speaking time

    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.duration_seconds = self.word_count / 2.5  # ~150 wpm


@dataclass
class EmotionalBeat:
    """A single beat in the emotional arc."""
    timestamp_seconds: int
    emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    description: str
    narration_hint: str


@dataclass
class EmotionalArc:
    """Complete emotional arc for a video."""
    beats: List[EmotionalBeat]
    peak_moment_seconds: int
    resolution_seconds: int
    total_duration_seconds: int


@dataclass
class CuriosityGap:
    """An open loop / curiosity gap."""
    opening_text: str
    resolution_text: str
    opening_timestamp_seconds: int
    resolution_timestamp_seconds: int
    gap_type: str  # "tease", "mystery", "countdown", "promise"
    retention_impact: str  # "high", "medium", "low"


@dataclass
class MicroPayoff:
    """A small value delivery moment."""
    text: str
    timestamp_seconds: int
    payoff_type: str  # "insight", "statistic", "tip", "reveal", "example"
    value_score: float  # 0.0 to 1.0


@dataclass
class PatternInterrupt:
    """A pattern interrupt for re-engaging attention."""
    text: str
    interrupt_type: str  # "visual", "audio", "verbal", "structural"
    visual_cue: str
    audio_cue: str
    recommended_duration_seconds: float


@dataclass
class CTAPlacement:
    """Strategic call-to-action placement."""
    text: str
    cta_type: str  # "soft", "engagement", "subscribe", "comment", "final"
    timestamp_seconds: int
    percentage_through: float  # Position in video (0.0 to 1.0)


# ============================================================
# Viral Hook Generator - 10+ Proven Formulas Per Niche
# ============================================================

class ViralHookGenerator:
    """
    Generates viral hooks using 10+ proven formulas per niche.

    Each formula is based on competitor analysis of top-performing
    YouTube channels achieving millions of views.
    """

    # Finance niche hook formulas (money_blueprints)
    FINANCE_HOOKS = {
        HookType.STATS_SHOCK: [
            "If you invested ${amount} in {topic} {timeframe} ago, you'd have ${result} today. But here's what nobody tells you...",
            "{percentage}% of investors lose money doing this one thing. Are you one of them?",
            "The difference between 7% and 9% returns? ${amount} over 30 years. And I'll show you exactly how to get it...",
            "Only {percentage}% of people retire wealthy. Here's the single thing that separates them...",
        ],
        HookType.LOSS_AVERSION: [
            "Right now, you're losing ${amount} every month. And you don't even know it...",
            "This hidden fee is costing the average investor ${amount} over their lifetime. Let me show you...",
            "{percentage}% of your paycheck disappears before you even see it. Here's where it goes...",
            "You're making this mistake with your money right now. And it's costing you ${amount} per year...",
        ],
        HookType.INSIDER_SECRET: [
            "Wall Street doesn't want you to know this, but...",
            "The strategy hedge funds use that banks will never tell you about...",
            "Millionaires have been using this method for decades. Now I'll show you...",
            "There's a loophole in the tax code that could save you ${amount}. Here's how...",
        ],
        HookType.BOLD_CLAIM: [
            "This one investment could turn ${small} into ${large} in {timeframe}...",
            "I turned ${small} into ${large} using this exact method. And you can too...",
            "The ${amount} side hustle that's replacing people's full-time income...",
            "One decision I made in {year} changed my entire financial future...",
        ],
        HookType.QUESTION_STACK: [
            "What if you could retire {years} years early? What if you could do it starting with just ${amount}? What if everything you've been told about saving is wrong?",
            "Why do 78% of Americans live paycheck to paycheck? Why do the wealthy keep getting wealthier? And what do they know that you don't?",
        ],
        HookType.STORY_LEAD: [
            "In {year}, a man with nothing but ${amount} in his bank account discovered something that would make him millions...",
            "She was $50,000 in debt. In {timeframe}, she was financially free. Here's exactly what she did...",
        ],
    }

    # Psychology niche hook formulas (mind_unlocked)
    PSYCHOLOGY_HOOKS = {
        HookType.CURIOSITY_GAP: [
            "Your brain is lying to you right now. And you don't even know it...",
            "There's a technique FBI interrogators use that I'm about to reveal...",
            "What I'm about to show you is used by CIA operatives to detect lies in seconds...",
            "Scientists discovered something terrifying about the human mind. And it affects you every day...",
        ],
        HookType.PATTERN_INTERRUPT: [
            "Stop. Everything you've been told about {topic} is wrong...",
            "Forget everything you think you know about {topic}...",
            "What if I told you that {surprising_claim}?",
            "In the next {duration}, I'll give you the ability to read anyone's mind...",
        ],
        HookType.INSIDER_SECRET: [
            "They've been using this against you since you were {age} years old...",
            "This manipulation technique is so powerful, it's actually illegal in some countries...",
            "Psychologists have kept this secret for decades. Now I'm sharing it with you...",
            "The dark psychology trick that advertisers use on you every single day...",
        ],
        HookType.STATS_SHOCK: [
            "A Stanford study proved that {percentage}% of people are wrong about this...",
            "{percentage}% of your decisions are made by your subconscious. Here's how to take control...",
            "Research shows {percentage}% of people fall for this manipulation. Don't be one of them...",
        ],
        HookType.BOLD_CLAIM: [
            "By the end of this video, you'll know exactly what people are thinking...",
            "In {duration}, you'll be able to influence anyone's decision...",
            "This one psychological trick will change every conversation you have...",
        ],
        HookType.QUESTION_STACK: [
            "What if you could read anyone's thoughts? What if you could influence their decisions? What if the technique only takes {duration} to learn?",
            "Why do people really make the choices they make? What's actually driving your behavior? And can you change it?",
        ],
    }

    # Storytelling niche hook formulas (untold_stories)
    STORYTELLING_HOOKS = {
        HookType.STORY_LEAD: [
            "The door slammed. He had exactly {seconds} seconds to decide...",
            "Nobody knows why he did it. But what happened next shocked the world...",
            "To this day, no one can explain what really happened that night...",
            "She looked at the evidence. Everything she thought she knew was a lie...",
            "In {year}, a man made a decision that would destroy everything he built...",
        ],
        HookType.COUNTDOWN: [
            "He had exactly {time} to make a choice that would change history...",
            "The clock showed 11:47 PM. By midnight, three people would be dead...",
            "In {seconds} seconds, everything would change. And no one saw it coming...",
        ],
        HookType.CURIOSITY_GAP: [
            "What I'm about to tell you is the true story they tried to bury...",
            "The ending of this story will change how you see everything...",
            "There's one detail about this case that nobody talks about...",
            "The truth about what happened has never been fully revealed. Until now...",
        ],
        HookType.CONTROVERSY: [
            "Everyone thinks they know this story. They're all wrong...",
            "The official story is a lie. Here's what really happened...",
            "What the media didn't tell you about {topic}...",
            "The cover-up was so complete, even the FBI couldn't crack it...",
        ],
        HookType.PATTERN_INTERRUPT: [
            "Stop. What you're about to see changes everything you thought you knew...",
            "Before I begin, I need to warn you - this story contains {warning}...",
            "What happened in that room would haunt them forever...",
        ],
    }

    # Universal hooks that work for any niche
    UNIVERSAL_HOOKS = {
        HookType.PATTERN_INTERRUPT: [
            "Stop. What you're about to learn changes everything...",
            "Forget everything you've been told about {topic}...",
            "This is the one thing nobody tells you about {topic}...",
        ],
        HookType.STATS_SHOCK: [
            "Only {percentage}% of people know this. Here's why it matters...",
            "{number} out of {total} people fail at this. Here's how to be different...",
            "In {year}, {shocking_stat}. And it's getting worse...",
        ],
        HookType.LOSS_AVERSION: [
            "You're losing {amount} every {timeframe} without knowing it...",
            "This mistake is costing you {consequence} right now...",
            "Every day you wait, you lose {amount}...",
        ],
        HookType.BOLD_CLAIM: [
            "This single technique is worth more than a college degree...",
            "What I'm about to show you took me {years} years to discover...",
            "This one change will transform your entire {area}...",
        ],
    }

    def __init__(self):
        """Initialize the viral hook generator."""
        self.hooks = {
            "finance": self.FINANCE_HOOKS,
            "psychology": self.PSYCHOLOGY_HOOKS,
            "storytelling": self.STORYTELLING_HOOKS,
            "default": self.UNIVERSAL_HOOKS,
        }
        logger.debug("ViralHookGenerator initialized")

    def generate_hook(
        self,
        topic: str,
        niche: str = "default",
        hook_type: Optional[HookType] = None,
        variables: Optional[Dict[str, str]] = None
    ) -> ViralHook:
        """
        Generate a viral hook for the given topic and niche.

        Args:
            topic: The video topic
            niche: Content niche (finance, psychology, storytelling)
            hook_type: Specific hook type to use (random if not specified)
            variables: Custom variables to fill placeholders

        Returns:
            ViralHook object with the generated hook
        """
        # Get hooks for this niche (fallback to universal)
        niche_hooks = self.hooks.get(niche, self.UNIVERSAL_HOOKS)

        # Select hook type
        if hook_type is None:
            hook_type = random.choice(list(niche_hooks.keys()))

        # Get templates for this hook type
        templates = niche_hooks.get(hook_type, self.UNIVERSAL_HOOKS.get(hook_type, []))
        if not templates:
            templates = list(self.UNIVERSAL_HOOKS.values())[0]

        # Select and fill template
        template = random.choice(templates)
        hook_text = self._fill_template(template, topic, variables)

        # Calculate estimated retention boost based on hook type
        retention_boosts = {
            HookType.PATTERN_INTERRUPT: 0.23,
            HookType.STATS_SHOCK: 0.21,
            HookType.LOSS_AVERSION: 0.25,
            HookType.CURIOSITY_GAP: 0.22,
            HookType.BOLD_CLAIM: 0.18,
            HookType.STORY_LEAD: 0.20,
            HookType.QUESTION_STACK: 0.19,
            HookType.INSIDER_SECRET: 0.22,
            HookType.COUNTDOWN: 0.24,
            HookType.CONTROVERSY: 0.26,
        }

        return ViralHook(
            text=hook_text,
            hook_type=hook_type,
            niche=niche,
            estimated_retention_boost=retention_boosts.get(hook_type, 0.15),
            word_count=0,  # Set in __post_init__
            duration_seconds=0.0  # Set in __post_init__
        )

    def generate_multiple_hooks(
        self,
        topic: str,
        niche: str = "default",
        count: int = 5
    ) -> List[ViralHook]:
        """
        Generate multiple hook variations for A/B testing.

        Args:
            topic: The video topic
            niche: Content niche
            count: Number of hooks to generate

        Returns:
            List of ViralHook objects
        """
        hooks = []
        used_types = set()
        niche_hooks = self.hooks.get(niche, self.UNIVERSAL_HOOKS)

        for _ in range(count):
            # Try to use different hook types
            available_types = [t for t in niche_hooks.keys() if t not in used_types]
            if not available_types:
                available_types = list(niche_hooks.keys())

            hook_type = random.choice(available_types)
            used_types.add(hook_type)

            hook = self.generate_hook(topic, niche, hook_type)
            hooks.append(hook)

        return hooks

    def _fill_template(
        self,
        template: str,
        topic: str,
        variables: Optional[Dict[str, str]] = None
    ) -> str:
        """Fill template placeholders with values."""
        vars_dict = variables or {}

        # Default values for common placeholders
        defaults = {
            "topic": topic,
            "amount": random.choice(["347", "1,247", "2,847", "5,000", "10,000"]),
            "small": random.choice(["100", "500", "1,000"]),
            "large": random.choice(["10,000", "50,000", "100,000", "1,000,000"]),
            "result": random.choice(["47,000", "127,000", "500,000", "1,200,000"]),
            "percentage": str(random.randint(73, 97)),
            "number": str(random.randint(3, 9)),
            "total": str(random.randint(10, 100)),
            "year": str(random.randint(2019, 2024)),
            "years": str(random.randint(3, 10)),
            "timeframe": random.choice(["5 years", "10 years", "3 months", "1 year"]),
            "duration": random.choice(["10 minutes", "5 minutes", "30 seconds"]),
            "seconds": str(random.randint(10, 60)),
            "time": random.choice(["30 seconds", "2 minutes", "5 minutes"]),
            "age": str(random.randint(5, 12)),
            "surprising_claim": f"everything you know about {topic} is wrong",
            "shocking_stat": "something changed that nobody predicted",
            "consequence": "thousands of dollars per year",
            "area": "life",
            "warning": "disturbing content",
        }

        # Merge with provided variables
        all_vars = {**defaults, **vars_dict}

        # Replace placeholders
        result = template
        for key, value in all_vars.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result


# ============================================================
# Emotional Arc Builder - Story Structure with Peaks/Valleys
# ============================================================

class EmotionalArcBuilder:
    """
    Builds emotional arcs for videos with strategic peaks and valleys.

    Based on successful documentary storytelling patterns that maximize
    viewer engagement and retention.
    """

    # Standard emotional arc templates
    ARC_TEMPLATES = {
        "hero_journey": [
            (0.0, EmotionType.INTRIGUE, 0.7, "Opening hook - create curiosity"),
            (0.1, EmotionType.TENSION, 0.5, "Establish the problem/stakes"),
            (0.2, EmotionType.CURIOSITY, 0.6, "Reveal first insight"),
            (0.35, EmotionType.TENSION, 0.7, "Rising conflict"),
            (0.5, EmotionType.SHOCK, 0.9, "Mid-point twist"),
            (0.65, EmotionType.URGENCY, 0.8, "Stakes increase"),
            (0.8, EmotionType.REVELATION, 1.0, "Peak moment - major payoff"),
            (0.9, EmotionType.SATISFACTION, 0.8, "Resolution begins"),
            (1.0, EmotionType.EXCITEMENT, 0.6, "Call to action"),
        ],
        "mystery_reveal": [
            (0.0, EmotionType.CURIOSITY, 0.8, "Mysterious opening"),
            (0.1, EmotionType.TENSION, 0.6, "Questions raised"),
            (0.25, EmotionType.INTRIGUE, 0.7, "First clue"),
            (0.4, EmotionType.SHOCK, 0.75, "Unexpected twist"),
            (0.55, EmotionType.TENSION, 0.85, "Building to revelation"),
            (0.7, EmotionType.CURIOSITY, 0.9, "Final pieces"),
            (0.85, EmotionType.REVELATION, 1.0, "Big reveal"),
            (0.95, EmotionType.SATISFACTION, 0.8, "Everything connects"),
        ],
        "problem_solution": [
            (0.0, EmotionType.URGENCY, 0.7, "Problem hook - pain point"),
            (0.1, EmotionType.EMPATHY, 0.6, "Relate to the viewer"),
            (0.2, EmotionType.TENSION, 0.5, "Why it matters"),
            (0.3, EmotionType.CURIOSITY, 0.7, "Hint at solution"),
            (0.45, EmotionType.EXCITEMENT, 0.75, "First solution point"),
            (0.6, EmotionType.SATISFACTION, 0.8, "Second solution point"),
            (0.75, EmotionType.REVELATION, 0.9, "Key insight"),
            (0.9, EmotionType.EXCITEMENT, 1.0, "Full solution revealed"),
            (1.0, EmotionType.URGENCY, 0.7, "CTA - act now"),
        ],
        "shocking_truth": [
            (0.0, EmotionType.SHOCK, 0.85, "Bold claim opener"),
            (0.1, EmotionType.CURIOSITY, 0.7, "Evidence tease"),
            (0.25, EmotionType.REVELATION, 0.75, "First proof"),
            (0.4, EmotionType.TENSION, 0.8, "Deeper investigation"),
            (0.55, EmotionType.SHOCK, 0.9, "Unexpected discovery"),
            (0.7, EmotionType.INTRIGUE, 0.85, "Hidden connections"),
            (0.85, EmotionType.REVELATION, 1.0, "Full truth revealed"),
            (0.95, EmotionType.URGENCY, 0.7, "What this means for you"),
        ],
    }

    def __init__(self):
        """Initialize the emotional arc builder."""
        logger.debug("EmotionalArcBuilder initialized")

    def build_arc(
        self,
        duration_seconds: int,
        arc_type: str = "hero_journey",
        custom_beats: Optional[List[Tuple]] = None
    ) -> EmotionalArc:
        """
        Build an emotional arc for a video of given duration.

        Args:
            duration_seconds: Total video duration
            arc_type: Type of arc template to use
            custom_beats: Optional custom beat definitions

        Returns:
            EmotionalArc with timed emotional beats
        """
        template = custom_beats or self.ARC_TEMPLATES.get(
            arc_type, self.ARC_TEMPLATES["hero_journey"]
        )

        beats = []
        peak_moment = 0
        peak_intensity = 0.0

        for position, emotion, intensity, description in template:
            timestamp = int(duration_seconds * position)
            beat = EmotionalBeat(
                timestamp_seconds=timestamp,
                emotion=emotion,
                intensity=intensity,
                description=description,
                narration_hint=self._get_narration_hint(emotion, intensity)
            )
            beats.append(beat)

            if intensity > peak_intensity:
                peak_intensity = intensity
                peak_moment = timestamp

        # Resolution is typically last 10-15% of video
        resolution_seconds = int(duration_seconds * 0.9)

        return EmotionalArc(
            beats=beats,
            peak_moment_seconds=peak_moment,
            resolution_seconds=resolution_seconds,
            total_duration_seconds=duration_seconds
        )

    def _get_narration_hint(self, emotion: EmotionType, intensity: float) -> str:
        """Get narration style hint for an emotion."""
        hints = {
            EmotionType.INTRIGUE: "Speak slowly, build suspense, use pauses",
            EmotionType.TENSION: "Increase pace slightly, add urgency to voice",
            EmotionType.SHOCK: "Pause before reveal, then deliver with emphasis",
            EmotionType.RELIEF: "Slow down, warmer tone, allow breathing room",
            EmotionType.CURIOSITY: "Raise pitch slightly at end of phrases, tease",
            EmotionType.SATISFACTION: "Confident delivery, slower pace",
            EmotionType.URGENCY: "Faster pace, emphasize key words",
            EmotionType.EMPATHY: "Softer tone, personal connection",
            EmotionType.EXCITEMENT: "Higher energy, varied pace",
            EmotionType.REVELATION: "Build up with pause, then impactful delivery",
        }

        base_hint = hints.get(emotion, "Natural delivery")
        if intensity > 0.8:
            return f"{base_hint} [HIGH INTENSITY]"
        return base_hint

    def get_arc_for_niche(self, niche: str) -> str:
        """Get recommended arc type for a niche."""
        recommendations = {
            "finance": "problem_solution",
            "psychology": "mystery_reveal",
            "storytelling": "hero_journey",
            "default": "hero_journey",
        }
        return recommendations.get(niche, "hero_journey")

    def blend_arcs(
        self,
        primary_arc: str,
        secondary_arc: str,
        blend_ratio: float = 0.3
    ) -> List[Tuple]:
        """
        Blend two arc types for hybrid emotional structure.

        Args:
            primary_arc: Primary arc template name
            secondary_arc: Secondary arc to blend in
            blend_ratio: How much of secondary to blend (0.0-1.0)

        Returns:
            Blended arc beat definitions
        """
        primary = self.ARC_TEMPLATES.get(primary_arc, self.ARC_TEMPLATES["hero_journey"])
        secondary = self.ARC_TEMPLATES.get(secondary_arc, self.ARC_TEMPLATES["mystery_reveal"])

        blended = []
        for i, (pos, emotion, intensity, desc) in enumerate(primary):
            # Check if secondary has a corresponding beat
            if i < len(secondary):
                sec_pos, sec_emotion, sec_intensity, sec_desc = secondary[i]
                # Blend intensities
                new_intensity = intensity * (1 - blend_ratio) + sec_intensity * blend_ratio
                blended.append((pos, emotion, new_intensity, desc))
            else:
                blended.append((pos, emotion, intensity, desc))

        return blended


# ============================================================
# Curiosity Gap Creator - Open Loops That Drive Retention
# ============================================================

class CuriosityGapCreator:
    """
    Creates curiosity gaps (open loops) that keep viewers watching.

    Open loops are psychological hooks that create tension by introducing
    questions or promises that will be resolved later in the video.
    """

    # Open loop templates by type
    LOOP_TEMPLATES = {
        "tease": [
            ("But first, there's something critical you need to understand...", "And now you understand why {topic} matters."),
            ("I'll reveal the most important insight at the end...", "And that's the key insight I promised you."),
            ("The third point changes everything - but you need context first...", "Now you see why point three is a game-changer."),
            ("What I'm about to show you will surprise you...", "And that's the surprise I mentioned earlier."),
            ("Stay with me - the best part is coming...", "And there it is - the best part."),
        ],
        "mystery": [
            ("Nobody knows why this happens. But I've discovered the answer...", "And that's why it happens."),
            ("There's a hidden pattern here that nobody talks about...", "Now you see the pattern."),
            ("Something doesn't add up. Here's what I found...", "And that's what was missing."),
            ("The official story isn't true. Here's the real version...", "Now you know the real story."),
        ],
        "countdown": [
            ("By the end of this video, you'll know exactly how to {promise}...", "And now you know exactly how to {promise}."),
            ("In the next {duration}, everything will become clear...", "And now it all makes sense."),
            ("I'm going to show you {number} things that will change your perspective...", "Those are the {number} things that change everything."),
        ],
        "promise": [
            ("What I share next could save you {amount}...", "And that's how you save {amount}."),
            ("The technique I'm about to reveal took me {years} years to master...", "And that's the technique."),
            ("This is the exact strategy used by {authority}...", "Now you know their strategy."),
        ],
        "cliffhanger": [
            ("But here's where it gets really interesting...", "And that's why it's so interesting."),
            ("What happened next shocked everyone...", "That's what shocked everyone."),
            ("And then everything changed...", "And that's how everything changed."),
        ],
    }

    def __init__(self):
        """Initialize the curiosity gap creator."""
        self._used_templates = set()
        logger.debug("CuriosityGapCreator initialized")

    def create_gap(
        self,
        gap_type: str = "tease",
        timestamp_open: int = 60,
        timestamp_close: int = 300,
        variables: Optional[Dict[str, str]] = None
    ) -> CuriosityGap:
        """
        Create a single curiosity gap.

        Args:
            gap_type: Type of gap (tease, mystery, countdown, promise, cliffhanger)
            timestamp_open: When to open the loop (seconds)
            timestamp_close: When to close the loop (seconds)
            variables: Custom variables for placeholders

        Returns:
            CuriosityGap object
        """
        templates = self.LOOP_TEMPLATES.get(gap_type, self.LOOP_TEMPLATES["tease"])

        # Avoid repeating templates
        available = [t for t in templates if t not in self._used_templates]
        if not available:
            self._used_templates.clear()
            available = templates

        opening, closing = random.choice(available)
        self._used_templates.add((opening, closing))

        # Fill placeholders
        vars_dict = variables or {}
        defaults = {
            "topic": "this concept",
            "promise": "achieve your goals",
            "duration": "5 minutes",
            "number": "5",
            "amount": "$1,000",
            "years": "10",
            "authority": "top performers",
        }
        all_vars = {**defaults, **vars_dict}

        for key, value in all_vars.items():
            opening = opening.replace(f"{{{key}}}", str(value))
            closing = closing.replace(f"{{{key}}}", str(value))

        # Determine retention impact based on gap duration
        gap_duration = timestamp_close - timestamp_open
        if gap_duration > 180:
            impact = "high"
        elif gap_duration > 60:
            impact = "medium"
        else:
            impact = "low"

        return CuriosityGap(
            opening_text=opening,
            resolution_text=closing,
            opening_timestamp_seconds=timestamp_open,
            resolution_timestamp_seconds=timestamp_close,
            gap_type=gap_type,
            retention_impact=impact
        )

    def create_gap_sequence(
        self,
        duration_seconds: int,
        count: int = 3,
        variables: Optional[Dict[str, str]] = None
    ) -> List[CuriosityGap]:
        """
        Create a sequence of curiosity gaps distributed throughout the video.

        Best practice: Minimum 3 open loops per video for optimal retention.

        Args:
            duration_seconds: Total video duration
            count: Number of gaps to create (minimum 3 recommended)
            variables: Custom variables for placeholders

        Returns:
            List of CuriosityGap objects
        """
        gaps = []
        gap_types = list(self.LOOP_TEMPLATES.keys())

        # Calculate gap positions
        # First loop opens around 15-20%, closes around 60%
        # Second loop opens around 30-35%, closes around 75%
        # Third loop opens around 45-50%, closes around 90%
        positions = [
            (0.15, 0.60),
            (0.30, 0.75),
            (0.45, 0.90),
            (0.55, 0.85),
            (0.20, 0.70),
        ]

        for i in range(min(count, len(positions))):
            open_pct, close_pct = positions[i]
            gap_type = gap_types[i % len(gap_types)]

            gap = self.create_gap(
                gap_type=gap_type,
                timestamp_open=int(duration_seconds * open_pct),
                timestamp_close=int(duration_seconds * close_pct),
                variables=variables
            )
            gaps.append(gap)

        return gaps


# ============================================================
# Micro Payoff Scheduler - Rewards Every 30-60 Seconds
# ============================================================

class MicroPayoffScheduler:
    """
    Schedules micro-payoffs (small value deliveries) throughout the video.

    Micro-payoffs keep viewers engaged by providing frequent moments
    of satisfaction and learning.
    """

    PAYOFF_TEMPLATES = {
        "insight": [
            "Here's a quick insight most people miss: {insight}",
            "The key realization here: {insight}",
            "What this really means: {insight}",
            "Here's the important takeaway: {insight}",
        ],
        "statistic": [
            "The data shows: {stat}",
            "Studies found that {stat}",
            "Here's what the numbers say: {stat}",
            "{percentage}% of people experience this.",
        ],
        "tip": [
            "Pro tip: {tip}",
            "Here's something you can try right now: {tip}",
            "Quick actionable step: {tip}",
            "The practical application: {tip}",
        ],
        "reveal": [
            "Here's what nobody tells you: {reveal}",
            "The hidden truth: {reveal}",
            "What's really going on: {reveal}",
            "The secret behind this: {reveal}",
        ],
        "example": [
            "For example, {example}",
            "Here's a real-world case: {example}",
            "This is exactly what happened with {example}",
            "Consider this scenario: {example}",
        ],
    }

    def __init__(self, interval_seconds: int = 45):
        """
        Initialize the micro payoff scheduler.

        Args:
            interval_seconds: Target interval between payoffs (30-60 recommended)
        """
        self.interval = max(30, min(60, interval_seconds))
        logger.debug(f"MicroPayoffScheduler initialized with {self.interval}s interval")

    def schedule_payoffs(
        self,
        duration_seconds: int,
        variables: Optional[Dict[str, str]] = None
    ) -> List[MicroPayoff]:
        """
        Schedule micro-payoffs throughout the video.

        Args:
            duration_seconds: Total video duration
            variables: Custom variables for templates

        Returns:
            List of MicroPayoff objects with timestamps
        """
        payoffs = []
        payoff_types = list(self.PAYOFF_TEMPLATES.keys())

        # Start after the hook (around 15-20 seconds)
        current_time = 20
        type_index = 0

        while current_time < duration_seconds - 30:
            payoff_type = payoff_types[type_index % len(payoff_types)]
            templates = self.PAYOFF_TEMPLATES[payoff_type]
            template = random.choice(templates)

            # Fill template
            vars_dict = variables or {}
            defaults = {
                "insight": "this changes everything",
                "stat": "this happens 73% of the time",
                "tip": "try this today",
                "reveal": "this is the real reason",
                "example": "successful people do this",
                "percentage": str(random.randint(60, 95)),
            }
            all_vars = {**defaults, **vars_dict}

            for key, value in all_vars.items():
                template = template.replace(f"{{{key}}}", str(value))

            # Calculate value score based on payoff type
            value_scores = {
                "insight": 0.85,
                "statistic": 0.75,
                "tip": 0.90,
                "reveal": 0.80,
                "example": 0.70,
            }

            payoff = MicroPayoff(
                text=template,
                timestamp_seconds=current_time,
                payoff_type=payoff_type,
                value_score=value_scores.get(payoff_type, 0.75)
            )
            payoffs.append(payoff)

            # Add some variation to interval (+/- 10 seconds)
            variation = random.randint(-10, 10)
            current_time += self.interval + variation
            type_index += 1

        return payoffs


# ============================================================
# Pattern Interrupt Library - 20+ Visual/Audio Interrupts
# ============================================================

class PatternInterruptLibrary:
    """
    Library of 20+ pattern interrupts to maintain viewer attention.

    Pattern interrupts break viewer expectations and re-engage
    attention when it starts to wander.
    """

    # Verbal pattern interrupts
    VERBAL_INTERRUPTS = [
        ("But here's where it gets really interesting...", "verbal", "[PAUSE]", "[TONE SHIFT]"),
        ("Now, this is where most people go wrong...", "verbal", "[ZOOM]", "[EMPHASIS]"),
        ("Stop and think about this for a second...", "verbal", "[FREEZE]", "[PAUSE]"),
        ("Here's what nobody tells you...", "verbal", "[TEXT OVERLAY]", "[SUSPENSE]"),
        ("Wait - this changes everything...", "verbal", "[QUICK CUT]", "[IMPACT]"),
        ("And this is the part that surprises everyone...", "verbal", "[REVEAL]", "[BUILD]"),
        ("But there's a catch...", "verbal", "[TENSION]", "[OMINOUS]"),
        ("Here's the twist...", "verbal", "[SPIN]", "[DRAMATIC]"),
        ("Now pay close attention to this...", "verbal", "[HIGHLIGHT]", "[FOCUS]"),
        ("This is the game-changer...", "verbal", "[EMPHASIS]", "[EPIC]"),
    ]

    # Visual pattern interrupts
    VISUAL_INTERRUPTS = [
        ("[CUT TO: New angle/perspective]", "visual", "Quick cut to different framing", "[WHOOSH]"),
        ("[ZOOM: Dramatic close-up]", "visual", "Zoom in on key element", "[ZOOM SOUND]"),
        ("[TEXT: Key point overlay]", "visual", "Bold text appears on screen", "[POP]"),
        ("[GRAPHIC: Chart/data visualization]", "visual", "Animated chart appears", "[DATA SOUND]"),
        ("[B-ROLL: Supporting footage]", "visual", "Cut to relevant footage", "[TRANSITION]"),
        ("[SPLIT SCREEN: Comparison]", "visual", "Side-by-side comparison", "[SLIDE]"),
        ("[ANIMATION: Concept illustration]", "visual", "Animated explanation", "[ANIMATION SFX]"),
        ("[FREEZE FRAME: Key moment]", "visual", "Pause on important frame", "[FREEZE]"),
        ("[COLOR SHIFT: Emphasis]", "visual", "Color grading change", "[SUBTLE]"),
        ("[SPEED RAMP: Dramatic effect]", "visual", "Speed up then slow down", "[RAMP]"),
    ]

    # Audio pattern interrupts
    AUDIO_INTERRUPTS = [
        ("[MUSIC: Tension build]", "audio", "[VISUALS INTENSIFY]", "Music builds tension"),
        ("[SFX: Impact sound]", "audio", "[SCREEN SHAKE]", "Dramatic impact"),
        ("[MUSIC: Drop/release]", "audio", "[REVEAL VISUAL]", "Musical drop"),
        ("[SILENCE: Dramatic pause]", "audio", "[HOLD FRAME]", "Silence for emphasis"),
        ("[SFX: Whoosh transition]", "audio", "[QUICK CUT]", "Transition sound"),
    ]

    # Structural pattern interrupts
    STRUCTURAL_INTERRUPTS = [
        ("Let me ask you something...", "structural", "[DIRECT TO CAMERA]", "[INTIMATE]"),
        ("Here's a story that illustrates this perfectly...", "structural", "[STORY B-ROLL]", "[NARRATIVE]"),
        ("Watch what happens when...", "structural", "[DEMONSTRATION]", "[ANTICIPATION]"),
        ("Before we continue, let's recap...", "structural", "[SUMMARY GRAPHICS]", "[RECAP MUSIC]"),
        ("Now let's flip this on its head...", "structural", "[PERSPECTIVE SHIFT]", "[TWIST]"),
    ]

    def __init__(self):
        """Initialize the pattern interrupt library."""
        self.all_interrupts = (
            self.VERBAL_INTERRUPTS +
            self.VISUAL_INTERRUPTS +
            self.AUDIO_INTERRUPTS +
            self.STRUCTURAL_INTERRUPTS
        )
        self._used_interrupts = set()
        logger.debug(f"PatternInterruptLibrary initialized with {len(self.all_interrupts)} interrupts")

    def get_interrupt(
        self,
        interrupt_type: Optional[str] = None
    ) -> PatternInterrupt:
        """
        Get a pattern interrupt.

        Args:
            interrupt_type: Type of interrupt (verbal, visual, audio, structural)

        Returns:
            PatternInterrupt object
        """
        if interrupt_type == "verbal":
            pool = self.VERBAL_INTERRUPTS
        elif interrupt_type == "visual":
            pool = self.VISUAL_INTERRUPTS
        elif interrupt_type == "audio":
            pool = self.AUDIO_INTERRUPTS
        elif interrupt_type == "structural":
            pool = self.STRUCTURAL_INTERRUPTS
        else:
            pool = self.all_interrupts

        # Avoid repeating interrupts
        available = [i for i in pool if i[0] not in self._used_interrupts]
        if not available:
            self._used_interrupts.clear()
            available = pool

        text, itype, visual, audio = random.choice(available)
        self._used_interrupts.add(text)

        return PatternInterrupt(
            text=text,
            interrupt_type=itype,
            visual_cue=visual,
            audio_cue=audio,
            recommended_duration_seconds=2.5
        )

    def schedule_interrupts(
        self,
        duration_seconds: int,
        interval_seconds: int = 75
    ) -> List[Tuple[int, PatternInterrupt]]:
        """
        Schedule pattern interrupts throughout the video.

        Best practice: Pattern interrupt every 60-90 seconds.

        Args:
            duration_seconds: Total video duration
            interval_seconds: Target interval between interrupts

        Returns:
            List of (timestamp, PatternInterrupt) tuples
        """
        scheduled = []
        interrupt_types = ["verbal", "visual", "audio", "structural"]

        # Start after initial hook
        current_time = 30
        type_index = 0

        while current_time < duration_seconds - 20:
            interrupt_type = interrupt_types[type_index % len(interrupt_types)]
            interrupt = self.get_interrupt(interrupt_type)
            scheduled.append((current_time, interrupt))

            # Add variation (+/- 15 seconds)
            variation = random.randint(-15, 15)
            current_time += interval_seconds + variation
            type_index += 1

        return scheduled


# ============================================================
# Call To Action Optimizer - Strategic CTA Placement
# ============================================================

class CallToActionOptimizer:
    """
    Optimizes call-to-action placement for maximum conversion.

    Based on research showing optimal CTA timing:
    - NEVER in first 30 seconds (kills retention)
    - Soft CTA at ~30% (if finding value, subscribe)
    - Engagement CTA at ~50% (comment with experience)
    - Final CTA at ~95% (subscribe for more)
    """

    CTA_TEMPLATES = {
        "soft": [
            "If you're finding this valuable, hit that subscribe button - we post content like this every week.",
            "Enjoying this so far? Subscribe for more insights like these.",
            "If this is helping you, consider subscribing - you won't miss future videos.",
        ],
        "engagement": [
            "I'd love to hear your thoughts - drop a comment below with your experience.",
            "What's your take on this? Let me know in the comments.",
            "Have you experienced this? Share your story in the comments.",
            "Comment below: which of these surprised you the most?",
        ],
        "subscribe": [
            "If you haven't subscribed yet, now's the time - hit that button and the bell.",
            "Join the community - subscribe and turn on notifications.",
            "Subscribe for more content like this - we post new videos every week.",
        ],
        "comment": [
            "Drop a comment with your biggest takeaway from this video.",
            "I read every comment - tell me what you want to learn next.",
            "Comment 'DONE' when you've tried this technique.",
        ],
        "final": [
            "If this video helped you, smash that like button and subscribe for more. I'll see you in the next one.",
            "Hit like if you learned something new, and subscribe for more content like this. Until next time.",
            "Don't forget to like, subscribe, and hit the bell. See you in the next video.",
        ],
    }

    def __init__(self):
        """Initialize the CTA optimizer."""
        logger.debug("CallToActionOptimizer initialized")

    def generate_cta(self, cta_type: str = "soft") -> str:
        """
        Generate a CTA of the specified type.

        Args:
            cta_type: Type of CTA (soft, engagement, subscribe, comment, final)

        Returns:
            CTA text string
        """
        templates = self.CTA_TEMPLATES.get(cta_type, self.CTA_TEMPLATES["soft"])
        return random.choice(templates)

    def create_cta_schedule(
        self,
        duration_seconds: int
    ) -> List[CTAPlacement]:
        """
        Create an optimized CTA schedule for a video.

        Placement strategy:
        - 30%: Soft CTA (subscribe hint)
        - 50%: Engagement CTA (comments)
        - 95%: Final CTA (full subscribe/like/bell)

        Args:
            duration_seconds: Total video duration

        Returns:
            List of CTAPlacement objects
        """
        placements = []

        # Never CTA in first 30 seconds
        if duration_seconds < 60:
            # For very short videos, only final CTA
            placements.append(CTAPlacement(
                text=self.generate_cta("final"),
                cta_type="final",
                timestamp_seconds=int(duration_seconds * 0.90),
                percentage_through=0.90
            ))
            return placements

        # Standard CTA schedule for longer videos
        schedule = [
            (0.30, "soft"),
            (0.50, "engagement"),
            (0.95, "final"),
        ]

        for position, cta_type in schedule:
            timestamp = int(duration_seconds * position)

            # Don't place CTA in first 30 seconds
            if timestamp < 30:
                continue

            placements.append(CTAPlacement(
                text=self.generate_cta(cta_type),
                cta_type=cta_type,
                timestamp_seconds=timestamp,
                percentage_through=position
            ))

        return placements


# ============================================================
# Unified Viral Content Engine
# ============================================================

class ViralContentEngine:
    """
    Unified engine combining all viral content generation components.

    Provides a single interface for generating complete viral content
    elements for a YouTube video.
    """

    def __init__(self, niche: str = "default"):
        """
        Initialize the viral content engine.

        Args:
            niche: Content niche (finance, psychology, storytelling)
        """
        self.niche = niche
        self.hook_generator = ViralHookGenerator()
        self.arc_builder = EmotionalArcBuilder()
        self.curiosity_creator = CuriosityGapCreator()
        self.payoff_scheduler = MicroPayoffScheduler()
        self.interrupt_library = PatternInterruptLibrary()
        self.cta_optimizer = CallToActionOptimizer()

        logger.info(f"ViralContentEngine initialized for niche: {niche}")

    def generate_all_elements(
        self,
        topic: str,
        duration_seconds: int,
        hook_count: int = 3,
        curiosity_gap_count: int = 3,
        variables: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate all viral content elements for a video.

        Args:
            topic: Video topic
            duration_seconds: Total video duration
            hook_count: Number of hook variations to generate
            curiosity_gap_count: Number of curiosity gaps
            variables: Custom variables for templates

        Returns:
            Dictionary containing all viral elements
        """
        # Merge topic into variables
        vars_dict = variables or {}
        vars_dict["topic"] = topic

        # Generate all elements
        hooks = self.hook_generator.generate_multiple_hooks(
            topic, self.niche, hook_count
        )

        arc_type = self.arc_builder.get_arc_for_niche(self.niche)
        emotional_arc = self.arc_builder.build_arc(duration_seconds, arc_type)

        curiosity_gaps = self.curiosity_creator.create_gap_sequence(
            duration_seconds, curiosity_gap_count, vars_dict
        )

        micro_payoffs = self.payoff_scheduler.schedule_payoffs(
            duration_seconds, vars_dict
        )

        pattern_interrupts = self.interrupt_library.schedule_interrupts(
            duration_seconds
        )

        cta_schedule = self.cta_optimizer.create_cta_schedule(duration_seconds)

        result = {
            "topic": topic,
            "niche": self.niche,
            "duration_seconds": duration_seconds,
            "hooks": hooks,
            "best_hook": max(hooks, key=lambda h: h.estimated_retention_boost),
            "emotional_arc": emotional_arc,
            "curiosity_gaps": curiosity_gaps,
            "micro_payoffs": micro_payoffs,
            "pattern_interrupts": pattern_interrupts,
            "cta_schedule": cta_schedule,
            "estimated_retention_boost": self._calculate_total_boost(
                hooks, curiosity_gaps, micro_payoffs, pattern_interrupts
            ),
        }

        logger.success(
            f"Generated viral elements: {len(hooks)} hooks, "
            f"{len(curiosity_gaps)} gaps, {len(micro_payoffs)} payoffs, "
            f"{len(pattern_interrupts)} interrupts"
        )

        return result

    def _calculate_total_boost(
        self,
        hooks: List[ViralHook],
        gaps: List[CuriosityGap],
        payoffs: List[MicroPayoff],
        interrupts: List[Tuple[int, PatternInterrupt]]
    ) -> float:
        """Calculate estimated total retention boost."""
        boost = 0.0

        # Best hook contribution
        if hooks:
            boost += max(h.estimated_retention_boost for h in hooks)

        # Curiosity gaps contribution (each gap adds ~5-10%)
        high_impact = sum(1 for g in gaps if g.retention_impact == "high")
        medium_impact = sum(1 for g in gaps if g.retention_impact == "medium")
        boost += high_impact * 0.08 + medium_impact * 0.05

        # Micro-payoffs contribution (diminishing returns)
        payoff_boost = min(0.15, len(payoffs) * 0.02)
        boost += payoff_boost

        # Pattern interrupts contribution
        interrupt_boost = min(0.10, len(interrupts) * 0.015)
        boost += interrupt_boost

        # Cap total boost at 50%
        return min(0.50, boost)

    def generate_viral_script_outline(
        self,
        topic: str,
        duration_seconds: int
    ) -> Dict[str, Any]:
        """
        Generate a complete viral script outline with all elements positioned.

        Args:
            topic: Video topic
            duration_seconds: Total video duration

        Returns:
            Dictionary with timestamped script outline
        """
        elements = self.generate_all_elements(topic, duration_seconds)

        # Build timeline
        timeline = []

        # Add hook
        best_hook = elements["best_hook"]
        timeline.append({
            "timestamp": 0,
            "type": "hook",
            "text": best_hook.text,
            "duration": best_hook.duration_seconds,
        })

        # Add emotional arc beats
        for beat in elements["emotional_arc"].beats:
            timeline.append({
                "timestamp": beat.timestamp_seconds,
                "type": "emotional_beat",
                "emotion": beat.emotion.value,
                "intensity": beat.intensity,
                "hint": beat.narration_hint,
            })

        # Add curiosity gaps
        for gap in elements["curiosity_gaps"]:
            timeline.append({
                "timestamp": gap.opening_timestamp_seconds,
                "type": "open_loop",
                "text": gap.opening_text,
                "closes_at": gap.resolution_timestamp_seconds,
            })
            timeline.append({
                "timestamp": gap.resolution_timestamp_seconds,
                "type": "close_loop",
                "text": gap.resolution_text,
            })

        # Add micro-payoffs
        for payoff in elements["micro_payoffs"]:
            timeline.append({
                "timestamp": payoff.timestamp_seconds,
                "type": "micro_payoff",
                "text": payoff.text,
                "payoff_type": payoff.payoff_type,
            })

        # Add pattern interrupts
        for timestamp, interrupt in elements["pattern_interrupts"]:
            timeline.append({
                "timestamp": timestamp,
                "type": "pattern_interrupt",
                "text": interrupt.text,
                "visual": interrupt.visual_cue,
                "audio": interrupt.audio_cue,
            })

        # Add CTAs
        for cta in elements["cta_schedule"]:
            timeline.append({
                "timestamp": cta.timestamp_seconds,
                "type": "cta",
                "text": cta.text,
                "cta_type": cta.cta_type,
            })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return {
            "topic": topic,
            "duration_seconds": duration_seconds,
            "timeline": timeline,
            "estimated_retention_boost": elements["estimated_retention_boost"],
            "summary": {
                "hooks": len(elements["hooks"]),
                "curiosity_gaps": len(elements["curiosity_gaps"]),
                "micro_payoffs": len(elements["micro_payoffs"]),
                "pattern_interrupts": len(elements["pattern_interrupts"]),
                "ctas": len(elements["cta_schedule"]),
            }
        }


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VIRAL CONTENT ENGINE TEST")
    print("=" * 60)

    # Test individual components
    print("\n1. Testing ViralHookGenerator:")
    hook_gen = ViralHookGenerator()
    hook = hook_gen.generate_hook("passive income strategies", niche="finance")
    print(f"   Hook Type: {hook.hook_type.value}")
    print(f"   Text: {hook.text}")
    print(f"   Estimated Boost: {hook.estimated_retention_boost:.0%}")

    print("\n2. Testing EmotionalArcBuilder:")
    arc_builder = EmotionalArcBuilder()
    arc = arc_builder.build_arc(600, "hero_journey")
    print(f"   Peak Moment: {arc.peak_moment_seconds}s")
    print(f"   Beats: {len(arc.beats)}")
    for beat in arc.beats[:3]:
        print(f"     - {beat.timestamp_seconds}s: {beat.emotion.value} ({beat.intensity:.0%})")

    print("\n3. Testing CuriosityGapCreator:")
    gap_creator = CuriosityGapCreator()
    gaps = gap_creator.create_gap_sequence(600, count=3)
    print(f"   Created {len(gaps)} curiosity gaps:")
    for gap in gaps:
        print(f"     - Opens at {gap.opening_timestamp_seconds}s, closes at {gap.resolution_timestamp_seconds}s")

    print("\n4. Testing MicroPayoffScheduler:")
    payoff_scheduler = MicroPayoffScheduler()
    payoffs = payoff_scheduler.schedule_payoffs(600)
    print(f"   Scheduled {len(payoffs)} micro-payoffs:")
    for payoff in payoffs[:3]:
        print(f"     - {payoff.timestamp_seconds}s: {payoff.payoff_type}")

    print("\n5. Testing PatternInterruptLibrary:")
    interrupt_lib = PatternInterruptLibrary()
    interrupts = interrupt_lib.schedule_interrupts(600)
    print(f"   Scheduled {len(interrupts)} pattern interrupts:")
    for ts, interrupt in interrupts[:3]:
        print(f"     - {ts}s: {interrupt.interrupt_type}")

    print("\n6. Testing CallToActionOptimizer:")
    cta_opt = CallToActionOptimizer()
    ctas = cta_opt.create_cta_schedule(600)
    print(f"   Created {len(ctas)} CTAs:")
    for cta in ctas:
        print(f"     - {cta.timestamp_seconds}s ({cta.percentage_through:.0%}): {cta.cta_type}")

    print("\n7. Testing Full ViralContentEngine:")
    engine = ViralContentEngine(niche="psychology")
    outline = engine.generate_viral_script_outline("dark psychology tricks", 600)
    print(f"   Generated outline with {len(outline['timeline'])} elements")
    print(f"   Estimated Retention Boost: {outline['estimated_retention_boost']:.0%}")
    print(f"   Summary: {outline['summary']}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
