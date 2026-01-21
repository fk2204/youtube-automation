"""
YouTube Algorithm Optimization Utilities

Advanced optimization tools for maximizing YouTube video performance:
- UploadTimingOptimizer: Calculate optimal upload times based on target audience regions
- FirstHourBooster: Schedule post-upload engagement actions for first-hour algorithm boost
- TitlePatternAnalyzer: Analyze and score titles against proven viral patterns

Usage:
    from src.utils.youtube_optimizer import (
        UploadTimingOptimizer,
        FirstHourBooster,
        TitlePatternAnalyzer,
        generate_chapters_from_script,
        optimize_description_keywords,
    )

    # Calculate optimal upload time
    timing = UploadTimingOptimizer()
    optimal_time = timing.calculate_optimal_time("finance", ["US_EST", "UK"])

    # Schedule first-hour engagement
    booster = FirstHourBooster()
    actions = booster.schedule_post_upload_actions("video_id_123")

    # Analyze title patterns
    analyzer = TitlePatternAnalyzer()
    result = analyzer.analyze_title("5 Money Mistakes Costing You $1000/Year", "finance")
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class OptimalUploadTime:
    """Result of optimal upload time calculation."""
    recommended_datetime: datetime
    target_regions: List[str]
    peak_hours_utc: List[int]
    confidence_score: float  # 0-1, how confident we are in this time
    reasoning: str
    alternative_times: List[datetime] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Upload at {self.recommended_datetime.strftime('%Y-%m-%d %H:%M %Z')} "
            f"(confidence: {self.confidence_score:.0%})"
        )


@dataclass
class EngagementAction:
    """A scheduled post-upload engagement action."""
    delay_seconds: int
    action_type: str
    description: str
    api_endpoint: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: str = "high"  # high, medium, low

    def __str__(self) -> str:
        return f"[{self.delay_seconds}s] {self.action_type}: {self.description}"


@dataclass
class TitleAnalysisResult:
    """Result of title pattern analysis."""
    title: str
    score: float  # 0-100
    has_number: bool
    has_power_word: bool
    has_question: bool
    has_brackets: bool
    character_count: int
    pattern_matches: List[str]
    suggestions: List[str]
    viral_potential: str  # "high", "medium", "low"

    def __str__(self) -> str:
        return f"Title Score: {self.score:.0f}/100 ({self.viral_potential} viral potential)"


@dataclass
class KeywordDensityResult:
    """Result of keyword density optimization."""
    original_text: str
    optimized_text: str
    original_density: float
    optimized_density: float
    keywords_added: List[str]
    target_density: float

    def __str__(self) -> str:
        return (
            f"Keyword density: {self.original_density:.1%} -> {self.optimized_density:.1%} "
            f"(target: {self.target_density:.1%})"
        )


# ============================================================
# UPLOAD TIMING OPTIMIZER
# ============================================================

class UploadTimingOptimizer:
    """
    Calculate optimal upload times based on target audience regions.

    YouTube's algorithm favors videos that get strong engagement in the first
    hour after upload. This class helps identify the best times to upload
    based on when your target audience is most active.

    Peak hours are typically:
    - Weekdays: 2-4 PM local time (post-work/school)
    - Weekends: 9-11 AM local time (morning leisure)
    - Evening: 7-9 PM local time (prime time)

    Usage:
        optimizer = UploadTimingOptimizer()
        result = optimizer.calculate_optimal_time("finance", ["US_EST", "US_PST"])
        print(f"Best time to upload: {result.recommended_datetime}")
    """

    # Peak viewing hours by region (UTC offset and peak hours in local time)
    PEAK_HOURS_BY_REGION = {
        "US_EST": {"hours": [14, 15, 19, 20, 21], "offset": -5, "name": "US Eastern"},
        "US_CST": {"hours": [14, 15, 19, 20, 21], "offset": -6, "name": "US Central"},
        "US_MST": {"hours": [14, 15, 19, 20, 21], "offset": -7, "name": "US Mountain"},
        "US_PST": {"hours": [14, 15, 18, 19, 20], "offset": -8, "name": "US Pacific"},
        "UK": {"hours": [12, 13, 19, 20, 21], "offset": 0, "name": "United Kingdom"},
        "EU_CET": {"hours": [12, 13, 19, 20, 21], "offset": 1, "name": "Central Europe"},
        "INDIA": {"hours": [18, 19, 20, 21, 22], "offset": 5.5, "name": "India"},
        "AUSTRALIA": {"hours": [18, 19, 20, 21], "offset": 10, "name": "Australia"},
        "JAPAN": {"hours": [19, 20, 21, 22], "offset": 9, "name": "Japan"},
        "BRAZIL": {"hours": [19, 20, 21, 22], "offset": -3, "name": "Brazil"},
    }

    # Niche-specific optimal times (adjustments)
    NICHE_ADJUSTMENTS = {
        "finance": {
            "best_days": [0, 1, 2, 3, 4],  # Monday-Friday (weekdays better for finance)
            "avoid_hours": [9, 10, 11],  # Avoid market hours
            "preference": "morning",  # Earlier in day preferred
        },
        "psychology": {
            "best_days": [1, 2, 3, 5, 6],  # Tue, Wed, Thu, Sat, Sun
            "avoid_hours": [],
            "preference": "evening",  # Evening content performs better
        },
        "storytelling": {
            "best_days": [4, 5, 6],  # Fri, Sat, Sun (weekend binge watching)
            "avoid_hours": [],
            "preference": "evening",  # Prime time viewing
        },
        "gaming": {
            "best_days": [4, 5, 6],  # Weekend gaming
            "avoid_hours": [6, 7, 8],  # Too early
            "preference": "afternoon",
        },
        "education": {
            "best_days": [0, 1, 2, 3, 6],  # Weekdays + Sunday
            "avoid_hours": [],
            "preference": "afternoon",
        },
        "default": {
            "best_days": [1, 2, 3, 4, 5],  # Tue-Sat
            "avoid_hours": [],
            "preference": "evening",
        },
    }

    def __init__(self):
        """Initialize the upload timing optimizer."""
        logger.debug("UploadTimingOptimizer initialized")

    def calculate_optimal_time(
        self,
        niche: str = "default",
        target_regions: Optional[List[str]] = None,
        days_ahead: int = 7,
        avoid_hours_utc: Optional[List[int]] = None
    ) -> OptimalUploadTime:
        """
        Calculate the optimal upload time for a video.

        Args:
            niche: Content niche (finance, psychology, storytelling, etc.)
            target_regions: List of region codes to target (US_EST, UK, etc.)
            days_ahead: Maximum days ahead to search for optimal time
            avoid_hours_utc: Hours to avoid in UTC (e.g., [2, 3, 4] to avoid night)

        Returns:
            OptimalUploadTime with recommended datetime and reasoning
        """
        logger.info(f"Calculating optimal upload time for niche: {niche}")

        # Default to US regions if none specified
        if not target_regions:
            target_regions = ["US_EST", "US_PST"]

        # Validate regions
        valid_regions = [r for r in target_regions if r in self.PEAK_HOURS_BY_REGION]
        if not valid_regions:
            logger.warning(f"No valid regions provided, using defaults")
            valid_regions = ["US_EST", "US_PST"]

        # Get niche adjustments
        niche_config = self.NICHE_ADJUSTMENTS.get(niche, self.NICHE_ADJUSTMENTS["default"])
        best_days = niche_config["best_days"]
        avoid_hours = set(niche_config.get("avoid_hours", []))
        if avoid_hours_utc:
            avoid_hours.update(avoid_hours_utc)

        # Calculate overlapping peak hours in UTC
        peak_hours_utc = self._find_overlapping_peak_hours(valid_regions)

        # Filter out avoided hours
        peak_hours_utc = [h for h in peak_hours_utc if h not in avoid_hours]

        if not peak_hours_utc:
            # Fallback to general evening hours
            peak_hours_utc = [19, 20, 21]

        # Find the next optimal datetime
        now = datetime.now(timezone.utc)
        optimal_dt = None
        alternative_times = []

        for day_offset in range(days_ahead):
            candidate_date = now + timedelta(days=day_offset)

            # Check if this day is in best_days
            if candidate_date.weekday() not in best_days:
                continue

            for hour in sorted(peak_hours_utc):
                candidate_dt = candidate_date.replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )

                # Skip if in the past
                if candidate_dt <= now:
                    continue

                if optimal_dt is None:
                    optimal_dt = candidate_dt
                elif len(alternative_times) < 3:
                    alternative_times.append(candidate_dt)

        if optimal_dt is None:
            # Fallback to tomorrow at peak hour
            optimal_dt = (now + timedelta(days=1)).replace(
                hour=peak_hours_utc[0] if peak_hours_utc else 19,
                minute=0, second=0, microsecond=0
            )

        # Calculate confidence score
        confidence = self._calculate_confidence(
            optimal_dt, valid_regions, best_days, peak_hours_utc
        )

        # Generate reasoning
        region_names = [self.PEAK_HOURS_BY_REGION[r]["name"] for r in valid_regions]
        reasoning = (
            f"Optimized for {', '.join(region_names)} audiences. "
            f"Day of week ({optimal_dt.strftime('%A')}) is optimal for {niche} content. "
            f"Hour ({optimal_dt.hour}:00 UTC) overlaps peak viewing times."
        )

        result = OptimalUploadTime(
            recommended_datetime=optimal_dt,
            target_regions=valid_regions,
            peak_hours_utc=peak_hours_utc,
            confidence_score=confidence,
            reasoning=reasoning,
            alternative_times=alternative_times
        )

        logger.success(f"Optimal upload time: {result}")
        return result

    def _find_overlapping_peak_hours(self, regions: List[str]) -> List[int]:
        """Find hours that are peak time in multiple regions."""
        hour_scores = {}

        for region in regions:
            config = self.PEAK_HOURS_BY_REGION.get(region)
            if not config:
                continue

            offset = config["offset"]
            peak_hours = config["hours"]

            for local_hour in peak_hours:
                # Convert local hour to UTC
                utc_hour = (local_hour - int(offset)) % 24
                hour_scores[utc_hour] = hour_scores.get(utc_hour, 0) + 1

        # Sort by score (most regions at peak) then by hour
        sorted_hours = sorted(
            hour_scores.items(),
            key=lambda x: (-x[1], x[0])
        )

        return [h for h, _ in sorted_hours[:5]]

    def _calculate_confidence(
        self,
        dt: datetime,
        regions: List[str],
        best_days: List[int],
        peak_hours: List[int]
    ) -> float:
        """Calculate confidence score for the recommended time."""
        score = 0.5  # Base score

        # Bonus for day of week
        if dt.weekday() in best_days:
            score += 0.2

        # Bonus for hour matching peak
        if dt.hour in peak_hours:
            score += 0.2

        # Bonus for multiple regions
        if len(regions) >= 2:
            score += 0.1

        return min(0.95, score)

    def get_region_info(self, region: str) -> Optional[Dict[str, Any]]:
        """Get information about a region."""
        return self.PEAK_HOURS_BY_REGION.get(region)

    def list_available_regions(self) -> List[str]:
        """List all available region codes."""
        return list(self.PEAK_HOURS_BY_REGION.keys())


# ============================================================
# FIRST HOUR BOOSTER
# ============================================================

class FirstHourBooster:
    """
    Schedule post-upload engagement actions for the critical first hour.

    YouTube's algorithm heavily weights engagement in the first 60 minutes
    after upload. This class schedules strategic actions to maximize
    early engagement signals.

    Key actions:
    1. Add to relevant playlists (immediate)
    2. Post community announcement (30 seconds)
    3. Pin an engagement comment (2 minutes)
    4. Share to social media (5 minutes)
    5. Reply to early comments (15-30 minutes)

    Usage:
        booster = FirstHourBooster()
        actions = booster.schedule_post_upload_actions("abc123xyz")
        for action in actions:
            print(f"{action.delay_seconds}s: {action.description}")
    """

    # Default engagement actions with timing
    DEFAULT_ACTIONS = [
        {
            "delay": 0,
            "action": "add_to_playlist",
            "description": "Add video to relevant playlist(s)",
            "priority": "high",
        },
        {
            "delay": 30,
            "action": "community_post",
            "description": "Create community post announcing the video",
            "priority": "high",
        },
        {
            "delay": 120,
            "action": "pin_engagement_comment",
            "description": "Pin a comment asking viewers to engage",
            "priority": "high",
        },
        {
            "delay": 300,
            "action": "social_share",
            "description": "Share video link on social media platforms",
            "priority": "medium",
        },
        {
            "delay": 600,
            "action": "end_screen_check",
            "description": "Verify end screens and cards are working",
            "priority": "medium",
        },
        {
            "delay": 900,
            "action": "reply_to_comments",
            "description": "Reply to early comments to boost engagement",
            "priority": "high",
        },
        {
            "delay": 1800,
            "action": "analytics_check",
            "description": "Check real-time analytics for early performance",
            "priority": "low",
        },
        {
            "delay": 3600,
            "action": "first_hour_report",
            "description": "Generate first hour performance report",
            "priority": "medium",
        },
    ]

    # Engagement comment templates by niche
    ENGAGEMENT_COMMENTS = {
        "finance": [
            "What's YOUR biggest money goal for this year? Drop it in the comments!",
            "Which tip from this video will you try first? Let me know below!",
            "Comment 'INVEST' if you want more content like this!",
        ],
        "psychology": [
            "Which psychological trick surprised you the most? Tell me below!",
            "Have you noticed any of these patterns in your own life? Share your experience!",
            "Comment 'MIND' if you want more psychology content!",
        ],
        "storytelling": [
            "What part of this story shocked you the most? Comment below!",
            "Did you know this story before watching? Let me know!",
            "Comment 'MORE' if you want more stories like this!",
        ],
        "default": [
            "What did you find most valuable in this video? Let me know in the comments!",
            "Which tip will you try first? Comment below!",
            "Hit LIKE if this helped you and comment what you want to see next!",
        ],
    }

    # Community post templates
    COMMUNITY_POST_TEMPLATES = {
        "finance": "NEW VIDEO: {title}\n\nLearn how to {hook}. Link in bio!",
        "psychology": "NEW VIDEO: {title}\n\nDiscover {hook}. Watch now!",
        "storytelling": "NEW VIDEO: {title}\n\nThe untold story of {hook}. Don't miss it!",
        "default": "NEW VIDEO: {title}\n\n{hook}. Watch now!",
    }

    def __init__(self):
        """Initialize the first hour booster."""
        logger.debug("FirstHourBooster initialized")

    def schedule_post_upload_actions(
        self,
        video_id: str,
        niche: str = "default",
        playlist_ids: Optional[List[str]] = None,
        custom_comment: Optional[str] = None,
        include_social: bool = True
    ) -> List[EngagementAction]:
        """
        Schedule all post-upload engagement actions.

        Args:
            video_id: The YouTube video ID
            niche: Content niche for customized messages
            playlist_ids: List of playlist IDs to add the video to
            custom_comment: Custom engagement comment (overrides template)
            include_social: Whether to include social media actions

        Returns:
            List of EngagementAction objects with timing and details
        """
        logger.info(f"Scheduling post-upload actions for video: {video_id}")

        actions = []

        for action_config in self.DEFAULT_ACTIONS:
            # Skip social actions if not requested
            if not include_social and action_config["action"] == "social_share":
                continue

            action = EngagementAction(
                delay_seconds=action_config["delay"],
                action_type=action_config["action"],
                description=action_config["description"],
                priority=action_config["priority"],
                parameters={"video_id": video_id}
            )

            # Add action-specific parameters
            if action.action_type == "add_to_playlist" and playlist_ids:
                action.parameters["playlist_ids"] = playlist_ids

            elif action.action_type == "pin_engagement_comment":
                comment = custom_comment or self._get_engagement_comment(niche)
                action.parameters["comment_text"] = comment

            elif action.action_type == "community_post":
                action.parameters["niche"] = niche

            actions.append(action)

        # Sort by delay
        actions.sort(key=lambda a: a.delay_seconds)

        logger.success(f"Scheduled {len(actions)} post-upload actions")
        return actions

    def _get_engagement_comment(self, niche: str) -> str:
        """Get an engagement comment for the given niche."""
        import random
        comments = self.ENGAGEMENT_COMMENTS.get(niche, self.ENGAGEMENT_COMMENTS["default"])
        return random.choice(comments)

    def get_community_post_template(self, niche: str) -> str:
        """Get a community post template for the given niche."""
        return self.COMMUNITY_POST_TEMPLATES.get(
            niche, self.COMMUNITY_POST_TEMPLATES["default"]
        )

    def get_action_timeline(self, actions: List[EngagementAction]) -> str:
        """Generate a human-readable timeline of actions."""
        lines = ["First Hour Engagement Timeline:", "=" * 40]

        for action in sorted(actions, key=lambda a: a.delay_seconds):
            if action.delay_seconds == 0:
                time_str = "Immediately"
            elif action.delay_seconds < 60:
                time_str = f"+{action.delay_seconds}s"
            else:
                minutes = action.delay_seconds // 60
                time_str = f"+{minutes}m"

            priority_marker = {
                "high": "[!]",
                "medium": "[*]",
                "low": "[ ]"
            }.get(action.priority, "[ ]")

            lines.append(f"{time_str:>12} {priority_marker} {action.description}")

        return "\n".join(lines)

    def execute_action(
        self,
        action: EngagementAction,
        youtube_service: Any = None
    ) -> bool:
        """
        Execute a single engagement action.

        Note: This requires a YouTube API service object for actual execution.
        Without it, this method just logs what would happen.

        Args:
            action: The EngagementAction to execute
            youtube_service: Authenticated YouTube API service object

        Returns:
            True if action was executed (or logged) successfully
        """
        logger.info(f"Executing action: {action.action_type}")

        if youtube_service is None:
            logger.warning(f"No YouTube service - would execute: {action.description}")
            return True

        video_id = action.parameters.get("video_id")

        try:
            if action.action_type == "add_to_playlist":
                playlist_ids = action.parameters.get("playlist_ids", [])
                for playlist_id in playlist_ids:
                    youtube_service.playlistItems().insert(
                        part="snippet",
                        body={
                            "snippet": {
                                "playlistId": playlist_id,
                                "resourceId": {
                                    "kind": "youtube#video",
                                    "videoId": video_id
                                }
                            }
                        }
                    ).execute()
                    logger.success(f"Added video to playlist: {playlist_id}")

            elif action.action_type == "pin_engagement_comment":
                comment_text = action.parameters.get("comment_text", "")
                # Insert comment
                response = youtube_service.commentThreads().insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "videoId": video_id,
                            "topLevelComment": {
                                "snippet": {
                                    "textOriginal": comment_text
                                }
                            }
                        }
                    }
                ).execute()
                comment_id = response["id"]
                logger.success(f"Posted engagement comment: {comment_id}")
                # Note: Pinning requires additional API call

            elif action.action_type == "community_post":
                # Community posts require different API access
                logger.info("Community post scheduled (requires YouTube Studio)")

            else:
                logger.info(f"Action '{action.action_type}' logged for manual execution")

            return True

        except Exception as e:
            logger.error(f"Failed to execute action {action.action_type}: {e}")
            return False


# ============================================================
# TITLE PATTERN ANALYZER
# ============================================================

class TitlePatternAnalyzer:
    """
    Analyze and score titles against proven viral patterns.

    Research shows that certain title patterns consistently achieve
    higher CTR (click-through rates) on YouTube:
    - Numbers (especially odd numbers like 7, 9)
    - Power words (shocking, secret, ultimate, etc.)
    - Questions
    - Brackets with bonus info
    - Emotional triggers

    Usage:
        analyzer = TitlePatternAnalyzer()
        result = analyzer.analyze_title("5 Money Mistakes Costing You $1000", "finance")
        print(f"Score: {result.score}/100")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")
    """

    # Power words that increase CTR
    POWER_WORDS = {
        "authority": [
            "ultimate", "proven", "expert", "complete", "definitive",
            "essential", "comprehensive", "official", "professional"
        ],
        "curiosity": [
            "secret", "hidden", "shocking", "unbelievable", "surprising",
            "actually", "finally", "revealed", "truth", "nobody",
            "untold", "dark", "mysterious", "unknown"
        ],
        "urgency": [
            "critical", "instant", "guaranteed", "revolutionary",
            "breakthrough", "urgent", "limited", "exclusive", "now"
        ],
        "emotional": [
            "powerful", "incredible", "amazing", "massive", "terrifying",
            "genius", "brilliant", "insane", "mind-blowing", "life-changing"
        ],
        "negative": [
            "mistake", "wrong", "fail", "never", "stop", "avoid",
            "worst", "bad", "dangerous", "warning"
        ],
    }

    # Viral title patterns by niche
    VIRAL_PATTERNS = {
        "finance": [
            r"\d+\s*(passive income|money|wealth|investment)",
            r"how .+ makes? money",
            r"\$[\d,]+",
            r"(save|earn|make)\s*\$?[\d,]+",
            r"(millionaire|rich|wealthy)",
            r"(mistake|secret|truth).*(money|wealth|invest)",
        ],
        "psychology": [
            r"\d+\s*signs?\s*(of|that|you)",
            r"(dark|secret|hidden)\s*psychology",
            r"(brain|mind)\s*(tricks?|hacks?)",
            r"why\s+(you|your|people)",
            r"(manipulation|persuasion|influence)",
            r"(narcissist|psychopath|introvert|empath)",
        ],
        "storytelling": [
            r"(untold|true|real)\s*story",
            r"(rise|fall)\s*(of|and)",
            r"what\s+(happened|really)",
            r"how .+ (became|went|changed)",
            r"(dark|secret|hidden)\s*side",
            r"(mystery|scandal|controversy)",
        ],
        "default": [
            r"\d+\s*(ways?|tips?|tricks?|steps?|things?)",
            r"how\s+to",
            r"why\s+(you|your|most)",
            r"(best|worst|top)\s+\d+",
            r"(complete|ultimate|definitive)\s+guide",
        ],
    }

    # Question words that increase engagement
    QUESTION_STARTERS = ["how", "why", "what", "when", "where", "who", "which", "can", "does", "is", "are", "will"]

    def __init__(self):
        """Initialize the title pattern analyzer."""
        logger.debug("TitlePatternAnalyzer initialized")

    def analyze_title(self, title: str, niche: str = "default") -> TitleAnalysisResult:
        """
        Analyze a title and score it against viral patterns.

        Args:
            title: The title to analyze
            niche: Content niche for pattern matching

        Returns:
            TitleAnalysisResult with score, matches, and suggestions
        """
        logger.debug(f"Analyzing title: {title}")

        score = 50.0  # Base score
        pattern_matches = []
        suggestions = []

        title_lower = title.lower()
        char_count = len(title)

        # Check for numbers
        has_number = bool(re.search(r'\d', title))
        if has_number:
            score += 10
            pattern_matches.append("Contains number")
            # Odd numbers perform better
            numbers = re.findall(r'\d+', title)
            for num in numbers:
                if int(num) % 2 == 1 and int(num) < 20:
                    score += 5
                    pattern_matches.append(f"Uses odd number ({num})")
                    break
        else:
            suggestions.append("Add a specific number (odd numbers like 5, 7, 9 work best)")

        # Check for power words
        has_power_word = False
        for category, words in self.POWER_WORDS.items():
            for word in words:
                if word in title_lower:
                    has_power_word = True
                    score += 8
                    pattern_matches.append(f"Power word: '{word}' ({category})")
                    break
            if has_power_word:
                break

        if not has_power_word:
            suggestions.append("Add a power word (e.g., 'secret', 'shocking', 'ultimate')")

        # Check for question format
        first_word = title_lower.split()[0] if title_lower.split() else ""
        has_question = first_word in self.QUESTION_STARTERS or title.endswith("?")
        if has_question:
            score += 8
            pattern_matches.append("Question format")

        # Check for brackets (bonus info pattern)
        has_brackets = bool(re.search(r'[\[\(].+[\]\)]', title))
        if has_brackets:
            score += 7
            pattern_matches.append("Brackets with bonus info")
        else:
            if char_count < 50:
                suggestions.append("Add brackets with bonus info [2024] or (Step by Step)")

        # Check title length (optimal: 40-60 characters)
        if 40 <= char_count <= 60:
            score += 10
            pattern_matches.append(f"Optimal length ({char_count} chars)")
        elif char_count < 40:
            suggestions.append(f"Title is short ({char_count} chars). Optimal is 40-60.")
        else:
            suggestions.append(f"Title is long ({char_count} chars). Consider shortening to 60 chars max.")

        # Check for niche-specific viral patterns
        niche_patterns = self.VIRAL_PATTERNS.get(niche, self.VIRAL_PATTERNS["default"])
        for pattern in niche_patterns:
            if re.search(pattern, title_lower):
                score += 5
                pattern_matches.append(f"Matches niche pattern: {pattern[:30]}...")

        # Check for dollar amounts (high CTR in finance)
        if niche == "finance" and re.search(r'\$[\d,]+', title):
            score += 10
            pattern_matches.append("Contains dollar amount")

        # Check for emotional triggers
        emotional_patterns = [
            (r"you('re|r)?\s+(wrong|right|missing)", "Direct address"),
            (r"(never|always|every)\s+", "Absolute language"),
            (r"(mistake|fail|wrong)", "Loss aversion trigger"),
        ]
        for pattern, label in emotional_patterns:
            if re.search(pattern, title_lower):
                score += 5
                pattern_matches.append(label)

        # Cap score at 100
        score = min(100, score)

        # Determine viral potential
        if score >= 80:
            viral_potential = "high"
        elif score >= 60:
            viral_potential = "medium"
        else:
            viral_potential = "low"

        result = TitleAnalysisResult(
            title=title,
            score=score,
            has_number=has_number,
            has_power_word=has_power_word,
            has_question=has_question,
            has_brackets=has_brackets,
            character_count=char_count,
            pattern_matches=pattern_matches,
            suggestions=suggestions,
            viral_potential=viral_potential
        )

        logger.info(f"Title analysis: {result}")
        return result

    def suggest_improvements(self, title: str, niche: str = "default") -> List[str]:
        """Get improvement suggestions for a title."""
        result = self.analyze_title(title, niche)
        return result.suggestions

    def generate_title_variations(
        self,
        topic: str,
        niche: str = "default",
        count: int = 5
    ) -> List[str]:
        """
        Generate title variations based on viral patterns.

        Args:
            topic: The main topic
            niche: Content niche
            count: Number of variations to generate

        Returns:
            List of title variations
        """
        import random

        templates = {
            "finance": [
                f"{random.randint(3, 9)} {topic} Secrets That Made Me ${{amount}}",
                f"The Hidden Truth About {topic} Nobody Tells You",
                f"How {topic} Can Make You ${random.randint(1, 9) * 1000}/Month",
                f"Why {random.randint(80, 99)}% of People Fail at {topic}",
                f"{topic}: The Complete Guide [{datetime.now().year}]",
                f"I Tried {topic} for 30 Days - Here's What Happened",
                f"The ${random.randint(1, 9) * 100} {topic} Strategy Nobody Knows",
            ],
            "psychology": [
                f"{random.randint(5, 9)} Signs of {topic} You're Ignoring",
                f"Dark Psychology: How {topic} Controls Your Mind",
                f"The Science Behind {topic} (You Won't Believe This)",
                f"Why Your Brain Makes You {topic}",
                f"{random.randint(5, 9)} {topic} Tricks That Work on Anyone",
                f"The {topic} That's Ruining Your Life",
                f"What Your {topic} Says About You",
            ],
            "storytelling": [
                f"The Untold Story of {topic}",
                f"How {topic} Changed Everything",
                f"What Really Happened to {topic}",
                f"The Rise and Fall of {topic}",
                f"The Dark Side of {topic} Nobody Talks About",
                f"Inside {topic}: The True Story",
                f"Why Everyone Was Wrong About {topic}",
            ],
            "default": [
                f"{random.randint(5, 9)} {topic} Tips You Need to Know",
                f"The Ultimate {topic} Guide [{datetime.now().year}]",
                f"How to Master {topic} (Step by Step)",
                f"Why {topic} Is More Important Than You Think",
                f"{topic}: Everything You Need to Know",
                f"The Secret to {topic} Nobody Tells You",
            ],
        }

        niche_templates = templates.get(niche, templates["default"])
        random.shuffle(niche_templates)

        variations = []
        for template in niche_templates[:count]:
            # Fill in random amounts if present
            title = template.replace("{amount}", f"{random.randint(1, 9) * 10000:,}")
            variations.append(title)

        return variations


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_chapters_from_script(
    script: Dict[str, Any],
    duration_seconds: int
) -> str:
    """
    Auto-generate YouTube chapters from script sections.

    YouTube chapters require:
    - First timestamp at 0:00
    - At least 3 chapters
    - Each chapter at least 10 seconds

    Args:
        script: Script dictionary with 'sections' key containing list of sections
        duration_seconds: Total video duration in seconds

    Returns:
        Formatted chapters string for video description

    Example:
        script = {
            "sections": [
                {"heading": "Introduction", "content": "..."},
                {"heading": "Main Point 1", "content": "..."},
                {"heading": "Conclusion", "content": "..."},
            ]
        }
        chapters = generate_chapters_from_script(script, 600)
        # Returns:
        # "CHAPTERS:
        # 0:00 Introduction
        # 2:00 Main Point 1
        # 8:00 Conclusion"
    """
    chapters = ["0:00 Introduction"]
    sections = script.get("sections", [])

    if not sections:
        logger.warning("No sections found in script")
        return "\n\nCHAPTERS:\n0:00 Introduction"

    # Calculate duration per section
    section_count = len(sections)
    section_duration = duration_seconds / max(1, section_count)

    for i, section in enumerate(sections):
        # Calculate timestamp
        timestamp_seconds = int((i + 1) * section_duration)

        # Skip if too close to start (< 10 seconds)
        if timestamp_seconds < 10:
            continue

        # Format timestamp
        minutes = timestamp_seconds // 60
        secs = timestamp_seconds % 60
        timestamp = f"{minutes}:{secs:02d}"

        # Get heading, truncate to 60 chars
        heading = section.get("heading", section.get("title", f"Part {i + 1}"))
        heading = str(heading)[:60]

        chapters.append(f"{timestamp} {heading}")

    # Ensure we have at least 3 chapters
    if len(chapters) < 3:
        # Add mid-point chapter
        mid_seconds = duration_seconds // 2
        mid_timestamp = f"{mid_seconds // 60}:{mid_seconds % 60:02d}"
        chapters.insert(1, f"{mid_timestamp} Main Content")

    return "\n\nCHAPTERS:\n" + "\n".join(chapters)


def optimize_description_keywords(
    description: str,
    target_keywords: List[str],
    target_density: float = 0.025
) -> KeywordDensityResult:
    """
    Ensure optimal keyword density (2-3%) in video descriptions.

    YouTube's algorithm uses descriptions for understanding video content.
    A keyword density of 2-3% is optimal - enough for SEO without
    appearing spammy.

    Args:
        description: Original video description
        target_keywords: List of keywords to optimize for
        target_density: Target keyword density (default 2.5%)

    Returns:
        KeywordDensityResult with original and optimized text

    Example:
        result = optimize_description_keywords(
            "Learn about investing in this video.",
            ["investing", "money", "wealth"],
            target_density=0.025
        )
        print(result.optimized_text)
    """
    if not description or not target_keywords:
        return KeywordDensityResult(
            original_text=description,
            optimized_text=description,
            original_density=0.0,
            optimized_density=0.0,
            keywords_added=[],
            target_density=target_density
        )

    # Clean keywords
    keywords = [kw.lower().strip() for kw in target_keywords if kw.strip()]
    if not keywords:
        return KeywordDensityResult(
            original_text=description,
            optimized_text=description,
            original_density=0.0,
            optimized_density=0.0,
            keywords_added=[],
            target_density=target_density
        )

    # Calculate original density
    words = description.lower().split()
    word_count = len(words)

    if word_count == 0:
        return KeywordDensityResult(
            original_text=description,
            optimized_text=description,
            original_density=0.0,
            optimized_density=0.0,
            keywords_added=[],
            target_density=target_density
        )

    keyword_count = sum(1 for w in words if any(kw in w for kw in keywords))
    original_density = keyword_count / word_count

    # If already at or above target, return as-is
    if original_density >= target_density:
        return KeywordDensityResult(
            original_text=description,
            optimized_text=description,
            original_density=original_density,
            optimized_density=original_density,
            keywords_added=[],
            target_density=target_density
        )

    # Calculate how many keywords to add
    target_keyword_count = int(word_count * target_density)
    keywords_to_add = target_keyword_count - keyword_count

    if keywords_to_add <= 0:
        return KeywordDensityResult(
            original_text=description,
            optimized_text=description,
            original_density=original_density,
            optimized_density=original_density,
            keywords_added=[],
            target_density=target_density
        )

    # Natural keyword insertion phrases
    insertion_templates = [
        "\n\nIn this video about {keyword}, you'll discover",
        "\n\nLearn more about {keyword} and",
        "\n\nThis {keyword} guide covers",
        "\n\nTopics covered: {keyword},",
        "\n\nRelated: {keyword}",
    ]

    # Add keywords naturally
    optimized = description
    added_keywords = []

    for i, keyword in enumerate(keywords[:keywords_to_add]):
        template = insertion_templates[i % len(insertion_templates)]
        insertion = template.format(keyword=keyword)
        optimized += insertion
        added_keywords.append(keyword)

    # Recalculate density
    new_words = optimized.lower().split()
    new_word_count = len(new_words)
    new_keyword_count = sum(1 for w in new_words if any(kw in w for kw in keywords))
    new_density = new_keyword_count / new_word_count if new_word_count > 0 else 0

    result = KeywordDensityResult(
        original_text=description,
        optimized_text=optimized,
        original_density=original_density,
        optimized_density=new_density,
        keywords_added=added_keywords,
        target_density=target_density
    )

    logger.info(f"Keyword density optimized: {result}")
    return result


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """CLI entry point for YouTube optimizer utilities."""
    import sys

    print("""
YouTube Algorithm Optimization Utilities
========================================

Available tools:
1. UploadTimingOptimizer - Calculate optimal upload times
2. FirstHourBooster - Schedule post-upload engagement
3. TitlePatternAnalyzer - Analyze and improve titles
4. generate_chapters_from_script - Auto-generate chapters
5. optimize_description_keywords - Keyword density optimizer

Examples:
    # Calculate optimal upload time
    from src.utils.youtube_optimizer import UploadTimingOptimizer
    timing = UploadTimingOptimizer()
    result = timing.calculate_optimal_time("finance", ["US_EST", "UK"])
    print(result)

    # Analyze title
    from src.utils.youtube_optimizer import TitlePatternAnalyzer
    analyzer = TitlePatternAnalyzer()
    result = analyzer.analyze_title("5 Money Mistakes", "finance")
    print(f"Score: {result.score}/100")

    # Schedule first hour actions
    from src.utils.youtube_optimizer import FirstHourBooster
    booster = FirstHourBooster()
    actions = booster.schedule_post_upload_actions("video_id")
    print(booster.get_action_timeline(actions))
    """)

    # Demo if no arguments
    if len(sys.argv) < 2:
        print("\n--- Demo ---\n")

        # Demo upload timing
        timing = UploadTimingOptimizer()
        optimal = timing.calculate_optimal_time("finance", ["US_EST", "UK"])
        print(f"Optimal upload time: {optimal}")
        print(f"Reasoning: {optimal.reasoning}\n")

        # Demo title analysis
        analyzer = TitlePatternAnalyzer()
        titles_to_test = [
            "5 Money Mistakes Costing You $1000/Year",
            "How to invest",
            "The Secret Psychology Trick That Changes Everything [2026]",
        ]

        for title in titles_to_test:
            result = analyzer.analyze_title(title, "finance")
            print(f"\nTitle: '{title}'")
            print(f"  Score: {result.score:.0f}/100 ({result.viral_potential})")
            print(f"  Patterns: {', '.join(result.pattern_matches[:3])}")
            if result.suggestions:
                print(f"  Suggestions: {result.suggestions[0]}")

        # Demo first hour booster
        print("\n--- First Hour Timeline ---")
        booster = FirstHourBooster()
        actions = booster.schedule_post_upload_actions("abc123", "finance")
        print(booster.get_action_timeline(actions))


if __name__ == "__main__":
    main()
