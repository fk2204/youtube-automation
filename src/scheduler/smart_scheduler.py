"""
Smart Scheduler - Advanced Scheduling with Optimization

Provides intelligent scheduling features for YouTube automation:
- OptimalTimeCalculator: Best upload times per channel/niche
- AudienceTimezoneAnalyzer: Post when audience is active
- CompetitorAvoidance: Avoid uploading when competitors do
- ContentCalendar: Weekly/monthly planning
- BatchScheduler: Schedule multiple videos efficiently
- HolidayAwareness: Adjust for holidays/events

Usage:
    from src.scheduler.smart_scheduler import SmartScheduler

    scheduler = SmartScheduler()

    # Get optimal upload time
    best_time = await scheduler.get_optimal_time("money_blueprints", "finance")

    # Plan content calendar
    calendar = await scheduler.plan_content_calendar(
        channel_id="money_blueprints",
        weeks=4
    )

    # Batch schedule videos
    result = await scheduler.batch_schedule([
        {"channel_id": "money_blueprints", "topic": "passive income"},
        {"channel_id": "mind_unlocked", "topic": "psychology tips"}
    ])
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger
import yaml
import random


class DayOfWeek(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class SchedulePriority(Enum):
    """Priority levels for scheduled content."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TimeSlot:
    """Represents a time slot for uploading."""
    hour: int
    minute: int = 0
    score: float = 0.0
    reason: str = ""

    @property
    def time_str(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hour": self.hour,
            "minute": self.minute,
            "time_str": self.time_str,
            "score": self.score,
            "reason": self.reason
        }


@dataclass
class ScheduledContent:
    """Content scheduled for upload."""
    content_id: str
    channel_id: str
    topic: str
    scheduled_time: datetime
    priority: int = SchedulePriority.NORMAL.value
    status: str = "pending"
    niche: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["scheduled_time"] = self.scheduled_time.isoformat()
        return result


@dataclass
class ContentCalendarEntry:
    """Entry in the content calendar."""
    date: str
    day_of_week: str
    channel_id: str
    topic: str
    scheduled_time: str
    niche: str
    status: str = "planned"
    content_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Holiday:
    """Holiday definition for scheduling awareness."""
    name: str
    date: str  # MM-DD format or YYYY-MM-DD for specific year
    recurring: bool = True
    country: str = "US"
    avoid_upload: bool = False  # Whether to avoid uploading on this day
    boost_topics: List[str] = field(default_factory=list)  # Topics that perform well

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OptimalTimeCalculator:
    """
    Calculate optimal upload times based on:
    - Historical channel performance
    - Niche-specific patterns
    - Audience activity windows
    """

    # Niche-specific optimal times (UTC hours)
    NICHE_OPTIMAL_TIMES: Dict[str, List[int]] = {
        "finance": [14, 15, 16, 19, 20, 21],  # Business hours + evening
        "psychology": [10, 11, 15, 16, 19, 20],  # Morning + afternoon
        "storytelling": [17, 18, 19, 20, 21, 22],  # Evening prime time
        "technology": [13, 14, 15, 18, 19],
        "education": [9, 10, 11, 14, 15, 16],
        "entertainment": [16, 17, 18, 19, 20, 21, 22],
        "gaming": [14, 15, 16, 19, 20, 21, 22],
        "lifestyle": [10, 11, 12, 17, 18, 19],
    }

    # Day-of-week modifiers (1.0 = no change)
    DAY_MODIFIERS: Dict[int, float] = {
        0: 0.95,   # Monday - slightly lower
        1: 1.0,    # Tuesday - baseline
        2: 1.0,    # Wednesday - baseline
        3: 1.05,   # Thursday - slight boost
        4: 0.9,    # Friday - lower engagement
        5: 0.85,   # Saturday - weekend dip
        6: 0.9,    # Sunday - recovering
    }

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/scheduler.db")
        self._init_db()

    def _init_db(self):
        """Initialize database for performance tracking."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upload_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    video_id TEXT,
                    upload_hour INTEGER,
                    upload_day INTEGER,
                    views_24h INTEGER DEFAULT 0,
                    engagement_rate REAL DEFAULT 0,
                    uploaded_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_upload_perf_channel
                ON upload_performance(channel_id, upload_hour, upload_day)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_upload_perf_created
                ON upload_performance(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_upload_perf_views
                ON upload_performance(views_24h)
            """)

    async def calculate_optimal_time(
        self,
        channel_id: str,
        niche: str,
        target_date: Optional[datetime] = None
    ) -> TimeSlot:
        """
        Calculate the optimal upload time for a channel.

        Args:
            channel_id: Channel identifier
            niche: Content niche
            target_date: Target date for upload

        Returns:
            TimeSlot with optimal hour and score
        """
        target_date = target_date or datetime.now(timezone.utc)
        day_of_week = target_date.weekday()

        # Get base optimal hours for niche
        base_hours = self.NICHE_OPTIMAL_TIMES.get(
            niche.lower(),
            [14, 15, 16, 19, 20]  # Default
        )

        # Score each hour
        hour_scores: List[Tuple[int, float, str]] = []

        for hour in base_hours:
            score = 1.0
            reasons = []

            # Apply day modifier
            day_mod = self.DAY_MODIFIERS.get(day_of_week, 1.0)
            score *= day_mod
            if day_mod > 1.0:
                reasons.append(f"Strong day ({DayOfWeek(day_of_week).name})")
            elif day_mod < 0.95:
                reasons.append(f"Weaker day ({DayOfWeek(day_of_week).name})")

            # Check historical performance
            historical_score = await self._get_historical_score(channel_id, hour, day_of_week)
            if historical_score > 0:
                score *= historical_score
                reasons.append(f"Historical: {historical_score:.2f}")

            # Peak hours boost
            if hour in [19, 20, 21]:
                score *= 1.1
                reasons.append("Prime time boost")

            hour_scores.append((hour, score, " | ".join(reasons)))

        # Sort by score and pick best
        hour_scores.sort(key=lambda x: x[1], reverse=True)
        best_hour, best_score, reason = hour_scores[0]

        return TimeSlot(
            hour=best_hour,
            minute=0,
            score=best_score,
            reason=reason or "Niche optimal time"
        )

    async def _get_historical_score(
        self,
        channel_id: str,
        hour: int,
        day_of_week: int
    ) -> float:
        """Get historical performance score for a time slot."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT AVG(views_24h), AVG(engagement_rate), COUNT(*)
                    FROM upload_performance
                    WHERE channel_id = ? AND upload_hour = ? AND upload_day = ?
                """, (channel_id, hour, day_of_week)).fetchone()

                if result and result[2] >= 3:  # Need at least 3 data points
                    avg_views = result[0] or 0
                    avg_engagement = result[1] or 0

                    # Normalize to a score (simplified)
                    # In production, compare against channel average
                    return 1.0 + (avg_engagement * 10)
        except Exception as e:
            logger.debug(f"Historical score lookup failed: {e}")

        return 0.0

    async def record_performance(
        self,
        channel_id: str,
        video_id: str,
        upload_time: datetime,
        views_24h: int,
        engagement_rate: float
    ):
        """Record upload performance for learning."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO upload_performance
                (channel_id, video_id, upload_hour, upload_day, views_24h,
                 engagement_rate, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                channel_id, video_id, upload_time.hour, upload_time.weekday(),
                views_24h, engagement_rate, upload_time.isoformat()
            ))


class AudienceTimezoneAnalyzer:
    """
    Analyze and optimize for audience timezone distribution.
    Posts when the majority of audience is active.
    """

    # Major timezone regions and their UTC offsets
    TIMEZONE_REGIONS: Dict[str, int] = {
        "US_PACIFIC": -8,
        "US_MOUNTAIN": -7,
        "US_CENTRAL": -6,
        "US_EASTERN": -5,
        "UK": 0,
        "EUROPE_CENTRAL": 1,
        "INDIA": 5,  # +5:30 rounded
        "SOUTHEAST_ASIA": 7,
        "AUSTRALIA_EAST": 10,
    }

    # Default audience distribution by region (percentages)
    DEFAULT_AUDIENCE_DISTRIBUTION: Dict[str, float] = {
        "US_EASTERN": 25.0,
        "US_CENTRAL": 15.0,
        "US_PACIFIC": 20.0,
        "UK": 12.0,
        "EUROPE_CENTRAL": 10.0,
        "INDIA": 8.0,
        "AUSTRALIA_EAST": 5.0,
        "SOUTHEAST_ASIA": 5.0,
    }

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/scheduler.db")
        self._audience_cache: Dict[str, Dict[str, float]] = {}

    async def analyze_optimal_window(
        self,
        channel_id: str,
        audience_distribution: Optional[Dict[str, float]] = None
    ) -> List[TimeSlot]:
        """
        Find optimal upload windows based on audience timezones.

        Args:
            channel_id: Channel identifier
            audience_distribution: Custom audience distribution by region

        Returns:
            List of optimal time slots sorted by score
        """
        distribution = audience_distribution or self._audience_cache.get(
            channel_id,
            self.DEFAULT_AUDIENCE_DISTRIBUTION
        )

        # Calculate audience awake percentage for each UTC hour
        hour_scores: Dict[int, float] = {}

        for utc_hour in range(24):
            total_awake = 0.0

            for region, percentage in distribution.items():
                if region not in self.TIMEZONE_REGIONS:
                    continue

                offset = self.TIMEZONE_REGIONS[region]
                local_hour = (utc_hour + offset) % 24

                # Audience activity model (awake and active)
                # Peak: 18-22 local, Good: 10-18 local, Low: 6-10, 22-24
                if 18 <= local_hour <= 22:
                    activity = 1.0  # Peak activity
                elif 10 <= local_hour < 18:
                    activity = 0.7  # Good activity
                elif 6 <= local_hour < 10:
                    activity = 0.4  # Morning
                elif 22 < local_hour <= 23:
                    activity = 0.5  # Late night
                else:
                    activity = 0.1  # Sleeping

                total_awake += percentage * activity / 100

            hour_scores[utc_hour] = total_awake

        # Convert to time slots
        slots = [
            TimeSlot(
                hour=hour,
                score=score,
                reason=f"Audience activity: {score:.1%}"
            )
            for hour, score in hour_scores.items()
        ]

        # Sort by score
        slots.sort(key=lambda x: x.score, reverse=True)

        return slots[:6]  # Return top 6 slots

    async def update_audience_distribution(
        self,
        channel_id: str,
        distribution: Dict[str, float]
    ):
        """Update stored audience distribution for a channel."""
        # Normalize to 100%
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total * 100 for k, v in distribution.items()}

        self._audience_cache[channel_id] = distribution

        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audience_distribution (
                        channel_id TEXT PRIMARY KEY,
                        distribution TEXT,
                        updated_at TEXT
                    )
                """)
                conn.execute("""
                    INSERT OR REPLACE INTO audience_distribution
                    (channel_id, distribution, updated_at)
                    VALUES (?, ?, ?)
                """, (channel_id, json.dumps(distribution), datetime.now().isoformat()))
        except Exception as e:
            logger.warning(f"Failed to persist audience distribution: {e}")

    async def get_best_time_for_regions(
        self,
        target_regions: List[str],
        weight_by_engagement: bool = True
    ) -> TimeSlot:
        """Get best upload time targeting specific regions."""
        distribution = {
            region: 100.0 / len(target_regions)
            for region in target_regions
            if region in self.TIMEZONE_REGIONS
        }

        slots = await self.analyze_optimal_window("_custom", distribution)
        return slots[0] if slots else TimeSlot(hour=15, score=0.5, reason="Default")


class CompetitorAvoidance:
    """
    Track competitor upload patterns and avoid scheduling conflicts.
    Helps content stand out by not competing directly.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/scheduler.db")
        self._init_db()
        self._competitor_patterns: Dict[str, List[int]] = {}

    def _init_db(self):
        """Initialize competitor tracking database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS competitor_uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competitor_id TEXT NOT NULL,
                    niche TEXT,
                    upload_hour INTEGER,
                    upload_day INTEGER,
                    video_title TEXT,
                    observed_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_competitor_niche
                ON competitor_uploads(niche, upload_hour, upload_day)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_competitor_observed
                ON competitor_uploads(observed_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_competitor_id
                ON competitor_uploads(competitor_id)
            """)

    async def record_competitor_upload(
        self,
        competitor_id: str,
        niche: str,
        upload_time: datetime,
        video_title: str = ""
    ):
        """Record a competitor's upload for pattern analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO competitor_uploads
                (competitor_id, niche, upload_hour, upload_day, video_title)
                VALUES (?, ?, ?, ?, ?)
            """, (competitor_id, niche, upload_time.hour, upload_time.weekday(), video_title))

        logger.debug(f"Recorded competitor upload: {competitor_id} at {upload_time}")

    async def get_competitor_hot_hours(
        self,
        niche: str,
        day_of_week: Optional[int] = None
    ) -> List[int]:
        """
        Get hours when competitors frequently upload.

        Args:
            niche: Content niche
            day_of_week: Specific day (0=Mon), None for all days

        Returns:
            List of hours with high competitor activity
        """
        query = """
            SELECT upload_hour, COUNT(*) as count
            FROM competitor_uploads
            WHERE niche = ?
        """
        params: List[Any] = [niche]

        if day_of_week is not None:
            query += " AND upload_day = ?"
            params.append(day_of_week)

        query += " GROUP BY upload_hour ORDER BY count DESC"

        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute(query, params).fetchall()

        # Return hours with above-average activity
        if not results:
            return []

        avg_count = sum(r[1] for r in results) / len(results)
        return [r[0] for r in results if r[1] > avg_count * 1.2]

    async def find_gap_slots(
        self,
        niche: str,
        target_date: datetime,
        preferred_hours: List[int]
    ) -> List[TimeSlot]:
        """
        Find time slots that avoid competitor uploads.

        Args:
            niche: Content niche
            target_date: Target upload date
            preferred_hours: Preferred hours to consider

        Returns:
            List of gap slots sorted by score
        """
        day_of_week = target_date.weekday()
        competitor_hours = await self.get_competitor_hot_hours(niche, day_of_week)

        gap_slots = []
        for hour in preferred_hours:
            if hour in competitor_hours:
                # Competitor active - lower score
                score = 0.5
                reason = "Competitor activity detected"
            else:
                # Gap found - higher score
                score = 1.0
                reason = "Low competitor activity"

            gap_slots.append(TimeSlot(hour=hour, score=score, reason=reason))

        gap_slots.sort(key=lambda x: x.score, reverse=True)
        return gap_slots

    async def suggest_counter_schedule(
        self,
        niche: str,
        our_upload_time: datetime
    ) -> Dict[str, Any]:
        """
        Analyze if our schedule competes with major competitors.

        Returns:
            Dict with competition analysis and suggestions
        """
        competitor_hours = await self.get_competitor_hot_hours(
            niche, our_upload_time.weekday()
        )

        our_hour = our_upload_time.hour
        is_competing = our_hour in competitor_hours

        suggestions = []
        if is_competing:
            # Find alternative hours
            for offset in [1, -1, 2, -2]:
                alt_hour = (our_hour + offset) % 24
                if alt_hour not in competitor_hours:
                    suggestions.append(f"{alt_hour:02d}:00 UTC")

        return {
            "current_hour": our_hour,
            "is_competing": is_competing,
            "competitor_hot_hours": competitor_hours,
            "suggested_alternatives": suggestions,
            "recommendation": (
                "Consider shifting upload time" if is_competing
                else "Good timing - low competition"
            )
        }


class ContentCalendar:
    """
    Plan and manage content calendars for weeks/months ahead.
    Ensures consistent posting and topic variety.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/channels.yaml")
        self._calendar: Dict[str, List[ContentCalendarEntry]] = {}
        self._load_channel_config()

    def _load_channel_config(self):
        """Load channel configuration."""
        self.channels_config: Dict[str, Dict[str, Any]] = {}

        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = yaml.safe_load(f)

                for channel in config.get("channels", []):
                    self.channels_config[channel["id"]] = channel
            except Exception as e:
                logger.warning(f"Failed to load channel config: {e}")

    async def generate_calendar(
        self,
        channel_id: str,
        weeks: int = 4,
        start_date: Optional[datetime] = None,
        topics: Optional[List[str]] = None
    ) -> List[ContentCalendarEntry]:
        """
        Generate a content calendar for the specified period.

        Args:
            channel_id: Channel identifier
            weeks: Number of weeks to plan
            start_date: Starting date (default: today)
            topics: Custom topic list (default: from config)

        Returns:
            List of calendar entries
        """
        start = start_date or datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(weeks=weeks)

        # Get channel settings
        channel_config = self.channels_config.get(channel_id, {})
        settings = channel_config.get("settings", {})

        posting_days = settings.get("posting_days", [0, 2, 4])  # Mon, Wed, Fri default
        niche = settings.get("niche", "general")
        available_topics = topics or settings.get("topics", ["General content"])

        # Generate entries
        entries: List[ContentCalendarEntry] = []
        current = start
        topic_index = 0

        while current < end:
            if current.weekday() in posting_days:
                topic = available_topics[topic_index % len(available_topics)]
                topic_index += 1

                # Get optimal time for this day
                time_calculator = OptimalTimeCalculator()
                optimal_slot = await time_calculator.calculate_optimal_time(
                    channel_id, niche, current
                )

                entry = ContentCalendarEntry(
                    date=current.strftime("%Y-%m-%d"),
                    day_of_week=DayOfWeek(current.weekday()).name,
                    channel_id=channel_id,
                    topic=topic,
                    scheduled_time=optimal_slot.time_str,
                    niche=niche
                )
                entries.append(entry)

            current += timedelta(days=1)

        self._calendar[channel_id] = entries
        return entries

    async def get_calendar(
        self,
        channel_id: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[ContentCalendarEntry]:
        """
        Get calendar entries for a channel within date range.
        """
        entries = self._calendar.get(channel_id, [])

        if not from_date and not to_date:
            return entries

        filtered = []
        for entry in entries:
            entry_date = datetime.strptime(entry.date, "%Y-%m-%d")

            if from_date and entry_date < from_date:
                continue
            if to_date and entry_date > to_date:
                continue

            filtered.append(entry)

        return filtered

    async def update_entry(
        self,
        channel_id: str,
        date: str,
        updates: Dict[str, Any]
    ) -> Optional[ContentCalendarEntry]:
        """Update a calendar entry."""
        entries = self._calendar.get(channel_id, [])

        for i, entry in enumerate(entries):
            if entry.date == date:
                for key, value in updates.items():
                    if hasattr(entry, key):
                        setattr(entry, key, value)
                return entry

        return None

    async def export_calendar(
        self,
        channel_id: str,
        format: str = "json"
    ) -> str:
        """Export calendar to JSON or CSV format."""
        entries = self._calendar.get(channel_id, [])

        if format == "json":
            return json.dumps([e.to_dict() for e in entries], indent=2)
        elif format == "csv":
            lines = ["date,day,channel,topic,time,niche,status"]
            for e in entries:
                lines.append(
                    f"{e.date},{e.day_of_week},{e.channel_id},"
                    f"\"{e.topic}\",{e.scheduled_time},{e.niche},{e.status}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def get_upcoming(
        self,
        channel_id: str,
        days: int = 7
    ) -> List[ContentCalendarEntry]:
        """Get upcoming content for the next N days."""
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)
        return await self.get_calendar(channel_id, now, end)


class BatchScheduler:
    """
    Schedule multiple videos efficiently with conflict resolution.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/scheduler.db")
        self._init_db()
        self._scheduled_content: Dict[str, ScheduledContent] = {}

    def _init_db(self):
        """Initialize batch scheduling database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_content (
                    content_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    topic TEXT,
                    scheduled_time TEXT,
                    priority INTEGER DEFAULT 2,
                    status TEXT DEFAULT 'pending',
                    niche TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scheduled_time
                ON scheduled_content(scheduled_time, status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scheduled_channel
                ON scheduled_content(channel_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scheduled_status
                ON scheduled_content(status, created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scheduled_priority
                ON scheduled_content(priority, scheduled_time)
            """)

    async def batch_schedule(
        self,
        items: List[Dict[str, Any]],
        avoid_conflicts: bool = True,
        min_gap_hours: int = 2
    ) -> Dict[str, Any]:
        """
        Schedule multiple videos with conflict resolution.

        Args:
            items: List of items to schedule, each with:
                   - channel_id: Channel identifier
                   - topic: Video topic
                   - preferred_time: Optional preferred datetime
                   - priority: Optional priority level
            avoid_conflicts: Whether to resolve time conflicts
            min_gap_hours: Minimum hours between uploads to same channel

        Returns:
            Dict with scheduled items and any conflicts resolved
        """
        scheduled = []
        conflicts_resolved = []
        failed = []

        # Group by channel
        by_channel: Dict[str, List[Dict]] = {}
        for item in items:
            channel = item.get("channel_id", "unknown")
            if channel not in by_channel:
                by_channel[channel] = []
            by_channel[channel].append(item)

        # Schedule each channel's items
        for channel_id, channel_items in by_channel.items():
            last_scheduled: Optional[datetime] = None

            for item in channel_items:
                try:
                    # Determine schedule time
                    preferred = item.get("preferred_time")
                    if isinstance(preferred, str):
                        schedule_time = datetime.fromisoformat(preferred)
                    elif isinstance(preferred, datetime):
                        schedule_time = preferred
                    else:
                        # Calculate optimal time
                        calculator = OptimalTimeCalculator()
                        slot = await calculator.calculate_optimal_time(
                            channel_id,
                            item.get("niche", "general")
                        )
                        now = datetime.now(timezone.utc)
                        schedule_time = now.replace(
                            hour=slot.hour, minute=slot.minute, second=0, microsecond=0
                        )
                        if schedule_time <= now:
                            schedule_time += timedelta(days=1)

                    # Check for conflicts
                    if avoid_conflicts and last_scheduled:
                        gap = (schedule_time - last_scheduled).total_seconds() / 3600
                        if gap < min_gap_hours:
                            # Resolve conflict
                            old_time = schedule_time
                            schedule_time = last_scheduled + timedelta(hours=min_gap_hours)
                            conflicts_resolved.append({
                                "item": item,
                                "original_time": old_time.isoformat(),
                                "new_time": schedule_time.isoformat(),
                                "reason": f"Too close to previous upload (gap: {gap:.1f}h)"
                            })

                    # Create scheduled content
                    content_id = f"batch_{channel_id}_{schedule_time.strftime('%Y%m%d%H%M')}"
                    content = ScheduledContent(
                        content_id=content_id,
                        channel_id=channel_id,
                        topic=item.get("topic", "Untitled"),
                        scheduled_time=schedule_time,
                        priority=item.get("priority", SchedulePriority.NORMAL.value),
                        niche=item.get("niche", ""),
                        metadata=item.get("metadata", {})
                    )

                    # Save to database
                    await self._save_content(content)
                    self._scheduled_content[content_id] = content
                    scheduled.append(content.to_dict())

                    last_scheduled = schedule_time

                except Exception as e:
                    failed.append({
                        "item": item,
                        "error": str(e)
                    })

        return {
            "scheduled": scheduled,
            "conflicts_resolved": conflicts_resolved,
            "failed": failed,
            "total_scheduled": len(scheduled),
            "total_conflicts": len(conflicts_resolved),
            "total_failed": len(failed)
        }

    async def _save_content(self, content: ScheduledContent):
        """Save scheduled content to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scheduled_content
                (content_id, channel_id, topic, scheduled_time, priority,
                 status, niche, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content.content_id, content.channel_id, content.topic,
                content.scheduled_time.isoformat(), content.priority,
                content.status, content.niche, json.dumps(content.metadata),
                content.created_at
            ))

    async def get_scheduled(
        self,
        channel_id: Optional[str] = None,
        status: str = "pending"
    ) -> List[ScheduledContent]:
        """Get scheduled content."""
        query = "SELECT * FROM scheduled_content WHERE status = ?"
        params: List[Any] = [status]

        if channel_id:
            query += " AND channel_id = ?"
            params.append(channel_id)

        query += " ORDER BY scheduled_time ASC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            ScheduledContent(
                content_id=row["content_id"],
                channel_id=row["channel_id"],
                topic=row["topic"],
                scheduled_time=datetime.fromisoformat(row["scheduled_time"]),
                priority=row["priority"],
                status=row["status"],
                niche=row["niche"] or "",
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"]
            )
            for row in rows
        ]

    async def update_status(
        self,
        content_id: str,
        status: str
    ):
        """Update content status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE scheduled_content SET status = ? WHERE content_id = ?
            """, (status, content_id))

        if content_id in self._scheduled_content:
            self._scheduled_content[content_id].status = status

    async def cancel(self, content_id: str) -> bool:
        """Cancel a scheduled item."""
        await self.update_status(content_id, "cancelled")
        return True

    async def reschedule(
        self,
        content_id: str,
        new_time: datetime
    ) -> Optional[ScheduledContent]:
        """Reschedule an item."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE scheduled_content
                SET scheduled_time = ?
                WHERE content_id = ?
            """, (new_time.isoformat(), content_id))

        if content_id in self._scheduled_content:
            self._scheduled_content[content_id].scheduled_time = new_time
            return self._scheduled_content[content_id]

        return None


class HolidayAwareness:
    """
    Adjust scheduling based on holidays and special events.
    """

    # US Federal holidays and common observances
    DEFAULT_HOLIDAYS: List[Holiday] = [
        Holiday("New Year's Day", "01-01", True, "US", True, ["new year", "goals", "resolutions"]),
        Holiday("Martin Luther King Jr. Day", "01-15", True, "US", False),  # Third Monday
        Holiday("Valentine's Day", "02-14", True, "US", False, ["love", "relationships"]),
        Holiday("Presidents Day", "02-15", True, "US", False),  # Third Monday
        Holiday("Memorial Day", "05-25", True, "US", True),  # Last Monday
        Holiday("Independence Day", "07-04", True, "US", True),
        Holiday("Labor Day", "09-01", True, "US", True),  # First Monday
        Holiday("Halloween", "10-31", True, "US", False, ["scary", "horror", "mystery"]),
        Holiday("Thanksgiving", "11-28", True, "US", True, ["gratitude", "family"]),
        Holiday("Black Friday", "11-29", True, "US", False, ["deals", "money", "shopping"]),
        Holiday("Christmas Eve", "12-24", True, "US", True),
        Holiday("Christmas Day", "12-25", True, "US", True, ["christmas", "holiday"]),
        Holiday("New Year's Eve", "12-31", True, "US", True, ["year review", "predictions"]),
    ]

    def __init__(self, custom_holidays: Optional[List[Holiday]] = None):
        self.holidays = self.DEFAULT_HOLIDAYS.copy()
        if custom_holidays:
            self.holidays.extend(custom_holidays)

    def _parse_holiday_date(self, holiday: Holiday, year: int) -> Optional[datetime]:
        """Parse holiday date for a specific year."""
        try:
            if len(holiday.date) == 5:  # MM-DD format
                return datetime(year, int(holiday.date[:2]), int(holiday.date[3:]))
            else:  # YYYY-MM-DD format
                return datetime.fromisoformat(holiday.date)
        except Exception:
            return None

    async def is_holiday(
        self,
        date: datetime,
        country: str = "US"
    ) -> Optional[Holiday]:
        """
        Check if a date is a holiday.

        Args:
            date: Date to check
            country: Country code

        Returns:
            Holiday if found, None otherwise
        """
        date_str = date.strftime("%m-%d")

        for holiday in self.holidays:
            if holiday.country != country:
                continue

            if holiday.recurring:
                if holiday.date == date_str:
                    return holiday
            else:
                if self._parse_holiday_date(holiday, date.year) == date.date():
                    return holiday

        return None

    async def should_avoid_upload(
        self,
        date: datetime,
        country: str = "US"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if uploads should be avoided on a date.

        Returns:
            Tuple of (should_avoid, reason)
        """
        holiday = await self.is_holiday(date, country)

        if holiday and holiday.avoid_upload:
            return True, f"Holiday: {holiday.name}"

        return False, None

    async def get_suggested_topics(
        self,
        date: datetime,
        base_niche: str
    ) -> List[str]:
        """
        Get topic suggestions for a date based on holidays.

        Args:
            date: Target date
            base_niche: Channel's base niche

        Returns:
            List of suggested topics
        """
        holiday = await self.is_holiday(date)
        suggestions = []

        if holiday and holiday.boost_topics:
            for topic in holiday.boost_topics:
                # Combine with niche
                suggestions.append(f"{topic} {base_niche}")
                suggestions.append(f"{base_niche} for {holiday.name}")

        return suggestions

    async def find_next_upload_day(
        self,
        start_date: datetime,
        posting_days: List[int],
        country: str = "US"
    ) -> datetime:
        """
        Find the next valid upload day avoiding holidays.

        Args:
            start_date: Date to start searching from
            posting_days: Valid posting days (0=Mon, 6=Sun)
            country: Country for holiday checking

        Returns:
            Next valid upload datetime
        """
        current = start_date
        max_search = 30  # Don't search more than 30 days

        for _ in range(max_search):
            if current.weekday() in posting_days:
                should_avoid, _ = await self.should_avoid_upload(current, country)
                if not should_avoid:
                    return current

            current += timedelta(days=1)

        # Fallback: return start_date + 1 day
        return start_date + timedelta(days=1)

    async def get_upcoming_holidays(
        self,
        days: int = 30,
        country: str = "US"
    ) -> List[Dict[str, Any]]:
        """Get upcoming holidays within the specified period."""
        now = datetime.now()
        end = now + timedelta(days=days)
        upcoming = []

        for holiday in self.holidays:
            if holiday.country != country:
                continue

            holiday_date = self._parse_holiday_date(holiday, now.year)
            if holiday_date and now <= holiday_date <= end:
                upcoming.append({
                    "name": holiday.name,
                    "date": holiday_date.strftime("%Y-%m-%d"),
                    "avoid_upload": holiday.avoid_upload,
                    "boost_topics": holiday.boost_topics
                })

        upcoming.sort(key=lambda x: x["date"])
        return upcoming


class SmartScheduler:
    """
    Main scheduler class that combines all scheduling intelligence.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/channels.yaml")
        self.db_path = Path("data/scheduler.db")

        # Initialize components
        self.time_calculator = OptimalTimeCalculator(self.db_path)
        self.timezone_analyzer = AudienceTimezoneAnalyzer(self.db_path)
        self.competitor_avoidance = CompetitorAvoidance(self.db_path)
        self.content_calendar = ContentCalendar(self.config_path)
        self.batch_scheduler = BatchScheduler(self.db_path)
        self.holiday_awareness = HolidayAwareness()

        logger.info("SmartScheduler initialized with all components")

    async def get_optimal_time(
        self,
        channel_id: str,
        niche: str,
        target_date: Optional[datetime] = None,
        avoid_competitors: bool = True,
        consider_holidays: bool = True
    ) -> Dict[str, Any]:
        """
        Get the optimal upload time considering all factors.

        Args:
            channel_id: Channel identifier
            niche: Content niche
            target_date: Target date for upload
            avoid_competitors: Whether to avoid competitor times
            consider_holidays: Whether to check for holidays

        Returns:
            Dict with optimal time and reasoning
        """
        target_date = target_date or datetime.now(timezone.utc)

        # Check for holiday
        if consider_holidays:
            should_avoid, reason = await self.holiday_awareness.should_avoid_upload(target_date)
            if should_avoid:
                # Find next valid day
                posting_days = [0, 1, 2, 3, 4, 5, 6]  # All days as fallback
                target_date = await self.holiday_awareness.find_next_upload_day(
                    target_date + timedelta(days=1),
                    posting_days
                )

        # Get base optimal time
        base_slot = await self.time_calculator.calculate_optimal_time(
            channel_id, niche, target_date
        )

        # Get audience-optimal times
        audience_slots = await self.timezone_analyzer.analyze_optimal_window(channel_id)

        # Check competitor activity
        competitor_analysis = None
        if avoid_competitors:
            preferred_hours = [base_slot.hour] + [s.hour for s in audience_slots[:3]]
            gap_slots = await self.competitor_avoidance.find_gap_slots(
                niche, target_date, preferred_hours
            )

            if gap_slots:
                # Prefer gap slots with good audience activity
                for gap_slot in gap_slots:
                    if gap_slot.hour in [s.hour for s in audience_slots[:5]]:
                        base_slot = gap_slot
                        competitor_analysis = await self.competitor_avoidance.suggest_counter_schedule(
                            niche, target_date.replace(hour=gap_slot.hour)
                        )
                        break

        # Build final recommendation
        final_time = target_date.replace(
            hour=base_slot.hour,
            minute=base_slot.minute,
            second=0,
            microsecond=0
        )

        return {
            "recommended_time": final_time.isoformat(),
            "hour_utc": base_slot.hour,
            "minute": base_slot.minute,
            "score": base_slot.score,
            "reasoning": base_slot.reason,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "day_of_week": DayOfWeek(target_date.weekday()).name,
            "audience_top_hours": [s.hour for s in audience_slots[:3]],
            "competitor_analysis": competitor_analysis,
            "factors_considered": [
                "niche_optimal_time",
                "day_of_week",
                "audience_timezones",
                "competitor_avoidance" if avoid_competitors else None,
                "holiday_awareness" if consider_holidays else None
            ]
        }

    async def plan_content_calendar(
        self,
        channel_id: str,
        weeks: int = 4,
        start_date: Optional[datetime] = None
    ) -> List[ContentCalendarEntry]:
        """
        Plan a content calendar for the specified period.
        """
        return await self.content_calendar.generate_calendar(
            channel_id, weeks, start_date
        )

    async def batch_schedule(
        self,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Schedule multiple videos efficiently.
        """
        return await self.batch_scheduler.batch_schedule(items, **kwargs)

    async def get_upcoming_content(
        self,
        channel_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get all upcoming scheduled content for a channel.
        """
        calendar = await self.content_calendar.get_upcoming(channel_id, days)
        scheduled = await self.batch_scheduler.get_scheduled(channel_id, "pending")
        holidays = await self.holiday_awareness.get_upcoming_holidays(days)

        return {
            "calendar": [e.to_dict() for e in calendar],
            "scheduled": [s.to_dict() for s in scheduled],
            "upcoming_holidays": holidays
        }

    async def record_upload_performance(
        self,
        channel_id: str,
        video_id: str,
        upload_time: datetime,
        views_24h: int,
        engagement_rate: float
    ):
        """
        Record upload performance for learning.
        """
        await self.time_calculator.record_performance(
            channel_id, video_id, upload_time, views_24h, engagement_rate
        )


# CLI entry point
if __name__ == "__main__":
    import sys

    async def main():
        print("\n" + "=" * 60)
        print("  SMART SCHEDULER")
        print("=" * 60 + "\n")

        scheduler = SmartScheduler()

        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "optimal":
                channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
                niche = sys.argv[3] if len(sys.argv) > 3 else "finance"
                result = await scheduler.get_optimal_time(channel, niche)
                print(f"Channel: {channel}")
                print(f"Niche: {niche}")
                print(f"Recommended Time: {result['recommended_time']}")
                print(f"Score: {result['score']:.2f}")
                print(f"Reasoning: {result['reasoning']}")

            elif command == "calendar":
                channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
                weeks = int(sys.argv[3]) if len(sys.argv) > 3 else 2
                calendar = await scheduler.plan_content_calendar(channel, weeks)
                print(f"Content Calendar for {channel} ({weeks} weeks):\n")
                for entry in calendar:
                    print(f"  {entry.date} ({entry.day_of_week}): {entry.topic[:40]}...")
                    print(f"    Time: {entry.scheduled_time} UTC")

            elif command == "holidays":
                days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
                holidays = await scheduler.holiday_awareness.get_upcoming_holidays(days)
                print(f"Upcoming Holidays ({days} days):\n")
                for h in holidays:
                    avoid = "[AVOID]" if h["avoid_upload"] else ""
                    print(f"  {h['date']}: {h['name']} {avoid}")

            else:
                print(f"Unknown command: {command}")
        else:
            print("Usage:")
            print("  python -m src.scheduler.smart_scheduler optimal [channel] [niche]")
            print("  python -m src.scheduler.smart_scheduler calendar [channel] [weeks]")
            print("  python -m src.scheduler.smart_scheduler holidays [days]")

    asyncio.run(main())
