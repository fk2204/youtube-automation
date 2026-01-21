"""
Content Calendar Module for YouTube Automation

Comprehensive content planning with weekly/monthly views, topic clustering,
optimal posting analysis, and gap identification.

Features:
- Weekly/monthly planning view
- Topic clustering to avoid repetition
- Optimal posting day by niche
- Gap analysis (what topics haven't been covered)
- Integration with research modules
- Visual calendar export

Usage:
    from src.scheduler.content_calendar import ContentCalendar

    calendar = ContentCalendar()

    # Generate monthly calendar
    monthly = await calendar.generate_monthly_calendar(
        channel_id="money_blueprints",
        month=1,
        year=2026
    )

    # Check topic clustering
    clustered = calendar.cluster_topics(topics_list)

    # Find content gaps
    gaps = await calendar.find_content_gaps(
        channel_id="money_blueprints",
        topics_covered=["budgeting", "investing"]
    )

    # Export to various formats
    calendar.export_calendar(monthly, format="ical")
"""

import os
import json
import sqlite3
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from enum import Enum
from collections import Counter, defaultdict
from loguru import logger

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ContentStatus(Enum):
    """Status of a content piece."""
    IDEA = "idea"
    PLANNED = "planned"
    SCRIPTED = "scripted"
    RECORDED = "recorded"
    EDITED = "edited"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    CANCELLED = "cancelled"


class ContentType(Enum):
    """Type of content."""
    VIDEO = "video"
    SHORT = "short"
    LIVESTREAM = "livestream"
    COMMUNITY_POST = "community_post"


class DayOfWeek(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class ContentEntry:
    """A single content entry in the calendar."""
    entry_id: str
    channel_id: str
    title: str
    topic: str
    niche: str
    content_type: ContentType
    scheduled_date: date
    scheduled_time: Optional[str] = None  # HH:MM format
    status: ContentStatus = ContentStatus.PLANNED
    priority: int = 5  # 1-10, higher = more important
    keywords: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    notes: str = ""
    video_id: Optional[str] = None  # After publishing
    performance_score: Optional[float] = None  # After publishing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["content_type"] = self.content_type.value
        result["status"] = self.status.value
        result["scheduled_date"] = self.scheduled_date.isoformat()
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentEntry':
        """Create ContentEntry from dict."""
        data["content_type"] = ContentType(data.get("content_type", "video"))
        data["status"] = ContentStatus(data.get("status", "planned"))
        data["scheduled_date"] = date.fromisoformat(data["scheduled_date"])
        data["created_at"] = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        data["updated_at"] = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        return cls(**data)


@dataclass
class CalendarWeek:
    """A week view of the content calendar."""
    week_number: int
    year: int
    start_date: date
    end_date: date
    entries: List[ContentEntry]
    channel_id: str

    @property
    def video_count(self) -> int:
        return sum(1 for e in self.entries if e.content_type == ContentType.VIDEO)

    @property
    def short_count(self) -> int:
        return sum(1 for e in self.entries if e.content_type == ContentType.SHORT)

    def get_day_entries(self, day: date) -> List[ContentEntry]:
        return [e for e in self.entries if e.scheduled_date == day]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week_number": self.week_number,
            "year": self.year,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "entries": [e.to_dict() for e in self.entries],
            "channel_id": self.channel_id,
            "video_count": self.video_count,
            "short_count": self.short_count,
        }


@dataclass
class CalendarMonth:
    """A month view of the content calendar."""
    month: int
    year: int
    channel_id: str
    weeks: List[CalendarWeek]
    entries: List[ContentEntry]

    @property
    def total_content(self) -> int:
        return len(self.entries)

    @property
    def topics_covered(self) -> Set[str]:
        return {e.topic for e in self.entries}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "month": self.month,
            "year": self.year,
            "channel_id": self.channel_id,
            "weeks": [w.to_dict() for w in self.weeks],
            "total_content": self.total_content,
            "topics_covered": list(self.topics_covered),
        }


@dataclass
class TopicCluster:
    """A cluster of related topics."""
    cluster_id: str
    name: str
    topics: List[str]
    core_topic: str
    related_keywords: List[str]
    recommended_spacing_days: int = 14  # Days between videos in same cluster

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContentGap:
    """An identified gap in content coverage."""
    topic: str
    niche: str
    gap_type: str  # "never_covered", "outdated", "low_coverage"
    priority_score: float  # 0-100
    recommended_title: Optional[str] = None
    last_covered: Optional[date] = None
    trend_score: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.last_covered:
            result["last_covered"] = self.last_covered.isoformat()
        return result


# Optimal posting days by niche (based on typical audience availability)
NICHE_OPTIMAL_DAYS: Dict[str, List[int]] = {
    "finance": [1, 2, 3],  # Tue, Wed, Thu - Business week
    "psychology": [1, 3, 5],  # Tue, Thu, Sat - Mid-week + weekend
    "storytelling": [0, 1, 2, 3, 4, 5, 6],  # Every day - Entertainment
    "technology": [1, 2, 3, 4],  # Tue-Fri
    "education": [0, 2, 4],  # Mon, Wed, Fri
    "entertainment": [4, 5, 6],  # Fri, Sat, Sun
    "gaming": [3, 4, 5, 6],  # Thu-Sun
    "default": [1, 3, 5],  # Tue, Thu, Sat
}

# Topic clusters by niche for avoiding repetition
NICHE_TOPIC_CLUSTERS: Dict[str, List[TopicCluster]] = {
    "finance": [
        TopicCluster(
            cluster_id="investing",
            name="Investing",
            topics=["stocks", "etfs", "bonds", "crypto", "real estate investing", "index funds"],
            core_topic="investing",
            related_keywords=["portfolio", "returns", "dividends", "compound"],
            recommended_spacing_days=10
        ),
        TopicCluster(
            cluster_id="budgeting",
            name="Budgeting & Saving",
            topics=["budgeting", "saving money", "frugal", "cutting expenses", "emergency fund"],
            core_topic="budgeting",
            related_keywords=["save", "budget", "expense", "frugal"],
            recommended_spacing_days=14
        ),
        TopicCluster(
            cluster_id="passive_income",
            name="Passive Income",
            topics=["passive income", "side hustle", "multiple income streams", "dividends", "rental income"],
            core_topic="passive income",
            related_keywords=["passive", "income", "streams", "residual"],
            recommended_spacing_days=14
        ),
        TopicCluster(
            cluster_id="debt",
            name="Debt Management",
            topics=["debt payoff", "credit cards", "student loans", "mortgage", "debt free"],
            core_topic="debt",
            related_keywords=["debt", "loan", "credit", "payoff"],
            recommended_spacing_days=21
        ),
    ],
    "psychology": [
        TopicCluster(
            cluster_id="manipulation",
            name="Manipulation & Persuasion",
            topics=["manipulation", "persuasion", "influence", "dark psychology", "mind tricks"],
            core_topic="manipulation",
            related_keywords=["manipulate", "persuade", "influence", "control"],
            recommended_spacing_days=10
        ),
        TopicCluster(
            cluster_id="habits",
            name="Habits & Productivity",
            topics=["habits", "productivity", "discipline", "routine", "focus"],
            core_topic="habits",
            related_keywords=["habit", "productive", "discipline", "routine"],
            recommended_spacing_days=14
        ),
        TopicCluster(
            cluster_id="emotions",
            name="Emotions & Mental Health",
            topics=["emotions", "anxiety", "confidence", "self-esteem", "mental health"],
            core_topic="emotions",
            related_keywords=["emotion", "feeling", "mental", "confidence"],
            recommended_spacing_days=14
        ),
    ],
    "storytelling": [
        TopicCluster(
            cluster_id="true_crime",
            name="True Crime",
            topics=["true crime", "unsolved mysteries", "serial killers", "cold cases"],
            core_topic="true crime",
            related_keywords=["crime", "murder", "mystery", "case"],
            recommended_spacing_days=7
        ),
        TopicCluster(
            cluster_id="history",
            name="Historical Stories",
            topics=["history", "historical events", "ancient civilizations", "wars"],
            core_topic="history",
            related_keywords=["history", "ancient", "historical", "civilization"],
            recommended_spacing_days=10
        ),
        TopicCluster(
            cluster_id="conspiracy",
            name="Mysteries & Conspiracies",
            topics=["conspiracy", "unexplained", "paranormal", "aliens", "secrets"],
            core_topic="conspiracy",
            related_keywords=["conspiracy", "secret", "hidden", "unexplained"],
            recommended_spacing_days=14
        ),
    ],
}


class ContentCalendar:
    """
    Comprehensive content calendar for YouTube automation.

    Handles planning, scheduling, topic clustering, and gap analysis
    for multiple channels.
    """

    def __init__(
        self,
        db_path: str = "data/content_calendar.db",
        config_path: str = "config/channels.yaml"
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)

        self._init_database()
        self._load_channel_config()

        # Cache for entries
        self._entries_cache: Dict[str, List[ContentEntry]] = {}

        logger.info("ContentCalendar initialized")

    def _init_database(self):
        """Initialize database for calendar storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_entries (
                    entry_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    title TEXT,
                    topic TEXT,
                    niche TEXT,
                    content_type TEXT,
                    scheduled_date TEXT,
                    scheduled_time TEXT,
                    status TEXT,
                    priority INTEGER,
                    keywords TEXT,
                    related_topics TEXT,
                    notes TEXT,
                    video_id TEXT,
                    performance_score REAL,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    cluster_id TEXT,
                    published_date TEXT,
                    video_id TEXT,
                    performance_score REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_channel
                ON content_entries(channel_id, scheduled_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_status
                ON content_entries(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_history_channel
                ON topic_history(channel_id, topic)
            """)

    def _load_channel_config(self):
        """Load channel configuration."""
        self.channels_config: Dict[str, Dict[str, Any]] = {}

        if self.config_path.exists() and YAML_AVAILABLE:
            try:
                with open(self.config_path) as f:
                    config = yaml.safe_load(f)

                for channel in config.get("channels", []):
                    self.channels_config[channel["id"]] = channel
            except Exception as e:
                logger.warning(f"Failed to load channel config: {e}")

    async def generate_monthly_calendar(
        self,
        channel_id: str,
        month: int,
        year: int,
        topics: Optional[List[str]] = None,
        videos_per_week: int = 3,
        shorts_per_week: int = 6
    ) -> CalendarMonth:
        """
        Generate a monthly content calendar.

        Args:
            channel_id: Channel identifier
            month: Month number (1-12)
            year: Year
            topics: Optional list of topics to schedule
            videos_per_week: Target videos per week
            shorts_per_week: Target Shorts per week

        Returns:
            CalendarMonth object with planned content
        """
        logger.info(f"Generating monthly calendar for {channel_id}: {month}/{year}")

        # Get channel config
        channel_config = self.channels_config.get(channel_id, {})
        settings = channel_config.get("settings", {})
        niche = settings.get("niche", "default")
        posting_days = settings.get("posting_days", NICHE_OPTIMAL_DAYS.get(niche, [1, 3, 5]))

        # Calculate date range
        first_day = date(year, month, 1)
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        # Get or generate topics
        if topics is None:
            topics = await self._get_topics_for_channel(channel_id, niche, videos_per_week * 5)

        # Generate entries
        entries = []
        weeks = []
        current = first_day
        topic_index = 0

        while current <= last_day:
            # Find week boundaries
            week_start = current - timedelta(days=current.weekday())
            week_end = week_start + timedelta(days=6)

            week_entries = []

            # Schedule videos for this week
            videos_scheduled = 0
            shorts_scheduled = 0

            for day_offset in range(7):
                day = week_start + timedelta(days=day_offset)

                if day < first_day or day > last_day:
                    continue

                day_of_week = day.weekday()

                # Schedule video on posting days
                if day_of_week in posting_days and videos_scheduled < videos_per_week:
                    if topic_index < len(topics):
                        entry = self._create_entry(
                            channel_id=channel_id,
                            topic=topics[topic_index],
                            niche=niche,
                            content_type=ContentType.VIDEO,
                            scheduled_date=day
                        )
                        entries.append(entry)
                        week_entries.append(entry)
                        topic_index += 1
                        videos_scheduled += 1

                # Schedule Shorts on most days
                if shorts_scheduled < shorts_per_week:
                    short_entry = self._create_entry(
                        channel_id=channel_id,
                        topic=topics[topic_index % len(topics)] if topics else "Short content",
                        niche=niche,
                        content_type=ContentType.SHORT,
                        scheduled_date=day
                    )
                    entries.append(short_entry)
                    week_entries.append(short_entry)
                    shorts_scheduled += 1

            # Create week object
            week = CalendarWeek(
                week_number=current.isocalendar()[1],
                year=year,
                start_date=week_start,
                end_date=week_end,
                entries=week_entries,
                channel_id=channel_id
            )
            weeks.append(week)

            # Move to next week
            current = week_end + timedelta(days=1)

        # Create month calendar
        calendar = CalendarMonth(
            month=month,
            year=year,
            channel_id=channel_id,
            weeks=weeks,
            entries=entries
        )

        # Save entries to database
        for entry in entries:
            self._save_entry(entry)

        logger.success(f"Generated calendar with {len(entries)} entries")
        return calendar

    async def _get_topics_for_channel(
        self,
        channel_id: str,
        niche: str,
        count: int
    ) -> List[str]:
        """Get or generate topics for a channel."""
        # Check config for predefined topics
        channel_config = self.channels_config.get(channel_id, {})
        settings = channel_config.get("settings", {})
        predefined_topics = settings.get("topics", [])

        if predefined_topics:
            # Cycle through predefined topics
            topics = []
            for i in range(count):
                topics.append(predefined_topics[i % len(predefined_topics)])
            return topics

        # Generate topics from clusters
        clusters = NICHE_TOPIC_CLUSTERS.get(niche, [])
        if clusters:
            topics = []
            for i in range(count):
                cluster = clusters[i % len(clusters)]
                topic = cluster.topics[i % len(cluster.topics)]
                topics.append(topic)
            return topics

        # Fallback generic topics
        return [f"Topic {i+1}" for i in range(count)]

    def _create_entry(
        self,
        channel_id: str,
        topic: str,
        niche: str,
        content_type: ContentType,
        scheduled_date: date
    ) -> ContentEntry:
        """Create a content entry."""
        import uuid

        # Generate title from topic
        title = self._generate_title(topic, niche, content_type)

        # Get related keywords
        keywords = self._get_topic_keywords(topic, niche)

        return ContentEntry(
            entry_id=f"entry_{uuid.uuid4().hex[:8]}",
            channel_id=channel_id,
            title=title,
            topic=topic,
            niche=niche,
            content_type=content_type,
            scheduled_date=scheduled_date,
            keywords=keywords
        )

    def _generate_title(
        self,
        topic: str,
        niche: str,
        content_type: ContentType
    ) -> str:
        """Generate a placeholder title for a topic."""
        if content_type == ContentType.SHORT:
            return f"{topic.title()} Quick Tip"
        return f"{topic.title()}"

    def _get_topic_keywords(self, topic: str, niche: str) -> List[str]:
        """Get related keywords for a topic."""
        # Check clusters
        clusters = NICHE_TOPIC_CLUSTERS.get(niche, [])
        for cluster in clusters:
            if topic.lower() in [t.lower() for t in cluster.topics]:
                return cluster.related_keywords[:5]

        # Fall back to topic words
        return topic.lower().split()[:5]

    def _save_entry(self, entry: ContentEntry):
        """Save entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO content_entries
                (entry_id, channel_id, title, topic, niche, content_type,
                 scheduled_date, scheduled_time, status, priority, keywords,
                 related_topics, notes, video_id, performance_score,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id, entry.channel_id, entry.title, entry.topic,
                entry.niche, entry.content_type.value, entry.scheduled_date.isoformat(),
                entry.scheduled_time, entry.status.value, entry.priority,
                json.dumps(entry.keywords), json.dumps(entry.related_topics),
                entry.notes, entry.video_id, entry.performance_score,
                entry.created_at.isoformat(), entry.updated_at.isoformat()
            ))

    def get_weekly_view(
        self,
        channel_id: str,
        week_start: date
    ) -> CalendarWeek:
        """
        Get weekly calendar view.

        Args:
            channel_id: Channel identifier
            week_start: First day of the week

        Returns:
            CalendarWeek object
        """
        week_end = week_start + timedelta(days=6)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM content_entries
                WHERE channel_id = ?
                AND scheduled_date >= ? AND scheduled_date <= ?
                ORDER BY scheduled_date, scheduled_time
            """, (channel_id, week_start.isoformat(), week_end.isoformat())).fetchall()

            entries = [self._row_to_entry(row) for row in rows]

        return CalendarWeek(
            week_number=week_start.isocalendar()[1],
            year=week_start.year,
            start_date=week_start,
            end_date=week_end,
            entries=entries,
            channel_id=channel_id
        )

    def _row_to_entry(self, row: sqlite3.Row) -> ContentEntry:
        """Convert database row to ContentEntry."""
        return ContentEntry(
            entry_id=row["entry_id"],
            channel_id=row["channel_id"],
            title=row["title"],
            topic=row["topic"],
            niche=row["niche"],
            content_type=ContentType(row["content_type"]),
            scheduled_date=date.fromisoformat(row["scheduled_date"]),
            scheduled_time=row["scheduled_time"],
            status=ContentStatus(row["status"]),
            priority=row["priority"],
            keywords=json.loads(row["keywords"]) if row["keywords"] else [],
            related_topics=json.loads(row["related_topics"]) if row["related_topics"] else [],
            notes=row["notes"] or "",
            video_id=row["video_id"],
            performance_score=row["performance_score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    def cluster_topics(
        self,
        topics: List[str],
        niche: str
    ) -> Dict[str, List[str]]:
        """
        Cluster topics to avoid repetition.

        Args:
            topics: List of topics to cluster
            niche: Content niche

        Returns:
            Dict mapping cluster name to list of topics
        """
        clusters = NICHE_TOPIC_CLUSTERS.get(niche, [])
        result = defaultdict(list)

        for topic in topics:
            topic_lower = topic.lower()
            assigned = False

            for cluster in clusters:
                # Check if topic matches cluster
                if any(t.lower() in topic_lower or topic_lower in t.lower()
                      for t in cluster.topics):
                    result[cluster.name].append(topic)
                    assigned = True
                    break

                # Check keywords
                if any(kw in topic_lower for kw in cluster.related_keywords):
                    result[cluster.name].append(topic)
                    assigned = True
                    break

            if not assigned:
                result["Other"].append(topic)

        return dict(result)

    def check_topic_spacing(
        self,
        channel_id: str,
        topic: str,
        proposed_date: date,
        niche: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if topic spacing is appropriate.

        Args:
            channel_id: Channel identifier
            topic: Topic to schedule
            proposed_date: Proposed date
            niche: Content niche

        Returns:
            Tuple of (is_ok, warning_message)
        """
        # Find cluster for topic
        clusters = NICHE_TOPIC_CLUSTERS.get(niche, [])
        topic_cluster = None
        min_spacing = 14  # Default

        for cluster in clusters:
            if any(t.lower() in topic.lower() or topic.lower() in t.lower()
                  for t in cluster.topics):
                topic_cluster = cluster
                min_spacing = cluster.recommended_spacing_days
                break

        if not topic_cluster:
            return True, None

        # Check recent content in same cluster
        with sqlite3.connect(self.db_path) as conn:
            cutoff = (proposed_date - timedelta(days=min_spacing)).isoformat()
            rows = conn.execute("""
                SELECT scheduled_date, topic FROM content_entries
                WHERE channel_id = ?
                AND scheduled_date >= ?
                AND scheduled_date < ?
                AND content_type = 'video'
            """, (channel_id, cutoff, proposed_date.isoformat())).fetchall()

            for row in rows:
                existing_topic = row[1]
                if any(t.lower() in existing_topic.lower() for t in topic_cluster.topics):
                    scheduled = row[0]
                    return False, f"Similar topic '{existing_topic}' scheduled on {scheduled}. Recommend {min_spacing}+ days spacing."

        return True, None

    async def find_content_gaps(
        self,
        channel_id: str,
        look_back_days: int = 90
    ) -> List[ContentGap]:
        """
        Find gaps in content coverage.

        Args:
            channel_id: Channel identifier
            look_back_days: Days to look back for coverage analysis

        Returns:
            List of ContentGap objects
        """
        logger.info(f"Finding content gaps for {channel_id}")

        channel_config = self.channels_config.get(channel_id, {})
        settings = channel_config.get("settings", {})
        niche = settings.get("niche", "default")

        gaps = []
        cutoff = (date.today() - timedelta(days=look_back_days)).isoformat()

        # Get all covered topics
        with sqlite3.connect(self.db_path) as conn:
            covered = conn.execute("""
                SELECT DISTINCT topic, MAX(scheduled_date) as last_date
                FROM content_entries
                WHERE channel_id = ? AND scheduled_date >= ?
                AND content_type = 'video' AND status != 'cancelled'
                GROUP BY topic
            """, (channel_id, cutoff)).fetchall()

        covered_topics = {row[0].lower(): row[1] for row in covered}

        # Check all cluster topics
        clusters = NICHE_TOPIC_CLUSTERS.get(niche, [])

        for cluster in clusters:
            for topic in cluster.topics:
                topic_lower = topic.lower()

                # Check if covered
                matched = False
                last_date = None
                for covered_topic, covered_date in covered_topics.items():
                    if topic_lower in covered_topic or covered_topic in topic_lower:
                        matched = True
                        last_date = date.fromisoformat(covered_date)
                        break

                if not matched:
                    # Never covered
                    gaps.append(ContentGap(
                        topic=topic,
                        niche=niche,
                        gap_type="never_covered",
                        priority_score=80.0,
                        reason=f"Topic '{topic}' has never been covered",
                    ))
                elif last_date:
                    days_since = (date.today() - last_date).days
                    if days_since > cluster.recommended_spacing_days * 3:
                        # Outdated
                        gaps.append(ContentGap(
                            topic=topic,
                            niche=niche,
                            gap_type="outdated",
                            priority_score=60.0,
                            last_covered=last_date,
                            reason=f"Last covered {days_since} days ago (recommend refresh)",
                        ))

        # Sort by priority
        gaps.sort(key=lambda g: g.priority_score, reverse=True)

        logger.info(f"Found {len(gaps)} content gaps")
        return gaps

    def get_optimal_posting_days(
        self,
        channel_id: str,
        niche: Optional[str] = None
    ) -> List[DayOfWeek]:
        """
        Get optimal posting days for a channel.

        Args:
            channel_id: Channel identifier
            niche: Optional niche override

        Returns:
            List of optimal posting days
        """
        # Check channel config
        channel_config = self.channels_config.get(channel_id, {})
        settings = channel_config.get("settings", {})

        # Use config posting days if available
        if "posting_days" in settings:
            return [DayOfWeek(d) for d in settings["posting_days"]]

        # Fall back to niche defaults
        niche = niche or settings.get("niche", "default")
        day_numbers = NICHE_OPTIMAL_DAYS.get(niche, NICHE_OPTIMAL_DAYS["default"])

        return [DayOfWeek(d) for d in day_numbers]

    def update_entry_status(
        self,
        entry_id: str,
        status: ContentStatus,
        video_id: Optional[str] = None,
        performance_score: Optional[float] = None
    ):
        """Update status of a content entry."""
        with sqlite3.connect(self.db_path) as conn:
            updates = ["status = ?", "updated_at = ?"]
            params = [status.value, datetime.now().isoformat()]

            if video_id:
                updates.append("video_id = ?")
                params.append(video_id)

            if performance_score is not None:
                updates.append("performance_score = ?")
                params.append(performance_score)

            params.append(entry_id)

            conn.execute(f"""
                UPDATE content_entries
                SET {', '.join(updates)}
                WHERE entry_id = ?
            """, params)

        logger.info(f"Updated entry {entry_id} status to {status.value}")

    def export_calendar(
        self,
        calendar: CalendarMonth,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export calendar to various formats.

        Args:
            calendar: CalendarMonth to export
            format: Export format (json, csv, ical, markdown)
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if not output_path:
            output_path = f"output/calendar_{calendar.channel_id}_{calendar.year}_{calendar.month:02d}.{format}"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(calendar.to_dict(), f, indent=2)

        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time", "Title", "Topic", "Type", "Status", "Priority"])
                for entry in calendar.entries:
                    writer.writerow([
                        entry.scheduled_date.isoformat(),
                        entry.scheduled_time or "",
                        entry.title,
                        entry.topic,
                        entry.content_type.value,
                        entry.status.value,
                        entry.priority
                    ])

        elif format == "ical":
            ical_content = self._generate_ical(calendar)
            with open(output_path, 'w') as f:
                f.write(ical_content)

        elif format == "markdown":
            md_content = self._generate_markdown(calendar)
            with open(output_path, 'w') as f:
                f.write(md_content)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Calendar exported to {output_path}")
        return output_path

    def _generate_ical(self, calendar: CalendarMonth) -> str:
        """Generate iCal format."""
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//YouTube Automation//Content Calendar//EN",
            f"X-WR-CALNAME:{calendar.channel_id} Content Calendar",
        ]

        for entry in calendar.entries:
            lines.extend([
                "BEGIN:VEVENT",
                f"UID:{entry.entry_id}@youtube-automation",
                f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}",
                f"DTSTART:{entry.scheduled_date.strftime('%Y%m%d')}",
                f"SUMMARY:{entry.title}",
                f"DESCRIPTION:Topic: {entry.topic}\\nType: {entry.content_type.value}\\nStatus: {entry.status.value}",
                f"CATEGORIES:{entry.content_type.value.upper()}",
                "END:VEVENT",
            ])

        lines.append("END:VCALENDAR")
        return "\r\n".join(lines)

    def _generate_markdown(self, calendar: CalendarMonth) -> str:
        """Generate Markdown format."""
        lines = [
            f"# Content Calendar: {calendar.channel_id}",
            f"## {datetime(calendar.year, calendar.month, 1).strftime('%B %Y')}",
            "",
            f"**Total Content Planned:** {calendar.total_content}",
            f"**Topics Covered:** {len(calendar.topics_covered)}",
            "",
        ]

        for week in calendar.weeks:
            lines.append(f"### Week {week.week_number}")
            lines.append(f"*{week.start_date} - {week.end_date}*")
            lines.append("")
            lines.append("| Date | Title | Type | Status |")
            lines.append("|------|-------|------|--------|")

            for entry in week.entries:
                lines.append(
                    f"| {entry.scheduled_date} | {entry.title} | "
                    f"{entry.content_type.value} | {entry.status.value} |"
                )

            lines.append("")

        return "\n".join(lines)

    def get_calendar_stats(self, channel_id: str, months: int = 3) -> Dict[str, Any]:
        """
        Get calendar statistics.

        Args:
            channel_id: Channel identifier
            months: Number of months to analyze

        Returns:
            Dict with calendar statistics
        """
        cutoff = (date.today() - timedelta(days=months * 30)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Count by status
            status_counts = dict(conn.execute("""
                SELECT status, COUNT(*)
                FROM content_entries
                WHERE channel_id = ? AND scheduled_date >= ?
                GROUP BY status
            """, (channel_id, cutoff)).fetchall())

            # Count by type
            type_counts = dict(conn.execute("""
                SELECT content_type, COUNT(*)
                FROM content_entries
                WHERE channel_id = ? AND scheduled_date >= ?
                GROUP BY content_type
            """, (channel_id, cutoff)).fetchall())

            # Topics frequency
            topic_freq = conn.execute("""
                SELECT topic, COUNT(*) as count
                FROM content_entries
                WHERE channel_id = ? AND scheduled_date >= ?
                AND content_type = 'video'
                GROUP BY topic
                ORDER BY count DESC
                LIMIT 10
            """, (channel_id, cutoff)).fetchall()

        return {
            "channel_id": channel_id,
            "period_months": months,
            "by_status": status_counts,
            "by_type": type_counts,
            "top_topics": [{"topic": t, "count": c} for t, c in topic_freq],
            "generated_at": datetime.now().isoformat(),
        }


# Convenience functions
async def generate_monthly(channel_id: str, month: int, year: int) -> CalendarMonth:
    """Quick function to generate monthly calendar."""
    calendar = ContentCalendar()
    return await calendar.generate_monthly_calendar(channel_id, month, year)


async def find_gaps(channel_id: str) -> List[ContentGap]:
    """Quick function to find content gaps."""
    calendar = ContentCalendar()
    return await calendar.find_content_gaps(channel_id)


if __name__ == "__main__":
    import sys

    async def main():
        print("\n" + "=" * 60)
        print("CONTENT CALENDAR")
        print("=" * 60 + "\n")

        calendar = ContentCalendar()

        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "generate" and len(sys.argv) >= 4:
                channel_id = sys.argv[2]
                month = int(sys.argv[3])
                year = int(sys.argv[4]) if len(sys.argv) > 4 else datetime.now().year

                print(f"Generating calendar for {channel_id}: {month}/{year}\n")
                cal = await calendar.generate_monthly_calendar(channel_id, month, year)

                print(f"Total entries: {cal.total_content}")
                print(f"Topics covered: {len(cal.topics_covered)}")
                print("\nWeekly breakdown:")
                for week in cal.weeks:
                    print(f"  Week {week.week_number}: {week.video_count} videos, {week.short_count} shorts")

                # Export
                export_path = calendar.export_calendar(cal, "markdown")
                print(f"\nExported to: {export_path}")

            elif command == "gaps" and len(sys.argv) >= 3:
                channel_id = sys.argv[2]

                print(f"Finding content gaps for {channel_id}...\n")
                gaps = await calendar.find_content_gaps(channel_id)

                print(f"Found {len(gaps)} gaps:\n")
                for gap in gaps[:10]:
                    print(f"  [{gap.gap_type.upper()}] {gap.topic}")
                    print(f"    Priority: {gap.priority_score:.0f} | {gap.reason}")
                    print()

            elif command == "stats" and len(sys.argv) >= 3:
                channel_id = sys.argv[2]

                stats = calendar.get_calendar_stats(channel_id)
                print(f"Calendar Stats for {channel_id}:\n")
                print(f"By Status: {stats['by_status']}")
                print(f"By Type: {stats['by_type']}")
                print(f"\nTop Topics:")
                for t in stats['top_topics']:
                    print(f"  - {t['topic']}: {t['count']} videos")

            elif command == "optimal" and len(sys.argv) >= 3:
                channel_id = sys.argv[2]

                days = calendar.get_optimal_posting_days(channel_id)
                print(f"Optimal posting days for {channel_id}:")
                for day in days:
                    print(f"  - {day.name}")
            else:
                print("Unknown command")
        else:
            print("Usage:")
            print("  python -m src.scheduler.content_calendar generate <channel_id> <month> [year]")
            print("  python -m src.scheduler.content_calendar gaps <channel_id>")
            print("  python -m src.scheduler.content_calendar stats <channel_id>")
            print("  python -m src.scheduler.content_calendar optimal <channel_id>")

    asyncio.run(main())
