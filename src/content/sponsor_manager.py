"""
Dynamic Sponsor Segment Manager for YouTube Automation

Manage sponsor reads with optimal placement, template-based generation,
and A/B testing for maximum sponsor satisfaction and viewer retention.

Features:
- Template-based sponsor reads
- Auto-insert at optimal times (30%, 50%, or end)
- Track sponsor deliverables
- A/B test sponsor placement impact
- Multiple sponsor types (integrated, dedicated, affiliate)

Usage:
    from src.content.sponsor_manager import SponsorManager

    manager = SponsorManager()

    # Add a sponsor deal
    deal = manager.create_sponsor_deal(
        sponsor_name="Skillshare",
        deal_type="integrated",
        talking_points=["Learn new skills", "1000+ classes"],
        discount_code="MYCHANNEL",
        url="https://skillshare.com/mychannel"
    )

    # Generate sponsor read script
    script = manager.generate_sponsor_read(deal, niche="finance", duration=30)

    # Insert into video script at optimal position
    full_script = manager.insert_sponsor_segment(
        video_script=original_script,
        sponsor_deal=deal,
        placement="optimal"
    )
"""

import os
import json
import random
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
from loguru import logger


class SponsorType(Enum):
    """Types of sponsor integrations."""
    INTEGRATED = "integrated"  # Woven into content
    DEDICATED = "dedicated"  # Standalone segment
    AFFILIATE = "affiliate"  # Affiliate link mention
    PRE_ROLL = "pre_roll"  # At video start
    POST_ROLL = "post_roll"  # At video end
    MID_ROLL = "mid_roll"  # In the middle


class PlacementPosition(Enum):
    """Sponsor placement positions."""
    OPTIMAL = "optimal"  # Algorithm decides
    EARLY = "early"  # ~30% through video
    MIDDLE = "middle"  # ~50% through video
    LATE = "late"  # ~70% through video
    END = "end"  # ~95% through video
    CUSTOM = "custom"  # Custom percentage


@dataclass
class SponsorDeal:
    """Represents a sponsorship deal."""
    deal_id: str
    sponsor_name: str
    sponsor_type: SponsorType
    talking_points: List[str]
    discount_code: Optional[str] = None
    url: Optional[str] = None
    cpm_rate: float = 0.0  # $ per 1000 views
    flat_rate: float = 0.0  # Flat payment
    min_duration_seconds: int = 30
    max_duration_seconds: int = 90
    required_phrases: List[str] = field(default_factory=list)
    forbidden_phrases: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    deliverables: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["sponsor_type"] = self.sponsor_type.value
        result["start_date"] = self.start_date.isoformat() if self.start_date else None
        result["end_date"] = self.end_date.isoformat() if self.end_date else None
        result["created_at"] = self.created_at.isoformat()
        return result

    def is_valid_for_date(self, date: datetime = None) -> bool:
        """Check if deal is valid for a given date."""
        date = date or datetime.now()
        if self.start_date and date < self.start_date:
            return False
        if self.end_date and date > self.end_date:
            return False
        return self.active


@dataclass
class SponsorRead:
    """A generated sponsor read script."""
    deal_id: str
    sponsor_name: str
    script: str
    duration_seconds: int
    placement_position: PlacementPosition
    placement_percentage: float  # 0-100
    includes_discount_code: bool
    includes_url: bool
    word_count: int
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["placement_position"] = self.placement_position.value
        result["generated_at"] = self.generated_at.isoformat()
        return result


@dataclass
class SponsorDeliverable:
    """Track a sponsor deliverable."""
    deliverable_id: str
    deal_id: str
    video_id: Optional[str]
    video_title: str
    upload_date: datetime
    sponsor_segment_start: float  # Seconds into video
    sponsor_segment_duration: float
    views_at_sponsor: int = 0
    clicks: int = 0
    conversions: int = 0
    status: str = "pending"  # pending, delivered, verified
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["upload_date"] = self.upload_date.isoformat()
        return result


@dataclass
class PlacementTest:
    """A/B test for sponsor placement."""
    test_id: str
    deal_id: str
    variant_a_position: PlacementPosition
    variant_b_position: PlacementPosition
    variant_a_videos: List[str] = field(default_factory=list)
    variant_b_videos: List[str] = field(default_factory=list)
    variant_a_retention_impact: float = 0.0
    variant_b_retention_impact: float = 0.0
    variant_a_click_rate: float = 0.0
    variant_b_click_rate: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    winner: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["variant_a_position"] = self.variant_a_position.value
        result["variant_b_position"] = self.variant_b_position.value
        result["started_at"] = self.started_at.isoformat()
        return result


# Sponsor read templates by type and niche
SPONSOR_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "integrated": {
        "finance": [
            "Speaking of building wealth, that's exactly what {sponsor_name} helps you do. {talking_point_1}. {talking_point_2}. Use code {discount_code} for {discount_offer}.",
            "Now, if you're serious about your financial future, you need {sponsor_name}. {talking_point_1}. I've been using it myself, and {personal_experience}. Link in the description.",
            "This brings me to today's sponsor, {sponsor_name}. {talking_point_1}. Whether you're {use_case}, {sponsor_name} makes it easy. {cta}",
        ],
        "psychology": [
            "Want to level up your mind? {sponsor_name} can help. {talking_point_1}. {talking_point_2}. Check them out at {url}.",
            "Speaking of self-improvement, I've partnered with {sponsor_name}. {talking_point_1}. It's perfect for {use_case}. {cta}",
        ],
        "storytelling": [
            "Before we continue, a quick word from {sponsor_name}. {talking_point_1}. {talking_point_2}. Link below.",
            "This video is brought to you by {sponsor_name}. {talking_point_1}. Use my link in the description for {discount_offer}.",
        ],
        "default": [
            "This video is sponsored by {sponsor_name}. {talking_point_1}. {talking_point_2}. Check them out using my link below.",
            "Quick shout-out to {sponsor_name} for sponsoring this video. {talking_point_1}. {cta}",
        ]
    },
    "dedicated": {
        "default": [
            "Now let's take a moment to talk about {sponsor_name}. {talking_point_1}. {talking_point_2}. {talking_point_3}. If you want to {benefit}, go to {url} and use code {discount_code} for {discount_offer}. Again, that's {url}. Now, back to the video.",
            "This video is brought to you by {sponsor_name}. {talking_point_1}. What makes them special is {talking_point_2}. I've personally used {sponsor_name} and {personal_experience}. Head to {url} - link's in the description - and use my code {discount_code} for {discount_offer}. Alright, let's continue.",
        ]
    },
    "affiliate": {
        "default": [
            "By the way, I use {sponsor_name} for {use_case}. If you want to try it, use my link in the description for {discount_offer}.",
            "Shout-out to {sponsor_name}. They're my go-to for {use_case}. Link below if you want to check them out.",
        ]
    }
}

# Transition phrases for natural integration
TRANSITIONS_IN = [
    "Speaking of which,",
    "This brings me to",
    "Now,",
    "Before we move on,",
    "Quick break for",
    "By the way,",
    "This is actually a perfect segue to",
]

TRANSITIONS_OUT = [
    "Now, back to",
    "Alright, let's continue with",
    "So, as I was saying,",
    "Anyway,",
    "Now, where were we?",
    "Let's get back to",
]


class SponsorManager:
    """
    Manage sponsor segments for YouTube videos.

    Handles deal tracking, script generation, optimal placement,
    and deliverable tracking.
    """

    def __init__(self, db_path: str = "data/sponsors.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._deals: Dict[str, SponsorDeal] = {}
        self._tests: Dict[str, PlacementTest] = {}

        self._init_database()
        self._load_deals()

        logger.info("SponsorManager initialized")

    def _init_database(self):
        """Initialize database for sponsor tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sponsor_deals (
                    deal_id TEXT PRIMARY KEY,
                    sponsor_name TEXT NOT NULL,
                    sponsor_type TEXT,
                    talking_points TEXT,
                    discount_code TEXT,
                    url TEXT,
                    cpm_rate REAL,
                    flat_rate REAL,
                    min_duration_seconds INTEGER,
                    max_duration_seconds INTEGER,
                    required_phrases TEXT,
                    forbidden_phrases TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    deliverables TEXT,
                    notes TEXT,
                    active INTEGER,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deliverables (
                    deliverable_id TEXT PRIMARY KEY,
                    deal_id TEXT,
                    video_id TEXT,
                    video_title TEXT,
                    upload_date TEXT,
                    sponsor_segment_start REAL,
                    sponsor_segment_duration REAL,
                    views_at_sponsor INTEGER DEFAULT 0,
                    clicks INTEGER DEFAULT 0,
                    conversions INTEGER DEFAULT 0,
                    status TEXT,
                    notes TEXT,
                    FOREIGN KEY (deal_id) REFERENCES sponsor_deals(deal_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS placement_tests (
                    test_id TEXT PRIMARY KEY,
                    deal_id TEXT,
                    variant_a_position TEXT,
                    variant_b_position TEXT,
                    variant_a_videos TEXT,
                    variant_b_videos TEXT,
                    variant_a_retention_impact REAL,
                    variant_b_retention_impact REAL,
                    variant_a_click_rate REAL,
                    variant_b_click_rate REAL,
                    started_at TEXT,
                    completed INTEGER DEFAULT 0,
                    winner TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deliverables_deal
                ON deliverables(deal_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deliverables_status
                ON deliverables(status)
            """)

    def _load_deals(self):
        """Load active deals from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM sponsor_deals WHERE active = 1").fetchall()

            for row in rows:
                deal = SponsorDeal(
                    deal_id=row["deal_id"],
                    sponsor_name=row["sponsor_name"],
                    sponsor_type=SponsorType(row["sponsor_type"]),
                    talking_points=json.loads(row["talking_points"]),
                    discount_code=row["discount_code"],
                    url=row["url"],
                    cpm_rate=row["cpm_rate"],
                    flat_rate=row["flat_rate"],
                    min_duration_seconds=row["min_duration_seconds"],
                    max_duration_seconds=row["max_duration_seconds"],
                    required_phrases=json.loads(row["required_phrases"] or "[]"),
                    forbidden_phrases=json.loads(row["forbidden_phrases"] or "[]"),
                    start_date=datetime.fromisoformat(row["start_date"]) if row["start_date"] else None,
                    end_date=datetime.fromisoformat(row["end_date"]) if row["end_date"] else None,
                    deliverables=json.loads(row["deliverables"] or "{}"),
                    notes=row["notes"] or "",
                    active=bool(row["active"]),
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                self._deals[deal.deal_id] = deal

        logger.info(f"Loaded {len(self._deals)} active sponsor deals")

    def create_sponsor_deal(
        self,
        sponsor_name: str,
        deal_type: str = "integrated",
        talking_points: List[str] = None,
        discount_code: Optional[str] = None,
        url: Optional[str] = None,
        cpm_rate: float = 0.0,
        flat_rate: float = 0.0,
        min_duration: int = 30,
        max_duration: int = 90,
        required_phrases: List[str] = None,
        forbidden_phrases: List[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        notes: str = ""
    ) -> SponsorDeal:
        """
        Create a new sponsor deal.

        Args:
            sponsor_name: Name of the sponsor
            deal_type: Type of integration (integrated, dedicated, affiliate)
            talking_points: Key points to mention
            discount_code: Promotional code
            url: Sponsor URL
            cpm_rate: Rate per 1000 views
            flat_rate: Flat payment amount
            min_duration: Minimum sponsor segment duration
            max_duration: Maximum sponsor segment duration
            required_phrases: Phrases that must be included
            forbidden_phrases: Phrases to avoid
            start_date: Deal start date
            end_date: Deal end date
            notes: Additional notes

        Returns:
            SponsorDeal object
        """
        import uuid

        deal_id = f"deal_{uuid.uuid4().hex[:8]}"

        deal = SponsorDeal(
            deal_id=deal_id,
            sponsor_name=sponsor_name,
            sponsor_type=SponsorType(deal_type),
            talking_points=talking_points or [],
            discount_code=discount_code,
            url=url,
            cpm_rate=cpm_rate,
            flat_rate=flat_rate,
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            required_phrases=required_phrases or [],
            forbidden_phrases=forbidden_phrases or [],
            start_date=start_date,
            end_date=end_date,
            notes=notes,
            active=True
        )

        # Save to database
        self._save_deal(deal)
        self._deals[deal_id] = deal

        logger.info(f"Created sponsor deal: {sponsor_name} ({deal_id})")
        return deal

    def _save_deal(self, deal: SponsorDeal):
        """Save deal to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sponsor_deals
                (deal_id, sponsor_name, sponsor_type, talking_points, discount_code,
                 url, cpm_rate, flat_rate, min_duration_seconds, max_duration_seconds,
                 required_phrases, forbidden_phrases, start_date, end_date,
                 deliverables, notes, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                deal.deal_id, deal.sponsor_name, deal.sponsor_type.value,
                json.dumps(deal.talking_points), deal.discount_code, deal.url,
                deal.cpm_rate, deal.flat_rate, deal.min_duration_seconds,
                deal.max_duration_seconds, json.dumps(deal.required_phrases),
                json.dumps(deal.forbidden_phrases),
                deal.start_date.isoformat() if deal.start_date else None,
                deal.end_date.isoformat() if deal.end_date else None,
                json.dumps(deal.deliverables), deal.notes, int(deal.active),
                deal.created_at.isoformat()
            ))

    def generate_sponsor_read(
        self,
        deal: SponsorDeal,
        niche: str = "default",
        target_duration: int = 45,
        style: str = "conversational",
        custom_context: Optional[str] = None
    ) -> SponsorRead:
        """
        Generate a sponsor read script.

        Args:
            deal: SponsorDeal to generate for
            niche: Content niche for template selection
            target_duration: Target duration in seconds
            style: Writing style (conversational, professional, casual)
            custom_context: Optional context from the video

        Returns:
            SponsorRead object with generated script
        """
        # Get appropriate templates
        type_key = deal.sponsor_type.value
        if type_key not in SPONSOR_TEMPLATES:
            type_key = "integrated"

        templates = SPONSOR_TEMPLATES[type_key].get(
            niche, SPONSOR_TEMPLATES[type_key].get("default", [])
        )

        if not templates:
            templates = SPONSOR_TEMPLATES["integrated"]["default"]

        # Select template
        template = random.choice(templates)

        # Prepare template variables
        variables = {
            "sponsor_name": deal.sponsor_name,
            "discount_code": deal.discount_code or "in the description",
            "url": deal.url or "the link in the description",
            "cta": f"Use code {deal.discount_code} at checkout" if deal.discount_code else "Check the link in the description",
            "discount_offer": "an exclusive discount" if deal.discount_code else "a special offer",
            "benefit": "level up your skills",
            "use_case": "improving my workflow",
            "personal_experience": "it's been a game-changer",
        }

        # Add talking points
        for i, point in enumerate(deal.talking_points[:5]):
            variables[f"talking_point_{i+1}"] = point

        # Fill missing talking points
        for i in range(1, 6):
            key = f"talking_point_{i}"
            if key not in variables:
                variables[key] = deal.talking_points[0] if deal.talking_points else "They're great"

        # Generate script
        script = template.format(**variables)

        # Add required phrases if not present
        for phrase in deal.required_phrases:
            if phrase.lower() not in script.lower():
                script += f" {phrase}"

        # Add transitions for better flow
        if style == "conversational" and deal.sponsor_type == SponsorType.INTEGRATED:
            transition_in = random.choice(TRANSITIONS_IN)
            script = f"{transition_in} {script}"

        # Estimate duration (150 words per minute average speaking rate)
        word_count = len(script.split())
        estimated_duration = int(word_count / 150 * 60)

        # Adjust if needed
        if estimated_duration < deal.min_duration_seconds:
            # Add more content
            additions = [
                f"I've been using {deal.sponsor_name} for a while now and I really recommend it.",
                f"Whether you're a beginner or advanced, {deal.sponsor_name} has something for you.",
                f"The best part about {deal.sponsor_name} is how easy it is to get started.",
            ]
            script += " " + random.choice(additions)
            word_count = len(script.split())
            estimated_duration = int(word_count / 150 * 60)

        # Determine optimal placement
        placement = self._determine_optimal_placement(deal)

        return SponsorRead(
            deal_id=deal.deal_id,
            sponsor_name=deal.sponsor_name,
            script=script,
            duration_seconds=estimated_duration,
            placement_position=placement,
            placement_percentage=self._get_placement_percentage(placement),
            includes_discount_code=deal.discount_code is not None,
            includes_url=deal.url is not None,
            word_count=word_count
        )

    def _determine_optimal_placement(self, deal: SponsorDeal) -> PlacementPosition:
        """Determine optimal placement based on sponsor type and data."""
        # Default placements by type
        type_placements = {
            SponsorType.INTEGRATED: PlacementPosition.MIDDLE,
            SponsorType.DEDICATED: PlacementPosition.EARLY,
            SponsorType.AFFILIATE: PlacementPosition.LATE,
            SponsorType.PRE_ROLL: PlacementPosition.EARLY,
            SponsorType.POST_ROLL: PlacementPosition.END,
            SponsorType.MID_ROLL: PlacementPosition.MIDDLE,
        }

        # Check if there's A/B test data
        if deal.deal_id in self._tests:
            test = self._tests[deal.deal_id]
            if test.completed and test.winner:
                return PlacementPosition(test.winner)

        return type_placements.get(deal.sponsor_type, PlacementPosition.MIDDLE)

    def _get_placement_percentage(self, position: PlacementPosition) -> float:
        """Get the percentage timestamp for a placement position."""
        percentages = {
            PlacementPosition.OPTIMAL: 50.0,
            PlacementPosition.EARLY: 30.0,
            PlacementPosition.MIDDLE: 50.0,
            PlacementPosition.LATE: 70.0,
            PlacementPosition.END: 95.0,
            PlacementPosition.CUSTOM: 50.0,
        }
        return percentages.get(position, 50.0)

    def insert_sponsor_segment(
        self,
        video_script: str,
        sponsor_deal: SponsorDeal,
        placement: str = "optimal",
        custom_percentage: Optional[float] = None
    ) -> Tuple[str, SponsorRead]:
        """
        Insert sponsor segment into a video script.

        Args:
            video_script: Original video script
            sponsor_deal: Deal to insert
            placement: Where to place (optimal, early, middle, late, end, custom)
            custom_percentage: Custom placement if placement="custom"

        Returns:
            Tuple of (modified script, SponsorRead object)
        """
        # Generate sponsor read
        sponsor_read = self.generate_sponsor_read(sponsor_deal)

        # Override placement if specified
        if placement != "optimal":
            sponsor_read.placement_position = PlacementPosition(placement)
            sponsor_read.placement_percentage = custom_percentage or self._get_placement_percentage(
                sponsor_read.placement_position
            )

        # Split script into sentences/paragraphs
        paragraphs = video_script.split('\n\n')
        total_paragraphs = len(paragraphs)

        if total_paragraphs <= 1:
            # Split by sentences instead
            import re
            sentences = re.split(r'(?<=[.!?])\s+', video_script)
            total_units = len(sentences)
            insert_index = int(total_units * (sponsor_read.placement_percentage / 100))
            insert_index = max(1, min(insert_index, total_units - 1))

            # Insert sponsor read
            sentences.insert(insert_index, f"\n\n{sponsor_read.script}\n\n")
            modified_script = ' '.join(sentences)
        else:
            # Insert between paragraphs
            insert_index = int(total_paragraphs * (sponsor_read.placement_percentage / 100))
            insert_index = max(1, min(insert_index, total_paragraphs - 1))

            # Add transition out
            transition_out = random.choice(TRANSITIONS_OUT)
            sponsor_segment = f"\n\n{sponsor_read.script}\n\n{transition_out}\n\n"

            paragraphs.insert(insert_index, sponsor_segment)
            modified_script = '\n\n'.join(paragraphs)

        return modified_script, sponsor_read

    def track_deliverable(
        self,
        deal_id: str,
        video_id: str,
        video_title: str,
        upload_date: datetime,
        segment_start: float,
        segment_duration: float
    ) -> SponsorDeliverable:
        """
        Track a sponsor deliverable.

        Args:
            deal_id: Sponsor deal ID
            video_id: YouTube video ID
            video_title: Video title
            upload_date: When video was uploaded
            segment_start: Seconds into video where sponsor starts
            segment_duration: Duration of sponsor segment

        Returns:
            SponsorDeliverable object
        """
        import uuid

        deliverable = SponsorDeliverable(
            deliverable_id=f"del_{uuid.uuid4().hex[:8]}",
            deal_id=deal_id,
            video_id=video_id,
            video_title=video_title,
            upload_date=upload_date,
            sponsor_segment_start=segment_start,
            sponsor_segment_duration=segment_duration,
            status="delivered"
        )

        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO deliverables
                (deliverable_id, deal_id, video_id, video_title, upload_date,
                 sponsor_segment_start, sponsor_segment_duration, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                deliverable.deliverable_id, deliverable.deal_id,
                deliverable.video_id, deliverable.video_title,
                deliverable.upload_date.isoformat(),
                deliverable.sponsor_segment_start,
                deliverable.sponsor_segment_duration,
                deliverable.status
            ))

        logger.info(f"Tracked deliverable: {video_title} for deal {deal_id}")
        return deliverable

    def start_placement_test(
        self,
        deal_id: str,
        variant_a: str = "middle",
        variant_b: str = "early"
    ) -> PlacementTest:
        """
        Start an A/B test for sponsor placement.

        Args:
            deal_id: Deal to test
            variant_a: First placement position
            variant_b: Second placement position

        Returns:
            PlacementTest object
        """
        import uuid

        test = PlacementTest(
            test_id=f"test_{uuid.uuid4().hex[:8]}",
            deal_id=deal_id,
            variant_a_position=PlacementPosition(variant_a),
            variant_b_position=PlacementPosition(variant_b)
        )

        self._tests[deal_id] = test

        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO placement_tests
                (test_id, deal_id, variant_a_position, variant_b_position,
                 variant_a_videos, variant_b_videos, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                test.test_id, test.deal_id,
                test.variant_a_position.value, test.variant_b_position.value,
                json.dumps([]), json.dumps([]),
                test.started_at.isoformat()
            ))

        logger.info(f"Started placement test: {variant_a} vs {variant_b}")
        return test

    def record_test_result(
        self,
        deal_id: str,
        video_id: str,
        variant: str,
        retention_impact: float,
        click_rate: float
    ):
        """Record results for a placement test variant."""
        if deal_id not in self._tests:
            logger.warning(f"No active test for deal {deal_id}")
            return

        test = self._tests[deal_id]

        if variant == "a":
            test.variant_a_videos.append(video_id)
            # Running average
            n = len(test.variant_a_videos)
            test.variant_a_retention_impact = (
                (test.variant_a_retention_impact * (n - 1) + retention_impact) / n
            )
            test.variant_a_click_rate = (
                (test.variant_a_click_rate * (n - 1) + click_rate) / n
            )
        else:
            test.variant_b_videos.append(video_id)
            n = len(test.variant_b_videos)
            test.variant_b_retention_impact = (
                (test.variant_b_retention_impact * (n - 1) + retention_impact) / n
            )
            test.variant_b_click_rate = (
                (test.variant_b_click_rate * (n - 1) + click_rate) / n
            )

        # Check if test should conclude
        if len(test.variant_a_videos) >= 5 and len(test.variant_b_videos) >= 5:
            self._conclude_test(test)

    def _conclude_test(self, test: PlacementTest):
        """Conclude a placement test and determine winner."""
        # Score based on retention impact and click rate
        # Lower retention impact is better (less drop), higher click rate is better
        score_a = (100 - test.variant_a_retention_impact) * 0.7 + test.variant_a_click_rate * 0.3
        score_b = (100 - test.variant_b_retention_impact) * 0.7 + test.variant_b_click_rate * 0.3

        if score_a > score_b:
            test.winner = test.variant_a_position.value
        else:
            test.winner = test.variant_b_position.value

        test.completed = True

        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE placement_tests
                SET completed = 1, winner = ?,
                    variant_a_retention_impact = ?, variant_b_retention_impact = ?,
                    variant_a_click_rate = ?, variant_b_click_rate = ?,
                    variant_a_videos = ?, variant_b_videos = ?
                WHERE test_id = ?
            """, (
                test.winner,
                test.variant_a_retention_impact, test.variant_b_retention_impact,
                test.variant_a_click_rate, test.variant_b_click_rate,
                json.dumps(test.variant_a_videos), json.dumps(test.variant_b_videos),
                test.test_id
            ))

        logger.success(f"Test concluded: Winner is {test.winner}")

    def get_next_placement_for_test(self, deal_id: str) -> PlacementPosition:
        """Get next placement variant for an active test."""
        if deal_id not in self._tests:
            return PlacementPosition.OPTIMAL

        test = self._tests[deal_id]
        if test.completed:
            return PlacementPosition(test.winner)

        # Alternate between variants
        if len(test.variant_a_videos) <= len(test.variant_b_videos):
            return test.variant_a_position
        return test.variant_b_position

    def get_deal(self, deal_id: str) -> Optional[SponsorDeal]:
        """Get a sponsor deal by ID."""
        return self._deals.get(deal_id)

    def get_active_deals(self, for_date: datetime = None) -> List[SponsorDeal]:
        """Get all active deals valid for a date."""
        for_date = for_date or datetime.now()
        return [d for d in self._deals.values() if d.is_valid_for_date(for_date)]

    def get_deal_deliverables(self, deal_id: str) -> List[SponsorDeliverable]:
        """Get all deliverables for a deal."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM deliverables WHERE deal_id = ? ORDER BY upload_date DESC",
                (deal_id,)
            ).fetchall()

            return [
                SponsorDeliverable(
                    deliverable_id=row["deliverable_id"],
                    deal_id=row["deal_id"],
                    video_id=row["video_id"],
                    video_title=row["video_title"],
                    upload_date=datetime.fromisoformat(row["upload_date"]),
                    sponsor_segment_start=row["sponsor_segment_start"],
                    sponsor_segment_duration=row["sponsor_segment_duration"],
                    views_at_sponsor=row["views_at_sponsor"],
                    clicks=row["clicks"],
                    conversions=row["conversions"],
                    status=row["status"],
                    notes=row["notes"] or ""
                )
                for row in rows
            ]

    def get_sponsor_report(self, deal_id: str = None) -> Dict[str, Any]:
        """Generate a sponsor performance report."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if deal_id:
                deals_filter = "WHERE d.deal_id = ?"
                params = (deal_id,)
            else:
                deals_filter = "WHERE d.active = 1"
                params = ()

            # Get deal summary
            deal_data = conn.execute(f"""
                SELECT d.*, COUNT(del.deliverable_id) as deliverable_count
                FROM sponsor_deals d
                LEFT JOIN deliverables del ON d.deal_id = del.deal_id
                {deals_filter}
                GROUP BY d.deal_id
            """, params).fetchall()

            # Get performance metrics
            performance = conn.execute(f"""
                SELECT
                    d.deal_id,
                    d.sponsor_name,
                    SUM(del.views_at_sponsor) as total_views,
                    SUM(del.clicks) as total_clicks,
                    SUM(del.conversions) as total_conversions,
                    AVG(del.sponsor_segment_duration) as avg_duration
                FROM sponsor_deals d
                LEFT JOIN deliverables del ON d.deal_id = del.deal_id
                {deals_filter}
                GROUP BY d.deal_id
            """, params).fetchall()

        deals_summary = []
        for deal in deal_data:
            perf = next((p for p in performance if p["deal_id"] == deal["deal_id"]), None)
            deals_summary.append({
                "deal_id": deal["deal_id"],
                "sponsor_name": deal["sponsor_name"],
                "type": deal["sponsor_type"],
                "deliverable_count": deal["deliverable_count"],
                "total_views": perf["total_views"] if perf else 0,
                "total_clicks": perf["total_clicks"] if perf else 0,
                "click_rate": (
                    (perf["total_clicks"] / perf["total_views"] * 100)
                    if perf and perf["total_views"] else 0
                ),
                "total_conversions": perf["total_conversions"] if perf else 0,
                "avg_segment_duration": perf["avg_duration"] if perf else 0,
            })

        return {
            "generated_at": datetime.now().isoformat(),
            "total_deals": len(deals_summary),
            "deals": deals_summary,
        }

    def deactivate_deal(self, deal_id: str):
        """Deactivate a sponsor deal."""
        if deal_id in self._deals:
            self._deals[deal_id].active = False
            self._save_deal(self._deals[deal_id])
            del self._deals[deal_id]
            logger.info(f"Deactivated deal {deal_id}")


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("SPONSOR MANAGER")
    print("=" * 60 + "\n")

    manager = SponsorManager()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "create":
            deal = manager.create_sponsor_deal(
                sponsor_name="TestSponsor",
                deal_type="integrated",
                talking_points=[
                    "Learn any skill online",
                    "Access to 1000+ courses",
                    "Start for free"
                ],
                discount_code="TESTCODE",
                url="https://example.com/sponsor"
            )
            print(f"Created deal: {deal.deal_id}")
            print(f"Sponsor: {deal.sponsor_name}")

        elif command == "generate" and len(sys.argv) >= 3:
            deal_id = sys.argv[2]
            niche = sys.argv[3] if len(sys.argv) > 3 else "default"

            deal = manager.get_deal(deal_id)
            if deal:
                read = manager.generate_sponsor_read(deal, niche=niche)
                print(f"Generated sponsor read for {deal.sponsor_name}:\n")
                print(read.script)
                print(f"\nEstimated duration: {read.duration_seconds} seconds")
                print(f"Placement: {read.placement_position.value} ({read.placement_percentage}%)")
            else:
                print(f"Deal not found: {deal_id}")

        elif command == "list":
            deals = manager.get_active_deals()
            print(f"Active sponsor deals ({len(deals)}):\n")
            for deal in deals:
                print(f"  {deal.deal_id}: {deal.sponsor_name} ({deal.sponsor_type.value})")
                print(f"    Talking points: {len(deal.talking_points)}")
                print(f"    Code: {deal.discount_code or 'None'}")
                print()

        elif command == "report":
            deal_id = sys.argv[2] if len(sys.argv) > 2 else None
            report = manager.get_sponsor_report(deal_id)
            print(f"Sponsor Report ({report['total_deals']} deals):\n")
            for deal in report["deals"]:
                print(f"  {deal['sponsor_name']}:")
                print(f"    Deliverables: {deal['deliverable_count']}")
                print(f"    Total views: {deal['total_views']:,}")
                print(f"    Click rate: {deal['click_rate']:.2f}%")
                print()
        else:
            print("Unknown command")
    else:
        print("Usage:")
        print("  python -m src.content.sponsor_manager create")
        print("  python -m src.content.sponsor_manager generate <deal_id> [niche]")
        print("  python -m src.content.sponsor_manager list")
        print("  python -m src.content.sponsor_manager report [deal_id]")
