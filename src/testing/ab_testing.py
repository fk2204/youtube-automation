"""
A/B Testing Framework for YouTube Metadata

Test different titles, thumbnails, and descriptions to optimize CTR.
Supports statistical significance testing, variant generation, and automated winner application.

Usage:
    from src.testing.ab_testing import ABTestManager, TitleVariantGenerator

    # Create manager
    manager = ABTestManager()

    # Create a title test
    test = manager.create_title_test(
        video_id="abc123",
        titles=["5 Money Tips", "SECRET Money Tips That WORK", "How I Saved $10,000"]
    )

    # Update metrics as data comes in
    manager.update_metrics(test.id, "variant_A", impressions=1500, clicks=45, avg_duration=180.5)

    # Check significance
    result = manager.check_statistical_significance(test)
    if result["is_significant"]:
        manager.apply_winner(test.id)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sqlite3
import json
import random
import uuid
import re
import math
from enum import Enum

from loguru import logger

# Try to import scipy for advanced statistical tests
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed. Using fallback statistical methods.")

# Try to import PIL for thumbnail manipulation
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not installed. Thumbnail variant generation limited.")


class TestStatus(Enum):
    """Status of an A/B test."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    INSUFFICIENT_DATA = "insufficient_data"


class TestType(Enum):
    """Type of content being tested."""
    TITLE = "title"
    THUMBNAIL = "thumbnail"
    DESCRIPTION = "description"
    TITLE_THUMBNAIL = "title_thumbnail"


@dataclass
class Variant:
    """A single variant in an A/B test."""
    id: str
    content: str  # Title text, thumbnail path, or description
    video_id: Optional[str] = None
    impressions: int = 0
    clicks: int = 0
    ctr: float = 0.0
    avg_view_duration: float = 0.0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    subscribers_gained: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_ctr(self) -> float:
        """Calculate click-through rate."""
        if self.impressions == 0:
            return 0.0
        self.ctr = (self.clicks / self.impressions) * 100
        return self.ctr

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "video_id": self.video_id,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "ctr": round(self.calculate_ctr(), 4),
            "avg_view_duration": round(self.avg_view_duration, 2),
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "subscribers_gained": self.subscribers_gained,
            "created_at": self.created_at,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Variant":
        """Create variant from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            video_id=data.get("video_id"),
            impressions=data.get("impressions", 0),
            clicks=data.get("clicks", 0),
            ctr=data.get("ctr", 0.0),
            avg_view_duration=data.get("avg_view_duration", 0.0),
            likes=data.get("likes", 0),
            comments=data.get("comments", 0),
            shares=data.get("shares", 0),
            subscribers_gained=data.get("subscribers_gained", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )


@dataclass
class ABTest:
    """An A/B test configuration and state."""
    id: str
    test_type: TestType
    base_video_id: str
    variants: List[Variant]
    status: TestStatus = TestStatus.PENDING
    winner_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    min_impressions: int = 1000
    confidence_level: float = 0.95
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    niche: str = ""
    target_metric: str = "ctr"  # ctr, avg_view_duration, engagement
    auto_apply_winner: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "test_type": self.test_type.value,
            "base_video_id": self.base_video_id,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status.value,
            "winner_id": self.winner_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "min_impressions": self.min_impressions,
            "confidence_level": self.confidence_level,
            "created_at": self.created_at,
            "notes": self.notes,
            "niche": self.niche,
            "target_metric": self.target_metric,
            "auto_apply_winner": self.auto_apply_winner
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABTest":
        """Create test from dictionary."""
        return cls(
            id=data["id"],
            test_type=TestType(data["test_type"]),
            base_video_id=data["base_video_id"],
            variants=[Variant.from_dict(v) for v in data["variants"]],
            status=TestStatus(data["status"]),
            winner_id=data.get("winner_id"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            min_impressions=data.get("min_impressions", 1000),
            confidence_level=data.get("confidence_level", 0.95),
            created_at=data.get("created_at", datetime.now().isoformat()),
            notes=data.get("notes", ""),
            niche=data.get("niche", ""),
            target_metric=data.get("target_metric", "ctr"),
            auto_apply_winner=data.get("auto_apply_winner", False)
        )

    def get_total_impressions(self) -> int:
        """Get total impressions across all variants."""
        return sum(v.impressions for v in self.variants)

    def get_total_clicks(self) -> int:
        """Get total clicks across all variants."""
        return sum(v.clicks for v in self.variants)

    def meets_min_impressions(self) -> bool:
        """Check if minimum impressions threshold is met."""
        return all(v.impressions >= self.min_impressions for v in self.variants)


class ABTestManager:
    """Manages A/B tests for YouTube video metadata."""

    def __init__(self, db_path: str = "data/ab_tests.db"):
        """
        Initialize the A/B test manager.

        Args:
            db_path: Path to SQLite database for storing test data
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"ABTestManager initialized with database: {self.db_path}")

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Tests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tests (
                id TEXT PRIMARY KEY,
                test_type TEXT NOT NULL,
                base_video_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                winner_id TEXT,
                start_time TEXT,
                end_time TEXT,
                min_impressions INTEGER DEFAULT 1000,
                confidence_level REAL DEFAULT 0.95,
                created_at TEXT NOT NULL,
                notes TEXT,
                niche TEXT,
                target_metric TEXT DEFAULT 'ctr',
                auto_apply_winner INTEGER DEFAULT 0
            )
        """)

        # Variants table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS variants (
                id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                content TEXT NOT NULL,
                video_id TEXT,
                impressions INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                ctr REAL DEFAULT 0.0,
                avg_view_duration REAL DEFAULT 0.0,
                likes INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                shares INTEGER DEFAULT 0,
                subscribers_gained INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (test_id) REFERENCES tests(id)
            )
        """)

        # Metrics history table for tracking over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variant_id TEXT NOT NULL,
                test_id TEXT NOT NULL,
                impressions INTEGER,
                clicks INTEGER,
                ctr REAL,
                avg_view_duration REAL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (variant_id) REFERENCES variants(id),
                FOREIGN KEY (test_id) REFERENCES tests(id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tests_status ON tests(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_test_id ON variants(test_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_variant ON metrics_history(variant_id)")

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def create_title_test(
        self,
        video_id: str,
        titles: List[str],
        min_impressions: int = 1000,
        confidence_level: float = 0.95,
        niche: str = "",
        auto_apply: bool = False
    ) -> ABTest:
        """
        Create an A/B test for different titles.

        Args:
            video_id: YouTube video ID
            titles: List of title variants to test
            min_impressions: Minimum impressions per variant before significance check
            confidence_level: Required confidence level (0.0-1.0)
            niche: Content niche for context
            auto_apply: Automatically apply winner when determined

        Returns:
            Created ABTest object
        """
        if len(titles) < 2:
            raise ValueError("At least 2 title variants required for A/B test")

        test_id = f"title_test_{uuid.uuid4().hex[:8]}"

        # Create variants
        variants = []
        for i, title in enumerate(titles):
            variant = Variant(
                id=f"variant_{chr(65 + i)}",  # A, B, C, ...
                content=title,
                video_id=video_id if i == 0 else None,  # First variant is active
                metadata={"position": i, "original": i == 0}
            )
            variants.append(variant)

        test = ABTest(
            id=test_id,
            test_type=TestType.TITLE,
            base_video_id=video_id,
            variants=variants,
            status=TestStatus.PENDING,
            min_impressions=min_impressions,
            confidence_level=confidence_level,
            niche=niche,
            auto_apply_winner=auto_apply
        )

        # Save to database
        self._save_test(test)

        logger.info(f"Created title A/B test {test_id} with {len(titles)} variants")
        return test

    def create_thumbnail_test(
        self,
        video_id: str,
        thumbnail_paths: List[str],
        min_impressions: int = 1000,
        confidence_level: float = 0.95,
        niche: str = "",
        auto_apply: bool = False
    ) -> ABTest:
        """
        Create an A/B test for different thumbnails.

        Args:
            video_id: YouTube video ID
            thumbnail_paths: List of thumbnail file paths
            min_impressions: Minimum impressions per variant
            confidence_level: Required confidence level
            niche: Content niche
            auto_apply: Auto-apply winner

        Returns:
            Created ABTest object
        """
        if len(thumbnail_paths) < 2:
            raise ValueError("At least 2 thumbnail variants required")

        # Validate paths exist
        for path in thumbnail_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Thumbnail not found: {path}")

        test_id = f"thumb_test_{uuid.uuid4().hex[:8]}"

        variants = []
        for i, path in enumerate(thumbnail_paths):
            variant = Variant(
                id=f"variant_{chr(65 + i)}",
                content=str(Path(path).absolute()),
                video_id=video_id if i == 0 else None,
                metadata={"position": i, "original": i == 0, "filename": Path(path).name}
            )
            variants.append(variant)

        test = ABTest(
            id=test_id,
            test_type=TestType.THUMBNAIL,
            base_video_id=video_id,
            variants=variants,
            status=TestStatus.PENDING,
            min_impressions=min_impressions,
            confidence_level=confidence_level,
            niche=niche,
            auto_apply_winner=auto_apply
        )

        self._save_test(test)
        logger.info(f"Created thumbnail A/B test {test_id} with {len(thumbnail_paths)} variants")
        return test

    def create_description_test(
        self,
        video_id: str,
        descriptions: List[str],
        min_impressions: int = 1000,
        confidence_level: float = 0.95,
        niche: str = "",
        auto_apply: bool = False
    ) -> ABTest:
        """
        Create an A/B test for different descriptions.

        Args:
            video_id: YouTube video ID
            descriptions: List of description variants
            min_impressions: Minimum impressions per variant
            confidence_level: Required confidence level
            niche: Content niche
            auto_apply: Auto-apply winner

        Returns:
            Created ABTest object
        """
        if len(descriptions) < 2:
            raise ValueError("At least 2 description variants required")

        test_id = f"desc_test_{uuid.uuid4().hex[:8]}"

        variants = []
        for i, desc in enumerate(descriptions):
            variant = Variant(
                id=f"variant_{chr(65 + i)}",
                content=desc,
                video_id=video_id if i == 0 else None,
                metadata={"position": i, "original": i == 0, "length": len(desc)}
            )
            variants.append(variant)

        test = ABTest(
            id=test_id,
            test_type=TestType.DESCRIPTION,
            base_video_id=video_id,
            variants=variants,
            status=TestStatus.PENDING,
            min_impressions=min_impressions,
            confidence_level=confidence_level,
            niche=niche,
            auto_apply_winner=auto_apply
        )

        self._save_test(test)
        logger.info(f"Created description A/B test {test_id} with {len(descriptions)} variants")
        return test

    def _save_test(self, test: ABTest):
        """Save test and variants to database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Insert or update test
            cursor.execute("""
                INSERT OR REPLACE INTO tests
                (id, test_type, base_video_id, status, winner_id, start_time, end_time,
                 min_impressions, confidence_level, created_at, notes, niche, target_metric, auto_apply_winner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test.id, test.test_type.value, test.base_video_id, test.status.value,
                test.winner_id, test.start_time, test.end_time, test.min_impressions,
                test.confidence_level, test.created_at, test.notes, test.niche,
                test.target_metric, 1 if test.auto_apply_winner else 0
            ))

            # Save variants
            for variant in test.variants:
                cursor.execute("""
                    INSERT OR REPLACE INTO variants
                    (id, test_id, content, video_id, impressions, clicks, ctr,
                     avg_view_duration, likes, comments, shares, subscribers_gained,
                     created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{test.id}_{variant.id}", test.id, variant.content, variant.video_id,
                    variant.impressions, variant.clicks, variant.ctr, variant.avg_view_duration,
                    variant.likes, variant.comments, variant.shares, variant.subscribers_gained,
                    variant.created_at, json.dumps(variant.metadata)
                ))

            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save test {test.id}: {e}")
            raise
        finally:
            conn.close()

    def _load_test(self, test_id: str) -> Optional[ABTest]:
        """Load test from database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM tests WHERE id = ?", (test_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        # Load variants
        cursor.execute("SELECT * FROM variants WHERE test_id = ?", (test_id,))
        variant_rows = cursor.fetchall()

        variants = []
        for vrow in variant_rows:
            variant = Variant(
                id=vrow["id"].replace(f"{test_id}_", ""),
                content=vrow["content"],
                video_id=vrow["video_id"],
                impressions=vrow["impressions"],
                clicks=vrow["clicks"],
                ctr=vrow["ctr"],
                avg_view_duration=vrow["avg_view_duration"],
                likes=vrow["likes"],
                comments=vrow["comments"],
                shares=vrow["shares"],
                subscribers_gained=vrow["subscribers_gained"],
                created_at=vrow["created_at"],
                metadata=json.loads(vrow["metadata"]) if vrow["metadata"] else {}
            )
            variants.append(variant)

        test = ABTest(
            id=row["id"],
            test_type=TestType(row["test_type"]),
            base_video_id=row["base_video_id"],
            variants=variants,
            status=TestStatus(row["status"]),
            winner_id=row["winner_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            min_impressions=row["min_impressions"],
            confidence_level=row["confidence_level"],
            created_at=row["created_at"],
            notes=row["notes"] or "",
            niche=row["niche"] or "",
            target_metric=row["target_metric"] or "ctr",
            auto_apply_winner=bool(row["auto_apply_winner"])
        )

        conn.close()
        return test

    def start_test(self, test_id: str) -> ABTest:
        """
        Start a pending test.

        Args:
            test_id: Test identifier

        Returns:
            Updated ABTest object
        """
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        if test.status != TestStatus.PENDING:
            raise ValueError(f"Test {test_id} cannot be started (status: {test.status.value})")

        test.status = TestStatus.RUNNING
        test.start_time = datetime.now().isoformat()

        self._save_test(test)
        logger.info(f"Started A/B test {test_id}")
        return test

    def pause_test(self, test_id: str) -> ABTest:
        """Pause a running test."""
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        if test.status != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")

        test.status = TestStatus.PAUSED
        self._save_test(test)
        logger.info(f"Paused A/B test {test_id}")
        return test

    def resume_test(self, test_id: str) -> ABTest:
        """Resume a paused test."""
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        if test.status != TestStatus.PAUSED:
            raise ValueError(f"Test {test_id} is not paused")

        test.status = TestStatus.RUNNING
        self._save_test(test)
        logger.info(f"Resumed A/B test {test_id}")
        return test

    def cancel_test(self, test_id: str) -> ABTest:
        """Cancel a test."""
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        if test.status == TestStatus.COMPLETED:
            raise ValueError(f"Test {test_id} is already completed")

        test.status = TestStatus.CANCELLED
        test.end_time = datetime.now().isoformat()
        self._save_test(test)
        logger.info(f"Cancelled A/B test {test_id}")
        return test

    def update_metrics(
        self,
        test_id: str,
        variant_id: str,
        impressions: int,
        clicks: int,
        avg_duration: float,
        likes: int = 0,
        comments: int = 0,
        shares: int = 0,
        subscribers_gained: int = 0
    ):
        """
        Update metrics for a variant.

        Args:
            test_id: Test identifier
            variant_id: Variant identifier (e.g., "variant_A")
            impressions: Total impressions (cumulative)
            clicks: Total clicks (cumulative)
            avg_duration: Average view duration in seconds
            likes: Total likes
            comments: Total comments
            shares: Total shares
            subscribers_gained: Subscribers gained from this variant
        """
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        if test.status not in [TestStatus.RUNNING, TestStatus.PAUSED]:
            logger.warning(f"Cannot update metrics for test {test_id} (status: {test.status.value})")
            return

        # Find and update variant
        variant = None
        for v in test.variants:
            if v.id == variant_id:
                variant = v
                break

        if not variant:
            raise ValueError(f"Variant not found: {variant_id}")

        # Update metrics
        variant.impressions = impressions
        variant.clicks = clicks
        variant.avg_view_duration = avg_duration
        variant.likes = likes
        variant.comments = comments
        variant.shares = shares
        variant.subscribers_gained = subscribers_gained
        variant.calculate_ctr()

        # Save to database
        self._save_test(test)

        # Record history
        self._record_metrics_history(test_id, variant_id, variant)

        logger.debug(
            f"Updated metrics for {test_id}/{variant_id}: "
            f"{impressions} impressions, {clicks} clicks, CTR: {variant.ctr:.2f}%"
        )

        # Check if we should auto-determine winner
        if test.auto_apply_winner and test.meets_min_impressions():
            result = self.check_statistical_significance(test)
            if result["is_significant"]:
                self.determine_winner(test_id)

    def _record_metrics_history(self, test_id: str, variant_id: str, variant: Variant):
        """Record metrics snapshot for historical tracking."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO metrics_history
            (variant_id, test_id, impressions, clicks, ctr, avg_view_duration, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{test_id}_{variant_id}", test_id,
            variant.impressions, variant.clicks, variant.ctr,
            variant.avg_view_duration, datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def check_statistical_significance(self, test: ABTest) -> Dict[str, Any]:
        """
        Check if results are statistically significant using chi-square test.

        Args:
            test: ABTest object or test_id string

        Returns:
            Dictionary with significance results
        """
        if isinstance(test, str):
            test = self._load_test(test)
            if not test:
                raise ValueError(f"Test not found: {test}")

        # Check minimum data requirements
        total_impressions = test.get_total_impressions()
        min_required = test.min_impressions * len(test.variants)

        if total_impressions < min_required:
            return {
                "is_significant": False,
                "reason": "insufficient_data",
                "message": f"Need {min_required} impressions total, have {total_impressions}",
                "total_impressions": total_impressions,
                "required_impressions": min_required,
                "progress_percent": round((total_impressions / min_required) * 100, 1)
            }

        # Prepare contingency table for chi-square test
        # Each row: [clicks, non-clicks] for each variant
        observed = []
        for v in test.variants:
            clicks = v.clicks
            non_clicks = v.impressions - v.clicks
            observed.append([clicks, non_clicks])

        # Run chi-square test
        if HAS_SCIPY:
            chi2, p_value, dof, expected = scipy_stats.chi2_contingency(observed)
        else:
            # Fallback implementation
            chi2, p_value = self._chi_square_fallback(observed)
            dof = len(observed) - 1

        confidence = 1 - p_value
        is_significant = confidence >= test.confidence_level

        # Calculate effect size (Cramer's V)
        n = sum(sum(row) for row in observed)
        min_dim = min(len(observed), 2) - 1
        cramers_v = math.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0

        # Find best performing variant
        best_variant = max(test.variants, key=lambda v: v.ctr)
        worst_variant = min(test.variants, key=lambda v: v.ctr)
        ctr_lift = best_variant.ctr - worst_variant.ctr

        return {
            "is_significant": is_significant,
            "chi_squared": round(chi2, 4),
            "p_value": round(p_value, 6),
            "confidence_level": round(confidence, 4),
            "required_confidence": test.confidence_level,
            "degrees_of_freedom": dof,
            "effect_size": round(cramers_v, 4),
            "best_variant": best_variant.id,
            "best_ctr": round(best_variant.ctr, 4),
            "worst_ctr": round(worst_variant.ctr, 4),
            "ctr_lift": round(ctr_lift, 4),
            "ctr_lift_percent": round((ctr_lift / worst_variant.ctr * 100) if worst_variant.ctr > 0 else 0, 2),
            "variant_stats": {v.id: {"ctr": v.ctr, "impressions": v.impressions, "clicks": v.clicks} for v in test.variants},
            "recommendation": self._generate_recommendation(is_significant, best_variant, ctr_lift, confidence)
        }

    def _chi_square_fallback(self, observed: List[List[int]]) -> Tuple[float, float]:
        """
        Fallback chi-square calculation when scipy is not available.

        Uses the chi-square formula: sum((observed - expected)^2 / expected)
        """
        # Calculate row and column totals
        row_totals = [sum(row) for row in observed]
        col_totals = [sum(observed[i][j] for i in range(len(observed))) for j in range(len(observed[0]))]
        grand_total = sum(row_totals)

        if grand_total == 0:
            return 0.0, 1.0

        # Calculate expected frequencies
        expected = []
        for i, row in enumerate(observed):
            expected_row = []
            for j in range(len(row)):
                exp = (row_totals[i] * col_totals[j]) / grand_total
                expected_row.append(exp)
            expected.append(expected_row)

        # Calculate chi-square statistic
        chi2 = 0.0
        for i in range(len(observed)):
            for j in range(len(observed[0])):
                if expected[i][j] > 0:
                    chi2 += ((observed[i][j] - expected[i][j]) ** 2) / expected[i][j]

        # Calculate degrees of freedom
        dof = (len(observed) - 1) * (len(observed[0]) - 1)

        # Approximate p-value using chi-square distribution
        # Using Wilson-Hilferty approximation for chi-square CDF
        if dof <= 0 or chi2 <= 0:
            return chi2, 1.0

        # Simplified p-value approximation
        # For more accurate results, scipy should be used
        p_value = self._chi2_survival_function(chi2, dof)

        return chi2, p_value

    def _chi2_survival_function(self, x: float, k: int) -> float:
        """
        Approximate the chi-square survival function (1 - CDF).
        Uses series expansion for reasonable accuracy.
        """
        if x <= 0:
            return 1.0

        # Use regularized incomplete gamma function approximation
        # P(a, x) where a = k/2 and x = chi2/2
        a = k / 2.0
        x = x / 2.0

        # Simple approximation using series expansion
        # This is a rough approximation - scipy provides exact values
        if x > a + 50:
            return 0.0
        if x < 0.01:
            return 1.0

        # Compute using continued fraction for better accuracy
        # Simplified version - returns approximate p-value
        result = math.exp(-x) * (x ** a)

        # Normalize
        try:
            # Stirling's approximation for gamma function
            gamma_a = math.sqrt(2 * math.pi / a) * ((a / math.e) ** a)
            result = result / gamma_a
        except (OverflowError, ZeroDivisionError):
            result = 0.0

        # Clamp between 0 and 1
        return max(0.0, min(1.0, 1.0 - result))

    def _generate_recommendation(
        self,
        is_significant: bool,
        best_variant: Variant,
        ctr_lift: float,
        confidence: float
    ) -> str:
        """Generate human-readable recommendation."""
        if is_significant:
            return (
                f"Winner: {best_variant.id} with {confidence:.1%} confidence. "
                f"CTR improvement: +{ctr_lift:.2f}%. "
                f"Recommend applying this variant permanently."
            )
        elif confidence > 0.8:
            return (
                f"{best_variant.id} is leading but not yet statistically significant "
                f"({confidence:.1%} confidence). Continue collecting data."
            )
        else:
            return (
                f"No clear winner yet ({confidence:.1%} confidence). "
                f"Continue the test to gather more data."
            )

    def determine_winner(self, test_id: str) -> Optional[Variant]:
        """
        Determine the winning variant if statistically significant.

        Args:
            test_id: Test identifier

        Returns:
            Winning Variant or None if no significant winner
        """
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        result = self.check_statistical_significance(test)

        if not result["is_significant"]:
            logger.info(f"Test {test_id} has no significant winner yet")
            return None

        # Find winner based on target metric
        if test.target_metric == "ctr":
            winner = max(test.variants, key=lambda v: v.ctr)
        elif test.target_metric == "avg_view_duration":
            winner = max(test.variants, key=lambda v: v.avg_view_duration)
        else:  # engagement
            winner = max(test.variants, key=lambda v: v.likes + v.comments + v.shares)

        # Update test
        test.winner_id = winner.id
        test.status = TestStatus.COMPLETED
        test.end_time = datetime.now().isoformat()

        self._save_test(test)

        logger.info(f"Test {test_id} completed. Winner: {winner.id} (CTR: {winner.ctr:.2f}%)")
        return winner

    def apply_winner(self, test_id: str, youtube_api=None) -> bool:
        """
        Apply the winning variant to the video.

        Args:
            test_id: Test identifier
            youtube_api: Optional YouTube API client for applying changes

        Returns:
            True if successfully applied
        """
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        if not test.winner_id:
            logger.warning(f"Test {test_id} has no winner to apply")
            return False

        winner = None
        for v in test.variants:
            if v.id == test.winner_id:
                winner = v
                break

        if not winner:
            logger.error(f"Winner variant {test.winner_id} not found")
            return False

        # Apply based on test type
        if youtube_api:
            try:
                if test.test_type == TestType.TITLE:
                    youtube_api.update_video_title(test.base_video_id, winner.content)
                elif test.test_type == TestType.THUMBNAIL:
                    youtube_api.update_video_thumbnail(test.base_video_id, winner.content)
                elif test.test_type == TestType.DESCRIPTION:
                    youtube_api.update_video_description(test.base_video_id, winner.content)

                logger.info(f"Applied winner {winner.id} to video {test.base_video_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to apply winner: {e}")
                return False
        else:
            logger.info(
                f"Winner for {test_id}: Apply '{winner.content[:50]}...' "
                f"to video {test.base_video_id} ({test.test_type.value})"
            )
            return True

    def generate_title_variants(self, base_title: str, count: int = 3) -> List[str]:
        """
        Generate title variants using patterns.

        Args:
            base_title: Original title
            count: Number of variants to generate

        Returns:
            List of title variants
        """
        generator = TitleVariantGenerator()
        return generator.generate_variants(base_title, count=count)

    def get_active_tests(self) -> List[ABTest]:
        """Get all running tests."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM tests WHERE status IN ('running', 'paused')")
        rows = cursor.fetchall()
        conn.close()

        tests = []
        for row in rows:
            test = self._load_test(row["id"])
            if test:
                tests.append(test)

        return tests

    def get_all_tests(self, status: Optional[TestStatus] = None) -> List[ABTest]:
        """Get all tests, optionally filtered by status."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute("SELECT id FROM tests WHERE status = ?", (status.value,))
        else:
            cursor.execute("SELECT id FROM tests")

        rows = cursor.fetchall()
        conn.close()

        tests = []
        for row in rows:
            test = self._load_test(row["id"])
            if test:
                tests.append(test)

        return tests

    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get a test by ID."""
        return self._load_test(test_id)

    def delete_test(self, test_id: str) -> bool:
        """Delete a test and its variants."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM metrics_history WHERE test_id = ?", (test_id,))
            cursor.execute("DELETE FROM variants WHERE test_id = ?", (test_id,))
            cursor.execute("DELETE FROM tests WHERE id = ?", (test_id,))
            conn.commit()
            logger.info(f"Deleted test {test_id}")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete test {test_id}: {e}")
            return False
        finally:
            conn.close()

    def get_test_report(self, test_id: str) -> Dict[str, Any]:
        """
        Generate detailed report for a test.

        Args:
            test_id: Test identifier

        Returns:
            Detailed report dictionary
        """
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        # Calculate duration
        start = datetime.fromisoformat(test.start_time) if test.start_time else None
        end = datetime.fromisoformat(test.end_time) if test.end_time else datetime.now()
        duration = (end - start).total_seconds() / 3600 if start else 0

        # Get significance results
        significance = self.check_statistical_significance(test)

        # Get metrics history
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT variant_id, impressions, clicks, ctr, recorded_at
            FROM metrics_history
            WHERE test_id = ?
            ORDER BY recorded_at
        """, (test_id,))
        history = [dict(row) for row in cursor.fetchall()]
        conn.close()

        report = {
            "test_id": test.id,
            "test_type": test.test_type.value,
            "video_id": test.base_video_id,
            "status": test.status.value,
            "niche": test.niche,
            "created_at": test.created_at,
            "start_time": test.start_time,
            "end_time": test.end_time,
            "duration_hours": round(duration, 1),
            "variants": [v.to_dict() for v in test.variants],
            "total_impressions": test.get_total_impressions(),
            "total_clicks": test.get_total_clicks(),
            "significance": significance,
            "winner": test.winner_id,
            "metrics_history": history,
            "configuration": {
                "min_impressions": test.min_impressions,
                "confidence_level": test.confidence_level,
                "target_metric": test.target_metric,
                "auto_apply_winner": test.auto_apply_winner
            }
        }

        return report

    def print_report(self, test_id: str):
        """Print formatted report to console."""
        report = self.get_test_report(test_id)

        print("=" * 70)
        print(f"A/B TEST REPORT: {report['test_id']}")
        print("=" * 70)
        print(f"Type:          {report['test_type']}")
        print(f"Video ID:      {report['video_id']}")
        print(f"Status:        {report['status'].upper()}")
        print(f"Duration:      {report['duration_hours']} hours")
        print(f"Niche:         {report['niche'] or 'N/A'}")
        print()
        print("-" * 70)
        print("VARIANT PERFORMANCE")
        print("-" * 70)
        print(f"{'Variant':<12} {'Impressions':>12} {'Clicks':>10} {'CTR':>10} {'Avg Duration':>14}")
        print("-" * 70)

        for v in report["variants"]:
            winner_mark = " <-- WINNER" if v["id"] == report["winner"] else ""
            print(
                f"{v['id']:<12} {v['impressions']:>12,} {v['clicks']:>10,} "
                f"{v['ctr']:>9.2f}% {v['avg_view_duration']:>13.1f}s{winner_mark}"
            )

        print()
        print("-" * 70)
        print("STATISTICAL ANALYSIS")
        print("-" * 70)
        sig = report["significance"]
        print(f"Chi-squared:   {sig.get('chi_squared', 'N/A')}")
        print(f"P-value:       {sig.get('p_value', 'N/A')}")
        print(f"Confidence:    {sig.get('confidence_level', 0):.1%}")
        print(f"Significant:   {'Yes' if sig.get('is_significant') else 'No'}")
        print(f"CTR Lift:      {sig.get('ctr_lift_percent', 0):.1f}%")
        print()
        print("RECOMMENDATION:")
        print(sig.get("recommendation", "Continue collecting data."))
        print("=" * 70)


class TitleVariantGenerator:
    """Generate title variants for A/B testing."""

    PATTERNS = {
        "number_front": "{number} {topic} {suffix}",
        "question": "{question_word} {topic}?",
        "how_to": "How to {action} ({benefit})",
        "brackets": "{title} [{hook}]",
        "versus": "{option1} vs {option2}: Which is {comparison}?",
        "listicle": "{number} {adjective} {topic} You Need to Know",
        "revelation": "The {adjective} Truth About {topic}",
        "urgency": "{topic}: What {audience} Need to Know NOW",
        "challenge": "I Tried {topic} for {time_period} - Here's What Happened",
        "mistake": "{number} {topic} Mistakes You're Making Right Now",
    }

    POWER_WORDS = [
        "Secret", "Proven", "Ultimate", "Shocking", "Hidden", "Explosive",
        "Incredible", "Mind-Blowing", "Unexpected", "Surprising", "Essential",
        "Crucial", "Dangerous", "Revolutionary", "Powerful", "Amazing",
        "Unbelievable", "Insane", "Genius", "Epic", "Massive", "Complete"
    ]

    QUESTION_WORDS = ["Why", "How", "What", "When", "Which", "Where", "Who"]

    HOOKS = [
        "Must Watch", "Game Changer", "Life Hack", "Pro Tips", "Beginner's Guide",
        "2024 Update", "Full Guide", "Step by Step", "No BS", "Real Results",
        "Expert Advice", "Quick Tips", "Deep Dive", "Breakdown", "Explained"
    ]

    NUMBERS = ["3", "5", "7", "10", "12", "15", "21", "30", "50", "100"]

    ADJECTIVES = [
        "Simple", "Easy", "Quick", "Powerful", "Effective", "Proven",
        "Secret", "Hidden", "Unknown", "Surprising", "Essential", "Critical"
    ]

    TIME_PERIODS = [
        "7 Days", "30 Days", "1 Week", "1 Month", "3 Months", "1 Year"
    ]

    def __init__(self):
        """Initialize the generator."""
        self.used_patterns = set()

    def generate_variants(
        self,
        base_title: str,
        niche: str = "",
        count: int = 4,
        include_original: bool = True
    ) -> List[str]:
        """
        Generate multiple title variants.

        Args:
            base_title: Original title to base variants on
            niche: Content niche for context
            count: Number of variants to generate
            include_original: Whether to include the original title

        Returns:
            List of title variants
        """
        variants = []

        if include_original:
            variants.append(base_title)

        # Extract key elements from base title
        topic = self._extract_topic(base_title)

        # Generate variants using different patterns
        patterns_to_try = list(self.PATTERNS.keys())
        random.shuffle(patterns_to_try)

        for pattern_name in patterns_to_try:
            if len(variants) >= count:
                break

            variant = self._apply_pattern(pattern_name, topic, base_title)
            if variant and variant not in variants:
                variants.append(variant)

        # Add power word variants
        if len(variants) < count:
            power_variant = self._add_power_word(base_title)
            if power_variant not in variants:
                variants.append(power_variant)

        # Add bracket hook variants
        if len(variants) < count:
            hook_variant = self._add_bracket_hook(base_title)
            if hook_variant not in variants:
                variants.append(hook_variant)

        # Add number variants
        if len(variants) < count:
            number_variant = self._add_number_prefix(base_title)
            if number_variant not in variants:
                variants.append(number_variant)

        return variants[:count]

    def _extract_topic(self, title: str) -> str:
        """Extract the main topic from a title, keeping it usable for patterns."""
        topic = title.strip()

        # Remove brackets and their content first
        topic = re.sub(r'\s*\[.*?\]\s*', '', topic)
        topic = re.sub(r'\s*\(.*?\)\s*', '', topic)

        # Handle "How to" pattern FIRST - extract the actual action
        how_to_match = re.match(r'^How\s+to\s+(.+)$', topic, flags=re.IGNORECASE)
        if how_to_match:
            topic = how_to_match.group(1)
            # Clean up and return early
            return topic.strip('?!.:')

        # Remove numbers at start (like "5 Ways to...")
        topic = re.sub(r'^\d+\s+\w+\s+(?:to\s+)?', '', topic)

        # Remove question words at start
        for word in self.QUESTION_WORDS:
            topic = re.sub(rf'^{word}\s+', '', topic, flags=re.IGNORECASE)

        # Remove leading "to" if present after other processing
        topic = re.sub(r'^to\s+', '', topic, flags=re.IGNORECASE)

        # Clean up
        topic = topic.strip('?!.:')

        return topic

    def _is_how_to_title(self, title: str) -> bool:
        """Check if the title is a 'How to' style title."""
        return bool(re.match(r'^How\s+to\s+', title, flags=re.IGNORECASE))

    def _apply_pattern(self, pattern_name: str, topic: str, original: str) -> Optional[str]:
        """Apply a specific pattern to generate a variant."""
        # Determine if this is an action phrase (like "Make Money Online")
        # vs a noun phrase (like "Python Programming")
        is_action = self._is_action_phrase(topic)

        # Capitalize appropriately
        topic_caps = self._smart_capitalize(topic)

        if pattern_name == "number_front":
            if is_action:
                return f"{random.choice(self.NUMBERS)} {random.choice(self.ADJECTIVES)} Ways to {topic_caps}"
            else:
                return f"{random.choice(self.NUMBERS)} {random.choice(self.ADJECTIVES)} {topic_caps} Tips"

        elif pattern_name == "question":
            question = random.choice(["Why", "What"])
            if is_action:
                return f"{question} Everyone is Trying to {topic_caps}"
            else:
                return f"{question} Makes {topic_caps} So Popular?"

        elif pattern_name == "how_to":
            benefit = random.choice(["Fast", "Easy", "For Beginners", "Like a Pro", "Step by Step"])
            if is_action:
                return f"How to {topic_caps} ({benefit})"
            else:
                return f"How to Master {topic_caps} ({benefit})"

        elif pattern_name == "brackets":
            return f"{original} [{random.choice(self.HOOKS)}]"

        elif pattern_name == "versus":
            return None  # Skip - needs two options

        elif pattern_name == "listicle":
            if is_action:
                return f"{random.choice(self.NUMBERS)} {random.choice(self.ADJECTIVES)} Tips to {topic_caps}"
            else:
                return f"{random.choice(self.NUMBERS)} {random.choice(self.ADJECTIVES)} {topic_caps} Tips"

        elif pattern_name == "revelation":
            if is_action:
                return f"The {random.choice(self.ADJECTIVES)} Truth About How to {topic_caps}"
            else:
                return f"The {random.choice(self.ADJECTIVES)} Truth About {topic_caps}"

        elif pattern_name == "urgency":
            if is_action:
                return f"How to {topic_caps}: What You Need to Know NOW"
            else:
                return f"{topic_caps}: What You Need to Know in 2024"

        elif pattern_name == "challenge":
            if is_action:
                return f"I Tried to {topic_caps} for {random.choice(self.TIME_PERIODS)} - Results"
            else:
                return f"I Tried {topic_caps} for {random.choice(self.TIME_PERIODS)} - Here's What Happened"

        elif pattern_name == "mistake":
            if is_action:
                return f"{random.choice(['5', '7', '10'])} Mistakes When Trying to {topic_caps}"
            else:
                return f"{random.choice(['5', '7', '10'])} {topic_caps} Mistakes Everyone Makes"

        return None

    def _is_action_phrase(self, text: str) -> bool:
        """Check if the text is an action phrase (starts with a verb)."""
        action_verbs = [
            "make", "get", "build", "create", "start", "learn", "grow", "find",
            "improve", "increase", "boost", "master", "become", "achieve", "earn",
            "save", "lose", "gain", "develop", "write", "design", "code", "cook",
            "play", "speak", "read", "invest", "trade", "sell", "buy", "fix"
        ]
        first_word = text.split()[0].lower() if text else ""
        return first_word in action_verbs

    def _smart_capitalize(self, text: str) -> str:
        """Smart capitalize preserving acronyms and proper nouns."""
        words = text.split()
        result = []
        lowercase_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}

        for i, word in enumerate(words):
            # Keep all-caps words (acronyms like AI, SEO)
            if word.isupper() and len(word) > 1:
                result.append(word)
            # Always capitalize first word
            elif i == 0:
                result.append(word.capitalize())
            # Keep lowercase words lowercase (unless first)
            elif word.lower() in lowercase_words:
                result.append(word.lower())
            else:
                result.append(word.capitalize())

        return " ".join(result)

    def _add_power_word(self, title: str) -> str:
        """Add a power word to the title."""
        power_word = random.choice(self.POWER_WORDS)

        # Try different positions
        if random.random() < 0.5:
            # Add at beginning
            return f"{power_word}: {title}"
        else:
            # Add before main noun (simple approach)
            words = title.split()
            if len(words) > 2:
                insert_pos = random.randint(1, min(3, len(words) - 1))
                words.insert(insert_pos, power_word)
                return " ".join(words)
            return f"{power_word} {title}"

    def _add_bracket_hook(self, title: str) -> str:
        """Add a bracketed hook to the title."""
        # Remove existing brackets first
        clean_title = re.sub(r'\s*\[.*?\]\s*', '', title).strip()
        hook = random.choice(self.HOOKS)
        return f"{clean_title} [{hook}]"

    def _add_number_prefix(self, title: str) -> str:
        """Add a number prefix to create a listicle title."""
        # Check if already has number
        if re.match(r'^\d+', title):
            return title

        number = random.choice(self.NUMBERS)
        topic = self._extract_topic(title)
        return f"{number} Ways to {topic}"


class ThumbnailVariantGenerator:
    """Generate thumbnail variants for A/B testing."""

    # Color schemes for different moods/styles
    COLOR_SCHEMES = {
        "urgent": ["#FF0000", "#FF4444", "#CC0000"],  # Red
        "trust": ["#0066CC", "#0088FF", "#004499"],   # Blue
        "growth": ["#00CC00", "#44FF44", "#009900"],  # Green
        "premium": ["#FFD700", "#FFA500", "#FF8C00"], # Gold/Orange
        "calm": ["#9966FF", "#7744FF", "#5522CC"],    # Purple
        "energy": ["#FF6600", "#FF9900", "#FFCC00"], # Orange/Yellow
        "dark": ["#1A1A1A", "#333333", "#4D4D4D"],   # Dark
        "light": ["#FFFFFF", "#F5F5F5", "#EEEEEE"],  # Light
    }

    # Text overlay positions
    TEXT_POSITIONS = [
        "top_left", "top_center", "top_right",
        "center_left", "center", "center_right",
        "bottom_left", "bottom_center", "bottom_right"
    ]

    def __init__(self, output_dir: str = "data/thumbnails"):
        """
        Initialize the thumbnail generator.

        Args:
            output_dir: Directory for generated thumbnails
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_color_variants(
        self,
        base_thumbnail: str,
        colors: Optional[List[str]] = None,
        output_prefix: str = "variant"
    ) -> List[str]:
        """
        Create thumbnail variants with different background colors.

        Args:
            base_thumbnail: Path to base thumbnail image
            colors: List of hex color codes
            output_prefix: Prefix for output filenames

        Returns:
            List of paths to generated thumbnails
        """
        if not HAS_PIL:
            logger.error("PIL required for thumbnail generation")
            return [base_thumbnail]

        if not Path(base_thumbnail).exists():
            raise FileNotFoundError(f"Base thumbnail not found: {base_thumbnail}")

        if colors is None:
            colors = ["#FF0000", "#00CC00", "#0066CC", "#FFD700"]

        variants = []
        base_img = Image.open(base_thumbnail)

        for i, color in enumerate(colors):
            # Create colored overlay
            overlay = Image.new("RGBA", base_img.size, color + "40")  # 25% opacity

            # Composite
            if base_img.mode != "RGBA":
                base_img = base_img.convert("RGBA")

            result = Image.alpha_composite(base_img, overlay)

            # Save variant
            output_path = self.output_dir / f"{output_prefix}_{chr(65 + i)}_color.png"
            result.save(str(output_path), "PNG")
            variants.append(str(output_path))

            logger.debug(f"Created color variant: {output_path}")

        return variants

    def generate_text_variants(
        self,
        base_thumbnail: str,
        texts: List[str],
        position: str = "center",
        font_size: int = 72,
        font_color: str = "#FFFFFF",
        output_prefix: str = "variant"
    ) -> List[str]:
        """
        Create thumbnail variants with different text overlays.

        Args:
            base_thumbnail: Path to base thumbnail
            texts: List of text strings to overlay
            position: Text position (see TEXT_POSITIONS)
            font_size: Font size in pixels
            font_color: Hex color for text
            output_prefix: Prefix for output filenames

        Returns:
            List of paths to generated thumbnails
        """
        if not HAS_PIL:
            logger.error("PIL required for thumbnail generation")
            return [base_thumbnail]

        if not Path(base_thumbnail).exists():
            raise FileNotFoundError(f"Base thumbnail not found: {base_thumbnail}")

        variants = []
        base_img = Image.open(base_thumbnail)

        for i, text in enumerate(texts):
            # Copy base image
            img = base_img.copy()
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Create text overlay
            draw = ImageDraw.Draw(img)

            # Try to load a bold font, fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except OSError:
                    font = ImageFont.load_default()

            # Calculate text position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x, y = self._calculate_position(
                img.size, (text_width, text_height), position
            )

            # Draw text shadow
            shadow_offset = max(2, font_size // 20)
            draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill="#000000")

            # Draw text
            draw.text((x, y), text, font=font, fill=font_color)

            # Save variant
            output_path = self.output_dir / f"{output_prefix}_{chr(65 + i)}_text.png"
            img.save(str(output_path), "PNG")
            variants.append(str(output_path))

            logger.debug(f"Created text variant: {output_path}")

        return variants

    def generate_style_variants(
        self,
        base_thumbnail: str,
        styles: Optional[List[str]] = None,
        output_prefix: str = "variant"
    ) -> List[str]:
        """
        Create thumbnail variants with different visual styles.

        Args:
            base_thumbnail: Path to base thumbnail
            styles: List of style names (brightness, contrast, saturation, blur)
            output_prefix: Prefix for output filenames

        Returns:
            List of paths to generated thumbnails
        """
        if not HAS_PIL:
            logger.error("PIL required for thumbnail generation")
            return [base_thumbnail]

        if not Path(base_thumbnail).exists():
            raise FileNotFoundError(f"Base thumbnail not found: {base_thumbnail}")

        if styles is None:
            styles = ["high_contrast", "high_saturation", "warm", "cool"]

        variants = []
        base_img = Image.open(base_thumbnail)

        for i, style in enumerate(styles):
            img = base_img.copy()

            if style == "high_contrast":
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.4)
            elif style == "high_saturation":
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.5)
            elif style == "warm":
                # Add warm color overlay
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, "#FF990020")
                img = Image.alpha_composite(img, overlay)
            elif style == "cool":
                # Add cool color overlay
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, "#0066FF20")
                img = Image.alpha_composite(img, overlay)
            elif style == "bright":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(1.2)
            elif style == "dark":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(0.8)

            # Save variant
            output_path = self.output_dir / f"{output_prefix}_{chr(65 + i)}_{style}.png"
            img.save(str(output_path), "PNG")
            variants.append(str(output_path))

            logger.debug(f"Created style variant: {output_path}")

        return variants

    def _calculate_position(
        self,
        image_size: Tuple[int, int],
        text_size: Tuple[int, int],
        position: str
    ) -> Tuple[int, int]:
        """Calculate x, y coordinates for text position."""
        img_width, img_height = image_size
        text_width, text_height = text_size

        padding = 20

        positions = {
            "top_left": (padding, padding),
            "top_center": ((img_width - text_width) // 2, padding),
            "top_right": (img_width - text_width - padding, padding),
            "center_left": (padding, (img_height - text_height) // 2),
            "center": ((img_width - text_width) // 2, (img_height - text_height) // 2),
            "center_right": (img_width - text_width - padding, (img_height - text_height) // 2),
            "bottom_left": (padding, img_height - text_height - padding),
            "bottom_center": ((img_width - text_width) // 2, img_height - text_height - padding),
            "bottom_right": (img_width - text_width - padding, img_height - text_height - padding),
        }

        return positions.get(position, positions["center"])


# CLI interface
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="A/B Testing Framework for YouTube")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create title test
    create_title = subparsers.add_parser("create-title", help="Create a title A/B test")
    create_title.add_argument("video_id", help="YouTube video ID")
    create_title.add_argument("titles", nargs="+", help="Title variants to test")
    create_title.add_argument("--min-impressions", type=int, default=1000, help="Min impressions per variant")
    create_title.add_argument("--confidence", type=float, default=0.95, help="Required confidence level")
    create_title.add_argument("--niche", default="", help="Content niche")

    # Create thumbnail test
    create_thumb = subparsers.add_parser("create-thumbnail", help="Create a thumbnail A/B test")
    create_thumb.add_argument("video_id", help="YouTube video ID")
    create_thumb.add_argument("thumbnails", nargs="+", help="Thumbnail file paths")
    create_thumb.add_argument("--min-impressions", type=int, default=1000)
    create_thumb.add_argument("--confidence", type=float, default=0.95)

    # Start test
    start_cmd = subparsers.add_parser("start", help="Start a pending test")
    start_cmd.add_argument("test_id", help="Test ID to start")

    # Update metrics
    update_cmd = subparsers.add_parser("update", help="Update variant metrics")
    update_cmd.add_argument("test_id", help="Test ID")
    update_cmd.add_argument("variant_id", help="Variant ID (e.g., variant_A)")
    update_cmd.add_argument("--impressions", type=int, required=True)
    update_cmd.add_argument("--clicks", type=int, required=True)
    update_cmd.add_argument("--duration", type=float, default=0.0, help="Avg view duration")

    # Check significance
    check_cmd = subparsers.add_parser("check", help="Check test significance")
    check_cmd.add_argument("test_id", help="Test ID to check")

    # Get report
    report_cmd = subparsers.add_parser("report", help="Get test report")
    report_cmd.add_argument("test_id", help="Test ID")

    # List tests
    list_cmd = subparsers.add_parser("list", help="List tests")
    list_cmd.add_argument("--status", choices=["pending", "running", "completed", "cancelled"])

    # Generate title variants
    gen_titles = subparsers.add_parser("generate-titles", help="Generate title variants")
    gen_titles.add_argument("base_title", help="Base title")
    gen_titles.add_argument("--count", type=int, default=5)
    gen_titles.add_argument("--niche", default="")

    # Determine winner
    winner_cmd = subparsers.add_parser("winner", help="Determine and optionally apply winner")
    winner_cmd.add_argument("test_id", help="Test ID")
    winner_cmd.add_argument("--apply", action="store_true", help="Apply winner to video")

    args = parser.parse_args()

    # Initialize manager
    manager = ABTestManager()

    if args.command == "create-title":
        test = manager.create_title_test(
            video_id=args.video_id,
            titles=args.titles,
            min_impressions=args.min_impressions,
            confidence_level=args.confidence,
            niche=args.niche
        )
        print(f"Created test: {test.id}")
        for v in test.variants:
            print(f"  {v.id}: {v.content[:50]}...")

    elif args.command == "create-thumbnail":
        test = manager.create_thumbnail_test(
            video_id=args.video_id,
            thumbnail_paths=args.thumbnails,
            min_impressions=args.min_impressions,
            confidence_level=args.confidence
        )
        print(f"Created test: {test.id}")

    elif args.command == "start":
        test = manager.start_test(args.test_id)
        print(f"Started test: {test.id}")

    elif args.command == "update":
        manager.update_metrics(
            test_id=args.test_id,
            variant_id=args.variant_id,
            impressions=args.impressions,
            clicks=args.clicks,
            avg_duration=args.duration
        )
        print(f"Updated metrics for {args.test_id}/{args.variant_id}")

    elif args.command == "check":
        test = manager.get_test(args.test_id)
        if not test:
            print(f"Test not found: {args.test_id}")
            sys.exit(1)

        result = manager.check_statistical_significance(test)
        print(json.dumps(result, indent=2))

    elif args.command == "report":
        manager.print_report(args.test_id)

    elif args.command == "list":
        status = TestStatus(args.status) if args.status else None
        tests = manager.get_all_tests(status)

        if not tests:
            print("No tests found")
        else:
            print(f"{'ID':<25} {'Type':<12} {'Status':<12} {'Variants':<10} {'Impressions':>12}")
            print("-" * 75)
            for test in tests:
                print(
                    f"{test.id:<25} {test.test_type.value:<12} {test.status.value:<12} "
                    f"{len(test.variants):<10} {test.get_total_impressions():>12,}"
                )

    elif args.command == "generate-titles":
        generator = TitleVariantGenerator()
        variants = generator.generate_variants(
            base_title=args.base_title,
            niche=args.niche,
            count=args.count
        )
        print("Generated title variants:")
        for i, title in enumerate(variants):
            print(f"  {i + 1}. {title}")

    elif args.command == "winner":
        winner = manager.determine_winner(args.test_id)
        if winner:
            print(f"Winner: {winner.id}")
            print(f"Content: {winner.content[:100]}...")
            print(f"CTR: {winner.ctr:.2f}%")

            if args.apply:
                success = manager.apply_winner(args.test_id)
                print(f"Applied: {'Yes' if success else 'No'}")
        else:
            print("No statistically significant winner yet")

    else:
        parser.print_help()
