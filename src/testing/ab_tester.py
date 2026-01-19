"""
A/B Testing Module for YouTube Thumbnails and Titles.

This module implements statistical A/B testing for comparing different
thumbnail and title variants to optimize click-through rates.

Test Strategy:
1. Upload video with thumbnail A
2. Wait 36 hours, collect metrics
3. Switch to thumbnail B
4. Wait 36 hours, collect metrics
5. Determine winner with statistical significance (Chi-squared test)

Usage:
    from src.testing.ab_tester import ABTester

    tester = ABTester()
    test_id = tester.start_thumbnail_test(
        video_id="abc123",
        thumbnail_variants=["/path/to/thumb_a.png", "/path/to/thumb_b.png"],
        test_duration_hours=72
    )

    # Check progress
    progress = tester.check_test_progress(test_id)

    # Get report
    report = tester.get_test_report(test_id)
    print(report)
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from loguru import logger

# Try to import scipy for statistical tests
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed. Statistical analysis will be limited.")


class ABTestStatus(Enum):
    """Status of an A/B test."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


class ABTestType(Enum):
    """Type of A/B test."""
    THUMBNAIL = "thumbnail"
    TITLE = "title"
    THUMBNAIL_AND_TITLE = "thumbnail_and_title"


@dataclass
class ABVariant:
    """A single variant in an A/B test."""
    id: str
    title: Optional[str] = None
    thumbnail_path: Optional[str] = None
    impressions: int = 0
    clicks: int = 0
    views: int = 0
    watch_time: float = 0.0  # Total watch time in seconds

    @property
    def ctr(self) -> float:
        """Calculate click-through rate (CTR)."""
        if self.impressions == 0:
            return 0.0
        return (self.clicks / self.impressions) * 100

    @property
    def avg_watch_time(self) -> float:
        """Calculate average watch time per view."""
        if self.views == 0:
            return 0.0
        return self.watch_time / self.views

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "thumbnail_path": self.thumbnail_path,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "views": self.views,
            "watch_time": self.watch_time,
            "ctr": round(self.ctr, 4),
            "avg_watch_time": round(self.avg_watch_time, 2)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABVariant":
        """Create variant from dictionary."""
        return cls(
            id=data["id"],
            title=data.get("title"),
            thumbnail_path=data.get("thumbnail_path"),
            impressions=data.get("impressions", 0),
            clicks=data.get("clicks", 0),
            views=data.get("views", 0),
            watch_time=data.get("watch_time", 0.0)
        )


@dataclass
class ABTest:
    """An A/B test configuration and results."""
    test_id: str
    video_id: str
    test_type: ABTestType
    variants: List[ABVariant]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ABTestStatus = ABTestStatus.RUNNING
    winner_id: Optional[str] = None
    confidence_level: float = 0.0
    current_variant_index: int = 0
    variant_switch_time: Optional[datetime] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "video_id": self.video_id,
            "test_type": self.test_type.value,
            "variants": [v.to_dict() for v in self.variants],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "winner_id": self.winner_id,
            "confidence_level": round(self.confidence_level, 4),
            "current_variant_index": self.current_variant_index,
            "variant_switch_time": self.variant_switch_time.isoformat() if self.variant_switch_time else None,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABTest":
        """Create test from dictionary."""
        return cls(
            test_id=data["test_id"],
            video_id=data["video_id"],
            test_type=ABTestType(data["test_type"]),
            variants=[ABVariant.from_dict(v) for v in data["variants"]],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=ABTestStatus(data["status"]),
            winner_id=data.get("winner_id"),
            confidence_level=data.get("confidence_level", 0.0),
            current_variant_index=data.get("current_variant_index", 0),
            variant_switch_time=datetime.fromisoformat(data["variant_switch_time"]) if data.get("variant_switch_time") else None,
            notes=data.get("notes", "")
        )


@dataclass
class AnalysisResult:
    """Result of statistical analysis."""
    winner_id: Optional[str]
    confidence_level: float
    chi_squared: float
    p_value: float
    is_significant: bool
    variant_stats: Dict[str, Dict[str, Any]]
    recommendation: str


class ABTester:
    """
    A/B Testing manager for YouTube thumbnails and titles.

    Uses a sequential testing strategy:
    1. Show variant A for half the test duration
    2. Collect metrics (impressions, clicks, views, watch time)
    3. Switch to variant B for remaining duration
    4. Collect metrics
    5. Run Chi-squared test for statistical significance
    """

    # Configuration constants
    MIN_IMPRESSIONS_PER_VARIANT = 1000
    MIN_TEST_HOURS = 48
    CONFIDENCE_THRESHOLD = 0.95  # 95% confidence required

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the A/B tester.

        Args:
            data_dir: Directory for storing test data. Defaults to data/
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Default to project's data directory
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / "data"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tests_file = self.data_dir / "ab_tests.json"

        # Load existing tests
        self.tests: Dict[str, ABTest] = self._load_tests()

        logger.info(f"ABTester initialized. Data file: {self.tests_file}")

    def _load_tests(self) -> Dict[str, ABTest]:
        """Load tests from JSON file."""
        if not self.tests_file.exists():
            return {}

        try:
            with open(self.tests_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    test_id: ABTest.from_dict(test_data)
                    for test_id, test_data in data.items()
                }
        except Exception as e:
            logger.error(f"Failed to load tests: {e}")
            return {}

    def _save_tests(self) -> None:
        """Save tests to JSON file."""
        try:
            data = {
                test_id: test.to_dict()
                for test_id, test in self.tests.items()
            }
            with open(self.tests_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.tests)} tests to {self.tests_file}")
        except Exception as e:
            logger.error(f"Failed to save tests: {e}")

    def start_thumbnail_test(
        self,
        video_id: str,
        thumbnail_variants: List[str],
        test_duration_hours: int = 72,
        titles: Optional[List[str]] = None
    ) -> str:
        """
        Start a new A/B test for thumbnails.

        Args:
            video_id: YouTube video ID
            thumbnail_variants: List of paths to thumbnail images
            test_duration_hours: Total test duration (default 72 hours)
            titles: Optional list of title variants (same length as thumbnails)

        Returns:
            Test ID for tracking

        Raises:
            ValueError: If invalid parameters provided
        """
        if len(thumbnail_variants) < 2:
            raise ValueError("At least 2 thumbnail variants required")

        if titles and len(titles) != len(thumbnail_variants):
            raise ValueError("Number of titles must match number of thumbnails")

        # Validate thumbnail files exist
        for thumb_path in thumbnail_variants:
            if not Path(thumb_path).exists():
                raise ValueError(f"Thumbnail not found: {thumb_path}")

        # Generate test ID
        test_id = f"ab_{uuid.uuid4().hex[:8]}"

        # Create variants
        variants = []
        for i, thumb_path in enumerate(thumbnail_variants):
            variant = ABVariant(
                id=f"variant_{chr(65 + i)}",  # A, B, C, ...
                thumbnail_path=str(Path(thumb_path).absolute()),
                title=titles[i] if titles else None
            )
            variants.append(variant)

        # Determine test type
        test_type = ABTestType.THUMBNAIL_AND_TITLE if titles else ABTestType.THUMBNAIL

        # Calculate end time
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=test_duration_hours)

        # Calculate when to switch variants
        switch_duration = test_duration_hours / len(variants)
        variant_switch_time = start_time + timedelta(hours=switch_duration)

        # Create test
        test = ABTest(
            test_id=test_id,
            video_id=video_id,
            test_type=test_type,
            variants=variants,
            start_time=start_time,
            end_time=end_time,
            status=ABTestStatus.RUNNING,
            current_variant_index=0,
            variant_switch_time=variant_switch_time
        )

        # Save test
        self.tests[test_id] = test
        self._save_tests()

        logger.info(f"Started A/B test {test_id} for video {video_id}")
        logger.info(f"  Variants: {len(variants)}")
        logger.info(f"  Duration: {test_duration_hours} hours")
        logger.info(f"  First switch at: {variant_switch_time}")

        return test_id

    def start_title_test(
        self,
        video_id: str,
        title_variants: List[str],
        thumbnail_path: str,
        test_duration_hours: int = 72
    ) -> str:
        """
        Start a new A/B test for titles only.

        Args:
            video_id: YouTube video ID
            title_variants: List of title options
            thumbnail_path: Single thumbnail to use
            test_duration_hours: Total test duration

        Returns:
            Test ID for tracking
        """
        if len(title_variants) < 2:
            raise ValueError("At least 2 title variants required")

        test_id = f"ab_{uuid.uuid4().hex[:8]}"

        variants = []
        for i, title in enumerate(title_variants):
            variant = ABVariant(
                id=f"variant_{chr(65 + i)}",
                title=title,
                thumbnail_path=str(Path(thumbnail_path).absolute())
            )
            variants.append(variant)

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=test_duration_hours)
        switch_duration = test_duration_hours / len(variants)
        variant_switch_time = start_time + timedelta(hours=switch_duration)

        test = ABTest(
            test_id=test_id,
            video_id=video_id,
            test_type=ABTestType.TITLE,
            variants=variants,
            start_time=start_time,
            end_time=end_time,
            status=ABTestStatus.RUNNING,
            current_variant_index=0,
            variant_switch_time=variant_switch_time
        )

        self.tests[test_id] = test
        self._save_tests()

        logger.info(f"Started title A/B test {test_id} for video {video_id}")

        return test_id

    def update_metrics(
        self,
        test_id: str,
        impressions: int,
        clicks: int,
        views: int,
        watch_time: float
    ) -> None:
        """
        Update metrics for the current variant.

        Args:
            test_id: Test identifier
            impressions: Number of impressions to add
            clicks: Number of clicks to add
            views: Number of views to add
            watch_time: Total watch time to add (seconds)
        """
        if test_id not in self.tests:
            raise ValueError(f"Test not found: {test_id}")

        test = self.tests[test_id]

        if test.status != ABTestStatus.RUNNING:
            logger.warning(f"Test {test_id} is not running, skipping metric update")
            return

        # Update current variant
        current_variant = test.variants[test.current_variant_index]
        current_variant.impressions += impressions
        current_variant.clicks += clicks
        current_variant.views += views
        current_variant.watch_time += watch_time

        self._save_tests()

        logger.debug(
            f"Updated metrics for {test_id}/{current_variant.id}: "
            f"+{impressions} impr, +{clicks} clicks, +{views} views"
        )

    def check_test_progress(self, test_id: str) -> Dict[str, Any]:
        """
        Check test progress and switch variants if needed.

        This method should be called periodically to:
        1. Update test status based on time
        2. Switch to next variant when time is up
        3. Analyze results when test is complete

        Args:
            test_id: Test identifier

        Returns:
            Progress report dictionary
        """
        if test_id not in self.tests:
            raise ValueError(f"Test not found: {test_id}")

        test = self.tests[test_id]
        now = datetime.now()

        # If test already completed or failed, return current status
        if test.status in [ABTestStatus.COMPLETED, ABTestStatus.FAILED]:
            return self._build_progress_report(test)

        # Check if test duration has ended
        if test.end_time and now >= test.end_time:
            # Run final analysis
            analysis = self._analyze_results(test)

            # Update test with results
            if analysis.is_significant:
                test.status = ABTestStatus.COMPLETED
                test.winner_id = analysis.winner_id
                test.confidence_level = analysis.confidence_level
            else:
                test.status = ABTestStatus.INSUFFICIENT_DATA
                test.notes = analysis.recommendation

            self._save_tests()
            logger.info(f"Test {test_id} completed. Status: {test.status.value}")

            return self._build_progress_report(test, analysis)

        # Check if we need to switch variants
        if test.variant_switch_time and now >= test.variant_switch_time:
            next_index = test.current_variant_index + 1

            if next_index < len(test.variants):
                test.current_variant_index = next_index

                # Calculate next switch time
                remaining_variants = len(test.variants) - next_index
                if remaining_variants > 0 and test.end_time:
                    remaining_time = test.end_time - now
                    switch_duration = remaining_time / remaining_variants
                    test.variant_switch_time = now + switch_duration
                else:
                    test.variant_switch_time = None

                self._save_tests()

                logger.info(
                    f"Test {test_id}: Switched to variant "
                    f"{test.variants[next_index].id}"
                )

                return {
                    "action": "switch_variant",
                    "new_variant": test.variants[next_index].to_dict(),
                    "progress": self._build_progress_report(test)
                }

        return self._build_progress_report(test)

    def _build_progress_report(
        self,
        test: ABTest,
        analysis: Optional[AnalysisResult] = None
    ) -> Dict[str, Any]:
        """Build a progress report dictionary."""
        now = datetime.now()
        elapsed = now - test.start_time

        if test.end_time:
            total_duration = test.end_time - test.start_time
            progress_pct = min(100, (elapsed / total_duration) * 100)
            remaining = max(timedelta(0), test.end_time - now)
        else:
            progress_pct = 0
            remaining = timedelta(0)

        report = {
            "test_id": test.test_id,
            "video_id": test.video_id,
            "status": test.status.value,
            "elapsed_hours": round(elapsed.total_seconds() / 3600, 1),
            "remaining_hours": round(remaining.total_seconds() / 3600, 1),
            "progress_percent": round(progress_pct, 1),
            "current_variant": test.variants[test.current_variant_index].id,
            "variants": [v.to_dict() for v in test.variants],
            "winner_id": test.winner_id,
            "confidence_level": test.confidence_level
        }

        if analysis:
            report["analysis"] = {
                "chi_squared": round(analysis.chi_squared, 4),
                "p_value": round(analysis.p_value, 6),
                "is_significant": analysis.is_significant,
                "recommendation": analysis.recommendation
            }

        return report

    def _analyze_results(self, test: ABTest) -> AnalysisResult:
        """
        Analyze test results using Chi-squared test.

        The Chi-squared test compares observed click frequencies
        against expected frequencies (if CTR were the same).

        Args:
            test: The ABTest to analyze

        Returns:
            AnalysisResult with statistical findings
        """
        variants = test.variants

        # Check minimum data requirements
        total_impressions = sum(v.impressions for v in variants)
        min_impressions = self.MIN_IMPRESSIONS_PER_VARIANT * len(variants)

        if total_impressions < min_impressions:
            return AnalysisResult(
                winner_id=None,
                confidence_level=0.0,
                chi_squared=0.0,
                p_value=1.0,
                is_significant=False,
                variant_stats={v.id: {"ctr": v.ctr, "impressions": v.impressions} for v in variants},
                recommendation=f"Need at least {min_impressions} total impressions. "
                             f"Currently have {total_impressions}."
            )

        # Prepare data for Chi-squared test
        # Contingency table: [clicks, non-clicks] for each variant
        observed = []
        for v in variants:
            clicks = v.clicks
            non_clicks = v.impressions - v.clicks
            observed.append([clicks, non_clicks])

        # Run Chi-squared test
        if HAS_SCIPY:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        else:
            # Fallback: simple CTR comparison without statistical test
            chi2 = 0.0
            p_value = 1.0
            logger.warning("scipy not available. Using simple CTR comparison.")

        # Determine confidence level (1 - p_value)
        confidence_level = 1 - p_value
        is_significant = confidence_level >= self.CONFIDENCE_THRESHOLD

        # Find winner (highest CTR)
        best_variant = max(variants, key=lambda v: v.ctr)

        # Build variant stats
        variant_stats = {}
        for v in variants:
            variant_stats[v.id] = {
                "ctr": round(v.ctr, 4),
                "impressions": v.impressions,
                "clicks": v.clicks,
                "views": v.views,
                "avg_watch_time": round(v.avg_watch_time, 2)
            }

        # Generate recommendation
        if is_significant:
            ctr_diff = best_variant.ctr - min(v.ctr for v in variants)
            recommendation = (
                f"Variant {best_variant.id} is the winner with {confidence_level:.1%} confidence. "
                f"CTR improvement: +{ctr_diff:.2f}%. "
                f"Recommend using {best_variant.id}'s thumbnail/title."
            )
        else:
            recommendation = (
                f"No statistically significant difference found (confidence: {confidence_level:.1%}). "
                f"Consider running a longer test or using other factors to decide."
            )

        return AnalysisResult(
            winner_id=best_variant.id if is_significant else None,
            confidence_level=confidence_level,
            chi_squared=chi2,
            p_value=p_value,
            is_significant=is_significant,
            variant_stats=variant_stats,
            recommendation=recommendation
        )

    def get_test_report(self, test_id: str) -> str:
        """
        Generate a human-readable report for a test.

        Args:
            test_id: Test identifier

        Returns:
            Formatted string report
        """
        if test_id not in self.tests:
            return f"Error: Test not found: {test_id}"

        test = self.tests[test_id]
        analysis = self._analyze_results(test)

        # Build report
        lines = [
            "=" * 60,
            f"A/B TEST REPORT: {test_id}",
            "=" * 60,
            "",
            f"Video ID:     {test.video_id}",
            f"Test Type:    {test.test_type.value}",
            f"Status:       {test.status.value.upper()}",
            f"Started:      {test.start_time.strftime('%Y-%m-%d %H:%M')}",
        ]

        if test.end_time:
            lines.append(f"End Time:     {test.end_time.strftime('%Y-%m-%d %H:%M')}")

        # Duration info
        elapsed = datetime.now() - test.start_time
        lines.append(f"Elapsed:      {elapsed.total_seconds() / 3600:.1f} hours")

        lines.extend(["", "-" * 60, "VARIANT PERFORMANCE", "-" * 60, ""])

        # Variant table
        header = f"{'Variant':<12} {'Impressions':>12} {'Clicks':>10} {'CTR':>10} {'Views':>10} {'Avg Watch':>12}"
        lines.append(header)
        lines.append("-" * len(header))

        for v in test.variants:
            row = (
                f"{v.id:<12} {v.impressions:>12,} {v.clicks:>10,} "
                f"{v.ctr:>9.2f}% {v.views:>10,} {v.avg_watch_time:>11.1f}s"
            )
            if test.winner_id == v.id:
                row += " <-- WINNER"
            lines.append(row)

        lines.extend(["", "-" * 60, "STATISTICAL ANALYSIS", "-" * 60, ""])

        lines.extend([
            f"Chi-squared:      {analysis.chi_squared:.4f}",
            f"P-value:          {analysis.p_value:.6f}",
            f"Confidence:       {analysis.confidence_level:.1%}",
            f"Significant:      {'Yes' if analysis.is_significant else 'No'}",
            "",
            "RECOMMENDATION:",
            analysis.recommendation,
            "",
            "=" * 60
        ])

        return "\n".join(lines)

    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get all currently running tests."""
        active = []
        for test in self.tests.values():
            if test.status == ABTestStatus.RUNNING:
                active.append(self._build_progress_report(test))
        return active

    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get a test by ID."""
        return self.tests.get(test_id)

    def delete_test(self, test_id: str) -> bool:
        """Delete a test."""
        if test_id in self.tests:
            del self.tests[test_id]
            self._save_tests()
            logger.info(f"Deleted test: {test_id}")
            return True
        return False

    def get_youtube_actions(self, test_id: str) -> Dict[str, Any]:
        """
        Get YouTube API actions needed for the current test state.

        Returns instructions for what YouTube API calls to make.

        Args:
            test_id: Test identifier

        Returns:
            Dictionary with action type and parameters
        """
        if test_id not in self.tests:
            return {"action": "none", "error": "Test not found"}

        test = self.tests[test_id]
        current_variant = test.variants[test.current_variant_index]

        if test.status != ABTestStatus.RUNNING:
            if test.winner_id:
                winner = next(v for v in test.variants if v.id == test.winner_id)
                return {
                    "action": "set_final",
                    "video_id": test.video_id,
                    "thumbnail_path": winner.thumbnail_path,
                    "title": winner.title
                }
            return {"action": "none", "status": test.status.value}

        return {
            "action": "set_variant",
            "video_id": test.video_id,
            "variant_id": current_variant.id,
            "thumbnail_path": current_variant.thumbnail_path,
            "title": current_variant.title
        }


# CLI interface for testing
if __name__ == "__main__":
    import sys

    tester = ABTester()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ab_tester.py start <video_id> <thumb1.png> <thumb2.png>")
        print("  python ab_tester.py check <test_id>")
        print("  python ab_tester.py report <test_id>")
        print("  python ab_tester.py list")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "start":
        if len(sys.argv) < 5:
            print("Error: Need video_id and at least 2 thumbnail paths")
            sys.exit(1)

        video_id = sys.argv[2]
        thumbnails = sys.argv[3:]

        test_id = tester.start_thumbnail_test(video_id, thumbnails)
        print(f"Started test: {test_id}")

    elif cmd == "check":
        if len(sys.argv) < 3:
            print("Error: Need test_id")
            sys.exit(1)

        test_id = sys.argv[2]
        progress = tester.check_test_progress(test_id)
        print(json.dumps(progress, indent=2))

    elif cmd == "report":
        if len(sys.argv) < 3:
            print("Error: Need test_id")
            sys.exit(1)

        test_id = sys.argv[2]
        print(tester.get_test_report(test_id))

    elif cmd == "list":
        active = tester.get_active_tests()
        if active:
            print(f"Active tests ({len(active)}):")
            for t in active:
                print(f"  {t['test_id']}: video={t['video_id']}, progress={t['progress_percent']}%")
        else:
            print("No active tests")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
