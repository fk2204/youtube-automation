"""
Video Quality Checker System

Validates videos before uploading to YouTube by checking:
1. File Quality - Size, existence, corruption, duration
2. Content Quality - Script engagement, title, description, tags (using AI)
3. Technical Quality - Resolution, audio presence, silent gaps

Usage:
    from src.content.quality_checker import VideoQualityChecker

    checker = VideoQualityChecker()
    report = checker.check_video(
        video_file="output/video.mp4",
        script_data=script_dict,
        is_short=False
    )

    if report.passed:
        # Safe to upload
        pass
    else:
        print(f"Quality check failed: {report.issues}")
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from loguru import logger
from enum import Enum


class IssueSeverity(Enum):
    """Severity levels for quality issues."""
    INFO = "info"           # Minor suggestion, doesn't affect score
    WARNING = "warning"     # Affects score slightly (-5 to -10)
    ERROR = "error"         # Affects score significantly (-15 to -25)
    CRITICAL = "critical"   # Fails the check entirely (-50 or auto-fail)


@dataclass
class QualityIssue:
    """Represents a single quality issue found during checking."""
    category: str           # file, content, technical
    issue: str              # Description of the issue
    severity: IssueSeverity
    score_impact: int       # How much this affects the score (negative)
    recommendation: str     # How to fix it


@dataclass
class QualityReport:
    """Complete quality report for a video."""
    video_file: str
    is_short: bool
    overall_score: int                          # 0-100
    passed: bool                                # True if score >= threshold
    threshold: int                              # Passing threshold used
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    file_checks: Dict[str, Any] = field(default_factory=dict)
    content_checks: Dict[str, Any] = field(default_factory=dict)
    technical_checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "video_file": self.video_file,
            "is_short": self.is_short,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "threshold": self.threshold,
            "issues": [
                {
                    "category": issue.category,
                    "issue": issue.issue,
                    "severity": issue.severity.value,
                    "score_impact": issue.score_impact,
                    "recommendation": issue.recommendation
                }
                for issue in self.issues
            ],
            "recommendations": self.recommendations,
            "file_checks": self.file_checks,
            "content_checks": self.content_checks,
            "technical_checks": self.technical_checks
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Quality Report: {status}",
            f"Score: {self.overall_score}/100 (threshold: {self.threshold})",
            f"Video: {self.video_file}",
            f"Format: {'YouTube Short' if self.is_short else 'Regular Video'}",
            ""
        ]

        if self.issues:
            lines.append(f"Issues Found ({len(self.issues)}):")
            for issue in self.issues:
                severity_icon = {
                    IssueSeverity.INFO: "i",
                    IssueSeverity.WARNING: "!",
                    IssueSeverity.ERROR: "X",
                    IssueSeverity.CRITICAL: "XX"
                }.get(issue.severity, "?")
                lines.append(f"  [{severity_icon}] {issue.category}: {issue.issue}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:  # Top 5 recommendations
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class VideoQualityChecker:
    """
    Comprehensive video quality checker for YouTube uploads.

    Validates:
    - File quality (size, existence, corruption, duration)
    - Content quality (script engagement, title, description, tags)
    - Technical quality (resolution, audio, silent gaps)
    """

    # File size requirements (in bytes)
    MIN_SIZE_REGULAR = 5 * 1024 * 1024      # 5MB for regular videos
    MIN_SIZE_SHORTS = 2 * 1024 * 1024       # 2MB for Shorts
    MAX_SIZE_REGULAR = 256 * 1024 * 1024 * 1024  # 256GB (YouTube limit)

    # Duration requirements (in seconds)
    MIN_DURATION_REGULAR = 60               # 1 minute minimum
    MAX_DURATION_REGULAR = 12 * 60 * 60     # 12 hours (YouTube limit)
    MIN_DURATION_SHORTS = 15                # 15 seconds minimum
    MAX_DURATION_SHORTS = 60                # 60 seconds maximum for Shorts
    OPTIMAL_DURATION_SHORTS = 30            # Optimal for engagement

    # Resolution requirements
    RESOLUTION_REGULAR = (1920, 1080)       # 16:9 landscape
    RESOLUTION_SHORTS = (1080, 1920)        # 9:16 portrait

    # Audio requirements
    MAX_SILENT_GAP_SECONDS = 3              # Maximum allowed silent gap
    MIN_AUDIO_LEVEL_DB = -50                # Minimum audio level (very quiet threshold)

    # Engagement check thresholds
    HOOK_WINDOW_SECONDS = 15                # First 15 seconds for hook check
    MIN_HOOK_WORDS = 20                     # Minimum words in hook

    # Title requirements
    MAX_TITLE_LENGTH = 100                  # YouTube limit
    MIN_TITLE_LENGTH = 10                   # Minimum for quality
    GENERIC_TITLE_PATTERNS = [
        "video", "my video", "new video", "untitled",
        "test", "clip", "upload"
    ]

    # Description requirements
    MIN_DESCRIPTION_LENGTH = 50             # Minimum characters
    OPTIMAL_DESCRIPTION_LENGTH = 200        # Recommended

    def __init__(self, ai_provider: Optional[str] = None):
        """
        Initialize the quality checker.

        Args:
            ai_provider: AI provider for content analysis (ollama, groq, claude, etc.)
                        If None, uses AI_PROVIDER environment variable
        """
        self.ai_provider = ai_provider or os.getenv("AI_PROVIDER", "ollama")
        self._ai_instance = None
        logger.info(f"VideoQualityChecker initialized (AI: {self.ai_provider})")

    def _get_ai_provider(self):
        """Lazy-load AI provider for content analysis."""
        if self._ai_instance is None:
            try:
                from src.content.script_writer import get_provider
                self._ai_instance = get_provider(self.ai_provider)
            except Exception as e:
                logger.warning(f"Could not initialize AI provider: {e}")
                self._ai_instance = None
        return self._ai_instance

    def check_video(
        self,
        video_file: str,
        script_data: Optional[Dict[str, Any]] = None,
        is_short: bool = False,
        threshold: int = 70,
        skip_ai_checks: bool = False
    ) -> QualityReport:
        """
        Perform comprehensive quality check on a video.

        Args:
            video_file: Path to the video file
            script_data: Optional script data dict with title, description, tags, sections
            is_short: Whether this is a YouTube Short
            threshold: Minimum score to pass (0-100)
            skip_ai_checks: Skip AI-based content analysis (faster but less thorough)

        Returns:
            QualityReport with score, issues, and recommendations
        """
        logger.info(f"Starting quality check for: {video_file}")

        issues = []
        file_checks = {}
        content_checks = {}
        technical_checks = {}

        # Start with perfect score
        score = 100

        # ============================================================
        # FILE QUALITY CHECKS
        # ============================================================
        file_issues, file_score, file_checks = self._check_file_quality(
            video_file, is_short
        )
        issues.extend(file_issues)
        score -= (100 - file_score)

        # ============================================================
        # CONTENT QUALITY CHECKS (if script data provided)
        # ============================================================
        if script_data:
            content_issues, content_score, content_checks = self._check_content_quality(
                script_data, is_short, skip_ai_checks
            )
            issues.extend(content_issues)
            score -= (100 - content_score)
        else:
            content_checks = {"skipped": True, "reason": "No script data provided"}

        # ============================================================
        # TECHNICAL QUALITY CHECKS
        # ============================================================
        technical_issues, technical_score, technical_checks = self._check_technical_quality(
            video_file, is_short
        )
        issues.extend(technical_issues)
        score -= (100 - technical_score)

        # Ensure score is in valid range
        score = max(0, min(100, score))

        # Generate recommendations from issues
        recommendations = [
            issue.recommendation for issue in issues
            if issue.severity in [IssueSeverity.WARNING, IssueSeverity.ERROR, IssueSeverity.CRITICAL]
        ]

        # Create report
        report = QualityReport(
            video_file=video_file,
            is_short=is_short,
            overall_score=score,
            passed=score >= threshold,
            threshold=threshold,
            issues=issues,
            recommendations=recommendations,
            file_checks=file_checks,
            content_checks=content_checks,
            technical_checks=technical_checks
        )

        logger.info(f"Quality check complete: {score}/100 ({'PASS' if report.passed else 'FAIL'})")
        return report

    def _check_file_quality(
        self,
        video_file: str,
        is_short: bool
    ) -> Tuple[List[QualityIssue], int, Dict[str, Any]]:
        """
        Check file-level quality metrics.

        Returns:
            Tuple of (issues, score, checks_dict)
        """
        issues = []
        checks = {}
        score = 100

        # Check file exists
        if not os.path.exists(video_file):
            issues.append(QualityIssue(
                category="file",
                issue="Video file does not exist",
                severity=IssueSeverity.CRITICAL,
                score_impact=-100,
                recommendation="Ensure video file is created before upload"
            ))
            checks["exists"] = False
            return issues, 0, checks

        checks["exists"] = True

        # Check file size
        file_size = os.path.getsize(video_file)
        checks["file_size_bytes"] = file_size
        checks["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        min_size = self.MIN_SIZE_SHORTS if is_short else self.MIN_SIZE_REGULAR
        if file_size < min_size:
            min_mb = min_size / (1024 * 1024)
            issues.append(QualityIssue(
                category="file",
                issue=f"File size too small ({checks['file_size_mb']}MB < {min_mb}MB)",
                severity=IssueSeverity.ERROR,
                score_impact=-20,
                recommendation="Ensure video has sufficient content and quality"
            ))
            score -= 20

        if file_size > self.MAX_SIZE_REGULAR:
            issues.append(QualityIssue(
                category="file",
                issue="File exceeds YouTube's 256GB limit",
                severity=IssueSeverity.CRITICAL,
                score_impact=-100,
                recommendation="Reduce file size by lowering bitrate or resolution"
            ))
            score -= 100

        # Check if file is corrupted (try to read video metadata)
        try:
            duration, width, height = self._get_video_info(video_file)
            checks["duration_seconds"] = duration
            checks["width"] = width
            checks["height"] = height
            checks["corrupted"] = False

            # Check duration
            if is_short:
                if duration < self.MIN_DURATION_SHORTS:
                    issues.append(QualityIssue(
                        category="file",
                        issue=f"Short too brief ({duration}s < {self.MIN_DURATION_SHORTS}s)",
                        severity=IssueSeverity.ERROR,
                        score_impact=-15,
                        recommendation="Shorts should be at least 15 seconds"
                    ))
                    score -= 15
                elif duration > self.MAX_DURATION_SHORTS:
                    issues.append(QualityIssue(
                        category="file",
                        issue=f"Short too long ({duration}s > {self.MAX_DURATION_SHORTS}s)",
                        severity=IssueSeverity.CRITICAL,
                        score_impact=-50,
                        recommendation="Shorts must be 60 seconds or less"
                    ))
                    score -= 50
                elif duration < self.OPTIMAL_DURATION_SHORTS - 10:
                    issues.append(QualityIssue(
                        category="file",
                        issue=f"Short duration ({duration}s) below optimal (~30s)",
                        severity=IssueSeverity.INFO,
                        score_impact=-5,
                        recommendation="Research shows 20-35 seconds performs best"
                    ))
                    score -= 5
            else:
                if duration < self.MIN_DURATION_REGULAR:
                    issues.append(QualityIssue(
                        category="file",
                        issue=f"Video too short ({duration}s < {self.MIN_DURATION_REGULAR}s)",
                        severity=IssueSeverity.WARNING,
                        score_impact=-10,
                        recommendation="Regular videos should be at least 1 minute"
                    ))
                    score -= 10
                elif duration > self.MAX_DURATION_REGULAR:
                    issues.append(QualityIssue(
                        category="file",
                        issue="Video exceeds YouTube's 12-hour limit",
                        severity=IssueSeverity.CRITICAL,
                        score_impact=-100,
                        recommendation="Split video into multiple parts"
                    ))
                    score -= 100

        except Exception as e:
            logger.error(f"Could not read video info: {e}")
            issues.append(QualityIssue(
                category="file",
                issue=f"Could not read video file (possibly corrupted): {str(e)[:50]}",
                severity=IssueSeverity.CRITICAL,
                score_impact=-100,
                recommendation="Re-generate the video file"
            ))
            checks["corrupted"] = True
            score -= 100

        return issues, max(0, score), checks

    def _check_content_quality(
        self,
        script_data: Dict[str, Any],
        is_short: bool,
        skip_ai_checks: bool
    ) -> Tuple[List[QualityIssue], int, Dict[str, Any]]:
        """
        Check content quality using script data.

        Returns:
            Tuple of (issues, score, checks_dict)
        """
        issues = []
        checks = {}
        score = 100

        # ============================================================
        # TITLE CHECKS
        # ============================================================
        title = script_data.get("title", "")
        checks["title"] = title
        checks["title_length"] = len(title)

        if not title:
            issues.append(QualityIssue(
                category="content",
                issue="Missing title",
                severity=IssueSeverity.CRITICAL,
                score_impact=-30,
                recommendation="Add a compelling title for the video"
            ))
            score -= 30
        else:
            # Check title length
            if len(title) > self.MAX_TITLE_LENGTH:
                issues.append(QualityIssue(
                    category="content",
                    issue=f"Title too long ({len(title)} > {self.MAX_TITLE_LENGTH} chars)",
                    severity=IssueSeverity.ERROR,
                    score_impact=-15,
                    recommendation="Shorten title to under 100 characters"
                ))
                score -= 15
            elif len(title) < self.MIN_TITLE_LENGTH:
                issues.append(QualityIssue(
                    category="content",
                    issue=f"Title too short ({len(title)} < {self.MIN_TITLE_LENGTH} chars)",
                    severity=IssueSeverity.WARNING,
                    score_impact=-10,
                    recommendation="Make title more descriptive"
                ))
                score -= 10

            # Check for generic titles
            title_lower = title.lower()
            if any(generic in title_lower for generic in self.GENERIC_TITLE_PATTERNS):
                issues.append(QualityIssue(
                    category="content",
                    issue="Title appears generic or placeholder",
                    severity=IssueSeverity.WARNING,
                    score_impact=-10,
                    recommendation="Use a specific, compelling title with keywords"
                ))
                score -= 10

            # Check for engagement elements
            has_numbers = any(c.isdigit() for c in title)
            has_power_words = any(word in title_lower for word in [
                "how", "why", "what", "secret", "proven", "best", "top",
                "ultimate", "complete", "guide", "free", "fast", "easy"
            ])

            checks["title_has_numbers"] = has_numbers
            checks["title_has_power_words"] = has_power_words

            if not has_numbers and not has_power_words:
                issues.append(QualityIssue(
                    category="content",
                    issue="Title lacks engagement triggers (numbers or power words)",
                    severity=IssueSeverity.INFO,
                    score_impact=-5,
                    recommendation="Add numbers or power words to boost CTR"
                ))
                score -= 5

        # ============================================================
        # DESCRIPTION CHECKS
        # ============================================================
        description = script_data.get("description", "")
        checks["description_length"] = len(description)

        if not description:
            issues.append(QualityIssue(
                category="content",
                issue="Missing description",
                severity=IssueSeverity.ERROR,
                score_impact=-20,
                recommendation="Add a detailed description with timestamps and keywords"
            ))
            score -= 20
        elif len(description) < self.MIN_DESCRIPTION_LENGTH:
            issues.append(QualityIssue(
                category="content",
                issue=f"Description too short ({len(description)} < {self.MIN_DESCRIPTION_LENGTH} chars)",
                severity=IssueSeverity.WARNING,
                score_impact=-10,
                recommendation="Expand description with timestamps and relevant keywords"
            ))
            score -= 10
        elif len(description) < self.OPTIMAL_DESCRIPTION_LENGTH:
            issues.append(QualityIssue(
                category="content",
                issue="Description could be more detailed",
                severity=IssueSeverity.INFO,
                score_impact=-3,
                recommendation="Add timestamps, links, and social media handles"
            ))
            score -= 3

        # Check for timestamps in description
        has_timestamps = "0:00" in description or "00:00" in description
        checks["description_has_timestamps"] = has_timestamps
        if not has_timestamps and not is_short:
            issues.append(QualityIssue(
                category="content",
                issue="Description lacks timestamps",
                severity=IssueSeverity.INFO,
                score_impact=-3,
                recommendation="Add chapter timestamps for better user experience"
            ))
            score -= 3

        # ============================================================
        # TAGS CHECKS
        # ============================================================
        tags = script_data.get("tags", [])
        checks["tags_count"] = len(tags)
        checks["tags"] = tags

        if not tags:
            issues.append(QualityIssue(
                category="content",
                issue="No tags provided",
                severity=IssueSeverity.ERROR,
                score_impact=-15,
                recommendation="Add 10-15 relevant tags for discoverability"
            ))
            score -= 15
        elif len(tags) < 5:
            issues.append(QualityIssue(
                category="content",
                issue=f"Too few tags ({len(tags)} < 5)",
                severity=IssueSeverity.WARNING,
                score_impact=-8,
                recommendation="Add more relevant tags (aim for 10-15)"
            ))
            score -= 8
        elif len(tags) > 30:
            issues.append(QualityIssue(
                category="content",
                issue=f"Too many tags ({len(tags)} > 30)",
                severity=IssueSeverity.INFO,
                score_impact=-3,
                recommendation="YouTube recommends 10-15 highly relevant tags"
            ))
            score -= 3

        # ============================================================
        # ENGAGEMENT HOOK CHECK
        # ============================================================
        sections = script_data.get("sections", [])
        full_narration = script_data.get("full_narration", "")

        # Check hook in first section(s)
        if sections:
            hook_text = ""
            hook_duration = 0
            for section in sections:
                if hook_duration >= self.HOOK_WINDOW_SECONDS:
                    break
                narration = section.get("narration", "")
                duration = section.get("duration_seconds", 10)
                hook_text += " " + narration
                hook_duration += duration

            hook_words = len(hook_text.split())
            checks["hook_word_count"] = hook_words
            checks["hook_text_preview"] = hook_text[:200] + "..." if len(hook_text) > 200 else hook_text

            if hook_words < self.MIN_HOOK_WORDS:
                issues.append(QualityIssue(
                    category="content",
                    issue=f"Hook too short ({hook_words} < {self.MIN_HOOK_WORDS} words)",
                    severity=IssueSeverity.WARNING,
                    score_impact=-10,
                    recommendation="First 15 seconds should have a compelling hook"
                ))
                score -= 10

            # Check for engagement patterns in hook
            hook_lower = hook_text.lower()
            engagement_patterns = [
                "you", "your", "?",  # Direct address and questions
                "secret", "mistake", "wrong", "truth",  # Curiosity triggers
                "never", "always", "everyone", "nobody",  # Absolutes
            ]
            engagement_count = sum(1 for p in engagement_patterns if p in hook_lower)
            checks["hook_engagement_elements"] = engagement_count

            if engagement_count < 2:
                issues.append(QualityIssue(
                    category="content",
                    issue="Hook lacks engagement elements",
                    severity=IssueSeverity.WARNING,
                    score_impact=-8,
                    recommendation="Use questions, 'you', or curiosity triggers in the hook"
                ))
                score -= 8

        # ============================================================
        # AI-BASED CONTENT ANALYSIS (if enabled)
        # ============================================================
        if not skip_ai_checks and full_narration:
            ai_issues, ai_score, ai_checks = self._ai_content_analysis(
                title, description, full_narration, tags, is_short
            )
            issues.extend(ai_issues)
            score = min(score, score - (100 - ai_score) // 2)  # AI impact is 50% weighted
            checks.update(ai_checks)

        return issues, max(0, score), checks

    def _ai_content_analysis(
        self,
        title: str,
        description: str,
        narration: str,
        tags: List[str],
        is_short: bool
    ) -> Tuple[List[QualityIssue], int, Dict[str, Any]]:
        """
        Use AI to analyze content quality.

        Returns:
            Tuple of (issues, score, checks_dict)
        """
        issues = []
        checks = {"ai_analysis": True}
        score = 100

        ai = self._get_ai_provider()
        if not ai:
            checks["ai_analysis"] = False
            checks["ai_error"] = "AI provider not available"
            return issues, score, checks

        try:
            # Build analysis prompt
            prompt = f"""Analyze this YouTube {'Short' if is_short else 'video'} content for quality:

TITLE: {title}

DESCRIPTION (first 500 chars): {description[:500]}

SCRIPT NARRATION (first 1000 chars): {narration[:1000]}

TAGS: {', '.join(tags[:15])}

Analyze and respond with ONLY a JSON object (no markdown, no explanation):
{{
    "title_score": 0-10,
    "title_feedback": "brief feedback",
    "hook_score": 0-10,
    "hook_feedback": "brief feedback on first 15 seconds engagement",
    "description_score": 0-10,
    "description_feedback": "brief feedback",
    "tags_relevance_score": 0-10,
    "tags_feedback": "brief feedback",
    "overall_engagement_score": 0-10,
    "top_issue": "single biggest improvement needed"
}}"""

            response = ai.generate(prompt, max_tokens=500)

            # Parse JSON from response
            # Try to extract JSON if wrapped in code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            analysis = json.loads(response.strip())
            checks["ai_analysis_result"] = analysis

            # Convert scores to issues
            title_score = analysis.get("title_score", 7)
            if title_score < 6:
                issues.append(QualityIssue(
                    category="content",
                    issue=f"AI: Title needs improvement ({title_score}/10)",
                    severity=IssueSeverity.WARNING,
                    score_impact=-8,
                    recommendation=analysis.get("title_feedback", "Improve title")
                ))
                score -= 8

            hook_score = analysis.get("hook_score", 7)
            if hook_score < 6:
                issues.append(QualityIssue(
                    category="content",
                    issue=f"AI: Hook/engagement weak ({hook_score}/10)",
                    severity=IssueSeverity.WARNING,
                    score_impact=-10,
                    recommendation=analysis.get("hook_feedback", "Strengthen hook")
                ))
                score -= 10

            description_score = analysis.get("description_score", 7)
            if description_score < 5:
                issues.append(QualityIssue(
                    category="content",
                    issue=f"AI: Description needs work ({description_score}/10)",
                    severity=IssueSeverity.INFO,
                    score_impact=-5,
                    recommendation=analysis.get("description_feedback", "Improve description")
                ))
                score -= 5

            tags_score = analysis.get("tags_relevance_score", 7)
            if tags_score < 5:
                issues.append(QualityIssue(
                    category="content",
                    issue=f"AI: Tags not optimal ({tags_score}/10)",
                    severity=IssueSeverity.INFO,
                    score_impact=-5,
                    recommendation=analysis.get("tags_feedback", "Use more relevant tags")
                ))
                score -= 5

            # Overall engagement
            engagement_score = analysis.get("overall_engagement_score", 7)
            checks["ai_engagement_score"] = engagement_score
            if engagement_score < 5:
                top_issue = analysis.get("top_issue", "Improve overall engagement")
                issues.append(QualityIssue(
                    category="content",
                    issue=f"AI: Low engagement potential ({engagement_score}/10)",
                    severity=IssueSeverity.ERROR,
                    score_impact=-15,
                    recommendation=top_issue
                ))
                score -= 15

        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse AI response: {e}")
            checks["ai_error"] = "Failed to parse AI response"
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            checks["ai_error"] = str(e)

        return issues, max(0, score), checks

    def _check_technical_quality(
        self,
        video_file: str,
        is_short: bool
    ) -> Tuple[List[QualityIssue], int, Dict[str, Any]]:
        """
        Check technical quality of the video.

        Returns:
            Tuple of (issues, score, checks_dict)
        """
        issues = []
        checks = {}
        score = 100

        try:
            # Get video info
            duration, width, height = self._get_video_info(video_file)
            checks["resolution"] = f"{width}x{height}"
            checks["duration"] = duration

            # ============================================================
            # RESOLUTION CHECK
            # ============================================================
            expected_res = self.RESOLUTION_SHORTS if is_short else self.RESOLUTION_REGULAR

            if (width, height) != expected_res:
                # Check if aspect ratio is at least correct
                expected_aspect = expected_res[0] / expected_res[1]
                actual_aspect = width / height
                aspect_diff = abs(expected_aspect - actual_aspect)

                if aspect_diff > 0.1:  # More than 10% aspect ratio difference
                    issues.append(QualityIssue(
                        category="technical",
                        issue=f"Wrong aspect ratio ({width}x{height}, expected {expected_res[0]}x{expected_res[1]})",
                        severity=IssueSeverity.ERROR if is_short else IssueSeverity.WARNING,
                        score_impact=-20 if is_short else -10,
                        recommendation=f"Use {'9:16 portrait' if is_short else '16:9 landscape'} aspect ratio"
                    ))
                    score -= 20 if is_short else 10
                elif width < expected_res[0] or height < expected_res[1]:
                    issues.append(QualityIssue(
                        category="technical",
                        issue=f"Resolution below recommended ({width}x{height} < {expected_res[0]}x{expected_res[1]})",
                        severity=IssueSeverity.WARNING,
                        score_impact=-10,
                        recommendation="Use higher resolution for better quality"
                    ))
                    score -= 10
            else:
                checks["resolution_correct"] = True

            # ============================================================
            # AUDIO CHECK
            # ============================================================
            has_audio, audio_level, silent_gaps = self._check_audio(video_file)
            checks["has_audio"] = has_audio
            checks["audio_level_db"] = audio_level
            checks["silent_gaps"] = silent_gaps

            if not has_audio:
                issues.append(QualityIssue(
                    category="technical",
                    issue="No audio track detected",
                    severity=IssueSeverity.CRITICAL,
                    score_impact=-50,
                    recommendation="Ensure video has audio/narration"
                ))
                score -= 50
            else:
                # Check audio level
                if audio_level is not None and audio_level < self.MIN_AUDIO_LEVEL_DB:
                    issues.append(QualityIssue(
                        category="technical",
                        issue=f"Audio level very low ({audio_level:.1f}dB)",
                        severity=IssueSeverity.WARNING,
                        score_impact=-10,
                        recommendation="Increase audio volume or check narration"
                    ))
                    score -= 10

                # Check for long silent gaps
                long_gaps = [gap for gap in silent_gaps if gap['duration'] > self.MAX_SILENT_GAP_SECONDS]
                if long_gaps:
                    max_gap = max(g['duration'] for g in long_gaps)
                    issues.append(QualityIssue(
                        category="technical",
                        issue=f"Silent gap detected ({max_gap:.1f}s > {self.MAX_SILENT_GAP_SECONDS}s)",
                        severity=IssueSeverity.WARNING,
                        score_impact=-8,
                        recommendation="Fill silent gaps with music or narration"
                    ))
                    score -= 8
                    checks["long_silent_gaps"] = len(long_gaps)

            # ============================================================
            # FRAMERATE CHECK
            # ============================================================
            fps = self._get_video_fps(video_file)
            checks["fps"] = fps

            if fps and fps < 24:
                issues.append(QualityIssue(
                    category="technical",
                    issue=f"Low framerate ({fps} fps < 24 fps)",
                    severity=IssueSeverity.WARNING,
                    score_impact=-8,
                    recommendation="Use at least 24 fps for smooth playback"
                ))
                score -= 8

        except Exception as e:
            logger.error(f"Technical check failed: {e}")
            checks["error"] = str(e)
            issues.append(QualityIssue(
                category="technical",
                issue=f"Could not complete technical checks: {str(e)[:50]}",
                severity=IssueSeverity.WARNING,
                score_impact=-10,
                recommendation="Verify video file is valid"
            ))
            score -= 10

        return issues, max(0, score), checks

    def _get_video_info(self, video_file: str) -> Tuple[float, int, int]:
        """
        Get video duration, width, and height using ffprobe.

        Returns:
            Tuple of (duration_seconds, width, height)
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            video_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)

        # Get dimensions from stream
        stream = data.get("streams", [{}])[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

        # Get duration (prefer stream duration, fallback to format duration)
        duration = float(stream.get("duration", 0))
        if duration == 0:
            duration = float(data.get("format", {}).get("duration", 0))

        return duration, width, height

    def _get_video_fps(self, video_file: str) -> Optional[float]:
        """Get video framerate using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "json",
                video_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            fps_str = data.get("streams", [{}])[0].get("r_frame_rate", "0/1")

            # Parse fraction like "30/1" or "30000/1001"
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den) if float(den) != 0 else None
            return float(fps_str)
        except Exception:
            return None

    def _check_audio(
        self,
        video_file: str
    ) -> Tuple[bool, Optional[float], List[Dict]]:
        """
        Check audio presence, level, and silent gaps.

        Returns:
            Tuple of (has_audio, mean_volume_db, silent_gaps)
        """
        # Check if audio stream exists
        cmd_check = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            video_file
        ]

        result = subprocess.run(cmd_check, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, None, []

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        has_audio = len(streams) > 0

        if not has_audio:
            return False, None, []

        # Get audio levels using ffmpeg volumedetect
        audio_level = None
        try:
            cmd_volume = [
                "ffmpeg",
                "-i", video_file,
                "-af", "volumedetect",
                "-f", "null",
                "-y",
                "NUL" if os.name == "nt" else "/dev/null"
            ]

            result = subprocess.run(
                cmd_volume,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parse mean volume from stderr
            for line in result.stderr.split("\n"):
                if "mean_volume:" in line:
                    parts = line.split("mean_volume:")[1].split()
                    audio_level = float(parts[0])
                    break
        except Exception as e:
            logger.warning(f"Could not get audio level: {e}")

        # Detect silent gaps using silencedetect filter
        silent_gaps = []
        try:
            cmd_silence = [
                "ffmpeg",
                "-i", video_file,
                "-af", f"silencedetect=noise=-40dB:d={self.MAX_SILENT_GAP_SECONDS}",
                "-f", "null",
                "-y",
                "NUL" if os.name == "nt" else "/dev/null"
            ]

            result = subprocess.run(
                cmd_silence,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parse silence_start and silence_end from stderr
            current_start = None
            for line in result.stderr.split("\n"):
                if "silence_start:" in line:
                    parts = line.split("silence_start:")[1].split()
                    current_start = float(parts[0])
                elif "silence_end:" in line and current_start is not None:
                    parts = line.split("silence_end:")[1].split()
                    end = float(parts[0])
                    duration_parts = line.split("silence_duration:")[1].split() if "silence_duration:" in line else None
                    duration = float(duration_parts[0]) if duration_parts else end - current_start

                    silent_gaps.append({
                        "start": current_start,
                        "end": end,
                        "duration": duration
                    })
                    current_start = None
        except Exception as e:
            logger.warning(f"Could not detect silent gaps: {e}")

        return has_audio, audio_level, silent_gaps


# Convenience function for quick checks
def quick_quality_check(
    video_file: str,
    script_data: Optional[Dict[str, Any]] = None,
    is_short: bool = False,
    threshold: int = 70
) -> Tuple[bool, int, str]:
    """
    Quick quality check that returns pass/fail, score, and summary.

    Args:
        video_file: Path to video file
        script_data: Optional script data
        is_short: Whether this is a YouTube Short
        threshold: Passing threshold (default 70)

    Returns:
        Tuple of (passed, score, summary_message)
    """
    checker = VideoQualityChecker()
    report = checker.check_video(
        video_file=video_file,
        script_data=script_data,
        is_short=is_short,
        threshold=threshold,
        skip_ai_checks=True  # Skip AI for quick checks
    )

    return report.passed, report.overall_score, report.summary()


if __name__ == "__main__":
    # Test the quality checker
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        is_short = "--short" in sys.argv

        checker = VideoQualityChecker()
        report = checker.check_video(
            video_file=video_path,
            is_short=is_short,
            skip_ai_checks="--no-ai" in sys.argv
        )

        print("\n" + "="*60)
        print(report.summary())
        print("="*60)

        # Print detailed JSON if requested
        if "--json" in sys.argv:
            print("\nDetailed Report:")
            print(json.dumps(report.to_dict(), indent=2))
    else:
        print("Usage: python -m src.content.quality_checker <video_file> [--short] [--no-ai] [--json]")
