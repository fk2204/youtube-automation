"""
Validator Agent - Content Validation Gate

A production agent for running comprehensive pre-publish validation checks
on video content before uploading to YouTube.

Features:
- Comprehensive pre-publish checklist enforcement
- Script quality validation (target score >= 75)
- Video technical specs validation via VideoQualityAgent
- Audio level verification via AudioQualityAgent
- Thumbnail requirements validation (1280x720, < 2MB)
- Metadata completeness checks (title, description, tags)
- Returns pass/fail with blockers and warnings

Usage:
    from src.agents.validator_agent import ValidatorAgent, ValidationResult

    agent = ValidatorAgent()
    result = agent.run(
        script_data={"title": "...", "description": "...", "tags": [...]},
        video_file="output/video.mp4",
        thumbnail_file="assets/thumbnail.png",
        niche="finance"
    )

    if result.success:
        validation = result.data
        if validation['passed']:
            print("Ready to publish!")
        else:
            print(f"Blockers: {validation['blockers']}")

CLI:
    python -m src.agents.validator_agent --video output/video.mp4 --thumbnail thumb.png
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger

from .base_agent import BaseAgent, AgentResult


# Validation thresholds
VALIDATION_THRESHOLDS = {
    "script_quality_min": 75,  # Minimum script quality score (0-100)
    "thumbnail_width": 1280,
    "thumbnail_height": 720,
    "thumbnail_max_size_kb": 2048,  # 2MB max
    "thumbnail_min_size_kb": 50,    # Sanity check
    "video_min_duration": 60,       # 1 minute for regular videos
    "video_max_duration": 43200,    # 12 hours (YouTube limit)
    "short_max_duration": 60,       # 60 seconds for Shorts
    "short_min_duration": 15,       # 15 seconds minimum
    "audio_min_level_db": -50,      # Minimum audio level
    "audio_max_level_db": -10,      # Maximum (too loud = clipping)
    "title_max_length": 100,
    "title_min_length": 10,
    "description_min_length": 50,
    "tags_min_count": 5,
    "tags_max_count": 30,
}

# Check categories and weights for scoring
CHECK_WEIGHTS = {
    "script_quality": {"weight": 25, "category": "content"},
    "video_technical": {"weight": 20, "category": "technical"},
    "audio_quality": {"weight": 15, "category": "technical"},
    "thumbnail_specs": {"weight": 15, "category": "assets"},
    "title_quality": {"weight": 10, "category": "metadata"},
    "description_complete": {"weight": 8, "category": "metadata"},
    "tags_complete": {"weight": 7, "category": "metadata"},
}


@dataclass
class ValidationCheck:
    """
    Individual validation check result.

    Attributes:
        name: Name of the check
        passed: Whether the check passed
        score: Score for this check (0-100)
        category: Check category (content, technical, assets, metadata)
        is_blocker: Whether failure blocks publishing
        message: Description of result
        details: Additional details
    """
    name: str
    passed: bool
    score: int = 100
    category: str = "general"
    is_blocker: bool = False
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """
    Complete validation result.

    Attributes:
        passed: Overall pass/fail status
        score: Overall score (0-100)
        blockers: List of blocking issues that must be fixed
        warnings: List of non-blocking issues
        checks: All individual check results
        ready_to_publish: Whether content is ready for publishing
        categories: Scores by category
    """
    passed: bool
    score: int = 0
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: List[ValidationCheck] = field(default_factory=list)
    ready_to_publish: bool = False
    categories: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["checks"] = [check.to_dict() for check in self.checks]
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "READY" if self.ready_to_publish else "NOT READY"
        passed_checks = sum(1 for c in self.checks if c.passed)

        lines = [
            f"Validation: {status}",
            f"Score: {self.score}/100",
            f"Checks: {passed_checks}/{len(self.checks)} passed",
        ]

        if self.blockers:
            lines.append(f"\nBlockers ({len(self.blockers)}):")
            for blocker in self.blockers[:5]:
                lines.append(f"  [X] {blocker}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings[:5]:
                lines.append(f"  [!] {warning}")

        lines.append("\nCategories:")
        for cat, score in self.categories.items():
            lines.append(f"  {cat}: {score}/100")

        return "\n".join(lines)


class ValidatorAgent(BaseAgent):
    """
    Production agent for comprehensive content validation.

    Acts as a quality gate before publishing to YouTube, checking:
    - Script quality and engagement metrics
    - Video technical specifications
    - Audio levels and quality
    - Thumbnail specifications
    - Metadata completeness
    """

    def __init__(self, provider: str = "groq", api_key: str = None):
        """
        Initialize the ValidatorAgent.

        Args:
            provider: AI provider (for quality analysis)
            api_key: API key for provider
        """
        super().__init__(provider=provider, api_key=api_key)
        self.name = "ValidatorAgent"

        # Lazy-load checkers
        self._video_checker = None
        self._quality_agent = None

        logger.info(f"ValidatorAgent initialized")

    def _get_video_checker(self):
        """Lazy-load VideoQualityChecker."""
        if self._video_checker is None:
            try:
                from ..content.quality_checker import VideoQualityChecker
                self._video_checker = VideoQualityChecker()
            except ImportError as e:
                logger.warning(f"Could not import VideoQualityChecker: {e}")
        return self._video_checker

    def _get_quality_agent(self):
        """Lazy-load QualityAgent for script analysis."""
        if self._quality_agent is None:
            try:
                from .quality_agent import QualityAgent
                self._quality_agent = QualityAgent(provider=self.provider, api_key=self.api_key)
            except ImportError as e:
                logger.warning(f"Could not import QualityAgent: {e}")
        return self._quality_agent

    def run(
        self,
        script_data: Dict[str, Any] = None,
        script_text: str = None,
        video_file: str = None,
        thumbnail_file: str = None,
        niche: str = "default",
        is_short: bool = False,
        min_score: int = 75,
        strict_mode: bool = False,
        **kwargs
    ) -> AgentResult:
        """
        Run comprehensive validation on video content.

        Args:
            script_data: Dict with title, description, tags, sections
            script_text: Raw script text (alternative to script_data)
            video_file: Path to video file
            thumbnail_file: Path to thumbnail file
            niche: Content niche for context-aware validation
            is_short: Whether this is a YouTube Short
            min_score: Minimum score to pass (default: 75)
            strict_mode: Treat warnings as blockers
            **kwargs: Additional parameters

        Returns:
            AgentResult containing ValidationResult
        """
        logger.info(f"[ValidatorAgent] Running validation (niche={niche}, short={is_short})")

        checks: List[ValidationCheck] = []
        blockers: List[str] = []
        warnings: List[str] = []
        category_scores: Dict[str, List[int]] = {
            "content": [],
            "technical": [],
            "assets": [],
            "metadata": []
        }

        # Extract script data if only text provided
        if script_text and not script_data:
            script_data = {"full_narration": script_text}

        # ============================================================
        # 1. SCRIPT QUALITY VALIDATION
        # ============================================================
        if script_data or script_text:
            script_check = self._validate_script_quality(
                script_data=script_data,
                script_text=script_text,
                niche=niche,
                is_short=is_short
            )
            checks.append(script_check)
            category_scores["content"].append(script_check.score)

            if not script_check.passed:
                if script_check.is_blocker:
                    blockers.append(script_check.message)
                else:
                    warnings.append(script_check.message)

        # ============================================================
        # 2. VIDEO TECHNICAL VALIDATION
        # ============================================================
        if video_file:
            video_check = self._validate_video_technical(
                video_file=video_file,
                is_short=is_short,
                script_data=script_data
            )
            checks.append(video_check)
            category_scores["technical"].append(video_check.score)

            if not video_check.passed:
                if video_check.is_blocker:
                    blockers.append(video_check.message)
                else:
                    warnings.append(video_check.message)

            # Audio quality check
            audio_check = self._validate_audio_quality(
                video_file=video_file
            )
            checks.append(audio_check)
            category_scores["technical"].append(audio_check.score)

            if not audio_check.passed:
                if audio_check.is_blocker:
                    blockers.append(audio_check.message)
                else:
                    warnings.append(audio_check.message)

        # ============================================================
        # 3. THUMBNAIL VALIDATION
        # ============================================================
        if thumbnail_file:
            thumb_check = self._validate_thumbnail(
                thumbnail_file=thumbnail_file
            )
            checks.append(thumb_check)
            category_scores["assets"].append(thumb_check.score)

            if not thumb_check.passed:
                if thumb_check.is_blocker:
                    blockers.append(thumb_check.message)
                else:
                    warnings.append(thumb_check.message)

        # ============================================================
        # 4. METADATA VALIDATION
        # ============================================================
        if script_data:
            # Title validation
            title_check = self._validate_title(
                title=script_data.get("title", ""),
                niche=niche
            )
            checks.append(title_check)
            category_scores["metadata"].append(title_check.score)

            if not title_check.passed:
                if title_check.is_blocker:
                    blockers.append(title_check.message)
                else:
                    warnings.append(title_check.message)

            # Description validation
            desc_check = self._validate_description(
                description=script_data.get("description", "")
            )
            checks.append(desc_check)
            category_scores["metadata"].append(desc_check.score)

            if not desc_check.passed:
                if desc_check.is_blocker:
                    blockers.append(desc_check.message)
                else:
                    warnings.append(desc_check.message)

            # Tags validation
            tags_check = self._validate_tags(
                tags=script_data.get("tags", []),
                niche=niche
            )
            checks.append(tags_check)
            category_scores["metadata"].append(tags_check.score)

            if not tags_check.passed:
                if tags_check.is_blocker:
                    blockers.append(tags_check.message)
                else:
                    warnings.append(tags_check.message)

        # ============================================================
        # CALCULATE OVERALL SCORE
        # ============================================================
        # Calculate weighted score
        total_weight = 0
        weighted_sum = 0

        for check in checks:
            check_info = CHECK_WEIGHTS.get(check.name.lower().replace(" ", "_"))
            if check_info:
                weight = check_info["weight"]
            else:
                weight = 10  # Default weight

            total_weight += weight
            weighted_sum += check.score * weight

        overall_score = int(weighted_sum / total_weight) if total_weight > 0 else 0

        # Calculate category scores
        categories = {}
        for cat, scores in category_scores.items():
            if scores:
                categories[cat] = int(sum(scores) / len(scores))
            else:
                categories[cat] = 100  # No checks = assume good

        # Determine pass/fail
        if strict_mode:
            blockers.extend(warnings)
            warnings = []

        passed = len(blockers) == 0 and overall_score >= min_score
        ready_to_publish = passed

        # Create validation result
        validation_result = ValidationResult(
            passed=passed,
            score=overall_score,
            blockers=blockers,
            warnings=warnings,
            checks=checks,
            ready_to_publish=ready_to_publish,
            categories=categories
        )

        # Log operation
        self.log_operation("validate", tokens=0, cost=0.0)

        status = "PASSED" if passed else "FAILED"
        logger.info(f"[ValidatorAgent] Validation {status}: {overall_score}/100")

        return AgentResult(
            success=True,
            data=validation_result.to_dict(),
            tokens_used=0,
            cost=0.0,
            metadata={
                "niche": niche,
                "is_short": is_short,
                "min_score": min_score,
                "strict_mode": strict_mode
            }
        )

    def _validate_script_quality(
        self,
        script_data: Dict[str, Any] = None,
        script_text: str = None,
        niche: str = "default",
        is_short: bool = False
    ) -> ValidationCheck:
        """
        Validate script quality using QualityAgent.

        Args:
            script_data: Script data dictionary
            script_text: Raw script text
            niche: Content niche
            is_short: Whether this is a Short

        Returns:
            ValidationCheck result
        """
        # Get script text
        text = script_text or script_data.get("full_narration", "")
        if not text and script_data.get("sections"):
            # Combine section narrations
            sections = script_data["sections"]
            text = " ".join(
                s.get("narration", "") if isinstance(s, dict) else getattr(s, "narration", "")
                for s in sections
            )

        if not text:
            return ValidationCheck(
                name="Script Quality",
                passed=False,
                score=0,
                category="content",
                is_blocker=True,
                message="No script content provided",
                details={}
            )

        # Use QualityAgent for analysis
        quality_agent = self._get_quality_agent()
        if quality_agent:
            try:
                result = quality_agent.quick_check(text, niche=niche, is_short=is_short)
                score = result.overall_score
                passed = score >= VALIDATION_THRESHOLDS["script_quality_min"]

                return ValidationCheck(
                    name="Script Quality",
                    passed=passed,
                    score=score,
                    category="content",
                    is_blocker=not passed,
                    message=f"Script quality: {score}/100 (min: {VALIDATION_THRESHOLDS['script_quality_min']})",
                    details={
                        "score": score,
                        "checks_passed": sum(1 for c in result.checks if c.passed),
                        "total_checks": len(result.checks),
                        "recommendations": result.recommendations[:3]
                    }
                )
            except Exception as e:
                logger.warning(f"Quality check failed: {e}")

        # Fallback: basic word count check
        word_count = len(text.split())
        if is_short:
            expected_range = (30, 200)
        else:
            expected_range = (500, 3000)

        score = 100 if expected_range[0] <= word_count <= expected_range[1] else 60
        passed = score >= VALIDATION_THRESHOLDS["script_quality_min"]

        return ValidationCheck(
            name="Script Quality",
            passed=passed,
            score=score,
            category="content",
            is_blocker=not passed,
            message=f"Script word count: {word_count} (expected: {expected_range[0]}-{expected_range[1]})",
            details={"word_count": word_count}
        )

    def _validate_video_technical(
        self,
        video_file: str,
        is_short: bool,
        script_data: Dict[str, Any] = None
    ) -> ValidationCheck:
        """
        Validate video technical specifications.

        Args:
            video_file: Path to video file
            is_short: Whether this is a Short
            script_data: Script data for content checks

        Returns:
            ValidationCheck result
        """
        if not os.path.exists(video_file):
            return ValidationCheck(
                name="Video Technical",
                passed=False,
                score=0,
                category="technical",
                is_blocker=True,
                message=f"Video file not found: {video_file}",
                details={}
            )

        video_checker = self._get_video_checker()
        if video_checker:
            try:
                threshold = VALIDATION_THRESHOLDS["script_quality_min"]
                report = video_checker.check_video(
                    video_file=video_file,
                    script_data=script_data,
                    is_short=is_short,
                    threshold=threshold,
                    skip_ai_checks=True
                )

                return ValidationCheck(
                    name="Video Technical",
                    passed=report.passed,
                    score=report.overall_score,
                    category="technical",
                    is_blocker=not report.passed,
                    message=f"Video quality: {report.overall_score}/100",
                    details={
                        "file_checks": report.file_checks,
                        "technical_checks": report.technical_checks,
                        "issues_count": len(report.issues)
                    }
                )
            except Exception as e:
                logger.warning(f"Video check failed: {e}")

        # Fallback: basic file checks
        try:
            file_size = os.path.getsize(video_file)
            file_size_mb = file_size / (1024 * 1024)

            # Check file size (min 5MB for regular, 2MB for shorts)
            min_size = 2 if is_short else 5
            if file_size_mb < min_size:
                return ValidationCheck(
                    name="Video Technical",
                    passed=False,
                    score=50,
                    category="technical",
                    is_blocker=True,
                    message=f"Video file too small ({file_size_mb:.1f}MB < {min_size}MB)",
                    details={"file_size_mb": file_size_mb}
                )

            return ValidationCheck(
                name="Video Technical",
                passed=True,
                score=80,
                category="technical",
                is_blocker=False,
                message=f"Video file size OK ({file_size_mb:.1f}MB)",
                details={"file_size_mb": file_size_mb}
            )
        except Exception as e:
            return ValidationCheck(
                name="Video Technical",
                passed=False,
                score=0,
                category="technical",
                is_blocker=True,
                message=f"Could not check video: {str(e)[:50]}",
                details={}
            )

    def _validate_audio_quality(self, video_file: str) -> ValidationCheck:
        """
        Validate audio levels in video.

        Args:
            video_file: Path to video file

        Returns:
            ValidationCheck result
        """
        try:
            # Use ffprobe to check for audio stream
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                video_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return ValidationCheck(
                    name="Audio Quality",
                    passed=False,
                    score=0,
                    category="technical",
                    is_blocker=True,
                    message="Could not analyze audio",
                    details={"error": result.stderr[:100]}
                )

            data = json.loads(result.stdout)
            streams = data.get("streams", [])

            if not streams:
                return ValidationCheck(
                    name="Audio Quality",
                    passed=False,
                    score=0,
                    category="technical",
                    is_blocker=True,
                    message="No audio track found in video",
                    details={}
                )

            # Audio exists, try to get volume level
            try:
                vol_cmd = [
                    "ffmpeg",
                    "-i", video_file,
                    "-af", "volumedetect",
                    "-f", "null",
                    "NUL" if os.name == "nt" else "/dev/null"
                ]

                vol_result = subprocess.run(vol_cmd, capture_output=True, text=True, timeout=120)

                mean_volume = None
                for line in vol_result.stderr.split("\n"):
                    if "mean_volume:" in line:
                        parts = line.split("mean_volume:")[1].split()
                        mean_volume = float(parts[0])
                        break

                if mean_volume is not None:
                    min_level = VALIDATION_THRESHOLDS["audio_min_level_db"]
                    max_level = VALIDATION_THRESHOLDS["audio_max_level_db"]

                    if mean_volume < min_level:
                        return ValidationCheck(
                            name="Audio Quality",
                            passed=False,
                            score=40,
                            category="technical",
                            is_blocker=False,
                            message=f"Audio too quiet ({mean_volume:.1f}dB < {min_level}dB)",
                            details={"mean_volume_db": mean_volume}
                        )
                    elif mean_volume > max_level:
                        return ValidationCheck(
                            name="Audio Quality",
                            passed=False,
                            score=60,
                            category="technical",
                            is_blocker=False,
                            message=f"Audio may be clipping ({mean_volume:.1f}dB > {max_level}dB)",
                            details={"mean_volume_db": mean_volume}
                        )
                    else:
                        return ValidationCheck(
                            name="Audio Quality",
                            passed=True,
                            score=100,
                            category="technical",
                            is_blocker=False,
                            message=f"Audio levels OK ({mean_volume:.1f}dB)",
                            details={"mean_volume_db": mean_volume}
                        )
            except:
                pass

            # Fallback: audio exists but couldn't measure level
            return ValidationCheck(
                name="Audio Quality",
                passed=True,
                score=80,
                category="technical",
                is_blocker=False,
                message="Audio track present",
                details={"has_audio": True}
            )

        except Exception as e:
            return ValidationCheck(
                name="Audio Quality",
                passed=False,
                score=50,
                category="technical",
                is_blocker=False,
                message=f"Could not check audio: {str(e)[:50]}",
                details={}
            )

    def _validate_thumbnail(self, thumbnail_file: str) -> ValidationCheck:
        """
        Validate thumbnail specifications.

        Args:
            thumbnail_file: Path to thumbnail file

        Returns:
            ValidationCheck result
        """
        if not os.path.exists(thumbnail_file):
            return ValidationCheck(
                name="Thumbnail Specs",
                passed=False,
                score=0,
                category="assets",
                is_blocker=True,
                message=f"Thumbnail file not found: {thumbnail_file}",
                details={}
            )

        try:
            from PIL import Image

            # Get file size
            file_size_kb = os.path.getsize(thumbnail_file) / 1024

            # Open image and get dimensions
            with Image.open(thumbnail_file) as img:
                width, height = img.size
                format_type = img.format

            issues = []
            score = 100

            # Check dimensions (must be 1280x720)
            expected_w = VALIDATION_THRESHOLDS["thumbnail_width"]
            expected_h = VALIDATION_THRESHOLDS["thumbnail_height"]

            if width != expected_w or height != expected_h:
                issues.append(f"Wrong dimensions: {width}x{height} (expected {expected_w}x{expected_h})")
                score -= 20

            # Check file size (must be < 2MB)
            max_size = VALIDATION_THRESHOLDS["thumbnail_max_size_kb"]
            if file_size_kb > max_size:
                issues.append(f"File too large: {file_size_kb:.0f}KB (max {max_size}KB)")
                score -= 30

            min_size = VALIDATION_THRESHOLDS["thumbnail_min_size_kb"]
            if file_size_kb < min_size:
                issues.append(f"File suspiciously small: {file_size_kb:.0f}KB")
                score -= 10

            passed = len(issues) == 0
            is_blocker = score < 70

            return ValidationCheck(
                name="Thumbnail Specs",
                passed=passed,
                score=max(0, score),
                category="assets",
                is_blocker=is_blocker,
                message=issues[0] if issues else f"Thumbnail OK ({width}x{height}, {file_size_kb:.0f}KB)",
                details={
                    "width": width,
                    "height": height,
                    "file_size_kb": file_size_kb,
                    "format": format_type,
                    "issues": issues
                }
            )

        except ImportError:
            # PIL not available, just check file exists and size
            file_size_kb = os.path.getsize(thumbnail_file) / 1024
            max_size = VALIDATION_THRESHOLDS["thumbnail_max_size_kb"]

            if file_size_kb > max_size:
                return ValidationCheck(
                    name="Thumbnail Specs",
                    passed=False,
                    score=50,
                    category="assets",
                    is_blocker=True,
                    message=f"Thumbnail too large: {file_size_kb:.0f}KB (max {max_size}KB)",
                    details={"file_size_kb": file_size_kb}
                )

            return ValidationCheck(
                name="Thumbnail Specs",
                passed=True,
                score=70,
                category="assets",
                is_blocker=False,
                message=f"Thumbnail file exists ({file_size_kb:.0f}KB) - dimensions not verified",
                details={"file_size_kb": file_size_kb}
            )

        except Exception as e:
            return ValidationCheck(
                name="Thumbnail Specs",
                passed=False,
                score=0,
                category="assets",
                is_blocker=True,
                message=f"Could not validate thumbnail: {str(e)[:50]}",
                details={}
            )

    def _validate_title(self, title: str, niche: str) -> ValidationCheck:
        """
        Validate video title.

        Args:
            title: Video title
            niche: Content niche

        Returns:
            ValidationCheck result
        """
        if not title:
            return ValidationCheck(
                name="Title Quality",
                passed=False,
                score=0,
                category="metadata",
                is_blocker=True,
                message="Title is missing",
                details={}
            )

        issues = []
        score = 100

        # Length checks
        if len(title) > VALIDATION_THRESHOLDS["title_max_length"]:
            issues.append(f"Title too long ({len(title)} chars, max {VALIDATION_THRESHOLDS['title_max_length']})")
            score -= 20
        elif len(title) < VALIDATION_THRESHOLDS["title_min_length"]:
            issues.append(f"Title too short ({len(title)} chars, min {VALIDATION_THRESHOLDS['title_min_length']})")
            score -= 15

        # Check for generic titles
        generic_words = ["video", "untitled", "test", "new video", "clip"]
        if any(word in title.lower() for word in generic_words):
            issues.append("Title appears generic")
            score -= 15

        # Check for engagement elements
        has_number = any(c.isdigit() for c in title)
        power_words = ["how", "why", "secret", "truth", "best", "top", "ultimate"]
        has_power_word = any(word in title.lower() for word in power_words)

        if not has_number and not has_power_word:
            issues.append("Title lacks engagement triggers (numbers or power words)")
            score -= 10

        # Use best practices validation if available
        try:
            from ..utils.best_practices import validate_title
            bp_result = validate_title(title, niche)
            if not bp_result.is_valid:
                issues.extend(bp_result.suggestions[:2])
                score = min(score, int(bp_result.score * 100))
        except:
            pass

        passed = score >= 70 and len(title) > 0
        is_blocker = len(title) == 0 or len(title) > VALIDATION_THRESHOLDS["title_max_length"]

        return ValidationCheck(
            name="Title Quality",
            passed=passed,
            score=max(0, score),
            category="metadata",
            is_blocker=is_blocker,
            message=issues[0] if issues else f"Title OK: {title[:50]}...",
            details={
                "title": title,
                "length": len(title),
                "has_numbers": has_number,
                "has_power_words": has_power_word,
                "issues": issues
            }
        )

    def _validate_description(self, description: str) -> ValidationCheck:
        """
        Validate video description.

        Args:
            description: Video description

        Returns:
            ValidationCheck result
        """
        if not description:
            return ValidationCheck(
                name="Description Complete",
                passed=False,
                score=0,
                category="metadata",
                is_blocker=True,
                message="Description is missing",
                details={}
            )

        issues = []
        score = 100

        # Length check
        if len(description) < VALIDATION_THRESHOLDS["description_min_length"]:
            issues.append(f"Description too short ({len(description)} chars, min {VALIDATION_THRESHOLDS['description_min_length']})")
            score -= 25

        # Check for timestamps
        import re
        has_timestamps = bool(re.search(r'\d{1,2}:\d{2}', description))
        if not has_timestamps and len(description) > 100:
            issues.append("No timestamps/chapters found")
            score -= 10

        # Check for links
        has_links = "http" in description.lower() or "www." in description.lower()

        # Check for keywords
        has_keywords = len(description.split()) > 20

        passed = score >= 60
        is_blocker = len(description) == 0

        return ValidationCheck(
            name="Description Complete",
            passed=passed,
            score=max(0, score),
            category="metadata",
            is_blocker=is_blocker,
            message=issues[0] if issues else f"Description OK ({len(description)} chars)",
            details={
                "length": len(description),
                "has_timestamps": has_timestamps,
                "has_links": has_links,
                "word_count": len(description.split()),
                "issues": issues
            }
        )

    def _validate_tags(self, tags: List[str], niche: str) -> ValidationCheck:
        """
        Validate video tags.

        Args:
            tags: List of tags
            niche: Content niche

        Returns:
            ValidationCheck result
        """
        if not tags:
            return ValidationCheck(
                name="Tags Complete",
                passed=False,
                score=0,
                category="metadata",
                is_blocker=True,
                message="No tags provided",
                details={}
            )

        issues = []
        score = 100

        # Count check
        min_tags = VALIDATION_THRESHOLDS["tags_min_count"]
        max_tags = VALIDATION_THRESHOLDS["tags_max_count"]

        if len(tags) < min_tags:
            issues.append(f"Too few tags ({len(tags)}, min {min_tags})")
            score -= 20
        elif len(tags) > max_tags:
            issues.append(f"Too many tags ({len(tags)}, max {max_tags})")
            score -= 10

        # Check for niche-relevant tags
        niche_keywords = {
            "finance": ["money", "investing", "finance", "wealth", "income", "stock"],
            "psychology": ["psychology", "mind", "brain", "manipulation", "behavior"],
            "storytelling": ["story", "true crime", "documentary", "mystery", "history"],
        }

        relevant_keywords = niche_keywords.get(niche, [])
        tags_lower = [t.lower() for t in tags]

        relevant_count = sum(1 for kw in relevant_keywords if any(kw in t for t in tags_lower))
        if relevant_count < 2 and relevant_keywords:
            issues.append(f"Few niche-relevant tags (found {relevant_count})")
            score -= 10

        passed = score >= 60 and len(tags) > 0
        is_blocker = len(tags) == 0

        return ValidationCheck(
            name="Tags Complete",
            passed=passed,
            score=max(0, score),
            category="metadata",
            is_blocker=is_blocker,
            message=issues[0] if issues else f"Tags OK ({len(tags)} tags)",
            details={
                "count": len(tags),
                "tags_preview": tags[:5],
                "niche_relevant_count": relevant_count,
                "issues": issues
            }
        )

    def quick_validate(
        self,
        video_file: str = None,
        thumbnail_file: str = None,
        title: str = None,
        description: str = None
    ) -> Tuple[bool, int, str]:
        """
        Quick validation returning simple pass/fail.

        Args:
            video_file: Path to video
            thumbnail_file: Path to thumbnail
            title: Video title
            description: Video description

        Returns:
            Tuple of (passed, score, summary_message)
        """
        script_data = {}
        if title:
            script_data["title"] = title
        if description:
            script_data["description"] = description

        result = self.run(
            script_data=script_data if script_data else None,
            video_file=video_file,
            thumbnail_file=thumbnail_file
        )

        if result.success:
            data = result.data
            return data["passed"], data["score"], data.get("blockers", ["OK"])[0] if not data["passed"] else "OK"
        else:
            return False, 0, result.error


# CLI entry point
def main():
    """CLI entry point for validator agent."""
    import sys
    import argparse

    if len(sys.argv) < 2:
        print("""
Validator Agent - Content Validation Gate

Usage:
    python -m src.agents.validator_agent [options]

Examples:
    python -m src.agents.validator_agent --video output/video.mp4 --thumbnail thumb.png
    python -m src.agents.validator_agent --script script.txt --niche finance
    python -m src.agents.validator_agent --video video.mp4 --title "My Title" --strict

Options:
    --video <path>          Video file to validate
    --thumbnail <path>      Thumbnail file to validate
    --script <path>         Script file to validate
    --title <title>         Video title
    --description <desc>    Video description
    --niche <niche>         Content niche (finance, psychology, storytelling)
    --short                 Validate as YouTube Short
    --strict                Treat warnings as blockers
    --min-score <n>         Minimum score to pass (default: 75)
    --json                  Output as JSON
        """)
        return

    parser = argparse.ArgumentParser(description="Validate YouTube content")
    parser.add_argument("--video", help="Video file path")
    parser.add_argument("--thumbnail", help="Thumbnail file path")
    parser.add_argument("--script", help="Script file path")
    parser.add_argument("--title", help="Video title")
    parser.add_argument("--description", help="Video description")
    parser.add_argument("--niche", default="default", help="Content niche")
    parser.add_argument("--short", action="store_true", help="YouTube Short")
    parser.add_argument("--strict", action="store_true", help="Strict mode")
    parser.add_argument("--min-score", type=int, default=75, help="Minimum score")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    # Build script data
    script_data = {}
    script_text = None

    if args.script and Path(args.script).exists():
        with open(args.script, "r", encoding="utf-8") as f:
            script_text = f.read()

    if args.title:
        script_data["title"] = args.title
    if args.description:
        script_data["description"] = args.description

    # Run agent
    agent = ValidatorAgent()
    result = agent.run(
        script_data=script_data if script_data else None,
        script_text=script_text,
        video_file=args.video,
        thumbnail_file=args.thumbnail,
        niche=args.niche,
        is_short=args.short,
        min_score=args.min_score,
        strict_mode=args.strict
    )

    # Output
    if args.json:
        print(json.dumps(result.data, indent=2))
    else:
        print("\n" + "=" * 60)
        print("VALIDATOR AGENT RESULT")
        print("=" * 60)

        if result.success:
            data = result.data
            validation = ValidationResult(**{k: v for k, v in data.items() if k != 'checks'})
            validation.checks = [ValidationCheck(**c) for c in data.get('checks', [])]

            print(validation.summary())
        else:
            print(f"Error: {result.error}")


if __name__ == "__main__":
    main()
