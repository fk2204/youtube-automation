"""
Accessibility Agent - Accessibility Features Validation

Validates and generates accessibility features for YouTube videos including
captions, subtitle sync accuracy, and text readability in thumbnails.

Usage:
    from src.agents.accessibility_agent import AccessibilityAgent

    agent = AccessibilityAgent()

    # Check accessibility
    result = agent.run(
        video_file="path/to/video.mp4",
        script="Video script text...",
        subtitle_file="path/to/subtitles.srt"  # Optional
    )

    if result.success:
        accessibility = result.data
        print(f"Score: {accessibility['score']}")
        print(f"Captions Accuracy: {accessibility['captions_accuracy']}")
        print(f"Improvements: {accessibility['improvements']}")

Example:
    >>> agent = AccessibilityAgent()
    >>> result = agent.run(
    ...     video_file="video.mp4",
    ...     script="Welcome to this video about Python programming."
    ... )
    >>> print(result.data['score'])
    85
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger

from .base_agent import BaseAgent, AgentResult


@dataclass
class AccessibilityResult:
    """
    Result of accessibility check.

    Attributes:
        score: Overall accessibility score (0-100)
        captions_accuracy: Estimated caption sync accuracy (0-100)
        has_captions: Whether captions are available/generated
        caption_coverage: Percentage of video covered by captions
        improvements: List of suggested accessibility improvements
        issues: List of accessibility issues found
        warnings: Non-critical accessibility warnings
        language_support: Languages supported/detected
        readability_score: Text readability score (for thumbnails/titles)
    """
    score: int
    captions_accuracy: Optional[float] = None
    has_captions: bool = False
    caption_coverage: Optional[float] = None
    improvements: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    language_support: List[str] = field(default_factory=lambda: ["en"])
    readability_score: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "captions_accuracy": self.captions_accuracy,
            "has_captions": self.has_captions,
            "caption_coverage": self.caption_coverage,
            "improvements": self.improvements,
            "issues": self.issues,
            "warnings": self.warnings,
            "language_support": self.language_support,
            "readability_score": self.readability_score
        }


class AccessibilityAgent(BaseAgent):
    """
    Agent for validating and generating accessibility features.

    Uses the existing subtitles.py module for caption generation and
    provides comprehensive accessibility validation.

    Features:
    - Caption generation and validation
    - Subtitle sync accuracy verification
    - Text readability analysis
    - Multi-language support preparation
    - Accessibility score calculation
    """

    # Readability thresholds
    MIN_TITLE_LENGTH = 10
    MAX_TITLE_LENGTH = 100
    OPTIMAL_WORDS_PER_CAPTION = (3, 8)  # Words per caption cue
    MAX_CAPTION_DURATION = 5.0  # Maximum seconds per cue
    MIN_CAPTION_DURATION = 0.8  # Minimum seconds per cue

    # Accessibility scoring weights
    WEIGHT_CAPTIONS = 40
    WEIGHT_SYNC_ACCURACY = 25
    WEIGHT_READABILITY = 20
    WEIGHT_COVERAGE = 15

    def __init__(self, provider: str = "rule_based", api_key: str = None):
        """
        Initialize the accessibility agent.

        Args:
            provider: Analysis provider (default: rule_based)
            api_key: Not used for this agent
        """
        super().__init__(provider=provider, api_key=api_key)
        self._subtitle_generator = None
        logger.info("AccessibilityAgent initialized")

    def _get_subtitle_generator(self):
        """Lazy-load the subtitle generator."""
        if self._subtitle_generator is None:
            try:
                from ..content.subtitles import SubtitleGenerator
                self._subtitle_generator = SubtitleGenerator()
            except ImportError as e:
                logger.warning(f"Could not import SubtitleGenerator: {e}")
        return self._subtitle_generator

    def run(
        self,
        video_file: str = "",
        script: str = "",
        subtitle_file: str = "",
        audio_file: str = "",
        title: str = "",
        thumbnail_text: str = "",
        target_languages: List[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        Check accessibility features for a video.

        Args:
            video_file: Path to video file
            script: Video script/narration text
            subtitle_file: Path to existing subtitle file (.srt)
            audio_file: Path to audio file (for sync validation)
            title: Video title (for readability check)
            thumbnail_text: Text in thumbnail (for readability check)
            target_languages: Target languages for captions
            **kwargs: Additional parameters

        Returns:
            AgentResult with AccessibilityResult data

        Example:
            >>> agent = AccessibilityAgent()
            >>> result = agent.run(
            ...     script="Welcome to this tutorial on Python.",
            ...     title="Learn Python in 10 Minutes"
            ... )
            >>> print(result.data['score'])
            85
        """
        logger.info("[AccessibilityAgent] Running accessibility check")

        target_languages = target_languages or ["en"]
        accessibility_result = AccessibilityResult(score=0, language_support=target_languages)

        scores = {}

        # Check captions
        caption_score, caption_accuracy = self._check_captions(
            script, subtitle_file, audio_file, accessibility_result
        )
        scores["captions"] = caption_score

        # Check caption sync accuracy
        sync_score = self._check_sync_accuracy(
            subtitle_file, audio_file, accessibility_result
        )
        scores["sync"] = sync_score

        # Check text readability
        readability_score = self._check_readability(
            title, thumbnail_text, accessibility_result
        )
        scores["readability"] = readability_score

        # Check coverage
        coverage_score = self._check_coverage(
            subtitle_file, video_file, accessibility_result
        )
        scores["coverage"] = coverage_score

        # Calculate overall score
        accessibility_result.score = int(
            (scores["captions"] * self.WEIGHT_CAPTIONS +
             scores["sync"] * self.WEIGHT_SYNC_ACCURACY +
             scores["readability"] * self.WEIGHT_READABILITY +
             scores["coverage"] * self.WEIGHT_COVERAGE) / 100
        )

        # Set caption accuracy
        if caption_accuracy is not None:
            accessibility_result.captions_accuracy = caption_accuracy

        # Generate improvements
        self._generate_improvements(accessibility_result, scores)

        # Log results
        logger.info(
            f"[AccessibilityAgent] Accessibility score: {accessibility_result.score}/100"
        )

        return AgentResult(
            success=True,
            data=accessibility_result.to_dict(),
            tokens_used=0,
            cost=0.0,
            metadata={
                "video_file": video_file,
                "subtitle_file": subtitle_file,
                "target_languages": target_languages,
                "component_scores": scores
            }
        )

    def _check_captions(
        self,
        script: str,
        subtitle_file: str,
        audio_file: str,
        result: AccessibilityResult
    ) -> Tuple[int, Optional[float]]:
        """
        Check caption availability and quality.

        Returns:
            Tuple of (score, accuracy)
        """
        score = 0
        accuracy = None

        # Check if subtitles exist
        if subtitle_file and os.path.exists(subtitle_file):
            result.has_captions = True
            score = 80  # Base score for having captions

            # Validate subtitle format and content
            try:
                cues = self._parse_srt(subtitle_file)
                if cues:
                    # Calculate accuracy based on cue quality
                    accuracy = self._estimate_caption_accuracy(cues, script)
                    score = min(100, score + int(accuracy * 0.2))

                    if len(cues) < 5:
                        result.warnings.append(
                            f"Only {len(cues)} caption cues found. "
                            "Consider adding more detailed captions."
                        )
                else:
                    result.issues.append("Subtitle file appears empty")
                    score = 30
            except Exception as e:
                result.warnings.append(f"Could not parse subtitle file: {str(e)[:50]}")
                score = 50

        elif script:
            # No subtitles but script available - can generate
            result.has_captions = False
            result.issues.append("No captions found. Captions can be generated from script.")
            score = 40  # Partial credit for having script

            # Try to generate captions
            generator = self._get_subtitle_generator()
            if generator:
                result.improvements.append(
                    "Use SubtitleGenerator to create captions from script"
                )
        else:
            # No subtitles and no script
            result.has_captions = False
            result.issues.append("No captions and no script available for caption generation")
            score = 0

        return score, accuracy

    def _check_sync_accuracy(
        self,
        subtitle_file: str,
        audio_file: str,
        result: AccessibilityResult
    ) -> int:
        """
        Check caption sync accuracy.

        Returns:
            Score (0-100)
        """
        if not subtitle_file or not os.path.exists(subtitle_file):
            # Can't check sync without subtitles
            return 50  # Neutral score

        try:
            cues = self._parse_srt(subtitle_file)
            if not cues:
                return 30

            score = 100
            issues = []

            for i, cue in enumerate(cues):
                # Check cue duration
                duration = cue['end'] - cue['start']

                if duration < self.MIN_CAPTION_DURATION:
                    issues.append(f"Cue {i+1} too short ({duration:.1f}s)")
                    score -= 5
                elif duration > self.MAX_CAPTION_DURATION:
                    issues.append(f"Cue {i+1} too long ({duration:.1f}s)")
                    score -= 3

                # Check word count per cue
                word_count = len(cue['text'].split())
                if word_count < self.OPTIMAL_WORDS_PER_CAPTION[0]:
                    score -= 1  # Minor penalty for very short cues
                elif word_count > self.OPTIMAL_WORDS_PER_CAPTION[1]:
                    score -= 2  # Penalty for too many words

                # Check for gaps between cues
                if i > 0:
                    gap = cue['start'] - cues[i-1]['end']
                    if gap > 3.0:  # More than 3 second gap
                        score -= 2

            if issues and len(issues) <= 5:
                for issue in issues[:3]:
                    result.warnings.append(issue)
            elif issues:
                result.warnings.append(f"{len(issues)} caption timing issues detected")

            return max(0, min(100, score))

        except Exception as e:
            logger.debug(f"Sync accuracy check failed: {e}")
            return 50

    def _check_readability(
        self,
        title: str,
        thumbnail_text: str,
        result: AccessibilityResult
    ) -> int:
        """
        Check text readability for title and thumbnail.

        Returns:
            Score (0-100)
        """
        score = 100

        # Check title readability
        if title:
            title_length = len(title)

            if title_length < self.MIN_TITLE_LENGTH:
                result.warnings.append(
                    f"Title very short ({title_length} chars). "
                    "May not be descriptive enough for accessibility."
                )
                score -= 10
            elif title_length > self.MAX_TITLE_LENGTH:
                result.issues.append(
                    f"Title too long ({title_length} chars). "
                    "Screen readers may truncate it."
                )
                score -= 15

            # Check for all caps (harder to read)
            if title.isupper():
                result.warnings.append(
                    "All-caps title is harder to read for many users"
                )
                score -= 10

            # Check for excessive punctuation
            punct_count = sum(1 for c in title if c in "!?#*@$%")
            if punct_count > 3:
                result.warnings.append(
                    "Excessive punctuation in title may confuse screen readers"
                )
                score -= 5

        # Check thumbnail text
        if thumbnail_text:
            # Thumbnail text should be short and high-contrast
            if len(thumbnail_text) > 50:
                result.warnings.append(
                    "Thumbnail text may be too long to read at small sizes"
                )
                score -= 10

            if thumbnail_text.isupper():
                # ALL CAPS in thumbnails is often OK for visibility
                pass

        result.readability_score = max(0, min(100, score))
        return result.readability_score

    def _check_coverage(
        self,
        subtitle_file: str,
        video_file: str,
        result: AccessibilityResult
    ) -> int:
        """
        Check how much of the video is covered by captions.

        Returns:
            Score (0-100)
        """
        if not subtitle_file or not os.path.exists(subtitle_file):
            return 0

        try:
            cues = self._parse_srt(subtitle_file)
            if not cues:
                return 0

            # Calculate total caption duration
            total_caption_time = sum(cue['end'] - cue['start'] for cue in cues)
            last_caption_time = max(cue['end'] for cue in cues)

            # Estimate coverage (captions typically don't cover silence)
            # Assume 70% coverage is excellent for typical video
            coverage = min(100, (total_caption_time / last_caption_time) * 100 * 1.4)
            result.caption_coverage = coverage

            if coverage < 50:
                result.issues.append(
                    f"Low caption coverage ({coverage:.0f}%). "
                    "Consider adding captions for more of the video."
                )
                return int(coverage)
            elif coverage < 70:
                result.warnings.append(
                    f"Caption coverage ({coverage:.0f}%) could be improved"
                )

            return min(100, int(coverage))

        except Exception as e:
            logger.debug(f"Coverage check failed: {e}")
            return 50

    def _parse_srt(self, srt_file: str) -> List[Dict[str, Any]]:
        """Parse an SRT subtitle file."""
        cues = []

        try:
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into blocks
            blocks = re.split(r'\n\n+', content.strip())

            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Parse timestamp line
                    timestamp_match = re.match(
                        r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                        lines[1]
                    )

                    if timestamp_match:
                        g = timestamp_match.groups()
                        start = (int(g[0]) * 3600 + int(g[1]) * 60 +
                                int(g[2]) + int(g[3]) / 1000)
                        end = (int(g[4]) * 3600 + int(g[5]) * 60 +
                              int(g[6]) + int(g[7]) / 1000)

                        text = '\n'.join(lines[2:])
                        cues.append({
                            'start': start,
                            'end': end,
                            'text': text
                        })

        except Exception as e:
            logger.debug(f"SRT parsing error: {e}")

        return cues

    def _estimate_caption_accuracy(
        self,
        cues: List[Dict[str, Any]],
        script: str
    ) -> float:
        """
        Estimate caption accuracy by comparing with script.

        Returns:
            Accuracy percentage (0-100)
        """
        if not cues or not script:
            return 50.0  # Neutral when can't compare

        # Combine all caption text
        caption_text = ' '.join(cue['text'] for cue in cues)

        # Simple word overlap calculation
        script_words = set(script.lower().split())
        caption_words = set(caption_text.lower().split())

        if not script_words:
            return 50.0

        # Calculate Jaccard similarity
        intersection = len(script_words & caption_words)
        union = len(script_words | caption_words)

        if union == 0:
            return 0.0

        similarity = (intersection / union) * 100

        # Adjust for typical caption differences (contractions, etc.)
        # Captions are rarely 100% match even when accurate
        adjusted = min(100, similarity * 1.3)

        return adjusted

    def _generate_improvements(
        self,
        result: AccessibilityResult,
        scores: Dict[str, int]
    ):
        """Generate accessibility improvement suggestions."""
        if not result.has_captions:
            result.improvements.append(
                "Generate captions using SubtitleGenerator for better accessibility"
            )

        if scores.get("captions", 0) < 60:
            result.improvements.append(
                "Improve caption quality or generate new captions from script"
            )

        if scores.get("sync", 0) < 70:
            result.improvements.append(
                "Review caption timing to ensure sync with audio"
            )

        if scores.get("readability", 0) < 80:
            result.improvements.append(
                "Simplify title text for better screen reader compatibility"
            )

        if result.caption_coverage and result.caption_coverage < 70:
            result.improvements.append(
                "Add captions to cover more of the video content"
            )

        # Language support suggestions
        if len(result.language_support) == 1:
            result.improvements.append(
                "Consider adding captions in additional languages for wider reach"
            )

        if not result.improvements:
            result.improvements.append(
                "Accessibility features are in good shape"
            )

    def generate_captions(
        self,
        script: str,
        audio_duration: float,
        output_file: str,
        max_chars: int = 50
    ) -> Optional[str]:
        """
        Generate captions from script text.

        Args:
            script: Video script/narration text
            audio_duration: Total audio duration in seconds
            output_file: Output path for .srt file
            max_chars: Maximum characters per caption line

        Returns:
            Path to generated .srt file or None on failure
        """
        generator = self._get_subtitle_generator()
        if not generator:
            logger.error("SubtitleGenerator not available")
            return None

        try:
            track = generator.generate_subtitles_from_script(
                script=script,
                audio_duration=audio_duration,
                max_chars=max_chars
            )

            if track and track.cues:
                output_path = generator.create_srt_file(track, output_file)
                logger.success(f"Generated {len(track.cues)} caption cues: {output_path}")
                return output_path
            else:
                logger.warning("No caption cues generated")
                return None

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return None

    def validate_captions(
        self,
        subtitle_file: str,
        script: str = ""
    ) -> Dict[str, Any]:
        """
        Validate caption file quality.

        Args:
            subtitle_file: Path to .srt file
            script: Optional script for accuracy comparison

        Returns:
            Dictionary with validation results
        """
        if not os.path.exists(subtitle_file):
            return {
                "valid": False,
                "error": "Subtitle file not found"
            }

        try:
            cues = self._parse_srt(subtitle_file)

            if not cues:
                return {
                    "valid": False,
                    "error": "No valid cues found in subtitle file"
                }

            # Calculate metrics
            total_duration = sum(cue['end'] - cue['start'] for cue in cues)
            avg_duration = total_duration / len(cues)
            avg_words = sum(len(cue['text'].split()) for cue in cues) / len(cues)

            # Check for issues
            issues = []
            for i, cue in enumerate(cues):
                duration = cue['end'] - cue['start']
                if duration < 0.3:
                    issues.append(f"Cue {i+1}: Too short ({duration:.2f}s)")
                elif duration > 7:
                    issues.append(f"Cue {i+1}: Too long ({duration:.2f}s)")

                if cue['start'] >= cue['end']:
                    issues.append(f"Cue {i+1}: Invalid timing")

            # Calculate accuracy if script provided
            accuracy = None
            if script:
                accuracy = self._estimate_caption_accuracy(cues, script)

            return {
                "valid": len(issues) == 0,
                "cue_count": len(cues),
                "total_duration": total_duration,
                "avg_cue_duration": avg_duration,
                "avg_words_per_cue": avg_words,
                "accuracy": accuracy,
                "issues": issues[:10]  # Limit to first 10 issues
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }


# CLI entry point
def main():
    """CLI entry point for accessibility agent."""
    import sys
    import json

    if len(sys.argv) < 2:
        print("""
Accessibility Agent - Accessibility Features Validation

Usage:
    python -m src.agents.accessibility_agent <video_file> [options]
    python -m src.agents.accessibility_agent --validate <srt_file>
    python -m src.agents.accessibility_agent --generate <script_file> --duration <seconds>

Options:
    --script <text>     Video script text
    --srt <file>        Existing subtitle file
    --title <title>     Video title for readability check
    --json              Output as JSON
    --validate          Validate an existing SRT file
    --generate          Generate captions from script
    --duration <sec>    Audio duration (for generation)
    --output <file>     Output file for generated captions

Examples:
    python -m src.agents.accessibility_agent video.mp4 --srt subtitles.srt
    python -m src.agents.accessibility_agent --validate subtitles.srt
    python -m src.agents.accessibility_agent --generate script.txt --duration 120 --output captions.srt
        """)
        return

    # Parse arguments
    video_file = ""
    script = ""
    subtitle_file = ""
    title = ""
    output_json = False
    validate_mode = False
    generate_mode = False
    duration = 0
    output_file = ""

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--script" and i + 1 < len(sys.argv):
            script_arg = sys.argv[i + 1]
            if os.path.exists(script_arg):
                with open(script_arg, 'r', encoding='utf-8') as f:
                    script = f.read()
            else:
                script = script_arg
            i += 2
        elif arg == "--srt" and i + 1 < len(sys.argv):
            subtitle_file = sys.argv[i + 1]
            i += 2
        elif arg == "--title" and i + 1 < len(sys.argv):
            title = sys.argv[i + 1]
            i += 2
        elif arg == "--json":
            output_json = True
            i += 1
        elif arg == "--validate":
            validate_mode = True
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                subtitle_file = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        elif arg == "--generate":
            generate_mode = True
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                script_arg = sys.argv[i + 1]
                if os.path.exists(script_arg):
                    with open(script_arg, 'r', encoding='utf-8') as f:
                        script = f.read()
                else:
                    script = script_arg
                i += 2
            else:
                i += 1
        elif arg == "--duration" and i + 1 < len(sys.argv):
            duration = float(sys.argv[i + 1])
            i += 2
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif not video_file and not arg.startswith("--"):
            video_file = arg
            i += 1
        else:
            i += 1

    agent = AccessibilityAgent()

    if validate_mode and subtitle_file:
        # Validate existing SRT file
        result = agent.validate_captions(subtitle_file, script)
        if output_json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 60)
            print("CAPTION VALIDATION RESULT")
            print("=" * 60)
            print(f"Valid: {result.get('valid', False)}")
            if result.get('error'):
                print(f"Error: {result['error']}")
            else:
                print(f"Cue Count: {result.get('cue_count', 0)}")
                print(f"Total Duration: {result.get('total_duration', 0):.1f}s")
                print(f"Avg Cue Duration: {result.get('avg_cue_duration', 0):.2f}s")
                print(f"Avg Words/Cue: {result.get('avg_words_per_cue', 0):.1f}")
                if result.get('accuracy'):
                    print(f"Accuracy: {result['accuracy']:.1f}%")
                if result.get('issues'):
                    print(f"\nIssues ({len(result['issues'])}):")
                    for issue in result['issues']:
                        print(f"  - {issue}")

    elif generate_mode and script and duration:
        # Generate captions
        if not output_file:
            output_file = "output/captions.srt"

        result = agent.generate_captions(
            script=script,
            audio_duration=duration,
            output_file=output_file
        )

        if result:
            print(f"Captions generated: {result}")
        else:
            print("Caption generation failed")

    else:
        # Run full accessibility check
        result = agent.run(
            video_file=video_file,
            script=script,
            subtitle_file=subtitle_file,
            title=title
        )

        if output_json:
            print(json.dumps(result.data, indent=2))
        else:
            print("\n" + "=" * 60)
            print("ACCESSIBILITY AGENT RESULT")
            print("=" * 60)

            data = result.data
            print(f"Accessibility Score: {data['score']}/100")
            print(f"Has Captions: {'Yes' if data['has_captions'] else 'No'}")

            if data['captions_accuracy']:
                print(f"Caption Accuracy: {data['captions_accuracy']:.1f}%")
            if data['caption_coverage']:
                print(f"Caption Coverage: {data['caption_coverage']:.1f}%")
            if data['readability_score']:
                print(f"Readability Score: {data['readability_score']}/100")

            print(f"Languages: {', '.join(data['language_support'])}")

            if data["issues"]:
                print(f"\nIssues ({len(data['issues'])}):")
                for issue in data["issues"]:
                    print(f"  [X] {issue}")

            if data["warnings"]:
                print(f"\nWarnings ({len(data['warnings'])}):")
                for warning in data["warnings"]:
                    print(f"  [!] {warning}")

            if data["improvements"]:
                print(f"\nSuggested Improvements:")
                for improvement in data["improvements"]:
                    print(f"  -> {improvement}")


if __name__ == "__main__":
    main()
