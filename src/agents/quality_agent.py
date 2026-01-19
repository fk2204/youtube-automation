"""
Quality Agent - Script and Content Validation

A token-efficient agent specialized in validating scripts, checking
engagement patterns, and ensuring content meets best practices.

Usage:
    from src.agents.quality_agent import QualityAgent

    agent = QualityAgent()

    # Quick validation (rule-based, no AI)
    result = agent.quick_check(script_text, niche="finance")

    # Full AI analysis
    result = agent.full_analysis(script_text, niche="finance")

    # Get improvement suggestions
    result = agent.suggest_improvements(script_text, niche="finance")
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger

from ..utils.token_manager import (
    get_token_manager,
    get_cost_optimizer,
    get_prompt_cache
)
from ..utils.best_practices import (
    validate_title,
    validate_hook,
    get_best_practices,
    suggest_improvements as get_suggestions,
    pre_publish_checklist,
    ValidationResult,
    PrePublishChecklist
)


@dataclass
class QualityCheckItem:
    """Single quality check result."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: str
    priority: str = "medium"  # high, medium, low


@dataclass
class QualityResult:
    """Result from quality agent operations."""
    success: bool
    operation: str
    overall_score: int  # 0-100
    is_valid: bool
    checks: List[QualityCheckItem] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    provider: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["checks"] = [asdict(c) for c in self.checks]
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASS" if self.is_valid else "NEEDS IMPROVEMENT"
        lines = [
            f"Quality Check: {status}",
            f"Score: {self.overall_score}/100",
            f"Checks: {sum(1 for c in self.checks if c.passed)}/{len(self.checks)} passed",
            ""
        ]

        # Group by priority
        high_priority = [c for c in self.checks if c.priority == "high"]
        if high_priority:
            lines.append("High Priority:")
            for check in high_priority:
                icon = "[OK]" if check.passed else "[!!]"
                lines.append(f"  {icon} {check.name}: {check.details}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def save(self, path: str = None):
        """Save result to JSON file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/quality_reports/report_{timestamp}.json"

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Quality report saved to {path}")


class QualityAgent:
    """
    Quality Agent for script and content validation.

    Token-efficient design:
    - Rule-based validation uses 0 tokens
    - AI analysis only when requested
    - Caches analysis results
    """

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the quality agent.

        Args:
            provider: AI provider (for full analysis mode)
            api_key: API key for cloud providers
        """
        self.tracker = get_token_manager()
        self.optimizer = get_cost_optimizer()
        self.cache = get_prompt_cache()

        # Select provider for AI analysis
        if provider is None:
            provider = self.optimizer.select_provider("script_revision")

        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        logger.info(f"QualityAgent initialized with provider: {provider}")

    def quick_check(
        self,
        script_text: str,
        niche: str = "default",
        is_short: bool = False
    ) -> QualityResult:
        """
        Quick quality check using rule-based validation.

        Token cost: ZERO (no AI calls)

        Args:
            script_text: Script content to validate
            niche: Content niche for context
            is_short: Whether this is a YouTube Short

        Returns:
            QualityResult with validation results
        """
        operation = f"quick_check_{niche}"
        logger.info(f"[QualityAgent] Quick check for {niche} content")

        checks = []
        recommendations = []

        # Extract components from script
        lines = script_text.strip().split("\n")
        title = self._extract_title(lines)
        hook = self._extract_hook(lines)
        word_count = len(script_text.split())

        # Check 1: Title quality
        if title:
            title_result = validate_title(title, niche)
            checks.append(QualityCheckItem(
                name="Title Quality",
                passed=title_result.is_valid,
                score=title_result.score,
                details=f"Score: {title_result.score:.0%}",
                priority="high"
            ))
            recommendations.extend(title_result.suggestions)
        else:
            checks.append(QualityCheckItem(
                name="Title Quality",
                passed=False,
                score=0.0,
                details="No title found",
                priority="high"
            ))
            recommendations.append("Add a compelling title at the start")

        # Check 2: Hook quality
        if hook:
            hook_result = validate_hook(hook, niche)
            checks.append(QualityCheckItem(
                name="Hook Quality",
                passed=hook_result.is_valid,
                score=hook_result.score,
                details=f"Score: {hook_result.score:.0%}",
                priority="high"
            ))
            recommendations.extend(hook_result.suggestions)
        else:
            checks.append(QualityCheckItem(
                name="Hook Quality",
                passed=False,
                score=0.0,
                details="No hook found in first 50 words",
                priority="high"
            ))

        # Check 3: Script length
        if is_short:
            # Shorts: 50-150 words optimal
            length_ok = 50 <= word_count <= 200
            length_details = f"{word_count} words (optimal: 50-150)"
        else:
            # Regular: 800-2000 words optimal (5-12 min video)
            length_ok = 500 <= word_count <= 3000
            length_details = f"{word_count} words (optimal: 800-2000)"

        checks.append(QualityCheckItem(
            name="Script Length",
            passed=length_ok,
            score=1.0 if length_ok else 0.5,
            details=length_details,
            priority="medium"
        ))

        # Check 4: Structure indicators
        has_sections = any(line.strip().startswith(("#", "##", "Section", "SECTION")) for line in lines)
        has_timestamps = any(":" in line and any(c.isdigit() for c in line.split(":")[0][-2:]) for line in lines)

        checks.append(QualityCheckItem(
            name="Structure",
            passed=has_sections or has_timestamps,
            score=1.0 if (has_sections or has_timestamps) else 0.3,
            details="Sections/timestamps detected" if has_sections else "No clear structure",
            priority="medium"
        ))

        # Check 5: Engagement elements
        engagement_words = ["you", "your", "?", "imagine", "secret", "truth"]
        engagement_count = sum(1 for word in engagement_words if word.lower() in script_text.lower())
        engagement_ok = engagement_count >= 3

        checks.append(QualityCheckItem(
            name="Engagement Elements",
            passed=engagement_ok,
            score=min(engagement_count / 5, 1.0),
            details=f"{engagement_count} engagement triggers found",
            priority="medium"
        ))

        if not engagement_ok:
            recommendations.append("Add more engagement elements: questions, 'you' statements, curiosity triggers")

        # Check 6: CTA presence
        cta_words = ["subscribe", "like", "comment", "share", "click", "link", "below"]
        has_cta = any(word in script_text.lower() for word in cta_words)

        checks.append(QualityCheckItem(
            name="Call to Action",
            passed=has_cta,
            score=1.0 if has_cta else 0.0,
            details="CTA found" if has_cta else "No CTA detected",
            priority="low"
        ))

        if not has_cta:
            recommendations.append("Add a call to action (subscribe, like, comment)")

        # Calculate overall score
        high_weight = 3
        medium_weight = 2
        low_weight = 1

        total_weight = sum(
            high_weight if c.priority == "high" else
            medium_weight if c.priority == "medium" else
            low_weight
            for c in checks
        )

        weighted_score = sum(
            c.score * (
                high_weight if c.priority == "high" else
                medium_weight if c.priority == "medium" else
                low_weight
            )
            for c in checks
        )

        overall_score = int((weighted_score / total_weight) * 100) if total_weight > 0 else 0
        is_valid = overall_score >= 60

        result = QualityResult(
            success=True,
            operation=operation,
            overall_score=overall_score,
            is_valid=is_valid,
            checks=checks,
            recommendations=recommendations[:10],
            tokens_used=0,
            cost=0.0,
            provider="rule_based"
        )

        logger.info(f"[QualityAgent] Quick check complete: {overall_score}/100")
        return result

    def full_analysis(
        self,
        script_text: str,
        niche: str = "default",
        is_short: bool = False
    ) -> QualityResult:
        """
        Full AI-powered quality analysis.

        Token cost: MEDIUM (~1,000-2,000 tokens)

        Args:
            script_text: Script content to analyze
            niche: Content niche for context
            is_short: Whether this is a YouTube Short

        Returns:
            QualityResult with detailed analysis
        """
        operation = f"full_analysis_{niche}"
        logger.info(f"[QualityAgent] Full analysis for {niche} content")

        # Start with quick check
        quick_result = self.quick_check(script_text, niche, is_short)

        # Check cache
        cache_key = f"quality_full_{hash(script_text[:500])}_{niche}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                cached_data = json.loads(cached)
                logger.info("[QualityAgent] Using cached AI analysis")

                # Merge cached AI analysis with quick check
                quick_result.checks.extend([
                    QualityCheckItem(**c) for c in cached_data.get("ai_checks", [])
                ])
                quick_result.recommendations.extend(cached_data.get("ai_recommendations", []))
                quick_result.provider = "cache"
                return quick_result
            except:
                pass

        # Get AI provider
        try:
            from ..content.script_writer import get_provider
            ai = get_provider(self.provider, self.api_key)
        except Exception as e:
            logger.warning(f"Could not load AI provider: {e}")
            return quick_result

        # Build analysis prompt
        script_preview = script_text[:2000]
        prompt = f"""Analyze this YouTube {'Short' if is_short else 'video'} script for quality and engagement:

SCRIPT:
{script_preview}

Analyze and respond with ONLY a JSON object:
{{
    "hook_effectiveness": {{
        "score": 0-10,
        "feedback": "brief feedback on first 5-10 seconds"
    }},
    "retention_structure": {{
        "score": 0-10,
        "open_loops_count": 0,
        "feedback": "brief feedback on retention techniques"
    }},
    "engagement_potential": {{
        "score": 0-10,
        "feedback": "brief feedback on engagement"
    }},
    "top_3_improvements": [
        "improvement 1",
        "improvement 2",
        "improvement 3"
    ]
}}"""

        try:
            response = ai.generate(prompt, max_tokens=800)

            # Parse response
            analysis = self._parse_json_response(response)

            # Record token usage
            tokens_used = 1500  # Estimate
            cost = self.tracker.record_usage(
                provider=self.provider,
                input_tokens=1000,
                output_tokens=500,
                operation="quality_full_analysis"
            )

            # Add AI checks
            ai_checks = []
            ai_recommendations = []

            if "hook_effectiveness" in analysis:
                score = analysis["hook_effectiveness"].get("score", 5) / 10
                ai_checks.append(QualityCheckItem(
                    name="AI: Hook Effectiveness",
                    passed=score >= 0.6,
                    score=score,
                    details=analysis["hook_effectiveness"].get("feedback", "")[:100],
                    priority="high"
                ))

            if "retention_structure" in analysis:
                score = analysis["retention_structure"].get("score", 5) / 10
                ai_checks.append(QualityCheckItem(
                    name="AI: Retention Structure",
                    passed=score >= 0.6,
                    score=score,
                    details=f"Open loops: {analysis['retention_structure'].get('open_loops_count', 0)}",
                    priority="high"
                ))

            if "engagement_potential" in analysis:
                score = analysis["engagement_potential"].get("score", 5) / 10
                ai_checks.append(QualityCheckItem(
                    name="AI: Engagement Potential",
                    passed=score >= 0.6,
                    score=score,
                    details=analysis["engagement_potential"].get("feedback", "")[:100],
                    priority="medium"
                ))

            ai_recommendations = analysis.get("top_3_improvements", [])

            # Cache the AI analysis
            cache_data = {
                "ai_checks": [asdict(c) for c in ai_checks],
                "ai_recommendations": ai_recommendations
            }
            self.cache.set(cache_key, json.dumps(cache_data), self.provider)

            # Merge with quick check results
            quick_result.checks.extend(ai_checks)
            quick_result.recommendations.extend(ai_recommendations)
            quick_result.tokens_used = tokens_used
            quick_result.cost = cost
            quick_result.provider = self.provider

            # Recalculate overall score
            all_scores = [c.score for c in quick_result.checks]
            quick_result.overall_score = int(sum(all_scores) / len(all_scores) * 100) if all_scores else 0
            quick_result.is_valid = quick_result.overall_score >= 60

            logger.success(f"[QualityAgent] Full analysis complete: {quick_result.overall_score}/100")
            return quick_result

        except Exception as e:
            logger.error(f"[QualityAgent] AI analysis failed: {e}")
            quick_result.error = str(e)
            return quick_result

    def check_video_file(
        self,
        video_file: str,
        script_data: Optional[Dict] = None,
        is_short: bool = False
    ) -> QualityResult:
        """
        Check video file quality.

        Uses VideoQualityChecker for technical validation.

        Args:
            video_file: Path to video file
            script_data: Optional script metadata
            is_short: Whether this is a YouTube Short

        Returns:
            QualityResult
        """
        from ..content.quality_checker import VideoQualityChecker

        operation = f"check_video_file"
        logger.info(f"[QualityAgent] Checking video file: {video_file}")

        try:
            checker = VideoQualityChecker()
            report = checker.check_video(
                video_file=video_file,
                script_data=script_data,
                is_short=is_short,
                skip_ai_checks=True  # Use separate AI checks
            )

            # Convert to QualityResult
            checks = []
            for key, value in report.file_checks.items():
                if key in ["exists", "corrupted"]:
                    checks.append(QualityCheckItem(
                        name=f"File: {key}",
                        passed=value if key == "exists" else not value,
                        score=1.0 if (value if key == "exists" else not value) else 0.0,
                        details=str(value),
                        priority="high"
                    ))

            for key, value in report.technical_checks.items():
                if key in ["resolution_correct", "has_audio"]:
                    checks.append(QualityCheckItem(
                        name=f"Technical: {key}",
                        passed=bool(value),
                        score=1.0 if value else 0.0,
                        details=str(value),
                        priority="high" if key == "has_audio" else "medium"
                    ))

            result = QualityResult(
                success=True,
                operation=operation,
                overall_score=report.overall_score,
                is_valid=report.passed,
                checks=checks,
                recommendations=report.recommendations[:5],
                tokens_used=0,
                cost=0.0,
                provider="ffprobe"
            )

            return result

        except Exception as e:
            logger.error(f"[QualityAgent] Video check failed: {e}")
            return QualityResult(
                success=False,
                operation=operation,
                overall_score=0,
                is_valid=False,
                error=str(e),
                provider="ffprobe"
            )

    def suggest_improvements(
        self,
        script_text: str,
        niche: str = "default"
    ) -> QualityResult:
        """
        Get improvement suggestions for a script.

        Token cost: LOW (uses pre-compiled patterns)

        Args:
            script_text: Script content
            niche: Content niche

        Returns:
            QualityResult with suggestions
        """
        operation = f"suggest_improvements_{niche}"
        logger.info(f"[QualityAgent] Getting improvement suggestions")

        # Extract content data
        lines = script_text.strip().split("\n")
        title = self._extract_title(lines)
        hook = self._extract_hook(lines)
        word_count = len(script_text.split())

        content = {
            "title": title or "",
            "hook": hook or "",
            "duration": word_count / 150,  # Estimate minutes
            "tags": [],  # Could extract hashtags
        }

        suggestions = get_suggestions(content, niche)

        result = QualityResult(
            success=True,
            operation=operation,
            overall_score=0,  # Not scored
            is_valid=True,
            recommendations=suggestions,
            tokens_used=0,
            cost=0.0,
            provider="rule_based"
        )

        return result

    def run(self, command: str, **kwargs) -> QualityResult:
        """
        Main entry point for CLI usage.

        Args:
            command: File path or command string
            **kwargs: Additional parameters (niche, full, short)

        Returns:
            QualityResult
        """
        # Check if command is a file path
        if os.path.exists(command):
            with open(command, "r", encoding="utf-8") as f:
                script_text = f.read()
        else:
            script_text = command

        niche = kwargs.get("niche", "default")
        is_short = kwargs.get("short", False)
        full = kwargs.get("full", False)

        if full:
            return self.full_analysis(script_text, niche, is_short)
        else:
            return self.quick_check(script_text, niche, is_short)

    def _extract_title(self, lines: List[str]) -> Optional[str]:
        """Extract title from script lines."""
        for line in lines[:5]:
            line = line.strip()
            if line and not line.startswith(("#", "//", "---")):
                # Check if it looks like a title
                if len(line) < 100 and not line.endswith((".", "?", "!")):
                    return line
                if line.upper() == line:  # All caps
                    return line
        return None

    def _extract_hook(self, lines: List[str]) -> Optional[str]:
        """Extract hook (first 50 words) from script."""
        text = " ".join(lines)
        words = text.split()[:50]
        return " ".join(words) if words else None

    def _parse_json_response(self, content: str) -> Dict:
        """Parse JSON from AI response."""
        # Try direct parse
        try:
            return json.loads(content)
        except:
            pass

        # Extract from markdown
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                try:
                    return json.loads(content[start:end].strip())
                except:
                    pass
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                try:
                    return json.loads(content[start:end].strip())
                except:
                    pass

        # Try to find JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end])
            except:
                pass

        return {}


# CLI entry point
def main():
    """CLI entry point for quality agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Quality Agent - Script and Content Validation

Usage:
    python -m src.agents.quality_agent <script_file> [options]
    python -m src.agents.quality_agent "script text here" [options]

Options:
    --niche <niche>     Content niche (finance, psychology, storytelling)
    --full              Run full AI analysis (uses tokens)
    --short             Validate as YouTube Short
    --save              Save report to data/quality_reports/
    --json              Output as JSON

Examples:
    python -m src.agents.quality_agent output/script.txt --niche finance
    python -m src.agents.quality_agent output/script.txt --full --niche psychology
    python -m src.agents.quality_agent output/short.txt --short
        """)
        return

    # Parse arguments
    script_input = sys.argv[1]
    kwargs = {}

    i = 2
    output_json = False
    save_report = False

    while i < len(sys.argv):
        if sys.argv[i] == "--niche" and i + 1 < len(sys.argv):
            kwargs["niche"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--full":
            kwargs["full"] = True
            i += 1
        elif sys.argv[i] == "--short":
            kwargs["short"] = True
            i += 1
        elif sys.argv[i] == "--save":
            save_report = True
            i += 1
        elif sys.argv[i] == "--json":
            output_json = True
            i += 1
        else:
            i += 1

    # Run agent
    agent = QualityAgent()
    result = agent.run(script_input, **kwargs)

    # Output
    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("QUALITY AGENT RESULT")
        print("=" * 60)
        print(result.summary())
        print(f"\nTokens used: {result.tokens_used}")
        print(f"Cost: ${result.cost:.4f}")

    # Save if requested
    if save_report:
        result.save()


if __name__ == "__main__":
    main()
