"""
Retention Optimizer Agent - Standalone Script Retention Optimization

A production agent for optimizing YouTube video scripts for maximum
viewer retention using open loops, micro-payoffs, and pattern interrupts.

Features:
- Wraps existing RetentionOptimizer from script_writer.py
- Analyzes scripts for open loops (target 3+ per video)
- Injects micro-payoffs every 30-60 seconds
- Adds pattern interrupts (visual cues in script)
- Calculates hook strength score
- Predicts drop-off risks at specific timestamps
- Returns comprehensive retention analysis

Usage:
    from src.agents.retention_optimizer_agent import RetentionOptimizerAgent, RetentionResult

    agent = RetentionOptimizerAgent()
    result = agent.run(
        script_text="Your script content here...",
        duration_seconds=600,  # 10 minutes
        niche="finance"
    )

    if result.success:
        print(f"Optimized Script: {result.data['optimized_script']}")
        print(f"Hook Strength: {result.data['hook_strength']}")
        print(f"Open Loop Count: {result.data['open_loop_count']}")
        print(f"Predicted Retention: {result.data['predicted_retention']}")

CLI:
    python -m src.agents.retention_optimizer_agent script.txt --duration 600
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger

from .base_agent import BaseAgent, AgentResult


# Hook strength patterns and weights
HOOK_PATTERNS = {
    "pattern_interrupt": {
        "patterns": [r"^stop", r"^wait", r"^hold on", r"^listen", r"^forget everything"],
        "weight": 0.15,
        "description": "Pattern interrupt opener"
    },
    "curiosity_gap": {
        "patterns": [r"you don't know", r"nobody tells", r"secret", r"hidden", r"truth"],
        "weight": 0.12,
        "description": "Curiosity gap"
    },
    "specific_numbers": {
        "patterns": [r"\$[\d,]+", r"\d+%", r"\d+\s*(?:million|billion|thousand)", r"\d+\s*(?:steps?|ways?|tips?)"],
        "weight": 0.10,
        "description": "Specific numbers/stats"
    },
    "direct_address": {
        "patterns": [r"\byou\b", r"\byour\b"],
        "weight": 0.08,
        "description": "Direct address (you/your)"
    },
    "urgency": {
        "patterns": [r"right now", r"immediately", r"today", r"before it's too late"],
        "weight": 0.08,
        "description": "Urgency trigger"
    },
    "bold_claim": {
        "patterns": [r"never", r"always", r"everyone", r"nobody", r"every single"],
        "weight": 0.07,
        "description": "Bold claim/absolute"
    },
    "question": {
        "patterns": [r"\?"],
        "weight": 0.06,
        "description": "Question hook"
    },
    "tension": {
        "patterns": [r"but\b", r"however", r"except", r"the catch", r"there's a problem"],
        "weight": 0.08,
        "description": "Tension/conflict"
    },
}

# Drop-off risk indicators at different video points
DROP_OFF_INDICATORS = {
    "early_drop": {
        "timestamp_range": (0, 30),
        "risk_factors": ["no hook", "slow start", "no curiosity gap"],
        "impact": 0.25,  # High impact
    },
    "middle_drop": {
        "timestamp_range": (30, 180),
        "risk_factors": ["no payoff", "repetitive", "no open loops"],
        "impact": 0.15,
    },
    "late_drop": {
        "timestamp_range": (180, None),
        "risk_factors": ["unresolved loops", "weak conclusion", "no cta"],
        "impact": 0.10,
    },
}

# Retention improvement phrases
VISUAL_CUE_MARKERS = [
    "[VISUAL: Show chart/graph]",
    "[VISUAL: Text overlay]",
    "[VISUAL: B-roll footage]",
    "[VISUAL: Zoom effect]",
    "[VISUAL: Split screen]",
    "[VISUAL: Highlight key point]",
    "[VISUAL: Animation]",
    "[VISUAL: Cut to face]",
]


@dataclass
class RetentionResult:
    """
    Result from retention optimization.

    Attributes:
        hook_strength: Hook strength score (0.0-1.0)
        open_loop_count: Number of open loops in script
        predicted_retention: Predicted average retention (0.0-1.0)
        improvements: List of improvements made
        drop_off_risks: Predicted drop-off points with risks
        original_word_count: Word count before optimization
        optimized_word_count: Word count after optimization
        micro_payoffs_added: Number of micro-payoffs injected
        pattern_interrupts_added: Number of pattern interrupts injected
        visual_cues_added: Number of visual cue markers added
        hook_analysis: Detailed hook analysis
    """
    hook_strength: float = 0.0
    open_loop_count: int = 0
    predicted_retention: float = 0.0
    improvements: List[str] = field(default_factory=list)
    drop_off_risks: List[Dict[str, Any]] = field(default_factory=list)
    original_word_count: int = 0
    optimized_word_count: int = 0
    micro_payoffs_added: int = 0
    pattern_interrupts_added: int = 0
    visual_cues_added: int = 0
    hook_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RetentionOptimizerAgent(BaseAgent):
    """
    Production agent for optimizing video scripts for viewer retention.

    Wraps the existing RetentionOptimizer class and adds:
    - Hook strength analysis and scoring
    - Drop-off risk prediction
    - Visual cue marker injection
    - Comprehensive retention metrics
    """

    # Target metrics
    TARGET_OPEN_LOOPS = 3  # Minimum open loops per video
    TARGET_PAYOFF_INTERVAL = 45  # Seconds between micro-payoffs
    TARGET_VISUAL_CHANGES = 15  # Visual changes per minute
    MIN_HOOK_STRENGTH = 0.6  # Minimum acceptable hook strength

    def __init__(self, provider: str = "groq", api_key: str = None):
        """
        Initialize the RetentionOptimizerAgent.

        Args:
            provider: AI provider (for potential enhancements)
            api_key: API key for provider
        """
        super().__init__(provider=provider, api_key=api_key)
        self.name = "RetentionOptimizerAgent"

        # Lazy-load the existing RetentionOptimizer
        self._retention_optimizer = None
        self._natural_pacer = None

        logger.info(f"RetentionOptimizerAgent initialized")

    def _get_retention_optimizer(self):
        """Lazy-load RetentionOptimizer from script_writer."""
        if self._retention_optimizer is None:
            try:
                from ..content.script_writer import RetentionOptimizer
                self._retention_optimizer = RetentionOptimizer()
            except ImportError as e:
                logger.warning(f"Could not import RetentionOptimizer: {e}")
        return self._retention_optimizer

    def _get_natural_pacer(self):
        """Lazy-load NaturalPacingInjector from script_writer."""
        if self._natural_pacer is None:
            try:
                from ..content.script_writer import NaturalPacingInjector
                self._natural_pacer = NaturalPacingInjector()
            except ImportError as e:
                logger.warning(f"Could not import NaturalPacingInjector: {e}")
        return self._natural_pacer

    def run(
        self,
        script_text: str,
        duration_seconds: int = 600,
        niche: str = "default",
        inject_open_loops: bool = True,
        inject_payoffs: bool = True,
        inject_interrupts: bool = True,
        inject_visual_cues: bool = True,
        add_natural_pacing: bool = True,
        target_open_loops: int = None,
        payoff_interval: int = None,
        **kwargs
    ) -> AgentResult:
        """
        Analyze and optimize a script for maximum viewer retention.

        Args:
            script_text: The script content to optimize
            duration_seconds: Expected video duration in seconds
            niche: Content niche (for niche-specific optimizations)
            inject_open_loops: Whether to inject open loop phrases
            inject_payoffs: Whether to inject micro-payoff phrases
            inject_interrupts: Whether to inject pattern interrupts
            inject_visual_cues: Whether to add visual cue markers
            add_natural_pacing: Whether to add natural pacing markers
            target_open_loops: Override default open loop target
            payoff_interval: Override default payoff interval (seconds)
            **kwargs: Additional parameters

        Returns:
            AgentResult containing optimized script and RetentionResult metrics
        """
        logger.info(f"[RetentionOptimizerAgent] Analyzing script ({len(script_text)} chars, {duration_seconds}s)")

        if not script_text or not script_text.strip():
            return AgentResult(
                success=False,
                error="Empty script provided",
                data={}
            )

        original_word_count = len(script_text.split())
        improvements = []
        warnings = []

        # Get the existing optimizer
        optimizer = self._get_retention_optimizer()
        if not optimizer:
            return AgentResult(
                success=False,
                error="RetentionOptimizer not available",
                data={"script_text": script_text}
            )

        # Calculate targets based on duration
        open_loop_target = target_open_loops or max(self.TARGET_OPEN_LOOPS, 3 + (duration_seconds - 300) // 120)
        payoff_int = payoff_interval or (45 if duration_seconds < 300 else 60 if duration_seconds < 600 else 75)

        # Phase 1: Analyze hook strength BEFORE optimization
        hook_text = self._extract_hook(script_text, word_count=50)
        hook_analysis = self._analyze_hook_strength(hook_text, niche)
        hook_strength = hook_analysis.get("score", 0.0)

        if hook_strength < self.MIN_HOOK_STRENGTH:
            warnings.append(f"Hook strength ({hook_strength:.0%}) below target ({self.MIN_HOOK_STRENGTH:.0%})")

        # Phase 2: Count existing open loops
        existing_open_loops = self._count_open_loops(script_text)
        logger.info(f"[RetentionOptimizerAgent] Found {existing_open_loops} existing open loops")

        # Phase 3: Apply retention optimizations
        optimized_script = script_text
        micro_payoffs_added = 0
        pattern_interrupts_added = 0
        visual_cues_added = 0

        try:
            if inject_open_loops:
                loops_to_add = max(0, open_loop_target - existing_open_loops)
                if loops_to_add > 0:
                    optimized_script = optimizer.inject_open_loops(optimized_script, count=loops_to_add)
                    improvements.append(f"Injected {loops_to_add} open loops")

            if inject_payoffs:
                before_len = len(optimized_script)
                optimized_script = optimizer.inject_micro_payoffs(optimized_script, interval_seconds=payoff_int)
                # Estimate payoffs added
                micro_payoffs_added = max(0, (len(optimized_script) - before_len) // 50)
                if micro_payoffs_added > 0:
                    improvements.append(f"Injected ~{micro_payoffs_added} micro-payoffs (every {payoff_int}s)")

            if inject_interrupts:
                before_len = len(optimized_script)
                optimized_script = optimizer.inject_pattern_interrupts(optimized_script)
                pattern_interrupts_added = max(0, (len(optimized_script) - before_len) // 40)
                if pattern_interrupts_added > 0:
                    improvements.append(f"Added {pattern_interrupts_added} pattern interrupts")

            if inject_visual_cues:
                optimized_script, visual_cues_added = self._inject_visual_cues(
                    optimized_script,
                    duration_seconds,
                    target_per_minute=self.TARGET_VISUAL_CHANGES
                )
                if visual_cues_added > 0:
                    improvements.append(f"Added {visual_cues_added} visual cue markers")

            if add_natural_pacing:
                pacer = self._get_natural_pacer()
                if pacer:
                    optimized_script = pacer.inject_breath_markers(optimized_script)
                    optimized_script = pacer.inject_pause_markers(optimized_script)
                    improvements.append("Added natural pacing markers")

        except Exception as e:
            logger.warning(f"Optimization error: {e}")
            warnings.append(f"Some optimizations failed: {str(e)[:50]}")

        # Phase 4: Count final open loops
        final_open_loops = self._count_open_loops(optimized_script)
        optimized_word_count = len(optimized_script.split())

        # Phase 5: Analyze drop-off risks
        drop_off_risks = self._analyze_drop_off_risks(optimized_script, duration_seconds)

        # Phase 6: Predict retention
        predicted_retention = self._predict_retention(
            hook_strength=hook_strength,
            open_loop_count=final_open_loops,
            micro_payoffs=micro_payoffs_added,
            pattern_interrupts=pattern_interrupts_added,
            duration_seconds=duration_seconds,
            drop_off_risks=drop_off_risks
        )

        # Create retention result
        retention_result = RetentionResult(
            hook_strength=hook_strength,
            open_loop_count=final_open_loops,
            predicted_retention=predicted_retention,
            improvements=improvements,
            drop_off_risks=drop_off_risks,
            original_word_count=original_word_count,
            optimized_word_count=optimized_word_count,
            micro_payoffs_added=micro_payoffs_added,
            pattern_interrupts_added=pattern_interrupts_added,
            visual_cues_added=visual_cues_added,
            hook_analysis=hook_analysis
        )

        # Log operation
        self.log_operation("optimize_retention", tokens=0, cost=0.0)

        logger.success(
            f"[RetentionOptimizerAgent] Optimization complete: "
            f"Hook={hook_strength:.0%}, Loops={final_open_loops}, Retention={predicted_retention:.0%}"
        )

        return AgentResult(
            success=True,
            data={
                "optimized_script": optimized_script,
                **retention_result.to_dict()
            },
            tokens_used=0,
            cost=0.0,
            metadata={
                "niche": niche,
                "duration_seconds": duration_seconds,
                "warnings": warnings
            }
        )

    def _extract_hook(self, script_text: str, word_count: int = 50) -> str:
        """Extract the hook (first N words) from script."""
        words = script_text.split()[:word_count]
        return " ".join(words)

    def _analyze_hook_strength(self, hook_text: str, niche: str) -> Dict[str, Any]:
        """
        Analyze hook strength based on proven patterns.

        Args:
            hook_text: First 50 words of script
            niche: Content niche

        Returns:
            Dict with score, patterns_found, and recommendations
        """
        hook_lower = hook_text.lower()
        score = 0.0
        patterns_found = []
        recommendations = []

        for pattern_name, pattern_info in HOOK_PATTERNS.items():
            for regex in pattern_info["patterns"]:
                if re.search(regex, hook_lower, re.IGNORECASE):
                    score += pattern_info["weight"]
                    patterns_found.append({
                        "name": pattern_name,
                        "description": pattern_info["description"],
                        "weight": pattern_info["weight"]
                    })
                    break  # Only count each pattern type once

        # Penalize weak hooks
        if score < 0.3:
            recommendations.append("Add a pattern interrupt opener (Stop, Wait, Here's the thing)")
        if "curiosity_gap" not in [p["name"] for p in patterns_found]:
            recommendations.append("Add curiosity gap (secret, hidden, truth)")
        if "specific_numbers" not in [p["name"] for p in patterns_found]:
            recommendations.append("Add specific numbers ($1,000, 5 steps, 97%)")
        if "direct_address" not in [p["name"] for p in patterns_found]:
            recommendations.append("Use direct address (you, your)")

        # Cap score at 1.0
        score = min(1.0, score)

        return {
            "score": score,
            "patterns_found": patterns_found,
            "recommendations": recommendations,
            "word_count": len(hook_text.split())
        }

    def _count_open_loops(self, script_text: str) -> int:
        """
        Count open loops in the script.

        Open loops are curiosity gaps that tease upcoming content.
        """
        open_loop_indicators = [
            r"but first",
            r"i'll reveal",
            r"stay with me",
            r"keep watching",
            r"coming up",
            r"in a moment",
            r"at the end",
            r"the best part is",
            r"what I'm about to show",
            r"there's a twist",
            r"but wait",
            r"the real secret",
            r"saved the most important",
        ]

        count = 0
        script_lower = script_text.lower()
        for pattern in open_loop_indicators:
            matches = re.findall(pattern, script_lower)
            count += len(matches)

        return count

    def _inject_visual_cues(
        self,
        script_text: str,
        duration_seconds: int,
        target_per_minute: int = 15
    ) -> Tuple[str, int]:
        """
        Inject visual cue markers for video editors.

        Args:
            script_text: Script content
            duration_seconds: Video duration
            target_per_minute: Target visual changes per minute

        Returns:
            Tuple of (modified_script, cues_added)
        """
        # Calculate how many visual cues needed
        duration_minutes = duration_seconds / 60
        target_cues = int(duration_minutes * target_per_minute)

        # Don't over-inject
        max_cues = min(target_cues // 3, 20)  # Only add ~1/3 as actual markers

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', script_text)
        if len(sentences) < max_cues * 2:
            return script_text, 0

        # Calculate spacing
        spacing = len(sentences) // (max_cues + 1)
        cues_added = 0
        result_sentences = []

        for i, sentence in enumerate(sentences):
            result_sentences.append(sentence)

            # Add visual cue after every N sentences
            if i > 0 and i % spacing == 0 and cues_added < max_cues:
                cue = VISUAL_CUE_MARKERS[cues_added % len(VISUAL_CUE_MARKERS)]
                result_sentences.append(f"\n{cue}\n")
                cues_added += 1

        return ' '.join(result_sentences), cues_added

    def _analyze_drop_off_risks(
        self,
        script_text: str,
        duration_seconds: int
    ) -> List[Dict[str, Any]]:
        """
        Analyze script for potential viewer drop-off points.

        Args:
            script_text: Optimized script
            duration_seconds: Video duration

        Returns:
            List of drop-off risk assessments
        """
        risks = []
        script_lower = script_text.lower()
        words = script_text.split()
        word_count = len(words)
        words_per_second = word_count / duration_seconds if duration_seconds > 0 else 2.5

        # Early drop-off risk (0-30 seconds)
        early_words = int(30 * words_per_second)
        early_text = " ".join(words[:early_words]).lower()

        early_risk = {
            "timestamp": "0:00 - 0:30",
            "risk_level": "low",
            "factors": [],
            "impact": DROP_OFF_INDICATORS["early_drop"]["impact"]
        }

        # Check for weak hook
        hook_patterns_found = 0
        for pattern_info in HOOK_PATTERNS.values():
            for regex in pattern_info["patterns"]:
                if re.search(regex, early_text, re.IGNORECASE):
                    hook_patterns_found += 1
                    break

        if hook_patterns_found < 3:
            early_risk["factors"].append("Weak hook (missing engagement triggers)")
            early_risk["risk_level"] = "medium"
        if hook_patterns_found < 2:
            early_risk["risk_level"] = "high"

        risks.append(early_risk)

        # Middle section risk (30s - 3 min)
        mid_start = int(30 * words_per_second)
        mid_end = int(180 * words_per_second)
        mid_text = " ".join(words[mid_start:mid_end]).lower() if mid_end < word_count else ""

        if mid_text:
            mid_risk = {
                "timestamp": "0:30 - 3:00",
                "risk_level": "low",
                "factors": [],
                "impact": DROP_OFF_INDICATORS["middle_drop"]["impact"]
            }

            # Check for open loops in middle section
            mid_loops = sum(1 for p in [r"but first", r"coming up", r"stay with me"] if p in mid_text)
            if mid_loops < 1:
                mid_risk["factors"].append("No open loops to maintain curiosity")
                mid_risk["risk_level"] = "medium"

            # Check for payoff phrases
            payoff_phrases = ["here's", "the key", "important", "takeaway", "truth"]
            payoffs_found = sum(1 for p in payoff_phrases if p in mid_text)
            if payoffs_found < 2:
                mid_risk["factors"].append("Few micro-payoffs to reward watching")
                if mid_risk["risk_level"] == "medium":
                    mid_risk["risk_level"] = "high"
                else:
                    mid_risk["risk_level"] = "medium"

            risks.append(mid_risk)

        # Late section risk (3 min+)
        if duration_seconds > 180:
            late_start = int(180 * words_per_second)
            late_text = " ".join(words[late_start:]).lower()

            late_risk = {
                "timestamp": "3:00+",
                "risk_level": "low",
                "factors": [],
                "impact": DROP_OFF_INDICATORS["late_drop"]["impact"]
            }

            # Check for CTA
            cta_phrases = ["subscribe", "like", "comment", "share", "click"]
            has_cta = any(p in late_text for p in cta_phrases)
            if not has_cta:
                late_risk["factors"].append("No call-to-action found")

            # Check for conclusion signals
            conclusion_phrases = ["in conclusion", "to wrap up", "finally", "remember"]
            has_conclusion = any(p in late_text for p in conclusion_phrases)
            if not has_conclusion:
                late_risk["factors"].append("Weak conclusion/summary")
                late_risk["risk_level"] = "medium"

            if late_risk["factors"]:
                risks.append(late_risk)

        return risks

    def _predict_retention(
        self,
        hook_strength: float,
        open_loop_count: int,
        micro_payoffs: int,
        pattern_interrupts: int,
        duration_seconds: int,
        drop_off_risks: List[Dict[str, Any]]
    ) -> float:
        """
        Predict average viewer retention based on script elements.

        Args:
            hook_strength: Hook strength score
            open_loop_count: Number of open loops
            micro_payoffs: Number of micro-payoffs
            pattern_interrupts: Number of pattern interrupts
            duration_seconds: Video duration
            drop_off_risks: Identified drop-off risks

        Returns:
            Predicted retention as float (0.0 to 1.0)
        """
        # Base retention (typically 40-60% for good content)
        base_retention = 0.45

        # Hook strength impact (strong hook = +15%, weak = -10%)
        hook_impact = (hook_strength - 0.5) * 0.30

        # Open loops impact (each loop adds ~2% retention, diminishing returns)
        loop_impact = min(0.15, open_loop_count * 0.02)

        # Micro-payoffs impact (regular payoffs add ~1% each, cap at 10%)
        payoff_impact = min(0.10, micro_payoffs * 0.01)

        # Pattern interrupts impact (help maintain attention)
        interrupt_impact = min(0.05, pattern_interrupts * 0.01)

        # Duration penalty (longer videos have lower retention)
        duration_minutes = duration_seconds / 60
        duration_penalty = max(-0.15, -0.01 * (duration_minutes - 5))  # Penalty starts after 5 min

        # Risk penalty
        risk_penalty = 0.0
        for risk in drop_off_risks:
            if risk["risk_level"] == "high":
                risk_penalty -= risk["impact"] * 0.5
            elif risk["risk_level"] == "medium":
                risk_penalty -= risk["impact"] * 0.25

        # Calculate final retention
        predicted = base_retention + hook_impact + loop_impact + payoff_impact + interrupt_impact + duration_penalty + risk_penalty

        # Clamp to realistic range (20% to 70%)
        return max(0.20, min(0.70, predicted))

    def analyze_only(
        self,
        script_text: str,
        duration_seconds: int = 600,
        niche: str = "default"
    ) -> AgentResult:
        """
        Analyze script without modifying it.

        Useful for getting metrics on an existing script.

        Args:
            script_text: Script content
            duration_seconds: Video duration
            niche: Content niche

        Returns:
            AgentResult with analysis only (no optimized_script)
        """
        return self.run(
            script_text=script_text,
            duration_seconds=duration_seconds,
            niche=niche,
            inject_open_loops=False,
            inject_payoffs=False,
            inject_interrupts=False,
            inject_visual_cues=False,
            add_natural_pacing=False
        )


# CLI entry point
def main():
    """CLI entry point for retention optimizer agent."""
    import sys
    import argparse

    if len(sys.argv) < 2:
        print("""
Retention Optimizer Agent - Script Retention Optimization

Usage:
    python -m src.agents.retention_optimizer_agent <script_file> [options]
    python -m src.agents.retention_optimizer_agent --text "Your script here..." [options]

Examples:
    python -m src.agents.retention_optimizer_agent script.txt --duration 600
    python -m src.agents.retention_optimizer_agent script.txt --niche finance --analyze-only
    python -m src.agents.retention_optimizer_agent script.txt --no-visual-cues

Options:
    --duration <seconds>    Expected video duration (default: 600)
    --niche <niche>         Content niche (finance, psychology, storytelling)
    --analyze-only          Only analyze, don't modify script
    --no-open-loops         Skip open loop injection
    --no-payoffs            Skip micro-payoff injection
    --no-interrupts         Skip pattern interrupt injection
    --no-visual-cues        Skip visual cue markers
    --output <path>         Save optimized script to file
    --json                  Output as JSON
        """)
        return

    parser = argparse.ArgumentParser(description="Optimize script retention")
    parser.add_argument("script", nargs="?", help="Script file or text")
    parser.add_argument("--text", help="Script text directly")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds")
    parser.add_argument("--niche", default="default", help="Content niche")
    parser.add_argument("--analyze-only", action="store_true", help="Analyze only")
    parser.add_argument("--no-open-loops", action="store_true", help="Skip open loops")
    parser.add_argument("--no-payoffs", action="store_true", help="Skip payoffs")
    parser.add_argument("--no-interrupts", action="store_true", help="Skip interrupts")
    parser.add_argument("--no-visual-cues", action="store_true", help="Skip visual cues")
    parser.add_argument("--output", "-o", help="Output file for optimized script")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    # Get script text
    if args.text:
        script_text = args.text
    elif args.script and Path(args.script).exists():
        with open(args.script, "r", encoding="utf-8") as f:
            script_text = f.read()
    elif args.script:
        script_text = args.script
    else:
        print("Error: Provide a script file or --text")
        return

    # Run agent
    agent = RetentionOptimizerAgent()

    if args.analyze_only:
        result = agent.analyze_only(
            script_text=script_text,
            duration_seconds=args.duration,
            niche=args.niche
        )
    else:
        result = agent.run(
            script_text=script_text,
            duration_seconds=args.duration,
            niche=args.niche,
            inject_open_loops=not args.no_open_loops,
            inject_payoffs=not args.no_payoffs,
            inject_interrupts=not args.no_interrupts,
            inject_visual_cues=not args.no_visual_cues
        )

    # Output
    if args.json:
        import json
        print(json.dumps(result.data, indent=2))
    else:
        print("\n" + "=" * 60)
        print("RETENTION OPTIMIZER RESULT")
        print("=" * 60)
        print(f"Success: {result.success}")

        if result.success:
            data = result.data
            print(f"\nHook Strength: {data.get('hook_strength', 0):.0%}")
            print(f"Open Loop Count: {data.get('open_loop_count', 0)} (target: 3+)")
            print(f"Predicted Retention: {data.get('predicted_retention', 0):.0%}")
            print(f"Original Words: {data.get('original_word_count', 0)}")
            print(f"Optimized Words: {data.get('optimized_word_count', 0)}")

            improvements = data.get('improvements', [])
            if improvements:
                print(f"\nImprovements Made:")
                for imp in improvements:
                    print(f"  - {imp}")

            hook_analysis = data.get('hook_analysis', {})
            if hook_analysis.get('recommendations'):
                print(f"\nHook Recommendations:")
                for rec in hook_analysis['recommendations'][:3]:
                    print(f"  - {rec}")

            risks = data.get('drop_off_risks', [])
            high_risks = [r for r in risks if r.get('risk_level') in ['high', 'medium']]
            if high_risks:
                print(f"\nDrop-off Risks:")
                for risk in high_risks:
                    print(f"  [{risk['risk_level'].upper()}] {risk['timestamp']}")
                    for factor in risk.get('factors', []):
                        print(f"    - {factor}")
        else:
            print(f"\nError: {result.error}")

    # Save optimized script if requested
    if args.output and result.success and not args.analyze_only:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.data.get("optimized_script", ""))
        print(f"\nOptimized script saved to: {args.output}")


if __name__ == "__main__":
    main()
