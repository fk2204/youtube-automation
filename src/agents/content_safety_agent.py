"""
Content Safety Agent - Content Moderation and Risk Assessment

Validates content for safety concerns including misinformation, harmful advice,
inappropriate content, and topics requiring human review.

Usage:
    from src.agents.content_safety_agent import ContentSafetyAgent

    agent = ContentSafetyAgent()

    # Check content safety
    result = agent.run(
        script="Your video script here...",
        title="Video Title",
        niche="finance"
    )

    if result.success:
        safety = result.data
        print(f"Safe: {safety['safe']}")
        print(f"Risk Level: {safety['risk_level']}")
        print(f"Concerns: {safety['concerns']}")

Example:
    >>> agent = ContentSafetyAgent()
    >>> result = agent.run(
    ...     script="This treatment cures cancer in 30 days guaranteed!",
    ...     niche="health"
    ... )
    >>> print(result.data['safe'])
    False
    >>> print(result.data['risk_level'])
    high
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from loguru import logger

from .base_agent import BaseAgent, AgentResult


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyResult:
    """
    Result of content safety check.

    Attributes:
        safe: Whether the content is safe to publish
        risk_level: Overall risk level (low/medium/high/critical)
        concerns: List of safety concerns found
        misinformation_flags: Specific misinformation patterns detected
        harmful_advice_flags: Harmful advice patterns detected
        inappropriate_flags: Inappropriate content patterns detected
        human_review_required: Whether human review is recommended
        human_review_reasons: Reasons for requiring human review
        score: Safety score from 0-100 (100 = safest)
    """
    safe: bool
    risk_level: str
    concerns: List[str] = field(default_factory=list)
    misinformation_flags: List[str] = field(default_factory=list)
    harmful_advice_flags: List[str] = field(default_factory=list)
    inappropriate_flags: List[str] = field(default_factory=list)
    human_review_required: bool = False
    human_review_reasons: List[str] = field(default_factory=list)
    score: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "safe": self.safe,
            "risk_level": self.risk_level,
            "concerns": self.concerns,
            "misinformation_flags": self.misinformation_flags,
            "harmful_advice_flags": self.harmful_advice_flags,
            "inappropriate_flags": self.inappropriate_flags,
            "human_review_required": self.human_review_required,
            "human_review_reasons": self.human_review_reasons,
            "score": self.score
        }


class ContentSafetyAgent(BaseAgent):
    """
    Agent for content safety and moderation.

    Detects potentially harmful content including:
    - Health misinformation
    - Financial misinformation
    - Dangerous advice
    - Inappropriate content
    - Content requiring human review

    Features:
    - Rule-based pattern matching for common issues
    - Context-aware niche-specific checks
    - Human review flagging for edge cases
    - Risk level scoring
    """

    # Health-related misinformation patterns
    HEALTH_MISINFORMATION = [
        # Cure claims
        (r'\bcure[sd]?\s+(cancer|diabetes|covid|aids|hiv)\b', "Dangerous cure claim"),
        (r'\b(miracle|secret|natural)\s+cure\b', "Unverified cure claim"),
        (r'\b(guaranteed|proven)\s+to\s+(cure|heal|treat)\b', "Unsubstantiated health claim"),
        # Anti-medical claims
        (r'\b(vaccines?\s+(cause|don\'?t\s+work)|anti-?vax)\b', "Anti-vaccine content"),
        (r'\bdoctors?\s+(don\'?t\s+want|are\s+hiding)\b', "Medical conspiracy claim"),
        (r'\b(big\s+pharma|pharma\s+conspiracy)\b', "Pharmaceutical conspiracy"),
        # Dangerous health advice
        (r'\b(stop|don\'?t)\s+taking\s+(your\s+)?(medication|medicine)\b', "Dangerous medication advice"),
        (r'\b(detox|cleanse)\s+your\s+(body|liver|kidneys)\b', "Unproven detox claim"),
        (r'\blose\s+\d+\s+(pounds?|lbs?|kg)\s+in\s+\d+\s+(days?|hours?)\b', "Unrealistic weight loss claim"),
    ]

    # Financial misinformation patterns
    FINANCIAL_MISINFORMATION = [
        # Get rich quick
        (r'\bguaranteed\s+(returns?|profit|income|money)\b', "Guaranteed returns claim"),
        (r'\bmake\s+\$?\d+[kK]?\s+(per\s+)?(day|week|month)\s+(easy|fast|passive)\b', "Unrealistic income claim"),
        (r'\b(double|triple)\s+your\s+money\s+in\s+\d+\b', "Unrealistic investment return"),
        (r'\brisk[- ]?free\s+(investment|trading|returns?)\b', "Risk-free investment claim"),
        # Pump and dump indicators
        (r'\b(buy|invest)\s+(now|today)\s+before\s+(it\'?s?\s+too\s+late|explodes?)\b', "FOMO manipulation"),
        (r'\bthis\s+(stock|crypto|coin)\s+will\s+(10x|100x|moon)\b', "Pump and dump indicator"),
        (r'\b(secret|hidden)\s+(stock|crypto|investment)\b', "Suspicious investment promotion"),
        # Tax advice
        (r'\b(avoid|evade)\s+taxes?\b', "Tax evasion suggestion"),
        (r'\b(irs|hmrc)\s+(hack|loophole|secret)\b', "Suspicious tax advice"),
    ]

    # Harmful advice patterns
    HARMFUL_ADVICE = [
        # Self-harm
        (r'\b(how\s+to\s+)?(hurt|harm)\s+(yourself|myself)\b', "Self-harm content"),
        (r'\b(suicide|suicidal|kill\s+(yourself|myself))\b', "Suicide-related content"),
        (r'\b(eating\s+disorder|anorexia|bulimia)\s+(tips?|how\s+to)\b', "Eating disorder promotion"),
        # Dangerous activities
        (r'\b(how\s+to\s+)?(make|build)\s+(a\s+)?(bomb|explosive|weapon)\b', "Weapons/explosives content"),
        (r'\b(hack|break\s+into|bypass)\s+(security|password|account)\b', "Hacking instructions"),
        (r'\bdrug\s+(recipe|synthesis|make|manufacture)\b', "Drug manufacturing content"),
        # Legal issues
        (r'\b(how\s+to\s+)?(steal|shoplift|commit\s+fraud)\b', "Criminal activity promotion"),
        (r'\b(fake\s+id|counterfeit|forge)\b', "Forgery/fraud content"),
    ]

    # Inappropriate content patterns
    INAPPROPRIATE_CONTENT = [
        # Age-inappropriate
        (r'\b(18\+|adult\s+only|nsfw|xxx)\b', "Age-restricted content indicator"),
        (r'\bexplicit\s+(content|material|sexual)\b', "Explicit content"),
        # Hate speech indicators
        (r'\b(hate|kill|destroy)\s+(all\s+)?(jews?|muslims?|blacks?|whites?|gays?)\b', "Hate speech"),
        (r'\b(racist?|bigot|supremac)\b', "Discriminatory content"),
        # Harassment
        (r'\b(dox|doxx|expose|leak)\s+(personal|private|address)\b', "Doxxing/harassment"),
        (r'\b(cyberbully|harass|stalk)\b', "Harassment content"),
    ]

    # Niche-specific sensitivity
    NICHE_SENSITIVITY = {
        "finance": {
            "extra_patterns": FINANCIAL_MISINFORMATION,
            "required_disclaimers": [
                "not financial advice",
                "consult a financial advisor",
                "do your own research"
            ],
            "risk_multiplier": 1.5
        },
        "health": {
            "extra_patterns": HEALTH_MISINFORMATION,
            "required_disclaimers": [
                "not medical advice",
                "consult your doctor",
                "consult a healthcare professional"
            ],
            "risk_multiplier": 2.0
        },
        "psychology": {
            "extra_patterns": [
                (r'\bdiagnose\s+(yourself|your)\b', "Self-diagnosis promotion"),
                (r'\b(therapy|medication)\s+(is\s+)?(useless|doesn\'t\s+work)\b', "Anti-treatment content"),
            ],
            "required_disclaimers": [
                "not professional advice",
                "seek professional help"
            ],
            "risk_multiplier": 1.3
        },
        "default": {
            "extra_patterns": [],
            "required_disclaimers": [],
            "risk_multiplier": 1.0
        }
    }

    # Content requiring human review
    HUMAN_REVIEW_TRIGGERS = [
        (r'\b(breaking|exclusive|leaked|confidential)\b', "Unverified claims"),
        (r'\b(allegedly|reportedly|sources?\s+say)\b', "Unverified sources"),
        (r'\b(controversy|controversial|scandal)\b', "Controversial topic"),
        (r'\b(political|politician|election|government)\b', "Political content"),
        (r'\b(religion|religious|god|allah|jesus|buddha)\b', "Religious content"),
        (r'\b(war|conflict|military|attack)\b', "Conflict-related content"),
    ]

    def __init__(self, provider: str = "rule_based", api_key: str = None):
        """
        Initialize the content safety agent.

        Args:
            provider: AI provider for advanced analysis (default: rule_based)
            api_key: API key for cloud providers
        """
        super().__init__(provider=provider, api_key=api_key)
        logger.info(f"ContentSafetyAgent initialized")

    def run(
        self,
        script: str = "",
        title: str = "",
        description: str = "",
        niche: str = "default",
        **kwargs
    ) -> AgentResult:
        """
        Check content for safety concerns.

        Args:
            script: Video script/narration text
            title: Video title
            description: Video description
            niche: Content niche (finance, health, psychology, etc.)
            **kwargs: Additional parameters

        Returns:
            AgentResult with SafetyResult data

        Example:
            >>> agent = ContentSafetyAgent()
            >>> result = agent.run(
            ...     script="Invest now and double your money in a week!",
            ...     niche="finance"
            ... )
            >>> print(result.data['risk_level'])
            high
        """
        logger.info(f"[ContentSafetyAgent] Running safety check for niche: {niche}")

        # Combine all text for analysis
        full_text = f"{title}\n{description}\n{script}"

        safety_result = SafetyResult(safe=True, risk_level="low", score=100)

        # Get niche-specific settings
        niche_config = self.NICHE_SENSITIVITY.get(
            niche, self.NICHE_SENSITIVITY["default"]
        )
        risk_multiplier = niche_config.get("risk_multiplier", 1.0)

        # Run all safety checks
        self._check_health_misinformation(full_text, safety_result, risk_multiplier)
        self._check_financial_misinformation(full_text, safety_result, risk_multiplier)
        self._check_harmful_advice(full_text, safety_result)
        self._check_inappropriate_content(full_text, safety_result)
        self._check_niche_specific(full_text, niche, niche_config, safety_result)
        self._check_human_review_needed(full_text, title, safety_result)
        self._check_disclaimers(full_text, niche_config, safety_result)

        # Calculate final risk level and safety status
        safety_result.score = max(0, safety_result.score)
        safety_result.risk_level = self._calculate_risk_level(safety_result)
        safety_result.safe = (
            safety_result.risk_level in ["low", "medium"] and
            len(safety_result.harmful_advice_flags) == 0
        )

        # Log results
        if safety_result.safe:
            logger.success(
                f"[ContentSafetyAgent] Content passed safety check "
                f"(risk: {safety_result.risk_level}, score: {safety_result.score})"
            )
        else:
            logger.warning(
                f"[ContentSafetyAgent] Safety concerns found "
                f"(risk: {safety_result.risk_level}, concerns: {len(safety_result.concerns)})"
            )

        return AgentResult(
            success=True,
            data=safety_result.to_dict(),
            tokens_used=0,
            cost=0.0,
            metadata={
                "niche": niche,
                "checks_performed": [
                    "health_misinformation", "financial_misinformation",
                    "harmful_advice", "inappropriate_content",
                    "niche_specific", "human_review", "disclaimers"
                ],
                "risk_multiplier": risk_multiplier
            }
        )

    def _check_health_misinformation(
        self,
        text: str,
        result: SafetyResult,
        multiplier: float = 1.0
    ):
        """Check for health-related misinformation."""
        for pattern, description in self.HEALTH_MISINFORMATION:
            if re.search(pattern, text, re.IGNORECASE):
                result.misinformation_flags.append(description)
                result.concerns.append(f"Health misinformation: {description}")
                result.score -= int(20 * multiplier)

    def _check_financial_misinformation(
        self,
        text: str,
        result: SafetyResult,
        multiplier: float = 1.0
    ):
        """Check for financial misinformation."""
        for pattern, description in self.FINANCIAL_MISINFORMATION:
            if re.search(pattern, text, re.IGNORECASE):
                result.misinformation_flags.append(description)
                result.concerns.append(f"Financial misinformation: {description}")
                result.score -= int(15 * multiplier)

    def _check_harmful_advice(self, text: str, result: SafetyResult):
        """Check for harmful or dangerous advice."""
        for pattern, description in self.HARMFUL_ADVICE:
            if re.search(pattern, text, re.IGNORECASE):
                result.harmful_advice_flags.append(description)
                result.concerns.append(f"Harmful content: {description}")
                result.score -= 40  # Severe penalty
                result.human_review_required = True
                result.human_review_reasons.append(
                    f"Flagged for: {description}"
                )

    def _check_inappropriate_content(self, text: str, result: SafetyResult):
        """Check for inappropriate content."""
        for pattern, description in self.INAPPROPRIATE_CONTENT:
            if re.search(pattern, text, re.IGNORECASE):
                result.inappropriate_flags.append(description)
                result.concerns.append(f"Inappropriate: {description}")
                result.score -= 35
                result.human_review_required = True
                result.human_review_reasons.append(
                    f"Inappropriate content: {description}"
                )

    def _check_niche_specific(
        self,
        text: str,
        niche: str,
        config: Dict,
        result: SafetyResult
    ):
        """Check niche-specific patterns."""
        extra_patterns = config.get("extra_patterns", [])
        for pattern, description in extra_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result.concerns.append(f"Niche concern ({niche}): {description}")
                result.score -= 10

    def _check_human_review_needed(
        self,
        text: str,
        title: str,
        result: SafetyResult
    ):
        """Check if content should be flagged for human review."""
        combined = f"{title}\n{text}"

        for pattern, reason in self.HUMAN_REVIEW_TRIGGERS:
            if re.search(pattern, combined, re.IGNORECASE):
                if reason not in result.human_review_reasons:
                    result.human_review_reasons.append(reason)

        # Set flag if we have reasons
        if result.human_review_reasons and not result.human_review_required:
            result.human_review_required = len(result.human_review_reasons) >= 2

    def _check_disclaimers(
        self,
        text: str,
        config: Dict,
        result: SafetyResult
    ):
        """Check if required disclaimers are present."""
        required = config.get("required_disclaimers", [])
        text_lower = text.lower()

        missing_disclaimers = []
        for disclaimer in required:
            if disclaimer.lower() not in text_lower:
                missing_disclaimers.append(disclaimer)

        if missing_disclaimers and result.score < 90:
            # Only warn about missing disclaimers if there are other concerns
            result.concerns.append(
                f"Consider adding disclaimers: {', '.join(missing_disclaimers)}"
            )

    def _calculate_risk_level(self, result: SafetyResult) -> str:
        """Calculate overall risk level based on score and flags."""
        if len(result.harmful_advice_flags) > 0:
            return "critical"

        if result.score >= 80:
            return "low"
        elif result.score >= 60:
            return "medium"
        elif result.score >= 40:
            return "high"
        else:
            return "critical"

    def get_safety_recommendations(
        self,
        safety_result: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on safety results.

        Args:
            safety_result: Dictionary from SafetyResult.to_dict()

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        if safety_result.get("misinformation_flags"):
            recommendations.append(
                "Review and remove or rephrase claims that could be interpreted as misinformation"
            )
            recommendations.append(
                "Add appropriate disclaimers (e.g., 'not professional advice')"
            )

        if safety_result.get("harmful_advice_flags"):
            recommendations.append(
                "CRITICAL: Remove content flagged as potentially harmful"
            )
            recommendations.append(
                "Consider whether this content is appropriate for your audience"
            )

        if safety_result.get("human_review_required"):
            recommendations.append(
                "Have someone else review this content before publishing"
            )

        if safety_result.get("risk_level") in ["high", "critical"]:
            recommendations.append(
                "Consider significant revisions before publishing"
            )
            recommendations.append(
                "Consult YouTube's Community Guidelines for compliance"
            )

        if not recommendations:
            recommendations.append("Content appears safe for publishing")

        return recommendations


# CLI entry point
def main():
    """CLI entry point for content safety agent."""
    import sys
    import json

    if len(sys.argv) < 2:
        print("""
Content Safety Agent - Content Moderation and Risk Assessment

Usage:
    python -m src.agents.content_safety_agent "script text" [options]
    python -m src.agents.content_safety_agent --file script.txt [options]

Options:
    --file <path>       Read script from file
    --title <title>     Video title to check
    --niche <niche>     Content niche (finance, health, psychology)
    --json              Output as JSON

Examples:
    python -m src.agents.content_safety_agent "Guaranteed 10x returns!" --niche finance
    python -m src.agents.content_safety_agent --file script.txt --niche health
        """)
        return

    # Parse arguments
    script = ""
    title = ""
    niche = "default"
    output_json = False

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--file" and i + 1 < len(sys.argv):
            with open(sys.argv[i + 1], "r", encoding="utf-8") as f:
                script = f.read()
            i += 2
        elif sys.argv[i] == "--title" and i + 1 < len(sys.argv):
            title = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--niche" and i + 1 < len(sys.argv):
            niche = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--json":
            output_json = True
            i += 1
        elif not script:
            script = sys.argv[i]
            i += 1
        else:
            i += 1

    # Run agent
    agent = ContentSafetyAgent()
    result = agent.run(script=script, title=title, niche=niche)

    # Output
    if output_json:
        print(json.dumps(result.data, indent=2))
    else:
        print("\n" + "=" * 60)
        print("CONTENT SAFETY AGENT RESULT")
        print("=" * 60)

        data = result.data
        status = "SAFE" if data["safe"] else "UNSAFE"
        risk_color = {
            "low": "",
            "medium": "[!]",
            "high": "[!!]",
            "critical": "[!!!]"
        }

        print(f"Status: {status}")
        print(f"Risk Level: {risk_color.get(data['risk_level'], '')} {data['risk_level'].upper()}")
        print(f"Safety Score: {data['score']}/100")

        if data["concerns"]:
            print(f"\nConcerns ({len(data['concerns'])}):")
            for concern in data["concerns"]:
                print(f"  - {concern}")

        if data["human_review_required"]:
            print(f"\n[ATTENTION] Human review recommended:")
            for reason in data["human_review_reasons"]:
                print(f"  - {reason}")

        # Get recommendations
        recommendations = agent.get_safety_recommendations(data)
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
