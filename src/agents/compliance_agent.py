"""
Compliance Agent - YouTube Policy Compliance Checking

Validates content against YouTube's policies to prevent strikes and demonetization.
Checks for copyright issues, flagged content, required disclosures, and trademark usage.

Usage:
    from src.agents.compliance_agent import ComplianceAgent

    agent = ComplianceAgent()

    # Check script compliance
    result = agent.run(
        script="Your video script here...",
        title="Video Title",
        description="Video description",
        music_sources=["Artist - Song Name"],
        footage_sources=["Pexels", "Pixabay"]
    )

    if result.success:
        compliance = result.data
        print(f"Compliant: {compliance['compliant']}")
        print(f"Issues: {compliance['issues']}")
        print(f"Required disclosures: {compliance['required_disclosures']}")

Example:
    >>> agent = ComplianceAgent()
    >>> result = agent.run(
    ...     script="Check out my affiliate link below for 20% off!",
    ...     title="Best Products Review"
    ... )
    >>> print(result.data['required_disclosures'])
    ['affiliate_disclosure']
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from loguru import logger

from .base_agent import BaseAgent, AgentResult


@dataclass
class ComplianceResult:
    """
    Result of compliance check.

    Attributes:
        compliant: Whether the content passes all compliance checks
        issues: List of compliance issues found
        required_disclosures: List of required disclosures that must be added
        warnings: Non-critical compliance warnings
        copyright_concerns: Specific copyright-related concerns
        trademark_concerns: Specific trademark-related concerns
        policy_violations: Direct YouTube policy violations
        score: Compliance score from 0-100
    """
    compliant: bool
    issues: List[str] = field(default_factory=list)
    required_disclosures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    copyright_concerns: List[str] = field(default_factory=list)
    trademark_concerns: List[str] = field(default_factory=list)
    policy_violations: List[str] = field(default_factory=list)
    score: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compliant": self.compliant,
            "issues": self.issues,
            "required_disclosures": self.required_disclosures,
            "warnings": self.warnings,
            "copyright_concerns": self.copyright_concerns,
            "trademark_concerns": self.trademark_concerns,
            "policy_violations": self.policy_violations,
            "score": self.score
        }


class ComplianceAgent(BaseAgent):
    """
    Agent for checking YouTube policy compliance.

    Validates content against YouTube's community guidelines,
    copyright policies, and advertising requirements.

    Features:
    - Copyright issue detection (music, footage sources)
    - Sensitive content flagging
    - Required disclosure validation (sponsored, affiliate)
    - Trademark usage checking
    - Policy violation detection
    """

    # Known safe stock footage sources
    SAFE_FOOTAGE_SOURCES = {
        "pexels", "pixabay", "coverr", "unsplash", "videvo",
        "mixkit", "lifeofvids", "videezy", "original", "self-recorded",
        "stock", "licensed"
    }

    # Patterns that suggest copyrighted content
    COPYRIGHT_PATTERNS = [
        r'\b(movie|film|trailer|clip|scene)\s+(from|of)\s+[\w\s]+',
        r'\b(song|music|track|beat)\s+by\s+[\w\s]+',
        r'\b(cover|remix)\s+of\s+[\w\s]+',
        r'\b(gameplay|walkthrough)\s+of\s+[\w\s]+',
        r'\b(copyright|copyrighted|all rights reserved)\b',
    ]

    # Patterns that indicate sponsored/affiliate content
    DISCLOSURE_PATTERNS = {
        "sponsored": [
            r'\bsponsored\b', r'\bpartner(ship|ed)?\b', r'\bpromo(tion|ted)?\b',
            r'\bpaid\s+(promotion|partnership|ad)\b', r'\bbrand\s+deal\b',
        ],
        "affiliate": [
            r'\baffiliate\b', r'\bcommission\b', r'\buse\s+(my|this)\s+link\b',
            r'\b(link|code)\s+(below|in\s+description)\b', r'\bdiscount\s+code\b',
            r'\bpromo\s+code\b', r'\b\d+%\s+off\b',
        ],
        "gifted": [
            r'\bgifted\b', r'\bfree\s+product\b', r'\bsent\s+(me|for\s+review)\b',
            r'\bprovided\s+by\b', r'\bcourtesy\s+of\b',
        ],
    }

    # Sensitive topics that may trigger demonetization or restrictions
    SENSITIVE_TOPICS = {
        "health_claims": [
            r'\bcure[sd]?\b', r'\btreat(s|ed|ment)?\b.*\b(disease|illness|cancer|diabetes)\b',
            r'\bmedical\s+advice\b', r'\bhealth\s+benefits?\b.*\b(proven|guaranteed)\b',
            r'\b(miracle|wonder)\s+(cure|drug|supplement)\b',
        ],
        "financial_claims": [
            r'\bguaranteed\s+(returns?|profit|income)\b',
            r'\bget\s+rich\s+quick\b', r'\bmake\s+\$?\d+[kK]?\s+(fast|easy|quick)\b',
            r'\bfinancial\s+freedom\s+in\s+\d+\s+(days?|weeks?|months?)\b',
            r'\bnot\s+financial\s+advice\b',  # Needs disclosure
        ],
        "controversial": [
            r'\b(conspiracy|hoax)\b', r'\bfake\s+news\b',
            r'\belection\s+(fraud|rigged)\b',
        ],
        "violence": [
            r'\b(graphic|explicit)\s+(violence|content)\b',
            r'\b(blood|gore|death)\b.*\b(real|actual)\b',
        ],
        "adult": [
            r'\b(nsfw|adult|18\+|xxx)\b',
            r'\bsexual(ly)?\s+(explicit|content)\b',
        ],
    }

    # Common trademarks to check
    TRADEMARK_PATTERNS = [
        # Tech companies
        (r'\b(iphone|ipad|macbook|airpods?|apple\s+watch)\b', "Apple"),
        (r'\b(google|youtube|android|pixel|chrome)\b', "Google"),
        (r'\b(microsoft|windows|xbox|office)\b', "Microsoft"),
        (r'\b(amazon|alexa|kindle|aws)\b', "Amazon"),
        (r'\b(facebook|instagram|whatsapp|meta)\b', "Meta"),
        (r'\b(tesla|spacex)\b', "Tesla/SpaceX"),
        # Avoid using these in titles/thumbnails
        (r'\b(netflix|disney\+?|hbo|hulu)\b', "Streaming Services"),
        (r'\b(coca-?cola|pepsi|mcdonald\'?s?|starbucks)\b', "Food/Beverage Brands"),
    ]

    # YouTube-specific policy triggers
    POLICY_TRIGGERS = [
        # Clickbait that violates policies
        (r'\b(shocking|you\s+won\'?t\s+believe)\b.*\b(died|dead|death)\b', "Misleading death claims"),
        (r'\b(leaked|exposed|hacked)\b', "Potential privacy violation claims"),
        (r'\b(free\s+download|crack|keygen|pirat(e|ed))\b', "Piracy promotion"),
        (r'\b(sub\s*4\s*sub|sub\s+for\s+sub)\b', "Fake engagement"),
        (r'\b(giveaway|contest).*\b(subscribe|like|comment)\b', "Potentially against giveaway rules"),
    ]

    def __init__(self, provider: str = "rule_based", api_key: str = None):
        """
        Initialize the compliance agent.

        Args:
            provider: AI provider for advanced analysis (default: rule_based)
            api_key: API key for cloud providers
        """
        super().__init__(provider=provider, api_key=api_key)
        logger.info(f"ComplianceAgent initialized")

    def run(
        self,
        script: str = "",
        title: str = "",
        description: str = "",
        tags: List[str] = None,
        music_sources: List[str] = None,
        footage_sources: List[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        Check content for YouTube policy compliance.

        Args:
            script: Video script/narration text
            title: Video title
            description: Video description
            tags: List of video tags
            music_sources: List of music sources used (e.g., ["Artist - Song"])
            footage_sources: List of footage sources (e.g., ["Pexels", "Original"])
            **kwargs: Additional parameters

        Returns:
            AgentResult with ComplianceResult data

        Example:
            >>> agent = ComplianceAgent()
            >>> result = agent.run(
            ...     script="Use my affiliate link for 20% off!",
            ...     title="Best Products 2024",
            ...     music_sources=["Royalty Free Music"],
            ...     footage_sources=["Pexels", "Pixabay"]
            ... )
            >>> print(result.data['compliant'])
            True
            >>> print(result.data['required_disclosures'])
            ['affiliate_disclosure']
        """
        logger.info(f"[ComplianceAgent] Running compliance check")

        tags = tags or []
        music_sources = music_sources or []
        footage_sources = footage_sources or []

        # Combine all text for analysis
        full_text = f"{title}\n{description}\n{script}\n{' '.join(tags)}"

        compliance_result = ComplianceResult(compliant=True, score=100)

        # Run all checks
        self._check_copyright(
            full_text, music_sources, footage_sources, compliance_result
        )
        self._check_disclosures(full_text, compliance_result)
        self._check_sensitive_content(full_text, compliance_result)
        self._check_trademarks(title, description, compliance_result)
        self._check_policy_violations(full_text, title, compliance_result)

        # Calculate final score
        compliance_result.score = max(0, compliance_result.score)

        # Determine final compliance status
        # Critical issues make content non-compliant
        compliance_result.compliant = (
            len(compliance_result.policy_violations) == 0 and
            compliance_result.score >= 60
        )

        # Log results
        if compliance_result.compliant:
            logger.success(f"[ComplianceAgent] Content is compliant (score: {compliance_result.score})")
        else:
            logger.warning(
                f"[ComplianceAgent] Compliance issues found: "
                f"{len(compliance_result.issues)} issues, "
                f"{len(compliance_result.policy_violations)} violations"
            )

        return AgentResult(
            success=True,
            data=compliance_result.to_dict(),
            tokens_used=0,
            cost=0.0,
            metadata={
                "checks_performed": [
                    "copyright", "disclosures", "sensitive_content",
                    "trademarks", "policy_violations"
                ]
            }
        )

    def _check_copyright(
        self,
        text: str,
        music_sources: List[str],
        footage_sources: List[str],
        result: ComplianceResult
    ):
        """Check for potential copyright issues."""
        text_lower = text.lower()

        # Check music sources
        for source in music_sources:
            source_lower = source.lower()
            # Check if it's from a known safe source
            safe_music_terms = ["royalty free", "creative commons", "cc0",
                               "no copyright", "ncs", "stock", "licensed"]
            is_safe = any(term in source_lower for term in safe_music_terms)

            if not is_safe:
                # Check if it looks like copyrighted music
                if re.search(r'[\w\s]+\s*-\s*[\w\s]+', source):  # Artist - Song format
                    result.copyright_concerns.append(
                        f"Music source '{source}' may be copyrighted. "
                        "Ensure you have proper licensing."
                    )
                    result.warnings.append(f"Verify license for music: {source}")
                    result.score -= 5

        # Check footage sources
        unknown_sources = []
        for source in footage_sources:
            source_lower = source.lower()
            is_safe = any(safe in source_lower for safe in self.SAFE_FOOTAGE_SOURCES)
            if not is_safe:
                unknown_sources.append(source)

        if unknown_sources:
            result.warnings.append(
                f"Unverified footage sources: {', '.join(unknown_sources)}. "
                "Ensure proper licensing."
            )
            result.score -= 3 * len(unknown_sources)

        # Check text for copyright-related patterns
        for pattern in self.COPYRIGHT_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                result.copyright_concerns.append(
                    f"Potential copyrighted content reference detected"
                )
                result.score -= 10
                break

    def _check_disclosures(self, text: str, result: ComplianceResult):
        """Check for required FTC/YouTube disclosures."""
        text_lower = text.lower()

        for disclosure_type, patterns in self.DISCLOSURE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    disclosure_name = f"{disclosure_type}_disclosure"
                    if disclosure_name not in result.required_disclosures:
                        result.required_disclosures.append(disclosure_name)
                        result.warnings.append(
                            f"Content requires {disclosure_type} disclosure. "
                            "Add proper disclosure to description and use 'Includes paid promotion' if applicable."
                        )
                    break

    def _check_sensitive_content(self, text: str, result: ComplianceResult):
        """Check for sensitive topics that may trigger demonetization."""
        text_lower = text.lower()

        for topic, patterns in self.SENSITIVE_TOPICS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if topic in ["health_claims", "financial_claims"]:
                        result.issues.append(
                            f"Potentially problematic {topic.replace('_', ' ')} detected. "
                            "May trigger limited ads or demonetization."
                        )
                        result.score -= 15
                    elif topic in ["controversial", "violence", "adult"]:
                        result.policy_violations.append(
                            f"Content flagged for: {topic}. This may violate YouTube policies."
                        )
                        result.score -= 30
                    break

    def _check_trademarks(
        self,
        title: str,
        description: str,
        result: ComplianceResult
    ):
        """Check for trademark usage that might cause issues."""
        title_lower = title.lower()

        # Trademarks in titles are more concerning than in descriptions
        for pattern, brand in self.TRADEMARK_PATTERNS:
            if re.search(pattern, title_lower, re.IGNORECASE):
                result.trademark_concerns.append(
                    f"{brand} trademark used in title. "
                    "Ensure fair use compliance and avoid implying endorsement."
                )
                result.score -= 3

    def _check_policy_violations(
        self,
        text: str,
        title: str,
        result: ComplianceResult
    ):
        """Check for direct YouTube policy violations."""
        combined = f"{title}\n{text}"

        for pattern, violation_type in self.POLICY_TRIGGERS:
            if re.search(pattern, combined, re.IGNORECASE):
                result.policy_violations.append(
                    f"Potential policy violation: {violation_type}"
                )
                result.issues.append(f"Review content for: {violation_type}")
                result.score -= 20

    def check_music_licensing(self, music_sources: List[str]) -> Dict[str, Any]:
        """
        Check music licensing status.

        Args:
            music_sources: List of music sources/tracks

        Returns:
            Dictionary with licensing information
        """
        results = {
            "safe": [],
            "needs_verification": [],
            "potentially_copyrighted": []
        }

        safe_terms = ["royalty free", "creative commons", "cc0", "no copyright",
                     "ncs", "stock", "licensed", "original", "self-composed"]

        for source in music_sources:
            source_lower = source.lower()

            if any(term in source_lower for term in safe_terms):
                results["safe"].append(source)
            elif re.search(r'[\w\s]+\s*-\s*[\w\s]+', source):
                results["potentially_copyrighted"].append(source)
            else:
                results["needs_verification"].append(source)

        return results

    def generate_disclosure_text(
        self,
        disclosure_types: List[str]
    ) -> Dict[str, str]:
        """
        Generate required disclosure text.

        Args:
            disclosure_types: List of required disclosures

        Returns:
            Dictionary with disclosure texts for description and pinned comment
        """
        disclosures = {
            "description": [],
            "pinned_comment": [],
        }

        templates = {
            "sponsored_disclosure": {
                "description": "This video is sponsored by [SPONSOR NAME]. "
                              "All opinions are my own.",
                "pinned_comment": "This video contains a paid promotion.",
            },
            "affiliate_disclosure": {
                "description": "Some links in this description are affiliate links. "
                              "If you purchase through these links, I may earn a commission "
                              "at no extra cost to you. Thank you for supporting the channel!",
                "pinned_comment": "Disclosure: This video contains affiliate links.",
            },
            "gifted_disclosure": {
                "description": "Products in this video were provided for review. "
                              "All opinions are my own and not influenced by the brand.",
                "pinned_comment": "Disclosure: Products shown were gifted for review.",
            },
        }

        for dtype in disclosure_types:
            if dtype in templates:
                disclosures["description"].append(templates[dtype]["description"])
                disclosures["pinned_comment"].append(templates[dtype]["pinned_comment"])

        return {
            "description": "\n\n".join(disclosures["description"]),
            "pinned_comment": " ".join(disclosures["pinned_comment"]),
        }


# CLI entry point
def main():
    """CLI entry point for compliance agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Compliance Agent - YouTube Policy Compliance Checking

Usage:
    python -m src.agents.compliance_agent "script text" [options]
    python -m src.agents.compliance_agent --file script.txt [options]

Options:
    --file <path>       Read script from file
    --title <title>     Video title to check
    --music <sources>   Comma-separated music sources
    --footage <sources> Comma-separated footage sources

Examples:
    python -m src.agents.compliance_agent "Check out my affiliate link!" --title "Product Review"
    python -m src.agents.compliance_agent --file script.txt --music "Royalty Free,NCS"
        """)
        return

    # Parse arguments
    script = ""
    title = ""
    music_sources = []
    footage_sources = []

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--file" and i + 1 < len(sys.argv):
            with open(sys.argv[i + 1], "r", encoding="utf-8") as f:
                script = f.read()
            i += 2
        elif sys.argv[i] == "--title" and i + 1 < len(sys.argv):
            title = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--music" and i + 1 < len(sys.argv):
            music_sources = [s.strip() for s in sys.argv[i + 1].split(",")]
            i += 2
        elif sys.argv[i] == "--footage" and i + 1 < len(sys.argv):
            footage_sources = [s.strip() for s in sys.argv[i + 1].split(",")]
            i += 2
        elif not script:
            script = sys.argv[i]
            i += 1
        else:
            i += 1

    # Run agent
    agent = ComplianceAgent()
    result = agent.run(
        script=script,
        title=title,
        music_sources=music_sources,
        footage_sources=footage_sources
    )

    # Print result
    print("\n" + "=" * 60)
    print("COMPLIANCE AGENT RESULT")
    print("=" * 60)

    data = result.data
    status = "COMPLIANT" if data["compliant"] else "NON-COMPLIANT"
    print(f"Status: {status}")
    print(f"Score: {data['score']}/100")

    if data["issues"]:
        print(f"\nIssues ({len(data['issues'])}):")
        for issue in data["issues"]:
            print(f"  - {issue}")

    if data["required_disclosures"]:
        print(f"\nRequired Disclosures:")
        for disclosure in data["required_disclosures"]:
            print(f"  - {disclosure}")

    if data["warnings"]:
        print(f"\nWarnings:")
        for warning in data["warnings"]:
            print(f"  - {warning}")

    if data["policy_violations"]:
        print(f"\nPolicy Violations:")
        for violation in data["policy_violations"]:
            print(f"  [!] {violation}")


if __name__ == "__main__":
    main()
