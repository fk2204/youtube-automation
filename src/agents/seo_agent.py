"""
SEO Optimizer Agent - Title, Description, and Tag Optimization

A token-efficient agent specialized in optimizing video metadata
for YouTube discoverability.

Usage:
    from src.agents.seo_agent import SEOAgent

    agent = SEOAgent()

    # Optimize title
    result = agent.optimize_title("How to save money", niche="finance")

    # Optimize all metadata
    result = agent.optimize_metadata(title, description, tags, niche="finance")

    # Generate tags from transcript
    result = agent.generate_tags(transcript, niche="finance")
"""

import os
import json
import re
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
    get_best_practices,
    POWER_WORDS,
    VIRAL_TITLE_PATTERNS,
    SEO_PATTERNS
)


@dataclass
class SEOResult:
    """Result from SEO agent operations."""
    success: bool
    operation: str
    original: Dict[str, Any] = field(default_factory=dict)
    optimized: Dict[str, Any] = field(default_factory=dict)
    score_before: int = 0
    score_after: int = 0
    changes: List[str] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    provider: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary."""
        improvement = self.score_after - self.score_before
        direction = "improved" if improvement > 0 else "unchanged" if improvement == 0 else "decreased"

        lines = [
            f"SEO Optimization: {direction.upper()}",
            f"Score: {self.score_before} -> {self.score_after} ({'+' if improvement >= 0 else ''}{improvement})",
            ""
        ]

        if self.changes:
            lines.append("Changes made:")
            for change in self.changes[:5]:
                lines.append(f"  - {change}")

        return "\n".join(lines)

    def save(self, path: str = None):
        """Save result to JSON file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/seo_optimizations/optimization_{timestamp}.json"

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"SEO optimization saved to {path}")


class SEOAgent:
    """
    SEO Optimizer Agent for YouTube metadata.

    Token-efficient design:
    - Uses rule-based optimization by default (0 tokens)
    - AI enhancement only when requested
    - Caches optimization patterns
    """

    # Current year for time-sensitive content
    CURRENT_YEAR = str(datetime.now().year)

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the SEO agent.

        Args:
            provider: AI provider (for AI-enhanced mode)
            api_key: API key for cloud providers
        """
        self.tracker = get_token_manager()
        self.optimizer = get_cost_optimizer()
        self.cache = get_prompt_cache()

        # Select provider for AI optimization
        if provider is None:
            provider = self.optimizer.select_provider("title_generation")

        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        logger.info(f"SEOAgent initialized with provider: {provider}")

    def optimize_title(
        self,
        title: str,
        niche: str = "default",
        use_ai: bool = False
    ) -> SEOResult:
        """
        Optimize a video title for SEO and CTR.

        Token cost: ZERO (rule-based) or LOW (AI-enhanced)

        Args:
            title: Original title
            niche: Content niche
            use_ai: Whether to use AI for enhancement

        Returns:
            SEOResult with optimized title
        """
        operation = f"optimize_title_{niche}"
        logger.info(f"[SEOAgent] Optimizing title: {title[:50]}...")

        original = {"title": title}
        changes = []

        # Score original
        original_validation = validate_title(title, niche)
        score_before = int(original_validation.score * 100)

        # Apply rule-based optimizations
        optimized_title = title

        # 1. Add current year if time-sensitive
        time_sensitive_words = ["best", "top", "guide", "tips", "strategy", "review"]
        if any(word in title.lower() for word in time_sensitive_words):
            if self.CURRENT_YEAR not in title:
                optimized_title = f"{optimized_title} ({self.CURRENT_YEAR})"
                changes.append(f"Added current year ({self.CURRENT_YEAR})")

        # 2. Add numbers if missing
        if not re.search(r'\d+', optimized_title):
            # Try to add a number at the start
            number_patterns = {
                "how to": "5 Ways to",
                "ways to": "7 Ways to",
                "tips": "10 Tips",
                "secrets": "5 Secrets",
                "mistakes": "7 Mistakes",
            }
            for pattern, replacement in number_patterns.items():
                if pattern in optimized_title.lower():
                    optimized_title = optimized_title.lower().replace(pattern, replacement.lower(), 1)
                    optimized_title = optimized_title.title()
                    changes.append("Added specific number")
                    break

        # 3. Add power words if missing
        has_power_word = any(
            word.lower() in optimized_title.lower()
            for category in POWER_WORDS.values()
            for word in category
        )
        if not has_power_word:
            # Add based on niche
            niche_power_words = {
                "finance": "Secret",
                "psychology": "Hidden",
                "storytelling": "Untold"
            }
            if niche in niche_power_words:
                power_word = niche_power_words[niche]
                optimized_title = f"The {power_word} {optimized_title}"
                changes.append(f"Added power word: {power_word}")

        # 4. Optimize length (under 60 chars)
        if len(optimized_title) > 60:
            # Try to shorten
            optimized_title = optimized_title[:57] + "..."
            changes.append("Shortened for display")

        # 5. Title case
        if optimized_title == optimized_title.lower() or optimized_title == optimized_title.upper():
            optimized_title = optimized_title.title()
            changes.append("Applied title case")

        # Use AI for further enhancement if requested
        if use_ai:
            ai_result = self._ai_enhance_title(optimized_title, niche)
            if ai_result:
                optimized_title = ai_result["title"]
                changes.extend(ai_result.get("changes", []))

        # Score optimized
        optimized_validation = validate_title(optimized_title, niche)
        score_after = int(optimized_validation.score * 100)

        result = SEOResult(
            success=True,
            operation=operation,
            original=original,
            optimized={"title": optimized_title},
            score_before=score_before,
            score_after=score_after,
            changes=changes,
            tokens_used=0 if not use_ai else 500,
            cost=0.0,
            provider="rule_based" if not use_ai else self.provider
        )

        logger.success(f"[SEOAgent] Title optimized: {score_before} -> {score_after}")
        return result

    def optimize_description(
        self,
        description: str,
        title: str = "",
        niche: str = "default"
    ) -> SEOResult:
        """
        Optimize a video description for SEO.

        Token cost: ZERO (rule-based)

        Args:
            description: Original description
            title: Video title for context
            niche: Content niche

        Returns:
            SEOResult with optimized description
        """
        operation = f"optimize_description_{niche}"
        logger.info(f"[SEOAgent] Optimizing description")

        original = {"description": description}
        changes = []
        optimized_desc = description

        # 1. Check length
        if len(description) < 200:
            changes.append("Warning: Description is short (< 200 chars)")

        # 2. Add timestamps if missing
        if not re.search(r'\d{1,2}:\d{2}', description):
            # Add placeholder timestamps
            timestamp_section = "\n\nTimestamps:\n00:00 Introduction\n01:00 Main Content\n05:00 Conclusion"
            optimized_desc += timestamp_section
            changes.append("Added timestamp placeholders")

        # 3. Ensure keywords at start
        practices = get_best_practices(niche)
        required_tags = SEO_PATTERNS.get(niche, {}).get("required_tags", [])

        # Check if primary keyword in first 200 chars
        first_200 = optimized_desc[:200].lower()
        keywords_in_start = [tag for tag in required_tags[:3] if tag.lower() in first_200]

        if not keywords_in_start and required_tags:
            # Prepend a keyword-rich sentence
            keyword = required_tags[0]
            intro = f"Learn about {keyword} in this video. "
            optimized_desc = intro + optimized_desc
            changes.append(f"Added keyword '{keyword}' to start")

        # 4. Add hashtags if missing
        if "#" not in optimized_desc:
            niche_hashtags = {
                "finance": "#PersonalFinance #MoneyTips #Investing",
                "psychology": "#Psychology #MindHacks #SelfImprovement",
                "storytelling": "#TrueStory #Documentary #Explained"
            }
            hashtags = niche_hashtags.get(niche, "#YouTube #Educational")
            optimized_desc += f"\n\n{hashtags}"
            changes.append("Added relevant hashtags")

        # 5. Add CTA if missing
        cta_words = ["subscribe", "like", "comment", "notification"]
        if not any(word in optimized_desc.lower() for word in cta_words):
            cta = "\n\nDon't forget to LIKE, SUBSCRIBE, and hit the notification bell for more content!"
            optimized_desc += cta
            changes.append("Added call to action")

        # Score (simple heuristic)
        score_before = min(100, len(description) // 10)
        score_after = min(100, len(optimized_desc) // 10 + len(changes) * 10)

        result = SEOResult(
            success=True,
            operation=operation,
            original=original,
            optimized={"description": optimized_desc},
            score_before=score_before,
            score_after=score_after,
            changes=changes,
            tokens_used=0,
            cost=0.0,
            provider="rule_based"
        )

        logger.success(f"[SEOAgent] Description optimized")
        return result

    def optimize_tags(
        self,
        tags: List[str],
        title: str = "",
        niche: str = "default"
    ) -> SEOResult:
        """
        Optimize video tags for SEO.

        Token cost: ZERO (rule-based)

        Args:
            tags: Original tags
            title: Video title for context
            niche: Content niche

        Returns:
            SEOResult with optimized tags
        """
        operation = f"optimize_tags_{niche}"
        logger.info(f"[SEOAgent] Optimizing {len(tags)} tags")

        original = {"tags": tags}
        changes = []

        # Get required tags for niche
        required_tags = SEO_PATTERNS.get(niche, {}).get("required_tags", [])
        current_tags_lower = [t.lower() for t in tags]

        optimized_tags = list(tags)

        # 1. Add missing required tags
        for req_tag in required_tags:
            if req_tag.lower() not in current_tags_lower:
                optimized_tags.append(req_tag)
                changes.append(f"Added required tag: {req_tag}")

        # 2. Add title words as tags
        title_words = re.findall(r'\b\w{4,}\b', title.lower())
        for word in title_words[:5]:
            if word not in current_tags_lower and word not in ["that", "this", "with", "from", "your"]:
                optimized_tags.append(word)
                changes.append(f"Added title keyword: {word}")

        # 3. Add year tag
        if self.CURRENT_YEAR not in [t for t in tags]:
            optimized_tags.append(self.CURRENT_YEAR)
            changes.append(f"Added year tag: {self.CURRENT_YEAR}")

        # 4. Remove duplicates (case-insensitive)
        seen = set()
        unique_tags = []
        for tag in optimized_tags:
            if tag.lower() not in seen:
                seen.add(tag.lower())
                unique_tags.append(tag)
        optimized_tags = unique_tags

        # 5. Limit to 15 most relevant
        if len(optimized_tags) > 15:
            changes.append(f"Trimmed from {len(optimized_tags)} to 15 tags")
            optimized_tags = optimized_tags[:15]

        # Score
        score_before = min(100, len(tags) * 6)
        score_after = min(100, len(optimized_tags) * 6 + sum(1 for t in required_tags if t.lower() in [x.lower() for x in optimized_tags]) * 5)

        result = SEOResult(
            success=True,
            operation=operation,
            original=original,
            optimized={"tags": optimized_tags},
            score_before=score_before,
            score_after=score_after,
            changes=changes,
            tokens_used=0,
            cost=0.0,
            provider="rule_based"
        )

        logger.success(f"[SEOAgent] Tags optimized: {len(tags)} -> {len(optimized_tags)}")
        return result

    def optimize_metadata(
        self,
        title: str,
        description: str,
        tags: List[str],
        niche: str = "default",
        use_ai: bool = False
    ) -> SEOResult:
        """
        Optimize all metadata at once.

        Args:
            title: Video title
            description: Video description
            tags: Video tags
            niche: Content niche
            use_ai: Whether to use AI enhancement

        Returns:
            SEOResult with all optimized metadata
        """
        operation = f"optimize_metadata_{niche}"
        logger.info(f"[SEOAgent] Optimizing full metadata")

        # Optimize each component
        title_result = self.optimize_title(title, niche, use_ai)
        desc_result = self.optimize_description(description, title, niche)
        tags_result = self.optimize_tags(tags, title, niche)

        # Combine results
        original = {
            "title": title,
            "description": description,
            "tags": tags
        }

        optimized = {
            "title": title_result.optimized.get("title", title),
            "description": desc_result.optimized.get("description", description),
            "tags": tags_result.optimized.get("tags", tags)
        }

        changes = title_result.changes + desc_result.changes + tags_result.changes

        # Combined score
        score_before = (title_result.score_before + desc_result.score_before + tags_result.score_before) // 3
        score_after = (title_result.score_after + desc_result.score_after + tags_result.score_after) // 3

        result = SEOResult(
            success=True,
            operation=operation,
            original=original,
            optimized=optimized,
            score_before=score_before,
            score_after=score_after,
            changes=changes,
            tokens_used=title_result.tokens_used,
            cost=title_result.cost,
            provider=title_result.provider
        )

        logger.success(f"[SEOAgent] Full metadata optimized: {score_before} -> {score_after}")
        return result

    def generate_tags(
        self,
        content: str,
        niche: str = "default",
        use_ai: bool = False
    ) -> SEOResult:
        """
        Generate tags from content/transcript.

        Args:
            content: Video transcript or script
            niche: Content niche
            use_ai: Use AI for tag generation

        Returns:
            SEOResult with generated tags
        """
        operation = f"generate_tags_{niche}"
        logger.info(f"[SEOAgent] Generating tags from content")

        # Extract keywords from content
        words = re.findall(r'\b\w{4,}\b', content.lower())

        # Count frequency
        word_counts = {}
        stopwords = {"this", "that", "with", "from", "your", "have", "been", "were", "they", "their", "about", "which", "would", "could", "should", "there", "where", "when", "what", "will", "into", "just", "more", "some", "very", "than"}

        for word in words:
            if word not in stopwords and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Generate tags
        tags = [word for word, _ in sorted_words[:10]]

        # Add niche-specific required tags
        required_tags = SEO_PATTERNS.get(niche, {}).get("required_tags", [])
        for tag in required_tags[:5]:
            if tag.lower() not in [t.lower() for t in tags]:
                tags.append(tag)

        result = SEOResult(
            success=True,
            operation=operation,
            original={"content_length": len(content)},
            optimized={"tags": tags[:15]},
            score_before=0,
            score_after=len(tags) * 6,
            changes=[f"Generated {len(tags)} tags from content"],
            tokens_used=0,
            cost=0.0,
            provider="rule_based"
        )

        return result

    def _ai_enhance_title(
        self,
        title: str,
        niche: str
    ) -> Optional[Dict[str, Any]]:
        """Use AI to enhance title."""
        # Check cache first
        cache_key = f"seo_title_{hash(title)}_{niche}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except:
                pass

        try:
            from ..content.script_writer import get_provider
            ai = get_provider(self.provider, self.api_key)

            prompt = f"""Improve this YouTube video title for {niche} content:

Original: {title}

Requirements:
- Under 60 characters
- Include a number if possible
- Include a power word (secret, proven, ultimate, etc.)
- Create curiosity or urgency

Respond with ONLY a JSON object:
{{
    "title": "improved title here",
    "changes": ["change 1", "change 2"]
}}"""

            response = ai.generate(prompt, max_tokens=200)

            # Parse response
            result = self._parse_json_response(response)

            # Record token usage
            self.tracker.record_usage(
                provider=self.provider,
                input_tokens=300,
                output_tokens=100,
                operation="seo_ai_title"
            )

            # Cache result
            self.cache.set(cache_key, json.dumps(result), self.provider)

            return result

        except Exception as e:
            logger.warning(f"AI title enhancement failed: {e}")
            return None

    def _parse_json_response(self, content: str) -> Dict:
        """Parse JSON from AI response."""
        try:
            return json.loads(content)
        except:
            pass

        # Try extracting from markdown
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                try:
                    return json.loads(content[start:end].strip())
                except:
                    pass

        # Try finding JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end])
            except:
                pass

        return {}

    def run(self, command: str = None, **kwargs) -> SEOResult:
        """
        Main entry point for CLI usage.

        Args:
            command: Command or title to optimize
            **kwargs: Parameters (title, description, tags, niche, etc.)

        Returns:
            SEOResult
        """
        niche = kwargs.get("niche", "default")
        use_ai = kwargs.get("ai", False)

        # If file provided, load metadata
        if kwargs.get("file"):
            file_path = kwargs["file"]
            with open(file_path) as f:
                data = json.load(f)
            return self.optimize_metadata(
                title=data.get("title", ""),
                description=data.get("description", ""),
                tags=data.get("tags", []),
                niche=niche,
                use_ai=use_ai
            )

        # If title provided
        title = kwargs.get("title") or command
        if title:
            return self.optimize_title(title, niche, use_ai)

        # If transcript provided
        if kwargs.get("transcript"):
            with open(kwargs["transcript"]) as f:
                content = f.read()
            return self.generate_tags(content, niche, use_ai)

        return SEOResult(
            success=False,
            operation="unknown",
            error="No input provided"
        )


# CLI entry point
def main():
    """CLI entry point for SEO agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
SEO Optimizer Agent - Metadata Optimization

Usage:
    python -m src.agents.seo_agent --title "How to save money"
    python -m src.agents.seo_agent --file output/metadata.json
    python -m src.agents.seo_agent --transcript output/transcript.txt --generate-tags

Options:
    --title <title>         Title to optimize
    --file <path>           JSON file with title, description, tags
    --transcript <path>     Generate tags from transcript
    --niche <niche>         Content niche (finance, psychology, storytelling)
    --ai                    Use AI enhancement (uses tokens)
    --save                  Save results
    --json                  Output as JSON

Examples:
    python -m src.agents.seo_agent --title "Money tips" --niche finance
    python -m src.agents.seo_agent --file output/video.json --niche psychology --ai
    python -m src.agents.seo_agent --transcript output/script.txt --generate-tags
        """)
        return

    # Parse arguments
    kwargs = {}
    i = 1
    output_json = False
    save_result = False

    while i < len(sys.argv):
        if sys.argv[i] == "--title" and i + 1 < len(sys.argv):
            kwargs["title"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--file" and i + 1 < len(sys.argv):
            kwargs["file"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--transcript" and i + 1 < len(sys.argv):
            kwargs["transcript"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--niche" and i + 1 < len(sys.argv):
            kwargs["niche"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ai":
            kwargs["ai"] = True
            i += 1
        elif sys.argv[i] == "--save":
            save_result = True
            i += 1
        elif sys.argv[i] == "--json":
            output_json = True
            i += 1
        elif sys.argv[i] == "--generate-tags":
            i += 1
        else:
            i += 1

    # Run agent
    agent = SEOAgent()
    result = agent.run(**kwargs)

    # Output
    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("SEO AGENT RESULT")
        print("=" * 60)
        print(result.summary())

        if result.optimized:
            print("\nOptimized:")
            for key, value in result.optimized.items():
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(value[:5])}...")
                else:
                    print(f"  {key}: {value[:80]}..." if len(str(value)) > 80 else f"  {key}: {value}")

    # Save if requested
    if save_result:
        result.save()


if __name__ == "__main__":
    main()
