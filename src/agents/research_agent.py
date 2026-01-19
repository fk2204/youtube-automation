"""
Research Agent - Topic Discovery and Trend Analysis

A token-efficient agent specialized in finding trending topics,
analyzing competitors, and generating video ideas.

Usage:
    from src.agents.research_agent import ResearchAgent

    agent = ResearchAgent()

    # Find trending topics
    result = agent.find_topics("finance", count=5)

    # Analyze competitors
    result = agent.analyze_competitors("psychology")

    # Generate viral ideas using templates
    result = agent.generate_viral_ideas(channel_id="money_blueprints")
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
from ..research.idea_generator import IdeaGenerator, ScoredIdea
from ..research.trends import TrendResearcher


@dataclass
class ResearchResult:
    """Result from research agent operations."""
    success: bool
    operation: str
    data: Dict[str, Any] = field(default_factory=dict)
    ideas: List[ScoredIdea] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    provider: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["ideas"] = [asdict(idea) for idea in self.ideas]
        return result

    def save(self, path: str = "data/research_results.json"):
        """Save result to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results
        existing = []
        if output_path.exists():
            try:
                with open(output_path) as f:
                    existing = json.load(f)
            except:
                existing = []

        # Append new result
        existing.append(self.to_dict())

        # Keep last 100 results
        if len(existing) > 100:
            existing = existing[-100:]

        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Research result saved to {path}")


class ResearchAgent:
    """
    Research Agent for topic discovery and trend analysis.

    Token-efficient design:
    - Uses Groq (free) or Ollama (local) by default
    - Caches research results to avoid duplicate calls
    - Batch processes multiple topics in single requests
    """

    # Channel to niche mapping
    CHANNEL_NICHE_MAP = {
        "money_blueprints": "finance",
        "mind_unlocked": "psychology",
        "untold_stories": "storytelling",
    }

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the research agent.

        Args:
            provider: AI provider (defaults to cost-optimized selection)
            api_key: API key for cloud providers
        """
        self.tracker = get_token_manager()
        self.optimizer = get_cost_optimizer()
        self.cache = get_prompt_cache()

        # Select provider based on budget and task complexity
        if provider is None:
            provider = self.optimizer.select_provider("idea_generation")

        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        # Initialize generators
        self.idea_gen = IdeaGenerator(provider=provider, api_key=self.api_key)
        self.trend_researcher = TrendResearcher()

        logger.info(f"ResearchAgent initialized with provider: {provider}")

    def find_topics(
        self,
        niche: str,
        count: int = 5,
        use_cache: bool = True
    ) -> ResearchResult:
        """
        Find trending topics for a niche.

        Args:
            niche: Content niche (finance, psychology, storytelling)
            count: Number of ideas to generate
            use_cache: Whether to use cached results

        Returns:
            ResearchResult with generated ideas
        """
        operation = f"find_topics_{niche}_{count}"
        logger.info(f"[ResearchAgent] Finding {count} topics for: {niche}")

        # Check cache
        cache_key = f"research_topics_{niche}_{count}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("[ResearchAgent] Using cached results")
                try:
                    cached_data = json.loads(cached)
                    ideas = [ScoredIdea(**idea) for idea in cached_data.get("ideas", [])]
                    return ResearchResult(
                        success=True,
                        operation=operation,
                        data={"source": "cache"},
                        ideas=ideas,
                        tokens_used=0,
                        cost=0.0,
                        provider="cache"
                    )
                except:
                    pass  # Cache miss or invalid data

        try:
            # Generate ideas
            ideas = self.idea_gen.generate_ideas(niche=niche, count=count)

            # Estimate tokens used (rough estimate)
            tokens_used = 500 + (count * 200)
            cost = self.tracker.record_usage(
                provider=self.provider,
                input_tokens=tokens_used // 2,
                output_tokens=tokens_used // 2,
                operation="research_find_topics"
            )

            # Cache the result
            if use_cache and ideas:
                cache_data = {"ideas": [asdict(idea) for idea in ideas]}
                self.cache.set(cache_key, json.dumps(cache_data), self.provider)

            result = ResearchResult(
                success=True,
                operation=operation,
                data={
                    "niche": niche,
                    "count_requested": count,
                    "count_generated": len(ideas)
                },
                ideas=ideas,
                tokens_used=tokens_used,
                cost=cost,
                provider=self.provider
            )

            logger.success(f"[ResearchAgent] Found {len(ideas)} topics")
            return result

        except Exception as e:
            logger.error(f"[ResearchAgent] Error finding topics: {e}")
            return ResearchResult(
                success=False,
                operation=operation,
                error=str(e),
                provider=self.provider
            )

    def generate_viral_ideas(
        self,
        channel_id: str = None,
        niche: str = None,
        count: int = 5
    ) -> ResearchResult:
        """
        Generate viral video ideas using proven templates.

        Token-efficient: Uses pre-built templates with minimal AI.

        Args:
            channel_id: Channel ID for automatic niche detection
            niche: Content niche (overrides channel_id)
            count: Number of ideas to generate

        Returns:
            ResearchResult with viral ideas
        """
        # Determine niche
        if channel_id and not niche:
            niche = self.CHANNEL_NICHE_MAP.get(channel_id, "finance")
        niche = niche or "finance"

        operation = f"viral_ideas_{niche}_{count}"
        logger.info(f"[ResearchAgent] Generating {count} viral ideas for: {niche}")

        try:
            # Use template-based generation (very few tokens)
            ideas = self.idea_gen.generate_viral_ideas(
                channel_id=channel_id,
                niche=niche,
                count=count
            )

            # Template-based = minimal tokens
            tokens_used = 50
            cost = 0.0  # Templates are essentially free

            result = ResearchResult(
                success=True,
                operation=operation,
                data={
                    "niche": niche,
                    "channel_id": channel_id,
                    "method": "template_based"
                },
                ideas=ideas,
                tokens_used=tokens_used,
                cost=cost,
                provider="template"
            )

            logger.success(f"[ResearchAgent] Generated {len(ideas)} viral ideas")
            return result

        except Exception as e:
            logger.error(f"[ResearchAgent] Error generating viral ideas: {e}")
            return ResearchResult(
                success=False,
                operation=operation,
                error=str(e),
                provider=self.provider
            )

    def analyze_trends(self, niche: str) -> ResearchResult:
        """
        Analyze current trends for a niche.

        Token-efficient: Uses Google Trends API (no AI tokens).

        Args:
            niche: Content niche to analyze

        Returns:
            ResearchResult with trend data
        """
        operation = f"analyze_trends_{niche}"
        logger.info(f"[ResearchAgent] Analyzing trends for: {niche}")

        try:
            trends = self.trend_researcher.get_trending_topics(niche)

            trend_data = []
            for trend in trends:
                trend_data.append({
                    "keyword": trend.keyword,
                    "interest_score": trend.interest_score,
                    "direction": trend.trend_direction,
                    "related_queries": trend.related_queries[:5] if trend.related_queries else []
                })

            result = ResearchResult(
                success=True,
                operation=operation,
                data={
                    "niche": niche,
                    "trends": trend_data,
                    "trend_count": len(trend_data)
                },
                tokens_used=0,  # No AI tokens for trend research
                cost=0.0,
                provider="google_trends"
            )

            logger.success(f"[ResearchAgent] Found {len(trend_data)} trends")
            return result

        except Exception as e:
            logger.error(f"[ResearchAgent] Error analyzing trends: {e}")
            return ResearchResult(
                success=False,
                operation=operation,
                error=str(e),
                provider="google_trends"
            )

    def analyze_competitors(self, niche: str) -> ResearchResult:
        """
        Get competitor insights for a niche.

        Token-efficient: Uses pre-compiled competitor data.

        Args:
            niche: Content niche

        Returns:
            ResearchResult with competitor insights
        """
        from ..utils.best_practices import get_best_practices, get_niche_metrics

        operation = f"analyze_competitors_{niche}"
        logger.info(f"[ResearchAgent] Analyzing competitors for: {niche}")

        try:
            practices = get_best_practices(niche)
            metrics = get_niche_metrics(niche)

            result = ResearchResult(
                success=True,
                operation=operation,
                data={
                    "niche": niche,
                    "metrics": metrics,
                    "viral_patterns": practices.get("viral_title_patterns", [])[:5],
                    "hook_formulas": practices.get("hook_formulas", [])[:3],
                    "best_practices": practices.get("retention_best_practices", {})
                },
                tokens_used=0,  # Pre-compiled data
                cost=0.0,
                provider="local"
            )

            logger.success(f"[ResearchAgent] Competitor analysis complete for {niche}")
            return result

        except Exception as e:
            logger.error(f"[ResearchAgent] Error analyzing competitors: {e}")
            return ResearchResult(
                success=False,
                operation=operation,
                error=str(e),
                provider="local"
            )

    def get_best_idea_for_channel(self, channel_id: str) -> Optional[ScoredIdea]:
        """
        Get the single best video idea for a channel.

        Combines viral templates with trend data for optimal results.

        Args:
            channel_id: Channel ID

        Returns:
            Best ScoredIdea or None
        """
        logger.info(f"[ResearchAgent] Getting best idea for channel: {channel_id}")

        # First try viral templates (free)
        viral_result = self.generate_viral_ideas(channel_id=channel_id, count=3)

        if viral_result.success and viral_result.ideas:
            best_idea = viral_result.ideas[0]
            logger.info(f"[ResearchAgent] Best idea: {best_idea.title} (score: {best_idea.score})")
            return best_idea

        # Fallback to AI-generated ideas
        niche = self.CHANNEL_NICHE_MAP.get(channel_id, "finance")
        ai_result = self.find_topics(niche, count=3)

        if ai_result.success and ai_result.ideas:
            return ai_result.ideas[0]

        return None

    def run(self, command: str, **kwargs) -> ResearchResult:
        """
        Main entry point for CLI usage.

        Args:
            command: Command string (e.g., "find trending finance topics")
            **kwargs: Additional parameters

        Returns:
            ResearchResult
        """
        command_lower = command.lower()

        # Parse command
        if "viral" in command_lower:
            niche = kwargs.get("niche") or self._extract_niche(command)
            return self.generate_viral_ideas(niche=niche, count=kwargs.get("count", 5))

        elif "trend" in command_lower:
            niche = kwargs.get("niche") or self._extract_niche(command)
            return self.analyze_trends(niche)

        elif "competitor" in command_lower:
            niche = kwargs.get("niche") or self._extract_niche(command)
            return self.analyze_competitors(niche)

        else:
            # Default: find topics
            niche = kwargs.get("niche") or self._extract_niche(command)
            return self.find_topics(niche, count=kwargs.get("count", 5))

    def _extract_niche(self, text: str) -> str:
        """Extract niche from text."""
        text_lower = text.lower()
        for niche in ["finance", "psychology", "storytelling", "money", "mind", "story"]:
            if niche in text_lower:
                # Normalize aliases
                if niche in ["money"]:
                    return "finance"
                if niche in ["mind"]:
                    return "psychology"
                if niche in ["story"]:
                    return "storytelling"
                return niche
        return "finance"  # Default


# CLI entry point
def main():
    """CLI entry point for research agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Research Agent - Topic Discovery and Trend Analysis

Usage:
    python -m src.agents.research_agent "find trending finance topics"
    python -m src.agents.research_agent "generate viral ideas" --niche psychology
    python -m src.agents.research_agent "analyze competitors" --niche storytelling
    python -m src.agents.research_agent "analyze trends" --niche finance

Options:
    --niche <niche>     Content niche (finance, psychology, storytelling)
    --count <n>         Number of ideas to generate (default: 5)
    --channel <id>      Channel ID for automatic niche detection
    --save              Save results to data/research_results.json
        """)
        return

    # Parse arguments
    command = sys.argv[1]
    kwargs = {}

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--niche" and i + 1 < len(sys.argv):
            kwargs["niche"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--count" and i + 1 < len(sys.argv):
            kwargs["count"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--channel" and i + 1 < len(sys.argv):
            kwargs["channel_id"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--save":
            kwargs["save"] = True
            i += 1
        else:
            command += " " + sys.argv[i]
            i += 1

    # Run agent
    agent = ResearchAgent()
    result = agent.run(command, **kwargs)

    # Print result
    print("\n" + "=" * 60)
    print("RESEARCH AGENT RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Operation: {result.operation}")
    print(f"Provider: {result.provider}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Cost: ${result.cost:.4f}")

    if result.ideas:
        print(f"\nIdeas ({len(result.ideas)}):")
        for i, idea in enumerate(result.ideas[:5], 1):
            print(f"  {i}. {idea.title}")
            print(f"     Score: {idea.score} | Trend: {idea.trend_score} | Engagement: {idea.engagement_score}")

    if result.data:
        print(f"\nData: {json.dumps(result.data, indent=2)[:500]}...")

    if result.error:
        print(f"\nError: {result.error}")

    # Save if requested
    if kwargs.get("save"):
        result.save()
        print(f"\nResults saved to data/research_results.json")


if __name__ == "__main__":
    main()
