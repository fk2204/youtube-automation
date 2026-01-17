"""
AI-Powered Video Idea Generator

Combines data from multiple sources (Trends, Reddit) and uses AI
to generate and score video ideas.

Usage:
    generator = IdeaGenerator(provider="ollama")  # or "groq", "gemini"
    ideas = generator.generate_ideas(niche="python programming", count=5)
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger

from .trends import TrendResearcher, TrendTopic
from .reddit import RedditResearcher, VideoIdea


@dataclass
class ScoredIdea:
    """A video idea with scoring and metadata."""
    title: str
    description: str
    keywords: List[str]
    niche: str
    score: int              # 0-100 overall score
    trend_score: int        # 0-100 trending potential
    competition_score: int  # 0-100 (higher = less competition)
    engagement_score: int   # 0-100 predicted engagement
    source: str             # Where the idea came from
    reasoning: str          # Why this is a good idea


class IdeaGenerator:
    """Generate and score video ideas using AI and research data."""

    IDEA_GENERATION_PROMPT = """You are an expert YouTube content strategist.

Based on the following research data, generate {count} unique video ideas for the niche: "{niche}"

## Research Data:

### Trending Topics:
{trends_data}

### Popular Reddit Questions:
{reddit_data}

## Requirements:
1. Each idea should be specific and actionable
2. Focus on topics with HIGH demand but LOW competition
3. Ideas should be suitable for educational/tutorial content
4. Consider current trends and what people are actively searching for

## Output Format:
Return a JSON array with exactly {count} ideas:
```json
[
    {{
        "title": "Video title (under 60 chars, SEO optimized)",
        "description": "2-3 sentence description of the video content",
        "keywords": ["keyword1", "keyword2", "keyword3"],
        "trend_score": 85,
        "competition_score": 70,
        "engagement_score": 80,
        "reasoning": "Why this idea will perform well"
    }}
]
```

Generate the ideas now:"""

    def __init__(
        self,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the idea generator.

        Args:
            provider: AI provider (ollama, groq, gemini, claude, openai)
            api_key: API key for cloud providers
            model: Model override
        """
        # Import here to avoid circular imports
        from ..content.script_writer import get_provider

        self.ai = get_provider(provider=provider, api_key=api_key, model=model)
        self.trend_researcher = TrendResearcher()
        self.reddit_researcher = RedditResearcher()

        logger.info(f"IdeaGenerator initialized with {provider}")

    def gather_research(
        self,
        niche: str,
        subreddits: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Gather research data from all sources.

        Args:
            niche: Topic niche to research
            subreddits: Specific subreddits to search

        Returns:
            Dict with trends and reddit data
        """
        logger.info(f"Gathering research for niche: {niche}")

        # Get trending topics
        trends = []
        try:
            trend_results = self.trend_researcher.get_trending_topics(niche)
            trends = [
                {
                    "keyword": t.keyword,
                    "interest": t.interest_score,
                    "direction": t.trend_direction,
                    "related": t.related_queries[:5]
                }
                for t in trend_results
            ]
        except Exception as e:
            logger.warning(f"Trends research failed: {e}")

        # Get Reddit ideas
        reddit_ideas = []
        if self.reddit_researcher.reddit:
            try:
                ideas = self.reddit_researcher.get_video_ideas(
                    subreddits=subreddits,
                    limit=20
                )
                reddit_ideas = [
                    {
                        "topic": idea.topic,
                        "subreddit": idea.subreddit,
                        "popularity": idea.popularity_score,
                        "type": idea.idea_type
                    }
                    for idea in ideas
                ]
            except Exception as e:
                logger.warning(f"Reddit research failed: {e}")

        return {
            "trends": trends,
            "reddit": reddit_ideas
        }

    def generate_ideas(
        self,
        niche: str,
        count: int = 5,
        subreddits: Optional[List[str]] = None
    ) -> List[ScoredIdea]:
        """
        Generate scored video ideas.

        Args:
            niche: Topic niche (e.g., "python programming")
            count: Number of ideas to generate
            subreddits: Specific subreddits to research

        Returns:
            List of ScoredIdea objects sorted by score
        """
        logger.info(f"Generating {count} video ideas for: {niche}")

        # Gather research
        research = self.gather_research(niche, subreddits)

        # Format research for prompt
        trends_data = json.dumps(research["trends"], indent=2) if research["trends"] else "No trend data available"
        reddit_data = json.dumps(research["reddit"], indent=2) if research["reddit"] else "No Reddit data available"

        # Generate ideas with AI
        prompt = self.IDEA_GENERATION_PROMPT.format(
            niche=niche,
            count=count,
            trends_data=trends_data,
            reddit_data=reddit_data
        )

        try:
            response = self.ai.generate(prompt, max_tokens=2000)
            ideas_data = self._parse_json_response(response)

            # Convert to ScoredIdea objects
            ideas = []
            for data in ideas_data:
                # Calculate overall score
                trend = data.get("trend_score", 50)
                competition = data.get("competition_score", 50)
                engagement = data.get("engagement_score", 50)
                overall = int((trend + competition + engagement) / 3)

                idea = ScoredIdea(
                    title=data.get("title", "Untitled"),
                    description=data.get("description", ""),
                    keywords=data.get("keywords", []),
                    niche=niche,
                    score=overall,
                    trend_score=trend,
                    competition_score=competition,
                    engagement_score=engagement,
                    source="ai_generated",
                    reasoning=data.get("reasoning", "")
                )
                ideas.append(idea)

            # Sort by score
            ideas.sort(key=lambda x: x.score, reverse=True)

            logger.success(f"Generated {len(ideas)} ideas")
            return ideas

        except Exception as e:
            logger.error(f"Idea generation failed: {e}")
            return []

    def _parse_json_response(self, content: str) -> List[Dict]:
        """Parse JSON array from AI response."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            return json.loads(content[start:end].strip())

        if "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            return json.loads(content[start:end].strip())

        # Find array in text
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])

        raise ValueError("Could not parse JSON from response")

    def get_best_idea(
        self,
        niche: str,
        subreddits: Optional[List[str]] = None
    ) -> Optional[ScoredIdea]:
        """
        Get the single best video idea for a niche.

        Args:
            niche: Topic niche
            subreddits: Subreddits to research

        Returns:
            Best ScoredIdea or None
        """
        ideas = self.generate_ideas(niche, count=5, subreddits=subreddits)
        return ideas[0] if ideas else None

    def expand_idea(self, idea: ScoredIdea) -> Dict[str, Any]:
        """
        Expand an idea with more details for video production.

        Args:
            idea: ScoredIdea to expand

        Returns:
            Dict with expanded details
        """
        prompt = f"""Expand this video idea into a detailed outline:

Title: {idea.title}
Description: {idea.description}

Provide:
1. Target audience (who is this for?)
2. Video length recommendation (in minutes)
3. Key points to cover (5-7 bullet points)
4. Suggested thumbnail elements
5. Best posting time recommendation
6. Related video ideas for a series

Return as JSON."""

        response = self.ai.generate(prompt, max_tokens=1000)

        try:
            return self._parse_json_response(response)
        except:
            return {"raw_response": response}


# Example usage
if __name__ == "__main__":
    generator = IdeaGenerator(provider="ollama")

    print("\n" + "="*60)
    print("GENERATING VIDEO IDEAS")
    print("="*60 + "\n")

    ideas = generator.generate_ideas(
        niche="Python programming tutorials",
        count=5
    )

    for i, idea in enumerate(ideas, 1):
        print(f"\n{i}. {idea.title}")
        print(f"   Score: {idea.score}/100")
        print(f"   - Trend: {idea.trend_score}")
        print(f"   - Competition: {idea.competition_score}")
        print(f"   - Engagement: {idea.engagement_score}")
        print(f"   Keywords: {', '.join(idea.keywords)}")
        print(f"   Reasoning: {idea.reasoning}")
