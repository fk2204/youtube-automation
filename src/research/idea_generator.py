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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        provider: str = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the idea generator.

        Args:
            provider: AI provider (ollama, groq, gemini, claude, openai). Defaults to AI_PROVIDER env var or "ollama".
            api_key: API key for cloud providers
            model: Model override
        """
        # Import here to avoid circular imports
        from ..content.script_writer import get_provider

        # Use environment variable if provider not specified
        if provider is None:
            provider = os.getenv("AI_PROVIDER", "ollama")

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
            Dict with trends and reddit data (always returns valid structure)
        """
        # Input validation
        if not niche or not isinstance(niche, str):
            niche = "general"
            logger.warning("Invalid niche provided, using 'general'")

        niche = niche.strip() or "general"
        logger.info(f"Gathering research for niche: {niche}")

        # Get trending topics
        trends = []
        try:
            trend_results = self.trend_researcher.get_trending_topics(niche)
            # Defensive check - ensure we have a list
            if trend_results and isinstance(trend_results, list):
                for t in trend_results:
                    # Validate each trend object before accessing
                    if t and hasattr(t, 'keyword') and hasattr(t, 'interest_score'):
                        related_queries = []
                        if hasattr(t, 'related_queries') and t.related_queries:
                            # Safe slice with bounds check
                            related_queries = list(t.related_queries)[:5] if isinstance(t.related_queries, (list, tuple)) else []

                        trends.append({
                            "keyword": str(t.keyword) if t.keyword else niche,
                            "interest": int(t.interest_score) if t.interest_score is not None else 50,
                            "direction": str(t.trend_direction) if hasattr(t, 'trend_direction') and t.trend_direction else "stable",
                            "related": related_queries
                        })
        except (AttributeError, TypeError, IndexError) as e:
            logger.warning(f"Trends research data parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Trends research failed: {e}")

        # Get Reddit ideas
        reddit_ideas = []
        try:
            if self.reddit_researcher and hasattr(self.reddit_researcher, 'reddit') and self.reddit_researcher.reddit:
                ideas = self.reddit_researcher.get_video_ideas(
                    subreddits=subreddits,
                    limit=20
                )
                # Defensive check - ensure we have a list
                if ideas and isinstance(ideas, list):
                    for idea in ideas:
                        # Validate each idea object before accessing
                        if idea and hasattr(idea, 'topic'):
                            reddit_ideas.append({
                                "topic": str(idea.topic) if idea.topic else "",
                                "subreddit": str(idea.subreddit) if hasattr(idea, 'subreddit') and idea.subreddit else "unknown",
                                "popularity": int(idea.popularity_score) if hasattr(idea, 'popularity_score') and idea.popularity_score is not None else 0,
                                "type": str(idea.idea_type) if hasattr(idea, 'idea_type') and idea.idea_type else "general"
                            })
        except (AttributeError, TypeError, IndexError) as e:
            logger.warning(f"Reddit research data parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Reddit research failed: {e}")

        return {
            "trends": trends,
            "reddit": reddit_ideas
        }

    def _get_fallback_ideas(self, niche: str, count: int = 5) -> List[ScoredIdea]:
        """
        Generate fallback ideas when AI generation fails.

        Args:
            niche: Topic niche
            count: Number of ideas to generate

        Returns:
            List of generic but usable ScoredIdea objects
        """
        fallback_templates = [
            {
                "title": f"Complete {niche} Tutorial for Beginners",
                "description": f"A comprehensive beginner's guide to {niche} covering all the basics you need to know.",
                "keywords": [niche, "tutorial", "beginners", "guide", "learn"],
                "trend_score": 70,
                "competition_score": 60,
                "engagement_score": 75,
                "reasoning": "Beginner tutorials have consistent demand and good engagement."
            },
            {
                "title": f"Top 10 {niche} Tips You Need to Know",
                "description": f"Essential tips and tricks for {niche} that will improve your skills immediately.",
                "keywords": [niche, "tips", "tricks", "best practices"],
                "trend_score": 65,
                "competition_score": 55,
                "engagement_score": 70,
                "reasoning": "List-based content performs well and is easy to consume."
            },
            {
                "title": f"{niche} in 2024: What's Changed",
                "description": f"Discover the latest updates and trends in {niche} for the current year.",
                "keywords": [niche, "2024", "trends", "updates", "new"],
                "trend_score": 75,
                "competition_score": 50,
                "engagement_score": 65,
                "reasoning": "Time-sensitive content attracts viewers looking for current information."
            },
            {
                "title": f"Common {niche} Mistakes to Avoid",
                "description": f"Learn from others' mistakes and avoid these common pitfalls in {niche}.",
                "keywords": [niche, "mistakes", "avoid", "errors", "problems"],
                "trend_score": 60,
                "competition_score": 65,
                "engagement_score": 70,
                "reasoning": "Problem-solving content addresses viewer pain points directly."
            },
            {
                "title": f"Advanced {niche} Techniques Explained",
                "description": f"Take your {niche} skills to the next level with these advanced techniques.",
                "keywords": [niche, "advanced", "techniques", "pro tips", "expert"],
                "trend_score": 55,
                "competition_score": 70,
                "engagement_score": 60,
                "reasoning": "Advanced content targets dedicated learners with high retention."
            },
        ]

        ideas = []
        for i, template in enumerate(fallback_templates[:count]):
            trend = template.get("trend_score", 50)
            competition = template.get("competition_score", 50)
            engagement = template.get("engagement_score", 50)
            overall = int((trend + competition + engagement) / 3)

            ideas.append(ScoredIdea(
                title=template["title"],
                description=template["description"],
                keywords=template.get("keywords", [niche]),
                niche=niche,
                score=overall,
                trend_score=trend,
                competition_score=competition,
                engagement_score=engagement,
                source="fallback",
                reasoning=template.get("reasoning", "Fallback idea generated due to API failure.")
            ))

        return ideas

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
            List of ScoredIdea objects sorted by score (never empty - falls back to generic ideas)
        """
        # Input validation
        if not niche or not isinstance(niche, str):
            niche = "general topics"
            logger.warning("Invalid niche provided, using 'general topics'")

        niche = niche.strip() or "general topics"
        count = max(1, min(count, 20)) if isinstance(count, int) else 5

        logger.info(f"Generating {count} video ideas for: {niche}")

        # Gather research
        try:
            research = self.gather_research(niche, subreddits)
        except Exception as e:
            logger.warning(f"Research gathering failed: {e}")
            research = {"trends": [], "reddit": []}

        # Format research for prompt with safe access
        trends_data = "No trend data available"
        reddit_data = "No Reddit data available"

        try:
            if research and isinstance(research, dict):
                trends_list = research.get("trends", [])
                reddit_list = research.get("reddit", [])

                if trends_list and isinstance(trends_list, list) and len(trends_list) > 0:
                    trends_data = json.dumps(trends_list, indent=2)
                if reddit_list and isinstance(reddit_list, list) and len(reddit_list) > 0:
                    reddit_data = json.dumps(reddit_list, indent=2)
        except (TypeError, ValueError) as e:
            logger.debug(f"Could not serialize research data: {e}")

        # Generate ideas with AI
        prompt = self.IDEA_GENERATION_PROMPT.format(
            niche=niche,
            count=count,
            trends_data=trends_data,
            reddit_data=reddit_data
        )

        try:
            response = self.ai.generate(prompt, max_tokens=2000)

            if not response or not isinstance(response, str) or not response.strip():
                logger.warning("Empty response from AI, using fallback ideas")
                return self._get_fallback_ideas(niche, count)

            ideas_data = self._parse_json_response(response)

            # Validate parsed data
            if not ideas_data or not isinstance(ideas_data, list) or len(ideas_data) == 0:
                logger.warning("No valid ideas parsed from AI response, using fallback ideas")
                return self._get_fallback_ideas(niche, count)

            # Convert to ScoredIdea objects with defensive checks
            ideas = []
            for data in ideas_data:
                if not data or not isinstance(data, dict):
                    continue

                try:
                    # Safe score extraction with validation
                    trend = data.get("trend_score", 50)
                    trend = int(trend) if trend is not None else 50
                    trend = max(0, min(100, trend))

                    competition = data.get("competition_score", 50)
                    competition = int(competition) if competition is not None else 50
                    competition = max(0, min(100, competition))

                    engagement = data.get("engagement_score", 50)
                    engagement = int(engagement) if engagement is not None else 50
                    engagement = max(0, min(100, engagement))

                    overall = int((trend + competition + engagement) / 3)

                    # Safe string extraction
                    title = data.get("title", "Untitled")
                    title = str(title)[:100] if title else "Untitled"

                    description = data.get("description", "")
                    description = str(description)[:500] if description else ""

                    keywords = data.get("keywords", [])
                    if not isinstance(keywords, list):
                        keywords = [str(keywords)] if keywords else []
                    keywords = [str(k) for k in keywords[:10] if k]

                    reasoning = data.get("reasoning", "")
                    reasoning = str(reasoning)[:500] if reasoning else ""

                    idea = ScoredIdea(
                        title=title,
                        description=description,
                        keywords=keywords,
                        niche=niche,
                        score=overall,
                        trend_score=trend,
                        competition_score=competition,
                        engagement_score=engagement,
                        source="ai_generated",
                        reasoning=reasoning
                    )
                    ideas.append(idea)
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug(f"Could not parse idea data: {e}")
                    continue

            # If no valid ideas were parsed, use fallback
            if not ideas:
                logger.warning("No valid ideas could be parsed, using fallback ideas")
                return self._get_fallback_ideas(niche, count)

            # Sort by score
            ideas.sort(key=lambda x: x.score, reverse=True)

            logger.success(f"Generated {len(ideas)} ideas")
            return ideas

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, using fallback ideas")
            return self._get_fallback_ideas(niche, count)
        except ValueError as e:
            logger.warning(f"Value error in idea generation: {e}, using fallback ideas")
            return self._get_fallback_ideas(niche, count)
        except Exception as e:
            logger.error(f"Idea generation failed: {e}, using fallback ideas")
            return self._get_fallback_ideas(niche, count)

    def _fix_json(self, json_str: str) -> str:
        """Fix common JSON issues from LLM outputs."""
        import re
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        # Fix missing quotes around keys
        json_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_str)
        # Remove duplicate quotes
        json_str = re.sub(r'""(\w+)""', r'"\1"', json_str)
        return json_str

    def _parse_json_response(self, content: str) -> List[Dict]:
        """Parse JSON array from AI response."""
        # Input validation
        if not content or not isinstance(content, str):
            logger.warning("Empty or invalid content for JSON parsing")
            return []

        content = content.strip()
        if not content:
            logger.warning("Empty content after stripping")
            return []

        # Try direct parse first
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # Wrap single dict in list
                return [result]
            else:
                logger.warning(f"Unexpected JSON type: {type(result)}")
                return []
        except json.JSONDecodeError:
            pass

        # Extract JSON from content
        json_str = content

        # Try to extract JSON from markdown
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if start > 6 and end > start:
                    json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if start > 2 and end > start:
                    json_str = content[start:end].strip()
            else:
                # Find array in text
                start = content.find("[")
                end = content.rfind("]") + 1
                if start != -1 and end > start:
                    json_str = content[start:end]
        except (ValueError, IndexError) as e:
            logger.debug(f"Error extracting JSON from content: {e}")

        # Try parsing with fixes
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
            return []
        except json.JSONDecodeError:
            # Apply fixes and try again
            try:
                fixed = self._fix_json(json_str)
                result = json.loads(fixed)
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict):
                    return [result]
                return []
            except (json.JSONDecodeError, ValueError):
                pass

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
            Best ScoredIdea or None (falls back to generic idea if generation fails)
        """
        try:
            ideas = self.generate_ideas(niche, count=5, subreddits=subreddits)
            # Safe access with validation
            if ideas and isinstance(ideas, list) and len(ideas) > 0:
                return ideas[0]
            logger.warning("No ideas generated, returning fallback")
            fallback_ideas = self._get_fallback_ideas(niche, 1)
            return fallback_ideas[0] if fallback_ideas else None
        except (IndexError, TypeError, AttributeError) as e:
            logger.error(f"Error getting best idea: {e}")
            fallback_ideas = self._get_fallback_ideas(niche, 1)
            return fallback_ideas[0] if fallback_ideas else None

    def expand_idea(self, idea: ScoredIdea) -> Dict[str, Any]:
        """
        Expand an idea with more details for video production.

        Args:
            idea: ScoredIdea to expand

        Returns:
            Dict with expanded details (always returns a valid dict)
        """
        # Input validation
        if not idea:
            logger.warning("No idea provided for expansion")
            return {"error": "No idea provided", "raw_response": ""}

        # Safe attribute access
        title = "Untitled"
        description = ""
        try:
            if hasattr(idea, 'title') and idea.title:
                title = str(idea.title)
            if hasattr(idea, 'description') and idea.description:
                description = str(idea.description)
        except (AttributeError, TypeError):
            pass

        prompt = f"""Expand this video idea into a detailed outline:

Title: {title}
Description: {description}

Provide:
1. Target audience (who is this for?)
2. Video length recommendation (in minutes)
3. Key points to cover (5-7 bullet points)
4. Suggested thumbnail elements
5. Best posting time recommendation
6. Related video ideas for a series

Return as JSON."""

        try:
            response = self.ai.generate(prompt, max_tokens=1000)

            if not response or not isinstance(response, str):
                logger.warning("Empty response from AI for idea expansion")
                return {"error": "Empty AI response", "raw_response": ""}

            parsed = self._parse_json_response(response)

            # Ensure we return a dict
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0] if isinstance(parsed[0], dict) else {"raw_response": response}
            elif isinstance(parsed, dict):
                return parsed
            else:
                return {"raw_response": response}
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Could not parse expansion response: {e}")
            return {"raw_response": response if response else ""}
        except Exception as e:
            logger.error(f"Error expanding idea: {e}")
            return {"error": str(e), "raw_response": ""}


# Example usage
if __name__ == "__main__":
    # Uses AI_PROVIDER from environment, falls back to "ollama"
    generator = IdeaGenerator()

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
