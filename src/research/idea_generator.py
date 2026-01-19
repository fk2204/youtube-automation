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
from datetime import datetime
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


# ============================================================
# VIRAL TOPIC TEMPLATES (From Competitor Analysis - January 2026)
# ============================================================
# Based on analysis of top faceless YouTube channels:
# Finance: The Swedish Investor, Practical Wisdom, Two Cents
# Psychology: Psych2Go (12.7M subs), Brainy Dose (10M views), The Infographics Show (15M subs)
# Storytelling: JCS Criminal Psychology, Truly Criminal, Lazy Masquerade (1.7M subs), Mr. Nightmare (6M subs)

VIRAL_TOPIC_TEMPLATES = {
    # money_blueprints channel (Finance)
    # Key insight: "Documentary-business hybrids" perform exceptionally well
    # Avg video length: 8-15 minutes | Best performing: Business explainers with charts
    "finance": {
        "templates": [
            # Business Model Explainers (The Swedish Investor style - 1M+ subs)
            "How {company} Makes Money (Business Model Explained)",
            "The Genius Strategy Behind {company}'s Success",
            "{company} vs {competitor}: Who Will Win?",
            # Stock Analysis (high engagement, visual evidence)
            "Why {stock} Will {multiplier}x in {year}",
            "{company} Stock: Buy, Sell, or Hold in {year}?",
            "I Analyzed {number} {investment_type} - Here's What I Found",
            # Passive Income (high search volume - Practical Wisdom style)
            "{number} Passive Income Ideas That Actually Work in {year}",
            "{number} Passive Income Streams I Use to Make ${amount}/Month",
            "The ${amount} Side Hustle Nobody Talks About",
            # Loss Aversion (highest CTR trigger)
            "{number} Money Mistakes Costing You ${amount}/Year",
            "Why {percentage}% of Investors Lose Money (And How to Win)",
            "The Hidden Fees Costing You ${amount} Over Your Lifetime",
            # Documentary-Business Hybrids
            "The Truth About {financial_trend}",
            "How Warren Buffett Built His ${amount} Fortune",
            "The Man Who Predicted the {year} {event}",
            # Wealth Building Guides
            "How to Turn ${small_amount} Into ${large_amount} ({timeframe})",
            "How I Made ${amount} in {time_period} With {strategy}",
            "Why {percentage}% of People Never Build Wealth",
            # New trending templates
            "The Real Reason {company} Is Worth ${valuation}",
            "What Billionaires Know That You Don't About {financial_trend}",
            "{number} Investing Rules That Changed My Life",
        ],
        "variables": {
            "company": ["Apple", "Tesla", "Amazon", "Microsoft", "Netflix", "Nvidia", "Meta", "Google", "Costco", "Berkshire Hathaway"],
            "competitor": ["Apple", "Microsoft", "Amazon", "Google", "Tesla", "Meta"],
            "stock": ["NVIDIA", "TSMC", "AMD", "Tesla", "Amazon", "Apple", "Microsoft", "Palantir", "Snowflake"],
            "financial_trend": ["Dividend Investing", "Index Funds", "Real Estate", "Crypto", "AI Stocks", "ETFs", "Bonds", "REITs"],
            "investment_type": ["Stocks", "ETFs", "REITs", "Dividend Stocks", "Growth Stocks", "Blue Chip Stocks"],
            "strategy": ["Dividend Investing", "Index Funds", "Real Estate Crowdfunding", "Side Hustles", "Value Investing"],
            "number": ["3", "5", "7", "10", "12"],
            "multiplier": ["2", "3", "5", "10"],
            "amount": ["1,000", "5,000", "10,000", "50,000", "100,000"],
            "small_amount": ["100", "500", "1,000", "5,000"],
            "large_amount": ["10,000", "50,000", "100,000", "1,000,000"],
            "percentage": ["73", "87", "92", "96", "78", "84"],
            "time_period": ["30 Days", "6 Months", "1 Year", "5 Years"],
            "timeframe": ["3 Years", "5 Years", "10 Years", "20 Years"],
            "year": [str(datetime.now().year), str(datetime.now().year + 1)],
            "event": ["Crash", "Recession", "Bull Run", "Market Correction"],
            "valuation": ["100 Billion", "500 Billion", "1 Trillion", "2 Trillion"],
        },
        "content_types": [
            "Stock analysis with charts/timelines",
            "Passive income explainers",
            "Business model documentaries (documentary-business hybrids)",
            "Budgeting tutorials with specific numbers",
            "Side hustle guides with proof",
            "Investing rule breakdowns (Warren Buffett, Charlie Munger)",
        ],
        # Competitor insights
        "competitor_channels": [
            "The Swedish Investor - Whiteboard animation, 1M+ subs, book summaries",
            "Practical Wisdom - Stock footage + clear narration, 10M views on best videos",
            "Two Cents - Animated explainers, personal finance focus",
        ],
        "best_practices": [
            "Use visual evidence (charts, timelines, citations)",
            "Include specific dollar amounts, not vague terms",
            "Reference credible sources (Buffett, academic studies)",
            "Documentary-business hybrid format performs best",
            "Target 8-15 minute videos for optimal retention",
        ],
    },

    # mind_unlocked channel (Psychology)
    # Key insight: "What if" hooks and dark psychology content drive highest engagement
    # Avg video length: 8-12 minutes | Best performing: Personality type content, manipulation awareness
    "psychology": {
        "templates": [
            # Personality Types (Psych2Go style - 12.7M subs, 18M views on "8 Toxic Things Parents Say")
            "{number} Signs of {personality_type}",
            "{number} Things Only {personality_type} Will Understand",
            "How to Spot a {personality_type} Instantly",
            "{number} Toxic Things {group} Say to {target}",
            # Brain Facts (high curiosity, Brainy Dose style - 10M views on best)
            "Why Your Brain {brain_action}",
            "The Science Behind {behavior}",
            "What Happens to Your Brain When You {action}",
            # Dark Psychology (manipulation awareness - highest engagement)
            "Dark Psychology Tricks {group} Uses Against You",
            "{number} Manipulation Tactics {group} Uses (And How to Defend)",
            "Why {percentage}% of People Fall for This Manipulation",
            "{number} Signs You're Being Manipulated",
            # Cognitive Biases (practical value)
            "The {cognitive_bias} That's Ruining Your Life",
            "{number} Cognitive Biases That Control Your Decisions",
            "How {group} Exploits the {cognitive_bias}",
            # Body Language (evergreen content)
            "{number} Body Language Signs Someone Is {emotion}",
            "How to Read Anyone in {number} Seconds",
            "FBI Techniques to Detect When Someone Is {emotion}",
            # Psychology Tricks (The Infographics Show style - 15M subs)
            "{number} Psychological Tricks That Work on Everyone",
            "The Psychology of {topic}: Why You {action}",
            "What Your {trait} Says About You (Psychology)",
            # New trending templates
            "Why Intelligent People {action}",
            "The Real Reason You {action} (Psychology Explained)",
            "{number} Habits of Highly Manipulative People",
        ],
        "variables": {
            "personality_type": ["a Narcissist", "a Manipulator", "High Intelligence", "an Introvert", "a Psychopath", "an Empath", "Emotional Intelligence", "a Covert Narcissist", "an INFJ", "a Sociopath"],
            "brain_action": ["Hates Change", "Remembers Embarrassing Moments", "Sabotages Your Success", "Fears Rejection", "Creates False Memories", "Craves Validation", "Procrastinates", "Gets Addicted"],
            "group": ["Salespeople", "Politicians", "Advertisers", "Narcissists", "Manipulators", "Social Media Companies", "Toxic Parents", "Gaslighters", "Marketing Teams"],
            "target": ["You", "Their Children", "Their Partners", "Their Employees", "Customers"],
            "behavior": ["Procrastination", "Jealousy", "Attraction", "Fear", "Addiction", "Motivation", "Anxiety", "Depression", "Love Bombing"],
            "cognitive_bias": ["Confirmation Bias", "Sunk Cost Fallacy", "Anchoring Effect", "Halo Effect", "Dunning-Kruger Effect", "Negativity Bias", "Availability Heuristic"],
            "trait": ["Handwriting", "Eye Color", "Sleep Position", "Favorite Color", "Walking Style", "Laugh", "Phone Wallpaper", "Music Taste"],
            "action": ["Remember Embarrassing Moments", "Procrastinate", "Fear Success", "Self-Sabotage", "Compare Yourself to Others", "Overthink", "People Please", "Avoid Conflict"],
            "emotion": ["Lying", "Attracted to You", "Hiding Something", "Nervous", "Confident", "Insecure", "Jealous", "Guilty"],
            "topic": ["Success", "Money", "Attraction", "Influence", "Fear", "Persuasion", "Charisma", "Power"],
            "number": ["3", "5", "7", "10", "8", "12"],
            "percentage": ["73", "87", "92", "96", "85", "78"],
        },
        "content_types": [
            "Personality type content (narcissist, empath, introvert)",
            "Brain facts and cognitive biases",
            "Dark psychology / manipulation awareness",
            "Psychology tricks used in marketing",
            "Behavior science explainers",
            "Body language analysis",
            "Toxic relationship red flags",
        ],
        # Competitor insights
        "competitor_channels": [
            "Psych2Go - Cute animations, 12.7M subs, personality content, merchandise",
            "Brainy Dose - Stock footage, 10M views, 'People Who Like To Be Alone' viral",
            "The Infographics Show - 15M subs, animated explainers, broad psychology topics",
            "The Art of Improvement - 1M subs, actionable advice format",
        ],
        "best_practices": [
            "Use 'What if' and curiosity-driven hooks",
            "Reference specific studies (Stanford, Milgram, Cialdini)",
            "Dark psychology content drives highest engagement",
            "Make viewers feel like they're learning 'forbidden knowledge'",
            "Animation or stock footage with calm voiceover works best",
        ],
    },

    # untold_stories channel (Storytelling)
    # Key insight: Documentary-style with dramatic tension, 30+ min episodes perform well
    # Best performing: Company rise/fall, true crime adjacent, unsolved mysteries
    "storytelling": {
        "templates": [
            # Company Documentaries (high watch time - Truly Criminal style)
            "The Untold Story of {company_person}",
            "How {company} Went From $0 to ${valuation}",
            "The Rise and Fall of {brand}",
            "What Happened to {forgotten_entity}?",
            "{company}'s Biggest Mistake Ever",
            "The Dark Side of {successful_entity}",
            # Person Stories (JCS Criminal Psychology style - interrogation analysis)
            "The {person} Who {achievement}",
            "How {person} Built a ${amount} Empire (Then Lost It All)",
            "Inside the Mind of {famous_person}",
            "What Really Happened to {famous_person}",
            # Business Scandals (high engagement)
            "Why {company} Is Secretly {adjective}",
            "The {year} {event} That Changed Everything",
            "The True Story Behind {famous_thing}",
            "Inside {company}'s Secret {noun}",
            # Mystery/Tension (Lazy Masquerade style - 1.7M subs)
            "The Mystery That Still Haunts {location}",
            "Why Everyone Was Wrong About {subject}",
            "The Case That Shocked the World",
            "Nobody Knows Why This Happened",
            # Crime/Investigation (Mr. Nightmare style - 6M subs)
            "The Criminal Who {achievement}",
            "The Heist That Went Horribly Wrong",
            "How They Caught {famous_criminal}",
            # New trending templates
            "The {time_period} That Destroyed {company}",
            "Why {company} Will Never Recover",
            "The Genius Strategy That Backfired",
            "The Truth They Don't Want You to Know About {subject}",
        ],
        "variables": {
            "company_person": ["Enron", "WeWork", "Theranos", "Bernie Madoff", "Elon Musk", "Steve Jobs", "FTX", "Lehman Brothers", "Elizabeth Holmes", "Sam Bankman-Fried", "Adam Neumann"],
            "company": ["Amazon", "Apple", "Netflix", "Google", "Tesla", "Uber", "Airbnb", "SpaceX", "Meta", "Twitter", "TikTok"],
            "brand": ["Blockbuster", "Kodak", "Nokia", "BlackBerry", "Toys R Us", "Sears", "RadioShack", "Vine", "MySpace", "Yahoo"],
            "forgotten_entity": ["MySpace", "AOL", "Yahoo", "Vine", "Palm", "Compaq", "Circuit City", "Borders", "Tower Records"],
            "person": ["Man", "Woman", "Teenager", "Dropout", "Immigrant", "Founder", "Genius", "Scammer", "Whistleblower"],
            "famous_person": ["Elon Musk", "Jeff Bezos", "Mark Zuckerberg", "Steve Jobs", "Elizabeth Holmes", "Adam Neumann"],
            "famous_criminal": ["Bernie Madoff", "Elizabeth Holmes", "The Zodiac Killer", "D.B. Cooper"],
            "achievement": ["Fooled Wall Street", "Predicted the Crash", "Broke the Internet", "Built a Billion Dollar Company", "Changed an Industry", "Stole Millions", "Escaped Justice"],
            "adjective": ["Terrifying", "Genius", "Evil", "Doomed", "Unstoppable", "Dangerous", "Brilliant"],
            "successful_entity": ["Amazon", "Facebook", "Google", "Tesla", "Apple", "Disney", "Netflix", "Microsoft"],
            "valuation": ["1 Billion", "10 Billion", "100 Billion", "1 Trillion"],
            "amount": ["1 Billion", "10 Billion", "100 Million", "500 Million"],
            "year": ["2008", "2020", "2001", "2019", "2023", "2022", "2024"],
            "time_period": ["24 Hours", "One Week", "30 Days", "One Decision"],
            "event": ["Crash", "Scandal", "Decision", "Mistake", "Discovery", "Betrayal", "Collapse"],
            "famous_thing": ["the iPhone", "Netflix", "Bitcoin", "Amazon Prime", "the Tesla Roadster", "Facebook", "ChatGPT"],
            "noun": ["Strategy", "Failure", "Success", "Scandal", "Project", "Deal", "Plan"],
            "subject": ["WeWork", "Theranos", "FTX", "Enron", "Lehman Brothers", "Crypto"],
            "location": ["Wall Street", "Silicon Valley", "Hollywood", "Washington"],
        },
        "content_types": [
            "Company rise/fall documentaries (30+ min for high watch time)",
            "Historical events with modern lessons",
            "Unsolved mysteries and true crime adjacent",
            "Founder stories with dramatic arcs",
            "Business scandals and fraud exposures",
            "Heist and crime documentaries",
        ],
        # Competitor insights
        "competitor_channels": [
            "JCS Criminal Psychology - Interrogation analysis, 5M+ subs, long-form",
            "Truly Criminal - Mini-documentaries 30+ min, solved and unsolved cases",
            "Lazy Masquerade - 1.7M subs, horror/true crime blend, calm narration",
            "Mr. Nightmare - 6M subs, animation + true crime, spooky visuals",
            "Stories to Remember - 143k subs, 35M views, compelling narratives",
        ],
        "best_practices": [
            "Start in media res - hook with tension immediately",
            "Use sensory details in every scene (visual, sound, physical)",
            "Mini-cliffhangers every 45-60 seconds",
            "Documentary-style production earns 30-50% higher CPM",
            "Longer videos (12-30 min) have better retention in this niche",
            "Historical crime content averages 2.3M views for established channels",
        ],
    }
}

# ============================================================
# RETENTION PATTERNS (From Competitor Analysis)
# ============================================================

RETENTION_BEST_PRACTICES = {
    "hook_timing": {
        "first_5_seconds": "Pattern interrupt or bold claim - critical for 70%+ retention",
        "5_15_seconds": "Context + first open loop planted",
        "30_seconds": "Must deliver first micro-payoff or risk 50% drop-off",
    },
    "engagement_techniques": {
        "open_loops": "Minimum 3 per video - 32% increase in watch time",
        "micro_cliffhangers": "Every 45-60 seconds - 'But here's where it gets interesting...'",
        "direct_address": "Use 'you' at least 3 times per minute",
        "rhetorical_questions": "Every 30-45 seconds - 'Sound familiar?'",
        "specific_numbers": "Always use exact figures, never vague terms",
    },
    "cta_placement": {
        "soft_cta_30_percent": "If you're finding this valuable, hit subscribe...",
        "engagement_cta_50_percent": "Comment below with your experience...",
        "final_cta_95_percent": "Like and subscribe for more...",
        "never_first_30_seconds": "CTAs in first 30 seconds kill retention",
    },
}


class IdeaGenerator:
    """Generate and score video ideas using AI and research data."""

    # ============================================================
    # CHANNEL-NICHE MAPPING (for automatic niche detection)
    # ============================================================
    CHANNEL_NICHE_MAP = {
        "money_blueprints": "finance",
        "mind_unlocked": "psychology",
        "untold_stories": "storytelling",
    }

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
                "title": f"{niche} in {datetime.now().year}: What's Changed",
                "description": f"Discover the latest updates and trends in {niche} for the current year.",
                "keywords": [niche, str(datetime.now().year), "trends", "updates", "new"],
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

    # ============================================================
    # VIRAL TOPIC GENERATION (From Report 5)
    # ============================================================

    def generate_viral_topic(
        self,
        channel_id: Optional[str] = None,
        niche: Optional[str] = None
    ) -> str:
        """
        Generate a viral video topic from proven templates.

        Uses the VIRAL_TOPIC_TEMPLATES to create engaging video titles
        based on Report 5 channel growth tactics.

        Args:
            channel_id: Channel ID (money_blueprints, mind_unlocked, untold_stories)
            niche: Niche name (finance, psychology, storytelling)

        Returns:
            A filled-in viral topic title string
        """
        import random

        # Determine niche from channel_id if provided
        if channel_id and not niche:
            niche = self.CHANNEL_NICHE_MAP.get(channel_id, "finance")

        # Default to finance if no niche specified
        if not niche:
            niche = "finance"

        # Get templates for this niche
        niche_data = VIRAL_TOPIC_TEMPLATES.get(niche, VIRAL_TOPIC_TEMPLATES["finance"])
        templates = niche_data["templates"]
        variables = niche_data["variables"]

        # Select a random template
        template = random.choice(templates)

        # Fill in the template with random variables
        result = template
        for var_name, var_options in variables.items():
            placeholder = "{" + var_name + "}"
            if placeholder in result:
                result = result.replace(placeholder, random.choice(var_options), 1)

        logger.info(f"Generated viral topic for {niche}: {result}")
        return result

    def generate_viral_ideas(
        self,
        channel_id: Optional[str] = None,
        niche: Optional[str] = None,
        count: int = 5
    ) -> List[ScoredIdea]:
        """
        Generate multiple viral video ideas from templates.

        Args:
            channel_id: Channel ID for automatic niche detection
            niche: Niche name (finance, psychology, storytelling)
            count: Number of ideas to generate

        Returns:
            List of ScoredIdea objects with viral topics
        """
        import random

        # Determine niche
        if channel_id and not niche:
            niche = self.CHANNEL_NICHE_MAP.get(channel_id, "finance")
        if not niche:
            niche = "finance"

        ideas = []
        used_templates = set()

        niche_data = VIRAL_TOPIC_TEMPLATES.get(niche, VIRAL_TOPIC_TEMPLATES["finance"])
        templates = niche_data["templates"]
        variables = niche_data["variables"]
        content_types = niche_data.get("content_types", [])

        for _ in range(count):
            # Try to get a unique template
            available_templates = [t for t in templates if t not in used_templates]
            if not available_templates:
                available_templates = templates  # Reset if all used

            template = random.choice(available_templates)
            used_templates.add(template)

            # Fill in the template
            title = template
            keywords = [niche]
            for var_name, var_options in variables.items():
                placeholder = "{" + var_name + "}"
                if placeholder in title:
                    chosen = random.choice(var_options)
                    title = title.replace(placeholder, chosen, 1)
                    # Add variable value as keyword
                    keywords.append(chosen.lower().replace(" ", "_"))

            # Generate description
            content_type = random.choice(content_types) if content_types else "educational content"
            description = f"A {content_type} video exploring: {title}"

            # Score based on template type (viral templates should score high)
            trend_score = random.randint(75, 95)
            competition_score = random.randint(60, 80)
            engagement_score = random.randint(70, 90)
            overall = int((trend_score + competition_score + engagement_score) / 3)

            idea = ScoredIdea(
                title=title,
                description=description,
                keywords=keywords[:10],
                niche=niche,
                score=overall,
                trend_score=trend_score,
                competition_score=competition_score,
                engagement_score=engagement_score,
                source="viral_template",
                reasoning=f"Based on proven viral template format for {niche} content"
            )
            ideas.append(idea)

        # Sort by score
        ideas.sort(key=lambda x: x.score, reverse=True)
        logger.success(f"Generated {len(ideas)} viral ideas for {niche}")
        return ideas

    def get_viral_idea_for_channel(self, channel_id: str) -> ScoredIdea:
        """
        Get a single viral video idea for a specific channel.

        Args:
            channel_id: Channel ID (money_blueprints, mind_unlocked, untold_stories)

        Returns:
            A ScoredIdea optimized for the channel's niche
        """
        ideas = self.generate_viral_ideas(channel_id=channel_id, count=1)
        if ideas:
            return ideas[0]
        # Fallback
        niche = self.CHANNEL_NICHE_MAP.get(channel_id, "finance")
        return self._get_fallback_ideas(niche, 1)[0]

    def get_content_types_for_niche(self, niche: str) -> List[str]:
        """
        Get the recommended content types for a niche.

        Args:
            niche: Niche name (finance, psychology, storytelling)

        Returns:
            List of content type descriptions
        """
        niche_data = VIRAL_TOPIC_TEMPLATES.get(niche, {})
        return niche_data.get("content_types", [
            "Educational tutorials",
            "Explainer videos",
            "Tips and tricks",
            "Analysis content",
        ])


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
