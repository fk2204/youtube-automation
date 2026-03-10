"""
Slavko Bot API - Clean wrapper around existing agents

Simple interface for Slavko bot to control Joe automation.
Uses existing agents without duplication.
"""

from typing import Dict, Any
from loguru import logger


class SlavkoBotAPI:
    """Simple API for Slavko bot."""

    def status(self, channel: str) -> Dict[str, Any]:
        """Get channel status using analytics agent."""
        try:
            from src.agents.analytics_agent import AnalyticsAgent

            agent = AnalyticsAgent()
            result = agent.analyze_channel(channel, period="7d")

            return {
                "status": "ok" if result.success else "error",
                "channel": channel,
                "metrics": result.metrics.__dict__ if hasattr(result, 'metrics') else {},
                "insights": result.insights if hasattr(result, 'insights') else [],
            }
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"status": "error", "message": str(e)}

    def trending(self, niche: str) -> Dict[str, Any]:
        """Get trending ideas using research agent."""
        try:
            from src.agents.research_agent import ResearchAgent

            agent = ResearchAgent()
            result = agent.find_trending_topics(niche)

            return {
                "status": "ok" if result.success else "error",
                "niche": niche,
                "topics": result.topics if hasattr(result, 'topics') else [],
                "recommendations": result.recommendations if hasattr(result, 'recommendations') else [],
            }
        except Exception as e:
            logger.error(f"Trending check failed: {e}")
            return {"status": "error", "message": str(e)}

    def optimize(self, channel: str) -> Dict[str, Any]:
        """Get optimization tips using SEO strategy agent."""
        try:
            from src.agents.seo_strategist import SEOStrategist

            agent = SEOStrategist()
            result = agent.generate_strategy(channel)

            return {
                "status": "ok" if result.success else "error",
                "channel": channel,
                "recommendations": result.recommendations if hasattr(result, 'recommendations') else [],
            }
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {"status": "error", "message": str(e)}

    def create_video(self, channel: str, topic: str = None) -> Dict[str, Any]:
        """Trigger video creation."""
        try:
            from src.automation.runner import run_video_creation

            result = run_video_creation(channel, topic)

            return {
                "status": "ok" if result.get("success") else "error",
                "channel": channel,
                "topic": topic,
                "video_url": result.get("video_url"),
                "message": result.get("message"),
            }
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_help(self) -> str:
        """Show available commands."""
        return """
**Joe Bot Commands:**
- /status <channel> - Get channel performance
- /trending <niche> - Get trending topics
- /optimize <channel> - Get optimization tips
- /create <channel> [topic] - Create a video
- /help - Show this message
        """


def get_bot_api() -> SlavkoBotAPI:
    """Get bot API instance."""
    return SlavkoBotAPI()
