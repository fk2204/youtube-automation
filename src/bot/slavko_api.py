"""
Slavko Bot API - Complete wrapper around all Joe features

Provides complete access to all 10 monetization features:
1. Analytics Dashboard
2. Trending Topics Research
3. A/B Testing System
4. Performance Monitoring
5. Multi-channel Analytics
6. Auto-Optimization
7. Competitor Analysis
8. SEO Keywords
9. Schedule Optimization
10. Revenue Projection
"""

from typing import Dict, Any, Optional, List
from loguru import logger


class SlavkoBotAPI:
    """Complete Slavko bot API with all 10 monetization features."""

    # Feature 1 & 5: Analytics Dashboard + Multi-channel Analytics
    def status(self, channel: str) -> Dict[str, Any]:
        """Get channel performance status."""
        try:
            from src.agents.analytics_agent import AnalyticsAgent

            agent = AnalyticsAgent()
            result = agent.analyze_channel(channel, period="7d")

            return {
                "status": "ok" if result.success else "error",
                "channel": channel,
                "videos": result.metrics.video_count,
                "views": result.metrics.total_views,
                "avg_ctr": f"{result.metrics.avg_ctr:.2f}%",
                "avg_retention": f"{result.metrics.avg_retention:.2f}%",
                "insights": result.insights[:3] if result.insights else [],
            }
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"status": "error", "message": str(e)}

    # Feature 2: Trending Topics
    def trending(self, niche: str) -> Dict[str, Any]:
        """Get trending video ideas for niche."""
        try:
            from src.agents.research_agent import ResearchAgent

            agent = ResearchAgent()
            result = agent.find_trending_topics(niche)

            return {
                "status": "ok" if result.success else "error",
                "niche": niche,
                "trending_topics": result.trending_topics if hasattr(result, 'trending_topics') else [],
                "keywords": result.keywords if hasattr(result, 'keywords') else [],
            }
        except Exception as e:
            logger.error(f"Trending failed: {e}")
            return {"status": "error", "message": str(e)}

    # Feature 3: A/B Testing
    def ab_test(self, video_id: str, variant_a: str, variant_b: str, metric: str = "ctr") -> Dict[str, Any]:
        """Start A/B test for video."""
        try:
            from src.analytics.ab_testing import ABTestingEngine

            db = None  # Would use real DB in production
            engine = ABTestingEngine(db)
            test = engine.start_test(video_id, "title", variant_a, variant_b, metric)

            return {
                "status": "ok",
                "test_id": video_id,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "metric": metric,
                "message": "A/B test started. Results will appear after 100+ views.",
            }
        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            return {"status": "error", "message": str(e)}

    # Feature 4: Performance Monitoring
    def alerts(self, channel: str) -> Dict[str, Any]:
        """Get performance alerts for channel."""
        try:
            from src.monitoring.performance_monitor import PerformanceMonitor

            monitor = PerformanceMonitor()
            summary = monitor.get_alert_summary(hours=24)

            return {
                "status": "ok",
                "channel": channel,
                "critical": summary.get("critical", 0),
                "warning": summary.get("warning", 0),
                "info": summary.get("info", 0),
                "recent_alerts": summary.get("recent_critical", [])[:3],
            }
        except Exception as e:
            logger.error(f"Alert check failed: {e}")
            return {"status": "error", "message": str(e)}

    # Feature 6: Auto-Optimization
    def optimize(self, channel: str) -> Dict[str, Any]:
        """Get auto-optimization recommendations."""
        try:
            from src.agents.retention_optimizer_agent import RetentionOptimizerAgent

            agent = RetentionOptimizerAgent()
            result = agent.analyze_and_recommend(channel)

            return {
                "status": "ok" if result.success else "error",
                "channel": channel,
                "recommendations": result.recommendations if hasattr(result, 'recommendations') else [],
            }
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {"status": "error", "message": str(e)}

    # Feature 7: Competitor Analysis
    def competitors(self, topic: str) -> Dict[str, Any]:
        """Analyze top competitor videos for topic."""
        try:
            from src.agents.research_agent import ResearchAgent

            agent = ResearchAgent()
            result = agent.analyze_competitors(topic)

            return {
                "status": "ok" if result.success else "error",
                "topic": topic,
                "competitors": result.competitors if hasattr(result, 'competitors') else [],
                "best_practices": result.best_practices if hasattr(result, 'best_practices') else [],
            }
        except Exception as e:
            logger.error(f"Competitor analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    # Feature 8: SEO Keywords
    def keywords(self, topic: str) -> Dict[str, Any]:
        """Get SEO keywords for topic."""
        try:
            from src.agents.seo_strategist import SEOStrategist

            agent = SEOStrategist()
            result = agent.research_keywords(topic)

            return {
                "status": "ok" if result.success else "error",
                "topic": topic,
                "primary_keywords": result.keywords if hasattr(result, 'keywords') else [],
                "long_tail": result.long_tail if hasattr(result, 'long_tail') else [],
            }
        except Exception as e:
            logger.error(f"Keyword research failed: {e}")
            return {"status": "error", "message": str(e)}

    # Feature 9: Schedule Optimization
    def best_time(self, channel: str) -> Dict[str, Any]:
        """Get optimal posting times for channel."""
        try:
            from src.agents.scheduler_agent import SchedulerAgent

            agent = SchedulerAgent()
            result = agent.get_optimal_schedule(channel)

            return {
                "status": "ok" if result.success else "error",
                "channel": channel,
                "best_days": result.best_days if hasattr(result, 'best_days') else [],
                "best_times": result.best_times if hasattr(result, 'best_times') else [],
            }
        except Exception as e:
            logger.error(f"Schedule optimization failed: {e}")
            return {"status": "error", "message": str(e)}

    # Feature 10: Revenue Projection
    def revenue(self, channel: str = None) -> Dict[str, Any]:
        """Get revenue projections and analytics."""
        try:
            from src.agents.revenue_agent import RevenueAgent

            agent = RevenueAgent()
            result = agent.analyze_revenue(channel or "all")

            return {
                "status": "ok" if result.success else "error",
                "channel": channel or "all",
                "projected_revenue": result.projected_revenue if hasattr(result, 'projected_revenue') else 0,
                "cpm": result.cpm if hasattr(result, 'cpm') else 0,
                "monthly_estimate": result.monthly_estimate if hasattr(result, 'monthly_estimate') else 0,
            }
        except Exception as e:
            logger.error(f"Revenue analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    # Helper: Create video
    def create_video(self, channel: str, topic: str = None) -> Dict[str, Any]:
        """Trigger video creation."""
        try:
            from src.automation.runner import task_full_pipeline

            result = task_full_pipeline(channel, topic)

            return {
                "status": "ok" if result.get("success") else "error",
                "channel": channel,
                "topic": topic,
                "video_file": result.get("output_file"),
            }
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_help(self) -> str:
        """Show all available commands."""
        return """
**Joe Bot - 10 Monetization Features:**

1. /status <channel> - Analytics Dashboard
2. /trending <niche> - Trending Topics Research
3. /ab-test <vid> <A> <B> - A/B Testing System
4. /alerts <channel> - Performance Monitoring
5. /optimize <channel> - Auto-Optimization
6. /competitors <topic> - Competitor Analysis
7. /keywords <topic> - SEO Keywords
8. /best-time <channel> - Schedule Optimization
9. /revenue [channel] - Revenue Projection
10. /create <channel> [topic] - Create Video

Niches: finance, psychology, ai, self-improvement
        """


def get_bot_api() -> SlavkoBotAPI:
    """Get bot API instance."""
    return SlavkoBotAPI()
