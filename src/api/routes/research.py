"""
Topic research endpoints.

Research trending topics using Google Trends, Reddit, and AI scoring.
"""

from fastapi import APIRouter, Depends
from loguru import logger

from src.api.auth import verify_api_key
from src.api.job_manager import get_job_manager, Job
from src.api.models import (
    ResearchTopicsRequest,
    JobResponse,
    TopicIdea,
    ResearchResponse,
)

router = APIRouter(prefix="/api/v1/research", tags=["research"])


async def _handle_research(job: Job) -> dict:
    """Job handler for topic research."""
    params = job.params
    niche = params["niche"]
    count = params.get("count", 5)
    min_score = params.get("min_score", 60)
    mgr = get_job_manager()

    ideas = []

    # Google Trends research
    if params.get("include_trends", True):
        mgr.update_progress(job.id, 20.0)
        try:
            from src.research.idea_generator import IdeaGenerator
            gen = IdeaGenerator(provider="groq")
            raw_ideas = gen.generate_ideas(niche=niche, count=count)
            for idea in raw_ideas:
                score = idea.score if hasattr(idea, "score") else 50
                if score >= min_score:
                    ideas.append({
                        "topic": idea.topic if hasattr(idea, "topic") else str(idea),
                        "title_suggestions": idea.title_suggestions if hasattr(idea, "title_suggestions") else [],
                        "score": score,
                        "source": "trends_ai",
                    })
        except Exception as e:
            logger.warning(f"Trend research failed: {e}")

    # Reddit research
    if params.get("include_reddit", True):
        mgr.update_progress(job.id, 60.0)
        try:
            from src.research.reddit import RedditResearcher
            researcher = RedditResearcher()
            reddit_ideas = researcher.get_video_ideas(
                subreddits=_get_subreddits_for_niche(niche),
                time_filter="week",
                limit=count,
            )
            for idea in reddit_ideas:
                score = idea.popularity_score if hasattr(idea, "popularity_score") else 50
                if score >= min_score:
                    ideas.append({
                        "topic": idea.topic if hasattr(idea, "topic") else str(idea),
                        "title_suggestions": [],
                        "score": score,
                        "source": "reddit",
                    })
        except Exception as e:
            logger.debug(f"Reddit research unavailable: {e}")

    # Sort by score descending
    ideas.sort(key=lambda x: x["score"], reverse=True)
    ideas = ideas[:count]

    mgr.update_progress(job.id, 100.0)
    return {
        "niche": niche,
        "ideas": ideas,
        "total_found": len(ideas),
    }


def _get_subreddits_for_niche(niche: str) -> list:
    """Map niche to relevant subreddits."""
    mapping = {
        "finance": ["personalfinance", "investing", "financialindependence", "money"],
        "psychology": ["psychology", "selfimprovement", "socialskills", "mentalhealth"],
        "storytelling": ["WritingPrompts", "nosleep", "UnresolvedMysteries", "todayilearned"],
        "programming": ["learnprogramming", "webdev", "Python", "javascript"],
        "technology": ["technology", "gadgets", "futurology", "artificial"],
    }
    return mapping.get(niche, ["learnprogramming", "todayilearned", "explainlikeimfive"])


@router.post("/topics", response_model=JobResponse)
async def research_topics(req: ResearchTopicsRequest, _: str = Depends(verify_api_key)):
    """Research trending topics for a niche.

    Uses Google Trends and Reddit to find high-potential topics,
    then scores them with AI. Returns a job_id for polling.
    """
    mgr = get_job_manager()
    job = mgr.submit(
        job_type="research_topics",
        params={
            "niche": req.niche,
            "count": req.count,
            "min_score": req.min_score,
            "include_reddit": req.include_reddit,
            "include_trends": req.include_trends,
        },
    )

    return JobResponse(
        job_id=job.id,
        status=job.status,
        estimated_duration_seconds=30,
        created_at=job.created_at,
    )
