"""
Smart AI Provider Router

Intelligent AI provider routing for cost optimization.
Routes tasks to the most cost-effective provider based on task type,
quality requirements, and budget constraints.

Cost savings: 50-70% vs using premium providers for everything.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ProviderCost:
    """Cost structure for AI providers (2026 pricing)."""

    name: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    rate_limit_rpm: int
    quality_score: float  # 0-1, subjective quality rating
    supports_streaming: bool = True
    max_tokens: int = 4096


# Provider costs as of January 2026
PROVIDER_COSTS_2026: Dict[str, ProviderCost] = {
    "groq": ProviderCost(
        name="Groq (Llama 3)",
        cost_per_1k_input=0.0,      # FREE tier: 30 req/min
        cost_per_1k_output=0.0,
        rate_limit_rpm=30,
        quality_score=0.75,
        max_tokens=8192,
    ),
    "gemini-flash": ProviderCost(
        name="Gemini 1.5 Flash",
        cost_per_1k_input=0.0,      # FREE tier
        cost_per_1k_output=0.0,
        rate_limit_rpm=15,
        quality_score=0.70,
        max_tokens=8192,
    ),
    "claude-haiku": ProviderCost(
        name="Claude 3.5 Haiku",
        cost_per_1k_input=0.25,     # $0.25/$1.25 per 1k tokens
        cost_per_1k_output=1.25,
        rate_limit_rpm=1000,
        quality_score=0.85,
        max_tokens=4096,
    ),
    "gpt-4o-mini": ProviderCost(
        name="GPT-4o Mini",
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        rate_limit_rpm=500,
        quality_score=0.80,
        max_tokens=4096,
    ),
    "claude-sonnet": ProviderCost(
        name="Claude 3.5 Sonnet",
        cost_per_1k_input=3.0,
        cost_per_1k_output=15.0,
        rate_limit_rpm=1000,
        quality_score=0.95,
        max_tokens=8192,
    ),
    "gpt-4o": ProviderCost(
        name="GPT-4o",
        cost_per_1k_input=2.50,
        cost_per_1k_output=10.0,
        rate_limit_rpm=500,
        quality_score=0.93,
        max_tokens=4096,
    ),
    "ollama": ProviderCost(
        name="Ollama (Local)",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        rate_limit_rpm=999,  # No limit
        quality_score=0.70,
        supports_streaming=True,
        max_tokens=4096,
    ),
}


# Task routing rules - prioritized provider lists per task type
TASK_ROUTING: Dict[str, List[str]] = {
    # Free providers first for simple tasks
    "script_generation": ["groq", "claude-haiku", "gpt-4o-mini"],
    "title_generation": ["groq", "gemini-flash", "claude-haiku"],
    "tag_generation": ["groq", "gemini-flash"],
    "description": ["groq", "claude-haiku"],
    "keyword_extraction": ["groq", "gemini-flash"],
    "simple_rewrite": ["groq", "gemini-flash"],

    # Quality-critical tasks use better providers
    "quality_check": ["claude-haiku", "claude-sonnet"],
    "hook_generation": ["claude-haiku", "gpt-4o-mini"],
    "script_revision": ["claude-haiku", "gpt-4o-mini"],

    # Premium tasks for best quality
    "complex_reasoning": ["claude-sonnet", "gpt-4o"],
    "creative_writing": ["claude-sonnet", "claude-haiku"],
    "final_review": ["claude-haiku", "claude-sonnet"],
}


class SmartProviderRouter:
    """
    Intelligent AI provider routing based on task and budget.

    Strategy:
    1. Use FREE providers (Groq, Gemini) for:
       - Tag generation
       - Keyword extraction
       - Simple rewrites
       - Script generation (draft)

    2. Use CHEAP providers (Haiku, GPT-4o Mini) for:
       - Script revision
       - Descriptions
       - Hook generation

    3. Use PREMIUM providers (Sonnet, GPT-4) only for:
       - Final quality checks
       - Complex reasoning tasks
       - When quality is critical

    Cost savings: 50-70% vs using Claude Sonnet for everything
    """

    def __init__(self, daily_budget: Optional[float] = None):
        """
        Initialize smart provider router.

        Args:
            daily_budget: Maximum daily spend in USD (default: from env or $10)
        """
        self.daily_budget = daily_budget or float(
            os.getenv("DAILY_AI_BUDGET", "10.0")
        )
        self.daily_spend = 0.0
        self.spend_reset_time = datetime.now()

        # Usage tracking
        self.usage_history: List[Dict] = []
        self.provider_usage: Dict[str, int] = {p: 0 for p in PROVIDER_COSTS_2026}

        logger.info(f"SmartProviderRouter initialized: daily_budget=${self.daily_budget:.2f}")

    def route_task(
        self,
        task_type: str,
        estimated_tokens: int = 1000,
        min_quality: float = 0.0,
        force_provider: Optional[str] = None
    ) -> str:
        """
        Select optimal provider for task.

        Args:
            task_type: Type of task (see TASK_ROUTING keys)
            estimated_tokens: Estimated total tokens (input + output)
            min_quality: Minimum required quality score (0-1)
            force_provider: Force specific provider (override routing)

        Returns:
            Provider name to use
        """
        # Reset daily spend if new day
        self._check_daily_reset()

        # Force provider override
        if force_provider and force_provider in PROVIDER_COSTS_2026:
            logger.debug(f"Using forced provider: {force_provider}")
            return force_provider

        # Check budget
        if self.daily_spend >= self.daily_budget:
            logger.warning("Daily budget exceeded, using free providers only")
            return self._get_best_free_provider()

        # Get preferred providers for task
        preferred = TASK_ROUTING.get(task_type, ["groq", "claude-haiku"])

        # Calculate cost and quality for each provider
        candidates = []
        for provider_key in preferred:
            provider = PROVIDER_COSTS_2026.get(provider_key)
            if not provider:
                continue

            # Skip if below minimum quality
            if provider.quality_score < min_quality:
                continue

            # Estimate cost (assume 1:3 input:output ratio)
            input_tokens = int(estimated_tokens * 0.25)
            output_tokens = int(estimated_tokens * 0.75)

            input_cost = (input_tokens / 1000) * provider.cost_per_1k_input
            output_cost = (output_tokens / 1000) * provider.cost_per_1k_output
            total_cost = input_cost + output_cost

            # Check if within remaining budget
            remaining_budget = self.daily_budget - self.daily_spend
            if total_cost > remaining_budget and total_cost > 0:
                continue

            # Quality-adjusted cost (prefer higher quality if cost is similar)
            # Lower adjusted cost = better choice
            if total_cost > 0:
                adjusted_cost = total_cost / provider.quality_score
            else:
                # Free providers: use inverse quality as "cost"
                adjusted_cost = (1 - provider.quality_score) * 0.01

            candidates.append({
                "provider": provider_key,
                "cost": total_cost,
                "adjusted_cost": adjusted_cost,
                "quality": provider.quality_score,
            })

        if not candidates:
            # Fallback to free provider
            logger.warning(f"No suitable provider for {task_type}, using fallback")
            return self._get_best_free_provider()

        # Select lowest adjusted cost
        best = min(candidates, key=lambda x: x["adjusted_cost"])

        logger.info(
            f"Routing {task_type} to {best['provider']} "
            f"(est. cost: ${best['cost']:.4f}, quality: {best['quality']:.0%})"
        )

        return best["provider"]

    def track_usage(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str = "unknown"
    ) -> float:
        """
        Track API usage and cost.

        Args:
            provider: Provider name
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            task_type: Type of task performed

        Returns:
            Cost of this request
        """
        provider_info = PROVIDER_COSTS_2026.get(provider)
        if not provider_info:
            logger.warning(f"Unknown provider: {provider}")
            return 0.0

        # Calculate cost
        input_cost = (input_tokens / 1000) * provider_info.cost_per_1k_input
        output_cost = (output_tokens / 1000) * provider_info.cost_per_1k_output
        total_cost = input_cost + output_cost

        # Update tracking
        self.daily_spend += total_cost
        self.provider_usage[provider] = self.provider_usage.get(provider, 0) + 1

        self.usage_history.append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "task_type": task_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost,
        })

        logger.debug(
            f"Usage tracked: {provider} - {input_tokens}+{output_tokens} tokens, "
            f"${total_cost:.4f} (daily: ${self.daily_spend:.4f}/${self.daily_budget:.2f})"
        )

        return total_cost

    def get_cost_estimate(
        self,
        provider: str,
        estimated_tokens: int,
        input_ratio: float = 0.25
    ) -> float:
        """
        Estimate cost for a request.

        Args:
            provider: Provider name
            estimated_tokens: Total estimated tokens
            input_ratio: Ratio of input to total tokens

        Returns:
            Estimated cost in USD
        """
        provider_info = PROVIDER_COSTS_2026.get(provider)
        if not provider_info:
            return 0.0

        input_tokens = int(estimated_tokens * input_ratio)
        output_tokens = estimated_tokens - input_tokens

        input_cost = (input_tokens / 1000) * provider_info.cost_per_1k_input
        output_cost = (output_tokens / 1000) * provider_info.cost_per_1k_output

        return input_cost + output_cost

    def get_daily_stats(self) -> Dict:
        """Get daily usage statistics."""
        return {
            "daily_spend": self.daily_spend,
            "daily_budget": self.daily_budget,
            "remaining_budget": self.daily_budget - self.daily_spend,
            "budget_used_pct": (self.daily_spend / self.daily_budget * 100) if self.daily_budget > 0 else 0,
            "provider_usage": dict(self.provider_usage),
            "total_requests": len(self.usage_history),
            "reset_time": self.spend_reset_time.isoformat(),
        }

    def get_provider_recommendation(
        self,
        task_types: List[str],
        total_videos: int = 1
    ) -> Dict:
        """
        Get cost recommendations for a batch of videos.

        Args:
            task_types: List of task types needed per video
            total_videos: Number of videos to process

        Returns:
            Cost breakdown and recommendations
        """
        # Estimate tokens per task type
        token_estimates = {
            "script_generation": 2000,
            "title_generation": 100,
            "tag_generation": 150,
            "description": 400,
            "hook_generation": 200,
            "quality_check": 500,
        }

        total_cost_optimized = 0.0
        total_cost_premium = 0.0
        breakdown = []

        for task_type in task_types:
            tokens = token_estimates.get(task_type, 500)

            # Optimized routing
            opt_provider = self.route_task(task_type, tokens)
            opt_cost = self.get_cost_estimate(opt_provider, tokens) * total_videos

            # Premium routing (Claude Sonnet)
            prem_cost = self.get_cost_estimate("claude-sonnet", tokens) * total_videos

            total_cost_optimized += opt_cost
            total_cost_premium += prem_cost

            breakdown.append({
                "task": task_type,
                "optimized_provider": opt_provider,
                "optimized_cost": opt_cost,
                "premium_cost": prem_cost,
                "savings": prem_cost - opt_cost,
            })

        savings_pct = (
            (total_cost_premium - total_cost_optimized) / total_cost_premium * 100
            if total_cost_premium > 0 else 0
        )

        return {
            "total_videos": total_videos,
            "optimized_total": total_cost_optimized,
            "premium_total": total_cost_premium,
            "total_savings": total_cost_premium - total_cost_optimized,
            "savings_percentage": savings_pct,
            "breakdown": breakdown,
        }

    def _check_daily_reset(self):
        """Reset daily spend if it's a new day."""
        now = datetime.now()
        if now.date() > self.spend_reset_time.date():
            logger.info(f"Daily reset: ${self.daily_spend:.4f} spent yesterday")
            self.daily_spend = 0.0
            self.spend_reset_time = now
            self.provider_usage = {p: 0 for p in PROVIDER_COSTS_2026}

    def _get_best_free_provider(self) -> str:
        """Get the best available free provider."""
        free_providers = [
            (k, v) for k, v in PROVIDER_COSTS_2026.items()
            if v.cost_per_1k_input == 0 and v.cost_per_1k_output == 0
        ]

        if not free_providers:
            return "groq"  # Default fallback

        # Return highest quality free provider
        best = max(free_providers, key=lambda x: x[1].quality_score)
        return best[0]


# Module-level singleton
_router: Optional[SmartProviderRouter] = None


def get_router(daily_budget: Optional[float] = None) -> SmartProviderRouter:
    """Get or create router singleton."""
    global _router
    if _router is None:
        _router = SmartProviderRouter(daily_budget=daily_budget)
    return _router


def route_task(task_type: str, estimated_tokens: int = 1000) -> str:
    """Convenience function to route a task."""
    return get_router().route_task(task_type, estimated_tokens)


def track_usage(
    provider: str,
    input_tokens: int,
    output_tokens: int,
    task_type: str = "unknown"
) -> float:
    """Convenience function to track usage."""
    return get_router().track_usage(provider, input_tokens, output_tokens, task_type)
