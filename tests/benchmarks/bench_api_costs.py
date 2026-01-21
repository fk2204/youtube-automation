"""
API Token Usage and Cost Benchmarks.

Tracks token consumption and costs across different AI providers,
estimates daily/monthly costs, and compares provider efficiency.

Run with: pytest tests/benchmarks/bench_api_costs.py -v
Or: python tests/benchmarks/bench_api_costs.py

Example:
    from tests.benchmarks.bench_api_costs import APICostBenchmark

    bench = APICostBenchmark()
    bench.benchmark_script_generation()
    bench.benchmark_idea_generation()
    print(bench.generate_cost_report())
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from loguru import logger


# Cost per 1M tokens (input/output) for each provider
PROVIDER_COSTS = {
    "groq": {"input": 0.05, "output": 0.08, "free_tier": True},
    "ollama": {"input": 0.0, "output": 0.0, "free_tier": True},
    "gemini": {"input": 0.075, "output": 0.30, "free_tier": True},
    "claude": {"input": 3.00, "output": 15.00, "free_tier": False},
    "openai": {"input": 2.50, "output": 10.00, "free_tier": False},
    "fish-audio": {"input": 0.0, "output": 0.01, "free_tier": False},
    "edge-tts": {"input": 0.0, "output": 0.0, "free_tier": True},
}

# Typical token usage per operation
TYPICAL_OPERATIONS = {
    "script_generation": {
        "description": "Generate a 10-minute video script",
        "input_tokens": 1500,
        "output_tokens": 3000,
    },
    "idea_generation": {
        "description": "Generate 5 video ideas",
        "input_tokens": 800,
        "output_tokens": 1200,
    },
    "title_optimization": {
        "description": "Optimize video title for SEO",
        "input_tokens": 300,
        "output_tokens": 200,
    },
    "description_generation": {
        "description": "Generate video description",
        "input_tokens": 500,
        "output_tokens": 800,
    },
    "thumbnail_prompt": {
        "description": "Generate thumbnail prompt",
        "input_tokens": 200,
        "output_tokens": 300,
    },
    "content_validation": {
        "description": "Validate content for compliance",
        "input_tokens": 2000,
        "output_tokens": 500,
    },
}


@dataclass
class TokenUsage:
    """Represents token usage for an operation."""
    operation: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost: float
    duration_seconds: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CostEstimate:
    """Cost estimate for a usage scenario."""
    scenario: str
    daily_operations: int
    daily_cost: float
    weekly_cost: float
    monthly_cost: float
    provider: str
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class APICostBenchmark:
    """Benchmark API token usage and costs."""

    def __init__(self):
        """Initialize API cost benchmark."""
        self.usage_records: List[TokenUsage] = []
        self.cost_estimates: List[CostEstimate] = []
        logger.info("APICostBenchmark initialized")

    @staticmethod
    def calculate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for given token usage.

        Args:
            provider: AI provider name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        costs = PROVIDER_COSTS.get(provider, {"input": 0, "output": 0})
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

    def benchmark_script_generation(
        self,
        providers: List[str] = None,
        iterations: int = 3
    ) -> Dict[str, TokenUsage]:
        """
        Benchmark script generation across providers.

        Args:
            providers: List of providers to test. Default: all providers
            iterations: Number of test iterations

        Returns:
            Dictionary mapping provider to TokenUsage
        """
        if providers is None:
            providers = ["groq", "ollama", "gemini", "claude", "openai"]

        logger.info(f"Benchmarking script generation for providers: {providers}")
        results = {}

        operation = TYPICAL_OPERATIONS["script_generation"]

        for provider in providers:
            total_input = 0
            total_output = 0
            total_duration = 0.0

            for _ in range(iterations):
                # Simulate token usage (in real use, this would call the actual API)
                input_tokens = operation["input_tokens"]
                output_tokens = operation["output_tokens"]

                # Add some variance
                import random
                input_tokens = int(input_tokens * random.uniform(0.9, 1.1))
                output_tokens = int(output_tokens * random.uniform(0.9, 1.1))

                # Simulate API call duration
                start = time.time()
                time.sleep(0.01)  # Minimal sleep to simulate
                duration = time.time() - start

                total_input += input_tokens
                total_output += output_tokens
                total_duration += duration

            avg_input = total_input // iterations
            avg_output = total_output // iterations
            avg_duration = total_duration / iterations
            cost = self.calculate_cost(provider, avg_input, avg_output)

            usage = TokenUsage(
                operation="script_generation",
                provider=provider,
                input_tokens=avg_input,
                output_tokens=avg_output,
                cost=cost,
                duration_seconds=avg_duration
            )
            results[provider] = usage
            self.usage_records.append(usage)

            logger.info(
                f"  {provider}: {avg_input}in/{avg_output}out = ${cost:.6f}"
            )

        return results

    def benchmark_idea_generation(
        self,
        providers: List[str] = None,
        iterations: int = 3
    ) -> Dict[str, TokenUsage]:
        """
        Benchmark idea generation across providers.

        Args:
            providers: List of providers to test
            iterations: Number of test iterations

        Returns:
            Dictionary mapping provider to TokenUsage
        """
        if providers is None:
            providers = ["groq", "ollama", "gemini"]

        logger.info(f"Benchmarking idea generation for providers: {providers}")
        results = {}

        operation = TYPICAL_OPERATIONS["idea_generation"]

        for provider in providers:
            import random
            input_tokens = int(operation["input_tokens"] * random.uniform(0.9, 1.1))
            output_tokens = int(operation["output_tokens"] * random.uniform(0.9, 1.1))
            cost = self.calculate_cost(provider, input_tokens, output_tokens)

            usage = TokenUsage(
                operation="idea_generation",
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                duration_seconds=0.01
            )
            results[provider] = usage
            self.usage_records.append(usage)

        return results

    def benchmark_full_video_workflow(self, provider: str = "groq") -> Dict[str, TokenUsage]:
        """
        Benchmark all operations in a full video workflow.

        Args:
            provider: Provider to use for the workflow

        Returns:
            Dictionary mapping operation to TokenUsage
        """
        logger.info(f"Benchmarking full video workflow with {provider}")
        results = {}

        workflow_operations = [
            "idea_generation",
            "script_generation",
            "title_optimization",
            "description_generation",
            "thumbnail_prompt",
            "content_validation",
        ]

        total_input = 0
        total_output = 0
        total_cost = 0.0

        for op_name in workflow_operations:
            operation = TYPICAL_OPERATIONS[op_name]
            import random
            input_tokens = int(operation["input_tokens"] * random.uniform(0.95, 1.05))
            output_tokens = int(operation["output_tokens"] * random.uniform(0.95, 1.05))
            cost = self.calculate_cost(provider, input_tokens, output_tokens)

            usage = TokenUsage(
                operation=op_name,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                duration_seconds=0.01
            )
            results[op_name] = usage
            self.usage_records.append(usage)

            total_input += input_tokens
            total_output += output_tokens
            total_cost += cost

        logger.info(
            f"Full workflow: {total_input}in/{total_output}out = ${total_cost:.6f}"
        )

        return results

    def compare_providers(self, operation: str = "script_generation") -> Dict[str, Any]:
        """
        Compare all providers for a specific operation.

        Args:
            operation: Operation type to compare

        Returns:
            Comparison dictionary with rankings
        """
        operation_data = TYPICAL_OPERATIONS.get(operation)
        if not operation_data:
            logger.error(f"Unknown operation: {operation}")
            return {}

        logger.info(f"Comparing providers for: {operation}")
        comparison = {
            "operation": operation,
            "description": operation_data["description"],
            "providers": {},
            "cheapest": None,
            "most_expensive": None,
            "free_options": [],
        }

        costs = []
        for provider, pricing in PROVIDER_COSTS.items():
            cost = self.calculate_cost(
                provider,
                operation_data["input_tokens"],
                operation_data["output_tokens"]
            )
            comparison["providers"][provider] = {
                "cost": cost,
                "cost_per_1k_operations": cost * 1000,
                "free_tier": pricing.get("free_tier", False),
            }
            costs.append((provider, cost))

            if pricing.get("free_tier"):
                comparison["free_options"].append(provider)

        # Sort by cost
        costs.sort(key=lambda x: x[1])
        comparison["cheapest"] = costs[0][0]
        comparison["most_expensive"] = costs[-1][0]
        comparison["cost_ranking"] = [p for p, _ in costs]

        return comparison

    def estimate_daily_costs(
        self,
        videos_per_day: int = 3,
        provider: str = "groq"
    ) -> CostEstimate:
        """
        Estimate daily costs for a given production level.

        Args:
            videos_per_day: Number of videos to produce daily
            provider: AI provider to use

        Returns:
            CostEstimate with daily/weekly/monthly projections
        """
        logger.info(f"Estimating costs for {videos_per_day} videos/day with {provider}")

        # Operations per video
        ops_per_video = {
            "script_generation": 1,
            "idea_generation": 0.5,  # Ideas can be batched
            "title_optimization": 1,
            "description_generation": 1,
            "thumbnail_prompt": 1,
            "content_validation": 1,
        }

        daily_cost = 0.0
        breakdown = {}

        for op_name, op_count in ops_per_video.items():
            operation = TYPICAL_OPERATIONS[op_name]
            cost_per_op = self.calculate_cost(
                provider,
                operation["input_tokens"],
                operation["output_tokens"]
            )
            op_daily_cost = cost_per_op * op_count * videos_per_day
            daily_cost += op_daily_cost
            breakdown[op_name] = op_daily_cost

        estimate = CostEstimate(
            scenario=f"{videos_per_day} videos/day",
            daily_operations=videos_per_day * len(ops_per_video),
            daily_cost=daily_cost,
            weekly_cost=daily_cost * 7,
            monthly_cost=daily_cost * 30,
            provider=provider,
            breakdown=breakdown
        )

        self.cost_estimates.append(estimate)
        logger.info(f"Daily: ${daily_cost:.4f}, Monthly: ${daily_cost * 30:.2f}")

        return estimate

    def estimate_monthly_budget(
        self,
        budget: float = 10.0,
        provider: str = "groq"
    ) -> Dict[str, Any]:
        """
        Estimate what can be achieved with a monthly budget.

        Args:
            budget: Monthly budget in USD
            provider: AI provider to use

        Returns:
            Dictionary with achievable operations/videos
        """
        logger.info(f"Estimating capacity for ${budget}/month with {provider}")

        # Calculate cost per video
        cost_per_video = 0.0
        for op_name in TYPICAL_OPERATIONS:
            operation = TYPICAL_OPERATIONS[op_name]
            cost_per_video += self.calculate_cost(
                provider,
                operation["input_tokens"],
                operation["output_tokens"]
            )

        videos_per_month = int(budget / cost_per_video) if cost_per_video > 0 else float('inf')
        videos_per_day = videos_per_month // 30

        result = {
            "budget": budget,
            "provider": provider,
            "cost_per_video": cost_per_video,
            "videos_per_month": videos_per_month,
            "videos_per_day": videos_per_day,
            "remaining_budget": budget - (videos_per_month * cost_per_video) if cost_per_video > 0 else budget,
        }

        if provider in ["ollama", "edge-tts"]:
            result["note"] = f"{provider} is free, budget only limited by hardware"
            result["videos_per_month"] = "Unlimited (free)"
            result["videos_per_day"] = "Unlimited (free)"

        logger.info(
            f"With ${budget}/month: {result['videos_per_month']} videos possible"
        )

        return result

    def track_operation(
        self,
        operation: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float = 0.0
    ) -> TokenUsage:
        """
        Track an actual API operation.

        Args:
            operation: Operation name
            provider: Provider used
            input_tokens: Actual input tokens
            output_tokens: Actual output tokens
            duration_seconds: API call duration

        Returns:
            TokenUsage record
        """
        cost = self.calculate_cost(provider, input_tokens, output_tokens)

        usage = TokenUsage(
            operation=operation,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            duration_seconds=duration_seconds
        )

        self.usage_records.append(usage)
        logger.debug(f"Tracked: {operation} on {provider} = ${cost:.6f}")

        return usage

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tracked usage.

        Returns:
            Dictionary with usage statistics
        """
        if not self.usage_records:
            return {"total_cost": 0, "total_operations": 0}

        total_input = sum(u.input_tokens for u in self.usage_records)
        total_output = sum(u.output_tokens for u in self.usage_records)
        total_cost = sum(u.cost for u in self.usage_records)

        # Group by provider
        by_provider = {}
        for u in self.usage_records:
            if u.provider not in by_provider:
                by_provider[u.provider] = {"operations": 0, "cost": 0}
            by_provider[u.provider]["operations"] += 1
            by_provider[u.provider]["cost"] += u.cost

        # Group by operation
        by_operation = {}
        for u in self.usage_records:
            if u.operation not in by_operation:
                by_operation[u.operation] = {"count": 0, "total_cost": 0}
            by_operation[u.operation]["count"] += 1
            by_operation[u.operation]["total_cost"] += u.cost

        return {
            "total_operations": len(self.usage_records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": total_cost,
            "by_provider": by_provider,
            "by_operation": by_operation,
        }

    def generate_cost_report(self) -> str:
        """
        Generate markdown cost report.

        Returns:
            Markdown-formatted report string
        """
        lines = [
            "# API Cost Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Provider Pricing (per 1M tokens)",
            "",
            "| Provider | Input | Output | Free Tier |",
            "|----------|-------|--------|-----------|",
        ]

        for provider, costs in PROVIDER_COSTS.items():
            free = "Yes" if costs.get("free_tier") else "No"
            lines.append(
                f"| {provider} | ${costs['input']:.2f} | ${costs['output']:.2f} | {free} |"
            )

        # Usage summary
        summary = self.get_usage_summary()
        lines.extend([
            "",
            "## Usage Summary",
            "",
            f"- **Total Operations:** {summary['total_operations']}",
            f"- **Total Input Tokens:** {summary['total_input_tokens']:,}",
            f"- **Total Output Tokens:** {summary['total_output_tokens']:,}",
            f"- **Total Cost:** ${summary['total_cost']:.4f}",
            "",
        ])

        # Cost by provider
        if summary.get("by_provider"):
            lines.extend([
                "### Cost by Provider",
                "",
                "| Provider | Operations | Cost |",
                "|----------|------------|------|",
            ])
            for provider, data in summary["by_provider"].items():
                lines.append(f"| {provider} | {data['operations']} | ${data['cost']:.4f} |")

        # Cost estimates
        if self.cost_estimates:
            lines.extend([
                "",
                "## Cost Estimates",
                "",
                "| Scenario | Daily | Weekly | Monthly | Provider |",
                "|----------|-------|--------|---------|----------|",
            ])
            for est in self.cost_estimates:
                lines.append(
                    f"| {est.scenario} | ${est.daily_cost:.4f} | "
                    f"${est.weekly_cost:.4f} | ${est.monthly_cost:.2f} | {est.provider} |"
                )

        # Recommendations
        lines.extend([
            "",
            "## Recommendations",
            "",
            "1. **For development:** Use `ollama` (free, local) or `groq` (generous free tier)",
            "2. **For production:** `groq` offers best cost/quality balance",
            "3. **For quality:** `claude` produces best scripts but costs more",
            "4. **For TTS:** `edge-tts` is free and high quality",
            "",
        ])

        return "\n".join(lines)


def run_api_benchmarks():
    """Run all API cost benchmarks."""
    logger.info("=" * 60)
    logger.info("  RUNNING API COST BENCHMARKS")
    logger.info("=" * 60)

    bench = APICostBenchmark()

    # Run benchmarks
    logger.info("\n--- Script Generation Benchmark ---")
    bench.benchmark_script_generation(["groq", "ollama", "gemini"])

    logger.info("\n--- Idea Generation Benchmark ---")
    bench.benchmark_idea_generation(["groq", "ollama"])

    logger.info("\n--- Full Workflow Benchmark ---")
    bench.benchmark_full_video_workflow("groq")

    logger.info("\n--- Provider Comparison ---")
    comparison = bench.compare_providers("script_generation")
    logger.info(f"Cheapest: {comparison['cheapest']}")
    logger.info(f"Free options: {comparison['free_options']}")

    logger.info("\n--- Cost Estimates ---")
    bench.estimate_daily_costs(3, "groq")
    bench.estimate_daily_costs(5, "groq")
    bench.estimate_monthly_budget(10.0, "groq")

    # Generate report
    report = bench.generate_cost_report()

    # Save report
    output_dir = Path("output/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "api_cost_report.md"

    with open(report_file, "w") as f:
        f.write(report)

    logger.success(f"Cost report saved to: {report_file}")
    print(report)

    logger.info("=" * 60)
    logger.info("  API COST BENCHMARKS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_api_benchmarks()
