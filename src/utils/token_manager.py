"""
Token Usage and Cost Management System
Tracks API usage, costs, and optimizes provider selection
"""

import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
from typing import Dict, Optional, List, Callable, Any
from pathlib import Path
from loguru import logger
import json
import yaml


class BudgetExceededError(Exception):
    """Raised when the daily budget limit has been exceeded."""

    def __init__(self, message: str, spent: float = 0.0, limit: float = 0.0):
        self.spent = spent
        self.limit = limit
        super().__init__(message)

# Cost per 1M tokens (input/output) for each provider
PROVIDER_COSTS = {
    "groq": {"input": 0.05, "output": 0.08},  # Free tier available
    "ollama": {"input": 0.0, "output": 0.0},   # Free, local
    "gemini": {"input": 0.075, "output": 0.30},
    "claude": {"input": 3.00, "output": 15.00},
    "openai": {"input": 2.50, "output": 10.00},
    "fish-audio": {"input": 0.0, "output": 0.01},  # Very cheap TTS
}


@dataclass
class UsageRecord:
    provider: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime
    operation: str  # e.g., "script_generation", "idea_research"


class TokenTracker:
    """Track token usage across all AI providers"""

    def __init__(self, db_path: str = "data/token_usage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    operation TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_budgets (
                    date TEXT PRIMARY KEY,
                    budget REAL NOT NULL,
                    spent REAL DEFAULT 0
                )
            """)

    def record_usage(self, provider: str, input_tokens: int, output_tokens: int, operation: str = ""):
        """Record token usage and calculate cost"""
        cost = self.calculate_cost(provider, input_tokens, output_tokens)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO token_usage (provider, input_tokens, output_tokens, cost, operation) VALUES (?, ?, ?, ?, ?)",
                (provider, input_tokens, output_tokens, cost, operation)
            )
            # Update daily spent
            today = datetime.now().strftime("%Y-%m-%d")
            conn.execute("""
                INSERT INTO daily_budgets (date, budget, spent)
                VALUES (?, 10.0, ?)
                ON CONFLICT(date) DO UPDATE SET spent = spent + ?
            """, (today, cost, cost))
        logger.info(f"Token usage: {provider} - {input_tokens}in/{output_tokens}out = ${cost:.4f}")
        return cost

    @staticmethod
    def calculate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token usage"""
        costs = PROVIDER_COSTS.get(provider, {"input": 0, "output": 0})
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

    def get_daily_usage(self, date: str = None) -> Dict:
        """Get token usage for a specific date"""
        date = date or datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT SUM(input_tokens), SUM(output_tokens), SUM(cost) FROM token_usage WHERE DATE(timestamp) = ?",
                (date,)
            ).fetchone()
        return {
            "date": date,
            "input_tokens": row[0] or 0,
            "output_tokens": row[1] or 0,
            "cost": row[2] or 0
        }

    def get_weekly_usage(self) -> Dict:
        """Get token usage for the last 7 days"""
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT SUM(input_tokens), SUM(output_tokens), SUM(cost) FROM token_usage WHERE DATE(timestamp) >= ?",
                (week_ago,)
            ).fetchone()
        return {
            "period": "7 days",
            "input_tokens": row[0] or 0,
            "output_tokens": row[1] or 0,
            "cost": row[2] or 0
        }

    def get_monthly_usage(self) -> Dict:
        """Get token usage for the last 30 days"""
        month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT SUM(input_tokens), SUM(output_tokens), SUM(cost) FROM token_usage WHERE DATE(timestamp) >= ?",
                (month_ago,)
            ).fetchone()
        return {
            "period": "30 days",
            "input_tokens": row[0] or 0,
            "output_tokens": row[1] or 0,
            "cost": row[2] or 0
        }

    def get_usage_by_provider(self) -> List[Dict]:
        """Get total usage broken down by provider"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT provider, SUM(input_tokens), SUM(output_tokens), SUM(cost) FROM token_usage GROUP BY provider ORDER BY SUM(cost) DESC"
            ).fetchall()
        return [{"provider": r[0], "input_tokens": r[1], "output_tokens": r[2], "cost": r[3]} for r in rows]

    def get_usage_by_operation(self) -> List[Dict]:
        """Get total usage broken down by operation type"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT operation, SUM(input_tokens), SUM(output_tokens), SUM(cost), COUNT(*) FROM token_usage WHERE operation != '' GROUP BY operation ORDER BY SUM(cost) DESC"
            ).fetchall()
        return [{"operation": r[0], "input_tokens": r[1], "output_tokens": r[2], "cost": r[3], "count": r[4]} for r in rows]

    def get_cost_per_video(self) -> float:
        """Calculate average cost per video produced"""
        with sqlite3.connect(self.db_path) as conn:
            # Count unique video productions (assume each script_generation = 1 video)
            video_count = conn.execute(
                "SELECT COUNT(*) FROM token_usage WHERE operation LIKE '%script%'"
            ).fetchone()[0] or 1
            total_cost = conn.execute(
                "SELECT SUM(cost) FROM token_usage"
            ).fetchone()[0] or 0
        return total_cost / max(video_count, 1)

    def check_budget(self, daily_budget: float = 10.0, warning_threshold: float = 0.8) -> Dict:
        """
        Check if daily budget has been exceeded.

        Args:
            daily_budget: Maximum daily spend limit in dollars
            warning_threshold: Fraction of budget at which to warn (default 0.8 = 80%)

        Returns:
            Dict with budget status including:
            - daily_budget: The budget limit
            - spent_today: Amount spent today
            - remaining: Amount remaining
            - exceeded: True if budget exceeded
            - warning: True if at or above warning threshold
            - usage_percent: Percentage of budget used
        """
        daily = self.get_daily_usage()
        spent = daily["cost"]
        remaining = daily_budget - spent
        usage_percent = (spent / daily_budget * 100) if daily_budget > 0 else 0
        warning = spent >= (daily_budget * warning_threshold)
        exceeded = remaining <= 0

        # Log warnings
        if exceeded:
            logger.error(f"BUDGET EXCEEDED: Spent ${spent:.4f} of ${daily_budget:.2f} daily limit")
        elif warning:
            logger.warning(f"Budget warning: {usage_percent:.1f}% used (${spent:.4f} of ${daily_budget:.2f})")

        return {
            "daily_budget": daily_budget,
            "spent_today": spent,
            "remaining": max(0, remaining),
            "exceeded": exceeded,
            "warning": warning,
            "usage_percent": usage_percent
        }


class CostOptimizer:
    """Automatically select the most cost-effective provider for each task"""

    # Task complexity ratings
    TASK_COMPLEXITY = {
        "idea_generation": "simple",
        "trend_research": "simple",
        "script_outline": "medium",
        "script_full": "complex",
        "script_revision": "medium",
        "title_generation": "simple",
        "description_generation": "simple",
        "tag_generation": "simple",
    }

    # Provider quality ratings (1-10)
    PROVIDER_QUALITY = {
        "claude": 10,
        "openai": 9,
        "gemini": 7,
        "groq": 6,
        "ollama": 5,
    }

    # Free providers (prioritize these when possible)
    FREE_PROVIDERS = ["ollama", "groq"]

    def __init__(self, tracker: TokenTracker, daily_budget: float = 10.0):
        self.tracker = tracker
        self.daily_budget = daily_budget

    def select_provider(self, task_type: str, prefer_quality: bool = False) -> str:
        """Select the best provider for a task based on complexity and budget"""
        complexity = self.TASK_COMPLEXITY.get(task_type, "medium")
        budget_status = self.tracker.check_budget(self.daily_budget)
        budget_remaining = budget_status["remaining"]

        # If budget is low, use free providers
        if budget_remaining < 1.0:
            logger.warning(f"Low budget (${budget_remaining:.2f}), using free provider")
            return "ollama" if self._is_ollama_available() else "groq"

        # If budget warning, be more conservative
        if budget_status["warning"]:
            logger.info("Budget warning - preferring cheaper providers")
            prefer_quality = False

        # Route based on complexity
        if complexity == "simple":
            return "groq"  # Free tier, fast
        elif complexity == "medium":
            return "groq" if not prefer_quality else "gemini"
        else:  # complex
            if prefer_quality and budget_remaining > 2.0:
                return "claude"
            elif budget_remaining > 1.0:
                return "gemini"
            else:
                return "groq"

    def _is_ollama_available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            return r.status_code == 200
        except:
            return False

    def get_recommendations(self) -> List[str]:
        """Get cost optimization recommendations"""
        recommendations = []

        usage_by_op = self.tracker.get_usage_by_operation()
        usage_by_provider = self.tracker.get_usage_by_provider()

        # Check if expensive providers are being used for simple tasks
        for op in usage_by_op:
            if op["operation"] in ["idea_generation", "title_generation", "tag_generation"]:
                if op["cost"] > 0.10:
                    recommendations.append(
                        f"Consider using Groq (free) for {op['operation']} - currently costing ${op['cost']:.2f}"
                    )

        # Check overall provider usage
        for p in usage_by_provider:
            if p["provider"] in ["claude", "openai"] and p["cost"] > 5.0:
                recommendations.append(
                    f"High spend on {p['provider']} (${p['cost']:.2f}) - consider Gemini or Groq for non-critical tasks"
                )

        # Check if Ollama could be used
        if self._is_ollama_available():
            recommendations.append(
                "Ollama is available locally - use it for development/testing to save costs"
            )

        return recommendations


class PromptCache:
    """Cache prompt responses to avoid duplicate API calls"""

    def __init__(self, db_path: str = "data/prompt_cache.db", ttl_hours: int = 24):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    prompt_hash TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    provider TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _hash_prompt(self, prompt: str) -> str:
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    def get(self, prompt: str) -> Optional[str]:
        """Get cached response for a prompt"""
        prompt_hash = self._hash_prompt(prompt)
        cutoff = (datetime.now() - timedelta(hours=self.ttl_hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response FROM cache WHERE prompt_hash = ? AND created_at > ?",
                (prompt_hash, cutoff)
            ).fetchone()

        if row:
            logger.debug(f"Cache hit for prompt hash {prompt_hash[:8]}...")
            return row[0]
        return None

    def set(self, prompt: str, response: str, provider: str = ""):
        """Cache a response for a prompt"""
        prompt_hash = self._hash_prompt(prompt)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (prompt_hash, response, provider) VALUES (?, ?, ?)",
                (prompt_hash, response, provider)
            )
        logger.debug(f"Cached response for prompt hash {prompt_hash[:8]}...")

    def clear_expired(self):
        """Remove expired cache entries"""
        cutoff = (datetime.now() - timedelta(hours=self.ttl_hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
            logger.info(f"Cleared {result.rowcount} expired cache entries")


# Singleton instances
_token_tracker: Optional[TokenTracker] = None
_cost_optimizer: Optional[CostOptimizer] = None
_prompt_cache: Optional[PromptCache] = None


def get_token_manager() -> TokenTracker:
    """Get singleton token tracker instance"""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker


def get_cost_optimizer(daily_budget: float = 10.0) -> CostOptimizer:
    """Get cost optimizer with specified daily budget"""
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer(get_token_manager(), daily_budget)
    return _cost_optimizer


def get_prompt_cache() -> PromptCache:
    """Get singleton prompt cache instance"""
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = PromptCache()
    return _prompt_cache


def print_usage_report():
    """Print a formatted usage report"""
    tracker = get_token_manager()
    daily = tracker.get_daily_usage()
    weekly = tracker.get_weekly_usage()
    monthly = tracker.get_monthly_usage()
    by_provider = tracker.get_usage_by_provider()
    by_operation = tracker.get_usage_by_operation()
    cost_per_video = tracker.get_cost_per_video()
    budget = tracker.check_budget()

    print("\n" + "="*50)
    print("       TOKEN USAGE & COST REPORT")
    print("="*50)

    print(f"\n{'Period':<15} {'Input Tokens':>15} {'Output Tokens':>15} {'Cost':>10}")
    print("-"*55)
    print(f"{'Today':<15} {daily['input_tokens']:>15,} {daily['output_tokens']:>15,} ${daily['cost']:>8.4f}")
    print(f"{'This Week':<15} {weekly['input_tokens']:>15,} {weekly['output_tokens']:>15,} ${weekly['cost']:>8.4f}")
    print(f"{'This Month':<15} {monthly['input_tokens']:>15,} {monthly['output_tokens']:>15,} ${monthly['cost']:>8.4f}")

    print(f"\n{'Provider':<15} {'Input Tokens':>15} {'Output Tokens':>15} {'Cost':>10}")
    print("-"*55)
    for p in by_provider:
        print(f"{p['provider']:<15} {p['input_tokens']:>15,} {p['output_tokens']:>15,} ${p['cost']:>8.4f}")

    if by_operation:
        print(f"\n{'Operation':<25} {'Count':>8} {'Cost':>10}")
        print("-"*45)
        for op in by_operation[:10]:
            print(f"{op['operation']:<25} {op['count']:>8} ${op['cost']:>8.4f}")

    print(f"\n--- Summary ---")
    print(f"Average cost per video: ${cost_per_video:.4f}")
    print(f"Daily budget: ${budget['daily_budget']:.2f} | Spent: ${budget['spent_today']:.4f} | Remaining: ${budget['remaining']:.4f}")

    if budget["warning"]:
        print("\n[WARNING] Daily budget is 80% depleted!")
    if budget["exceeded"]:
        print("\n[ALERT] Daily budget EXCEEDED!")

    # Get recommendations
    optimizer = get_cost_optimizer()
    recommendations = optimizer.get_recommendations()
    if recommendations:
        print(f"\n--- Recommendations ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

    print("\n" + "="*50)


def load_budget_config() -> Dict[str, Any]:
    """
    Load budget configuration from config.yaml.

    Returns:
        Dict with budget settings:
        - daily_limit: Maximum daily spend
        - warning_threshold: Fraction at which to warn (0.0-1.0)
        - enforce: Whether to enforce budget (raise exceptions)
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    defaults = {
        "daily_limit": 10.0,
        "warning_threshold": 0.8,
        "enforce": True
    }

    try:
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                budget_config = config.get("budget", {})
                return {
                    "daily_limit": budget_config.get("daily_limit", defaults["daily_limit"]),
                    "warning_threshold": budget_config.get("warning_threshold", defaults["warning_threshold"]),
                    "enforce": budget_config.get("enforce", defaults["enforce"])
                }
    except Exception as e:
        logger.warning(f"Failed to load budget config: {e}. Using defaults.")

    return defaults


def check_budget_status(daily_budget: float = None, warning_threshold: float = None) -> Dict[str, Any]:
    """
    Standalone function to check budget status.

    Args:
        daily_budget: Override daily budget limit (uses config if None)
        warning_threshold: Override warning threshold (uses config if None)

    Returns:
        Dict with budget status
    """
    config = load_budget_config()
    budget = daily_budget if daily_budget is not None else config["daily_limit"]
    threshold = warning_threshold if warning_threshold is not None else config["warning_threshold"]

    tracker = get_token_manager()
    return tracker.check_budget(budget, threshold)


def enforce_budget(daily_budget: float = None, raise_on_warning: bool = False):
    """
    Decorator to check budget before API calls.

    Checks the daily budget before executing the wrapped function.
    If budget is exceeded, raises BudgetExceededError.
    If at warning threshold, logs a warning (or raises if raise_on_warning=True).

    Args:
        daily_budget: Override daily budget limit (uses config if None)
        raise_on_warning: If True, also raise exception at warning threshold

    Usage:
        @enforce_budget()
        def my_api_call():
            # Makes API call
            pass

        @enforce_budget(daily_budget=5.0)
        def expensive_operation():
            # Custom budget limit
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = load_budget_config()

            # Skip enforcement if disabled in config
            if not config["enforce"]:
                return func(*args, **kwargs)

            budget = daily_budget if daily_budget is not None else config["daily_limit"]
            threshold = config["warning_threshold"]

            status = check_budget_status(budget, threshold)

            if status["exceeded"]:
                raise BudgetExceededError(
                    f"Daily budget ${budget:.2f} exceeded. Spent: ${status['spent_today']:.4f}",
                    spent=status["spent_today"],
                    limit=budget
                )

            if status["warning"]:
                logger.warning(
                    f"Budget warning for {func.__name__}: "
                    f"{status['usage_percent']:.1f}% used (${status['spent_today']:.4f} of ${budget:.2f})"
                )
                if raise_on_warning:
                    raise BudgetExceededError(
                        f"Budget at warning threshold ({status['usage_percent']:.1f}% used)",
                        spent=status["spent_today"],
                        limit=budget
                    )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_budget(min_remaining: float = 0.0):
    """
    Decorator to require a minimum budget remaining before executing.

    Args:
        min_remaining: Minimum dollars that must remain in budget

    Usage:
        @require_budget(min_remaining=2.0)
        def expensive_ai_call():
            # Only runs if at least $2 remaining
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = load_budget_config()

            if not config["enforce"]:
                return func(*args, **kwargs)

            status = check_budget_status()

            if status["remaining"] < min_remaining:
                raise BudgetExceededError(
                    f"Insufficient budget: ${status['remaining']:.4f} remaining, "
                    f"${min_remaining:.2f} required for {func.__name__}",
                    spent=status["spent_today"],
                    limit=config["daily_limit"]
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


class BudgetGuard:
    """
    Context manager for budget-aware operations.

    Usage:
        with BudgetGuard(estimated_cost=0.50) as guard:
            # Perform API call
            result = api_call()

        # Or check status
        with BudgetGuard() as guard:
            if guard.can_afford(0.50):
                result = expensive_api_call()
            else:
                result = cheap_fallback()
    """

    def __init__(self, estimated_cost: float = 0.0, enforce: bool = True):
        self.estimated_cost = estimated_cost
        self.enforce = enforce
        self.status = None
        self.config = load_budget_config()

    def __enter__(self):
        self.status = check_budget_status()

        if self.enforce and self.config["enforce"]:
            # Check if we can afford the estimated cost
            if self.estimated_cost > 0 and self.status["remaining"] < self.estimated_cost:
                raise BudgetExceededError(
                    f"Cannot afford estimated cost ${self.estimated_cost:.4f}. "
                    f"Only ${self.status['remaining']:.4f} remaining.",
                    spent=self.status["spent_today"],
                    limit=self.config["daily_limit"]
                )

            if self.status["exceeded"]:
                raise BudgetExceededError(
                    f"Daily budget exceeded. Spent: ${self.status['spent_today']:.4f}",
                    spent=self.status["spent_today"],
                    limit=self.config["daily_limit"]
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def can_afford(self, cost: float) -> bool:
        """Check if the budget can afford a specific cost."""
        return self.status["remaining"] >= cost

    @property
    def remaining(self) -> float:
        """Get remaining budget."""
        return self.status["remaining"] if self.status else 0.0

    @property
    def spent(self) -> float:
        """Get amount spent today."""
        return self.status["spent_today"] if self.status else 0.0
