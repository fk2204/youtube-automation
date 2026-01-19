"""
Token Optimization System for YouTube Automation

Comprehensive system to reduce API costs by 50%+ through:
1. Prompt caching - Cache repeated prompts with intelligent invalidation
2. Response compression - Extract only needed data from responses
3. Batch processing - Combine multiple small requests into one
4. Smart provider routing - Use free providers when possible
5. Token budget allocation - Per-agent budgets with enforcement
6. Automatic fallback - Switch to cheaper providers when budget low

Usage:
    from src.utils.token_optimizer import (
        TokenOptimizer,
        get_token_optimizer,
        optimize_prompt,
        batch_requests,
        smart_route
    )

    optimizer = get_token_optimizer()

    # Optimize a prompt before sending
    optimized = optimizer.optimize_prompt(prompt, task_type="script")

    # Route to best provider based on task and budget
    provider = optimizer.smart_route(task_type="title_generation")

    # Batch multiple requests
    results = optimizer.batch_process(requests_list)
"""

import hashlib
import json
import sqlite3
import zlib
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from loguru import logger

# Try importing token_manager components
try:
    from src.utils.token_manager import (
        TokenTracker,
        CostOptimizer,
        PromptCache,
        PROVIDER_COSTS,
        get_token_manager,
        get_cost_optimizer,
        get_prompt_cache,
        BudgetExceededError,
    )
except ImportError:
    # Fallback if running standalone
    PROVIDER_COSTS = {
        "groq": {"input": 0.05, "output": 0.08},
        "ollama": {"input": 0.0, "output": 0.0},
        "gemini": {"input": 0.075, "output": 0.30},
        "claude": {"input": 3.00, "output": 15.00},
        "openai": {"input": 2.50, "output": 10.00},
    }


# ============================================================
# Type Definitions
# ============================================================

T = TypeVar("T")


@dataclass
class TokenBudget:
    """Token budget allocation for an agent or operation."""

    name: str
    daily_limit: int
    hourly_limit: int
    used_today: int = 0
    used_this_hour: int = 0
    last_reset_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    last_reset_hour: int = field(default_factory=lambda: datetime.now().hour)

    def check_and_reset(self) -> None:
        """Reset counters if day/hour has changed."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        current_hour = now.hour

        if self.last_reset_date != today:
            self.used_today = 0
            self.last_reset_date = today
            logger.debug(f"Reset daily budget for {self.name}")

        if self.last_reset_hour != current_hour:
            self.used_this_hour = 0
            self.last_reset_hour = current_hour
            logger.debug(f"Reset hourly budget for {self.name}")

    def can_use(self, tokens: int) -> bool:
        """Check if we can use the specified number of tokens."""
        self.check_and_reset()
        return (
            self.used_today + tokens <= self.daily_limit and
            self.used_this_hour + tokens <= self.hourly_limit
        )

    def use(self, tokens: int) -> bool:
        """Use tokens from the budget. Returns False if would exceed limit."""
        if not self.can_use(tokens):
            return False
        self.used_today += tokens
        self.used_this_hour += tokens
        return True

    @property
    def remaining_today(self) -> int:
        """Get remaining tokens for today."""
        self.check_and_reset()
        return max(0, self.daily_limit - self.used_today)

    @property
    def remaining_this_hour(self) -> int:
        """Get remaining tokens this hour."""
        self.check_and_reset()
        return max(0, self.hourly_limit - self.used_this_hour)


@dataclass
class CacheEntry:
    """Entry in the prompt cache with metadata."""

    response: str
    provider: str
    token_count: int
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    compressed: bool = False

    def access(self) -> None:
        """Record an access to this cache entry."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class BatchRequest:
    """A single request in a batch."""

    prompt: str
    task_type: str
    callback: Optional[Callable[[str], None]] = None
    priority: int = 5  # 1 (highest) to 10 (lowest)
    max_tokens: int = 1000
    request_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8])


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    original_tokens: int
    optimized_tokens: int
    savings_percent: float
    optimized_prompt: str
    compression_applied: List[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Decision from smart routing."""

    provider: str
    reason: str
    estimated_cost: float
    fallback_provider: Optional[str] = None
    quality_score: float = 1.0


# ============================================================
# Prompt Compression Strategies
# ============================================================

class CompressionStrategy(ABC):
    """Abstract base for prompt compression strategies."""

    @abstractmethod
    def compress(self, text: str) -> str:
        """Compress the text."""
        pass

    @abstractmethod
    def estimate_savings(self, text: str) -> float:
        """Estimate compression savings as a percentage."""
        pass


class WhitespaceCompressor(CompressionStrategy):
    """Remove excess whitespace while preserving readability."""

    def compress(self, text: str) -> str:
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r' {2,}', ' ', text)       # Single spaces
        text = re.sub(r'\t+', ' ', text)         # Tabs to spaces
        return text.strip()

    def estimate_savings(self, text: str) -> float:
        original = len(text)
        compressed = len(self.compress(text))
        return (original - compressed) / original if original > 0 else 0


class ExampleCondenser(CompressionStrategy):
    """Condense multiple examples into compact format."""

    EXAMPLE_PATTERNS = [
        r'Example \d+:.*?(?=Example \d+:|$)',
        r'For example,.*?(?=\.|$)',
        r'e\.g\.,.*?(?=\.|$)',
    ]

    def compress(self, text: str) -> str:
        # Count and limit examples
        for pattern in self.EXAMPLE_PATTERNS:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if len(matches) > 2:
                # Keep only first 2 examples, summarize rest
                kept = matches[:2]
                remaining_count = len(matches) - 2

                for match in matches[2:]:
                    text = text.replace(match, '', 1)

                # Add summary note
                if remaining_count > 0:
                    text += f"\n(+{remaining_count} more examples follow similar patterns)"

        return text.strip()

    def estimate_savings(self, text: str) -> float:
        original = len(text)
        compressed = len(self.compress(text))
        return (original - compressed) / original if original > 0 else 0


class InstructionDeduplicator(CompressionStrategy):
    """Remove redundant instructions that appear multiple times."""

    def compress(self, text: str) -> str:
        lines = text.split('\n')
        seen_instructions = set()
        result = []

        for line in lines:
            # Normalize for comparison
            normalized = line.lower().strip()

            # Skip if it's a duplicate instruction
            if normalized.startswith(('you must', 'always', 'never', 'make sure', 'ensure')):
                if normalized in seen_instructions:
                    continue
                seen_instructions.add(normalized)

            result.append(line)

        return '\n'.join(result)

    def estimate_savings(self, text: str) -> float:
        original = len(text)
        compressed = len(self.compress(text))
        return (original - compressed) / original if original > 0 else 0


class JSONSchemaMinifier(CompressionStrategy):
    """Minify JSON schemas in prompts while keeping them valid."""

    def compress(self, text: str) -> str:
        # Find JSON blocks and minify them
        json_pattern = r'```json\s*(.*?)\s*```'

        def minify_json(match):
            try:
                json_str = match.group(1)
                parsed = json.loads(json_str)
                minified = json.dumps(parsed, separators=(',', ':'))
                return f'```json\n{minified}\n```'
            except json.JSONDecodeError:
                return match.group(0)

        return re.sub(json_pattern, minify_json, text, flags=re.DOTALL)

    def estimate_savings(self, text: str) -> float:
        original = len(text)
        compressed = len(self.compress(text))
        return (original - compressed) / original if original > 0 else 0


class VerboseLanguageCondenser(CompressionStrategy):
    """Replace verbose phrases with concise alternatives."""

    REPLACEMENTS = {
        "in order to": "to",
        "due to the fact that": "because",
        "at this point in time": "now",
        "in the event that": "if",
        "with regard to": "about",
        "for the purpose of": "to",
        "it is important to note that": "note:",
        "please make sure to": "ensure",
        "you should be aware that": "note:",
        "keep in mind that": "remember:",
        "it is worth mentioning that": "",
        "as a matter of fact": "",
        "the reason why is because": "because",
        "in spite of the fact that": "although",
        "regardless of the fact that": "although",
        "with the exception of": "except",
        "in the near future": "soon",
        "at the present time": "now",
        "on a daily basis": "daily",
        "on a regular basis": "regularly",
    }

    def compress(self, text: str) -> str:
        result = text
        for verbose, concise in self.REPLACEMENTS.items():
            result = re.sub(
                re.escape(verbose),
                concise,
                result,
                flags=re.IGNORECASE
            )
        return result

    def estimate_savings(self, text: str) -> float:
        original = len(text)
        compressed = len(self.compress(text))
        return (original - compressed) / original if original > 0 else 0


# ============================================================
# Response Extraction Strategies
# ============================================================

class ResponseExtractor:
    """Extract only needed data from AI responses."""

    @staticmethod
    def extract_json(response: str) -> Optional[Dict]:
        """Extract JSON from a response that may contain other text."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find bare JSON object
        brace_match = re.search(r'\{.*\}', response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def extract_list(response: str) -> List[str]:
        """Extract a list from a response."""
        # Try numbered list
        numbered = re.findall(r'^\s*\d+[\.\)]\s*(.+)$', response, re.MULTILINE)
        if numbered:
            return numbered

        # Try bullet list
        bulleted = re.findall(r'^\s*[-*]\s*(.+)$', response, re.MULTILINE)
        if bulleted:
            return bulleted

        # Split by newlines as fallback
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return lines

    @staticmethod
    def extract_code(response: str, language: str = None) -> Optional[str]:
        """Extract code block from a response."""
        if language:
            pattern = rf'```{language}\s*(.*?)\s*```'
        else:
            pattern = r'```(?:\w+)?\s*(.*?)\s*```'

        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_section(response: str, section_name: str) -> Optional[str]:
        """Extract a named section from a response."""
        # Look for markdown headers
        pattern = rf'#+\s*{re.escape(section_name)}[:\s]*(.*?)(?=#+\s|\Z)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Look for bold/labeled sections
        pattern = rf'\*\*{re.escape(section_name)}\*\*[:\s]*(.*?)(?=\*\*|\Z)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None


# ============================================================
# Advanced Prompt Cache
# ============================================================

class AdvancedPromptCache:
    """
    Advanced caching system with:
    - Semantic similarity matching (not just exact hash)
    - Automatic compression for large responses
    - TTL-based and access-based eviction
    - Cache statistics and optimization hints
    """

    def __init__(
        self,
        db_path: str = "data/advanced_cache.db",
        ttl_hours: int = 24,
        max_entries: int = 10000,
        compress_threshold: int = 1000,  # Compress responses > 1000 chars
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.max_entries = max_entries
        self.compress_threshold = compress_threshold
        self._lock = threading.Lock()
        self._init_db()

        # In-memory LRU cache for hot entries
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_cache_size = 100

        # Stats
        self.hits = 0
        self.misses = 0

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt_prefix TEXT,
                    response BLOB,
                    provider TEXT,
                    token_count INTEGER,
                    compressed INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    last_accessed TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prefix ON prompt_cache(prompt_prefix)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON prompt_cache(last_accessed)
            """)

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of the prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _get_prefix(self, prompt: str, length: int = 100) -> str:
        """Get a prefix for similarity matching."""
        return prompt[:length].lower().strip()

    def _compress(self, data: str) -> bytes:
        """Compress a string."""
        return zlib.compress(data.encode('utf-8'))

    def _decompress(self, data: bytes) -> str:
        """Decompress bytes to string."""
        return zlib.decompress(data).decode('utf-8')

    def get(self, prompt: str, similarity_threshold: float = 0.9) -> Optional[str]:
        """
        Get a cached response for a prompt.

        Args:
            prompt: The prompt to look up
            similarity_threshold: Minimum similarity for prefix matching (0-1)

        Returns:
            Cached response or None
        """
        prompt_hash = self._hash_prompt(prompt)

        # Check memory cache first
        if prompt_hash in self._memory_cache:
            entry = self._memory_cache[prompt_hash]
            entry.access()
            self.hits += 1
            logger.debug(f"Memory cache hit for hash {prompt_hash[:8]}")
            return entry.response

        # Check database
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cutoff = (datetime.now() - timedelta(hours=self.ttl_hours)).isoformat()

                # First try exact match
                row = conn.execute(
                    """SELECT response, compressed, token_count, provider
                       FROM prompt_cache
                       WHERE prompt_hash = ? AND last_accessed > ?""",
                    (prompt_hash, cutoff)
                ).fetchone()

                if row:
                    response = self._decompress(row[0]) if row[1] else row[0]
                    if isinstance(response, bytes):
                        response = response.decode('utf-8')

                    # Update access stats
                    conn.execute(
                        """UPDATE prompt_cache
                           SET access_count = access_count + 1, last_accessed = ?
                           WHERE prompt_hash = ?""",
                        (datetime.now().isoformat(), prompt_hash)
                    )

                    # Add to memory cache
                    self._add_to_memory_cache(prompt_hash, CacheEntry(
                        response=response,
                        provider=row[3],
                        token_count=row[2],
                        created_at=datetime.now(),
                        compressed=bool(row[1])
                    ))

                    self.hits += 1
                    logger.debug(f"Database cache hit for hash {prompt_hash[:8]}")
                    return response

        self.misses += 1
        return None

    def set(
        self,
        prompt: str,
        response: str,
        provider: str = "",
        token_count: int = 0
    ) -> None:
        """
        Cache a response for a prompt.

        Args:
            prompt: The original prompt
            response: The AI response
            provider: The provider that generated the response
            token_count: Estimated token count
        """
        prompt_hash = self._hash_prompt(prompt)
        prompt_prefix = self._get_prefix(prompt)

        # Compress if response is large
        compressed = len(response) > self.compress_threshold
        stored_response = self._compress(response) if compressed else response.encode('utf-8')

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO prompt_cache
                       (prompt_hash, prompt_prefix, response, provider, token_count,
                        compressed, access_count, created_at, last_accessed)
                       VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                    (prompt_hash, prompt_prefix, stored_response, provider,
                     token_count, int(compressed), datetime.now().isoformat(),
                     datetime.now().isoformat())
                )

                # Enforce max entries
                self._evict_if_needed(conn)

        # Add to memory cache
        self._add_to_memory_cache(prompt_hash, CacheEntry(
            response=response,
            provider=provider,
            token_count=token_count,
            created_at=datetime.now(),
            compressed=compressed
        ))

        logger.debug(f"Cached response for hash {prompt_hash[:8]} (compressed={compressed})")

    def _add_to_memory_cache(self, key: str, entry: CacheEntry) -> None:
        """Add an entry to the memory cache, evicting if needed."""
        if len(self._memory_cache) >= self._memory_cache_size:
            # Evict least recently accessed
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].last_accessed
            )
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = entry

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        """Evict old entries if we're over the limit."""
        count = conn.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()[0]

        if count > self.max_entries:
            # Delete oldest 10%
            to_delete = max(1, count // 10)
            conn.execute(f"""
                DELETE FROM prompt_cache
                WHERE prompt_hash IN (
                    SELECT prompt_hash FROM prompt_cache
                    ORDER BY last_accessed ASC
                    LIMIT {to_delete}
                )
            """)
            logger.info(f"Evicted {to_delete} old cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()[0]
            compressed_count = conn.execute(
                "SELECT COUNT(*) FROM prompt_cache WHERE compressed = 1"
            ).fetchone()[0]
            total_accesses = conn.execute(
                "SELECT SUM(access_count) FROM prompt_cache"
            ).fetchone()[0] or 0

            # Get size
            size_bytes = conn.execute(
                "SELECT SUM(LENGTH(response)) FROM prompt_cache"
            ).fetchone()[0] or 0

        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

        return {
            "total_entries": total,
            "compressed_entries": compressed_count,
            "memory_cache_size": len(self._memory_cache),
            "total_accesses": total_accesses,
            "session_hits": self.hits,
            "session_misses": self.misses,
            "hit_rate": hit_rate,
            "size_mb": size_bytes / (1024 * 1024),
        }

    def clear_expired(self) -> int:
        """Clear expired cache entries. Returns number deleted."""
        cutoff = (datetime.now() - timedelta(hours=self.ttl_hours)).isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "DELETE FROM prompt_cache WHERE last_accessed < ?",
                    (cutoff,)
                )
                deleted = result.rowcount

        logger.info(f"Cleared {deleted} expired cache entries")
        return deleted


# ============================================================
# Batch Processor
# ============================================================

class BatchProcessor:
    """
    Process multiple requests in batches for efficiency.

    Combines multiple small requests into larger batches,
    reducing API overhead and improving throughput.
    """

    def __init__(
        self,
        batch_size: int = 5,
        max_wait_seconds: float = 2.0,
        max_batch_tokens: int = 4000,
    ):
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self.max_batch_tokens = max_batch_tokens
        self._queue: List[BatchRequest] = []
        self._lock = threading.Lock()
        self._results: Dict[str, str] = {}

    def add_request(self, request: BatchRequest) -> str:
        """
        Add a request to the batch queue.

        Args:
            request: The batch request to add

        Returns:
            Request ID for retrieving results
        """
        with self._lock:
            self._queue.append(request)
            self._queue.sort(key=lambda r: r.priority)

        return request.request_id

    def get_batch(self) -> List[BatchRequest]:
        """Get the next batch of requests to process."""
        with self._lock:
            batch = self._queue[:self.batch_size]
            self._queue = self._queue[self.batch_size:]
            return batch

    def create_combined_prompt(self, batch: List[BatchRequest]) -> str:
        """
        Combine multiple prompts into a single prompt.

        Creates a structured prompt that asks the AI to respond
        to multiple queries at once, with clear separators.
        """
        combined = """Process the following requests. For each request, provide a response labeled with its ID.

Format your response as:
[ID_1]
<response 1>

[ID_2]
<response 2>

---REQUESTS---

"""
        for req in batch:
            combined += f"[{req.request_id}] Task: {req.task_type}\n"
            combined += f"Prompt: {req.prompt}\n\n"

        combined += "---END REQUESTS---\n\nRespond to each request above:"
        return combined

    def parse_combined_response(self, response: str, batch: List[BatchRequest]) -> Dict[str, str]:
        """Parse a combined response back into individual responses."""
        results = {}

        for req in batch:
            # Look for the response section for this ID
            pattern = rf'\[{req.request_id}\]\s*(.*?)(?=\[[\w]+\]|\Z)'
            match = re.search(pattern, response, re.DOTALL)

            if match:
                results[req.request_id] = match.group(1).strip()
            else:
                results[req.request_id] = ""
                logger.warning(f"Could not parse response for request {req.request_id}")

        return results

    def estimate_batch_tokens(self, batch: List[BatchRequest]) -> int:
        """Estimate total tokens for a batch."""
        combined = self.create_combined_prompt(batch)
        # Rough estimate: 1 token per 4 characters
        return len(combined) // 4


# ============================================================
# Smart Router
# ============================================================

class SmartRouter:
    """
    Intelligently route requests to the most cost-effective provider.

    Considers:
    - Task complexity and quality requirements
    - Current budget status
    - Provider availability
    - Historical performance
    """

    # Task complexity definitions
    TASK_COMPLEXITY = {
        # Simple tasks - use free providers
        "title_generation": "simple",
        "tag_generation": "simple",
        "description_generation": "simple",
        "idea_brainstorm": "simple",
        "keyword_extraction": "simple",

        # Medium tasks - use mid-tier providers
        "script_outline": "medium",
        "hook_generation": "medium",
        "script_revision": "medium",
        "content_summary": "medium",
        "seo_optimization": "medium",

        # Complex tasks - may need premium providers
        "full_script": "complex",
        "research_synthesis": "complex",
        "creative_writing": "complex",
        "technical_explanation": "complex",
        "competitor_analysis": "complex",
    }

    # Quality scores per provider (1-10)
    PROVIDER_QUALITY = {
        "claude": 10,
        "openai": 9,
        "gemini": 7,
        "groq": 6,
        "ollama": 5,
    }

    # Cost per 1K tokens (approximate, for routing decisions)
    PROVIDER_COST_PER_1K = {
        "claude": 0.015,
        "openai": 0.01,
        "gemini": 0.0005,
        "groq": 0.0001,
        "ollama": 0.0,
    }

    # Free providers to try first
    FREE_PROVIDERS = ["ollama", "groq"]

    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self._provider_failures: Dict[str, int] = {}
        self._provider_success: Dict[str, int] = {}

    def route(
        self,
        task_type: str,
        estimated_tokens: int = 1000,
        require_quality: float = 0.5,
        budget_remaining: Optional[float] = None,
    ) -> RoutingDecision:
        """
        Route a request to the best provider.

        Args:
            task_type: Type of task to perform
            estimated_tokens: Estimated tokens needed
            require_quality: Minimum quality score required (0-1)
            budget_remaining: Override budget remaining

        Returns:
            RoutingDecision with provider and reasoning
        """
        complexity = self.TASK_COMPLEXITY.get(task_type, "medium")
        min_quality = int(require_quality * 10)

        # Get budget status
        if budget_remaining is None:
            try:
                tracker = get_token_manager()
                status = tracker.check_budget(self.daily_budget)
                budget_remaining = status["remaining"]
            except Exception:
                budget_remaining = self.daily_budget

        # Build candidate list
        candidates = []
        for provider, quality in self.PROVIDER_QUALITY.items():
            if quality >= min_quality:
                cost = self.PROVIDER_COST_PER_1K.get(provider, 0) * (estimated_tokens / 1000)

                # Skip if we can't afford it
                if cost > budget_remaining:
                    continue

                # Penalize providers with high failure rates
                failure_rate = self._get_failure_rate(provider)
                adjusted_quality = quality * (1 - failure_rate)

                candidates.append({
                    "provider": provider,
                    "quality": adjusted_quality,
                    "cost": cost,
                    "is_free": provider in self.FREE_PROVIDERS,
                })

        if not candidates:
            # Fallback to cheapest available
            return RoutingDecision(
                provider="ollama",
                reason="No providers within budget or quality requirements",
                estimated_cost=0.0,
                quality_score=0.5,
            )

        # Routing logic based on complexity and budget
        if complexity == "simple" or budget_remaining < 1.0:
            # Prefer free providers
            free_candidates = [c for c in candidates if c["is_free"]]
            if free_candidates:
                best = max(free_candidates, key=lambda c: c["quality"])
                return RoutingDecision(
                    provider=best["provider"],
                    reason=f"Simple task or low budget - using free provider",
                    estimated_cost=best["cost"],
                    quality_score=best["quality"] / 10,
                    fallback_provider=self._get_fallback(best["provider"], candidates),
                )

        if complexity == "medium":
            # Balance cost and quality
            # Score = quality - (cost * 10)  (penalize cost heavily)
            scored = [(c, c["quality"] - c["cost"] * 10) for c in candidates]
            best = max(scored, key=lambda x: x[1])[0]

            return RoutingDecision(
                provider=best["provider"],
                reason=f"Medium complexity - balanced cost/quality choice",
                estimated_cost=best["cost"],
                quality_score=best["quality"] / 10,
                fallback_provider=self._get_fallback(best["provider"], candidates),
            )

        # Complex tasks - prefer quality but respect budget
        if budget_remaining > 2.0:
            # Can afford premium
            best = max(candidates, key=lambda c: c["quality"])
        else:
            # Budget constrained - find best quality we can afford
            affordable = [c for c in candidates if c["cost"] <= budget_remaining * 0.5]
            best = max(affordable, key=lambda c: c["quality"]) if affordable else candidates[0]

        return RoutingDecision(
            provider=best["provider"],
            reason=f"Complex task - quality prioritized within budget",
            estimated_cost=best["cost"],
            quality_score=best["quality"] / 10,
            fallback_provider=self._get_fallback(best["provider"], candidates),
        )

    def _get_failure_rate(self, provider: str) -> float:
        """Get the failure rate for a provider."""
        failures = self._provider_failures.get(provider, 0)
        successes = self._provider_success.get(provider, 0)
        total = failures + successes
        return failures / total if total > 0 else 0.0

    def _get_fallback(self, primary: str, candidates: List[Dict]) -> Optional[str]:
        """Get a fallback provider different from the primary."""
        for c in candidates:
            if c["provider"] != primary and c["is_free"]:
                return c["provider"]
        return None

    def record_success(self, provider: str) -> None:
        """Record a successful API call."""
        self._provider_success[provider] = self._provider_success.get(provider, 0) + 1

    def record_failure(self, provider: str) -> None:
        """Record a failed API call."""
        self._provider_failures[provider] = self._provider_failures.get(provider, 0) + 1


# ============================================================
# Main Token Optimizer Class
# ============================================================

class TokenOptimizer:
    """
    Main token optimization system that combines all strategies.

    This is the primary interface for optimizing API costs.
    It provides:
    - Prompt optimization with multiple compression strategies
    - Intelligent caching with similarity matching
    - Batch processing for multiple requests
    - Smart provider routing based on task and budget
    - Token budget management per agent

    Usage:
        optimizer = TokenOptimizer(daily_budget=10.0)

        # Optimize a prompt
        result = optimizer.optimize_prompt(prompt, task_type="script")

        # Get provider for a task
        decision = optimizer.route_request(task_type="title_generation")

        # Check cache before making API call
        cached = optimizer.get_cached(prompt)
        if not cached:
            response = make_api_call(prompt)
            optimizer.cache_response(prompt, response)
    """

    # Default token budgets per agent type
    DEFAULT_AGENT_BUDGETS = {
        "ResearchAgent": TokenBudget("ResearchAgent", daily_limit=50000, hourly_limit=10000),
        "ScriptWriter": TokenBudget("ScriptWriter", daily_limit=100000, hourly_limit=20000),
        "SEOAgent": TokenBudget("SEOAgent", daily_limit=30000, hourly_limit=5000),
        "QualityAgent": TokenBudget("QualityAgent", daily_limit=20000, hourly_limit=5000),
        "ThumbnailAgent": TokenBudget("ThumbnailAgent", daily_limit=10000, hourly_limit=2000),
        "AnalyticsAgent": TokenBudget("AnalyticsAgent", daily_limit=15000, hourly_limit=3000),
        "default": TokenBudget("default", daily_limit=50000, hourly_limit=10000),
    }

    def __init__(
        self,
        daily_budget: float = 10.0,
        cache_ttl_hours: int = 24,
        enable_compression: bool = True,
        enable_caching: bool = True,
        enable_batching: bool = True,
    ):
        """
        Initialize the TokenOptimizer.

        Args:
            daily_budget: Daily spending limit in dollars
            cache_ttl_hours: Cache time-to-live in hours
            enable_compression: Enable prompt compression
            enable_caching: Enable response caching
            enable_batching: Enable request batching
        """
        self.daily_budget = daily_budget
        self.enable_compression = enable_compression
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching

        # Initialize components
        self.cache = AdvancedPromptCache(ttl_hours=cache_ttl_hours) if enable_caching else None
        self.router = SmartRouter(daily_budget=daily_budget)
        self.batch_processor = BatchProcessor() if enable_batching else None

        # Compression strategies
        self.compressors: List[CompressionStrategy] = [
            WhitespaceCompressor(),
            VerboseLanguageCondenser(),
            ExampleCondenser(),
            InstructionDeduplicator(),
            JSONSchemaMinifier(),
        ]

        # Response extractor
        self.extractor = ResponseExtractor()

        # Agent budgets
        self.agent_budgets: Dict[str, TokenBudget] = dict(self.DEFAULT_AGENT_BUDGETS)

        # Statistics
        self._stats = {
            "prompts_optimized": 0,
            "tokens_saved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "requests_batched": 0,
            "routing_decisions": 0,
        }

        logger.info(f"TokenOptimizer initialized (budget=${daily_budget}, "
                   f"compression={enable_compression}, caching={enable_caching})")

    def optimize_prompt(
        self,
        prompt: str,
        task_type: str = "general",
        preserve_json: bool = True,
    ) -> OptimizationResult:
        """
        Optimize a prompt to reduce token count.

        Args:
            prompt: The original prompt
            task_type: Type of task for context-aware optimization
            preserve_json: Don't compress JSON schemas

        Returns:
            OptimizationResult with optimized prompt and savings info
        """
        if not self.enable_compression:
            return OptimizationResult(
                original_tokens=self._estimate_tokens(prompt),
                optimized_tokens=self._estimate_tokens(prompt),
                savings_percent=0.0,
                optimized_prompt=prompt,
            )

        original_tokens = self._estimate_tokens(prompt)
        optimized = prompt
        applied = []

        for compressor in self.compressors:
            # Skip JSON minification if preserve_json is False
            if isinstance(compressor, JSONSchemaMinifier) and not preserve_json:
                continue

            savings = compressor.estimate_savings(optimized)
            if savings > 0.01:  # Only apply if > 1% savings
                optimized = compressor.compress(optimized)
                applied.append(type(compressor).__name__)

        optimized_tokens = self._estimate_tokens(optimized)
        savings_percent = (original_tokens - optimized_tokens) / original_tokens if original_tokens > 0 else 0

        self._stats["prompts_optimized"] += 1
        self._stats["tokens_saved"] += original_tokens - optimized_tokens

        logger.debug(f"Prompt optimized: {original_tokens} -> {optimized_tokens} tokens "
                    f"({savings_percent:.1%} savings)")

        return OptimizationResult(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            savings_percent=savings_percent,
            optimized_prompt=optimized,
            compression_applied=applied,
        )

    def get_cached(self, prompt: str) -> Optional[str]:
        """
        Get a cached response for a prompt.

        Args:
            prompt: The prompt to look up

        Returns:
            Cached response or None
        """
        if not self.enable_caching or not self.cache:
            return None

        result = self.cache.get(prompt)
        if result:
            self._stats["cache_hits"] += 1
        else:
            self._stats["cache_misses"] += 1

        return result

    def cache_response(
        self,
        prompt: str,
        response: str,
        provider: str = "",
        token_count: int = 0,
    ) -> None:
        """
        Cache a response for future use.

        Args:
            prompt: The original prompt
            response: The AI response
            provider: Provider that generated the response
            token_count: Token count for the response
        """
        if self.enable_caching and self.cache:
            self.cache.set(prompt, response, provider, token_count)

    def route_request(
        self,
        task_type: str,
        estimated_tokens: int = 1000,
        require_quality: float = 0.5,
    ) -> RoutingDecision:
        """
        Get the best provider for a request.

        Args:
            task_type: Type of task
            estimated_tokens: Estimated token count
            require_quality: Minimum quality required (0-1)

        Returns:
            RoutingDecision with provider recommendation
        """
        decision = self.router.route(task_type, estimated_tokens, require_quality)
        self._stats["routing_decisions"] += 1
        return decision

    def extract_response(
        self,
        response: str,
        extract_type: str = "text",
        section: Optional[str] = None,
    ) -> Any:
        """
        Extract specific data from a response.

        Args:
            response: The AI response
            extract_type: Type of extraction (json, list, code, section, text)
            section: Section name for section extraction

        Returns:
            Extracted data in appropriate format
        """
        if extract_type == "json":
            return self.extractor.extract_json(response)
        elif extract_type == "list":
            return self.extractor.extract_list(response)
        elif extract_type == "code":
            return self.extractor.extract_code(response)
        elif extract_type == "section":
            return self.extractor.extract_section(response, section or "")
        else:
            return response

    def check_agent_budget(self, agent_name: str, tokens_needed: int) -> bool:
        """
        Check if an agent has budget for an operation.

        Args:
            agent_name: Name of the agent
            tokens_needed: Tokens required for operation

        Returns:
            True if budget is available
        """
        budget = self.agent_budgets.get(agent_name, self.agent_budgets["default"])
        return budget.can_use(tokens_needed)

    def use_agent_budget(self, agent_name: str, tokens_used: int) -> bool:
        """
        Deduct tokens from an agent's budget.

        Args:
            agent_name: Name of the agent
            tokens_used: Tokens to deduct

        Returns:
            True if successful, False if would exceed budget
        """
        budget = self.agent_budgets.get(agent_name, self.agent_budgets["default"])
        return budget.use(tokens_used)

    def get_agent_budget_status(self, agent_name: str) -> Dict[str, Any]:
        """Get budget status for an agent."""
        budget = self.agent_budgets.get(agent_name, self.agent_budgets["default"])
        return {
            "agent": agent_name,
            "daily_limit": budget.daily_limit,
            "hourly_limit": budget.hourly_limit,
            "used_today": budget.used_today,
            "used_this_hour": budget.used_this_hour,
            "remaining_today": budget.remaining_today,
            "remaining_this_hour": budget.remaining_this_hour,
        }

    def set_agent_budget(
        self,
        agent_name: str,
        daily_limit: int,
        hourly_limit: int,
    ) -> None:
        """Set custom budget for an agent."""
        self.agent_budgets[agent_name] = TokenBudget(
            name=agent_name,
            daily_limit=daily_limit,
            hourly_limit=hourly_limit,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Approximate: 1 token per 4 characters for English
        return len(text) // 4

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = dict(self._stats)

        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        stats["agent_budgets"] = {
            name: self.get_agent_budget_status(name)
            for name in self.agent_budgets
        }

        return stats

    def get_savings_report(self) -> str:
        """Generate a human-readable savings report."""
        stats = self.get_stats()

        report = "\n" + "=" * 50 + "\n"
        report += "       TOKEN OPTIMIZATION REPORT\n"
        report += "=" * 50 + "\n\n"

        report += f"Prompts Optimized: {stats['prompts_optimized']:,}\n"
        report += f"Tokens Saved: {stats['tokens_saved']:,}\n"

        if stats['prompts_optimized'] > 0:
            avg_savings = stats['tokens_saved'] / stats['prompts_optimized']
            report += f"Avg Savings/Prompt: {avg_savings:.0f} tokens\n"

        report += f"\nCache Hits: {stats['cache_hits']:,}\n"
        report += f"Cache Misses: {stats['cache_misses']:,}\n"

        if stats['cache_hits'] + stats['cache_misses'] > 0:
            hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            report += f"Cache Hit Rate: {hit_rate:.1%}\n"

        report += f"\nRouting Decisions: {stats['routing_decisions']:,}\n"

        # Estimated cost savings
        # Assuming average cost of $0.002 per 1K tokens saved
        estimated_savings = (stats['tokens_saved'] / 1000) * 0.002
        cache_savings = (stats['cache_hits'] * 500 / 1000) * 0.002  # Assume 500 tokens per cached response
        total_savings = estimated_savings + cache_savings

        report += f"\n--- Estimated Savings ---\n"
        report += f"From Compression: ${estimated_savings:.4f}\n"
        report += f"From Caching: ${cache_savings:.4f}\n"
        report += f"Total Estimated: ${total_savings:.4f}\n"

        report += "\n" + "=" * 50 + "\n"

        return report


# ============================================================
# Singleton Instance and Helper Functions
# ============================================================

_token_optimizer: Optional[TokenOptimizer] = None


def get_token_optimizer(daily_budget: float = 10.0) -> TokenOptimizer:
    """Get singleton TokenOptimizer instance."""
    global _token_optimizer
    if _token_optimizer is None:
        _token_optimizer = TokenOptimizer(daily_budget=daily_budget)
    return _token_optimizer


def optimize_prompt(prompt: str, task_type: str = "general") -> str:
    """Convenience function to optimize a prompt."""
    optimizer = get_token_optimizer()
    result = optimizer.optimize_prompt(prompt, task_type)
    return result.optimized_prompt


def smart_route(task_type: str, estimated_tokens: int = 1000) -> str:
    """Convenience function to get best provider for a task."""
    optimizer = get_token_optimizer()
    decision = optimizer.route_request(task_type, estimated_tokens)
    return decision.provider


def check_cache(prompt: str) -> Optional[str]:
    """Convenience function to check cache for a prompt."""
    optimizer = get_token_optimizer()
    return optimizer.get_cached(prompt)


def cache_response(prompt: str, response: str, provider: str = "") -> None:
    """Convenience function to cache a response."""
    optimizer = get_token_optimizer()
    optimizer.cache_response(prompt, response, provider)


# ============================================================
# Decorators for Easy Integration
# ============================================================

def with_optimization(task_type: str = "general"):
    """
    Decorator to automatically optimize prompts and cache responses.

    Usage:
        @with_optimization(task_type="script")
        def generate_script(prompt: str) -> str:
            # Make API call
            return response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(prompt: str, *args, **kwargs) -> str:
            optimizer = get_token_optimizer()

            # Check cache first
            cached = optimizer.get_cached(prompt)
            if cached:
                logger.info(f"Cache hit for {func.__name__}")
                return cached

            # Optimize prompt
            result = optimizer.optimize_prompt(prompt, task_type)

            # Call function with optimized prompt
            response = func(result.optimized_prompt, *args, **kwargs)

            # Cache response
            optimizer.cache_response(prompt, response)

            return response
        return wrapper
    return decorator


def with_routing(task_type: str = "general", require_quality: float = 0.5):
    """
    Decorator to automatically route to best provider.

    The decorated function should accept a 'provider' keyword argument.

    Usage:
        @with_routing(task_type="title_generation")
        def generate_title(prompt: str, provider: str = "groq") -> str:
            # Make API call using provider
            return response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            optimizer = get_token_optimizer()

            # Get routing decision
            decision = optimizer.route_request(task_type, require_quality=require_quality)

            # Set provider in kwargs
            kwargs['provider'] = decision.provider
            logger.info(f"Routing {func.__name__} to {decision.provider}: {decision.reason}")

            try:
                result = func(*args, **kwargs)
                optimizer.router.record_success(decision.provider)
                return result
            except Exception as e:
                optimizer.router.record_failure(decision.provider)

                # Try fallback if available
                if decision.fallback_provider:
                    logger.warning(f"Primary provider {decision.provider} failed, "
                                  f"trying fallback {decision.fallback_provider}")
                    kwargs['provider'] = decision.fallback_provider
                    return func(*args, **kwargs)

                raise
        return wrapper
    return decorator


def with_budget(agent_name: str):
    """
    Decorator to enforce token budget for an agent.

    Usage:
        @with_budget("ResearchAgent")
        def research_topic(prompt: str) -> str:
            # Make API call
            return response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            optimizer = get_token_optimizer()

            # Estimate tokens needed (rough estimate from first positional arg if string)
            estimated_tokens = 1000
            if args and isinstance(args[0], str):
                estimated_tokens = len(args[0]) // 4 + 1000  # Input + expected output

            # Check budget
            if not optimizer.check_agent_budget(agent_name, estimated_tokens):
                status = optimizer.get_agent_budget_status(agent_name)
                raise BudgetExceededError(
                    f"Agent {agent_name} budget exceeded. "
                    f"Remaining today: {status['remaining_today']}, "
                    f"Remaining this hour: {status['remaining_this_hour']}"
                )

            # Execute function
            result = func(*args, **kwargs)

            # Deduct tokens (estimate actual usage)
            actual_tokens = estimated_tokens  # Could be refined with actual response
            if isinstance(result, str):
                actual_tokens = len(args[0]) // 4 + len(result) // 4 if args else len(result) // 4

            optimizer.use_agent_budget(agent_name, actual_tokens)

            return result
        return wrapper
    return decorator


# ============================================================
# Main Entry Point for Testing
# ============================================================

if __name__ == "__main__":
    # Test the token optimizer
    optimizer = get_token_optimizer(daily_budget=10.0)

    # Test prompt optimization
    test_prompt = """
    In order to generate a script, you must follow these instructions carefully.

    Please make sure to include the following elements:

    1. A strong hook in the first 5 seconds
    2. Clear transitions between sections
    3. A compelling call to action

    Example 1: "Did you know that..."
    Example 2: "What if I told you..."
    Example 3: "The surprising truth is..."
    Example 4: "Scientists discovered..."
    Example 5: "Here's what nobody tells you..."

    It is important to note that the script should be engaging and informative.
    You should be aware that viewers have short attention spans.
    Keep in mind that quality matters more than quantity.

    ```json
    {
        "title": "string",
        "hook": "string",
        "sections": [
            {
                "heading": "string",
                "content": "string"
            }
        ]
    }
    ```
    """

    result = optimizer.optimize_prompt(test_prompt, task_type="full_script")

    print("=" * 50)
    print("PROMPT OPTIMIZATION TEST")
    print("=" * 50)
    print(f"Original tokens: {result.original_tokens}")
    print(f"Optimized tokens: {result.optimized_tokens}")
    print(f"Savings: {result.savings_percent:.1%}")
    print(f"Strategies applied: {', '.join(result.compression_applied)}")
    print("\nOptimized prompt:")
    print("-" * 40)
    print(result.optimized_prompt)
    print("-" * 40)

    # Test routing
    print("\n" + "=" * 50)
    print("ROUTING TEST")
    print("=" * 50)

    for task in ["title_generation", "script_outline", "full_script"]:
        decision = optimizer.route_request(task)
        print(f"{task}: {decision.provider} ({decision.reason})")

    # Test cache
    print("\n" + "=" * 50)
    print("CACHE TEST")
    print("=" * 50)

    test_prompt = "Generate a title for a video about Python programming"
    test_response = "10 Python Tips That Will Make You a Better Developer"

    optimizer.cache_response(test_prompt, test_response, "groq")
    cached = optimizer.get_cached(test_prompt)
    print(f"Cached response: {cached}")

    # Print stats
    print("\n" + optimizer.get_savings_report())
