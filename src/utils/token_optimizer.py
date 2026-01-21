"""
Token Optimization System for YouTube Automation

Comprehensive system to reduce API costs by 50%+ through:
1. Prompt caching - Cache repeated prompts with intelligent invalidation
2. Semantic caching - Find similar prompts even if not exact matches
3. Response compression - Extract only needed data from responses
4. Batch processing - Combine multiple small requests into one
5. Smart provider routing - Use free providers when possible
6. Token budget allocation - Per-agent budgets with enforcement
7. Automatic fallback - Switch to cheaper providers when budget low

Usage:
    from src.utils.token_optimizer import (
        TokenOptimizer,
        get_token_optimizer,
        optimize_prompt,
        batch_requests,
        smart_route,
        SemanticCache,
        check_semantic_cache,
    )

    optimizer = get_token_optimizer()

    # Optimize a prompt before sending
    optimized = optimizer.optimize_prompt(prompt, task_type="script")

    # Route to best provider based on task and budget
    provider = optimizer.smart_route(task_type="title_generation")

    # Check cache (includes semantic matching by default)
    cached = optimizer.get_cached(prompt)  # Tries exact, then semantic match

    # Batch multiple requests
    results = optimizer.batch_process(requests_list)

Semantic Caching:
    The semantic cache finds similar prompts using text similarity algorithms.
    This allows cache hits even when prompts vary slightly:

    # These prompts would match with semantic caching:
    # - "Generate a title for Python tutorial"
    # - "Generate title for a Python programming video"
    # - "Create a title for Python lesson"

    # Configure similarity threshold (default 0.85 = 85% similar)
    optimizer = TokenOptimizer(semantic_threshold=0.80)

    # Use the semantic cache directly
    from src.utils.token_optimizer import SemanticCache
    cache = SemanticCache(similarity_threshold=0.85)
    cache.set("prompt", "response")
    result = cache.get_similar("similar prompt")
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

# Try importing text similarity utilities
try:
    from src.utils.text_similarity import (
        word_overlap_similarity,
        character_ngram_similarity,
        levenshtein_similarity,
        combined_similarity,
        prompt_similarity,
        normalize_prompt_for_comparison,
        fast_similarity,
    )
    TEXT_SIMILARITY_AVAILABLE = True
except ImportError:
    TEXT_SIMILARITY_AVAILABLE = False
    # Fallback implementations will be provided in class methods

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
# Task-Specific Token Limits (API Cost Optimization)
# ============================================================

# Maximum tokens for each task type to minimize API costs
# These limits are calibrated to provide sufficient output while avoiding waste
TASK_MAX_TOKENS = {
    "title_generation": 50,
    "tag_generation": 100,
    "description_generation": 300,
    "hook_generation": 150,
    "script_outline": 500,
    "full_script": 4000,
    "thumbnail_text": 30,
    "seo_keywords": 150,
    "idea_generation": 200,
    "trend_research": 300,
    "script_revision": 1000,
    "content_summary": 250,
    "seo_optimization": 200,
    "research_synthesis": 1500,
    "creative_writing": 2000,
    "technical_explanation": 1000,
    "competitor_analysis": 500,
}


# ============================================================
# Cache TTL by Content Type (Extended Caching)
# ============================================================

# Different TTLs based on content type for more efficient caching
CACHE_TTL_BY_TYPE = {
    "evergreen": 90,      # 90 days for timeless content
    "trending": 7,        # 7 days for trending topics
    "news": 1,            # 1 day for news-related content
    "template": 365,      # 1 year for reusable templates
    "default": 30,        # 30 days default
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
# Semantic Cache
# ============================================================

class SemanticCache:
    """
    Cache that can find similar prompts using text similarity.
    Falls back to exact matching if similarity libraries unavailable.

    This cache is designed to find cached responses even when the prompts
    are not exact matches, using various text similarity algorithms.

    Features:
    - Exact match lookup (fast, hash-based)
    - Prefix-based candidate filtering (medium speed)
    - Full semantic similarity calculation (slower but more accurate)
    - Configurable similarity threshold
    - TTL-based expiration
    - Statistics tracking for semantic vs exact hits

    Usage:
        cache = SemanticCache(similarity_threshold=0.85)

        # Store a response
        cache.set("Generate a title for Python tutorial", "10 Python Tips...")

        # Find similar - will match even with slight variations
        response = cache.get_similar("Generate title for a Python video tutorial")
        # Returns "10 Python Tips..." because similarity > 0.85
    """

    def __init__(
        self,
        db_path: str = "data/semantic_cache.db",
        similarity_threshold: float = 0.85,
        ttl_days: int = 30,
        max_candidates: int = 50,
    ):
        """
        Initialize the SemanticCache.

        Args:
            db_path: Path to SQLite database file
            similarity_threshold: Minimum similarity score (0-1) to consider a match
            ttl_days: Time-to-live in days for cache entries
            max_candidates: Maximum number of candidates to check for similarity
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.ttl_days = ttl_days
        self.max_candidates = max_candidates
        self._lock = threading.Lock()
        self._init_db()

        # Statistics
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0

    def _init_db(self):
        """Initialize database with prompt prefixes for fast lookup."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt_text TEXT NOT NULL,
                    prompt_prefix TEXT NOT NULL,
                    normalized_prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    provider TEXT DEFAULT '',
                    token_count INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            # Indexes for fast lookup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_prefix
                ON semantic_cache(prompt_prefix)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_expires
                ON semantic_cache(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_accessed
                ON semantic_cache(last_accessed)
            """)

    def _normalize_prompt(self, prompt: str) -> str:
        """
        Normalize prompt for comparison.

        Removes extra whitespace, converts to lowercase, and strips
        punctuation variations that don't affect meaning.
        """
        if TEXT_SIMILARITY_AVAILABLE:
            return normalize_prompt_for_comparison(prompt)

        # Fallback normalization
        normalized = prompt.lower().strip()
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove common punctuation variations
        normalized = re.sub(r'["\']', '', normalized)
        return normalized

    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Calculate similarity between two prompts.

        Uses the text_similarity module if available, otherwise
        falls back to simple word overlap ratio.
        """
        if TEXT_SIMILARITY_AVAILABLE:
            return prompt_similarity(prompt1, prompt2)

        # Fallback: simple word overlap ratio
        return self._word_overlap_similarity(prompt1, prompt2)

    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple word overlap (Jaccard) similarity.

        This is a fallback when the text_similarity module is unavailable.
        """
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Levenshtein-based similarity.

        Uses the text_similarity module if available, otherwise
        uses difflib as fallback.
        """
        if TEXT_SIMILARITY_AVAILABLE:
            return levenshtein_similarity(text1, text2)

        # Fallback: use difflib SequenceMatcher
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        except Exception:
            # Ultimate fallback to word overlap
            return self._word_overlap_similarity(text1, text2)

    def _get_prompt_signature(self, prompt: str, length: int = 100) -> str:
        """
        Get a signature for fast prefix matching.

        The signature is the normalized first N characters of the prompt,
        used for quickly filtering candidates before full similarity check.
        """
        normalized = self._normalize_prompt(prompt)
        return normalized[:length]

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of the prompt for exact matching."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _get_exact(self, prompt: str) -> Optional[str]:
        """Try to get an exact match from the cache."""
        prompt_hash = self._hash_prompt(prompt)
        now = datetime.now().isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT response FROM semantic_cache
                       WHERE prompt_hash = ? AND expires_at > ?""",
                    (prompt_hash, now)
                ).fetchone()

                if row:
                    # Update access stats
                    conn.execute(
                        """UPDATE semantic_cache
                           SET access_count = access_count + 1, last_accessed = ?
                           WHERE prompt_hash = ?""",
                        (now, prompt_hash)
                    )
                    return row[0]

        return None

    def _find_candidates(self, signature: str) -> List[Tuple[str, str]]:
        """
        Find candidate prompts with similar prefix.

        Returns list of (prompt_text, response) tuples for prompts
        that have a similar prefix to the given signature.
        """
        now = datetime.now().isoformat()
        # Use first 50 chars for prefix matching to get wider candidate pool
        prefix_match = signature[:50] if len(signature) >= 50 else signature

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Get candidates with similar prefix
                rows = conn.execute(
                    """SELECT normalized_prompt, response
                       FROM semantic_cache
                       WHERE prompt_prefix LIKE ? AND expires_at > ?
                       ORDER BY access_count DESC
                       LIMIT ?""",
                    (prefix_match[:20] + '%', now, self.max_candidates)
                ).fetchall()

                return [(row[0], row[1]) for row in rows]

    def get_similar(self, prompt: str) -> Optional[str]:
        """
        Find cached response for similar prompt.

        This method tries three strategies in order:
        1. Exact match (fastest) - hash-based lookup
        2. Prefix filtering - find candidates with similar beginnings
        3. Full similarity - calculate similarity for candidates

        Args:
            prompt: The prompt to find a similar cached response for

        Returns:
            Cached response if found with similarity >= threshold, None otherwise
        """
        normalized = self._normalize_prompt(prompt)
        signature = self._get_prompt_signature(normalized)

        # First: try exact match
        exact = self._get_exact(prompt)
        if exact:
            self.exact_hits += 1
            logger.debug(f"Semantic cache exact hit")
            return exact

        # Second: find candidates with similar prefix
        candidates = self._find_candidates(signature)

        if not candidates:
            self.misses += 1
            return None

        # Third: calculate full similarity for candidates
        best_match = None
        best_similarity = 0.0

        for candidate_prompt, response in candidates:
            # Quick filter using fast similarity
            if TEXT_SIMILARITY_AVAILABLE:
                quick_score = fast_similarity(normalized, candidate_prompt)
                if quick_score < self.similarity_threshold * 0.7:
                    continue

            similarity = self._calculate_similarity(normalized, candidate_prompt)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = response

        if best_match:
            self.semantic_hits += 1
            logger.debug(f"Semantic cache hit: {best_similarity:.2%} similarity")
            return best_match

        self.misses += 1
        return None

    def set(
        self,
        prompt: str,
        response: str,
        provider: str = "",
        token_count: int = 0,
        ttl_days: Optional[int] = None,
    ):
        """
        Store prompt-response pair in the cache.

        Args:
            prompt: The original prompt
            response: The AI response to cache
            provider: Provider that generated the response
            token_count: Token count for the response
            ttl_days: Custom TTL in days (uses default if not specified)
        """
        prompt_hash = self._hash_prompt(prompt)
        normalized = self._normalize_prompt(prompt)
        prefix = self._get_prompt_signature(normalized)
        now = datetime.now()
        ttl = ttl_days if ttl_days is not None else self.ttl_days
        expires_at = (now + timedelta(days=ttl)).isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO semantic_cache
                       (prompt_hash, prompt_text, prompt_prefix, normalized_prompt,
                        response, provider, token_count, access_count,
                        created_at, last_accessed, expires_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                    (prompt_hash, prompt, prefix, normalized, response,
                     provider, token_count, now.isoformat(), now.isoformat(),
                     expires_at)
                )

        logger.debug(f"Semantic cache stored: {prompt_hash[:8]}...")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including semantic hit rate.

        Returns:
            Dictionary with cache statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM semantic_cache"
            ).fetchone()[0]

            expired = conn.execute(
                "SELECT COUNT(*) FROM semantic_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            ).fetchone()[0]

            total_accesses = conn.execute(
                "SELECT SUM(access_count) FROM semantic_cache"
            ).fetchone()[0] or 0

        total_lookups = self.exact_hits + self.semantic_hits + self.misses
        exact_rate = self.exact_hits / total_lookups if total_lookups > 0 else 0
        semantic_rate = self.semantic_hits / total_lookups if total_lookups > 0 else 0
        total_hit_rate = (self.exact_hits + self.semantic_hits) / total_lookups if total_lookups > 0 else 0

        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "total_accesses": total_accesses,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "exact_hit_rate": exact_rate,
            "semantic_hit_rate": semantic_rate,
            "total_hit_rate": total_hit_rate,
            "similarity_threshold": self.similarity_threshold,
        }

    def clear_expired(self) -> int:
        """
        Clear expired cache entries.

        Returns:
            Number of entries deleted
        """
        now = datetime.now().isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "DELETE FROM semantic_cache WHERE expires_at < ?",
                    (now,)
                )
                deleted = result.rowcount

        logger.info(f"Cleared {deleted} expired semantic cache entries")
        return deleted

    def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("DELETE FROM semantic_cache")
                deleted = result.rowcount

        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0

        logger.info(f"Cleared all {deleted} semantic cache entries")
        return deleted


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
            # Additional indexes for optimization
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_prefix ON prompt_cache(prompt_prefix)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_accessed ON prompt_cache(last_accessed)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_provider ON prompt_cache(provider)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created ON prompt_cache(created_at)
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
    # Semantic Similarity Methods
    # ============================================================

    def get_semantic_similar(
        self,
        prompt: str,
        threshold: float = 0.85,
        max_candidates: int = 20,
    ) -> Optional[str]:
        """
        Find similar cached responses using semantic similarity.

        This method searches for cached responses that are semantically
        similar to the given prompt, even if they don't match exactly.

        Args:
            prompt: The prompt to find similar cached responses for
            threshold: Minimum similarity score (0-1) to consider a match
            max_candidates: Maximum number of candidates to check

        Returns:
            Cached response if found with similarity >= threshold, None otherwise

        Example:
            # Original cached prompt: "Generate a title for Python tutorial"
            # Query: "Generate title for a Python programming video"
            # Will return the cached response if similarity >= threshold
        """
        # First try exact match (fast path)
        exact_result = self.get(prompt)
        if exact_result:
            return exact_result

        # Get prefix for candidate filtering
        prompt_prefix = self._get_prefix(prompt, length=50)
        cutoff = (datetime.now() - timedelta(hours=self.ttl_hours)).isoformat()

        # Find candidates with similar prefix
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Get candidates with similar prefix (first 20 chars match)
                rows = conn.execute(
                    """SELECT prompt_prefix, response, compressed
                       FROM prompt_cache
                       WHERE prompt_prefix LIKE ? AND last_accessed > ?
                       ORDER BY access_count DESC
                       LIMIT ?""",
                    (prompt_prefix[:20] + '%', cutoff, max_candidates)
                ).fetchall()

        if not rows:
            return None

        # Calculate similarity for each candidate
        normalized_prompt = prompt.lower().strip()
        best_match = None
        best_similarity = 0.0

        for candidate_prefix, response_data, compressed in rows:
            # Calculate similarity between prompts
            similarity = self._calculate_prompt_similarity(
                normalized_prompt,
                candidate_prefix
            )

            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                # Decompress if needed
                if compressed:
                    best_match = self._decompress(response_data)
                else:
                    best_match = response_data
                    if isinstance(best_match, bytes):
                        best_match = best_match.decode('utf-8')

        if best_match:
            self.hits += 1
            logger.debug(f"Semantic cache hit: {best_similarity:.2%} similarity")
            return best_match

        return None

    def _calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Calculate similarity between two prompts.

        Uses the text_similarity module if available, otherwise falls back
        to a combination of word overlap and Levenshtein similarity.

        Args:
            prompt1: First prompt (normalized)
            prompt2: Second prompt

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if TEXT_SIMILARITY_AVAILABLE:
            return prompt_similarity(prompt1, prompt2)

        # Fallback: combine word overlap and Levenshtein
        word_sim = self._word_overlap_similarity(prompt1, prompt2)
        lev_sim = self._levenshtein_similarity(prompt1, prompt2)

        # Weight word overlap more heavily for prompts
        return word_sim * 0.6 + lev_sim * 0.4

    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """
        Simple word overlap (Jaccard) similarity calculation.

        Calculates the ratio of shared words to total unique words.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0

        # Tokenize into words
        words1 = set(re.findall(r'\b[a-z0-9]+\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-z0-9]+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Levenshtein-based similarity calculation.

        Uses difflib SequenceMatcher as the implementation since it's
        available in the standard library.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if TEXT_SIMILARITY_AVAILABLE:
            return levenshtein_similarity(text1, text2)

        # Fallback: use difflib SequenceMatcher
        try:
            from difflib import SequenceMatcher

            if not text1 or not text2:
                return 0.0 if text1 != text2 else 1.0

            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        except Exception:
            # Ultimate fallback
            return self._word_overlap_similarity(text1, text2)


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
    # UPDATED: Route 90% of tasks to Groq (free) for cost optimization
    # Only complex full scripts use paid providers
    TASK_COMPLEXITY = {
        # Simple tasks - use Groq (free)
        "title_generation": "simple",
        "tag_generation": "simple",
        "description_generation": "simple",
        "idea_brainstorm": "simple",
        "keyword_extraction": "simple",
        "thumbnail_text": "simple",
        "seo_keywords": "simple",

        # UPDATED: These tasks now use Groq (moved from medium to simple)
        "script_outline": "simple",      # Use Groq - outlines don't need premium
        "hook_generation": "simple",     # Use Groq - hooks are short
        "script_revision": "simple",     # Use Groq - revisions are iterative
        "content_summary": "simple",     # Use Groq - summaries are straightforward
        "seo_optimization": "simple",    # Use Groq - SEO is formulaic
        "idea_generation": "simple",     # Use Groq - idea generation is creative but simple
        "trend_research": "simple",      # Use Groq - research can be done in steps
        "competitor_analysis": "simple", # Use Groq - analysis can be done incrementally

        # Complex tasks - ONLY these use paid providers when budget allows
        "full_script": "complex",        # Only full scripts justify paid providers
        "research_synthesis": "medium",  # Medium complexity, use Gemini if available
        "creative_writing": "medium",    # Medium complexity for creative tasks
        "technical_explanation": "medium", # Medium complexity for technical content
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
        enable_semantic_cache: bool = True,
        semantic_threshold: float = 0.85,
    ):
        """
        Initialize the TokenOptimizer.

        Args:
            daily_budget: Daily spending limit in dollars
            cache_ttl_hours: Cache time-to-live in hours
            enable_compression: Enable prompt compression
            enable_caching: Enable response caching
            enable_batching: Enable request batching
            enable_semantic_cache: Enable semantic similarity caching
            semantic_threshold: Similarity threshold for semantic cache (0-1)
        """
        self.daily_budget = daily_budget
        self.enable_compression = enable_compression
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        self.enable_semantic_cache = enable_semantic_cache
        self.semantic_threshold = semantic_threshold

        # Initialize components
        self.cache = AdvancedPromptCache(ttl_hours=cache_ttl_hours) if enable_caching else None
        self.semantic_cache = SemanticCache(
            similarity_threshold=semantic_threshold,
            ttl_days=cache_ttl_hours // 24 if cache_ttl_hours >= 24 else 1
        ) if enable_semantic_cache else None
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
            "semantic_hits": 0,
            "requests_batched": 0,
            "routing_decisions": 0,
        }

        logger.info(f"TokenOptimizer initialized (budget=${daily_budget}, "
                   f"compression={enable_compression}, caching={enable_caching}, "
                   f"semantic_cache={enable_semantic_cache})")

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

    def get_cached(
        self,
        prompt: str,
        try_semantic: bool = True,
        semantic_threshold: Optional[float] = None,
    ) -> Optional[str]:
        """
        Get a cached response for a prompt.

        This method tries multiple caching strategies in order:
        1. Exact match from advanced cache (fastest)
        2. Semantic similarity match from semantic cache (if enabled)
        3. Semantic similarity match from advanced cache

        Args:
            prompt: The prompt to look up
            try_semantic: Whether to try semantic matching if exact match fails
            semantic_threshold: Override similarity threshold (uses instance default if None)

        Returns:
            Cached response or None
        """
        if not self.enable_caching:
            return None

        threshold = semantic_threshold if semantic_threshold is not None else self.semantic_threshold

        # Strategy 1: Try exact match from advanced cache
        if self.cache:
            result = self.cache.get(prompt)
            if result:
                self._stats["cache_hits"] += 1
                logger.debug("Exact cache hit from advanced cache")
                return result

        # Strategy 2: Try semantic cache for similar prompts
        if try_semantic and self.enable_semantic_cache and self.semantic_cache:
            result = self.semantic_cache.get_similar(prompt)
            if result:
                self._stats["semantic_hits"] += 1
                self._stats["cache_hits"] += 1
                logger.debug("Semantic cache hit")
                return result

        # Strategy 3: Try semantic similarity from advanced cache
        if try_semantic and self.cache:
            result = self.cache.get_semantic_similar(
                prompt,
                threshold=threshold,
            )
            if result:
                self._stats["semantic_hits"] += 1
                self._stats["cache_hits"] += 1
                logger.debug("Semantic hit from advanced cache")
                return result

        self._stats["cache_misses"] += 1
        return None

    def get_cached_with_similarity(
        self,
        prompt: str,
        threshold: float = 0.85,
    ) -> Tuple[Optional[str], float]:
        """
        Get cached response with similarity score.

        Similar to get_cached but also returns the similarity score
        of the match (1.0 for exact match, <1.0 for semantic match).

        Args:
            prompt: The prompt to look up
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (response or None, similarity score)
        """
        # Try exact match first
        if self.cache:
            result = self.cache.get(prompt)
            if result:
                return result, 1.0

        # Try semantic match from semantic cache
        if self.semantic_cache:
            result = self.semantic_cache.get_similar(prompt)
            if result:
                # Estimate similarity (we know it's at least the threshold)
                return result, self.semantic_threshold

        # Try semantic match from advanced cache
        if self.cache:
            result = self.cache.get_semantic_similar(prompt, threshold=threshold)
            if result:
                return result, threshold

        return None, 0.0

    def cache_response(
        self,
        prompt: str,
        response: str,
        provider: str = "",
        token_count: int = 0,
    ) -> None:
        """
        Cache a response for future use.

        Stores the response in both the advanced cache and semantic cache
        for maximum retrieval options.

        Args:
            prompt: The original prompt
            response: The AI response
            provider: Provider that generated the response
            token_count: Token count for the response
        """
        # Store in advanced cache
        if self.enable_caching and self.cache:
            self.cache.set(prompt, response, provider, token_count)

        # Also store in semantic cache for similarity matching
        if self.enable_semantic_cache and self.semantic_cache:
            self.semantic_cache.set(prompt, response, provider, token_count)

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

    def get_max_tokens_for_task(self, task_type: str) -> int:
        """
        Get the appropriate max_tokens limit for a task type.

        Args:
            task_type: Type of task (e.g., "title_generation", "full_script")

        Returns:
            Maximum tokens to request for this task type
        """
        return TASK_MAX_TOKENS.get(task_type, 1000)  # Default to 1000 if unknown

    def truncate_response(
        self,
        response: str,
        task_type: str,
        preserve_structure: bool = True,
    ) -> str:
        """
        Truncate response to appropriate length based on task type.

        This helps reduce token usage by ensuring responses don't exceed
        what's actually needed for each task type.

        Args:
            response: The AI response to truncate
            task_type: Type of task for determining max length
            preserve_structure: If True, try to preserve JSON/list structure

        Returns:
            Truncated response
        """
        max_tokens = self.get_max_tokens_for_task(task_type)
        max_chars = max_tokens * 4  # Approximate chars from tokens

        if len(response) <= max_chars:
            return response

        logger.debug(f"Truncating response from {len(response)} to {max_chars} chars for {task_type}")

        if preserve_structure:
            # Try to preserve JSON structure
            if response.strip().startswith('{') or response.strip().startswith('['):
                try:
                    # Parse and re-serialize with limits
                    import json
                    data = json.loads(response)
                    # Truncate string values recursively
                    truncated_data = self._truncate_json_values(data, max_chars // 2)
                    return json.dumps(truncated_data, separators=(',', ':'))
                except json.JSONDecodeError:
                    pass

            # Try to preserve list structure
            lines = response.split('\n')
            if len(lines) > 1:
                # Keep as many complete lines as possible
                result = []
                current_len = 0
                for line in lines:
                    if current_len + len(line) + 1 <= max_chars:
                        result.append(line)
                        current_len += len(line) + 1
                    else:
                        break
                return '\n'.join(result)

        # Simple truncation
        return response[:max_chars]

    def _truncate_json_values(self, data: Any, max_total_chars: int) -> Any:
        """Recursively truncate string values in JSON data."""
        if isinstance(data, str):
            max_str_len = min(500, max_total_chars // 10)
            return data[:max_str_len] if len(data) > max_str_len else data
        elif isinstance(data, dict):
            return {k: self._truncate_json_values(v, max_total_chars) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._truncate_json_values(item, max_total_chars) for item in data[:20]]  # Max 20 items
        return data

    def batch_requests(
        self,
        requests: List[Dict[str, Any]],
        combine_similar: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch multiple similar requests into fewer API calls.

        This method combines multiple requests of the same task type into
        a single prompt, reducing API overhead and costs.

        Args:
            requests: List of request dicts with keys:
                - prompt: The prompt text
                - task_type: Type of task
                - callback: Optional callback function for results
            combine_similar: If True, combine requests of same task_type

        Returns:
            List of batched request dicts ready for processing

        Example:
            requests = [
                {"prompt": "Generate title for: AI video", "task_type": "title_generation"},
                {"prompt": "Generate title for: Python tutorial", "task_type": "title_generation"},
                {"prompt": "Generate tags for: AI video", "task_type": "tag_generation"},
            ]
            batched = optimizer.batch_requests(requests)
            # Returns 2 batched requests: one for titles, one for tags
        """
        if not combine_similar or len(requests) <= 1:
            return requests

        # Group by task type
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for req in requests:
            task_type = req.get("task_type", "general")
            if task_type not in grouped:
                grouped[task_type] = []
            grouped[task_type].append(req)

        batched_requests = []
        for task_type, group in grouped.items():
            if len(group) == 1:
                # Single request, no batching needed
                batched_requests.append(group[0])
            else:
                # Combine multiple requests into one
                combined_prompt = self._create_batch_prompt(group, task_type)
                max_tokens = self.get_max_tokens_for_task(task_type) * len(group)

                batched_requests.append({
                    "prompt": combined_prompt,
                    "task_type": task_type,
                    "is_batch": True,
                    "batch_size": len(group),
                    "original_requests": group,
                    "max_tokens": max_tokens,
                })

        self._stats["requests_batched"] += sum(len(g) for g in grouped.values() if len(g) > 1)
        logger.info(f"Batched {len(requests)} requests into {len(batched_requests)} API calls")

        return batched_requests

    def _create_batch_prompt(self, requests: List[Dict[str, Any]], task_type: str) -> str:
        """Create a combined prompt for batched requests."""
        prompt = f"""Process the following {len(requests)} requests for {task_type}.
For each request, provide a response labeled with its number.

Format your response as:
[1] <response for request 1>
[2] <response for request 2>
...

---REQUESTS---

"""
        for i, req in enumerate(requests, 1):
            prompt += f"[{i}] {req.get('prompt', '')}\n\n"

        prompt += "---END REQUESTS---\n\nProvide responses for each request above:"
        return prompt

    def parse_batch_response(
        self,
        response: str,
        batch_request: Dict[str, Any],
    ) -> List[str]:
        """
        Parse a batched response back into individual responses.

        Args:
            response: The combined response from the AI
            batch_request: The batched request dict (from batch_requests)

        Returns:
            List of individual responses in the same order as original requests
        """
        batch_size = batch_request.get("batch_size", 1)
        results = []

        for i in range(1, batch_size + 1):
            # Look for the response section for this number
            pattern = rf'\[{i}\]\s*(.*?)(?=\[{i+1}\]|\Z)'
            match = re.search(pattern, response, re.DOTALL)

            if match:
                results.append(match.group(1).strip())
            else:
                results.append("")
                logger.warning(f"Could not parse response for batch item {i}")

        return results

    def get_cache_ttl(self, content_type: str = "default") -> int:
        """
        Get the appropriate cache TTL (in days) for a content type.

        Args:
            content_type: Type of content ("evergreen", "trending", "news", "template")

        Returns:
            TTL in days
        """
        return CACHE_TTL_BY_TYPE.get(content_type, CACHE_TTL_BY_TYPE["default"])

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = dict(self._stats)

        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        if self.semantic_cache:
            stats["semantic_cache_stats"] = self.semantic_cache.get_stats()

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
        report += f"  - Semantic Hits: {stats.get('semantic_hits', 0):,}\n"
        report += f"  - Exact Hits: {stats['cache_hits'] - stats.get('semantic_hits', 0):,}\n"
        report += f"Cache Misses: {stats['cache_misses']:,}\n"

        if stats['cache_hits'] + stats['cache_misses'] > 0:
            hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            report += f"Cache Hit Rate: {hit_rate:.1%}\n"

            # Semantic contribution
            if stats['cache_hits'] > 0:
                semantic_contribution = stats.get('semantic_hits', 0) / stats['cache_hits']
                report += f"Semantic Contribution: {semantic_contribution:.1%}\n"

        # Semantic cache specific stats
        if "semantic_cache_stats" in stats:
            sem_stats = stats["semantic_cache_stats"]
            report += f"\n--- Semantic Cache ---\n"
            report += f"Total Entries: {sem_stats['total_entries']:,}\n"
            report += f"Active Entries: {sem_stats['active_entries']:,}\n"
            report += f"Similarity Threshold: {sem_stats['similarity_threshold']:.0%}\n"
            report += f"Semantic Hit Rate: {sem_stats['semantic_hit_rate']:.1%}\n"

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


def check_cache(prompt: str, try_semantic: bool = True) -> Optional[str]:
    """
    Convenience function to check cache for a prompt.

    Args:
        prompt: The prompt to look up
        try_semantic: Whether to try semantic matching if exact match fails

    Returns:
        Cached response or None
    """
    optimizer = get_token_optimizer()
    return optimizer.get_cached(prompt, try_semantic=try_semantic)


def check_semantic_cache(prompt: str, threshold: float = 0.85) -> Optional[str]:
    """
    Convenience function to check semantic cache for similar prompts.

    Args:
        prompt: The prompt to find similar matches for
        threshold: Minimum similarity threshold (0-1)

    Returns:
        Cached response from similar prompt or None
    """
    optimizer = get_token_optimizer()
    if optimizer.semantic_cache:
        return optimizer.semantic_cache.get_similar(prompt)
    return None


def cache_response(prompt: str, response: str, provider: str = "") -> None:
    """Convenience function to cache a response."""
    optimizer = get_token_optimizer()
    optimizer.cache_response(prompt, response, provider)


def get_semantic_cache() -> Optional[SemanticCache]:
    """Get the semantic cache instance from the token optimizer."""
    optimizer = get_token_optimizer()
    return optimizer.semantic_cache


# ============================================================
# Decorators for Easy Integration
# ============================================================

def with_optimization(task_type: str = "general", use_semantic: bool = True):
    """
    Decorator to automatically optimize prompts and cache responses.

    This decorator:
    1. Checks exact cache first (fastest)
    2. If use_semantic=True, checks semantic cache for similar prompts
    3. Optimizes the prompt to reduce tokens
    4. Calls the wrapped function
    5. Caches the response for future use

    Args:
        task_type: Type of task for context-aware optimization
        use_semantic: Whether to try semantic matching if exact match fails

    Usage:
        @with_optimization(task_type="script")
        def generate_script(prompt: str) -> str:
            # Make API call
            return response

        @with_optimization(task_type="title_generation", use_semantic=True)
        def generate_title(prompt: str) -> str:
            # Will find similar cached prompts even with variations
            return response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(prompt: str, *args, **kwargs) -> str:
            optimizer = get_token_optimizer()

            # Check cache first (includes semantic matching if enabled)
            cached = optimizer.get_cached(prompt, try_semantic=use_semantic)
            if cached:
                logger.info(f"Cache hit for {func.__name__} (semantic={use_semantic})")
                return cached

            # Optimize prompt
            result = optimizer.optimize_prompt(prompt, task_type)

            # Call function with optimized prompt
            response = func(result.optimized_prompt, *args, **kwargs)

            # Cache response (stores in both exact and semantic caches)
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
