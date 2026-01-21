"""Query caching for repeated database lookups."""
from functools import lru_cache
from typing import Optional, Dict, Any
from loguru import logger


@lru_cache(maxsize=128)
def get_cached_channel_config(channel_id: str) -> Dict[str, Any]:
    """Cached channel config lookup."""
    from src.config import load_channel_config
    logger.debug(f"Cache miss: loading channel config for {channel_id}")
    return load_channel_config(channel_id)


@lru_cache(maxsize=256)
def get_cached_stock_query(query: str, provider: str = "pexels") -> str:
    """Cache stock footage query results hash."""
    return f"{provider}:{query}"


def clear_cache():
    """Clear all caches."""
    get_cached_channel_config.cache_clear()
    get_cached_stock_query.cache_clear()
    logger.info("Query caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "channel_config": get_cached_channel_config.cache_info()._asdict(),
        "stock_query": get_cached_stock_query.cache_info()._asdict(),
    }
