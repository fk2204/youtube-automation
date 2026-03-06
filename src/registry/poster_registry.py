"""
PosterRegistry — global registry for all social platform posters.

Each poster module calls _register() at module level so that
importing the module is sufficient to make it available here.

Usage:
    from registry.poster_registry import PosterRegistry

    poster = PosterRegistry.get("tiktok")
    poster.authenticate()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Type

if TYPE_CHECKING:
    from social.base_poster import BasePoster


class PosterRegistry:
    """Singleton registry mapping platform names to poster classes."""

    _registry: Dict[str, Type["BasePoster"]] = {}

    @classmethod
    def register(cls, platform: str, poster_class: Type["BasePoster"]) -> None:
        """
        Register a poster class under a platform key.

        Called by each poster module at import time via _register().
        Overwrites silently if the same key is registered twice —
        this allows hot-reloading during development.
        """
        cls._registry[platform.lower()] = poster_class

    @classmethod
    def get(cls, platform: str) -> Optional[Type["BasePoster"]]:
        """
        Return the poster class for the given platform name, or None.

        Callers should check for None before instantiating.
        """
        return cls._registry.get(platform.lower())

    @classmethod
    def all_platforms(cls) -> list[str]:
        """Return sorted list of all registered platform names."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, platform: str) -> bool:
        """Return True if a poster is registered for the given platform."""
        return platform.lower() in cls._registry
