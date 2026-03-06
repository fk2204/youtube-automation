"""
RepurposerRegistry — global registry for all content repurposers.

Each repurposer module calls _register() at module level so that
importing the module is sufficient to make it available here.

Usage:
    from distribution.repurposers.repurposer_registry import RepurposerRegistry

    repurposer_class = RepurposerRegistry.get("medium")
    repurposer = repurposer_class(config)
    await repurposer.authenticate()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Type

if TYPE_CHECKING:
    from distribution.repurposers.base_repurposer import BaseRepurposer


class RepurposerRegistry:
    """Singleton registry mapping platform names to repurposer classes."""

    _registry: Dict[str, Type["BaseRepurposer"]] = {}

    @classmethod
    def register(cls, platform: str, repurposer_class: Type["BaseRepurposer"]) -> None:
        """
        Register a repurposer class under a platform key.

        Called by each repurposer module at import time via _register().
        Overwrites silently if the same key is registered twice —
        this allows hot-reloading during development.
        """
        cls._registry[platform.lower()] = repurposer_class

    @classmethod
    def get(cls, platform: str) -> Optional[Type["BaseRepurposer"]]:
        """
        Return the repurposer class for the given platform name, or None.

        Callers should check for None before instantiating.
        """
        return cls._registry.get(platform.lower())

    @classmethod
    def all_platforms(cls) -> List[str]:
        """Return sorted list of all registered platform names."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, platform: str) -> bool:
        """Return True if a repurposer is registered for the given platform."""
        return platform.lower() in cls._registry
