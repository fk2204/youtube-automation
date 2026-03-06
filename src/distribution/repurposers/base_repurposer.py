"""
BaseRepurposer — abstract base class for all content repurposers.

Defines the interface and shared utilities that every repurposer must implement.
Repurposers transform source content into platform-specific formats and
distribute it (or prepare it for distribution) to a target platform.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger


class BaseRepurposer(ABC):
    """
    Abstract base providing the minimum interface for a content repurposer.

    Subclasses must implement: platform_name(), authenticate(), repurpose().
    All repurposers support simulation mode via SOCIAL_SIMULATION_MODE env var.
    """

    # Subclasses override to identify themselves in logs and registry.
    PLATFORM_NAME: str = "unknown"

    # Subclasses set to describe their automation level.
    # "full" = API posting, "content_only" = generates files only.
    AUTOMATION_LEVEL: str = "full"

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the repurposer with a configuration dict.

        Args:
            config: Platform-specific configuration (API keys, paths, etc.)
                    Must not contain plaintext secrets — use env vars for those.
        """
        self._config = config
        self._authenticated: bool = False

    @property
    def platform_name(self) -> str:
        """Return the platform identifier string."""
        return self.PLATFORM_NAME

    @property
    def simulation_mode(self) -> bool:
        """True when SOCIAL_SIMULATION_MODE env var is set to 'true'."""
        return os.environ.get("SOCIAL_SIMULATION_MODE", "").lower() == "true"

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the platform using credentials from env vars or config.

        Returns:
            True on success, False on failure (never raises).
        """
        ...

    @abstractmethod
    async def repurpose(
        self,
        source_content: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transform source content and distribute (or prepare) for the platform.

        Args:
            source_content: Raw content string (markdown, text, HTML, etc.)
            **kwargs:        Platform-specific parameters.

        Returns:
            Status dict — shape varies by platform but always contains:
            {"success": bool, "error": str | None}
        """
        ...

    def _simulate_repurpose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a simulated success response without making any API call.

        Used when SOCIAL_SIMULATION_MODE=true.
        """
        logger.info(
            "[SIMULATION] {platform} repurpose would run with: {payload}",
            platform=self.PLATFORM_NAME,
            payload=payload,
        )
        return {
            "success": True,
            "platform": self.PLATFORM_NAME,
            "simulated": True,
            "error": None,
        }

    def _build_error_response(
        self,
        error: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a standardized failure response dict."""
        response: Dict[str, Any] = {
            "success": False,
            "error": error,
        }
        if extra:
            response.update(extra)
        return response

    def guard_missing_config(self, *keys: str) -> List[str]:
        """
        Return list of missing required config keys.

        Returns an empty list when all keys are present and non-empty.
        Use this to validate config in __init__ or authenticate().
        """
        missing: List[str] = []
        for key in keys:
            value = self._config.get(key)
            if not value or (isinstance(value, str) and not value.strip()):
                logger.warning(
                    "Missing required config key for {platform}: {key}",
                    platform=self.PLATFORM_NAME,
                    key=key,
                )
                missing.append(key)
        return missing

    def guard_unconfigured_env(self, *env_vars: str) -> bool:
        """
        Return True if any required env var is missing or empty.

        Pattern matches BasePoster.guard_unconfigured() for consistency.
        """
        for var in env_vars:
            value = os.environ.get(var, "").strip()
            if not value:
                logger.warning(
                    "Missing required env var for {platform}: {var}",
                    platform=self.PLATFORM_NAME,
                    var=var,
                )
                return True
        return False
