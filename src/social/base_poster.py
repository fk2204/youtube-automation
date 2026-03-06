"""
BasePoster — shared mixin for all social platform posters.

Provides common utility methods so individual poster classes
stay focused on platform-specific logic only.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from loguru import logger


class BasePoster:
    """
    Mixin providing shared utilities for every social poster.

    Platform-specific posters inherit both SocialPlatform and BasePoster.
    BasePoster must appear after SocialPlatform in the MRO so that
    SocialPlatform.__init__ runs correctly.
    """

    # Subclasses set this to identify themselves in logs and responses.
    platform_name: str = "unknown"

    def guard_unconfigured(self, *env_vars: str) -> bool:
        """
        Return True if any required env var is missing or empty.

        Use this at the start of authenticate() to short-circuit
        when credentials are absent. Tests can leave env vars unset
        and check for the expected AuthError.

        Example:
            if self.guard_unconfigured("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET"):
                raise AuthError("TikTok credentials not configured")
        """
        for var in env_vars:
            value = os.environ.get(var, "").strip()
            if not value:
                logger.warning(
                    "Missing required env var for {platform}: {var}",
                    platform=self.platform_name,
                    var=var,
                )
                return True
        return False

    def _simulate_post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a simulated success response without making any API call.

        Used in simulation mode when SOCIAL_SIMULATION_MODE=true.
        Enables full workflow testing without real credentials.
        """
        logger.info(
            "[SIMULATION] {platform} would post: {payload}",
            platform=self.platform_name,
            payload=payload,
        )
        return {
            "success": True,
            "post_id": f"sim_{self.platform_name}_12345",
            "platform": self.platform_name,
            "url": f"https://{self.platform_name}.com/sim/post/12345",
            "error": None,
            "simulated": True,
        }

    async def _execute_with_library(
        self,
        operation: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an async operation with standardized error logging.

        Wraps the call so every external library interaction logs on
        failure before re-raising. Individual methods still decide
        whether to catch and return or let the exception propagate.
        """
        try:
            return await operation(*args, **kwargs)
        except Exception as exc:
            logger.error(
                "{platform} library operation failed: {exc}",
                platform=self.platform_name,
                exc=exc,
            )
            raise

    @property
    def simulation_mode(self) -> bool:
        """True when SOCIAL_SIMULATION_MODE env var is set to 'true'."""
        return os.environ.get("SOCIAL_SIMULATION_MODE", "").lower() == "true"

    def _build_error_response(self, error: str) -> Dict[str, Any]:
        """Return a standardized failure response dict."""
        return {
            "success": False,
            "post_id": None,
            "platform": self.platform_name,
            "url": None,
            "error": error,
        }

    def _build_success_response(
        self,
        post_id: str,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a standardized success response dict."""
        return {
            "success": True,
            "post_id": post_id,
            "platform": self.platform_name,
            "url": url,
            "error": None,
        }
