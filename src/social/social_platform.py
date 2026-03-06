"""
SocialPlatform — abstract base class for all social platform posters.

Defines the interface that every poster must implement.
Platform-specific posters inherit: class MyPoster(SocialPlatform, BasePoster).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class SocialPlatform(ABC):
    """
    Abstract base defining the minimum interface for a social platform poster.

    The authenticate() method must be called before any post method.
    Every post method returns a standardized dict:
        {
            "success": bool,
            "post_id": str | None,
            "platform": str,
            "url": str | None,
            "error": str | None,
        }
    """

    @abstractmethod
    async def authenticate(self) -> None:
        """
        Authenticate with the platform using credentials from env vars.

        Raises:
            AuthError: on invalid credentials, missing env vars, or 401 response.
        """
        ...

    @abstractmethod
    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Return current rate limit state from the platform.

        Return shape:
            {
                "requests_remaining": int,
                "reset_at": str,   # ISO 8601
                "window_seconds": int,
            }
        """
        ...
