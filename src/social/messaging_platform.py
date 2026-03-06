"""
MessagingPlatform — abstract base class for all messaging channel posters.

Messaging platforms differ from social platforms in that they deliver
content to individual users or groups via direct messages rather than
public feeds. Every messaging poster inherits from this class and BasePoster.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class MessagingPlatform(ABC):
  """
  Abstract base defining the minimum interface for a messaging channel poster.

  authenticate() must be called before any send method.
  Every send method returns a standardized dict:
      {
          "success": bool,
          "message_id": str | None,
          "platform": str,
          "error": str | None,
      }
  """

  @abstractmethod
  async def authenticate(self) -> None:
    """
    Authenticate with the messaging platform using credentials from env vars.

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
            "reset_at": str,   # ISO 8601 or descriptive string
            "window_seconds": int,
        }
    """
    ...
