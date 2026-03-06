"""
Base classes and mixins for social media platform posters.

Consolidates common poster logic:
- Unconfigured platform guards
- Simulation mode helpers
- Library availability checks
- Standard error handling
"""

from typing import Dict, Any, Optional, Callable, Tuple
from loguru import logger


class BasePoster:
    """
    Mixin class providing shared poster functionality for all social platforms.

    Subclasses should:
    1. Define self.platform_name (e.g., "Twitter")
    2. Implement is_configured() -> bool
    3. Call guard_unconfigured() at top of post()
    4. Use _execute_with_library() to wrap API calls
    """

    platform_name: str = "Unknown"

    def guard_unconfigured(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Return a safe "not configured" response if platform is not set up.

        Args:
            content: The content being posted (for preview logging)

        Returns:
            Dict with error info if not configured, None if configured.
        """
        if not self.is_configured():
            preview = content[:50].replace("\n", " ")
            logger.warning(f"[{self.platform_name}] Not configured - would post: {preview}...")
            return {
                "success": False,
                "error": f"{self.platform_name} not configured",
                "platform": self.platform_name.lower(),
                "simulated": True,
                "content_preview": content[:100],
            }
        return None

    def _simulate_post(
        self,
        content: str,
        url: Optional[str] = None,
        image: Optional[str] = None,
        **extra_fields
    ) -> Dict[str, Any]:
        """
        Generate a simulated post response (for testing/demo mode).

        Args:
            content: The posted content
            url: Optional URL in the post
            image: Optional image path
            **extra_fields: Platform-specific fields (e.g., subreddit for Reddit)

        Returns:
            Simulated post response dict
        """
        import time
        post_id = f"sim_{self.platform_name.lower()}_{int(time.time())}"

        result = {
            "success": True,
            "post_id": post_id,
            "platform": self.platform_name.lower(),
            "simulated": True,
            "content_preview": content[:100],
        }

        if url:
            result["url"] = url
        if image:
            result["image_path"] = image

        result.update(extra_fields)

        logger.debug(f"[{self.platform_name}] Simulated post: {post_id}")
        return result

    def _execute_with_library(
        self,
        lib_name: str,
        fn: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute a function that requires an external library, handling import errors gracefully.

        Args:
            lib_name: Name of the required library (for logging)
            fn: Callable that performs the actual post operation

        Returns:
            Success dict from fn, or simulated post if library unavailable.
        """
        try:
            return fn()
        except ImportError:
            logger.warning(f"[{self.platform_name}] {lib_name} not installed - using simulation mode")
            # Return a simulated response with basic content (caller should provide this context)
            return {
                "success": False,
                "error": f"{lib_name} not installed",
                "platform": self.platform_name.lower(),
                "simulated": True,
            }
        except Exception as e:
            logger.error(f"[{self.platform_name}] Failed to post: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "platform": self.platform_name.lower(),
            }

    def is_configured(self) -> bool:
        """Check if this platform is properly configured. Must be implemented by subclass."""
        raise NotImplementedError(f"{self.platform_name} must implement is_configured()")
