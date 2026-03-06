"""
MediumRepurposer — publishes articles to Medium via the Integration Token API.

NOTE: The Medium API (https://api.medium.com/v1/) is officially unmaintained
as of 2023 but remains functional for existing integration tokens. New tokens
can still be generated from Medium account settings. Monitor Medium's
developer announcements for deprecation notices before relying on this in
production workflows.

API flow:
  1. GET /v1/me  — verify token and fetch authorId.
  2. (Optional) GET /v1/users/{authorId}/publications — list publications.
  3. POST /v1/users/{authorId}/posts  — publish to user profile, OR
     POST /v1/publications/{publicationId}/posts — publish to a publication.

Rate limits:
  - Not officially documented. In practice: ~100 requests/day per token.
  - Use exponential backoff on 429 responses.

Credentials (env vars):
    MEDIUM_INTEGRATION_TOKEN  — Generated at medium.com/me/settings → Integration Token
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from distribution.repurposers.base_repurposer import BaseRepurposer
from distribution.repurposers.repurposer_registry import RepurposerRegistry
from exceptions import AuthError, RateLimitError

MEDIUM_API_BASE = "https://api.medium.com/v1"

# Maximum tags Medium accepts per post.
MAX_TAGS = 5


class MediumRepurposer(BaseRepurposer):
    """
    Publishes articles to Medium using the Integration Token API.

    The API is unmaintained but functional — see module docstring.

    Requires env var:
        MEDIUM_INTEGRATION_TOKEN  — Medium Integration Token
    """

    PLATFORM_NAME = "medium"
    AUTOMATION_LEVEL = "full"

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with config dict.

        Config keys (optional):
            default_status:       "draft" | "public" | "unlisted" (default: "draft")
            default_publication_id: str | None
        """
        super().__init__(config)
        self._token: Optional[str] = None
        self._author_id: Optional[str] = None

    @property
    def platform_name(self) -> str:
        return self.PLATFORM_NAME

    async def authenticate(self) -> bool:
        """
        Load the integration token and verify it by calling GET /v1/me.

        Returns:
            True on success, False on failure (never raises).
        """
        if self.guard_unconfigured_env("MEDIUM_INTEGRATION_TOKEN"):
            logger.error("MEDIUM_INTEGRATION_TOKEN env var not set.")
            return False

        token = os.environ["MEDIUM_INTEGRATION_TOKEN"]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MEDIUM_API_BASE}/me",
                    headers=self._auth_headers(token),
                    timeout=10.0,
                )
            if response.status_code == 401:
                logger.error("Medium token rejected (401). Check MEDIUM_INTEGRATION_TOKEN.")
                return False
            if response.status_code == 429:
                logger.error("Medium rate limit hit during auth check.")
                return False
            response.raise_for_status()
        except Exception as exc:
            logger.error("Medium auth validation failed: {exc}", exc=exc)
            return False

        data = response.json().get("data", {})
        self._token = token
        self._author_id = data.get("id")
        self._authenticated = True
        logger.info(
            "Medium authenticated: author_id={aid}",
            aid=self._author_id,
        )
        return True

    async def repurpose(
        self,
        source_content: str,
        publication_id: Optional[str] = None,
        status: str = "draft",
        tags: Optional[List[str]] = None,
        canonical_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Publish source content as a Medium article.

        Args:
            source_content:   Article body in Markdown or HTML.
            publication_id:   Optional Medium publication ID. When provided,
                              posts to the publication instead of user profile.
            status:           "draft" | "public" | "unlisted". Default: "draft".
            tags:             List of tags. Capped at MAX_TAGS (5). Extra tags dropped.
            canonical_url:    Optional canonical URL for SEO cross-posting.

        Returns:
            {
                "success": bool,
                "post_id": str | None,
                "url": str | None,
                "status": str,
                "error": str | None,
            }
        """
        if self.simulation_mode:
            sim = self._simulate_repurpose({
                "content_length": len(source_content),
                "publication_id": publication_id,
                "status": status,
            })
            sim.update({"post_id": "sim_medium_123", "url": "https://medium.com/sim/123", "status": status})
            return sim

        if not self._authenticated:
            return self._build_error_response(
                "Not authenticated. Call authenticate() first.",
                extra={"post_id": None, "url": None, "status": "failed"},
            )

        safe_tags = (tags or [])[:MAX_TAGS]
        if tags and len(tags) > MAX_TAGS:
            logger.warning(
                "Medium: {n} tags provided, capped at {max}.",
                n=len(tags),
                max=MAX_TAGS,
            )

        payload = self._transform_to_medium_format(
            content=source_content,
            status=status,
            tags=safe_tags,
            canonical_url=canonical_url,
        )

        endpoint = self._resolve_endpoint(publication_id)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    headers={**self._auth_headers(self._token), "Content-Type": "application/json"},
                    json=payload,
                    timeout=30.0,
                )
            if response.status_code == 429:
                raise RateLimitError("Medium rate limit hit during post.")
            response.raise_for_status()
        except RateLimitError as exc:
            return self._build_error_response(
                str(exc),
                extra={"post_id": None, "url": None, "status": "rate_limited"},
            )
        except Exception as exc:
            logger.error("Medium repurpose failed: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={"post_id": None, "url": None, "status": "failed"},
            )

        data = response.json().get("data", {})
        post_id = str(data.get("id", ""))
        url = data.get("url")
        logger.info("Medium post created: post_id={pid}, url={url}", pid=post_id, url=url)

        return {
            "success": True,
            "post_id": post_id,
            "url": url,
            "status": data.get("publishStatus", status),
            "error": None,
        }

    def _transform_to_medium_format(
        self,
        content: str,
        status: str,
        tags: List[str],
        canonical_url: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build the Medium API post payload from source content.

        Medium accepts "html" or "markdown" as contentFormat.
        We detect HTML by checking for '<' characters; otherwise use markdown.
        """
        content_format = "html" if content.strip().startswith("<") else "markdown"

        payload: Dict[str, Any] = {
            "title": self._extract_title(content),
            "contentFormat": content_format,
            "content": content,
            "publishStatus": status,
            "tags": tags,
        }
        if canonical_url:
            payload["canonicalUrl"] = canonical_url

        return payload

    async def get_publications(self) -> List[Dict[str, Any]]:
        """
        Fetch the list of publications the authenticated user can post to.

        Returns:
            List of publication dicts from the Medium API.
            Returns empty list on error (does not raise).
        """
        if not self._authenticated:
            logger.error("Medium: cannot fetch publications — not authenticated.")
            return []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MEDIUM_API_BASE}/users/{self._author_id}/publications",
                    headers=self._auth_headers(self._token),
                    timeout=15.0,
                )
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as exc:
            logger.error("Medium get_publications failed: {exc}", exc=exc)
            return []

    def _resolve_endpoint(self, publication_id: Optional[str]) -> str:
        """Return the correct API endpoint based on whether a publication is targeted."""
        if publication_id:
            return f"{MEDIUM_API_BASE}/publications/{publication_id}/posts"
        return f"{MEDIUM_API_BASE}/users/{self._author_id}/posts"

    @staticmethod
    def _auth_headers(token: Optional[str]) -> Dict[str, str]:
        """Return Authorization header for Medium API."""
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _extract_title(content: str) -> str:
        """
        Extract a title from the first heading or first line of content.

        Falls back to "Untitled" when content is empty.
        """
        lines = content.strip().splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            if stripped:
                return stripped[:100]
        return "Untitled"


def _register() -> None:
    """Register MediumRepurposer with the global RepurposerRegistry."""
    RepurposerRegistry.register("medium", MediumRepurposer)
    logger.debug("MediumRepurposer registered.")


_register()
