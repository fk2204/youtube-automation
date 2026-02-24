"""
API key authentication for Content Empire API.

Open Claw Bot authenticates via X-API-Key header.
Key is stored in .env as CONTENT_API_KEY.
"""

import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from loguru import logger

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> str:
    """Get configured API key from environment."""
    key = os.environ.get("CONTENT_API_KEY", "")
    if not key:
        logger.warning(
            "CONTENT_API_KEY not set. Generating ephemeral key for this session. "
            "Set CONTENT_API_KEY in .env for persistent authentication."
        )
        key = secrets.token_urlsafe(32)
        os.environ["CONTENT_API_KEY"] = key
        logger.info(f"Ephemeral API key: {key}")
    return key


async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """FastAPI dependency that verifies the API key.

    Raises:
        HTTPException: 401 if key is missing, 403 if key is invalid.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    expected = get_api_key()
    if not secrets.compare_digest(api_key, expected):
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key
