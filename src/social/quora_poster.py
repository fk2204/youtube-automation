"""
Quora answer posting via Playwright browser automation.

Quora has NO public API for content publishing.
Browser automation is the only programmatic option.

Strategy: Post answers to relevant questions in the niche
to drive traffic back to YouTube/blog content.
"""

import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.social.platform_adapter import (
    ContentPlatform,
    ContentMetadata,
    PlatformMetadata,
    UploadResult,
    UploadStatus,
)

SESSION_DIR = Path("config/browser_sessions/quora")
SCREENSHOT_DIR = Path("output/screenshots/quora")


class QuoraPlatform(ContentPlatform):
    """Quora answer posting via browser automation."""

    def __init__(self, session_dir: Optional[Path] = None, headless: bool = True):
        self._session_dir = session_dir or SESSION_DIR
        self._headless = headless
        self._session_dir.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        return UploadResult(platform="quora", status=UploadStatus.FAILED, error="Quora does not support direct video uploads")

    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        return UploadResult(platform="quora", status=UploadStatus.FAILED, error="Use post_text with embedded image references")

    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        """Post an answer on Quora via browser automation."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return UploadResult(platform="quora", status=UploadStatus.FAILED, error="Playwright not installed")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir=str(self._session_dir),
                    headless=self._headless,
                    viewport={"width": 1280, "height": 900},
                )
                page = await browser.new_page()

                # Search for a relevant question
                search_query = metadata.title[:80]
                await page.goto(f"https://www.quora.com/search?q={search_query}", wait_until="networkidle", timeout=30000)
                await asyncio.sleep(random.uniform(2, 4))

                if await self._check_login_required(page):
                    await browser.close()
                    return UploadResult(platform="quora", status=UploadStatus.AUTH_REQUIRED, error="Login required")

                # Find first question link
                question_link = await page.query_selector('a[href*="/answer"]') or await page.query_selector('.q-text a')
                if question_link:
                    await question_link.click()
                    await asyncio.sleep(random.uniform(2, 4))

                # Click "Answer" button
                answer_btn = await page.query_selector('button:has-text("Answer")')
                if answer_btn:
                    await answer_btn.click()
                    await asyncio.sleep(random.uniform(1, 3))

                # Fill answer
                editor = await page.query_selector('[contenteditable="true"]') or await page.query_selector('.doc')
                if editor:
                    await editor.click()
                    # Type answer (first 2000 chars)
                    answer_text = content[:2000]
                    await page.keyboard.type(answer_text, delay=random.randint(10, 40))
                    await asyncio.sleep(random.uniform(1, 3))

                # Submit
                submit_btn = await page.query_selector('button:has-text("Post")')  or await page.query_selector('button:has-text("Submit")')
                if submit_btn:
                    await submit_btn.click()
                    await asyncio.sleep(random.uniform(3, 6))

                await page.screenshot(path=str(SCREENSHOT_DIR / f"answer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"))
                await browser.close()

                return UploadResult(platform="quora", status=UploadStatus.SUCCESS, uploaded_at=datetime.utcnow())

        except Exception as e:
            logger.error(f"Quora posting failed: {e}")
            return UploadResult(platform="quora", status=UploadStatus.FAILED, error=str(e))

    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        return PlatformMetadata(
            title=base.title[:150],
            description=base.description[:2000],
            tags=base.tags[:5],
        )

    def get_platform_name(self) -> str:
        return "quora"

    def is_configured(self) -> bool:
        return self._session_dir.exists() and any(self._session_dir.iterdir()) if self._session_dir.exists() else False

    async def _check_login_required(self, page) -> bool:
        login = await page.query_selector('input[placeholder*="email"]')
        return login is not None

    async def setup_session(self) -> bool:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return False

        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(self._session_dir), headless=False, viewport={"width": 1280, "height": 900},
            )
            page = await browser.new_page()
            await page.goto("https://www.quora.com/")
            logger.info("Log in to Quora manually...")
            await asyncio.sleep(300)
            await browser.close()
        return True
