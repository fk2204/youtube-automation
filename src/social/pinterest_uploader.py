"""
Pinterest pin upload via Playwright browser automation.

Creates rich pins with:
- Video previews
- Image pins (1000x1500 optimal)
- Board organization by niche
- SEO-rich descriptions

Pinterest pins have 6+ month lifespan — best for evergreen traffic.
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

SESSION_DIR = Path("config/browser_sessions/pinterest")
SCREENSHOT_DIR = Path("output/screenshots/pinterest")


class PinterestPlatform(ContentPlatform):
    """Pinterest pin creation via browser automation."""

    def __init__(self, session_dir: Optional[Path] = None, headless: bool = True):
        self._session_dir = session_dir or SESSION_DIR
        self._headless = headless
        self._session_dir.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Create a video pin on Pinterest."""
        return await self._create_pin(video_path, metadata, is_video=True)

    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Create an image pin on Pinterest."""
        return await self._create_pin(image_path, metadata, is_video=False)

    async def _create_pin(self, file_path: str, metadata: PlatformMetadata, is_video: bool) -> UploadResult:
        if not Path(file_path).exists():
            return UploadResult(platform="pinterest", status=UploadStatus.FAILED, error=f"File not found: {file_path}")

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return UploadResult(platform="pinterest", status=UploadStatus.FAILED, error="Playwright not installed")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir=str(self._session_dir),
                    headless=self._headless,
                    viewport={"width": 1280, "height": 900},
                )
                page = await browser.new_page()

                await page.goto("https://www.pinterest.com/pin-creation-tool/", wait_until="networkidle", timeout=30000)
                await asyncio.sleep(random.uniform(2, 4))

                if await self._check_login_required(page):
                    await browser.close()
                    return UploadResult(platform="pinterest", status=UploadStatus.AUTH_REQUIRED, error="Login required")

                # Upload file
                file_input = await page.wait_for_selector('input[type="file"]', timeout=10000)
                if file_input:
                    await file_input.set_input_files(file_path)
                    await asyncio.sleep(random.uniform(3, 8))

                # Fill title
                title_input = await page.query_selector('input[placeholder*="title"]') or await page.query_selector('[data-test-id="pin-draft-title"]')
                if title_input:
                    await title_input.fill(metadata.title[:100])
                    await asyncio.sleep(random.uniform(0.5, 1.5))

                # Fill description (SEO-rich for Pinterest search)
                desc_input = await page.query_selector('textarea') or await page.query_selector('[data-test-id="pin-draft-description"]')
                if desc_input:
                    desc = metadata.description
                    if metadata.hashtags:
                        desc += " " + " ".join(f"#{h}" for h in metadata.hashtags[:10])
                    await desc_input.fill(desc[:500])
                    await asyncio.sleep(random.uniform(0.5, 1.5))

                # Select board
                board_name = metadata.platform_specific.get("board")
                if board_name:
                    board_select = await page.query_selector('[data-test-id="board-dropdown"]') or await page.query_selector('button:has-text("Select a board")')
                    if board_select:
                        await board_select.click()
                        await asyncio.sleep(1)
                        board_option = await page.query_selector(f'div:has-text("{board_name}")')
                        if board_option:
                            await board_option.click()
                            await asyncio.sleep(1)

                # Publish
                publish_btn = await page.query_selector('button:has-text("Publish")') or await page.query_selector('[data-test-id="board-dropdown-save-button"]')
                if publish_btn:
                    await publish_btn.click()
                    await asyncio.sleep(random.uniform(3, 6))

                await page.screenshot(path=str(SCREENSHOT_DIR / f"pin_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"))
                await browser.close()

                return UploadResult(platform="pinterest", status=UploadStatus.SUCCESS, uploaded_at=datetime.utcnow())

        except Exception as e:
            logger.error(f"Pinterest pin creation failed: {e}")
            return UploadResult(platform="pinterest", status=UploadStatus.FAILED, error=str(e))

    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        return UploadResult(platform="pinterest", status=UploadStatus.FAILED, error="Pinterest requires image/video content")

    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        # Pinterest: SEO-rich descriptions (500 char), hashtags in description
        hashtags = (base.hashtags or base.tags)[:10]
        desc = base.description[:400] if base.description else base.title
        if hashtags:
            desc += " " + " ".join(f"#{h}" for h in hashtags)

        board_map = {
            "finance": "Money & Investing Tips",
            "psychology": "Self Improvement",
            "storytelling": "Amazing Stories",
        }

        return PlatformMetadata(
            title=base.title[:100],
            description=desc[:500],
            hashtags=hashtags,
            platform_specific={"board": board_map.get(base.niche, "General")},
        )

    def get_platform_name(self) -> str:
        return "pinterest"

    def is_configured(self) -> bool:
        return self._session_dir.exists() and any(self._session_dir.iterdir()) if self._session_dir.exists() else False

    async def _check_login_required(self, page) -> bool:
        login = await page.query_selector('input[name="id"]') or await page.query_selector('button:has-text("Log in")')
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
            await page.goto("https://www.pinterest.com/login/")
            logger.info("Log in to Pinterest manually...")
            await asyncio.sleep(300)
            await browser.close()
        return True
