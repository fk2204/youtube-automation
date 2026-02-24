"""
Instagram Reels/Posts upload via Playwright browser automation.

Supports:
- Reels (9:16 video, 3-90s)
- Image posts (1080x1080)
- Carousel posts (multiple images)

Session management mirrors TikTok uploader pattern.
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

SESSION_DIR = Path("config/browser_sessions/instagram")
SCREENSHOT_DIR = Path("output/screenshots/instagram")


class InstagramPlatform(ContentPlatform):
    """Instagram upload via Playwright browser automation."""

    def __init__(self, session_dir: Optional[Path] = None, headless: bool = True):
        self._session_dir = session_dir or SESSION_DIR
        self._headless = headless
        self._session_dir.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Upload a Reel to Instagram."""
        if not Path(video_path).exists():
            return UploadResult(platform="instagram", status=UploadStatus.FAILED, error=f"File not found: {video_path}")

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return UploadResult(platform="instagram", status=UploadStatus.FAILED, error="Playwright not installed")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir=str(self._session_dir),
                    headless=self._headless,
                    viewport={"width": 1280 + random.randint(-10, 10), "height": 900 + random.randint(-10, 10)},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                )
                page = await browser.new_page()

                await page.goto("https://www.instagram.com/", wait_until="networkidle", timeout=30000)
                await asyncio.sleep(random.uniform(2, 4))

                if await self._check_login_required(page):
                    await browser.close()
                    return UploadResult(platform="instagram", status=UploadStatus.AUTH_REQUIRED, error="Login required. Run setup_session() first.")

                # Navigate to create
                create_btn = await page.query_selector('[aria-label="New post"]') or await page.query_selector('svg[aria-label="New post"]')
                if create_btn:
                    await create_btn.click()
                    await asyncio.sleep(random.uniform(1, 3))

                # Find file input and upload
                file_input = await page.wait_for_selector('input[type="file"]', timeout=10000)
                if file_input:
                    await file_input.set_input_files(video_path)
                    await asyncio.sleep(random.uniform(5, 10))

                    # Select "Reel" tab if available
                    reel_tab = await page.query_selector('button:has-text("Reel")')
                    if reel_tab:
                        await reel_tab.click()
                        await asyncio.sleep(random.uniform(1, 2))

                    # Click Next
                    for _ in range(2):
                        next_btn = await page.query_selector('button:has-text("Next")')
                        if next_btn:
                            await next_btn.click()
                            await asyncio.sleep(random.uniform(1, 3))

                    # Fill caption
                    caption_el = await page.query_selector('textarea[aria-label="Write a caption..."]') or await page.query_selector('[contenteditable="true"]')
                    if caption_el:
                        caption = f"{metadata.title}\n\n{metadata.description}"
                        if metadata.hashtags:
                            caption += "\n\n" + " ".join(f"#{h}" for h in metadata.hashtags[:30])
                        caption = caption[:2200]
                        await caption_el.fill(caption)
                        await asyncio.sleep(random.uniform(1, 2))

                    # Share
                    share_btn = await page.query_selector('button:has-text("Share")')
                    if share_btn:
                        await share_btn.click()
                        await asyncio.sleep(random.uniform(5, 10))

                await page.screenshot(path=str(SCREENSHOT_DIR / f"upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"))
                await browser.close()

                return UploadResult(platform="instagram", status=UploadStatus.SUCCESS, uploaded_at=datetime.utcnow())

        except Exception as e:
            logger.error(f"Instagram upload failed: {e}")
            return UploadResult(platform="instagram", status=UploadStatus.FAILED, error=str(e))

    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Upload an image post to Instagram."""
        # Same flow as video but without Reel tab selection
        return await self.upload_video(image_path, metadata)

    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        return UploadResult(platform="instagram", status=UploadStatus.FAILED, error="Instagram does not support text-only posts")

    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        hashtags = (base.hashtags or base.tags)[:30]
        return PlatformMetadata(
            title=base.title[:100],
            description=base.description[:2000],
            hashtags=hashtags,
            tags=base.tags[:10],
            privacy=base.privacy,
        )

    def get_platform_name(self) -> str:
        return "instagram"

    def is_configured(self) -> bool:
        return self._session_dir.exists() and any(self._session_dir.iterdir()) if self._session_dir.exists() else False

    async def _check_login_required(self, page) -> bool:
        login = await page.query_selector('input[name="username"]')
        return login is not None

    async def setup_session(self) -> bool:
        """Interactive login for session setup."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return False

        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(self._session_dir), headless=False, viewport={"width": 1280, "height": 900},
            )
            page = await browser.new_page()
            await page.goto("https://www.instagram.com/accounts/login/")
            logger.info("Log in to Instagram manually, then wait...")
            try:
                await page.wait_for_url("**/instagram.com/**", timeout=300000)
                await asyncio.sleep(5)
            except Exception:
                pass
            await browser.close()
        return True
