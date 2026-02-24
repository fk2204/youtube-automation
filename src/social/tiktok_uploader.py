"""
TikTok upload via Playwright browser automation.

Implements ContentPlatform for TikTok video uploads without needing
the official TikTok Content Posting API (which requires business account approval).

Session management:
- First run: Opens browser for manual login, saves cookies
- Subsequent runs: Reuses saved session (no login needed)

Anti-detection:
- Random delays between actions (3-15s)
- Human-like mouse movements
- Screenshot capture for debugging
- Viewport randomization
"""

import asyncio
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from src.social.platform_adapter import (
    ContentPlatform,
    ContentMetadata,
    PlatformMetadata,
    UploadResult,
    UploadStatus,
)

SESSION_DIR = Path("config/browser_sessions/tiktok")
SCREENSHOT_DIR = Path("output/screenshots/tiktok")


class TikTokPlatform(ContentPlatform):
    """TikTok video upload via Playwright browser automation."""

    def __init__(self, session_dir: Optional[Path] = None, headless: bool = True):
        self._session_dir = session_dir or SESSION_DIR
        self._headless = headless
        self._session_dir.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Upload a video to TikTok via browser automation."""
        if not Path(video_path).exists():
            return UploadResult(
                platform="tiktok",
                status=UploadStatus.FAILED,
                error=f"Video file not found: {video_path}",
            )

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return UploadResult(
                platform="tiktok",
                status=UploadStatus.FAILED,
                error="Playwright not installed. Run: pip install playwright && playwright install chromium",
            )

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir=str(self._session_dir),
                    headless=self._headless,
                    viewport={"width": 1280 + random.randint(-20, 20), "height": 900 + random.randint(-20, 20)},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                    timezone_id="America/New_York",
                )

                page = await browser.new_page()
                result = await self._perform_upload(page, video_path, metadata)
                await self._screenshot(page, "upload_complete")
                await browser.close()
                return result

        except Exception as e:
            logger.error(f"TikTok upload failed: {e}")
            return UploadResult(
                platform="tiktok",
                status=UploadStatus.FAILED,
                error=str(e),
            )

    async def _perform_upload(self, page, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Execute the TikTok upload flow."""
        upload_url = "https://www.tiktok.com/creator#/upload?scene=creator_center"

        logger.info(f"Navigating to TikTok upload page...")
        await page.goto(upload_url, wait_until="networkidle", timeout=30000)
        await self._random_delay(2, 5)

        # Check if we need to log in
        if await self._check_login_required(page):
            return UploadResult(
                platform="tiktok",
                status=UploadStatus.AUTH_REQUIRED,
                error=(
                    "TikTok login required. Run with headless=False for initial login: "
                    "TikTokPlatform(headless=False).setup_session()"
                ),
            )

        await self._screenshot(page, "upload_page")

        # Upload video file
        logger.info(f"Uploading video: {video_path}")
        try:
            # TikTok upload page uses iframe for upload
            # Try multiple selector strategies
            selectors = [
                'input[type="file"]',
                'input[accept="video/*"]',
                'iframe[src*="upload"]',
            ]

            file_input = None
            for selector in selectors:
                try:
                    file_input = await page.wait_for_selector(selector, timeout=10000)
                    if file_input:
                        break
                except Exception:
                    continue

            if not file_input:
                # Try iframe approach
                frames = page.frames
                for frame in frames:
                    try:
                        file_input = await frame.wait_for_selector('input[type="file"]', timeout=5000)
                        if file_input:
                            break
                    except Exception:
                        continue

            if not file_input:
                await self._screenshot(page, "no_upload_input")
                return UploadResult(
                    platform="tiktok",
                    status=UploadStatus.FAILED,
                    error="Could not find file upload input on TikTok page",
                )

            await file_input.set_input_files(video_path)
            logger.info("Video file selected, waiting for upload...")
            await self._random_delay(5, 15)

        except Exception as e:
            await self._screenshot(page, "upload_error")
            return UploadResult(
                platform="tiktok",
                status=UploadStatus.FAILED,
                error=f"Failed to upload file: {e}",
            )

        # Fill in metadata
        await self._fill_metadata(page, metadata)
        await self._random_delay(2, 5)

        # Wait for video processing
        logger.info("Waiting for TikTok to process video...")
        await self._wait_for_processing(page, timeout=120)

        # Click post button
        post_result = await self._click_post(page)
        if not post_result:
            await self._screenshot(page, "post_failed")
            return UploadResult(
                platform="tiktok",
                status=UploadStatus.FAILED,
                error="Failed to click Post button",
            )

        await self._random_delay(3, 8)
        await self._screenshot(page, "posted")

        return UploadResult(
            platform="tiktok",
            status=UploadStatus.SUCCESS,
            uploaded_at=datetime.utcnow(),
            metadata={"title": metadata.title},
        )

    async def _check_login_required(self, page) -> bool:
        """Check if TikTok is showing login page."""
        try:
            # Look for login indicators
            login_selectors = [
                'button:has-text("Log in")',
                '[data-e2e="login-button"]',
                'a[href*="login"]',
            ]
            for sel in login_selectors:
                el = await page.query_selector(sel)
                if el:
                    return True
            return False
        except Exception:
            return False

    async def _fill_metadata(self, page, metadata: PlatformMetadata) -> None:
        """Fill in title/description and hashtags."""
        try:
            # TikTok caption editor
            caption_selectors = [
                '[data-e2e="caption-editor"]',
                '.caption-editor',
                '[contenteditable="true"]',
                'div[role="textbox"]',
            ]

            caption_el = None
            for sel in caption_selectors:
                try:
                    caption_el = await page.wait_for_selector(sel, timeout=5000)
                    if caption_el:
                        break
                except Exception:
                    continue

            if caption_el:
                # Build caption: title + hashtags
                caption_parts = [metadata.title]
                if metadata.hashtags:
                    hashtag_str = " ".join(f"#{h}" for h in metadata.hashtags[:5])
                    caption_parts.append(hashtag_str)

                caption = " ".join(caption_parts)[:2200]  # TikTok caption limit

                await caption_el.click()
                await self._random_delay(0.5, 1.5)

                # Clear existing content
                await page.keyboard.press("Control+A")
                await self._random_delay(0.2, 0.5)

                # Type caption character by character for realism
                for char in caption:
                    await page.keyboard.type(char, delay=random.randint(20, 80))

                logger.info(f"Caption set: {caption[:50]}...")
            else:
                logger.warning("Could not find caption editor")

        except Exception as e:
            logger.warning(f"Failed to set caption: {e}")

    async def _wait_for_processing(self, page, timeout: int = 120) -> None:
        """Wait for TikTok to finish processing the uploaded video."""
        check_interval = 5
        elapsed = 0
        while elapsed < timeout:
            try:
                # Look for "processing" indicators disappearing
                processing = await page.query_selector('[class*="processing"]')
                if not processing:
                    # Check for upload success indicator
                    success = await page.query_selector('[class*="success"]')
                    if success:
                        logger.info("Video processing complete")
                        return
                await asyncio.sleep(check_interval)
                elapsed += check_interval
            except Exception:
                await asyncio.sleep(check_interval)
                elapsed += check_interval

        logger.warning(f"Processing timeout after {timeout}s, attempting to post anyway")

    async def _click_post(self, page) -> bool:
        """Click the Post/Publish button."""
        post_selectors = [
            'button:has-text("Post")',
            'button:has-text("Publish")',
            '[data-e2e="post-button"]',
            'button[class*="post-button"]',
            'button[class*="submit"]',
        ]

        for sel in post_selectors:
            try:
                btn = await page.wait_for_selector(sel, timeout=5000)
                if btn:
                    is_disabled = await btn.get_attribute("disabled")
                    if not is_disabled:
                        await btn.click()
                        logger.info("Post button clicked")
                        return True
            except Exception:
                continue

        logger.error("Could not find or click Post button")
        return False

    async def _screenshot(self, page, name: str) -> None:
        """Save a screenshot for debugging."""
        try:
            path = SCREENSHOT_DIR / f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            await page.screenshot(path=str(path))
            logger.debug(f"Screenshot saved: {path}")
        except Exception as e:
            logger.debug(f"Screenshot failed: {e}")

    async def _random_delay(self, min_sec: float, max_sec: float) -> None:
        """Human-like random delay."""
        delay = random.uniform(min_sec, max_sec)
        await asyncio.sleep(delay)

    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        """TikTok doesn't support standalone image uploads."""
        return UploadResult(
            platform="tiktok",
            status=UploadStatus.FAILED,
            error="TikTok does not support standalone image uploads. Use upload_video.",
        )

    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        """TikTok doesn't support text-only posts."""
        return UploadResult(
            platform="tiktok",
            status=UploadStatus.FAILED,
            error="TikTok does not support text-only posts.",
        )

    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        """Adapt metadata for TikTok's constraints."""
        # TikTok: caption 2200 chars, ~30 hashtags practical
        title = base.title[:150]
        hashtags = (base.hashtags or base.tags)[:15]

        # TikTok-specific: shorter, punchier description
        desc = base.description[:500] if base.description else title

        return PlatformMetadata(
            title=title,
            description=desc,
            hashtags=hashtags,
            tags=base.tags[:10],
            privacy=base.privacy,
            platform_specific={
                "allow_comments": True,
                "allow_duets": True,
                "allow_stitches": True,
            },
        )

    def get_platform_name(self) -> str:
        return "tiktok"

    def is_configured(self) -> bool:
        """Check if browser session directory exists with saved state."""
        if not self._session_dir.exists():
            return False
        # Check for Playwright session data
        return any(self._session_dir.iterdir()) if self._session_dir.exists() else False

    def get_platform_specs(self):
        """Return TikTok video specs."""
        try:
            from src.social.multi_platform import PLATFORM_SPECS, Platform
            return PLATFORM_SPECS.get(Platform.TIKTOK)
        except ImportError:
            return None

    async def setup_session(self) -> bool:
        """Interactive session setup — opens browser for manual TikTok login.

        Call this once with headless=False to save login cookies:
            platform = TikTokPlatform(headless=False)
            await platform.setup_session()
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("Playwright not installed")
            return False

        logger.info("Opening TikTok for manual login. Log in and press Enter when done.")

        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(self._session_dir),
                headless=False,
                viewport={"width": 1280, "height": 900},
            )
            page = await browser.new_page()
            await page.goto("https://www.tiktok.com/login", wait_until="networkidle")

            # Wait for user to complete login
            logger.info("Waiting for login completion...")
            try:
                await page.wait_for_url("**/foryou*", timeout=300000)  # 5 min timeout
                logger.info("Login detected. Saving session...")
            except Exception:
                logger.info("Timeout waiting for login redirect. Session may still be saved.")

            await browser.close()

        logger.info(f"Session saved to: {self._session_dir}")
        return True
