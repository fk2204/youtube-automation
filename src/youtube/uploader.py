"""
YouTube Video Uploader Module

Uploads videos to YouTube with metadata, thumbnails, and scheduling.
Includes YouTube SEO best practices for algorithm virality.

Usage:
    uploader = YouTubeUploader()
    video_id = uploader.upload_video(
        video_file="output/video.mp4",
        title="My Tutorial",
        description="Learn something new!",
        tags=["tutorial", "python"]
    )
"""

import os
import time
import random
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger

try:
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    raise ImportError("Please install google-api-python-client")

from .auth import YouTubeAuth

# Import YouTube optimization utilities
try:
    from src.utils.youtube_optimizer import (
        UploadTimingOptimizer,
        FirstHourBooster,
        TitlePatternAnalyzer,
        generate_chapters_from_script,
        optimize_description_keywords,
    )
    YOUTUBE_OPTIMIZER_AVAILABLE = True
except ImportError:
    YOUTUBE_OPTIMIZER_AVAILABLE = False
    logger.debug("youtube_optimizer module not available - advanced optimizations disabled")

# Import AI Disclosure Tracker
try:
    from src.compliance.ai_disclosure import AIDisclosureTracker, DisclosureMetadata
    AI_DISCLOSURE_AVAILABLE = True
except ImportError:
    AI_DISCLOSURE_AVAILABLE = False
    logger.debug("AI disclosure module not available")

# Import Metadata Optimizer
try:
    from src.seo.metadata_optimizer import MetadataOptimizer, OptimizedMetadata
    METADATA_OPTIMIZER_AVAILABLE = True
except ImportError:
    METADATA_OPTIMIZER_AVAILABLE = False
    logger.debug("Metadata optimizer module not available")


# ============================================================
# SEO OPTIMIZATION CONSTANTS
# ============================================================

# Niche-specific hashtag sets for maximum discoverability
NICHE_HASHTAGS = {
    "finance": ["#finance", "#money", "#investing", "#passiveincome", "#wealthbuilding"],
    "psychology": ["#psychology", "#mindset", "#selfimprovement", "#humanmind", "#darkpsychology"],
    "storytelling": ["#truecrime", "#documentary", "#mystery", "#truestory", "#untoldstories"],
    "programming": ["#coding", "#programming", "#developer", "#tech", "#tutorial"],
    "education": ["#education", "#learning", "#howto", "#tutorial", "#explained"],
    "default": ["#viral", "#trending", "#mustwatch", "#facts", "#knowledge"]
}

# Trending tags by niche for SEO
NICHE_TRENDING_TAGS = {
    "finance": [
        "passive income 2026", "how to make money", "financial freedom",
        "investing for beginners", "side hustle ideas", "money tips",
        "stock market", "wealth building", "personal finance", "budgeting",
        "crypto", "dividend investing", "real estate investing", "retire early"
    ],
    "psychology": [
        "psychology facts", "dark psychology", "manipulation tactics",
        "body language", "cognitive bias", "human behavior", "mind tricks",
        "self improvement", "stoicism", "narcissist signs", "emotional intelligence",
        "subconscious mind", "persuasion techniques", "mental health"
    ],
    "storytelling": [
        "true story", "documentary", "unsolved mysteries", "true crime",
        "what happened to", "rise and fall", "company documentary",
        "business story", "scandal", "untold story", "investigation",
        "mystery explained", "crime documentary", "horror story"
    ],
    "default": [
        "explained", "how to", "tutorial", "guide", "tips and tricks",
        "facts", "interesting facts", "did you know", "education"
    ]
}

# Category mapping optimized by niche
NICHE_CATEGORIES = {
    "finance": "22",        # People & Blogs (performs better than Education for finance)
    "psychology": "27",     # Education
    "storytelling": "24",   # Entertainment
    "programming": "28",    # Science & Technology
    "education": "27",      # Education
    "default": "27"         # Education
}


@dataclass
class UploadResult:
    """Result of a video upload."""
    success: bool
    video_id: Optional[str]
    video_url: Optional[str]
    error: Optional[str]


@dataclass
class VideoSEOMetadata:
    """SEO-optimized metadata for YouTube upload."""
    title: str
    description: str
    tags: List[str]
    category_id: str
    hashtags: List[str] = field(default_factory=list)
    chapters: str = ""  # Timestamp chapters string

    def get_full_description(self) -> str:
        """Build the full SEO-optimized description."""
        return self.description


class YouTubeSEOOptimizer:
    """
    YouTube SEO Optimization Helper.

    Implements best practices for:
    - Description optimization (hook + keywords in first 2 lines)
    - Timestamps/chapters for navigation
    - Hashtags (3-5 relevant ones)
    - Tags optimization (trending + niche keywords)
    - Category optimization per niche
    """

    def __init__(self, niche: str = "default"):
        self.niche = niche.lower()
        self.current_year = str(datetime.now().year)

    def build_seo_description(
        self,
        hook: str,
        main_description: str,
        chapters: Optional[List[Dict[str, Any]]] = None,
        related_videos: Optional[List[str]] = None,
        channel_name: str = ""
    ) -> str:
        """
        Build an SEO-optimized YouTube description.

        Structure:
        1. Hook + keywords (first 2 lines - shown in search)
        2. Main description
        3. Timestamps/chapters
        4. Call-to-action
        5. Related video links
        6. Hashtags (3-5)

        Args:
            hook: Attention-grabbing first line with keywords
            main_description: Main body of description
            chapters: List of {"timestamp": "00:00", "title": "Chapter Name"}
            related_videos: List of video URLs to link
            channel_name: Channel name for CTA

        Returns:
            Full SEO-optimized description
        """
        parts = []

        # 1. HOOK + KEYWORDS (First 2 lines - CRITICAL for search)
        # YouTube shows first ~100 chars in search results
        hook = hook.strip()
        if not hook.endswith(('.', '!', '?')):
            hook += '.'
        parts.append(hook)
        parts.append("")  # Blank line

        # 2. MAIN DESCRIPTION
        if main_description:
            parts.append(main_description.strip())
            parts.append("")

        # 3. TIMESTAMPS/CHAPTERS (Improves navigation and watch time)
        if chapters and len(chapters) >= 3:
            parts.append("=" * 30)
            parts.append("TIMESTAMPS:")
            parts.append("=" * 30)
            for chapter in chapters:
                timestamp = chapter.get("timestamp", "00:00")
                title = chapter.get("title", "")
                parts.append(f"{timestamp} - {title}")
            parts.append("")

        # 4. CALL-TO-ACTION (Engagement signals help algorithm)
        parts.append("-" * 30)
        parts.append("SUBSCRIBE for more content!")
        parts.append("LIKE this video if you found it valuable!")
        parts.append("COMMENT below with your thoughts!")
        if channel_name:
            parts.append(f"Turn on NOTIFICATIONS to never miss a video from {channel_name}!")
        parts.append("")

        # 5. RELATED VIDEO LINKS (Increases session time)
        if related_videos and len(related_videos) > 0:
            parts.append("-" * 30)
            parts.append("WATCH NEXT:")
            for i, video_url in enumerate(related_videos[:3], 1):
                parts.append(f"Video {i}: {video_url}")
            parts.append("")

        # 6. HASHTAGS (3-5, shown above title on mobile)
        hashtags = self.get_hashtags()
        parts.append(" ".join(hashtags[:5]))

        description = "\n".join(parts)

        # YouTube description limit is 5000 characters
        if len(description) > 5000:
            description = description[:4997] + "..."

        return description

    def get_hashtags(self, custom_hashtags: Optional[List[str]] = None) -> List[str]:
        """Get 3-5 relevant hashtags for the niche."""
        niche_tags = NICHE_HASHTAGS.get(self.niche, NICHE_HASHTAGS["default"])

        if custom_hashtags:
            # Combine custom with niche, prioritize custom
            all_tags = list(custom_hashtags)
            for tag in niche_tags:
                if tag not in all_tags:
                    all_tags.append(tag)
            return all_tags[:5]

        return niche_tags[:5]

    def optimize_tags(
        self,
        base_tags: List[str],
        title: str = "",
        max_tags: int = 30
    ) -> List[str]:
        """
        Optimize video tags for SEO.

        Combines:
        - Base tags from script/config
        - Trending niche keywords
        - Keywords extracted from title
        - Current year variations

        Args:
            base_tags: Original tags list
            title: Video title for keyword extraction
            max_tags: Maximum number of tags (YouTube limit is 500 chars total)

        Returns:
            Optimized tags list
        """
        optimized = []
        seen = set()

        def add_tag(tag: str):
            tag_lower = tag.lower().strip()
            if tag_lower and tag_lower not in seen and len(tag) <= 30:
                seen.add(tag_lower)
                optimized.append(tag)

        # 1. Add base tags first (most relevant)
        for tag in base_tags:
            add_tag(tag)

        # 2. Add trending niche keywords
        trending = NICHE_TRENDING_TAGS.get(self.niche, NICHE_TRENDING_TAGS["default"])
        for tag in trending:
            add_tag(tag)

        # 3. Extract keywords from title
        if title:
            # Remove special characters and split
            title_words = re.sub(r'[^\w\s]', '', title.lower()).split()
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or',
                         'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'how',
                         'what', 'why', 'when', 'where', 'who', 'this', 'that'}
            for word in title_words:
                if word not in stop_words and len(word) > 3:
                    add_tag(word)

        # 4. Add year variations for relevant tags
        year_relevant = ['guide', 'tutorial', 'tips', 'how to', 'best']
        for tag in list(optimized)[:10]:
            tag_lower = tag.lower()
            if any(kw in tag_lower for kw in year_relevant):
                add_tag(f"{tag} {self.current_year}")

        return optimized[:max_tags]

    def get_optimal_category(self, override_category: Optional[str] = None) -> str:
        """
        Get the optimal YouTube category ID for the niche.

        Args:
            override_category: Optional category name to override niche default

        Returns:
            YouTube category ID string
        """
        if override_category:
            # Map category name to ID
            category_map = {
                "film": "1", "autos": "2", "music": "10", "pets": "15",
                "sports": "17", "travel": "19", "gaming": "20", "blogs": "22",
                "comedy": "23", "entertainment": "24", "news": "25",
                "howto": "26", "education": "27", "science": "28", "nonprofits": "29"
            }
            return category_map.get(override_category.lower(), NICHE_CATEGORIES.get(self.niche, "27"))

        return NICHE_CATEGORIES.get(self.niche, "27")

    def format_chapters(self, sections: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format script sections into YouTube chapter format.

        Args:
            sections: List of script sections with timestamp and title

        Returns:
            List of {"timestamp": "00:00", "title": "Chapter"}
        """
        chapters = []
        current_time = 0

        for section in sections:
            minutes = current_time // 60
            seconds = current_time % 60
            timestamp = f"{minutes:02d}:{seconds:02d}"

            title = section.get("title", section.get("section_type", "Content"))

            # Ensure first chapter is at 00:00
            if not chapters:
                timestamp = "00:00"

            chapters.append({
                "timestamp": timestamp,
                "title": title
            })

            current_time += section.get("duration_seconds", 30)

        return chapters


class YouTubeUploader:
    """Upload videos to YouTube."""

    # YouTube category IDs
    CATEGORIES = {
        "film": "1",
        "autos": "2",
        "music": "10",
        "pets": "15",
        "sports": "17",
        "travel": "19",
        "gaming": "20",
        "blogs": "22",
        "comedy": "23",
        "entertainment": "24",
        "news": "25",
        "howto": "26",
        "education": "27",
        "science": "28",
        "nonprofits": "29",
    }

    # Retry settings for resumable uploads
    MAX_RETRIES = 10
    RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

    def __init__(self, credentials_file: Optional[str] = None):
        """
        Initialize the uploader.

        Args:
            credentials_file: Path to OAuth credentials file
        """
        self.auth = YouTubeAuth(credentials_file=credentials_file)
        self._youtube = None

    @property
    def youtube(self):
        """Get authenticated YouTube service (lazy load)."""
        if self._youtube is None:
            self._youtube = self.auth.get_authenticated_service()
        return self._youtube

    def upload_video_seo_optimized(
        self,
        video_file: str,
        title: str,
        description: str,
        tags: Optional[List[str]] = None,
        niche: str = "default",
        privacy: str = "public",
        thumbnail_file: Optional[str] = None,
        playlist_id: Optional[str] = None,
        publish_at: Optional[datetime] = None,
        made_for_kids: bool = False,
        default_language: str = "en",
        chapters: Optional[List[Dict[str, Any]]] = None,
        related_videos: Optional[List[str]] = None,
        channel_name: str = "",
        hook_text: str = ""
    ) -> UploadResult:
        """
        Upload a video with full YouTube SEO optimization.

        This method applies all SEO best practices:
        - Optimized description with hook + keywords in first 2 lines
        - Timestamps/chapters for navigation
        - Hashtags (3-5 relevant to niche)
        - Optimized tags (trending + niche keywords)
        - Optimal category for niche

        Args:
            video_file: Path to video file
            title: Video title (max 100 chars)
            description: Base video description
            tags: List of tags (will be optimized)
            niche: Content niche (finance, psychology, storytelling, etc.)
            privacy: Privacy status (public recommended for algorithm)
            thumbnail_file: Optional custom thumbnail image
            playlist_id: Optional playlist to add video to
            publish_at: Schedule publish time
            made_for_kids: Whether video is made for kids
            default_language: Default language code
            chapters: Script sections for timestamps
            related_videos: URLs of related videos
            channel_name: Channel name for CTA
            hook_text: First line hook for description

        Returns:
            UploadResult with video ID and URL on success
        """
        # Initialize SEO optimizer
        seo = YouTubeSEOOptimizer(niche=niche)

        # Build hook from title if not provided
        if not hook_text:
            hook_text = f"Discover the secrets of {title}. Watch until the end for a game-changing insight!"

        # Build SEO-optimized description
        formatted_chapters = seo.format_chapters(chapters) if chapters else None
        optimized_description = seo.build_seo_description(
            hook=hook_text,
            main_description=description,
            chapters=formatted_chapters,
            related_videos=related_videos,
            channel_name=channel_name
        )

        # Optimize tags
        optimized_tags = seo.optimize_tags(
            base_tags=tags or [],
            title=title
        )

        # Get optimal category
        category_id = seo.get_optimal_category()

        logger.info(f"SEO-optimized upload: {title}")
        logger.info(f"Niche: {niche}, Category ID: {category_id}")
        logger.info(f"Tags: {len(optimized_tags)}, Privacy: {privacy}")

        # Use the standard upload method with optimized parameters
        return self._upload_video_internal(
            video_file=video_file,
            title=title,
            description=optimized_description,
            tags=optimized_tags,
            category_id=category_id,
            privacy=privacy,
            thumbnail_file=thumbnail_file,
            playlist_id=playlist_id,
            publish_at=publish_at,
            made_for_kids=made_for_kids,
            default_language=default_language
        )

    def upload_video(
        self,
        video_file: str,
        title: str,
        description: str,
        tags: Optional[List[str]] = None,
        category: str = "education",
        privacy: str = "public",
        thumbnail_file: Optional[str] = None,
        playlist_id: Optional[str] = None,
        publish_at: Optional[datetime] = None,
        made_for_kids: bool = False,
        default_language: str = "en"
    ) -> UploadResult:
        """
        Upload a video to YouTube.

        Args:
            video_file: Path to video file
            title: Video title (max 100 chars)
            description: Video description (max 5000 chars)
            tags: List of tags
            category: Category name (education, howto, etc.)
            privacy: Privacy status (public, unlisted, private)
            thumbnail_file: Optional custom thumbnail image
            playlist_id: Optional playlist to add video to
            publish_at: Schedule publish time (requires privacy="private")
            made_for_kids: Whether video is made for kids
            default_language: Default language code

        Returns:
            UploadResult with video ID and URL on success
        """
        # Get category ID from category name
        category_id = self.CATEGORIES.get(category.lower(), "27")  # Default: Education

        return self._upload_video_internal(
            video_file=video_file,
            title=title,
            description=description,
            tags=tags,
            category_id=category_id,
            privacy=privacy,
            thumbnail_file=thumbnail_file,
            playlist_id=playlist_id,
            publish_at=publish_at,
            made_for_kids=made_for_kids,
            default_language=default_language
        )

    def _upload_video_internal(
        self,
        video_file: str,
        title: str,
        description: str,
        tags: Optional[List[str]] = None,
        category_id: str = "27",
        privacy: str = "public",
        thumbnail_file: Optional[str] = None,
        playlist_id: Optional[str] = None,
        publish_at: Optional[datetime] = None,
        made_for_kids: bool = False,
        default_language: str = "en"
    ) -> UploadResult:
        """
        Internal upload method with category_id directly.

        This is the core upload logic used by both upload_video and
        upload_video_seo_optimized methods.
        """
        # Validate file exists
        if not os.path.exists(video_file):
            return UploadResult(
                success=False,
                video_id=None,
                video_url=None,
                error=f"Video file not found: {video_file}"
            )

        logger.info(f"Uploading video: {title}")

        body = {
            "snippet": {
                "title": title[:100],  # Max 100 chars
                "description": description[:5000],  # Max 5000 chars
                "tags": tags or [],
                "categoryId": category_id,
                "defaultLanguage": default_language,
                "defaultAudioLanguage": default_language,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": made_for_kids,
            }
        }

        # Add scheduled publish time
        if publish_at and privacy == "private":
            body["status"]["publishAt"] = publish_at.isoformat() + "Z"
            logger.info(f"Scheduled for: {publish_at}")

        # Create media upload
        media = MediaFileUpload(
            video_file,
            mimetype="video/*",
            resumable=True,
            chunksize=1024 * 1024  # 1MB chunks
        )

        # Execute upload with retry logic
        try:
            request = self.youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )

            video_id = self._resumable_upload(request)

            if video_id:
                video_url = f"https://youtube.com/watch?v={video_id}"
                logger.success(f"Upload complete: {video_url}")

                # Upload thumbnail if provided
                if thumbnail_file and os.path.exists(thumbnail_file):
                    self.set_thumbnail(video_id, thumbnail_file)

                # Add to playlist if specified
                if playlist_id:
                    self.add_to_playlist(video_id, playlist_id)

                return UploadResult(
                    success=True,
                    video_id=video_id,
                    video_url=video_url,
                    error=None
                )

        except HttpError as e:
            error_msg = f"HTTP error: {e.resp.status} - {e.content}"
            logger.error(error_msg)
            return UploadResult(
                success=False,
                video_id=None,
                video_url=None,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            logger.error(error_msg)
            return UploadResult(
                success=False,
                video_id=None,
                video_url=None,
                error=error_msg
            )

        return UploadResult(
            success=False,
            video_id=None,
            video_url=None,
            error="Upload failed for unknown reason"
        )

    def _resumable_upload(self, request) -> Optional[str]:
        """
        Execute a resumable upload with retry logic.

        Args:
            request: YouTube API request object

        Returns:
            Video ID on success, None on failure
        """
        response = None
        retry = 0

        while response is None:
            try:
                logger.info("Uploading...")
                status, response = request.next_chunk()

                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Upload progress: {progress}%")

            except HttpError as e:
                if e.resp.status in self.RETRIABLE_STATUS_CODES:
                    retry += 1
                    if retry > self.MAX_RETRIES:
                        raise Exception(f"Max retries exceeded")

                    sleep_seconds = random.random() * (2 ** retry)
                    logger.warning(f"Retry {retry}/{self.MAX_RETRIES} in {sleep_seconds:.1f}s")
                    time.sleep(sleep_seconds)
                else:
                    raise

        if response:
            return response.get("id")
        return None

    def set_thumbnail(self, video_id: str, thumbnail_file: str) -> bool:
        """
        Set a custom thumbnail for a video.

        Args:
            video_id: YouTube video ID
            thumbnail_file: Path to thumbnail image (JPEG, PNG, GIF, BMP)

        Returns:
            True on success
        """
        if not os.path.exists(thumbnail_file):
            logger.error(f"Thumbnail not found: {thumbnail_file}")
            return False

        logger.info(f"Setting thumbnail for video {video_id}")

        try:
            media = MediaFileUpload(thumbnail_file, mimetype="image/png")

            self.youtube.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()

            logger.success("Thumbnail set successfully")
            return True

        except HttpError as e:
            logger.error(f"Failed to set thumbnail: {e}")
            return False

    def add_to_playlist(self, video_id: str, playlist_id: str) -> bool:
        """
        Add a video to a playlist.

        Args:
            video_id: YouTube video ID
            playlist_id: YouTube playlist ID

        Returns:
            True on success
        """
        logger.info(f"Adding video to playlist {playlist_id}")

        try:
            self.youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id
                        }
                    }
                }
            ).execute()

            logger.success("Added to playlist")
            return True

        except HttpError as e:
            logger.error(f"Failed to add to playlist: {e}")
            return False

    def get_channel_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the authenticated channel.

        Returns:
            Dict with channel info or None
        """
        try:
            response = self.youtube.channels().list(
                part="snippet,statistics",
                mine=True
            ).execute()

            if response.get("items"):
                channel = response["items"][0]
                return {
                    "id": channel["id"],
                    "title": channel["snippet"]["title"],
                    "description": channel["snippet"].get("description", ""),
                    "subscribers": channel["statistics"].get("subscriberCount", "0"),
                    "videos": channel["statistics"].get("videoCount", "0"),
                    "views": channel["statistics"].get("viewCount", "0"),
                }

        except HttpError as e:
            logger.error(f"Failed to get channel info: {e}")

        return None

    def get_my_playlists(self) -> List[Dict[str, str]]:
        """
        Get list of playlists for the authenticated channel.

        Returns:
            List of playlist dicts with id and title
        """
        playlists = []

        try:
            response = self.youtube.playlists().list(
                part="snippet",
                mine=True,
                maxResults=50
            ).execute()

            for item in response.get("items", []):
                playlists.append({
                    "id": item["id"],
                    "title": item["snippet"]["title"]
                })

        except HttpError as e:
            logger.error(f"Failed to get playlists: {e}")

        return playlists

    def create_playlist(
        self,
        title: str,
        description: str = "",
        privacy: str = "unlisted"
    ) -> Optional[str]:
        """
        Create a new playlist.

        Args:
            title: Playlist title
            description: Playlist description
            privacy: Privacy status

        Returns:
            Playlist ID on success
        """
        logger.info(f"Creating playlist: {title}")

        try:
            response = self.youtube.playlists().insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": title,
                        "description": description
                    },
                    "status": {
                        "privacyStatus": privacy
                    }
                }
            ).execute()

            playlist_id = response["id"]
            logger.success(f"Playlist created: {playlist_id}")
            return playlist_id

        except HttpError as e:
            logger.error(f"Failed to create playlist: {e}")
            return None

    # ============================================================
    # YOUTUBE ALGORITHM OPTIMIZATION METHODS
    # ============================================================

    def get_optimal_upload_time(
        self,
        niche: str = "default",
        target_regions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate the optimal upload time for maximum algorithm boost.

        Uses UploadTimingOptimizer to determine the best time to upload
        based on target audience regions and niche-specific patterns.

        Args:
            niche: Content niche (finance, psychology, storytelling, etc.)
            target_regions: List of region codes (US_EST, UK, etc.)

        Returns:
            Dict with recommended_datetime, reasoning, and alternatives

        Example:
            uploader = YouTubeUploader()
            result = uploader.get_optimal_upload_time("finance", ["US_EST", "UK"])
            print(f"Best time: {result['recommended_datetime']}")
        """
        if not YOUTUBE_OPTIMIZER_AVAILABLE:
            logger.warning("YouTube optimizer not available")
            return {
                "error": "youtube_optimizer module not installed",
                "recommended_datetime": None
            }

        optimizer = UploadTimingOptimizer()
        result = optimizer.calculate_optimal_time(niche, target_regions)

        return {
            "recommended_datetime": result.recommended_datetime.isoformat(),
            "target_regions": result.target_regions,
            "peak_hours_utc": result.peak_hours_utc,
            "confidence_score": result.confidence_score,
            "reasoning": result.reasoning,
            "alternative_times": [dt.isoformat() for dt in result.alternative_times]
        }

    def schedule_first_hour_actions(
        self,
        video_id: str,
        niche: str = "default",
        playlist_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Schedule engagement actions for the critical first hour after upload.

        YouTube's algorithm heavily weights engagement in the first 60 minutes.
        This schedules strategic actions to maximize early engagement signals.

        Args:
            video_id: The uploaded video's ID
            niche: Content niche for customized engagement messages
            playlist_ids: List of playlist IDs to add the video to

        Returns:
            List of scheduled actions with timing

        Example:
            uploader = YouTubeUploader()
            actions = uploader.schedule_first_hour_actions("abc123", "finance")
            for action in actions:
                print(f"{action['delay_seconds']}s: {action['description']}")
        """
        if not YOUTUBE_OPTIMIZER_AVAILABLE:
            logger.warning("YouTube optimizer not available")
            return []

        booster = FirstHourBooster()
        actions = booster.schedule_post_upload_actions(
            video_id=video_id,
            niche=niche,
            playlist_ids=playlist_ids
        )

        # Convert to dict format
        return [
            {
                "delay_seconds": a.delay_seconds,
                "action_type": a.action_type,
                "description": a.description,
                "priority": a.priority,
                "parameters": a.parameters
            }
            for a in actions
        ]

    def execute_first_hour_action(
        self,
        action: Dict[str, Any]
    ) -> bool:
        """
        Execute a single first-hour engagement action.

        Args:
            action: Action dict from schedule_first_hour_actions

        Returns:
            True if action was executed successfully
        """
        action_type = action.get("action_type", "")
        video_id = action.get("parameters", {}).get("video_id", "")

        logger.info(f"Executing first-hour action: {action_type}")

        try:
            if action_type == "add_to_playlist":
                playlist_ids = action.get("parameters", {}).get("playlist_ids", [])
                for playlist_id in playlist_ids:
                    self.add_video_to_playlist(video_id, playlist_id)
                return True

            elif action_type == "pin_engagement_comment":
                comment_text = action.get("parameters", {}).get("comment_text", "")
                if comment_text:
                    # Post comment (pinning requires separate API call)
                    response = self.youtube.commentThreads().insert(
                        part="snippet",
                        body={
                            "snippet": {
                                "videoId": video_id,
                                "topLevelComment": {
                                    "snippet": {
                                        "textOriginal": comment_text
                                    }
                                }
                            }
                        }
                    ).execute()
                    logger.success(f"Posted engagement comment: {response.get('id', 'unknown')}")
                return True

            else:
                logger.info(f"Action '{action_type}' logged for manual execution")
                return True

        except HttpError as e:
            logger.error(f"Failed to execute action {action_type}: {e}")
            return False

    def analyze_title(
        self,
        title: str,
        niche: str = "default"
    ) -> Dict[str, Any]:
        """
        Analyze a video title against viral patterns.

        Uses TitlePatternAnalyzer to score titles and provide
        improvement suggestions based on proven patterns.

        Args:
            title: The video title to analyze
            niche: Content niche for pattern matching

        Returns:
            Dict with score, matches, and suggestions

        Example:
            uploader = YouTubeUploader()
            result = uploader.analyze_title("5 Money Mistakes", "finance")
            print(f"Score: {result['score']}/100")
        """
        if not YOUTUBE_OPTIMIZER_AVAILABLE:
            logger.warning("YouTube optimizer not available")
            return {"error": "youtube_optimizer module not installed", "score": 0}

        analyzer = TitlePatternAnalyzer()
        result = analyzer.analyze_title(title, niche)

        return {
            "title": result.title,
            "score": result.score,
            "viral_potential": result.viral_potential,
            "has_number": result.has_number,
            "has_power_word": result.has_power_word,
            "has_question": result.has_question,
            "has_brackets": result.has_brackets,
            "character_count": result.character_count,
            "pattern_matches": result.pattern_matches,
            "suggestions": result.suggestions
        }

    def optimize_video_metadata(
        self,
        title: str,
        description: str,
        tags: List[str],
        niche: str = "default",
        target_keywords: Optional[List[str]] = None,
        script: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fully optimize video metadata for YouTube algorithm.

        Applies all available optimizations:
        1. Title analysis and scoring
        2. Description keyword density optimization
        3. Auto-generated chapters from script
        4. SEO-optimized tags

        Args:
            title: Video title
            description: Video description
            tags: Video tags
            niche: Content niche
            target_keywords: Keywords to optimize for
            script: Optional script dict for chapter generation
            duration_seconds: Video duration for chapter timestamps

        Returns:
            Dict with optimized metadata and analysis

        Example:
            uploader = YouTubeUploader()
            result = uploader.optimize_video_metadata(
                title="5 Money Mistakes",
                description="Learn about money...",
                tags=["money", "finance"],
                niche="finance",
                target_keywords=["money", "investing", "wealth"]
            )
            print(result["optimized_title"])
            print(result["optimized_description"])
        """
        if not YOUTUBE_OPTIMIZER_AVAILABLE:
            logger.warning("YouTube optimizer not available")
            return {
                "optimized_title": title,
                "optimized_description": description,
                "optimized_tags": tags,
                "error": "youtube_optimizer module not installed"
            }

        result = {
            "original_title": title,
            "original_description": description,
            "optimized_title": title,
            "optimized_description": description,
            "optimized_tags": list(tags) if tags else []
        }

        # 1. Analyze and potentially improve title
        title_analysis = self.analyze_title(title, niche)
        result["title_analysis"] = title_analysis

        # 2. Optimize description keywords
        keywords = target_keywords or []
        if not keywords and tags:
            keywords = list(tags)[:5]

        if keywords:
            keyword_result = optimize_description_keywords(
                description,
                keywords,
                target_density=0.025
            )
            result["optimized_description"] = keyword_result.optimized_text
            result["keyword_optimization"] = {
                "original_density": keyword_result.original_density,
                "optimized_density": keyword_result.optimized_density,
                "keywords_added": keyword_result.keywords_added
            }

        # 3. Generate chapters if script provided
        if script and duration_seconds:
            chapters = generate_chapters_from_script(script, duration_seconds)
            result["optimized_description"] = result["optimized_description"] + chapters
            result["chapters_generated"] = True

        # 4. Add niche-specific hashtags
        niche_hashtags = NICHE_HASHTAGS.get(niche, NICHE_HASHTAGS.get("default", []))
        if niche_hashtags:
            # Add hashtags to end of description
            hashtag_line = "\n\n" + " ".join(niche_hashtags[:5])
            result["optimized_description"] += hashtag_line

        logger.success("Video metadata optimized for YouTube algorithm")
        return result

    def upload_with_optimization(
        self,
        video_file: str,
        title: str,
        description: str,
        tags: List[str] = None,
        niche: str = "default",
        target_keywords: List[str] = None,
        script: Dict[str, Any] = None,
        privacy: str = "unlisted",
        thumbnail: str = None,
        playlist_id: str = None,
        schedule_first_hour: bool = True
    ) -> Optional["UploadResult"]:
        """
        Upload a video with full YouTube algorithm optimization.

        This method:
        1. Optimizes all metadata before upload
        2. Uploads the video
        3. Optionally schedules first-hour engagement actions

        Args:
            video_file: Path to the video file
            title: Video title
            description: Video description
            tags: Video tags
            niche: Content niche for optimization
            target_keywords: Keywords to optimize for
            script: Script dict for chapter generation
            privacy: Privacy status (public, unlisted, private)
            thumbnail: Path to thumbnail image
            playlist_id: Playlist to add video to
            schedule_first_hour: Whether to schedule first-hour actions

        Returns:
            UploadResult with video details and scheduled actions

        Example:
            uploader = YouTubeUploader()
            result = uploader.upload_with_optimization(
                video_file="video.mp4",
                title="5 Money Mistakes",
                description="Learn about money...",
                tags=["money", "finance"],
                niche="finance",
                target_keywords=["money", "investing"]
            )
            if result:
                print(f"Video uploaded: {result.video_url}")
        """
        # Optimize metadata
        duration_seconds = None
        if script:
            # Try to get duration from script
            sections = script.get("sections", [])
            duration_seconds = sum(
                s.get("duration_seconds", 30) for s in sections
            ) if sections else None

        optimized = self.optimize_video_metadata(
            title=title,
            description=description,
            tags=tags or [],
            niche=niche,
            target_keywords=target_keywords,
            script=script,
            duration_seconds=duration_seconds
        )

        # Upload with optimized metadata
        result = self.upload_video(
            video_file=video_file,
            title=optimized.get("optimized_title", title),
            description=optimized.get("optimized_description", description),
            tags=optimized.get("optimized_tags", tags),
            privacy=privacy,
            thumbnail=thumbnail
        )

        if result and schedule_first_hour:
            # Schedule first-hour engagement actions
            playlist_ids = [playlist_id] if playlist_id else None
            actions = self.schedule_first_hour_actions(
                video_id=result.video_id,
                niche=niche,
                playlist_ids=playlist_ids
            )
            result.scheduled_actions = actions
            logger.info(f"Scheduled {len(actions)} first-hour engagement actions")

        return result


    def upload_with_full_optimization(
        self,
        video_file: str,
        title: str,
        description: str,
        tags: List[str] = None,
        niche: str = "default",
        video_id: str = None,
        script: str = "",
        duration_seconds: int = 600,
        privacy: str = "unlisted",
        thumbnail: str = None,
        playlist_id: str = None,
        ai_disclosure_enabled: bool = True,
        metadata_optimization_enabled: bool = True,
        auto_chapters: bool = True
    ) -> UploadResult:
        """
        Upload a video with AI disclosure and full metadata optimization.

        This method combines:
        1. AI disclosure tracking and description append
        2. Metadata optimization (IMPACT formula titles, keyword front-loading)
        3. Auto-generated chapters from script
        4. SEO-optimized tags

        Args:
            video_file: Path to the video file
            title: Video title
            description: Video description
            tags: Video tags
            niche: Content niche for optimization
            video_id: Unique video ID for tracking (auto-generated if not provided)
            script: Script text for chapter generation
            duration_seconds: Video duration for chapter timestamps
            privacy: Privacy status (public, unlisted, private)
            thumbnail: Path to thumbnail image
            playlist_id: Playlist to add video to
            ai_disclosure_enabled: Add AI disclosure to description
            metadata_optimization_enabled: Optimize title/description/tags
            auto_chapters: Auto-generate chapters from script

        Returns:
            UploadResult with video details
        """
        import uuid

        # Generate video ID if not provided
        if not video_id:
            video_id = f"vid_{uuid.uuid4().hex[:12]}"

        optimized_title = title
        optimized_description = description
        optimized_tags = tags or []
        chapters = []

        # 1. Metadata Optimization
        if metadata_optimization_enabled and METADATA_OPTIMIZER_AVAILABLE:
            try:
                optimizer = MetadataOptimizer()

                # Get keywords from tags or generate from title
                keywords = list(tags) if tags else [title.split()[0]] if title else []

                # Create complete optimized metadata
                metadata = optimizer.create_complete_metadata(
                    topic=title,
                    keywords=keywords,
                    script=script,
                    video_duration=duration_seconds
                )

                optimized_title = metadata.title
                optimized_tags = metadata.tags
                chapters = metadata.chapters

                # Build optimized description with chapters
                optimized_description = description

                # Add chapters if generated and auto_chapters enabled
                if auto_chapters and chapters:
                    chapter_text = "\n\n" + "=" * 30 + "\nCHAPTERS:\n" + "=" * 30 + "\n"
                    for chapter in chapters:
                        timestamp = self._format_timestamp(chapter.get("start", 0))
                        chapter_title = chapter.get("title", "Chapter")
                        chapter_text += f"{timestamp} - {chapter_title}\n"
                    optimized_description += chapter_text

                logger.info(f"[Upload] Metadata optimized: title_score={metadata.title_score:.1f}")

            except Exception as e:
                logger.warning(f"[Upload] Metadata optimization failed: {e}")

        # 2. AI Disclosure
        if ai_disclosure_enabled and AI_DISCLOSURE_AVAILABLE:
            try:
                tracker = AIDisclosureTracker()

                # Track AI usage (TTS and script generation are common)
                tracker.track_voice_generation(video_id, "edge-tts")
                tracker.track_script_generation(video_id, "groq")

                # Get disclosure metadata
                disclosure = tracker.get_disclosure_metadata(video_id)

                if disclosure.requires_disclosure:
                    # Append disclosure to description
                    disclaimer = disclosure.get_description_disclaimer()
                    optimized_description += disclaimer
                    logger.info(f"[Upload] AI disclosure added: {disclosure.disclosure_level.value}")

            except Exception as e:
                logger.warning(f"[Upload] AI disclosure tracking failed: {e}")

        # 3. SEO optimization (existing)
        seo = YouTubeSEOOptimizer(niche=niche)
        optimized_tags = seo.optimize_tags(optimized_tags, optimized_title)

        # Add niche hashtags to description
        hashtags = seo.get_hashtags()
        optimized_description += "\n\n" + " ".join(hashtags[:5])

        # 4. Upload with optimized metadata
        return self._upload_video_internal(
            video_file=video_file,
            title=optimized_title,
            description=optimized_description,
            tags=optimized_tags,
            category_id=seo.get_optimal_category(),
            privacy=privacy,
            thumbnail_file=thumbnail,
            playlist_id=playlist_id
        )

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def add_video_to_playlist(self, video_id: str, playlist_id: str) -> bool:
        """
        Add a video to a playlist (alias for add_to_playlist).

        Args:
            video_id: YouTube video ID
            playlist_id: YouTube playlist ID

        Returns:
            True on success
        """
        return self.add_to_playlist(video_id, playlist_id)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOUTUBE UPLOADER TEST")
    print("="*60 + "\n")

    uploader = YouTubeUploader()

    # Get channel info
    try:
        info = uploader.get_channel_info()
        if info:
            print(f"Channel: {info['title']}")
            print(f"Subscribers: {info['subscribers']}")
            print(f"Total videos: {info['videos']}")
        else:
            print("Could not get channel info. Check authentication.")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Created OAuth credentials in Google Cloud Console")
        print("2. Saved client_secret.json to config/")
