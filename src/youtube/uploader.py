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
