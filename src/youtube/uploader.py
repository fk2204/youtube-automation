"""
YouTube Video Uploader Module

Uploads videos to YouTube with metadata, thumbnails, and scheduling.

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
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

try:
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    raise ImportError("Please install google-api-python-client")

from .auth import YouTubeAuth


@dataclass
class UploadResult:
    """Result of a video upload."""
    success: bool
    video_id: Optional[str]
    video_url: Optional[str]
    error: Optional[str]


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

    def upload_video(
        self,
        video_file: str,
        title: str,
        description: str,
        tags: Optional[List[str]] = None,
        category: str = "education",
        privacy: str = "unlisted",
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
        # Validate file exists
        if not os.path.exists(video_file):
            return UploadResult(
                success=False,
                video_id=None,
                video_url=None,
                error=f"Video file not found: {video_file}"
            )

        logger.info(f"Uploading video: {title}")

        # Prepare metadata
        category_id = self.CATEGORIES.get(category.lower(), "27")  # Default: Education

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
