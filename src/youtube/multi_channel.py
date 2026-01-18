"""
Multi-Channel YouTube Manager

Authenticate and upload to multiple YouTube channels.

Usage:
    manager = MultiChannelManager()

    # Authenticate all channels (one-time)
    manager.authenticate_channel("money_blueprints")
    manager.authenticate_channel("mind_unlocked")
    manager.authenticate_channel("untold_stories")

    # Upload to specific channel
    manager.upload_to_channel("money_blueprints", video_file, title, description)
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from loguru import logger

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


@dataclass
class ChannelInfo:
    """Information about a YouTube channel."""
    channel_id: str
    name: str
    credentials_file: str
    authenticated: bool = False


class MultiChannelManager:
    """Manage multiple YouTube channels."""

    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.readonly"
    ]

    # Channel configurations
    CHANNELS = {
        "money_blueprints": {
            "name": "Money Blueprints",
            "niche": "finance",
            "voice": "en-US-GuyNeural"
        },
        "mind_unlocked": {
            "name": "Mind Unlocked",
            "niche": "psychology",
            "voice": "en-US-JennyNeural"
        },
        "untold_stories": {
            "name": "Untold Stories",
            "niche": "storytelling",
            "voice": "en-GB-RyanNeural"
        }
    }

    def __init__(self):
        self.config_dir = Path("config")
        self.credentials_dir = self.config_dir / "credentials"
        self.client_secrets = self.config_dir / "client_secret.json"

        # Create credentials directory
        self.credentials_dir.mkdir(parents=True, exist_ok=True)

        # Track authenticated channels
        self.channels: Dict[str, ChannelInfo] = {}

        # Load existing credentials
        self._load_existing_credentials()

        logger.info(f"MultiChannelManager initialized")

    def _load_existing_credentials(self):
        """Load any existing channel credentials."""
        for channel_id in self.CHANNELS.keys():
            cred_file = self.credentials_dir / f"{channel_id}.pickle"
            if cred_file.exists():
                self.channels[channel_id] = ChannelInfo(
                    channel_id=channel_id,
                    name=self.CHANNELS[channel_id]["name"],
                    credentials_file=str(cred_file),
                    authenticated=True
                )
                logger.info(f"Found credentials for: {self.CHANNELS[channel_id]['name']}")

    def authenticate_channel(self, channel_id: str) -> bool:
        """
        Authenticate a specific channel.

        Args:
            channel_id: Channel ID (money_blueprints, mind_unlocked, untold_stories)

        Returns:
            True if successful
        """
        if channel_id not in self.CHANNELS:
            logger.error(f"Unknown channel: {channel_id}")
            return False

        channel_name = self.CHANNELS[channel_id]["name"]
        cred_file = self.credentials_dir / f"{channel_id}.pickle"

        print()
        print("="*60)
        print(f"  AUTHENTICATE: {channel_name}")
        print("="*60)
        print()
        print("A browser will open. Please:")
        print(f"1. Sign in with Google")
        print(f"2. Select the '{channel_name}' channel")
        print(f"3. Click 'Allow'")
        print()

        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.client_secrets),
                self.SCOPES
            )

            credentials = flow.run_local_server(
                port=0,
                prompt="consent"
            )

            # Save credentials
            with open(cred_file, "wb") as f:
                pickle.dump(credentials, f)

            # Verify by getting channel info
            youtube = build("youtube", "v3", credentials=credentials)
            request = youtube.channels().list(part="snippet", mine=True)
            response = request.execute()

            if response.get("items"):
                actual_name = response["items"][0]["snippet"]["title"]
                logger.success(f"Authenticated: {actual_name}")

                self.channels[channel_id] = ChannelInfo(
                    channel_id=channel_id,
                    name=actual_name,
                    credentials_file=str(cred_file),
                    authenticated=True
                )
                return True

        except Exception as e:
            logger.error(f"Authentication failed: {e}")

        return False

    def get_youtube_service(self, channel_id: str):
        """Get authenticated YouTube service for a channel."""
        if channel_id not in self.channels:
            logger.error(f"Channel not authenticated: {channel_id}")
            return None

        cred_file = self.credentials_dir / f"{channel_id}.pickle"

        try:
            with open(cred_file, "rb") as f:
                credentials = pickle.load(f)

            # Refresh if needed
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                with open(cred_file, "wb") as f:
                    pickle.dump(credentials, f)

            return build("youtube", "v3", credentials=credentials)

        except Exception as e:
            logger.error(f"Failed to get service for {channel_id}: {e}")
            return None

    def upload_to_channel(
        self,
        channel_id: str,
        video_file: str,
        title: str,
        description: str,
        tags: List[str] = None,
        privacy: str = "unlisted",
        thumbnail_file: str = None
    ) -> Optional[str]:
        """
        Upload video to a specific channel.

        Returns:
            Video URL if successful, None otherwise
        """
        youtube = self.get_youtube_service(channel_id)
        if not youtube:
            return None

        channel_name = self.channels[channel_id].name
        logger.info(f"Uploading to {channel_name}: {title}")

        try:
            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags or [],
                    "categoryId": "22"  # People & Blogs
                },
                "status": {
                    "privacyStatus": privacy,
                    "selfDeclaredMadeForKids": False
                }
            }

            media = MediaFileUpload(
                video_file,
                mimetype="video/mp4",
                resumable=True
            )

            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )

            response = request.execute()
            video_id = response["id"]
            video_url = f"https://youtube.com/watch?v={video_id}"

            logger.success(f"Uploaded to {channel_name}: {video_url}")

            # Set thumbnail if provided
            if thumbnail_file and os.path.exists(thumbnail_file):
                try:
                    youtube.thumbnails().set(
                        videoId=video_id,
                        media_body=MediaFileUpload(thumbnail_file)
                    ).execute()
                    logger.success("Thumbnail set")
                except:
                    pass

            return video_url

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return None

    def list_channels(self):
        """List all channels and their status."""
        print()
        print("="*60)
        print("  YOUR YOUTUBE CHANNELS")
        print("="*60)
        print()

        for channel_id, config in self.CHANNELS.items():
            if channel_id in self.channels:
                status = "[OK] Authenticated"
                name = self.channels[channel_id].name
            else:
                status = "[--] Not authenticated"
                name = config["name"]

            print(f"  {channel_id}")
            print(f"    Name: {name}")
            print(f"    Niche: {config['niche']}")
            print(f"    Status: {status}")
            print()

    def authenticate_all(self):
        """Authenticate all channels one by one."""
        for channel_id in self.CHANNELS.keys():
            if channel_id not in self.channels:
                print(f"\nNext: {self.CHANNELS[channel_id]['name']}")
                input("Press Enter to continue...")
                self.authenticate_channel(channel_id)
            else:
                print(f"Already authenticated: {self.channels[channel_id].name}")


# CLI
if __name__ == "__main__":
    import sys

    manager = MultiChannelManager()

    if len(sys.argv) < 2:
        manager.list_channels()
        print("\nUsage:")
        print("  python multi_channel.py list          - List channels")
        print("  python multi_channel.py auth <id>     - Authenticate channel")
        print("  python multi_channel.py auth-all      - Authenticate all channels")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "list":
        manager.list_channels()

    elif cmd == "auth" and len(sys.argv) > 2:
        channel_id = sys.argv[2]
        manager.authenticate_channel(channel_id)

    elif cmd == "auth-all":
        manager.authenticate_all()
