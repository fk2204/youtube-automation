"""
YouTube OAuth2 Authentication Module

Handles authentication with YouTube Data API v3.

Setup:
1. Go to Google Cloud Console (console.cloud.google.com)
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create OAuth 2.0 credentials (Desktop app)
5. Download client_secret.json to config/

Usage:
    auth = YouTubeAuth()
    youtube = auth.get_authenticated_service()
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional

from loguru import logger

try:
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except ImportError:
    raise ImportError(
        "Please install Google API libraries:\n"
        "pip install google-auth-oauthlib google-api-python-client"
    )


class YouTubeAuth:
    """Handle YouTube API authentication."""

    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/youtube.force-ssl",
    ]

    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"

    def __init__(
        self, client_secrets_file: Optional[str] = None, credentials_file: Optional[str] = None
    ):
        """
        Initialize YouTube authentication.

        Args:
            client_secrets_file: Path to client_secret.json from Google Cloud
            credentials_file: Path to store/load OAuth credentials
        """
        self.client_secrets_file = client_secrets_file or os.getenv(
            "YOUTUBE_CLIENT_SECRETS_FILE", "config/client_secret.json"
        )
        self.credentials_file = credentials_file or "config/youtube_credentials.pickle"

        # Ensure config directory exists
        Path(self.credentials_file).parent.mkdir(parents=True, exist_ok=True)

    def get_credentials(self) -> Any: # Google OAuth2 Credentials
        """
        Get valid OAuth credentials, refreshing or creating as needed.

        Returns:
            Valid Google OAuth credentials
        """
        credentials = None

        # Try to load existing credentials
        if os.path.exists(self.credentials_file):
            logger.info("Loading existing credentials...")
            with open(self.credentials_file, "rb") as f:
                credentials = pickle.load(f)

        # Check if credentials are valid
        if credentials and credentials.valid:
            logger.info("Credentials are valid")
            return credentials

        # Try to refresh expired credentials
        if credentials and credentials.expired and credentials.refresh_token:
            logger.info("Refreshing expired credentials...")
            try:
                credentials.refresh(Request())
                self._save_credentials(credentials)
                return credentials
            except Exception as e:
                logger.warning(f"Could not refresh credentials: {e}")
                credentials = None

        # Need to create new credentials
        if not credentials:
            credentials = self._create_new_credentials()

        return credentials

    def _create_new_credentials(self) -> Any:
        """Create new OAuth credentials via browser flow with fallback to OOB."""
        if not self.client_secrets_file or not os.path.exists(self.client_secrets_file):
            raise FileNotFoundError(
                f"Client secrets file not found: {self.client_secrets_file}\n\n"
                "To set up YouTube API:\n"
                "1. Go to console.cloud.google.com\n"
                "2. Create a project and enable 'YouTube Data API v3'\n"
                "3. Create OAuth 2.0 credentials (Desktop app)\n"
                "4. Download and save as config/client_secret.json"
            )

        logger.info("Creating new credentials (browser will open)...")

        flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_file, self.SCOPES)

        # Try local server first
        try:
            logger.debug("Attempting local server authentication (port 0 - auto-select)...")
            credentials = flow.run_local_server(
                port=0,
                prompt="consent",
                authorization_prompt_message="Please authorize YouTube access in your browser.",
            )
            self._save_credentials(credentials)
            logger.success("New credentials created and saved")
            return credentials

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Local server failed: {error_msg}")

            # If redirect_uri_mismatch, try OOB flow
            if "redirect_uri_mismatch" in error_msg or "invalid_request" in error_msg:
                logger.info("Falling back to Out-of-Band (OOB) authorization...")
                return self._create_new_credentials_oob(flow)
            else:
                raise

    def _create_new_credentials_oob(self, flow) -> Any:
        """Fallback to Out-of-Band OAuth flow."""
        logger.info("\n" + "="*70)
        logger.info("MANUAL AUTHORIZATION REQUIRED (Out-of-Band Flow)")
        logger.info("="*70)

        # Get authorization URL
        auth_url, _ = flow.authorization_url(prompt='consent')

        print(f"\n[STEP 1] Open this link in your browser:")
        print(f"  {auth_url}")

        print(f"\n[STEP 2] Sign in and authorize YouTube access")

        print(f"\n[STEP 3] You'll be redirected with an authorization code")
        print(f"         Copy the entire code (it looks like: 4/0AX...)")

        print(f"\n[STEP 4] Paste the code here:")

        try:
            auth_code = input("  Authorization code: ").strip()
        except KeyboardInterrupt:
            logger.error("Authorization cancelled by user")
            raise

        if not auth_code:
            raise ValueError("No authorization code provided")

        # Exchange code for credentials
        try:
            credentials = flow.fetch_token(code=auth_code)
            logger.success("Authorization successful!")
            self._save_credentials(credentials)
            return credentials
        except Exception as e:
            logger.error(f"Failed to exchange authorization code: {e}")
            raise

    def _save_credentials(self, credentials: Any) -> None:
        """Save credentials to file."""
        with open(self.credentials_file, "wb") as f:
            pickle.dump(credentials, f)
        logger.info(f"Credentials saved to {self.credentials_file}")

    def get_authenticated_service(self) -> Any:
        """
        Get an authenticated YouTube API service.

        Returns:
            YouTube API service object
        """
        credentials = self.get_credentials()

        service = build(self.API_SERVICE_NAME, self.API_VERSION, credentials=credentials)

        logger.info("YouTube API service created")
        return service

    def revoke_credentials(self) -> None:
        """Revoke and delete stored credentials."""
        if os.path.exists(self.credentials_file):
            os.remove(self.credentials_file)
            logger.info("Credentials revoked and deleted")
        else:
            logger.info("No credentials to revoke")


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOUTUBE AUTHENTICATION TEST")
    print("=" * 60 + "\n")

    auth = YouTubeAuth()

    try:
        youtube = auth.get_authenticated_service()
        print("Successfully authenticated!")

        # Test by getting channel info
        request = youtube.channels().list(part="snippet", mine=True)
        response = request.execute()

        if response.get("items"):
            channel = response["items"][0]["snippet"]
            print(f"\nChannel: {channel['title']}")
            print(f"Description: {channel.get('description', 'N/A')[:100]}...")

    except FileNotFoundError as e:
        print(f"\nSetup required:\n{e}")
    except Exception as e:
        print(f"\nError: {e}")
