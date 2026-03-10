"""
YouTube OAuth2 - Out-of-Band (OOB) Flow

Doesn't require redirect URI configuration in Google Cloud.
User manually enters authorization code from browser.
"""

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


class YouTubeAuthOOB:
    """Handle YouTube API authentication using Out-of-Band (OOB) flow."""

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
        """Initialize YouTube authentication (OOB flow)."""
        self.client_secrets_file = client_secrets_file or os.getenv(
            "YOUTUBE_CLIENT_SECRETS_FILE", "config/client_secret.json"
        )
        self.credentials_file = credentials_file or "config/youtube_credentials.pickle"
        Path(self.credentials_file).parent.mkdir(parents=True, exist_ok=True)

    def get_credentials(self) -> Any:
        """Get valid OAuth credentials."""
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
            credentials = self._create_new_credentials_oob()

        return credentials

    def _create_new_credentials_oob(self) -> Any:
        """Create new OAuth credentials using Out-of-Band (OOB) flow."""
        if not self.client_secrets_file or not os.path.exists(self.client_secrets_file):
            raise FileNotFoundError(
                f"Client secrets file not found: {self.client_secrets_file}\n\n"
                "To set up YouTube API:\n"
                "1. Go to console.cloud.google.com\n"
                "2. Create a project and enable 'YouTube Data API v3'\n"
                "3. Create OAuth 2.0 credentials\n"
                "4. Download and save as config/client_secret.json"
            )

        logger.info("Starting Out-of-Band (OOB) OAuth flow...")
        logger.info("No redirect URI needed - you'll authorize manually")

        flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_file, self.SCOPES)

        # Desktop apps use OOB (Out-of-Band) flow directly
        logger.info("Using Out-of-Band authorization (no local server needed)...")

        logger.info("\n" + "="*70)
        logger.info("MANUAL AUTHORIZATION REQUIRED")
        logger.info("="*70)

        # Get authorization URL
        auth_url, _ = flow.authorization_url(prompt='consent')

        print(f"\n[STEP 1] Open this link in your browser:")
        print(f"  {auth_url}")

        print(f"\n[STEP 2] You'll see a Google login screen")
        print(f"         Sign in if prompted")

        print(f"\n[STEP 3] Click 'Allow' to authorize YouTube access")

        print(f"\n[STEP 4] You'll be redirected to a page with an authorization code")
        print(f"         Copy the entire code (it looks like: 4/0AX...) ")

        print(f"\n[STEP 5] Paste the code here and press Enter:")

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
        except Exception as e:
            logger.error(f"Failed to exchange authorization code: {e}")
            raise

        self._save_credentials(credentials)
        logger.success("Credentials saved to config/youtube_credentials.pickle")

        return credentials

    def _save_credentials(self, credentials: Any) -> None:
        """Save credentials to file."""
        with open(self.credentials_file, "wb") as f:
            pickle.dump(credentials, f)
        logger.info(f"Credentials saved to {self.credentials_file}")

    def get_authenticated_service(self) -> Any:
        """Get an authenticated YouTube API service."""
        credentials = self.get_credentials()
        service = build(self.API_SERVICE_NAME, self.API_VERSION, credentials=credentials)
        logger.success("YouTube API service ready")
        return service

    def get_youtube_service(self) -> Any:
        """Alias for get_authenticated_service()."""
        return self.get_authenticated_service()
