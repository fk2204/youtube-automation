"""
Unit tests for EmailRepurposer.

All external API calls and SMTP connections are mocked.
No real credentials required. Tests are fully isolated.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from distribution.repurposers.repurposer_registry import RepurposerRegistry
import distribution.repurposers.email_repurposer as email_module
from distribution.repurposers.email_repurposer import EmailRepurposer, SUPPORTED_PROVIDERS
from exceptions import ConfigError

SENDGRID_CONFIG = {"provider": "sendgrid"}
MAILCHIMP_CONFIG = {
    "provider": "mailchimp",
    "from_name": "Test Newsletter",
    "reply_to": "reply@test.com",
}
SMTP_CONFIG = {"provider": "smtp"}
SAMPLE_CONTENT = "# Newsletter Title\n\nThis is the newsletter body content."


@pytest.fixture(autouse=True)
def clear_simulation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOCIAL_SIMULATION_MODE", raising=False)



class TestInit:
    def test_init_raises_config_error_on_invalid_provider(self) -> None:
        """__init__ must raise ConfigError when provider is not in SUPPORTED_PROVIDERS."""
        with pytest.raises(ConfigError, match="provider"):
            EmailRepurposer({"provider": "myspace"})

    def test_init_raises_config_error_on_empty_provider(self) -> None:
        """__init__ must raise ConfigError when provider key is missing."""
        with pytest.raises(ConfigError):
            EmailRepurposer({})

    def test_init_succeeds_sendgrid(self) -> None:
        """__init__ succeeds with 'sendgrid' provider."""
        rp = EmailRepurposer(SENDGRID_CONFIG)
        assert rp._provider == "sendgrid"

    def test_init_succeeds_mailchimp(self) -> None:
        """__init__ succeeds with 'mailchimp' provider."""
        rp = EmailRepurposer(MAILCHIMP_CONFIG)
        assert rp._provider == "mailchimp"

    def test_init_succeeds_smtp(self) -> None:
        """__init__ succeeds with 'smtp' provider."""
        rp = EmailRepurposer(SMTP_CONFIG)
        assert rp._provider == "smtp"


class TestAuthenticateSendGrid:
    @pytest.mark.asyncio
    async def test_authenticate_sendgrid_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns True for valid SendGrid API key."""
        monkeypatch.setenv("SENDGRID_API_KEY", "SG.test_key_123")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": [], "contact_count": 0}

        with patch("distribution.repurposers.email_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            rp = EmailRepurposer(SENDGRID_CONFIG)
            result = await rp.authenticate()

        assert result is True
        assert rp._authenticated is True

    @pytest.mark.asyncio
    async def test_authenticate_sendgrid_returns_false_on_missing_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns False when SENDGRID_API_KEY is not set."""
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        rp = EmailRepurposer(SENDGRID_CONFIG)
        result = await rp.authenticate()
        assert result is False


class TestAuthenticateMailchimp:
    @pytest.mark.asyncio
    async def test_authenticate_mailchimp_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns True for valid Mailchimp credentials."""
        monkeypatch.setenv("MAILCHIMP_API_KEY", "abc123-us1")
        monkeypatch.setenv("MAILCHIMP_SERVER_PREFIX", "us1")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"health_status": "Everything's Chimpy!"}

        with patch("distribution.repurposers.email_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            rp = EmailRepurposer(MAILCHIMP_CONFIG)
            result = await rp.authenticate()

        assert result is True

    @pytest.mark.asyncio
    async def test_authenticate_mailchimp_returns_false_on_missing_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns False when MAILCHIMP_API_KEY is not set."""
        monkeypatch.delenv("MAILCHIMP_API_KEY", raising=False)
        monkeypatch.delenv("MAILCHIMP_SERVER_PREFIX", raising=False)
        rp = EmailRepurposer(MAILCHIMP_CONFIG)
        result = await rp.authenticate()
        assert result is False


class TestAuthenticateSmtp:
    @pytest.mark.asyncio
    async def test_authenticate_smtp_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns True when SMTP connection succeeds."""
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_PORT", "587")
        monkeypatch.setenv("SMTP_USERNAME", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        monkeypatch.setenv("SMTP_FROM_ADDRESS", "from@test.com")

        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp.ehlo = MagicMock()
        mock_smtp.starttls = MagicMock()
        mock_smtp.login = MagicMock()

        with patch("distribution.repurposers.email_repurposer.smtplib.SMTP", return_value=mock_smtp):
            rp = EmailRepurposer(SMTP_CONFIG)
            result = await rp.authenticate()

        assert result is True

    @pytest.mark.asyncio
    async def test_authenticate_smtp_returns_false_on_missing_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns False when SMTP env vars are missing."""
        monkeypatch.delenv("SMTP_HOST", raising=False)
        monkeypatch.delenv("SMTP_USERNAME", raising=False)
        monkeypatch.delenv("SMTP_PASSWORD", raising=False)
        monkeypatch.delenv("SMTP_FROM_ADDRESS", raising=False)
        rp = EmailRepurposer(SMTP_CONFIG)
        result = await rp.authenticate()
        assert result is False


class TestRepurpose:
    @pytest.mark.asyncio
    async def test_repurpose_sendgrid_creates_draft(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """repurpose() creates a SendGrid campaign draft with send_immediately=False."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"id": "sg_camp_001"}

        with patch("distribution.repurposers.email_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            rp = EmailRepurposer(SENDGRID_CONFIG)
            rp._sendgrid_key = "SG.test"
            rp._authenticated = True
            result = await rp.repurpose(SAMPLE_CONTENT, send_immediately=False)

        assert result["success"] is True
        assert result["campaign_id"] == "sg_camp_001"
        assert result["status"] == "draft"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_repurpose_not_authenticated_returns_error(self) -> None:
        """repurpose() returns error dict when not authenticated."""
        rp = EmailRepurposer(SENDGRID_CONFIG)
        result = await rp.repurpose(SAMPLE_CONTENT)
        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_repurpose_smtp_draft_returns_draft_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """repurpose() with SMTP and send_immediately=False returns 'draft' status."""
        rp = EmailRepurposer(SMTP_CONFIG)
        rp._authenticated = True
        result = await rp.repurpose(SAMPLE_CONTENT, send_immediately=False)
        assert result["success"] is True
        assert result["status"] == "draft"
