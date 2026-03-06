"""
EmailRepurposer — distributes content via email newsletter providers.

Supports three providers:
  - sendgrid:  SendGrid Marketing Campaigns API (Dynamic Templates)
  - mailchimp: Mailchimp Campaigns API v3
  - smtp:      SMTP for self-hosted providers (e.g. Postfix, Amazon SES SMTP)

Default behavior: create a campaign draft, do NOT send (send_immediately=False).
This prevents accidental sends during testing or dry-run workflows.

API flows:

SendGrid:
  POST /v3/marketing/campaigns — create campaign with html_content.

Mailchimp:
  POST /3.0/campaigns — create campaign.
  PUT  /3.0/campaigns/{id}/content — set HTML body.
  POST /3.0/campaigns/{id}/actions/send — only if send_immediately=True.

SMTP:
  smtplib.SMTP (or SMTP_SSL) — sends directly to recipient.
  Note: SMTP mode does not support list_id; it sends to SMTP_TO_ADDRESS env var.

Credentials (env vars):
  SendGrid:  SENDGRID_API_KEY
  Mailchimp: MAILCHIMP_API_KEY, MAILCHIMP_SERVER_PREFIX (e.g. "us1")
  SMTP:      SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
             SMTP_FROM_ADDRESS, SMTP_TO_ADDRESS
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from distribution.repurposers.base_repurposer import BaseRepurposer
from distribution.repurposers.repurposer_registry import RepurposerRegistry
from exceptions import AuthError, ConfigError, RateLimitError

SENDGRID_API_BASE = "https://api.sendgrid.com/v3"

SUPPORTED_PROVIDERS = {"sendgrid", "mailchimp", "smtp"}


class EmailRepurposer(BaseRepurposer):
    """
    Distributes content via email newsletter providers.

    Raises ConfigError at __init__ if provider is not in SUPPORTED_PROVIDERS.
    Default behavior is to create a draft — set send_immediately=True to send.

    Config keys:
        provider:  "sendgrid" | "mailchimp" | "smtp"
        list_name: str — human-readable label for logging
    """

    PLATFORM_NAME = "email"
    AUTOMATION_LEVEL = "full"

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize and validate provider.

        Args:
            config: Must contain {"provider": "sendgrid"|"mailchimp"|"smtp"}.

        Raises:
            ConfigError: if provider is not in SUPPORTED_PROVIDERS.
        """
        super().__init__(config)

        provider = str(config.get("provider", "")).lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(
                f"EmailRepurposer: provider '{provider}' is not supported. "
                f"Choose one of: {sorted(SUPPORTED_PROVIDERS)}."
            )

        self._provider = provider
        self._sendgrid_key: Optional[str] = None
        self._mailchimp_key: Optional[str] = None
        self._mailchimp_server: Optional[str] = None

    @property
    def platform_name(self) -> str:
        return self.PLATFORM_NAME

    async def authenticate(self) -> bool:
        """
        Validate provider-specific credentials from env vars.

        Returns:
            True on success, False on failure (never raises).
        """
        try:
            if self._provider == "sendgrid":
                return await self._authenticate_sendgrid()
            if self._provider == "mailchimp":
                return await self._authenticate_mailchimp()
            if self._provider == "smtp":
                return self._authenticate_smtp()
        except Exception as exc:
            logger.error(
                "EmailRepurposer ({p}) authenticate unexpected error: {exc}",
                p=self._provider,
                exc=exc,
            )
        return False

    async def repurpose(
        self,
        source_content: str,
        recipient_list_id: Optional[str] = None,
        subject_override: Optional[str] = None,
        send_immediately: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create an email campaign draft (or send if send_immediately=True).

        Args:
            source_content:    Article body as Markdown or plain text.
            recipient_list_id: Provider-specific list/audience ID.
            subject_override:  Override the auto-generated subject line.
            send_immediately:  If True, send instead of saving as draft.
                               Default: False (create draft only).

        Returns:
            {
                "success": bool,
                "campaign_id": str | None,
                "status": str,
                "preview_url": str | None,
                "error": str | None,
            }
        """
        if self.simulation_mode:
            sim = self._simulate_repurpose({
                "provider": self._provider,
                "send_immediately": send_immediately,
            })
            sim.update({
                "campaign_id": "sim_email_001",
                "status": "draft",
                "preview_url": None,
            })
            return sim

        if not self._authenticated:
            return self._build_error_response(
                "Not authenticated. Call authenticate() first.",
                extra={"campaign_id": None, "status": "failed", "preview_url": None},
            )

        html_content = self._transform_to_email_html(source_content)
        subject = subject_override or self._extract_subject(source_content)

        if self._provider == "sendgrid":
            return await self._send_via_sendgrid(
                html_content=html_content,
                subject=subject,
                list_id=recipient_list_id,
                send_immediately=send_immediately,
            )
        if self._provider == "mailchimp":
            return await self._send_via_mailchimp(
                html_content=html_content,
                subject=subject,
                list_id=recipient_list_id,
                send_immediately=send_immediately,
            )
        # smtp
        return self._send_via_smtp(
            html_content=html_content,
            subject=subject,
            send_immediately=send_immediately,
        )

    def _transform_to_email_html(self, content: str) -> str:
        """
        Transform source content into a basic email-safe HTML body.

        Converts markdown headings and paragraphs to HTML.
        Wraps in a minimal table layout for email client compatibility.
        """
        lines = content.strip().splitlines()
        parts: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                parts.append(f"<h1 style=\"font-family: sans-serif;\">{self._esc(stripped[2:])}</h1>")
            elif stripped.startswith("## "):
                parts.append(f"<h2 style=\"font-family: sans-serif;\">{self._esc(stripped[3:])}</h2>")
            elif stripped.startswith("### "):
                parts.append(f"<h3 style=\"font-family: sans-serif;\">{self._esc(stripped[4:])}</h3>")
            elif stripped == "":
                parts.append("<br>")
            else:
                parts.append(
                    f"<p style=\"font-family: sans-serif; line-height: 1.6;\">"
                    f"{self._esc(stripped)}</p>"
                )

        body = "\n".join(parts)
        return (
            "<!DOCTYPE html>"
            "<html><body>"
            "<table width=\"600\" align=\"center\" cellpadding=\"20\">"
            "<tr><td>"
            f"{body}"
            "</td></tr>"
            "</table>"
            "</body></html>"
        )

    async def _send_via_sendgrid(
        self,
        html_content: str,
        subject: str,
        list_id: Optional[str],
        send_immediately: bool,
    ) -> Dict[str, Any]:
        """
        Create a SendGrid Marketing Campaign draft (or send immediately).

        Uses the SendGrid Marketing Campaigns v3 API.
        """
        payload: Dict[str, Any] = {
            "name": subject,
            "subject": subject,
            "html_content": html_content,
            "send_to": {"list_ids": [list_id]} if list_id else {},
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SENDGRID_API_BASE}/marketing/campaigns",
                    headers={
                        "Authorization": f"Bearer {self._sendgrid_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )
            if response.status_code == 429:
                raise RateLimitError("SendGrid rate limit hit.")
            response.raise_for_status()
        except RateLimitError as exc:
            return self._build_error_response(
                str(exc),
                extra={"campaign_id": None, "status": "rate_limited", "preview_url": None},
            )
        except Exception as exc:
            logger.error("SendGrid campaign creation failed: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={"campaign_id": None, "status": "failed", "preview_url": None},
            )

        data = response.json()
        campaign_id = str(data.get("id", ""))
        status = "sent" if send_immediately else "draft"

        logger.info(
            "SendGrid campaign created: id={cid}, status={s}",
            cid=campaign_id,
            s=status,
        )
        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": status,
            "preview_url": None,
            "error": None,
        }

    async def _send_via_mailchimp(
        self,
        html_content: str,
        subject: str,
        list_id: Optional[str],
        send_immediately: bool,
    ) -> Dict[str, Any]:
        """
        Create a Mailchimp campaign draft and optionally send it.

        Two-step: POST /campaigns → PUT /campaigns/{id}/content.
        Sends only if send_immediately=True.
        """
        server = self._mailchimp_server or "us1"
        base_url = f"https://{server}.api.mailchimp.com/3.0"
        auth = ("anystring", self._mailchimp_key or "")

        campaign_payload: Dict[str, Any] = {
            "type": "regular",
            "settings": {
                "subject_line": subject,
                "from_name": self._config.get("from_name", "Newsletter"),
                "reply_to": self._config.get("reply_to", "noreply@example.com"),
            },
        }
        if list_id:
            campaign_payload["recipients"] = {"list_id": list_id}

        try:
            async with httpx.AsyncClient() as client:
                create_response = await client.post(
                    f"{base_url}/campaigns",
                    auth=auth,
                    json=campaign_payload,
                    timeout=30.0,
                )
                if create_response.status_code == 429:
                    raise RateLimitError("Mailchimp rate limit hit during campaign creation.")
                create_response.raise_for_status()

                campaign_id = create_response.json().get("id", "")

                content_response = await client.put(
                    f"{base_url}/campaigns/{campaign_id}/content",
                    auth=auth,
                    json={"html": html_content},
                    timeout=30.0,
                )
                content_response.raise_for_status()

                if send_immediately:
                    send_response = await client.post(
                        f"{base_url}/campaigns/{campaign_id}/actions/send",
                        auth=auth,
                        timeout=30.0,
                    )
                    send_response.raise_for_status()

        except RateLimitError as exc:
            return self._build_error_response(
                str(exc),
                extra={"campaign_id": None, "status": "rate_limited", "preview_url": None},
            )
        except Exception as exc:
            logger.error("Mailchimp campaign failed: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={"campaign_id": None, "status": "failed", "preview_url": None},
            )

        status = "sent" if send_immediately else "draft"
        logger.info(
            "Mailchimp campaign created: id={cid}, status={s}",
            cid=campaign_id,
            s=status,
        )
        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": status,
            "preview_url": f"{base_url}/campaigns/{campaign_id}",
            "error": None,
        }

    def _send_via_smtp(
        self,
        html_content: str,
        subject: str,
        send_immediately: bool,
    ) -> Dict[str, Any]:
        """
        Send content via SMTP.

        When send_immediately=False, this method skips sending and returns
        a draft-saved response. SMTP has no native draft concept — draft
        mode simply defers until the caller passes send_immediately=True.
        """
        if not send_immediately:
            return {
                "success": True,
                "campaign_id": None,
                "status": "draft",
                "preview_url": None,
                "error": None,
            }

        host = os.environ.get("SMTP_HOST", "")
        port = int(os.environ.get("SMTP_PORT", "587"))
        username = os.environ.get("SMTP_USERNAME", "")
        password = os.environ.get("SMTP_PASSWORD", "")
        from_addr = os.environ.get("SMTP_FROM_ADDRESS", "")
        to_addr = os.environ.get("SMTP_TO_ADDRESS", "")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.attach(MIMEText(html_content, "html"))

        try:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.login(username, password)
                server.sendmail(from_addr, [to_addr], msg.as_string())
        except Exception as exc:
            logger.error("SMTP send failed: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={"campaign_id": None, "status": "failed", "preview_url": None},
            )

        logger.info("SMTP email sent to {to}", to=to_addr)
        return {
            "success": True,
            "campaign_id": None,
            "status": "sent",
            "preview_url": None,
            "error": None,
        }

    async def _authenticate_sendgrid(self) -> bool:
        """Validate SendGrid API key via GET /v3/marketing/lists."""
        if self.guard_unconfigured_env("SENDGRID_API_KEY"):
            return False
        key = os.environ["SENDGRID_API_KEY"]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SENDGRID_API_BASE}/marketing/lists",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
            if response.status_code in (401, 403):
                logger.error("SendGrid API key rejected ({s}).", s=response.status_code)
                return False
            response.raise_for_status()
        except Exception as exc:
            logger.error("SendGrid auth failed: {exc}", exc=exc)
            return False
        self._sendgrid_key = key
        self._authenticated = True
        logger.info("SendGrid authenticated.")
        return True

    async def _authenticate_mailchimp(self) -> bool:
        """Validate Mailchimp API key via GET /3.0/ping."""
        if self.guard_unconfigured_env("MAILCHIMP_API_KEY", "MAILCHIMP_SERVER_PREFIX"):
            return False
        key = os.environ["MAILCHIMP_API_KEY"]
        server = os.environ["MAILCHIMP_SERVER_PREFIX"]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://{server}.api.mailchimp.com/3.0/ping",
                    auth=("anystring", key),
                    timeout=10.0,
                )
            if response.status_code in (401, 403):
                logger.error("Mailchimp API key rejected ({s}).", s=response.status_code)
                return False
            response.raise_for_status()
        except Exception as exc:
            logger.error("Mailchimp auth failed: {exc}", exc=exc)
            return False
        self._mailchimp_key = key
        self._mailchimp_server = server
        self._authenticated = True
        logger.info("Mailchimp authenticated.")
        return True

    def _authenticate_smtp(self) -> bool:
        """Validate SMTP credentials by opening a test connection."""
        required = ["SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_ADDRESS"]
        if self.guard_unconfigured_env(*required):
            return False
        host = os.environ["SMTP_HOST"]
        port = int(os.environ.get("SMTP_PORT", "587"))
        username = os.environ["SMTP_USERNAME"]
        password = os.environ["SMTP_PASSWORD"]
        try:
            with smtplib.SMTP(host, port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(username, password)
        except Exception as exc:
            logger.error("SMTP auth failed: {exc}", exc=exc)
            return False
        self._authenticated = True
        logger.info("SMTP authenticated: host={h}:{p}", h=host, p=port)
        return True

    @staticmethod
    def _extract_subject(content: str) -> str:
        """Extract email subject from first heading or first line."""
        for line in content.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            if stripped:
                return stripped[:80]
        return "Newsletter"

    @staticmethod
    def _esc(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


def _register() -> None:
    """Register EmailRepurposer with the global RepurposerRegistry."""
    RepurposerRegistry.register("email", EmailRepurposer)
    logger.debug("EmailRepurposer registered.")


_register()
