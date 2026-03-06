"""
SubstackRepurposer — generates HTML draft files for manual Substack publishing.

AUTOMATION_LEVEL: content_only

Substack has no public write API for publishing posts programmatically.
This repurposer does NOT call any Substack endpoint. Instead it:
  1. Transforms source content into a well-structured HTML file.
  2. Writes the file to a configured export directory.
  3. Returns the draft path so the operator can open it, copy the HTML,
     and paste into the Substack editor manually.

Decision #2 from Wave 2 Blueprint: Substack = content_only, no API calls.

Operator workflow after repurpose():
  1. Open the draft_path HTML file.
  2. Log in to substack.com/publish.
  3. Create new post → switch to HTML mode → paste content.
  4. Review and publish.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from distribution.repurposers.base_repurposer import BaseRepurposer
from distribution.repurposers.repurposer_registry import RepurposerRegistry

AUTOMATION_LEVEL = "content_only"


class SubstackRepurposer(BaseRepurposer):
    """
    Generates HTML draft files for manual Substack publishing.

    No Substack API calls are made — see module docstring for operator workflow.

    Config keys:
        export_path:  Directory path where HTML drafts will be written.
                      Must be writable. authenticate() validates this.
    """

    PLATFORM_NAME = "substack"
    AUTOMATION_LEVEL = "content_only"

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with config dict.

        Config keys:
            export_path: str — directory where draft HTML files are saved.
        """
        super().__init__(config)
        self._export_path: str = config.get("export_path", "")

    @property
    def platform_name(self) -> str:
        return self.PLATFORM_NAME

    async def authenticate(self) -> bool:
        """
        Validate that export_path exists and is writable.

        No network call is made. Returns True/False without raising.

        Returns:
            True if export_path is a writable directory, False otherwise.
        """
        if not self._export_path:
            logger.error("SubstackRepurposer: export_path not set in config.")
            return False

        export_dir = Path(self._export_path)

        if not export_dir.exists():
            try:
                export_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "SubstackRepurposer: created export_path at {p}",
                    p=str(export_dir),
                )
            except Exception as exc:
                logger.error(
                    "SubstackRepurposer: cannot create export_path {p}: {exc}",
                    p=str(export_dir),
                    exc=exc,
                )
                return False

        if not os.access(str(export_dir), os.W_OK):
            logger.error(
                "SubstackRepurposer: export_path {p} is not writable.",
                p=str(export_dir),
            )
            return False

        self._authenticated = True
        logger.info(
            "SubstackRepurposer: export_path validated: {p}",
            p=str(export_dir),
        )
        return True

    async def repurpose(
        self,
        source_content: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Write source content as an HTML draft file to export_path.

        Does NOT call any Substack endpoint.

        Args:
            source_content: Article body in Markdown or plain text.
            **kwargs:        Unused — accepted for interface compatibility.

        Returns:
            {
                "success": bool,
                "draft_path": str | None,
                "status": "draft_saved",
                "action_required": "manual_publish",
                "error": str | None,
            }
        """
        if self.simulation_mode:
            sim = self._simulate_repurpose({"content_length": len(source_content)})
            sim.update({
                "draft_path": "/tmp/substack_sim_draft.html",
                "status": "draft_saved",
                "action_required": "manual_publish",
            })
            return sim

        if not self._authenticated:
            return self._build_error_response(
                "Not authenticated. Call authenticate() first.",
                extra={
                    "draft_path": None,
                    "status": "failed",
                    "action_required": "manual_publish",
                },
            )

        html_content = self._transform_to_substack_html(source_content)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"substack_draft_{timestamp}.html"
        draft_path = os.path.join(self._export_path, filename)

        try:
            with open(draft_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as exc:
            logger.error("SubstackRepurposer: failed to write draft: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={
                    "draft_path": None,
                    "status": "failed",
                    "action_required": "manual_publish",
                },
            )

        logger.info(
            "SubstackRepurposer: draft saved to {p}",
            p=draft_path,
        )
        return {
            "success": True,
            "draft_path": draft_path,
            "status": "draft_saved",
            "action_required": "manual_publish",
            "error": None,
        }

    def _transform_to_substack_html(self, content: str) -> str:
        """
        Transform source content into a Substack-compatible HTML document.

        Converts markdown headings, paragraphs, and bold/italic markers
        to HTML. Returns a full HTML document string.
        """
        lines = content.strip().splitlines()
        html_parts: list[str] = []
        title = "Draft"

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("# ") and i == 0:
                title = stripped[2:].strip()
                html_parts.append(f"<h1>{self._escape_html(title)}</h1>")
            elif stripped.startswith("# "):
                html_parts.append(f"<h1>{self._escape_html(stripped[2:])}</h1>")
            elif stripped.startswith("## "):
                html_parts.append(f"<h2>{self._escape_html(stripped[3:])}</h2>")
            elif stripped.startswith("### "):
                html_parts.append(f"<h3>{self._escape_html(stripped[4:])}</h3>")
            elif stripped == "":
                html_parts.append("")
            else:
                html_parts.append(f"<p>{self._escape_html(stripped)}</p>")

        body = "\n".join(html_parts)
        return (
            f"<!DOCTYPE html>\n"
            f"<html lang=\"en\">\n"
            f"<head>\n"
            f"  <meta charset=\"UTF-8\">\n"
            f"  <title>{self._escape_html(title)}</title>\n"
            f"</head>\n"
            f"<body>\n"
            f"{body}\n"
            f"</body>\n"
            f"</html>"
        )

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape special HTML characters in text."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


def _register() -> None:
    """Register SubstackRepurposer with the global RepurposerRegistry."""
    RepurposerRegistry.register("substack", SubstackRepurposer)
    logger.debug("SubstackRepurposer registered.")


_register()
