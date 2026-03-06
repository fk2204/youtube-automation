"""
JobResult — data class for a single platform distribution attempt.

One JobResult is produced per platform per distribute() call.
The orchestrator collects them into a list and returns them to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class JobResult:
  """
  Captures the outcome of distributing content to one platform.

  status values:
    "success"   — platform accepted the content.
    "failed"    — platform rejected or an error occurred.
    "skipped"   — platform is disabled, blocked, or not applicable.
    "simulated" — simulation_mode=True; no real API call was made.
  """

  platform: str
  status: str  # "success" | "failed" | "skipped" | "simulated"
  started_at: datetime = field(default_factory=datetime.utcnow)
  completed_at: Optional[datetime] = None
  post_id: Optional[str] = None
  post_url: Optional[str] = None
  error_code: Optional[str] = None
  error_message: Optional[str] = None
  retry_count: int = 0
  simulated: bool = False
  metadata: Dict[str, Any] = field(default_factory=dict)

  def is_success(self) -> bool:
    """Return True when the platform accepted the content without error."""
    return self.status == "success"

  def to_dict(self) -> Dict[str, Any]:
    """
    Convert to a JSON-serializable dict for logging and reporting.

    datetime fields are converted to ISO 8601 strings.
    """
    return {
      "platform": self.platform,
      "status": self.status,
      "started_at": self.started_at.isoformat() if self.started_at else None,
      "completed_at": self.completed_at.isoformat() if self.completed_at else None,
      "post_id": self.post_id,
      "post_url": self.post_url,
      "error_code": self.error_code,
      "error_message": self.error_message,
      "retry_count": self.retry_count,
      "simulated": self.simulated,
      "metadata": self.metadata,
    }
