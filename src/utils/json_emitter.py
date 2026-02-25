import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def emit(msg_type: str, **kwargs) -> None:
    """
    Emit a JSON message to stdout for Node.js bridge consumption.

    Always include timestamp. Flush immediately for real-time visibility.

    Args:
        msg_type: one of "agent_thinking", "progress", "token_usage", "result", "error"
        **kwargs: type-specific fields (agent, percent, message, data, etc)
    """
    payload = {
        "type": msg_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs,
    }
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()  # Critical: ensures real-time delivery with PYTHONUNBUFFERED=1
