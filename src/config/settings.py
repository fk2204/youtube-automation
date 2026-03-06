"""
Config loader for the social distribution system.

Reads distribution.yaml and returns a plain dict.
All callers depend on this as the single source of config truth.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from loguru import logger

try:
  import yaml
  _YAML_AVAILABLE = True
except ImportError:
  _YAML_AVAILABLE = False

# Default retry policy used when the yaml file is missing or has no retry key.
DEFAULT_RETRY_CONFIG: Dict[str, Any] = {
  "max_attempts": 3,
  "base_delay_seconds": 1.0,
  "max_delay_seconds": 60.0,
  "backoff_factor": 2.0,
}


def load_distribution_config(config_path: str) -> Dict[str, Any]:
  """
  Load and return the distribution config from a YAML file.

  Falls back to an empty platforms dict when the file is missing or
  when PyYAML is not installed, so the orchestrator never crashes at init.

  Args:
    config_path: Absolute or relative path to distribution.yaml.

  Returns:
    Parsed config dict. Always contains at least "platforms" and "retry" keys.
  """
  if not _YAML_AVAILABLE:
    logger.warning(
      "PyYAML not installed. Returning empty config. "
      "Install with: pip install pyyaml"
    )
    return {"platforms": {}, "retry": DEFAULT_RETRY_CONFIG}

  abs_path = os.path.abspath(config_path)

  if not os.path.exists(abs_path):
    logger.warning(
      "distribution.yaml not found at {path}. Returning empty config.",
      path=abs_path,
    )
    return {"platforms": {}, "retry": DEFAULT_RETRY_CONFIG}

  try:
    with open(abs_path, "r", encoding="utf-8") as fh:
      config: Dict[str, Any] = yaml.safe_load(fh) or {}
  except Exception as exc:
    logger.error(
      "Failed to parse distribution.yaml at {path}: {exc}",
      path=abs_path,
      exc=exc,
    )
    return {"platforms": {}, "retry": DEFAULT_RETRY_CONFIG}

  if "platforms" not in config:
    config["platforms"] = {}

  if "retry" not in config:
    config["retry"] = DEFAULT_RETRY_CONFIG

  logger.debug(
    "Loaded distribution config from {path} ({n} platforms).",
    path=abs_path,
    n=len(config["platforms"]),
  )
  return config
