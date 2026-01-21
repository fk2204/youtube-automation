"""
Compliance Module - AI disclosure and platform compliance.

Provides compliance tracking for AI-generated content:
- AIDisclosureTracker: Track AI usage and generate disclosure metadata
- YouTube AI disclosure requirements for altered/synthetic content
- Automatic disclosure text generation

Integrated (2026-01-20) into:
- Pipeline Orchestrator for automatic disclosure tracking
- Uploader for description disclaimer
"""

from .ai_disclosure import (
    AIDisclosureTracker,
    AIContentType,
    DisclosureLevel,
    DisclosureMetadata,
)

__all__ = [
    "AIDisclosureTracker",
    "AIContentType",
    "DisclosureLevel",
    "DisclosureMetadata",
]
