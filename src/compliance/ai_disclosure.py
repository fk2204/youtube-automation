"""
AI Disclosure Compliance System

Handle YouTube's 2025 AI content disclosure requirements.
Automatically track and inject proper metadata for uploads.

YouTube requires disclosure for:
- Synthetic/realistic voices (TTS, voice cloning)
- AI-generated realistic visuals (faces, scenes)
- Altered content that appears realistic
- Deepfakes and synthetic media

This module:
1. Tracks which content uses AI generation
2. Generates proper disclosure metadata
3. Injects disclosures into upload metadata
4. Maintains compliance records

Usage:
    tracker = AIDisclosureTracker()

    # Track AI usage
    tracker.track_voice_generation(video_id="vid123", method="edge-tts")
    tracker.track_visual_generation(video_id="vid123", method="stock_footage")

    # Get disclosure metadata for upload
    disclosure = tracker.get_disclosure_metadata("vid123")

    # Upload with disclosure
    uploader.upload_video(..., ai_disclosure=disclosure)
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
from loguru import logger


class AIContentType(Enum):
    """Types of AI-generated content that require disclosure."""

    # Voice/Audio
    SYNTHETIC_VOICE = "synthetic_voice"  # TTS, voice cloning
    VOICE_ALTERATION = "voice_alteration"  # Pitch, speed changes beyond normal

    # Visual
    SYNTHETIC_VISUALS = "synthetic_visuals"  # AI-generated images/video
    DEEPFAKE = "deepfake"  # Face swaps, synthetic faces
    ALTERED_FOOTAGE = "altered_footage"  # Significant AI alterations

    # Text
    AI_SCRIPT = "ai_script"  # AI-generated scripts (lower risk, but good to track)

    # Other
    SYNTHETIC_AUDIO = "synthetic_audio"  # AI music, sound effects


class DisclosureLevel(Enum):
    """Disclosure requirement levels."""
    NONE = "none"  # No disclosure needed
    OPTIONAL = "optional"  # Good practice but not required
    RECOMMENDED = "recommended"  # Should disclose
    REQUIRED = "required"  # Must disclose per YouTube policy


@dataclass
class AIUsageRecord:
    """Record of AI usage in content."""
    video_id: str
    content_type: AIContentType
    method: str  # e.g., "edge-tts", "whisper", "replicate-face"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict = field(default_factory=dict)
    disclosure_level: DisclosureLevel = DisclosureLevel.RECOMMENDED

    def to_dict(self) -> Dict:
        return {
            "video_id": self.video_id,
            "content_type": self.content_type.value,
            "method": self.method,
            "timestamp": self.timestamp,
            "details": self.details,
            "disclosure_level": self.disclosure_level.value
        }


@dataclass
class DisclosureMetadata:
    """YouTube disclosure metadata."""
    requires_disclosure: bool
    disclosure_text: str
    content_types_used: List[str]
    methods_used: List[str]
    disclosure_level: DisclosureLevel
    detailed_breakdown: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "requires_disclosure": self.requires_disclosure,
            "disclosure_text": self.disclosure_text,
            "content_types_used": self.content_types_used,
            "methods_used": self.methods_used,
            "disclosure_level": self.disclosure_level.value,
            "detailed_breakdown": self.detailed_breakdown
        }

    def get_description_disclaimer(self) -> str:
        """Get disclaimer text to append to video description."""
        if not self.requires_disclosure:
            return ""

        disclaimer = "\n\n" + "─" * 50 + "\n"
        disclaimer += "📢 AI Disclosure\n"
        disclaimer += "─" * 50 + "\n"
        disclaimer += self.disclosure_text + "\n"

        if self.methods_used:
            disclaimer += "\nTools used: " + ", ".join(self.methods_used) + "\n"

        return disclaimer


class AIDisclosureTracker:
    """
    Track AI content usage and generate YouTube disclosure metadata.

    Complies with YouTube's 2025 AI disclosure requirements.
    """

    # Disclosure rules by content type
    DISCLOSURE_RULES = {
        AIContentType.SYNTHETIC_VOICE: {
            "level": DisclosureLevel.RECOMMENDED,
            "text": "This video uses AI-generated voiceover (text-to-speech).",
            "youtube_flag": "altered_content"
        },
        AIContentType.VOICE_ALTERATION: {
            "level": DisclosureLevel.OPTIONAL,
            "text": "Audio has been processed for clarity.",
            "youtube_flag": None
        },
        AIContentType.SYNTHETIC_VISUALS: {
            "level": DisclosureLevel.REQUIRED,
            "text": "This video contains AI-generated realistic visuals.",
            "youtube_flag": "synthetic_content"
        },
        AIContentType.DEEPFAKE: {
            "level": DisclosureLevel.REQUIRED,
            "text": "This video contains synthetic media (deepfake/AI-generated faces).",
            "youtube_flag": "synthetic_content"
        },
        AIContentType.ALTERED_FOOTAGE: {
            "level": DisclosureLevel.REQUIRED,
            "text": "Footage in this video has been altered using AI.",
            "youtube_flag": "altered_content"
        },
        AIContentType.AI_SCRIPT: {
            "level": DisclosureLevel.OPTIONAL,
            "text": "Script was generated with AI assistance.",
            "youtube_flag": None
        },
        AIContentType.SYNTHETIC_AUDIO: {
            "level": DisclosureLevel.RECOMMENDED,
            "text": "Background music/audio generated using AI.",
            "youtube_flag": "altered_content"
        }
    }

    # Common TTS methods and their disclosure needs
    TTS_METHODS = {
        "edge-tts": DisclosureLevel.RECOMMENDED,
        "fish-audio": DisclosureLevel.RECOMMENDED,
        "elevenlabs": DisclosureLevel.REQUIRED,  # Very realistic
        "chatterbox": DisclosureLevel.RECOMMENDED,
        "natural-voice": DisclosureLevel.RECOMMENDED
    }

    def __init__(self, db_path: str = "data/compliance/ai_disclosure.db"):
        """
        Initialize AI disclosure tracker.

        Args:
            db_path: Path to SQLite database for tracking
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("[AIDisclosureTracker] Initialized")

    def _init_database(self):
        """Initialize SQLite database for tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    method TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    disclosure_level TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_video_id ON ai_usage(video_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON ai_usage(timestamp)
            """)

    def track_voice_generation(
        self,
        video_id: str,
        method: str,
        voice_name: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """
        Track TTS/voice generation usage.

        Args:
            video_id: Unique video identifier
            method: TTS method used (e.g., "edge-tts", "fish-audio")
            voice_name: Voice name if applicable
            details: Additional details
        """
        disclosure_level = self.TTS_METHODS.get(method, DisclosureLevel.RECOMMENDED)

        extra_details = details or {}
        if voice_name:
            extra_details["voice_name"] = voice_name

        record = AIUsageRecord(
            video_id=video_id,
            content_type=AIContentType.SYNTHETIC_VOICE,
            method=method,
            details=extra_details,
            disclosure_level=disclosure_level
        )

        self._save_record(record)
        logger.info(f"[AIDisclosure] Tracked voice generation: {method} for {video_id}")

    def track_visual_generation(
        self,
        video_id: str,
        method: str,
        content_type: AIContentType = AIContentType.SYNTHETIC_VISUALS,
        details: Optional[Dict] = None
    ):
        """
        Track AI visual generation.

        Args:
            video_id: Unique video identifier
            method: Method used (e.g., "replicate-face", "midjourney")
            content_type: Type of visual content
            details: Additional details
        """
        # Determine disclosure level
        rule = self.DISCLOSURE_RULES.get(content_type, {})
        disclosure_level = rule.get("level", DisclosureLevel.RECOMMENDED)

        record = AIUsageRecord(
            video_id=video_id,
            content_type=content_type,
            method=method,
            details=details or {},
            disclosure_level=disclosure_level
        )

        self._save_record(record)
        logger.info(f"[AIDisclosure] Tracked visual generation: {method} for {video_id}")

    def track_script_generation(
        self,
        video_id: str,
        ai_provider: str,
        model: Optional[str] = None
    ):
        """
        Track AI script generation.

        Args:
            video_id: Unique video identifier
            ai_provider: AI provider (e.g., "groq", "claude", "ollama")
            model: Model name if applicable
        """
        details = {}
        if model:
            details["model"] = model

        record = AIUsageRecord(
            video_id=video_id,
            content_type=AIContentType.AI_SCRIPT,
            method=ai_provider,
            details=details,
            disclosure_level=DisclosureLevel.OPTIONAL
        )

        self._save_record(record)
        logger.info(f"[AIDisclosure] Tracked script generation: {ai_provider} for {video_id}")

    def track_audio_generation(
        self,
        video_id: str,
        method: str,
        audio_type: str = "background_music"
    ):
        """
        Track AI audio generation (music, sound effects).

        Args:
            video_id: Unique video identifier
            method: Method used
            audio_type: Type of audio
        """
        record = AIUsageRecord(
            video_id=video_id,
            content_type=AIContentType.SYNTHETIC_AUDIO,
            method=method,
            details={"audio_type": audio_type},
            disclosure_level=DisclosureLevel.RECOMMENDED
        )

        self._save_record(record)
        logger.info(f"[AIDisclosure] Tracked audio generation: {method} for {video_id}")

    def get_disclosure_metadata(self, video_id: str) -> DisclosureMetadata:
        """
        Get disclosure metadata for a video.

        Args:
            video_id: Video identifier

        Returns:
            DisclosureMetadata with requirements and text
        """
        records = self._get_records(video_id)

        if not records:
            return DisclosureMetadata(
                requires_disclosure=False,
                disclosure_text="No AI disclosure required.",
                content_types_used=[],
                methods_used=[],
                disclosure_level=DisclosureLevel.NONE
            )

        # Determine highest disclosure level
        max_level = max(
            (r.disclosure_level for r in records),
            key=lambda x: list(DisclosureLevel).index(x)
        )

        # Check if disclosure is required
        requires_disclosure = max_level in [
            DisclosureLevel.REQUIRED,
            DisclosureLevel.RECOMMENDED
        ]

        # Collect unique content types and methods
        content_types = list(set(r.content_type.value for r in records))
        methods = list(set(r.method for r in records))

        # Build disclosure text
        disclosure_lines = []
        content_type_groups = {}

        for record in records:
            ct = record.content_type
            if ct not in content_type_groups:
                content_type_groups[ct] = []
            content_type_groups[ct].append(record)

        for content_type, type_records in content_type_groups.items():
            rule = self.DISCLOSURE_RULES.get(content_type, {})
            text = rule.get("text", "AI-generated content used.")
            disclosure_lines.append(text)

        disclosure_text = " ".join(disclosure_lines)

        # Build detailed breakdown
        breakdown = {}
        for ct, type_records in content_type_groups.items():
            breakdown[ct.value] = {
                "count": len(type_records),
                "methods": list(set(r.method for r in type_records)),
                "disclosure_level": self.DISCLOSURE_RULES.get(ct, {}).get("level", DisclosureLevel.OPTIONAL).value
            }

        metadata = DisclosureMetadata(
            requires_disclosure=requires_disclosure,
            disclosure_text=disclosure_text,
            content_types_used=content_types,
            methods_used=methods,
            disclosure_level=max_level,
            detailed_breakdown=breakdown
        )

        logger.success(
            f"[AIDisclosure] Generated metadata for {video_id}: "
            f"Disclosure {'REQUIRED' if requires_disclosure else 'optional'}"
        )

        return metadata

    def get_youtube_metadata_fields(self, video_id: str) -> Dict:
        """
        Get YouTube API metadata fields for AI disclosure.

        Returns dict to merge into upload metadata.

        Args:
            video_id: Video identifier

        Returns:
            Dict with YouTube metadata fields
        """
        disclosure = self.get_disclosure_metadata(video_id)

        if not disclosure.requires_disclosure:
            return {}

        # YouTube's AI disclosure fields (as of 2025)
        metadata = {}

        # Add disclaimer to description
        metadata["description_disclaimer"] = disclosure.get_description_disclaimer()

        # Flag for altered/synthetic content
        if any(
            ct in disclosure.content_types_used
            for ct in [
                AIContentType.SYNTHETIC_VISUALS.value,
                AIContentType.DEEPFAKE.value,
                AIContentType.ALTERED_FOOTAGE.value
            ]
        ):
            metadata["contains_synthetic_media"] = True

        # Flag for AI voice
        if AIContentType.SYNTHETIC_VOICE.value in disclosure.content_types_used:
            metadata["contains_synthetic_voice"] = True

        return metadata

    def clear_records(self, video_id: str):
        """
        Clear AI usage records for a video.

        Args:
            video_id: Video identifier
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM ai_usage WHERE video_id = ?", (video_id,))
        logger.info(f"[AIDisclosure] Cleared records for {video_id}")

    def get_all_tracked_videos(self) -> List[str]:
        """Get list of all tracked video IDs."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT video_id FROM ai_usage").fetchall()
        return [row[0] for row in rows]

    def get_compliance_report(self, days: int = 30) -> Dict:
        """
        Generate compliance report for recent videos.

        Args:
            days: Number of days to include

        Returns:
            Report dictionary
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT video_id, content_type, method, disclosure_level FROM ai_usage WHERE timestamp >= ?",
                (cutoff_iso,)
            ).fetchall()

        if not rows:
            return {
                "period_days": days,
                "videos_tracked": 0,
                "total_ai_usage": 0,
                "disclosure_breakdown": {}
            }

        video_ids = set(row[0] for row in rows)
        disclosure_counts = {}

        for level in DisclosureLevel:
            count = sum(1 for row in rows if row[3] == level.value)
            disclosure_counts[level.value] = count

        content_type_counts = {}
        for row in rows:
            ct = row[1]
            content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

        return {
            "period_days": days,
            "videos_tracked": len(video_ids),
            "total_ai_usage": len(rows),
            "disclosure_breakdown": disclosure_counts,
            "content_type_breakdown": content_type_counts,
            "most_common_methods": self._get_most_common_methods(rows)
        }

    def _save_record(self, record: AIUsageRecord):
        """Save usage record to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO ai_usage (video_id, content_type, method, timestamp, details, disclosure_level)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.video_id,
                record.content_type.value,
                record.method,
                record.timestamp,
                json.dumps(record.details),
                record.disclosure_level.value
            ))

    def _get_records(self, video_id: str) -> List[AIUsageRecord]:
        """Get all usage records for a video."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT video_id, content_type, method, timestamp, details, disclosure_level FROM ai_usage WHERE video_id = ?",
                (video_id,)
            ).fetchall()

        records = []
        for row in rows:
            records.append(AIUsageRecord(
                video_id=row[0],
                content_type=AIContentType(row[1]),
                method=row[2],
                timestamp=row[3],
                details=json.loads(row[4]) if row[4] else {},
                disclosure_level=DisclosureLevel(row[5])
            ))

        return records

    def _get_most_common_methods(self, rows: List, limit: int = 5) -> List[Dict]:
        """Get most common AI methods used."""
        method_counts = {}
        for row in rows:
            method = row[2]
            method_counts[method] = method_counts.get(method, 0) + 1

        sorted_methods = sorted(
            method_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {"method": method, "count": count}
            for method, count in sorted_methods[:limit]
        ]


# Convenience functions
def create_disclosure_tracker(db_path: str = "data/compliance/ai_disclosure.db") -> AIDisclosureTracker:
    """Create a new disclosure tracker instance."""
    return AIDisclosureTracker(db_path=db_path)


# CLI entry point
def main():
    """CLI entry point for AI disclosure management."""
    import sys

    if len(sys.argv) < 2:
        print("""
AI Disclosure Compliance Tracker

Commands:
    track-voice <video_id> <method> [--voice <name>]
        Track TTS usage

    track-visual <video_id> <method> [--type <type>]
        Track visual AI usage

    track-script <video_id> <provider> [--model <model>]
        Track script generation

    get-disclosure <video_id>
        Get disclosure metadata for video

    report [--days <n>]
        Generate compliance report

    list
        List all tracked videos

Examples:
    python -m src.compliance.ai_disclosure track-voice vid123 edge-tts --voice en-US-GuyNeural
    python -m src.compliance.ai_disclosure track-visual vid123 stock_footage
    python -m src.compliance.ai_disclosure get-disclosure vid123
    python -m src.compliance.ai_disclosure report --days 30
        """)
        return

    tracker = AIDisclosureTracker()
    cmd = sys.argv[1]

    if cmd == "track-voice" and len(sys.argv) >= 4:
        video_id = sys.argv[2]
        method = sys.argv[3]
        voice = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--voice" else None
        tracker.track_voice_generation(video_id, method, voice)
        print(f"Tracked voice generation for {video_id}")

    elif cmd == "track-visual" and len(sys.argv) >= 4:
        video_id = sys.argv[2]
        method = sys.argv[3]
        tracker.track_visual_generation(video_id, method)
        print(f"Tracked visual generation for {video_id}")

    elif cmd == "track-script" and len(sys.argv) >= 4:
        video_id = sys.argv[2]
        provider = sys.argv[3]
        model = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--model" else None
        tracker.track_script_generation(video_id, provider, model)
        print(f"Tracked script generation for {video_id}")

    elif cmd == "get-disclosure" and len(sys.argv) >= 3:
        video_id = sys.argv[2]
        disclosure = tracker.get_disclosure_metadata(video_id)
        print(json.dumps(disclosure.to_dict(), indent=2))

    elif cmd == "report":
        days = 30
        if len(sys.argv) > 2 and sys.argv[2] == "--days":
            days = int(sys.argv[3])
        report = tracker.get_compliance_report(days)
        print(json.dumps(report, indent=2))

    elif cmd == "list":
        videos = tracker.get_all_tracked_videos()
        print(f"Tracked videos ({len(videos)}):")
        for vid in videos:
            print(f"  - {vid}")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
