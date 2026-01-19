"""
Real-time performance monitoring with alerts for YouTube videos.

This module provides:
- Performance threshold monitoring (CTR, retention)
- Alert generation with actionable recommendations
- Discord/Slack webhook integration
- Channel-wide performance tracking
"""

import os
import json
import requests
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from loguru import logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of performance alerts."""
    LOW_CTR = "low_ctr"
    LOW_RETENTION = "low_retention"
    UNDERPERFORMING = "underperforming"
    VIRAL_POTENTIAL = "viral_potential"


@dataclass
class PerformanceAlert:
    """
    Represents a performance alert for a video.

    Attributes:
        video_id: YouTube video ID
        alert_type: Type of alert (low_ctr, low_retention, etc.)
        severity: Alert severity (info, warning, critical)
        message: Human-readable alert message
        metric_value: Current value of the metric
        threshold_value: Threshold that was crossed
        recommendation: Actionable recommendation to improve performance
        timestamp: When the alert was generated
        video_title: Optional video title for context
        channel_id: Optional channel ID
    """
    video_id: str
    alert_type: str
    severity: str
    message: str
    metric_value: float
    threshold_value: float
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)
    video_title: Optional[str] = None
    channel_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_webhook_payload(self) -> Dict[str, Any]:
        """Convert to webhook-friendly payload (Discord/Slack compatible)."""
        severity_colors = {
            "info": 0x3498db,      # Blue
            "warning": 0xf39c12,   # Orange
            "critical": 0xe74c3c,  # Red
        }
        severity_emojis = {
            "info": ":information_source:",
            "warning": ":warning:",
            "critical": ":rotating_light:",
        }

        color = severity_colors.get(self.severity, 0x95a5a6)
        emoji = severity_emojis.get(self.severity, ":bell:")

        # Discord embed format
        return {
            "embeds": [{
                "title": f"{emoji} Performance Alert: {self.alert_type.upper().replace('_', ' ')}",
                "description": self.message,
                "color": color,
                "fields": [
                    {
                        "name": "Video",
                        "value": f"[{self.video_title or self.video_id}](https://youtube.com/watch?v={self.video_id})",
                        "inline": True
                    },
                    {
                        "name": "Metric Value",
                        "value": f"{self.metric_value:.2f}%",
                        "inline": True
                    },
                    {
                        "name": "Threshold",
                        "value": f"{self.threshold_value:.2f}%",
                        "inline": True
                    },
                    {
                        "name": "Recommendation",
                        "value": self.recommendation,
                        "inline": False
                    }
                ],
                "timestamp": self.timestamp.isoformat(),
                "footer": {
                    "text": f"Channel: {self.channel_id or 'Unknown'}"
                }
            }]
        }

    def __str__(self) -> str:
        """String representation of the alert."""
        return (
            f"[{self.severity.upper()}] {self.alert_type}: {self.message} "
            f"(Value: {self.metric_value:.2f}%, Threshold: {self.threshold_value:.2f}%)"
        )


class PerformanceMonitor:
    """
    Real-time performance monitor for YouTube videos.

    Monitors video metrics and generates alerts when performance
    falls below thresholds or shows viral potential.

    Attributes:
        CTR_WARNING: CTR below this triggers a warning (4%)
        CTR_CRITICAL: CTR below this triggers a critical alert (2%)
        RETENTION_WARNING: Retention below this triggers a warning (35%)
        RETENTION_CRITICAL: Retention below this triggers a critical alert (25%)
        VIRAL_MULTIPLIER: Views above this x channel average triggers viral alert (3x)
    """

    # Alert thresholds (as percentages)
    CTR_WARNING = 4.0
    CTR_CRITICAL = 2.0
    RETENTION_WARNING = 35.0
    RETENTION_CRITICAL = 25.0
    VIRAL_MULTIPLIER = 3.0

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        alert_log_path: Optional[str] = None
    ):
        """
        Initialize the performance monitor.

        Args:
            discord_webhook_url: Discord webhook URL for alerts
            slack_webhook_url: Slack webhook URL for alerts
            alert_log_path: Path to save alert logs (JSON)
        """
        self.discord_webhook_url = discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.alert_log_path = alert_log_path or "logs/performance_alerts.json"

        # Cache for channel averages
        self._channel_averages: Dict[str, Dict[str, float]] = {}

        # Recommendations for each alert type
        self._recommendations = {
            AlertType.LOW_CTR.value: {
                "warning": (
                    "Consider A/B testing your thumbnail and title. "
                    "Try adding curiosity gaps, numbers, or emotional triggers. "
                    "Check if thumbnail text is readable on mobile."
                ),
                "critical": (
                    "URGENT: Your CTR is critically low. Actions to take: "
                    "1) Change thumbnail immediately - use high contrast faces/expressions. "
                    "2) Rewrite title with power words. "
                    "3) Check if topic is saturated - consider different angle."
                ),
            },
            AlertType.LOW_RETENTION.value: {
                "warning": (
                    "Viewers are dropping off early. Consider: "
                    "1) Strengthen your hook in first 30 seconds. "
                    "2) Add pattern interrupts every 30-60 seconds. "
                    "3) Promise specific value early and deliver it."
                ),
                "critical": (
                    "URGENT: Retention is critically low. Actions to take: "
                    "1) Analyze where viewers drop off in YouTube Studio. "
                    "2) Add open loops and tease upcoming content. "
                    "3) Consider shorter, more focused content. "
                    "4) Re-edit intro to hook faster."
                ),
            },
            AlertType.UNDERPERFORMING.value: {
                "warning": (
                    "Video is below channel average. Consider: "
                    "1) Promote on social media/community tab. "
                    "2) Create a Short from best moment. "
                    "3) Reply to all comments to boost engagement."
                ),
                "critical": (
                    "Video is significantly underperforming. Actions: "
                    "1) Analyze successful videos for what works. "
                    "2) Consider updating title/thumbnail. "
                    "3) Cross-promote from other videos via end screens/cards."
                ),
            },
            AlertType.VIRAL_POTENTIAL.value: {
                "info": (
                    "GREAT NEWS! Video is outperforming channel average. "
                    "1) Create follow-up content on this topic. "
                    "2) Pin a comment linking to related videos. "
                    "3) Share widely on all social channels. "
                    "4) Engage with every comment to boost algorithm."
                ),
            },
        }

        logger.info("PerformanceMonitor initialized")

    def check_video(
        self,
        video_id: str,
        hours_since_upload: int,
        metrics: Optional[Dict[str, Any]] = None
    ) -> List[PerformanceAlert]:
        """
        Check video performance and generate alerts.

        Args:
            video_id: YouTube video ID
            hours_since_upload: Hours since the video was uploaded
            metrics: Optional pre-fetched metrics. If None, will fetch from API.
                     Expected keys: ctr, retention, views, title, channel_id

        Returns:
            List of PerformanceAlert objects
        """
        alerts = []

        # Fetch metrics if not provided
        if metrics is None:
            metrics = self._fetch_video_metrics(video_id)

        if not metrics:
            logger.warning(f"Could not fetch metrics for video {video_id}")
            return alerts

        video_title = metrics.get("title", video_id)
        channel_id = metrics.get("channel_id")
        ctr = metrics.get("ctr", 0)
        retention = metrics.get("retention", 0)
        views = metrics.get("views", 0)

        # Check CTR thresholds
        if ctr < self.CTR_CRITICAL:
            alerts.append(PerformanceAlert(
                video_id=video_id,
                alert_type=AlertType.LOW_CTR.value,
                severity=AlertSeverity.CRITICAL.value,
                message=f"CTR is critically low at {ctr:.2f}% (threshold: {self.CTR_CRITICAL}%)",
                metric_value=ctr,
                threshold_value=self.CTR_CRITICAL,
                recommendation=self._recommendations[AlertType.LOW_CTR.value]["critical"],
                video_title=video_title,
                channel_id=channel_id
            ))
        elif ctr < self.CTR_WARNING:
            alerts.append(PerformanceAlert(
                video_id=video_id,
                alert_type=AlertType.LOW_CTR.value,
                severity=AlertSeverity.WARNING.value,
                message=f"CTR is below target at {ctr:.2f}% (threshold: {self.CTR_WARNING}%)",
                metric_value=ctr,
                threshold_value=self.CTR_WARNING,
                recommendation=self._recommendations[AlertType.LOW_CTR.value]["warning"],
                video_title=video_title,
                channel_id=channel_id
            ))

        # Check retention thresholds
        if retention < self.RETENTION_CRITICAL:
            alerts.append(PerformanceAlert(
                video_id=video_id,
                alert_type=AlertType.LOW_RETENTION.value,
                severity=AlertSeverity.CRITICAL.value,
                message=f"Retention is critically low at {retention:.2f}% (threshold: {self.RETENTION_CRITICAL}%)",
                metric_value=retention,
                threshold_value=self.RETENTION_CRITICAL,
                recommendation=self._recommendations[AlertType.LOW_RETENTION.value]["critical"],
                video_title=video_title,
                channel_id=channel_id
            ))
        elif retention < self.RETENTION_WARNING:
            alerts.append(PerformanceAlert(
                video_id=video_id,
                alert_type=AlertType.LOW_RETENTION.value,
                severity=AlertSeverity.WARNING.value,
                message=f"Retention is below target at {retention:.2f}% (threshold: {self.RETENTION_WARNING}%)",
                metric_value=retention,
                threshold_value=self.RETENTION_WARNING,
                recommendation=self._recommendations[AlertType.LOW_RETENTION.value]["warning"],
                video_title=video_title,
                channel_id=channel_id
            ))

        # Check against channel average (if available)
        if channel_id:
            channel_avg = self._get_channel_average(channel_id)
            if channel_avg and channel_avg.get("views", 0) > 0:
                avg_views = channel_avg["views"]

                # Check for viral potential (above 3x average)
                if views > avg_views * self.VIRAL_MULTIPLIER:
                    alerts.append(PerformanceAlert(
                        video_id=video_id,
                        alert_type=AlertType.VIRAL_POTENTIAL.value,
                        severity=AlertSeverity.INFO.value,
                        message=f"Video is going viral! {views:,} views vs {avg_views:,.0f} channel average ({views/avg_views:.1f}x)",
                        metric_value=(views / avg_views) * 100,
                        threshold_value=self.VIRAL_MULTIPLIER * 100,
                        recommendation=self._recommendations[AlertType.VIRAL_POTENTIAL.value]["info"],
                        video_title=video_title,
                        channel_id=channel_id
                    ))

                # Check for underperforming (below 50% of average at 24+ hours)
                elif hours_since_upload >= 24 and views < avg_views * 0.5:
                    severity = AlertSeverity.CRITICAL if views < avg_views * 0.25 else AlertSeverity.WARNING
                    alerts.append(PerformanceAlert(
                        video_id=video_id,
                        alert_type=AlertType.UNDERPERFORMING.value,
                        severity=severity.value,
                        message=f"Video underperforming: {views:,} views vs {avg_views:,.0f} channel average ({views/avg_views:.1%})",
                        metric_value=(views / avg_views) * 100,
                        threshold_value=50.0,
                        recommendation=self._recommendations[AlertType.UNDERPERFORMING.value][severity.value],
                        video_title=video_title,
                        channel_id=channel_id
                    ))

        # Log alerts
        for alert in alerts:
            logger.log(
                "WARNING" if alert.severity == "warning" else
                "ERROR" if alert.severity == "critical" else "INFO",
                str(alert)
            )

        return alerts

    def monitor_all_recent(self, hours: int = 72) -> Dict[str, List[PerformanceAlert]]:
        """
        Monitor all videos uploaded in the last N hours.

        Args:
            hours: Number of hours to look back (default: 72)

        Returns:
            Dictionary mapping video_id to list of alerts
        """
        all_alerts = {}

        # Get recent videos from database or API
        recent_videos = self._get_recent_videos(hours)

        if not recent_videos:
            logger.info(f"No videos found in the last {hours} hours")
            return all_alerts

        logger.info(f"Monitoring {len(recent_videos)} videos from the last {hours} hours")

        for video in recent_videos:
            video_id = video.get("video_id")
            upload_time = video.get("upload_time")

            if not video_id:
                continue

            # Calculate hours since upload
            if isinstance(upload_time, str):
                upload_time = datetime.fromisoformat(upload_time.replace("Z", "+00:00"))

            hours_since = (datetime.now(upload_time.tzinfo if upload_time.tzinfo else None) - upload_time).total_seconds() / 3600

            # Check video performance
            alerts = self.check_video(video_id, int(hours_since), video.get("metrics"))

            if alerts:
                all_alerts[video_id] = alerts
                # Send alerts for this video
                self._send_alerts(video_id, alerts)

        # Save alerts to log
        self._save_alerts_log(all_alerts)

        return all_alerts

    def _send_alerts(self, video_id: str, alerts: List[PerformanceAlert]) -> bool:
        """
        Send alerts to Discord/Slack webhooks.

        Args:
            video_id: Video ID for logging
            alerts: List of alerts to send

        Returns:
            True if alerts were sent successfully
        """
        success = True

        for alert in alerts:
            payload = alert.to_webhook_payload()

            # Send to Discord
            if self.discord_webhook_url:
                try:
                    response = requests.post(
                        self.discord_webhook_url,
                        json=payload,
                        timeout=10
                    )
                    response.raise_for_status()
                    logger.info(f"Discord alert sent for {video_id}: {alert.alert_type}")
                except requests.RequestException as e:
                    logger.error(f"Failed to send Discord alert: {e}")
                    success = False

            # Send to Slack (convert format)
            if self.slack_webhook_url:
                try:
                    slack_payload = self._convert_to_slack_format(alert)
                    response = requests.post(
                        self.slack_webhook_url,
                        json=slack_payload,
                        timeout=10
                    )
                    response.raise_for_status()
                    logger.info(f"Slack alert sent for {video_id}: {alert.alert_type}")
                except requests.RequestException as e:
                    logger.error(f"Failed to send Slack alert: {e}")
                    success = False

        return success

    def _convert_to_slack_format(self, alert: PerformanceAlert) -> Dict[str, Any]:
        """Convert alert to Slack message format."""
        severity_emojis = {
            "info": ":information_source:",
            "warning": ":warning:",
            "critical": ":rotating_light:",
        }
        emoji = severity_emojis.get(alert.severity, ":bell:")

        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Performance Alert: {alert.alert_type.upper().replace('_', ' ')}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": alert.message
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Video:*\n<https://youtube.com/watch?v={alert.video_id}|{alert.video_title or alert.video_id}>"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Metric:* {alert.metric_value:.2f}% (Threshold: {alert.threshold_value:.2f}%)"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Recommendation:*\n{alert.recommendation}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Channel: {alert.channel_id or 'Unknown'} | {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                }
            ]
        }

    def _fetch_video_metrics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch video metrics from YouTube API.

        This is a placeholder that should be implemented with actual
        YouTube Data API and Analytics API calls.
        """
        try:
            # Try to import YouTube API client
            from src.youtube.auth import get_authenticated_service

            youtube = get_authenticated_service()

            # Get video details
            video_response = youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            ).execute()

            if not video_response.get("items"):
                return None

            video_data = video_response["items"][0]
            snippet = video_data.get("snippet", {})
            stats = video_data.get("statistics", {})

            # Note: CTR and retention require YouTube Analytics API
            # This is a simplified version that uses available data
            return {
                "video_id": video_id,
                "title": snippet.get("title"),
                "channel_id": snippet.get("channelId"),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                # CTR and retention would come from Analytics API
                "ctr": None,  # Requires Analytics API
                "retention": None,  # Requires Analytics API
            }
        except ImportError:
            logger.warning("YouTube API client not available")
            return None
        except Exception as e:
            logger.error(f"Error fetching video metrics: {e}")
            return None

    def _get_recent_videos(self, hours: int) -> List[Dict[str, Any]]:
        """
        Get videos uploaded in the last N hours.

        This should be implemented to fetch from your database or YouTube API.
        """
        try:
            # Try to get from database first
            from src.database.models import get_recent_uploads
            return get_recent_uploads(hours)
        except ImportError:
            pass

        try:
            # Fallback to reading from output directory
            import glob
            from pathlib import Path

            output_dir = Path("output")
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_videos = []

            # Look for video metadata files
            for meta_file in output_dir.glob("**/metadata.json"):
                try:
                    with open(meta_file, "r") as f:
                        metadata = json.load(f)

                    upload_time_str = metadata.get("upload_time")
                    if upload_time_str:
                        upload_time = datetime.fromisoformat(upload_time_str.replace("Z", "+00:00"))
                        if upload_time > cutoff_time:
                            recent_videos.append({
                                "video_id": metadata.get("video_id"),
                                "upload_time": upload_time,
                                "metrics": metadata.get("metrics")
                            })
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

            return recent_videos
        except Exception as e:
            logger.error(f"Error getting recent videos: {e}")
            return []

    def _get_channel_average(self, channel_id: str) -> Optional[Dict[str, float]]:
        """
        Get average metrics for a channel.

        Caches results to avoid repeated API calls.
        """
        if channel_id in self._channel_averages:
            return self._channel_averages[channel_id]

        try:
            # Try to calculate from database or API
            from src.database.models import get_channel_stats
            stats = get_channel_stats(channel_id)
            if stats:
                self._channel_averages[channel_id] = stats
                return stats
        except ImportError:
            pass

        # Default averages if no data available
        default_averages = {
            "views": 1000,
            "ctr": 5.0,
            "retention": 40.0,
        }
        self._channel_averages[channel_id] = default_averages
        return default_averages

    def _save_alerts_log(self, alerts: Dict[str, List[PerformanceAlert]]) -> None:
        """Save alerts to JSON log file."""
        try:
            os.makedirs(os.path.dirname(self.alert_log_path), exist_ok=True)

            # Load existing alerts
            existing_alerts = []
            if os.path.exists(self.alert_log_path):
                try:
                    with open(self.alert_log_path, "r") as f:
                        existing_alerts = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

            # Add new alerts
            for video_id, video_alerts in alerts.items():
                for alert in video_alerts:
                    existing_alerts.append(alert.to_dict())

            # Keep only last 1000 alerts
            existing_alerts = existing_alerts[-1000:]

            with open(self.alert_log_path, "w") as f:
                json.dump(existing_alerts, f, indent=2)

            logger.info(f"Alerts saved to {self.alert_log_path}")
        except Exception as e:
            logger.error(f"Error saving alerts log: {e}")

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get a summary of alerts from the last N hours.

        Returns:
            Dictionary with alert counts and recent critical alerts
        """
        try:
            with open(self.alert_log_path, "r") as f:
                all_alerts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"total": 0, "critical": 0, "warning": 0, "info": 0, "recent_critical": []}

        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = []

        for alert_data in all_alerts:
            try:
                timestamp = datetime.fromisoformat(alert_data["timestamp"])
                if timestamp > cutoff:
                    recent_alerts.append(alert_data)
            except (KeyError, ValueError):
                continue

        summary = {
            "total": len(recent_alerts),
            "critical": sum(1 for a in recent_alerts if a.get("severity") == "critical"),
            "warning": sum(1 for a in recent_alerts if a.get("severity") == "warning"),
            "info": sum(1 for a in recent_alerts if a.get("severity") == "info"),
            "recent_critical": [a for a in recent_alerts if a.get("severity") == "critical"][-5:],
            "by_type": {}
        }

        for alert in recent_alerts:
            alert_type = alert.get("alert_type", "unknown")
            summary["by_type"][alert_type] = summary["by_type"].get(alert_type, 0) + 1

        return summary


def monitor_video(video_id: str, hours_since_upload: int = 24) -> List[PerformanceAlert]:
    """
    Convenience function to monitor a single video.

    Args:
        video_id: YouTube video ID
        hours_since_upload: Hours since upload (for context)

    Returns:
        List of alerts generated
    """
    monitor = PerformanceMonitor()
    return monitor.check_video(video_id, hours_since_upload)


def monitor_all(hours: int = 72) -> Dict[str, List[PerformanceAlert]]:
    """
    Convenience function to monitor all recent videos.

    Args:
        hours: Number of hours to look back

    Returns:
        Dictionary of video_id -> alerts
    """
    monitor = PerformanceMonitor()
    return monitor.monitor_all_recent(hours)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24

        print(f"\nMonitoring video: {video_id}")
        print("=" * 60)

        alerts = monitor_video(video_id, hours)

        if alerts:
            for alert in alerts:
                print(f"\n{alert}")
                print(f"  Recommendation: {alert.recommendation}")
        else:
            print("No alerts - video is performing well!")
    else:
        print("\nMonitoring all recent videos (last 72 hours)...")
        print("=" * 60)

        all_alerts = monitor_all(72)

        if all_alerts:
            for video_id, alerts in all_alerts.items():
                print(f"\nVideo: {video_id}")
                for alert in alerts:
                    print(f"  {alert}")
        else:
            print("No alerts - all videos are performing well!")
