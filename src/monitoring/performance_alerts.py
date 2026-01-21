"""
Real-Time Performance Alerts for YouTube Videos

Monitor video performance during critical first 24-48 hours and
generate actionable alerts when metrics fall below thresholds.

Features:
- Monitor CTR, retention, and engagement in real-time
- Alert if CTR drops below threshold
- Alert if retention is underperforming
- Suggest immediate actions (thumbnail swap, etc.)
- Email/Discord webhook notifications
- Scheduled monitoring checks

Usage:
    from src.monitoring.performance_alerts import PerformanceAlertSystem

    alert_system = PerformanceAlertSystem()

    # Start monitoring a newly uploaded video
    await alert_system.start_monitoring(
        video_id="abc123",
        channel_id="money_blueprints",
        duration_hours=48
    )

    # Check all monitored videos
    alerts = await alert_system.check_all_monitored()

    # Get action recommendations
    actions = alert_system.get_recommended_actions("abc123")
"""

import os
import json
import asyncio
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"


class AlertType(Enum):
    """Types of performance alerts."""
    LOW_CTR = "low_ctr"
    LOW_RETENTION = "low_retention"
    LOW_ENGAGEMENT = "low_engagement"
    UNDERPERFORMING = "underperforming"
    VIRAL_POTENTIAL = "viral_potential"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    RAPID_DROP = "rapid_drop"


class ActionType(Enum):
    """Types of recommended actions."""
    SWAP_THUMBNAIL = "swap_thumbnail"
    UPDATE_TITLE = "update_title"
    PROMOTE_SOCIAL = "promote_social"
    ENGAGE_COMMENTS = "engage_comments"
    CREATE_SHORT = "create_short"
    ADD_CARDS = "add_cards"
    UPDATE_DESCRIPTION = "update_description"
    PIN_COMMENT = "pin_comment"


@dataclass
class AlertThresholds:
    """Configurable alert thresholds."""
    ctr_warning: float = 4.0      # CTR below this triggers warning
    ctr_critical: float = 2.0     # CTR below this triggers critical
    retention_warning: float = 35.0  # Retention % below triggers warning
    retention_critical: float = 25.0  # Retention % below triggers critical
    engagement_warning: float = 3.0   # Engagement rate % below triggers warning
    views_underperform_ratio: float = 0.5  # Below 50% of channel avg
    views_viral_ratio: float = 3.0   # Above 3x channel avg = viral
    sentiment_negative_threshold: float = 0.3  # 30% negative comments

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Video performance metrics at a point in time."""
    video_id: str
    timestamp: datetime
    hours_since_upload: float
    views: int = 0
    ctr: float = 0.0
    avg_view_duration: float = 0.0
    retention_percentage: float = 0.0
    likes: int = 0
    dislikes: int = 0
    comments: int = 0
    shares: int = 0
    impressions: int = 0
    subscribers_gained: int = 0

    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.views == 0:
            return 0.0
        return ((self.likes + self.comments * 5 + self.shares * 10) / self.views) * 100

    @property
    def like_ratio(self) -> float:
        """Calculate like ratio."""
        total = self.likes + self.dislikes
        if total == 0:
            return 0.0
        return (self.likes / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["engagement_rate"] = self.engagement_rate
        result["like_ratio"] = self.like_ratio
        return result


@dataclass
class PerformanceAlert:
    """A performance alert for a video."""
    alert_id: str
    video_id: str
    video_title: str
    channel_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold_value: float
    recommended_actions: List[ActionType]
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["alert_type"] = self.alert_type.value
        result["severity"] = self.severity.value
        result["recommended_actions"] = [a.value for a in self.recommended_actions]
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class RecommendedAction:
    """A recommended action to improve performance."""
    action_type: ActionType
    priority: int  # 1 = highest
    title: str
    description: str
    estimated_impact: str
    implementation_steps: List[str]
    time_to_implement: str
    urgency: str  # immediate, soon, when_possible

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["action_type"] = self.action_type.value
        return result


@dataclass
class MonitoredVideo:
    """A video being monitored for performance."""
    video_id: str
    video_title: str
    channel_id: str
    upload_time: datetime
    monitoring_end_time: datetime
    check_interval_minutes: int = 60
    last_check: Optional[datetime] = None
    metrics_history: List[PerformanceMetrics] = field(default_factory=list)
    alerts: List[PerformanceAlert] = field(default_factory=list)
    is_active: bool = True

    @property
    def hours_since_upload(self) -> float:
        return (datetime.now() - self.upload_time).total_seconds() / 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "channel_id": self.channel_id,
            "upload_time": self.upload_time.isoformat(),
            "monitoring_end_time": self.monitoring_end_time.isoformat(),
            "check_interval_minutes": self.check_interval_minutes,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "hours_since_upload": self.hours_since_upload,
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "alerts": [a.to_dict() for a in self.alerts],
            "is_active": self.is_active,
        }


# Action recommendations by alert type
ACTION_RECOMMENDATIONS: Dict[AlertType, List[RecommendedAction]] = {
    AlertType.LOW_CTR: [
        RecommendedAction(
            action_type=ActionType.SWAP_THUMBNAIL,
            priority=1,
            title="Swap Thumbnail",
            description="Replace thumbnail with a high-contrast variant featuring a face with strong emotion",
            estimated_impact="Can increase CTR by 20-50%",
            implementation_steps=[
                "Generate 2-3 new thumbnail variants",
                "Ensure face is visible and expressive",
                "Use bright, contrasting colors",
                "Add curiosity-inducing text overlay",
                "Upload new thumbnail via YouTube Studio"
            ],
            time_to_implement="15-30 minutes",
            urgency="immediate"
        ),
        RecommendedAction(
            action_type=ActionType.UPDATE_TITLE,
            priority=2,
            title="Optimize Title",
            description="Rewrite title with power words and curiosity gap",
            estimated_impact="Can increase CTR by 10-30%",
            implementation_steps=[
                "Add numbers or statistics",
                "Use emotional trigger words",
                "Create curiosity gap (don't give away conclusion)",
                "Keep under 60 characters",
                "Front-load important keywords"
            ],
            time_to_implement="5-10 minutes",
            urgency="immediate"
        ),
    ],
    AlertType.LOW_RETENTION: [
        RecommendedAction(
            action_type=ActionType.ADD_CARDS,
            priority=1,
            title="Add Cards at Drop Points",
            description="Add info cards at retention drop points to re-engage viewers",
            estimated_impact="Can retain 5-10% more viewers",
            implementation_steps=[
                "Analyze retention graph in YouTube Studio",
                "Identify major drop-off points",
                "Add card polls or video suggestions at these points",
                "Link to more engaging content"
            ],
            time_to_implement="10-15 minutes",
            urgency="soon"
        ),
        RecommendedAction(
            action_type=ActionType.CREATE_SHORT,
            priority=2,
            title="Create Short from Best Moment",
            description="Extract the highest-retention segment as a Short to drive traffic",
            estimated_impact="Can generate 10-50% additional views",
            implementation_steps=[
                "Find highest retention segment in analytics",
                "Extract 15-60 second clip",
                "Add captions and optimize for vertical",
                "Post as YouTube Short with link to full video"
            ],
            time_to_implement="20-30 minutes",
            urgency="soon"
        ),
        RecommendedAction(
            action_type=ActionType.PIN_COMMENT,
            priority=3,
            title="Pin Engaging Comment",
            description="Pin a comment that encourages viewers to watch till the end",
            estimated_impact="Can improve retention 2-5%",
            implementation_steps=[
                "Write or find engaging comment",
                "Tease content coming later in video",
                "Pin comment to top",
                "Reply to boost engagement"
            ],
            time_to_implement="5 minutes",
            urgency="immediate"
        ),
    ],
    AlertType.LOW_ENGAGEMENT: [
        RecommendedAction(
            action_type=ActionType.ENGAGE_COMMENTS,
            priority=1,
            title="Engage with Comments",
            description="Reply to all comments to boost algorithm signals and encourage more comments",
            estimated_impact="Can increase engagement by 50-100%",
            implementation_steps=[
                "Reply to every comment with thoughtful responses",
                "Ask follow-up questions to drive conversation",
                "Heart/like positive comments",
                "Address any concerns or questions"
            ],
            time_to_implement="15-30 minutes",
            urgency="immediate"
        ),
        RecommendedAction(
            action_type=ActionType.PROMOTE_SOCIAL,
            priority=2,
            title="Promote on Social Media",
            description="Cross-promote on all social platforms to drive engagement",
            estimated_impact="Can drive 10-30% additional views",
            implementation_steps=[
                "Post on Twitter with video preview",
                "Share in relevant Reddit communities",
                "Post to Discord servers",
                "Send to email list",
                "Create Instagram Story/Reel"
            ],
            time_to_implement="20-30 minutes",
            urgency="immediate"
        ),
    ],
    AlertType.UNDERPERFORMING: [
        RecommendedAction(
            action_type=ActionType.SWAP_THUMBNAIL,
            priority=1,
            title="Complete Package Refresh",
            description="Update thumbnail, title, and description together",
            estimated_impact="Can revive underperforming video",
            implementation_steps=[
                "Generate new thumbnail with different approach",
                "Rewrite title with different angle",
                "Update description with better hooks",
                "Add timestamps/chapters if missing"
            ],
            time_to_implement="30-45 minutes",
            urgency="immediate"
        ),
        RecommendedAction(
            action_type=ActionType.CREATE_SHORT,
            priority=2,
            title="Create Multiple Shorts",
            description="Extract 3-5 Shorts to funnel traffic to main video",
            estimated_impact="Can double or triple video views",
            implementation_steps=[
                "Extract most interesting segments",
                "Optimize each for vertical format",
                "Add compelling hooks at start",
                "Post over several days with links to full video"
            ],
            time_to_implement="1-2 hours",
            urgency="soon"
        ),
    ],
    AlertType.VIRAL_POTENTIAL: [
        RecommendedAction(
            action_type=ActionType.PROMOTE_SOCIAL,
            priority=1,
            title="Maximize Momentum",
            description="Promote heavily while video has algorithmic momentum",
            estimated_impact="Can 10x viral reach",
            implementation_steps=[
                "Share across ALL social platforms",
                "Engage with every comment immediately",
                "Create and post follow-up content",
                "Cross-promote from other videos via end screens",
                "Consider paid promotion to accelerate"
            ],
            time_to_implement="1-2 hours",
            urgency="immediate"
        ),
        RecommendedAction(
            action_type=ActionType.PIN_COMMENT,
            priority=2,
            title="Pin Strategic Comment",
            description="Pin a comment linking to related videos to capture traffic",
            estimated_impact="Can drive 5-15% of views to other videos",
            implementation_steps=[
                "Write comment linking to related content",
                "Include call-to-action for subscription",
                "Pin and heart the comment"
            ],
            time_to_implement="5 minutes",
            urgency="immediate"
        ),
    ],
}


class NotificationService:
    """Handle sending notifications via various channels."""

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None
    ):
        self.discord_webhook_url = discord_webhook_url or os.getenv("DISCORD_ALERT_WEBHOOK")
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_ALERT_WEBHOOK")
        self.email_config = email_config or {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "sender_email": os.getenv("ALERT_EMAIL_SENDER"),
            "sender_password": os.getenv("ALERT_EMAIL_PASSWORD"),
            "recipient_email": os.getenv("ALERT_EMAIL_RECIPIENT"),
        }

    async def send_discord(self, alert: PerformanceAlert) -> bool:
        """Send alert to Discord webhook."""
        if not self.discord_webhook_url or not REQUESTS_AVAILABLE:
            return False

        severity_colors = {
            AlertSeverity.INFO: 0x3498db,
            AlertSeverity.WARNING: 0xf39c12,
            AlertSeverity.CRITICAL: 0xe74c3c,
            AlertSeverity.URGENT: 0x9b59b6,
        }

        severity_emojis = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
            AlertSeverity.URGENT: ":fire:",
        }

        payload = {
            "embeds": [{
                "title": f"{severity_emojis[alert.severity]} Performance Alert: {alert.alert_type.value.replace('_', ' ').upper()}",
                "description": alert.message,
                "color": severity_colors[alert.severity],
                "fields": [
                    {
                        "name": "Video",
                        "value": f"[{alert.video_title}](https://youtube.com/watch?v={alert.video_id})",
                        "inline": True
                    },
                    {
                        "name": f"{alert.metric_name}",
                        "value": f"{alert.metric_value:.2f}% (threshold: {alert.threshold_value:.2f}%)",
                        "inline": True
                    },
                    {
                        "name": "Recommended Actions",
                        "value": "\n".join([f"- {a.value.replace('_', ' ').title()}" for a in alert.recommended_actions[:3]]),
                        "inline": False
                    }
                ],
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": f"Channel: {alert.channel_id}"}
            }]
        }

        try:
            response = requests.post(self.discord_webhook_url, json=payload, timeout=10)
            return response.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False

    async def send_slack(self, alert: PerformanceAlert) -> bool:
        """Send alert to Slack webhook."""
        if not self.slack_webhook_url or not REQUESTS_AVAILABLE:
            return False

        severity_emojis = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
            AlertSeverity.URGENT: ":fire:",
        }

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{severity_emojis[alert.severity]} Performance Alert: {alert.alert_type.value.replace('_', ' ').upper()}"
                    }
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": alert.message}
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Video:*\n<https://youtube.com/watch?v={alert.video_id}|{alert.video_title}>"},
                        {"type": "mrkdwn", "text": f"*{alert.metric_name}:* {alert.metric_value:.2f}%"}
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Recommended Actions:*\n" + "\n".join([f"- {a.value.replace('_', ' ').title()}" for a in alert.recommended_actions[:3]])
                    }
                }
            ]
        }

        try:
            response = requests.post(self.slack_webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False

    async def send_email(self, alert: PerformanceAlert) -> bool:
        """Send alert via email."""
        if not all([
            self.email_config.get("sender_email"),
            self.email_config.get("sender_password"),
            self.email_config.get("recipient_email")
        ]):
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] YouTube Alert: {alert.alert_type.value.replace('_', ' ')}"
            msg["From"] = self.email_config["sender_email"]
            msg["To"] = self.email_config["recipient_email"]

            text_content = f"""
YouTube Performance Alert

Severity: {alert.severity.value.upper()}
Type: {alert.alert_type.value.replace('_', ' ').title()}

Video: {alert.video_title}
Link: https://youtube.com/watch?v={alert.video_id}

{alert.message}

{alert.metric_name}: {alert.metric_value:.2f}% (threshold: {alert.threshold_value:.2f}%)

Recommended Actions:
{chr(10).join(['- ' + a.value.replace('_', ' ').title() for a in alert.recommended_actions])}

Channel: {alert.channel_id}
Time: {alert.timestamp.isoformat()}
            """

            html_content = f"""
<html>
<body>
<h2 style="color: {'#e74c3c' if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.URGENT] else '#f39c12'}">
YouTube Performance Alert
</h2>
<p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
<p><strong>Type:</strong> {alert.alert_type.value.replace('_', ' ').title()}</p>
<hr>
<p><strong>Video:</strong> <a href="https://youtube.com/watch?v={alert.video_id}">{alert.video_title}</a></p>
<p>{alert.message}</p>
<p><strong>{alert.metric_name}:</strong> {alert.metric_value:.2f}% (threshold: {alert.threshold_value:.2f}%)</p>
<h3>Recommended Actions:</h3>
<ul>
{''.join([f'<li>{a.value.replace("_", " ").title()}</li>' for a in alert.recommended_actions])}
</ul>
<hr>
<small>Channel: {alert.channel_id} | Time: {alert.timestamp.isoformat()}</small>
</body>
</html>
            """

            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"]) as server:
                server.starttls()
                server.login(self.email_config["sender_email"], self.email_config["sender_password"])
                server.send_message(msg)

            return True
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False

    async def send_all(self, alert: PerformanceAlert) -> Dict[str, bool]:
        """Send alert via all configured channels."""
        results = {}

        if self.discord_webhook_url:
            results["discord"] = await self.send_discord(alert)

        if self.slack_webhook_url:
            results["slack"] = await self.send_slack(alert)

        if self.email_config.get("sender_email"):
            results["email"] = await self.send_email(alert)

        return results


class PerformanceAlertSystem:
    """
    Main class for real-time performance monitoring and alerting.

    Monitors newly uploaded videos during critical first 24-48 hours
    and generates actionable alerts when metrics fall below thresholds.
    """

    def __init__(
        self,
        db_path: str = "data/performance_alerts.db",
        thresholds: Optional[AlertThresholds] = None
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.thresholds = thresholds or AlertThresholds()
        self.notification_service = NotificationService()

        # In-memory storage for monitored videos
        self.monitored_videos: Dict[str, MonitoredVideo] = {}

        # Channel baselines for comparison
        self._channel_baselines: Dict[str, Dict[str, float]] = {}

        self._init_database()
        self._load_monitored_videos()

        logger.info("PerformanceAlertSystem initialized")

    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitored_videos (
                    video_id TEXT PRIMARY KEY,
                    video_title TEXT,
                    channel_id TEXT,
                    upload_time TEXT,
                    monitoring_end_time TEXT,
                    check_interval_minutes INTEGER,
                    last_check TEXT,
                    is_active INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    timestamp TEXT,
                    hours_since_upload REAL,
                    views INTEGER,
                    ctr REAL,
                    avg_view_duration REAL,
                    retention_percentage REAL,
                    likes INTEGER,
                    comments INTEGER,
                    shares INTEGER,
                    impressions INTEGER,
                    FOREIGN KEY (video_id) REFERENCES monitored_videos(video_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    video_id TEXT,
                    video_title TEXT,
                    channel_id TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    threshold_value REAL,
                    timestamp TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channel_baselines (
                    channel_id TEXT PRIMARY KEY,
                    avg_views_24h REAL,
                    avg_ctr REAL,
                    avg_retention REAL,
                    avg_engagement REAL,
                    updated_at TEXT
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_video ON performance_metrics(video_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_video ON alerts(video_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved, severity)")

    def _load_monitored_videos(self):
        """Load active monitored videos from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM monitored_videos WHERE is_active = 1"
            ).fetchall()

            for row in rows:
                video = MonitoredVideo(
                    video_id=row["video_id"],
                    video_title=row["video_title"],
                    channel_id=row["channel_id"],
                    upload_time=datetime.fromisoformat(row["upload_time"]),
                    monitoring_end_time=datetime.fromisoformat(row["monitoring_end_time"]),
                    check_interval_minutes=row["check_interval_minutes"],
                    last_check=datetime.fromisoformat(row["last_check"]) if row["last_check"] else None,
                    is_active=bool(row["is_active"])
                )
                self.monitored_videos[video.video_id] = video

        logger.info(f"Loaded {len(self.monitored_videos)} monitored videos")

    async def start_monitoring(
        self,
        video_id: str,
        video_title: str,
        channel_id: str,
        duration_hours: int = 48,
        check_interval_minutes: int = 60
    ) -> MonitoredVideo:
        """
        Start monitoring a newly uploaded video.

        Args:
            video_id: YouTube video ID
            video_title: Video title
            channel_id: Channel ID
            duration_hours: How long to monitor (default: 48 hours)
            check_interval_minutes: How often to check (default: 60 minutes)

        Returns:
            MonitoredVideo object
        """
        logger.info(f"Starting monitoring for video {video_id}: {video_title}")

        now = datetime.now()
        video = MonitoredVideo(
            video_id=video_id,
            video_title=video_title,
            channel_id=channel_id,
            upload_time=now,
            monitoring_end_time=now + timedelta(hours=duration_hours),
            check_interval_minutes=check_interval_minutes,
            is_active=True
        )

        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO monitored_videos
                (video_id, video_title, channel_id, upload_time, monitoring_end_time,
                 check_interval_minutes, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (
                video.video_id, video.video_title, video.channel_id,
                video.upload_time.isoformat(), video.monitoring_end_time.isoformat(),
                video.check_interval_minutes
            ))

        self.monitored_videos[video_id] = video

        # Send initial notification
        await self._send_monitoring_started_notification(video)

        return video

    async def _send_monitoring_started_notification(self, video: MonitoredVideo):
        """Send notification that monitoring has started."""
        if self.notification_service.discord_webhook_url and REQUESTS_AVAILABLE:
            payload = {
                "embeds": [{
                    "title": ":rocket: Monitoring Started",
                    "description": f"Now monitoring **{video.video_title}**",
                    "color": 0x2ecc71,
                    "fields": [
                        {"name": "Video ID", "value": video.video_id, "inline": True},
                        {"name": "Duration", "value": f"{(video.monitoring_end_time - video.upload_time).total_seconds() / 3600:.0f} hours", "inline": True},
                        {"name": "Check Interval", "value": f"{video.check_interval_minutes} minutes", "inline": True}
                    ],
                    "timestamp": datetime.now().isoformat()
                }]
            }
            try:
                requests.post(self.notification_service.discord_webhook_url, json=payload, timeout=10)
            except Exception:
                pass

    async def check_video(
        self,
        video_id: str,
        metrics: Optional[PerformanceMetrics] = None
    ) -> List[PerformanceAlert]:
        """
        Check a video's performance and generate alerts if needed.

        Args:
            video_id: Video ID to check
            metrics: Optional pre-fetched metrics

        Returns:
            List of generated alerts
        """
        if video_id not in self.monitored_videos:
            logger.warning(f"Video {video_id} is not being monitored")
            return []

        video = self.monitored_videos[video_id]

        # Fetch metrics if not provided
        if metrics is None:
            metrics = await self._fetch_video_metrics(video_id, video.hours_since_upload)

        if metrics is None:
            logger.warning(f"Could not fetch metrics for {video_id}")
            return []

        # Update video state
        video.metrics_history.append(metrics)
        video.last_check = datetime.now()

        # Save metrics to database
        self._save_metrics(metrics)

        # Generate alerts
        alerts = await self._generate_alerts(video, metrics)

        # Save and send alerts
        for alert in alerts:
            self._save_alert(alert)
            video.alerts.append(alert)
            await self.notification_service.send_all(alert)

        # Check if monitoring should end
        if datetime.now() >= video.monitoring_end_time:
            await self.stop_monitoring(video_id)

        return alerts

    async def _generate_alerts(
        self,
        video: MonitoredVideo,
        metrics: PerformanceMetrics
    ) -> List[PerformanceAlert]:
        """Generate alerts based on current metrics."""
        alerts = []

        # Get channel baseline for comparison
        baseline = await self._get_channel_baseline(video.channel_id)

        # Check CTR
        if metrics.ctr > 0:
            if metrics.ctr < self.thresholds.ctr_critical:
                alerts.append(self._create_alert(
                    video, AlertType.LOW_CTR, AlertSeverity.CRITICAL,
                    f"CTR is critically low at {metrics.ctr:.2f}%",
                    "CTR", metrics.ctr, self.thresholds.ctr_critical,
                    [ActionType.SWAP_THUMBNAIL, ActionType.UPDATE_TITLE]
                ))
            elif metrics.ctr < self.thresholds.ctr_warning:
                alerts.append(self._create_alert(
                    video, AlertType.LOW_CTR, AlertSeverity.WARNING,
                    f"CTR is below target at {metrics.ctr:.2f}%",
                    "CTR", metrics.ctr, self.thresholds.ctr_warning,
                    [ActionType.SWAP_THUMBNAIL, ActionType.UPDATE_TITLE]
                ))

        # Check retention
        if metrics.retention_percentage > 0:
            if metrics.retention_percentage < self.thresholds.retention_critical:
                alerts.append(self._create_alert(
                    video, AlertType.LOW_RETENTION, AlertSeverity.CRITICAL,
                    f"Retention is critically low at {metrics.retention_percentage:.1f}%",
                    "Retention", metrics.retention_percentage, self.thresholds.retention_critical,
                    [ActionType.ADD_CARDS, ActionType.CREATE_SHORT, ActionType.PIN_COMMENT]
                ))
            elif metrics.retention_percentage < self.thresholds.retention_warning:
                alerts.append(self._create_alert(
                    video, AlertType.LOW_RETENTION, AlertSeverity.WARNING,
                    f"Retention is below target at {metrics.retention_percentage:.1f}%",
                    "Retention", metrics.retention_percentage, self.thresholds.retention_warning,
                    [ActionType.ADD_CARDS, ActionType.CREATE_SHORT]
                ))

        # Check engagement
        engagement = metrics.engagement_rate
        if engagement > 0 and engagement < self.thresholds.engagement_warning:
            alerts.append(self._create_alert(
                video, AlertType.LOW_ENGAGEMENT, AlertSeverity.WARNING,
                f"Engagement rate is low at {engagement:.2f}%",
                "Engagement", engagement, self.thresholds.engagement_warning,
                [ActionType.ENGAGE_COMMENTS, ActionType.PROMOTE_SOCIAL]
            ))

        # Check against channel baseline
        if baseline and baseline.get("avg_views_24h", 0) > 0:
            avg_views = baseline["avg_views_24h"]

            if metrics.views < avg_views * self.thresholds.views_underperform_ratio:
                alerts.append(self._create_alert(
                    video, AlertType.UNDERPERFORMING, AlertSeverity.WARNING,
                    f"Views ({metrics.views:,}) are {metrics.views/avg_views*100:.0f}% of channel average ({avg_views:,.0f})",
                    "Views", (metrics.views/avg_views)*100, self.thresholds.views_underperform_ratio * 100,
                    [ActionType.SWAP_THUMBNAIL, ActionType.CREATE_SHORT, ActionType.PROMOTE_SOCIAL]
                ))

            elif metrics.views > avg_views * self.thresholds.views_viral_ratio:
                alerts.append(self._create_alert(
                    video, AlertType.VIRAL_POTENTIAL, AlertSeverity.INFO,
                    f"Video is going viral! Views ({metrics.views:,}) are {metrics.views/avg_views:.1f}x channel average",
                    "Views", (metrics.views/avg_views)*100, self.thresholds.views_viral_ratio * 100,
                    [ActionType.PROMOTE_SOCIAL, ActionType.PIN_COMMENT]
                ))

        # Check for rapid drop (comparing to previous metrics)
        if len(video.metrics_history) >= 2:
            prev_metrics = video.metrics_history[-2]
            if prev_metrics.ctr > 0 and metrics.ctr > 0:
                ctr_drop = prev_metrics.ctr - metrics.ctr
                if ctr_drop > 1.0:  # More than 1% CTR drop
                    alerts.append(self._create_alert(
                        video, AlertType.RAPID_DROP, AlertSeverity.WARNING,
                        f"CTR dropped {ctr_drop:.2f}% since last check",
                        "CTR Drop", ctr_drop, 1.0,
                        [ActionType.SWAP_THUMBNAIL, ActionType.UPDATE_TITLE]
                    ))

        return alerts

    def _create_alert(
        self,
        video: MonitoredVideo,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold_value: float,
        recommended_actions: List[ActionType]
    ) -> PerformanceAlert:
        """Create a performance alert."""
        import uuid

        return PerformanceAlert(
            alert_id=str(uuid.uuid4())[:8],
            video_id=video.video_id,
            video_title=video.video_title,
            channel_id=video.channel_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold_value=threshold_value,
            recommended_actions=recommended_actions
        )

    def _save_metrics(self, metrics: PerformanceMetrics):
        """Save metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics
                (video_id, timestamp, hours_since_upload, views, ctr,
                 avg_view_duration, retention_percentage, likes, comments,
                 shares, impressions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.video_id, metrics.timestamp.isoformat(),
                metrics.hours_since_upload, metrics.views, metrics.ctr,
                metrics.avg_view_duration, metrics.retention_percentage,
                metrics.likes, metrics.comments, metrics.shares, metrics.impressions
            ))

    def _save_alert(self, alert: PerformanceAlert):
        """Save alert to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts
                (alert_id, video_id, video_title, channel_id, alert_type,
                 severity, message, metric_name, metric_value, threshold_value,
                 timestamp, acknowledged, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """, (
                alert.alert_id, alert.video_id, alert.video_title,
                alert.channel_id, alert.alert_type.value, alert.severity.value,
                alert.message, alert.metric_name, alert.metric_value,
                alert.threshold_value, alert.timestamp.isoformat()
            ))

    async def _fetch_video_metrics(
        self,
        video_id: str,
        hours_since_upload: float
    ) -> Optional[PerformanceMetrics]:
        """Fetch video metrics from YouTube API."""
        try:
            from src.youtube.analytics_api import YouTubeAnalyticsAPI

            api = YouTubeAnalyticsAPI()
            analytics = api.get_video_analytics(video_id, days=7)

            return PerformanceMetrics(
                video_id=video_id,
                timestamp=datetime.now(),
                hours_since_upload=hours_since_upload,
                views=analytics.views,
                ctr=analytics.ctr,
                avg_view_duration=analytics.avg_view_duration,
                retention_percentage=analytics.avg_percentage_viewed,
                likes=analytics.likes,
                comments=analytics.comments,
                shares=analytics.shares,
                impressions=analytics.impressions,
                subscribers_gained=analytics.subscribers_gained
            )
        except ImportError:
            logger.warning("YouTube Analytics API not available")
            return None
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return None

    async def _get_channel_baseline(self, channel_id: str) -> Optional[Dict[str, float]]:
        """Get channel baseline metrics."""
        if channel_id in self._channel_baselines:
            return self._channel_baselines[channel_id]

        # Try to load from database
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM channel_baselines WHERE channel_id = ?",
                (channel_id,)
            ).fetchone()

            if row:
                baseline = {
                    "avg_views_24h": row["avg_views_24h"],
                    "avg_ctr": row["avg_ctr"],
                    "avg_retention": row["avg_retention"],
                    "avg_engagement": row["avg_engagement"],
                }
                self._channel_baselines[channel_id] = baseline
                return baseline

        # Return default baseline if none exists
        return {
            "avg_views_24h": 1000,
            "avg_ctr": 5.0,
            "avg_retention": 40.0,
            "avg_engagement": 3.0,
        }

    async def update_channel_baseline(
        self,
        channel_id: str,
        avg_views_24h: float,
        avg_ctr: float,
        avg_retention: float,
        avg_engagement: float
    ):
        """Update channel baseline metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO channel_baselines
                (channel_id, avg_views_24h, avg_ctr, avg_retention, avg_engagement, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (channel_id, avg_views_24h, avg_ctr, avg_retention, avg_engagement, datetime.now().isoformat()))

        self._channel_baselines[channel_id] = {
            "avg_views_24h": avg_views_24h,
            "avg_ctr": avg_ctr,
            "avg_retention": avg_retention,
            "avg_engagement": avg_engagement,
        }

    async def check_all_monitored(self) -> List[PerformanceAlert]:
        """Check all monitored videos and return any alerts."""
        all_alerts = []

        for video_id, video in list(self.monitored_videos.items()):
            if not video.is_active:
                continue

            # Check if it's time to check
            if video.last_check:
                minutes_since_check = (datetime.now() - video.last_check).total_seconds() / 60
                if minutes_since_check < video.check_interval_minutes:
                    continue

            alerts = await self.check_video(video_id)
            all_alerts.extend(alerts)

        return all_alerts

    async def stop_monitoring(self, video_id: str):
        """Stop monitoring a video."""
        if video_id in self.monitored_videos:
            self.monitored_videos[video_id].is_active = False

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE monitored_videos SET is_active = 0 WHERE video_id = ?",
                    (video_id,)
                )

            logger.info(f"Stopped monitoring video {video_id}")

    def get_recommended_actions(self, video_id: str) -> List[RecommendedAction]:
        """Get recommended actions for a video based on its alerts."""
        if video_id not in self.monitored_videos:
            return []

        video = self.monitored_videos[video_id]
        actions = []
        seen_action_types = set()

        for alert in video.alerts:
            if alert.resolved:
                continue

            for action in ACTION_RECOMMENDATIONS.get(alert.alert_type, []):
                if action.action_type not in seen_action_types:
                    actions.append(action)
                    seen_action_types.add(action.action_type)

        # Sort by priority
        actions.sort(key=lambda a: a.priority)
        return actions

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts in the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Count by severity
            severity_counts = {}
            for row in conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM alerts
                WHERE timestamp > ?
                GROUP BY severity
            """, (cutoff.isoformat(),)).fetchall():
                severity_counts[row["severity"]] = row["count"]

            # Count by type
            type_counts = {}
            for row in conn.execute("""
                SELECT alert_type, COUNT(*) as count
                FROM alerts
                WHERE timestamp > ?
                GROUP BY alert_type
            """, (cutoff.isoformat(),)).fetchall():
                type_counts[row["alert_type"]] = row["count"]

            # Recent alerts
            recent = conn.execute("""
                SELECT * FROM alerts
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (cutoff.isoformat(),)).fetchall()

        return {
            "period_hours": hours,
            "total_alerts": sum(severity_counts.values()),
            "by_severity": severity_counts,
            "by_type": type_counts,
            "recent_alerts": [dict(row) for row in recent],
            "monitored_videos": len([v for v in self.monitored_videos.values() if v.is_active]),
        }

    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?",
                (alert_id,)
            )

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE alerts SET resolved = 1 WHERE alert_id = ?",
                (alert_id,)
            )


# Convenience functions
async def start_monitoring(video_id: str, video_title: str, channel_id: str) -> MonitoredVideo:
    """Quick function to start monitoring a video."""
    system = PerformanceAlertSystem()
    return await system.start_monitoring(video_id, video_title, channel_id)


async def check_all_videos() -> List[PerformanceAlert]:
    """Quick function to check all monitored videos."""
    system = PerformanceAlertSystem()
    return await system.check_all_monitored()


if __name__ == "__main__":
    import sys

    async def main():
        print("\n" + "=" * 60)
        print("PERFORMANCE ALERT SYSTEM")
        print("=" * 60 + "\n")

        system = PerformanceAlertSystem()

        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "start" and len(sys.argv) >= 4:
                video_id = sys.argv[2]
                video_title = sys.argv[3]
                channel_id = sys.argv[4] if len(sys.argv) > 4 else "default_channel"

                video = await system.start_monitoring(video_id, video_title, channel_id)
                print(f"Started monitoring: {video.video_title}")
                print(f"  Video ID: {video.video_id}")
                print(f"  Will monitor until: {video.monitoring_end_time}")

            elif command == "check":
                alerts = await system.check_all_monitored()
                print(f"Checked {len(system.monitored_videos)} videos")
                print(f"Generated {len(alerts)} alerts")
                for alert in alerts:
                    print(f"\n[{alert.severity.value.upper()}] {alert.alert_type.value}")
                    print(f"  Video: {alert.video_title}")
                    print(f"  {alert.message}")

            elif command == "summary":
                summary = system.get_alert_summary(24)
                print(f"Alert Summary (last 24 hours):")
                print(f"  Total Alerts: {summary['total_alerts']}")
                print(f"  By Severity: {summary['by_severity']}")
                print(f"  Monitored Videos: {summary['monitored_videos']}")

            elif command == "actions" and len(sys.argv) >= 3:
                video_id = sys.argv[2]
                actions = system.get_recommended_actions(video_id)
                print(f"Recommended Actions for {video_id}:")
                for i, action in enumerate(actions, 1):
                    print(f"\n{i}. {action.title} (Priority: {action.priority})")
                    print(f"   {action.description}")
                    print(f"   Impact: {action.estimated_impact}")
                    print(f"   Time: {action.time_to_implement}")
            else:
                print("Unknown command or missing arguments")
        else:
            print("Usage:")
            print("  python -m src.monitoring.performance_alerts start <video_id> <title> [channel_id]")
            print("  python -m src.monitoring.performance_alerts check")
            print("  python -m src.monitoring.performance_alerts summary")
            print("  python -m src.monitoring.performance_alerts actions <video_id>")

    asyncio.run(main())
