"""
Success Metrics Tracker for YouTube Automation
Tracks KPIs, goals, and progress toward channel success.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from loguru import logger


@dataclass
class ChannelMetrics:
    """Metrics for a single channel."""
    channel_id: str
    subscribers: int = 0
    total_views: int = 0
    watch_time_hours: float = 0.0
    avg_view_duration: float = 0.0
    avg_ctr: float = 0.0
    avg_retention: float = 0.0
    revenue_estimate: float = 0.0
    videos_published: int = 0
    shorts_published: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score (0-100)."""
        ctr_score = min(self.avg_ctr * 10, 30)  # Max 30 points
        retention_score = min(self.avg_retention * 0.5, 40)  # Max 40 points
        growth_score = min(self.subscribers / 100, 30)  # Max 30 points
        return ctr_score + retention_score + growth_score


@dataclass
class DailySnapshot:
    """Daily performance snapshot."""
    date: str
    channels: Dict[str, ChannelMetrics]
    total_api_cost: float = 0.0
    videos_created: int = 0
    shorts_created: int = 0
    errors_count: int = 0


@dataclass
class GoalProgress:
    """Track progress toward a goal."""
    goal_name: str
    target_value: float
    current_value: float
    unit: str
    deadline: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        if self.target_value == 0:
            return 100.0
        return min((self.current_value / self.target_value) * 100, 100)

    @property
    def on_track(self) -> bool:
        if not self.deadline:
            return True
        days_remaining = (self.deadline - datetime.now()).days
        if days_remaining <= 0:
            return self.progress_percent >= 100
        # Linear progress check
        total_days = 30  # Assume 30 day goals
        expected_progress = ((total_days - days_remaining) / total_days) * 100
        return self.progress_percent >= expected_progress * 0.8  # 80% buffer


class SuccessTracker:
    """
    Comprehensive success tracking system.

    Features:
    - Daily metrics snapshots
    - Goal tracking with progress
    - Trend analysis
    - Performance alerts
    - Token efficiency tracking
    """

    def __init__(self, db_path: str = "data/success_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # Default goals
        self.goals: Dict[str, GoalProgress] = {
            "token_efficiency": GoalProgress(
                goal_name="Reduce token usage by 50%",
                target_value=50.0,
                current_value=0.0,
                unit="%",
                deadline=datetime.now() + timedelta(days=30)
            ),
            "content_quality": GoalProgress(
                goal_name="Improve engagement by 50%",
                target_value=50.0,
                current_value=0.0,
                unit="%",
                deadline=datetime.now() + timedelta(days=30)
            ),
            "subscribers_1k": GoalProgress(
                goal_name="Reach 1,000 subscribers (combined)",
                target_value=1000,
                current_value=0,
                unit="subs",
                deadline=datetime.now() + timedelta(days=30)
            ),
            "monthly_views_50k": GoalProgress(
                goal_name="Reach 50,000 monthly views",
                target_value=50000,
                current_value=0,
                unit="views",
                deadline=datetime.now() + timedelta(days=30)
            ),
        }

        logger.info("SuccessTracker initialized")

    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                subscribers INTEGER DEFAULT 0,
                total_views INTEGER DEFAULT 0,
                watch_time_hours REAL DEFAULT 0,
                avg_ctr REAL DEFAULT 0,
                avg_retention REAL DEFAULT 0,
                revenue_estimate REAL DEFAULT 0,
                videos_published INTEGER DEFAULT 0,
                shorts_published INTEGER DEFAULT 0,
                api_cost REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, channel_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_name TEXT NOT NULL,
                target_value REAL NOT NULL,
                current_value REAL DEFAULT 0,
                unit TEXT,
                deadline TEXT,
                achieved INTEGER DEFAULT 0,
                achieved_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                provider TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                operation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_metrics(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_channel ON daily_metrics(channel_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_date ON token_usage(date)")

        conn.commit()
        conn.close()

    def record_daily_metrics(self, metrics: ChannelMetrics):
        """Record daily metrics for a channel."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        cursor.execute("""
            INSERT OR REPLACE INTO daily_metrics
            (date, channel_id, subscribers, total_views, watch_time_hours,
             avg_ctr, avg_retention, revenue_estimate, videos_published, shorts_published)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            today, metrics.channel_id, metrics.subscribers, metrics.total_views,
            metrics.watch_time_hours, metrics.avg_ctr, metrics.avg_retention,
            metrics.revenue_estimate, metrics.videos_published, metrics.shorts_published
        ))

        conn.commit()
        conn.close()
        logger.info(f"Recorded metrics for {metrics.channel_id}")

    def record_token_usage(self, provider: str, tokens: int, cost: float, operation: str):
        """Record token usage for efficiency tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        cursor.execute("""
            INSERT INTO token_usage (date, provider, tokens_used, cost, operation)
            VALUES (?, ?, ?, ?, ?)
        """, (today, provider, tokens, cost, operation))

        conn.commit()
        conn.close()

    def get_token_efficiency(self, days: int = 7) -> Dict[str, Any]:
        """Calculate token efficiency metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        cursor.execute("""
            SELECT
                SUM(tokens_used) as total_tokens,
                SUM(cost) as total_cost,
                COUNT(DISTINCT operation) as operations,
                AVG(tokens_used) as avg_tokens_per_op
            FROM token_usage
            WHERE date >= ?
        """, (start_date,))

        row = cursor.fetchone()
        conn.close()

        return {
            "total_tokens": row[0] or 0,
            "total_cost": row[1] or 0.0,
            "operations": row[2] or 0,
            "avg_tokens_per_op": row[3] or 0,
            "period_days": days
        }

    def get_channel_trend(self, channel_id: str, days: int = 30) -> List[Dict]:
        """Get trend data for a channel."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        cursor.execute("""
            SELECT date, subscribers, total_views, avg_ctr, avg_retention, revenue_estimate
            FROM daily_metrics
            WHERE channel_id = ? AND date >= ?
            ORDER BY date ASC
        """, (channel_id, start_date))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "date": row[0],
                "subscribers": row[1],
                "views": row[2],
                "ctr": row[3],
                "retention": row[4],
                "revenue": row[5]
            }
            for row in rows
        ]

    def update_goal(self, goal_key: str, current_value: float):
        """Update progress toward a goal."""
        if goal_key in self.goals:
            self.goals[goal_key].current_value = current_value

            if self.goals[goal_key].progress_percent >= 100:
                logger.success(f"Goal achieved: {self.goals[goal_key].goal_name}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        # Today's metrics
        cursor.execute("""
            SELECT
                SUM(total_views) as views,
                SUM(subscribers) as subs,
                SUM(revenue_estimate) as revenue,
                SUM(videos_published) as videos,
                SUM(shorts_published) as shorts
            FROM daily_metrics
            WHERE date = ?
        """, (today,))

        today_row = cursor.fetchone()

        # This week's metrics
        week_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        cursor.execute("""
            SELECT
                SUM(total_views) as views,
                SUM(subscribers) as subs,
                SUM(revenue_estimate) as revenue
            FROM daily_metrics
            WHERE date >= ?
        """, (week_start,))

        week_row = cursor.fetchone()

        # Token costs today
        cursor.execute("""
            SELECT SUM(cost) FROM token_usage WHERE date = ?
        """, (today,))

        cost_row = cursor.fetchone()

        conn.close()

        return {
            "today": {
                "views": today_row[0] or 0,
                "subscribers": today_row[1] or 0,
                "revenue": today_row[2] or 0.0,
                "videos": today_row[3] or 0,
                "shorts": today_row[4] or 0,
                "api_cost": cost_row[0] or 0.0
            },
            "week": {
                "views": week_row[0] or 0,
                "subscribers": week_row[1] or 0,
                "revenue": week_row[2] or 0.0
            },
            "goals": {
                key: {
                    "name": goal.goal_name,
                    "progress": goal.progress_percent,
                    "current": goal.current_value,
                    "target": goal.target_value,
                    "unit": goal.unit,
                    "on_track": goal.on_track
                }
                for key, goal in self.goals.items()
            }
        }

    def print_dashboard(self):
        """Print a formatted dashboard to console."""
        data = self.get_dashboard_data()

        print("\n" + "=" * 60)
        print("📊 YOUTUBE AUTOMATION SUCCESS DASHBOARD")
        print("=" * 60)

        print("\n📅 TODAY'S METRICS")
        print("-" * 40)
        print(f"  Views:        {data['today']['views']:,}")
        print(f"  Subscribers:  {data['today']['subscribers']:,}")
        print(f"  Revenue:      ${data['today']['revenue']:.2f}")
        print(f"  Videos:       {data['today']['videos']}")
        print(f"  Shorts:       {data['today']['shorts']}")
        print(f"  API Cost:     ${data['today']['api_cost']:.4f}")

        print("\n📈 THIS WEEK")
        print("-" * 40)
        print(f"  Total Views:  {data['week']['views']:,}")
        print(f"  New Subs:     {data['week']['subscribers']:,}")
        print(f"  Revenue:      ${data['week']['revenue']:.2f}")

        print("\n🎯 GOAL PROGRESS")
        print("-" * 40)
        for key, goal in data['goals'].items():
            status = "✅" if goal['progress'] >= 100 else ("🔄" if goal['on_track'] else "⚠️")
            bar_filled = int(goal['progress'] / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(f"  {status} {goal['name'][:30]}")
            print(f"     [{bar}] {goal['progress']:.1f}%")
            print(f"     {goal['current']:.0f} / {goal['target']:.0f} {goal['unit']}")

        print("\n" + "=" * 60)

    def generate_report(self, period_days: int = 30) -> str:
        """Generate a comprehensive report."""
        data = self.get_dashboard_data()
        token_data = self.get_token_efficiency(period_days)

        report = f"""
# Success Tracker Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Period: Last {period_days} days

## Key Metrics

### Views & Growth
- Total Views (Week): {data['week']['views']:,}
- Total Subscribers: {data['week']['subscribers']:,}
- Estimated Revenue: ${data['week']['revenue']:.2f}

### Token Efficiency
- Total Tokens Used: {token_data['total_tokens']:,}
- Total API Cost: ${token_data['total_cost']:.4f}
- Avg Tokens/Operation: {token_data['avg_tokens_per_op']:.0f}

### Goal Progress
"""
        for key, goal in data['goals'].items():
            status = "ACHIEVED" if goal['progress'] >= 100 else ("ON TRACK" if goal['on_track'] else "BEHIND")
            report += f"- {goal['name']}: {goal['progress']:.1f}% [{status}]\n"

        return report


# Singleton instance
_tracker = None


def get_success_tracker() -> SuccessTracker:
    """Get or create the global success tracker."""
    global _tracker
    if _tracker is None:
        _tracker = SuccessTracker()
    return _tracker


if __name__ == "__main__":
    # Demo usage
    tracker = get_success_tracker()

    # Record some test data
    test_metrics = ChannelMetrics(
        channel_id="money_blueprints",
        subscribers=150,
        total_views=5000,
        avg_ctr=4.5,
        avg_retention=45.0,
        revenue_estimate=25.0,
        videos_published=2,
        shorts_published=4
    )
    tracker.record_daily_metrics(test_metrics)

    # Record token usage
    tracker.record_token_usage("groq", 5000, 0.01, "script_generation")

    # Update goals
    tracker.update_goal("subscribers_1k", 150)
    tracker.update_goal("token_efficiency", 25)

    # Print dashboard
    tracker.print_dashboard()
