"""
YouTube Analytics API v2 Integration Module

Provides comprehensive video analytics, retention analysis, and performance comparison.
Uses YouTube Analytics API v2 for detailed metrics.

Setup:
1. Enable "YouTube Analytics API" in Google Cloud Console
2. Add these scopes to your OAuth consent screen:
   - https://www.googleapis.com/auth/yt-analytics.readonly
   - https://www.googleapis.com/auth/yt-analytics-monetary.readonly
3. Re-authenticate if credentials don't include analytics scopes

Usage:
    from src.youtube.analytics_api import YouTubeAnalyticsAPI

    analytics = YouTubeAnalyticsAPI()
    video_stats = analytics.get_video_analytics("VIDEO_ID", days=28)
    print(f"Views: {video_stats.views}")
    print(f"Algorithm Score: {video_stats.get_algorithm_score()}")
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    raise ImportError(
        "Please install Google API libraries:\n"
        "pip install google-auth-oauthlib google-api-python-client"
    )

from .auth import YouTubeAuth


@dataclass
class VideoAnalytics:
    """
    Comprehensive video analytics data from YouTube Analytics API.

    Contains all key metrics that affect YouTube algorithm ranking
    and video performance assessment.
    """
    video_id: str
    views: int = 0
    watch_time_minutes: float = 0.0
    avg_view_duration: float = 0.0  # in seconds
    avg_percentage_viewed: float = 0.0  # 0-100
    ctr: float = 0.0  # Click-through rate (0-100)
    impressions: int = 0
    subscribers_gained: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    traffic_source_breakdown: Dict[str, float] = field(default_factory=dict)
    audience_retention_curve: List[float] = field(default_factory=list)

    # Additional metrics for algorithm analysis
    dislikes: int = 0
    estimated_minutes_watched: float = 0.0
    annotation_click_rate: float = 0.0
    annotation_close_rate: float = 0.0
    card_click_rate: float = 0.0
    card_teaser_click_rate: float = 0.0
    end_screen_element_click_rate: float = 0.0

    def get_algorithm_score(self) -> float:
        """
        Calculate YouTube algorithm favorability score (0-100).

        The YouTube algorithm primarily weighs:
        1. CTR (Click-Through Rate) - How enticing is the thumbnail/title
        2. Average Percentage Viewed - How engaging is the content
        3. Watch Time - Total minutes watched
        4. Engagement Rate - Likes, comments, shares relative to views

        Higher scores indicate better algorithm performance.

        Returns:
            Float between 0 and 100 representing algorithm favorability
        """
        score = 0.0

        # 1. CTR Score (30% weight) - Excellent: >10%, Good: 5-10%, Average: 2-5%
        if self.ctr >= 10:
            ctr_score = 30.0
        elif self.ctr >= 5:
            ctr_score = 20.0 + ((self.ctr - 5) / 5) * 10
        elif self.ctr >= 2:
            ctr_score = 10.0 + ((self.ctr - 2) / 3) * 10
        else:
            ctr_score = (self.ctr / 2) * 10
        score += ctr_score

        # 2. Average Percentage Viewed (35% weight) - Excellent: >70%, Good: 50-70%, Average: 30-50%
        if self.avg_percentage_viewed >= 70:
            retention_score = 35.0
        elif self.avg_percentage_viewed >= 50:
            retention_score = 25.0 + ((self.avg_percentage_viewed - 50) / 20) * 10
        elif self.avg_percentage_viewed >= 30:
            retention_score = 15.0 + ((self.avg_percentage_viewed - 30) / 20) * 10
        else:
            retention_score = (self.avg_percentage_viewed / 30) * 15
        score += retention_score

        # 3. Engagement Rate (25% weight) - Calculated as (likes + comments*5 + shares*10) / views
        if self.views > 0:
            engagement = (self.likes + self.comments * 5 + self.shares * 10) / self.views * 100
            # Excellent: >5%, Good: 2-5%, Average: 1-2%
            if engagement >= 5:
                engagement_score = 25.0
            elif engagement >= 2:
                engagement_score = 15.0 + ((engagement - 2) / 3) * 10
            elif engagement >= 1:
                engagement_score = 10.0 + ((engagement - 1) / 1) * 5
            else:
                engagement_score = engagement * 10
            score += engagement_score

        # 4. Watch Time Factor (10% weight) - More watch time = better
        # Normalized based on views (avg 3 min watch time is good baseline)
        if self.views > 0:
            avg_watch_minutes = self.watch_time_minutes / self.views
            if avg_watch_minutes >= 5:
                watch_score = 10.0
            elif avg_watch_minutes >= 3:
                watch_score = 7.0 + ((avg_watch_minutes - 3) / 2) * 3
            elif avg_watch_minutes >= 1:
                watch_score = 3.0 + ((avg_watch_minutes - 1) / 2) * 4
            else:
                watch_score = avg_watch_minutes * 3
            score += watch_score

        return min(100.0, max(0.0, score))

    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary."""
        algorithm_score = self.get_algorithm_score()

        if algorithm_score >= 80:
            performance = "EXCELLENT - Viral potential"
        elif algorithm_score >= 60:
            performance = "GOOD - Above average performance"
        elif algorithm_score >= 40:
            performance = "AVERAGE - Room for improvement"
        elif algorithm_score >= 20:
            performance = "BELOW AVERAGE - Needs optimization"
        else:
            performance = "POOR - Significant issues"

        engagement_rate = 0.0
        if self.views > 0:
            engagement_rate = (self.likes + self.comments * 5 + self.shares * 10) / self.views * 100

        lines = [
            f"Video Performance Summary for {self.video_id}",
            "=" * 50,
            f"Algorithm Score: {algorithm_score:.1f}/100 ({performance})",
            "",
            "Key Metrics:",
            f"  Views: {self.views:,}",
            f"  Watch Time: {self.watch_time_minutes:,.0f} minutes",
            f"  Avg View Duration: {self.avg_view_duration:.1f} seconds",
            f"  Avg Percentage Viewed: {self.avg_percentage_viewed:.1f}%",
            "",
            "Click Performance:",
            f"  Impressions: {self.impressions:,}",
            f"  CTR: {self.ctr:.2f}%",
            "",
            "Engagement:",
            f"  Likes: {self.likes:,}",
            f"  Comments: {self.comments:,}",
            f"  Shares: {self.shares:,}",
            f"  Subscribers Gained: {self.subscribers_gained:,}",
            f"  Engagement Rate: {engagement_rate:.2f}%",
        ]

        if self.traffic_source_breakdown:
            lines.append("")
            lines.append("Traffic Sources:")
            for source, percentage in sorted(
                self.traffic_source_breakdown.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                lines.append(f"  {source}: {percentage:.1f}%")

        return "\n".join(lines)


@dataclass
class RetentionDropoff:
    """Represents a significant audience retention dropoff point."""
    timestamp_seconds: int
    percentage_drop: float
    retention_before: float
    retention_after: float
    severity: str  # "minor", "moderate", "severe"

    def get_timestamp_str(self) -> str:
        """Get formatted timestamp string (MM:SS)."""
        minutes = self.timestamp_seconds // 60
        seconds = self.timestamp_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class ChannelComparison:
    """Comparison of video performance against channel averages."""
    video_id: str
    channel_id: str
    days_analyzed: int

    # Video metrics
    video_views: int
    video_ctr: float
    video_retention: float
    video_engagement_rate: float

    # Channel averages
    channel_avg_views: float
    channel_avg_ctr: float
    channel_avg_retention: float
    channel_avg_engagement_rate: float

    # Percentile rankings (0-100, higher is better)
    views_percentile: float
    ctr_percentile: float
    retention_percentile: float
    engagement_percentile: float

    def get_summary(self) -> str:
        """Get human-readable comparison summary."""
        lines = [
            f"Video vs Channel Comparison ({self.days_analyzed} days)",
            "=" * 50,
            "",
            f"{'Metric':<20} {'Video':<15} {'Channel Avg':<15} {'Percentile':<10}",
            "-" * 60,
            f"{'Views':<20} {self.video_views:<15,} {self.channel_avg_views:<15,.0f} {self.views_percentile:.0f}%",
            f"{'CTR':<20} {self.video_ctr:<15.2f}% {self.channel_avg_ctr:<15.2f}% {self.ctr_percentile:.0f}%",
            f"{'Retention':<20} {self.video_retention:<15.1f}% {self.channel_avg_retention:<15.1f}% {self.retention_percentile:.0f}%",
            f"{'Engagement':<20} {self.video_engagement_rate:<15.2f}% {self.channel_avg_engagement_rate:<15.2f}% {self.engagement_percentile:.0f}%",
            "",
        ]

        # Overall assessment
        avg_percentile = (
            self.views_percentile +
            self.ctr_percentile +
            self.retention_percentile +
            self.engagement_percentile
        ) / 4

        if avg_percentile >= 75:
            assessment = "TOP PERFORMER - This video significantly outperforms your channel average"
        elif avg_percentile >= 50:
            assessment = "ABOVE AVERAGE - This video performs better than most of your content"
        elif avg_percentile >= 25:
            assessment = "BELOW AVERAGE - This video underperforms compared to your typical content"
        else:
            assessment = "UNDERPERFORMER - Consider analyzing what went wrong"

        lines.append(f"Overall: {assessment}")
        lines.append(f"Average Percentile: {avg_percentile:.0f}%")

        return "\n".join(lines)


class YouTubeAnalyticsAPI:
    """
    YouTube Analytics API v2 integration for video performance analysis.

    Provides comprehensive analytics including:
    - Video metrics (views, watch time, engagement)
    - Audience retention analysis
    - Traffic source breakdown
    - Channel comparison and benchmarking
    """

    # Analytics API scopes
    ANALYTICS_SCOPES = [
        "https://www.googleapis.com/auth/yt-analytics.readonly",
        "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"
    ]

    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"

    # Data API for supplementary info
    DATA_API_SERVICE = "youtube"
    DATA_API_VERSION = "v3"

    def __init__(self, credentials_file: Optional[str] = None):
        """
        Initialize YouTube Analytics API client.

        Args:
            credentials_file: Path to OAuth credentials file.
                            If None, uses default from YouTubeAuth.
        """
        self.credentials_file = credentials_file

        # Extend scopes in YouTubeAuth to include analytics
        self.auth = YouTubeAuth(credentials_file=credentials_file)

        # Ensure analytics scopes are included
        original_scopes = list(self.auth.SCOPES)
        for scope in self.ANALYTICS_SCOPES:
            if scope not in original_scopes:
                original_scopes.append(scope)
        self.auth.SCOPES = original_scopes

        self._analytics_service = None
        self._data_service = None

    @property
    def analytics(self):
        """Get authenticated YouTube Analytics API service (lazy load)."""
        if self._analytics_service is None:
            credentials = self.auth.get_credentials()
            self._analytics_service = build(
                self.API_SERVICE_NAME,
                self.API_VERSION,
                credentials=credentials
            )
            logger.info("YouTube Analytics API service created")
        return self._analytics_service

    @property
    def youtube(self):
        """Get authenticated YouTube Data API service (lazy load)."""
        if self._data_service is None:
            credentials = self.auth.get_credentials()
            self._data_service = build(
                self.DATA_API_SERVICE,
                self.DATA_API_VERSION,
                credentials=credentials
            )
            logger.info("YouTube Data API service created")
        return self._data_service

    def _get_channel_id(self) -> Optional[str]:
        """Get the authenticated user's channel ID."""
        try:
            response = self.youtube.channels().list(
                part="id",
                mine=True
            ).execute()

            if response.get("items"):
                return response["items"][0]["id"]
        except HttpError as e:
            logger.error(f"Failed to get channel ID: {e}")
        return None

    def get_video_analytics(
        self,
        video_id: str,
        days: int = 28
    ) -> VideoAnalytics:
        """
        Get comprehensive analytics for a specific video.

        Args:
            video_id: YouTube video ID
            days: Number of days to analyze (default 28)

        Returns:
            VideoAnalytics object with all metrics
        """
        logger.info(f"Fetching analytics for video {video_id} (last {days} days)")

        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Initialize analytics object
        analytics = VideoAnalytics(video_id=video_id)

        # Get channel ID for queries
        channel_id = self._get_channel_id()
        if not channel_id:
            logger.error("Could not determine channel ID")
            return analytics

        try:
            # Query 1: Basic metrics
            basic_response = self.analytics.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,likes,comments,shares,subscribersGained",
                filters=f"video=={video_id}"
            ).execute()

            if basic_response.get("rows"):
                row = basic_response["rows"][0]
                analytics.views = int(row[0])
                analytics.watch_time_minutes = float(row[1])
                analytics.avg_view_duration = float(row[2])
                analytics.avg_percentage_viewed = float(row[3])
                analytics.likes = int(row[4])
                analytics.comments = int(row[5])
                analytics.shares = int(row[6])
                analytics.subscribers_gained = int(row[7])

            # Query 2: Impressions and CTR
            try:
                impressions_response = self.analytics.reports().query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date,
                    endDate=end_date,
                    metrics="impressions,impressionClickThroughRate",
                    filters=f"video=={video_id}"
                ).execute()

                if impressions_response.get("rows"):
                    row = impressions_response["rows"][0]
                    analytics.impressions = int(row[0])
                    analytics.ctr = float(row[1]) * 100  # Convert to percentage
            except HttpError as e:
                logger.warning(f"Could not fetch impressions data: {e}")

            # Query 3: Traffic sources
            try:
                traffic_response = self.analytics.reports().query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date,
                    endDate=end_date,
                    dimensions="insightTrafficSourceType",
                    metrics="views",
                    filters=f"video=={video_id}",
                    sort="-views"
                ).execute()

                if traffic_response.get("rows"):
                    total_views = sum(int(row[1]) for row in traffic_response["rows"])
                    if total_views > 0:
                        for row in traffic_response["rows"]:
                            source = row[0]
                            views = int(row[1])
                            percentage = (views / total_views) * 100
                            analytics.traffic_source_breakdown[source] = percentage
            except HttpError as e:
                logger.warning(f"Could not fetch traffic source data: {e}")

            # Query 4: Audience retention curve (if available)
            try:
                retention_response = self.analytics.reports().query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date,
                    endDate=end_date,
                    dimensions="elapsedVideoTimeRatio",
                    metrics="audienceWatchRatio",
                    filters=f"video=={video_id}",
                    sort="elapsedVideoTimeRatio"
                ).execute()

                if retention_response.get("rows"):
                    analytics.audience_retention_curve = [
                        float(row[1]) * 100 for row in retention_response["rows"]
                    ]
            except HttpError as e:
                logger.warning(f"Could not fetch retention curve: {e}")

            # Query 5: Card and annotation metrics (optional)
            try:
                engagement_response = self.analytics.reports().query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date,
                    endDate=end_date,
                    metrics="cardClickRate,cardTeaserClickRate",
                    filters=f"video=={video_id}"
                ).execute()

                if engagement_response.get("rows"):
                    row = engagement_response["rows"][0]
                    analytics.card_click_rate = float(row[0]) * 100
                    analytics.card_teaser_click_rate = float(row[1]) * 100
            except HttpError:
                pass  # These metrics may not be available for all videos

            logger.success(f"Analytics fetched for video {video_id}")

        except HttpError as e:
            logger.error(f"Failed to fetch analytics: {e}")
            # Return analytics object with zeros if API fails

        return analytics

    def get_retention_dropoff_points(
        self,
        video_id: str,
        threshold: float = 5.0
    ) -> List[RetentionDropoff]:
        """
        Identify significant audience retention dropoff points.

        Analyzes the retention curve to find moments where viewers
        leave the video at higher rates than normal.

        Args:
            video_id: YouTube video ID
            threshold: Minimum percentage drop to consider significant (default 5%)

        Returns:
            List of RetentionDropoff objects sorted by severity
        """
        logger.info(f"Analyzing retention dropoffs for video {video_id}")

        dropoffs = []

        # Get video analytics with retention curve
        analytics = self.get_video_analytics(video_id)

        if not analytics.audience_retention_curve:
            logger.warning("No retention curve data available")
            return dropoffs

        retention = analytics.audience_retention_curve

        # Get video duration for timestamp calculation
        video_duration = self._get_video_duration(video_id)
        if video_duration == 0:
            video_duration = len(retention) * 10  # Estimate 10 seconds per data point

        # Analyze retention curve for dropoffs
        for i in range(1, len(retention)):
            drop = retention[i - 1] - retention[i]

            if drop >= threshold:
                # Calculate timestamp
                progress_ratio = i / len(retention)
                timestamp_seconds = int(progress_ratio * video_duration)

                # Determine severity
                if drop >= 15:
                    severity = "severe"
                elif drop >= 10:
                    severity = "moderate"
                else:
                    severity = "minor"

                dropoffs.append(RetentionDropoff(
                    timestamp_seconds=timestamp_seconds,
                    percentage_drop=drop,
                    retention_before=retention[i - 1],
                    retention_after=retention[i],
                    severity=severity
                ))

        # Sort by severity (severe first) then by drop percentage
        severity_order = {"severe": 0, "moderate": 1, "minor": 2}
        dropoffs.sort(key=lambda x: (severity_order[x.severity], -x.percentage_drop))

        logger.info(f"Found {len(dropoffs)} significant dropoff points")
        return dropoffs

    def _get_video_duration(self, video_id: str) -> int:
        """Get video duration in seconds from Data API."""
        try:
            response = self.youtube.videos().list(
                part="contentDetails",
                id=video_id
            ).execute()

            if response.get("items"):
                duration_str = response["items"][0]["contentDetails"]["duration"]
                return self._parse_duration(duration_str)
        except HttpError as e:
            logger.warning(f"Could not get video duration: {e}")
        return 0

    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration string to seconds."""
        import re

        # PT#H#M#S format
        hours = 0
        minutes = 0
        seconds = 0

        hour_match = re.search(r'(\d+)H', duration_str)
        min_match = re.search(r'(\d+)M', duration_str)
        sec_match = re.search(r'(\d+)S', duration_str)

        if hour_match:
            hours = int(hour_match.group(1))
        if min_match:
            minutes = int(min_match.group(1))
        if sec_match:
            seconds = int(sec_match.group(1))

        return hours * 3600 + minutes * 60 + seconds

    def compare_to_channel_average(
        self,
        video_id: str,
        channel_id: Optional[str] = None,
        days: int = 90
    ) -> ChannelComparison:
        """
        Compare a video's performance to the channel average.

        Calculates percentile rankings for key metrics to show
        how the video performs relative to other channel content.

        Args:
            video_id: YouTube video ID to analyze
            channel_id: Channel ID (uses authenticated channel if None)
            days: Number of days to calculate averages (default 90)

        Returns:
            ChannelComparison object with metrics and percentiles
        """
        logger.info(f"Comparing video {video_id} to channel average ({days} days)")

        # Get channel ID if not provided
        if not channel_id:
            channel_id = self._get_channel_id()
        if not channel_id:
            logger.error("Could not determine channel ID")
            # Return empty comparison
            return ChannelComparison(
                video_id=video_id,
                channel_id="unknown",
                days_analyzed=days,
                video_views=0, video_ctr=0, video_retention=0, video_engagement_rate=0,
                channel_avg_views=0, channel_avg_ctr=0, channel_avg_retention=0, channel_avg_engagement_rate=0,
                views_percentile=0, ctr_percentile=0, retention_percentile=0, engagement_percentile=0
            )

        # Get video metrics
        video_analytics = self.get_video_analytics(video_id, days=days)

        # Calculate video engagement rate
        video_engagement_rate = 0.0
        if video_analytics.views > 0:
            video_engagement_rate = (
                video_analytics.likes +
                video_analytics.comments * 5 +
                video_analytics.shares * 10
            ) / video_analytics.views * 100

        # Get channel-wide metrics
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            # Get all videos' metrics for the channel
            channel_response = self.analytics.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                dimensions="video",
                metrics="views,estimatedMinutesWatched,averageViewPercentage,likes,comments,shares",
                maxResults=200,
                sort="-views"
            ).execute()

            videos_data = []
            if channel_response.get("rows"):
                for row in channel_response["rows"]:
                    vid = row[0]
                    views = int(row[1])
                    watch_time = float(row[2])
                    retention = float(row[3])
                    likes = int(row[4])
                    comments = int(row[5])
                    shares = int(row[6])

                    engagement = 0.0
                    if views > 0:
                        engagement = (likes + comments * 5 + shares * 10) / views * 100

                    videos_data.append({
                        "video_id": vid,
                        "views": views,
                        "retention": retention,
                        "engagement": engagement
                    })

            # Get CTR data separately (may not be available for all videos)
            ctr_data = {}
            try:
                ctr_response = self.analytics.reports().query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date,
                    endDate=end_date,
                    dimensions="video",
                    metrics="impressionClickThroughRate",
                    maxResults=200
                ).execute()

                if ctr_response.get("rows"):
                    for row in ctr_response["rows"]:
                        ctr_data[row[0]] = float(row[1]) * 100
            except HttpError:
                pass

            # Add CTR to videos data
            for vid_data in videos_data:
                vid_data["ctr"] = ctr_data.get(vid_data["video_id"], 0.0)

            # Calculate channel averages
            if videos_data:
                channel_avg_views = sum(v["views"] for v in videos_data) / len(videos_data)
                channel_avg_retention = sum(v["retention"] for v in videos_data) / len(videos_data)
                channel_avg_engagement = sum(v["engagement"] for v in videos_data) / len(videos_data)
                ctrs = [v["ctr"] for v in videos_data if v["ctr"] > 0]
                channel_avg_ctr = sum(ctrs) / len(ctrs) if ctrs else 0.0
            else:
                channel_avg_views = 0
                channel_avg_retention = 0
                channel_avg_engagement = 0
                channel_avg_ctr = 0

            # Calculate percentiles
            def calculate_percentile(value: float, values: List[float]) -> float:
                if not values:
                    return 50.0
                sorted_values = sorted(values)
                count_below = sum(1 for v in sorted_values if v < value)
                return (count_below / len(sorted_values)) * 100

            views_list = [v["views"] for v in videos_data]
            retention_list = [v["retention"] for v in videos_data]
            engagement_list = [v["engagement"] for v in videos_data]
            ctr_list = [v["ctr"] for v in videos_data if v["ctr"] > 0]

            views_percentile = calculate_percentile(video_analytics.views, views_list)
            retention_percentile = calculate_percentile(video_analytics.avg_percentage_viewed, retention_list)
            engagement_percentile = calculate_percentile(video_engagement_rate, engagement_list)
            ctr_percentile = calculate_percentile(video_analytics.ctr, ctr_list) if ctr_list else 50.0

            comparison = ChannelComparison(
                video_id=video_id,
                channel_id=channel_id,
                days_analyzed=days,
                video_views=video_analytics.views,
                video_ctr=video_analytics.ctr,
                video_retention=video_analytics.avg_percentage_viewed,
                video_engagement_rate=video_engagement_rate,
                channel_avg_views=channel_avg_views,
                channel_avg_ctr=channel_avg_ctr,
                channel_avg_retention=channel_avg_retention,
                channel_avg_engagement_rate=channel_avg_engagement,
                views_percentile=views_percentile,
                ctr_percentile=ctr_percentile,
                retention_percentile=retention_percentile,
                engagement_percentile=engagement_percentile
            )

            logger.success(f"Comparison complete for video {video_id}")
            return comparison

        except HttpError as e:
            logger.error(f"Failed to get channel comparison data: {e}")
            return ChannelComparison(
                video_id=video_id,
                channel_id=channel_id,
                days_analyzed=days,
                video_views=video_analytics.views,
                video_ctr=video_analytics.ctr,
                video_retention=video_analytics.avg_percentage_viewed,
                video_engagement_rate=video_engagement_rate,
                channel_avg_views=0, channel_avg_ctr=0, channel_avg_retention=0, channel_avg_engagement_rate=0,
                views_percentile=50, ctr_percentile=50, retention_percentile=50, engagement_percentile=50
            )

    def get_top_performing_videos(
        self,
        days: int = 30,
        limit: int = 10,
        metric: str = "views"
    ) -> List[Tuple[str, VideoAnalytics]]:
        """
        Get top performing videos by a specific metric.

        Args:
            days: Number of days to analyze
            limit: Maximum number of videos to return
            metric: Metric to sort by (views, watchTime, ctr, engagement)

        Returns:
            List of (video_id, VideoAnalytics) tuples
        """
        logger.info(f"Fetching top {limit} videos by {metric} (last {days} days)")

        channel_id = self._get_channel_id()
        if not channel_id:
            return []

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Map metric name to API metric
        metric_map = {
            "views": "views",
            "watchTime": "estimatedMinutesWatched",
            "ctr": "impressionClickThroughRate",
            "engagement": "likes"
        }
        api_metric = metric_map.get(metric, "views")

        try:
            response = self.analytics.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                dimensions="video",
                metrics=f"{api_metric}",
                maxResults=limit,
                sort=f"-{api_metric}"
            ).execute()

            results = []
            if response.get("rows"):
                for row in response["rows"]:
                    video_id = row[0]
                    analytics = self.get_video_analytics(video_id, days=days)
                    results.append((video_id, analytics))

            return results

        except HttpError as e:
            logger.error(f"Failed to get top performing videos: {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOUTUBE ANALYTICS API TEST")
    print("=" * 60 + "\n")

    analytics_api = YouTubeAnalyticsAPI()

    # Test with a sample video ID (replace with actual ID)
    test_video_id = "dQw4w9WgXcQ"  # Example video ID

    try:
        # Get video analytics
        print(f"Fetching analytics for video: {test_video_id}\n")
        video_stats = analytics_api.get_video_analytics(test_video_id, days=28)
        print(video_stats.get_performance_summary())
        print()

        # Get retention dropoffs
        print("Analyzing retention dropoffs...")
        dropoffs = analytics_api.get_retention_dropoff_points(test_video_id)
        if dropoffs:
            print(f"\nFound {len(dropoffs)} significant dropoff points:")
            for dropoff in dropoffs[:5]:  # Show top 5
                print(f"  {dropoff.get_timestamp_str()} - {dropoff.severity.upper()}: "
                      f"{dropoff.percentage_drop:.1f}% drop "
                      f"({dropoff.retention_before:.1f}% -> {dropoff.retention_after:.1f}%)")
        else:
            print("No significant dropoffs found")
        print()

        # Compare to channel average
        print("Comparing to channel average...")
        comparison = analytics_api.compare_to_channel_average(test_video_id)
        print(comparison.get_summary())

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Enabled YouTube Analytics API in Google Cloud Console")
        print("2. Added analytics scopes to OAuth consent screen")
        print("3. Re-authenticated if credentials were created before adding scopes")
