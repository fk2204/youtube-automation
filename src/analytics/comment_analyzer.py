"""
Comment Sentiment Analysis Module for YouTube Automation

Analyze YouTube comments to extract insights, identify trends, and
generate content ideas from audience feedback.

Features:
- Fetch comments via YouTube API
- Sentiment scoring (positive/negative/neutral)
- Extract common questions
- Identify viral comment threads
- Generate content ideas from comments
- Alert on negative sentiment spikes

Usage:
    from src.analytics.comment_analyzer import CommentAnalyzer

    analyzer = CommentAnalyzer()

    # Analyze comments for a video
    analysis = await analyzer.analyze_video_comments("VIDEO_ID")
    print(f"Sentiment: {analysis.overall_sentiment}")
    print(f"Common Questions: {analysis.questions[:5]}")

    # Get content ideas from comments
    ideas = await analyzer.generate_content_ideas("VIDEO_ID")

    # Monitor sentiment across channel
    alerts = await analyzer.check_sentiment_alerts("CHANNEL_ID")
"""

import os
import re
import json
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from pathlib import Path
from loguru import logger

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    logger.warning("Google API libraries not installed")
    YOUTUBE_API_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    logger.warning("TextBlob not installed. Install with: pip install textblob")
    TEXTBLOB_AVAILABLE = False


@dataclass
class Comment:
    """Represents a YouTube comment."""
    comment_id: str
    video_id: str
    author: str
    text: str
    like_count: int
    reply_count: int
    published_at: datetime
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_label: str = "neutral"  # positive, negative, neutral
    is_question: bool = False
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["published_at"] = self.published_at.isoformat()
        return result


@dataclass
class CommentThread:
    """A comment thread with replies."""
    top_comment: Comment
    replies: List[Comment] = field(default_factory=list)
    engagement_score: float = 0.0
    is_viral: bool = False

    @property
    def total_likes(self) -> int:
        return self.top_comment.like_count + sum(r.like_count for r in self.replies)

    @property
    def total_replies(self) -> int:
        return len(self.replies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_comment": self.top_comment.to_dict(),
            "replies": [r.to_dict() for r in self.replies],
            "engagement_score": self.engagement_score,
            "is_viral": self.is_viral,
            "total_likes": self.total_likes,
            "total_replies": self.total_replies,
        }


@dataclass
class SentimentAnalysis:
    """Results of sentiment analysis."""
    video_id: str
    total_comments: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_sentiment: float
    sentiment_distribution: Dict[str, float]  # label -> percentage

    @property
    def overall_sentiment(self) -> str:
        if self.average_sentiment > 0.1:
            return "positive"
        elif self.average_sentiment < -0.1:
            return "negative"
        return "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoCommentAnalysis:
    """Comprehensive comment analysis for a video."""
    video_id: str
    video_title: Optional[str]
    analysis_time: datetime
    total_comments: int
    sentiment: SentimentAnalysis
    questions: List[str]
    common_topics: List[Tuple[str, int]]  # topic, count
    viral_threads: List[CommentThread]
    content_ideas: List[str]
    top_positive_comments: List[Comment]
    top_negative_comments: List[Comment]
    engagement_insights: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "analysis_time": self.analysis_time.isoformat(),
            "total_comments": self.total_comments,
            "sentiment": self.sentiment.to_dict(),
            "questions": self.questions,
            "common_topics": self.common_topics,
            "viral_threads": [t.to_dict() for t in self.viral_threads],
            "content_ideas": self.content_ideas,
            "top_positive_comments": [c.to_dict() for c in self.top_positive_comments],
            "top_negative_comments": [c.to_dict() for c in self.top_negative_comments],
            "engagement_insights": self.engagement_insights,
        }


@dataclass
class SentimentAlert:
    """Alert for sentiment anomalies."""
    video_id: str
    video_title: str
    alert_type: str  # spike_negative, spike_positive, trending_topic
    severity: str  # info, warning, critical
    message: str
    current_sentiment: float
    baseline_sentiment: float
    sample_comments: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


class SentimentAnalyzer:
    """Analyze sentiment of text using multiple methods."""

    # Words that strongly indicate sentiment
    POSITIVE_WORDS = {
        "amazing", "awesome", "excellent", "fantastic", "great", "incredible",
        "love", "perfect", "wonderful", "best", "helpful", "informative",
        "brilliant", "outstanding", "superb", "thank", "thanks", "appreciated",
        "learned", "valuable", "insightful", "quality", "recommend", "subscribed"
    }

    NEGATIVE_WORDS = {
        "terrible", "awful", "horrible", "bad", "worst", "hate", "boring",
        "waste", "useless", "disappointing", "clickbait", "misleading",
        "wrong", "false", "stupid", "dumb", "annoying", "scam", "fake",
        "unsubscribed", "dislike", "garbage", "trash", "pathetic"
    }

    # Question patterns
    QUESTION_PATTERNS = [
        r"\?$",
        r"^(how|what|why|when|where|who|which|can|could|would|should|is|are|do|does)",
        r"anyone (know|else|tried|have)",
        r"(please|pls) (explain|help|tell|show)",
        r"(what is|what's|whats) (the|a|your)",
    ]

    def __init__(self):
        self.use_textblob = TEXTBLOB_AVAILABLE

    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of text.

        Returns:
            Tuple of (score from -1 to 1, label)
        """
        if not text or not text.strip():
            return 0.0, "neutral"

        text_lower = text.lower()

        # Count sentiment words
        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in text_lower)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in text_lower)

        # Word-based score
        word_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)

        # Use TextBlob if available for more nuanced analysis
        if self.use_textblob:
            try:
                blob = TextBlob(text)
                textblob_score = blob.sentiment.polarity

                # Combine scores (weighted average)
                score = (word_score * 0.4 + textblob_score * 0.6)
            except Exception:
                score = word_score
        else:
            score = word_score

        # Clamp score
        score = max(-1.0, min(1.0, score))

        # Determine label
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        return score, label

    def is_question(self, text: str) -> bool:
        """Check if text is a question."""
        text_lower = text.lower().strip()

        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def extract_topics(self, text: str, min_length: int = 4) -> List[str]:
        """Extract potential topics/keywords from text."""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s#@]', ' ', text)

        # Extract words
        words = text.lower().split()

        # Common stop words to filter
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because", "until",
            "while", "this", "that", "these", "those", "i", "me", "my", "you",
            "your", "he", "she", "it", "we", "they", "them", "his", "her", "its",
            "video", "watch", "like", "comment", "subscribe", "channel", "youtube"
        }

        topics = [w for w in words if len(w) >= min_length and w not in stop_words]

        return list(set(topics))


class CommentFetcher:
    """Fetch comments from YouTube API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        self._youtube = None

    @property
    def youtube(self):
        """Lazy-load YouTube API client."""
        if self._youtube is None:
            if not self.api_key:
                raise ValueError("YouTube API key required. Set YOUTUBE_API_KEY env var.")

            if not YOUTUBE_API_AVAILABLE:
                raise ImportError("Google API libraries required. pip install google-api-python-client")

            self._youtube = build("youtube", "v3", developerKey=self.api_key)

        return self._youtube

    async def fetch_comments(
        self,
        video_id: str,
        max_results: int = 100,
        include_replies: bool = True
    ) -> List[CommentThread]:
        """
        Fetch comments for a video.

        Args:
            video_id: YouTube video ID
            max_results: Maximum number of top-level comments
            include_replies: Whether to fetch replies

        Returns:
            List of CommentThread objects
        """
        logger.info(f"Fetching comments for video {video_id}")

        threads = []

        try:
            # Fetch comment threads
            request = self.youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(max_results, 100),
                order="relevance",
                textFormat="plainText"
            )

            while request and len(threads) < max_results:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, request.execute
                )

                for item in response.get("items", []):
                    thread = self._parse_comment_thread(item, video_id, include_replies)
                    if thread:
                        threads.append(thread)

                # Get next page
                if "nextPageToken" in response and len(threads) < max_results:
                    request = self.youtube.commentThreads().list(
                        part="snippet,replies",
                        videoId=video_id,
                        maxResults=min(max_results - len(threads), 100),
                        order="relevance",
                        textFormat="plainText",
                        pageToken=response["nextPageToken"]
                    )
                else:
                    break

        except HttpError as e:
            if e.resp.status == 403:
                logger.warning(f"Comments disabled for video {video_id}")
            else:
                logger.error(f"Error fetching comments: {e}")
        except Exception as e:
            logger.error(f"Error fetching comments: {e}")

        logger.info(f"Fetched {len(threads)} comment threads")
        return threads

    def _parse_comment_thread(
        self,
        item: Dict,
        video_id: str,
        include_replies: bool
    ) -> Optional[CommentThread]:
        """Parse a comment thread from API response."""
        try:
            snippet = item["snippet"]["topLevelComment"]["snippet"]

            top_comment = Comment(
                comment_id=item["id"],
                video_id=video_id,
                author=snippet.get("authorDisplayName", "Unknown"),
                text=snippet.get("textDisplay", ""),
                like_count=snippet.get("likeCount", 0),
                reply_count=item["snippet"].get("totalReplyCount", 0),
                published_at=datetime.fromisoformat(
                    snippet["publishedAt"].replace("Z", "+00:00")
                )
            )

            replies = []
            if include_replies and "replies" in item:
                for reply_item in item["replies"]["comments"]:
                    reply_snippet = reply_item["snippet"]
                    reply = Comment(
                        comment_id=reply_item["id"],
                        video_id=video_id,
                        author=reply_snippet.get("authorDisplayName", "Unknown"),
                        text=reply_snippet.get("textDisplay", ""),
                        like_count=reply_snippet.get("likeCount", 0),
                        reply_count=0,
                        published_at=datetime.fromisoformat(
                            reply_snippet["publishedAt"].replace("Z", "+00:00")
                        )
                    )
                    replies.append(reply)

            return CommentThread(top_comment=top_comment, replies=replies)

        except Exception as e:
            logger.debug(f"Error parsing comment: {e}")
            return None

    async def get_video_title(self, video_id: str) -> Optional[str]:
        """Get video title."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.youtube.videos().list(
                    part="snippet",
                    id=video_id
                ).execute()
            )

            if response.get("items"):
                return response["items"][0]["snippet"]["title"]
        except Exception as e:
            logger.debug(f"Could not get video title: {e}")

        return None


class CommentAnalyzer:
    """
    Main class for YouTube comment analysis.

    Provides comprehensive comment analysis including sentiment,
    question extraction, topic analysis, and content idea generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/comment_cache"
    ):
        self.fetcher = CommentFetcher(api_key)
        self.sentiment_analyzer = SentimentAnalyzer()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Sentiment baseline for alerts
        self._channel_baselines: Dict[str, float] = {}

        logger.info("CommentAnalyzer initialized")

    async def analyze_video_comments(
        self,
        video_id: str,
        max_comments: int = 100,
        include_replies: bool = True
    ) -> VideoCommentAnalysis:
        """
        Perform comprehensive analysis of video comments.

        Args:
            video_id: YouTube video ID
            max_comments: Maximum comments to analyze
            include_replies: Whether to include replies

        Returns:
            VideoCommentAnalysis object with all insights
        """
        logger.info(f"Analyzing comments for video {video_id}")

        # Fetch comments
        threads = await self.fetcher.fetch_comments(video_id, max_comments, include_replies)

        # Get video title
        video_title = await self.fetcher.get_video_title(video_id)

        # Analyze each comment
        all_comments = []
        questions = []
        all_topics = []

        for thread in threads:
            # Analyze top comment
            self._analyze_comment(thread.top_comment)
            all_comments.append(thread.top_comment)

            if thread.top_comment.is_question:
                questions.append(thread.top_comment.text)

            all_topics.extend(thread.top_comment.topics)

            # Analyze replies
            for reply in thread.replies:
                self._analyze_comment(reply)
                all_comments.append(reply)

                if reply.is_question:
                    questions.append(reply.text)

                all_topics.extend(reply.topics)

            # Calculate engagement score for thread
            thread.engagement_score = self._calculate_engagement_score(thread)

        # Calculate sentiment analysis
        sentiment = self._calculate_sentiment_analysis(video_id, all_comments)

        # Find common topics
        topic_counts = Counter(all_topics)
        common_topics = topic_counts.most_common(20)

        # Identify viral threads (high engagement)
        viral_threads = sorted(threads, key=lambda t: t.engagement_score, reverse=True)[:5]
        for thread in viral_threads:
            thread.is_viral = True

        # Generate content ideas
        content_ideas = self._generate_content_ideas(questions, common_topics, sentiment)

        # Get top comments by sentiment
        sorted_by_sentiment = sorted(all_comments, key=lambda c: c.sentiment_score, reverse=True)
        top_positive = [c for c in sorted_by_sentiment if c.sentiment_score > 0][:5]
        top_negative = [c for c in sorted_by_sentiment if c.sentiment_score < 0][-5:]

        # Engagement insights
        engagement_insights = self._calculate_engagement_insights(threads, all_comments)

        analysis = VideoCommentAnalysis(
            video_id=video_id,
            video_title=video_title,
            analysis_time=datetime.now(),
            total_comments=len(all_comments),
            sentiment=sentiment,
            questions=questions[:20],  # Limit to 20 questions
            common_topics=common_topics,
            viral_threads=viral_threads,
            content_ideas=content_ideas,
            top_positive_comments=top_positive,
            top_negative_comments=top_negative,
            engagement_insights=engagement_insights
        )

        # Cache analysis
        self._cache_analysis(analysis)

        logger.success(f"Analysis complete for {video_id}: {len(all_comments)} comments")
        return analysis

    def _analyze_comment(self, comment: Comment) -> None:
        """Analyze a single comment and update its properties."""
        # Sentiment
        score, label = self.sentiment_analyzer.analyze_text(comment.text)
        comment.sentiment_score = score
        comment.sentiment_label = label

        # Question detection
        comment.is_question = self.sentiment_analyzer.is_question(comment.text)

        # Topic extraction
        comment.topics = self.sentiment_analyzer.extract_topics(comment.text)

    def _calculate_sentiment_analysis(
        self,
        video_id: str,
        comments: List[Comment]
    ) -> SentimentAnalysis:
        """Calculate overall sentiment analysis."""
        if not comments:
            return SentimentAnalysis(
                video_id=video_id,
                total_comments=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                average_sentiment=0.0,
                sentiment_distribution={"positive": 0, "negative": 0, "neutral": 0}
            )

        positive = sum(1 for c in comments if c.sentiment_label == "positive")
        negative = sum(1 for c in comments if c.sentiment_label == "negative")
        neutral = sum(1 for c in comments if c.sentiment_label == "neutral")
        total = len(comments)

        avg_sentiment = sum(c.sentiment_score for c in comments) / total

        return SentimentAnalysis(
            video_id=video_id,
            total_comments=total,
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            average_sentiment=avg_sentiment,
            sentiment_distribution={
                "positive": (positive / total) * 100,
                "negative": (negative / total) * 100,
                "neutral": (neutral / total) * 100,
            }
        )

    def _calculate_engagement_score(self, thread: CommentThread) -> float:
        """Calculate engagement score for a comment thread."""
        # Factors: likes, replies, sentiment engagement
        likes = thread.top_comment.like_count
        replies = len(thread.replies)

        # Reply engagement (more replies = more engaging)
        reply_likes = sum(r.like_count for r in thread.replies)

        # Score formula: likes + (replies * 5) + reply_likes
        score = likes + (replies * 5) + reply_likes

        # Bonus for questions (tend to drive engagement)
        if thread.top_comment.is_question:
            score *= 1.2

        return score

    def _calculate_engagement_insights(
        self,
        threads: List[CommentThread],
        comments: List[Comment]
    ) -> Dict[str, Any]:
        """Calculate engagement insights from comments."""
        if not comments:
            return {}

        total_likes = sum(c.like_count for c in comments)
        question_count = sum(1 for c in comments if c.is_question)
        avg_comment_length = sum(len(c.text) for c in comments) / len(comments)

        # Time distribution
        comment_times = [c.published_at for c in comments]
        if comment_times:
            earliest = min(comment_times)
            latest = max(comment_times)
            time_span = (latest - earliest).total_seconds() / 3600  # hours
        else:
            time_span = 0

        # Most engaging authors
        author_engagement = Counter()
        for c in comments:
            author_engagement[c.author] += c.like_count

        top_authors = author_engagement.most_common(5)

        return {
            "total_likes": total_likes,
            "avg_likes_per_comment": total_likes / len(comments) if comments else 0,
            "question_percentage": (question_count / len(comments)) * 100 if comments else 0,
            "avg_comment_length": avg_comment_length,
            "comment_time_span_hours": time_span,
            "top_engaging_authors": top_authors,
            "total_threads": len(threads),
            "threads_with_replies": sum(1 for t in threads if t.replies),
        }

    def _generate_content_ideas(
        self,
        questions: List[str],
        common_topics: List[Tuple[str, int]],
        sentiment: SentimentAnalysis
    ) -> List[str]:
        """Generate content ideas from comment analysis."""
        ideas = []

        # Ideas from questions
        if questions:
            ideas.append(f"FAQ Video: Answer the top {min(5, len(questions))} questions from comments")
            for q in questions[:3]:
                # Clean question for idea
                clean_q = q[:100].replace('\n', ' ').strip()
                if clean_q:
                    ideas.append(f"Topic request: {clean_q}")

        # Ideas from common topics
        if common_topics:
            top_topics = [t[0] for t in common_topics[:5]]
            ideas.append(f"Deep dive into trending topic: {', '.join(top_topics[:3])}")

        # Ideas from sentiment
        if sentiment.negative_count > sentiment.total_comments * 0.3:
            ideas.append("Address concerns: Create clarification video for common misconceptions")

        if sentiment.positive_count > sentiment.total_comments * 0.7:
            ideas.append("Follow-up: Create sequel/continuation of this popular content")

        return ideas[:10]  # Limit to 10 ideas

    def _cache_analysis(self, analysis: VideoCommentAnalysis) -> None:
        """Cache analysis results."""
        cache_file = self.cache_dir / f"{analysis.video_id}_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Could not cache analysis: {e}")

    async def check_sentiment_alerts(
        self,
        video_ids: List[str],
        threshold_negative: float = 0.3,  # 30% negative triggers alert
        threshold_drop: float = 0.2  # 20% drop from baseline
    ) -> List[SentimentAlert]:
        """
        Check for sentiment anomalies across videos.

        Args:
            video_ids: List of video IDs to check
            threshold_negative: Negative percentage threshold for alert
            threshold_drop: Sentiment drop from baseline threshold

        Returns:
            List of SentimentAlert objects
        """
        alerts = []

        for video_id in video_ids:
            try:
                analysis = await self.analyze_video_comments(video_id, max_comments=50)

                # Check for high negative sentiment
                neg_percentage = analysis.sentiment.sentiment_distribution.get("negative", 0)

                if neg_percentage > threshold_negative * 100:
                    # Get sample negative comments
                    samples = [c.text[:100] for c in analysis.top_negative_comments[:3]]

                    alert = SentimentAlert(
                        video_id=video_id,
                        video_title=analysis.video_title or video_id,
                        alert_type="spike_negative",
                        severity="warning" if neg_percentage < 50 else "critical",
                        message=f"High negative sentiment detected: {neg_percentage:.1f}% negative comments",
                        current_sentiment=analysis.sentiment.average_sentiment,
                        baseline_sentiment=self._channel_baselines.get(video_id, 0.0),
                        sample_comments=samples
                    )
                    alerts.append(alert)

                # Check for sentiment drop from baseline
                baseline = self._channel_baselines.get(video_id)
                if baseline is not None:
                    drop = baseline - analysis.sentiment.average_sentiment
                    if drop > threshold_drop:
                        alert = SentimentAlert(
                            video_id=video_id,
                            video_title=analysis.video_title or video_id,
                            alert_type="sentiment_drop",
                            severity="warning",
                            message=f"Sentiment dropped {drop:.2f} from baseline",
                            current_sentiment=analysis.sentiment.average_sentiment,
                            baseline_sentiment=baseline,
                            sample_comments=[]
                        )
                        alerts.append(alert)

                # Update baseline
                self._channel_baselines[video_id] = analysis.sentiment.average_sentiment

            except Exception as e:
                logger.error(f"Error checking sentiment for {video_id}: {e}")

        return alerts

    async def generate_content_ideas(
        self,
        video_id: str,
        max_ideas: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate content ideas from video comments.

        Args:
            video_id: Video ID to analyze
            max_ideas: Maximum number of ideas to generate

        Returns:
            List of content idea dicts with title, description, source
        """
        analysis = await self.analyze_video_comments(video_id, max_comments=100)

        ideas = []

        # From questions
        for question in analysis.questions[:5]:
            ideas.append({
                "title": f"Answering: {question[:50]}...",
                "description": f"Based on audience question from comments",
                "source": "question",
                "original_text": question,
                "confidence": 0.8,
            })

        # From viral threads
        for thread in analysis.viral_threads[:3]:
            if thread.top_comment.text:
                ideas.append({
                    "title": f"Deep dive: {thread.top_comment.text[:50]}...",
                    "description": f"Highly engaged comment thread ({thread.total_likes} likes)",
                    "source": "viral_thread",
                    "original_text": thread.top_comment.text,
                    "confidence": 0.7,
                })

        # From common topics
        for topic, count in analysis.common_topics[:5]:
            if count >= 3:  # Mentioned at least 3 times
                ideas.append({
                    "title": f"Everything about {topic.title()}",
                    "description": f"Topic mentioned {count} times in comments",
                    "source": "common_topic",
                    "original_text": topic,
                    "confidence": 0.6,
                })

        # Sort by confidence and limit
        ideas.sort(key=lambda x: x["confidence"], reverse=True)
        return ideas[:max_ideas]

    async def get_question_summary(
        self,
        video_id: str,
        max_questions: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get a summary of questions from comments.

        Useful for creating FAQ videos or addressing audience concerns.

        Args:
            video_id: Video ID to analyze
            max_questions: Maximum questions to return

        Returns:
            List of question dicts with text, likes, answer_priority
        """
        analysis = await self.analyze_video_comments(video_id)

        # Find all question comments
        threads = await self.fetcher.fetch_comments(video_id, 100)

        questions = []
        for thread in threads:
            if thread.top_comment.is_question:
                questions.append({
                    "text": thread.top_comment.text,
                    "likes": thread.top_comment.like_count,
                    "replies": len(thread.replies),
                    "author": thread.top_comment.author,
                    "answer_priority": thread.top_comment.like_count + len(thread.replies) * 5,
                })

        # Sort by priority
        questions.sort(key=lambda q: q["answer_priority"], reverse=True)

        return questions[:max_questions]

    def export_analysis(
        self,
        analysis: VideoCommentAnalysis,
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export analysis to file.

        Args:
            analysis: VideoCommentAnalysis object
            output_path: Output file path
            format: Output format (json, csv)

        Returns:
            Path to exported file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(analysis.to_dict(), f, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Video ID", "Total Comments", "Avg Sentiment",
                               "Positive %", "Negative %", "Questions"])
                writer.writerow([
                    analysis.video_id,
                    analysis.total_comments,
                    f"{analysis.sentiment.average_sentiment:.3f}",
                    f"{analysis.sentiment.sentiment_distribution['positive']:.1f}%",
                    f"{analysis.sentiment.sentiment_distribution['negative']:.1f}%",
                    len(analysis.questions)
                ])
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Analysis exported to {output_path}")
        return output_path


# Convenience functions
async def analyze_video(video_id: str) -> VideoCommentAnalysis:
    """Quick function to analyze a video's comments."""
    analyzer = CommentAnalyzer()
    return await analyzer.analyze_video_comments(video_id)


async def get_content_ideas(video_id: str) -> List[Dict[str, Any]]:
    """Quick function to get content ideas from comments."""
    analyzer = CommentAnalyzer()
    return await analyzer.generate_content_ideas(video_id)


if __name__ == "__main__":
    import sys

    async def main():
        print("\n" + "=" * 60)
        print("COMMENT SENTIMENT ANALYZER")
        print("=" * 60 + "\n")

        if len(sys.argv) > 1:
            video_id = sys.argv[1]

            analyzer = CommentAnalyzer()

            print(f"Analyzing comments for video: {video_id}\n")

            try:
                analysis = await analyzer.analyze_video_comments(video_id)

                print(f"Video: {analysis.video_title or video_id}")
                print(f"Total Comments: {analysis.total_comments}")
                print(f"\nSentiment Analysis:")
                print(f"  Overall: {analysis.sentiment.overall_sentiment.upper()}")
                print(f"  Average Score: {analysis.sentiment.average_sentiment:.3f}")
                print(f"  Positive: {analysis.sentiment.sentiment_distribution['positive']:.1f}%")
                print(f"  Negative: {analysis.sentiment.sentiment_distribution['negative']:.1f}%")
                print(f"  Neutral: {analysis.sentiment.sentiment_distribution['neutral']:.1f}%")

                if analysis.questions:
                    print(f"\nTop Questions ({len(analysis.questions)} found):")
                    for q in analysis.questions[:5]:
                        print(f"  - {q[:80]}...")

                if analysis.common_topics:
                    print(f"\nCommon Topics:")
                    for topic, count in analysis.common_topics[:10]:
                        print(f"  - {topic}: {count} mentions")

                if analysis.content_ideas:
                    print(f"\nContent Ideas:")
                    for idea in analysis.content_ideas[:5]:
                        print(f"  - {idea}")

                # Export analysis
                export_path = f"output/comment_analysis_{video_id}.json"
                analyzer.export_analysis(analysis, export_path)
                print(f"\nFull analysis exported to: {export_path}")

            except Exception as e:
                print(f"Error: {e}")
                print("\nMake sure YOUTUBE_API_KEY is set in environment variables.")

        else:
            print("Usage: python -m src.analytics.comment_analyzer <video_id>")
            print("\nExample: python -m src.analytics.comment_analyzer dQw4w9WgXcQ")
            print("\nRequirements:")
            print("  - YOUTUBE_API_KEY environment variable")
            print("  - pip install google-api-python-client textblob")

    asyncio.run(main())
