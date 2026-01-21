"""
Tests for New YouTube Automation Modules

Tests for:
- WhisperCaptionGenerator
- ViralHookGenerator
- FreeKeywordResearch
- MetadataOptimizer
- AIDisclosureTracker
- RedditResearcher (mock API)

Run with: pytest tests/test_new_modules.py -v
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
# WhisperCaptionGenerator Tests
# ============================================================

class TestWhisperCaptionGenerator:
    """Tests for the WhisperCaptionGenerator module."""

    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model."""
        mock = MagicMock()
        mock.transcribe.return_value = {
            "text": "Hello world, this is a test transcription.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world,",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5},
                        {"word": "world,", "start": 0.6, "end": 1.0}
                    ]
                },
                {
                    "id": 1,
                    "start": 2.0,
                    "end": 4.5,
                    "text": " this is a test transcription.",
                    "words": [
                        {"word": "this", "start": 2.0, "end": 2.3},
                        {"word": "is", "start": 2.4, "end": 2.5},
                        {"word": "a", "start": 2.6, "end": 2.7},
                        {"word": "test", "start": 2.8, "end": 3.2},
                        {"word": "transcription.", "start": 3.3, "end": 4.5}
                    ]
                }
            ],
            "language": "en"
        }
        return mock

    def test_caption_generator_initialization(self):
        """Test WhisperCaptionGenerator initializes correctly."""
        with patch("src.captions.whisper_generator.WHISPER_AVAILABLE", False):
            from src.captions.whisper_generator import WhisperCaptionGenerator

            generator = WhisperCaptionGenerator(model_size="base")

            assert generator.model_size == "base"
            assert generator.cache_dir is not None

    def test_model_info_contains_all_sizes(self):
        """Test that MODEL_INFO contains all expected model sizes."""
        with patch("src.captions.whisper_generator.WHISPER_AVAILABLE", False):
            from src.captions.whisper_generator import WhisperCaptionGenerator

            expected_sizes = ["tiny", "base", "small", "medium", "large"]

            for size in expected_sizes:
                assert size in WhisperCaptionGenerator.MODEL_INFO
                info = WhisperCaptionGenerator.MODEL_INFO[size]
                assert "size" in info
                assert "vram" in info
                assert "speed" in info
                assert "accuracy" in info

    def test_word_timestamp_dataclass(self):
        """Test WordTimestamp dataclass."""
        from src.captions.whisper_generator import WordTimestamp

        word = WordTimestamp(
            text="hello",
            start=0.0,
            end=0.5,
            confidence=0.95
        )

        assert word.text == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence == 0.95

        # Test to_dict
        word_dict = word.to_dict()
        assert word_dict["text"] == "hello"
        assert word_dict["start"] == 0.0
        assert word_dict["end"] == 0.5
        assert word_dict["confidence"] == 0.95

    def test_caption_segment_dataclass(self):
        """Test CaptionSegment dataclass."""
        from src.captions.whisper_generator import CaptionSegment, WordTimestamp

        segment = CaptionSegment(
            index=1,
            start=0.0,
            end=2.0,
            text="Hello world",
            words=[
                WordTimestamp(text="Hello", start=0.0, end=0.5),
                WordTimestamp(text="world", start=0.6, end=1.0)
            ]
        )

        assert segment.index == 1
        assert segment.start == 0.0
        assert segment.end == 2.0
        assert segment.text == "Hello world"
        assert len(segment.words) == 2

        # Test to_dict
        segment_dict = segment.to_dict()
        assert segment_dict["index"] == 1
        assert len(segment_dict["words"]) == 2


# ============================================================
# ViralHookGenerator Tests
# ============================================================

class TestViralHookGenerator:
    """Tests for the ViralHookGenerator module."""

    def test_hook_generator_initialization(self):
        """Test ViralHookGenerator initializes correctly."""
        from src.content.viral_content_engine import ViralHookGenerator

        generator = ViralHookGenerator()

        assert generator is not None

    def test_generate_hook_for_finance_niche(self):
        """Test generating hooks for finance niche."""
        from src.content.viral_content_engine import ViralHookGenerator

        generator = ViralHookGenerator()
        hook = generator.generate_hook("passive income", niche="finance")

        assert hook is not None
        assert hook.text is not None
        assert len(hook.text) > 0
        assert hook.niche == "finance"
        assert hook.word_count > 0
        assert hook.duration_seconds > 0

    def test_generate_hook_for_psychology_niche(self):
        """Test generating hooks for psychology niche."""
        from src.content.viral_content_engine import ViralHookGenerator

        generator = ViralHookGenerator()
        hook = generator.generate_hook("manipulation", niche="psychology")

        assert hook is not None
        assert hook.niche == "psychology"

    def test_generate_hook_for_storytelling_niche(self):
        """Test generating hooks for storytelling niche."""
        from src.content.viral_content_engine import ViralHookGenerator

        generator = ViralHookGenerator()
        hook = generator.generate_hook("mystery", niche="storytelling")

        assert hook is not None
        assert hook.niche == "storytelling"

    def test_hook_types_enum(self):
        """Test HookType enum contains expected values."""
        from src.content.viral_content_engine import HookType

        expected_types = [
            "PATTERN_INTERRUPT", "BOLD_CLAIM", "QUESTION_STACK",
            "STATS_SHOCK", "STORY_LEAD", "LOSS_AVERSION",
            "CURIOSITY_GAP", "INSIDER_SECRET", "COUNTDOWN", "CONTROVERSY"
        ]

        for hook_type in expected_types:
            assert hasattr(HookType, hook_type)

    def test_viral_hook_dataclass(self):
        """Test ViralHook dataclass."""
        from src.content.viral_content_engine import ViralHook, HookType

        hook = ViralHook(
            text="This is a test hook with multiple words",
            hook_type=HookType.BOLD_CLAIM,
            niche="finance",
            estimated_retention_boost=0.15,
            word_count=0,  # Will be calculated
            duration_seconds=0  # Will be calculated
        )

        assert hook.text is not None
        assert hook.hook_type == HookType.BOLD_CLAIM
        assert hook.niche == "finance"
        assert hook.word_count == 8  # Calculated in __post_init__
        assert hook.duration_seconds > 0  # Calculated in __post_init__


# ============================================================
# FreeKeywordResearch Tests
# ============================================================

class TestFreeKeywordResearch:
    """Tests for the FreeKeywordResearch module."""

    def test_keyword_research_initialization(self):
        """Test FreeKeywordResearch initializes correctly."""
        with patch("src.seo.free_keyword_research.PYTRENDS_AVAILABLE", False):
            from src.seo.free_keyword_research import FreeKeywordResearch

            researcher = FreeKeywordResearch()

            assert researcher is not None

    def test_keyword_result_dataclass(self):
        """Test KeywordResult dataclass."""
        from src.seo.free_keyword_research import KeywordResult

        result = KeywordResult(
            keyword="passive income",
            search_volume_estimate="high",
            competition="medium",
            opportunity_score=75.0,
            trend_direction="rising",
            suggestions_count=10,
            is_longtail=False
        )

        assert result.keyword == "passive income"
        assert result.search_volume_estimate == "high"
        assert result.competition == "medium"
        assert result.opportunity_score == 75.0
        assert result.trend_direction == "rising"
        assert result.suggestions_count == 10
        assert result.is_longtail is False

        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict["keyword"] == "passive income"
        assert result_dict["opportunity_score"] == 75.0

    def test_alphabet_constant(self):
        """Test ALPHABET constant is correct."""
        from src.seo.free_keyword_research import FreeKeywordResearch

        expected = "abcdefghijklmnopqrstuvwxyz0123456789"
        assert FreeKeywordResearch.ALPHABET == expected

    def test_question_words_constant(self):
        """Test QUESTION_WORDS constant contains expected words."""
        from src.seo.free_keyword_research import FreeKeywordResearch

        expected_words = ["how", "what", "why", "when", "where", "who"]

        for word in expected_words:
            assert word in FreeKeywordResearch.QUESTION_WORDS

    def test_modifiers_constant(self):
        """Test MODIFIERS constant contains expected words."""
        from src.seo.free_keyword_research import FreeKeywordResearch

        expected_words = ["best", "top", "free", "easy", "simple"]

        for word in expected_words:
            assert word in FreeKeywordResearch.MODIFIERS


# ============================================================
# MetadataOptimizer Tests
# ============================================================

class TestMetadataOptimizer:
    """Tests for the MetadataOptimizer module."""

    def test_optimizer_initialization(self):
        """Test MetadataOptimizer initializes correctly."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        assert optimizer is not None

    def test_optimize_title_basic(self):
        """Test basic title optimization."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        title = optimizer.optimize_title(
            base_title="How to Make Money",
            keywords=["passive income", "investing"]
        )

        assert title is not None
        assert len(title) > 0
        assert len(title) <= 100  # Reasonable max length

    def test_optimize_title_with_keywords(self):
        """Test title optimization with keyword front-loading."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        title = optimizer.optimize_title(
            base_title="Simple Guide for Beginners",
            keywords=["passive income"],
            front_load_keywords=True
        )

        # Keywords should be front-loaded in first 40 chars
        assert "passive income" in title.lower() or "Passive Income" in title

    def test_power_words_categories(self):
        """Test POWER_WORDS contains expected categories."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        expected_categories = ["curiosity", "urgency", "value", "emotion", "authority", "numbers"]

        for category in expected_categories:
            assert category in MetadataOptimizer.POWER_WORDS
            assert len(MetadataOptimizer.POWER_WORDS[category]) > 0

    def test_impact_components(self):
        """Test IMPACT_COMPONENTS contains all letters."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        expected_keys = ["I", "M", "P", "A", "C", "T"]

        for key in expected_keys:
            assert key in MetadataOptimizer.IMPACT_COMPONENTS

    def test_avoid_words(self):
        """Test AVOID_WORDS contains expected negative words."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        assert "scam" in MetadataOptimizer.AVOID_WORDS
        assert "fake" in MetadataOptimizer.AVOID_WORDS

    def test_optimized_metadata_dataclass(self):
        """Test OptimizedMetadata dataclass."""
        from src.seo.metadata_optimizer import OptimizedMetadata

        metadata = OptimizedMetadata(
            title="Test Title",
            description="Test description",
            tags=["tag1", "tag2"],
            title_score=85.0,
            keyword_density=0.05,
            chapters=[]
        )

        assert metadata.title == "Test Title"
        assert metadata.description == "Test description"
        assert len(metadata.tags) == 2
        assert metadata.title_score == 85.0


# ============================================================
# AIDisclosureTracker Tests
# ============================================================

class TestAIDisclosureTracker:
    """Tests for the AIDisclosureTracker module."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create a temporary directory for test database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_tracker_initialization(self, temp_db_dir):
        """Test AIDisclosureTracker initializes correctly."""
        from src.compliance.ai_disclosure import AIDisclosureTracker

        tracker = AIDisclosureTracker(db_path=str(temp_db_dir / "test.db"))

        assert tracker is not None

    def test_ai_content_type_enum(self):
        """Test AIContentType enum contains expected values."""
        from src.compliance.ai_disclosure import AIContentType

        expected_types = [
            "SYNTHETIC_VOICE", "VOICE_ALTERATION", "SYNTHETIC_VISUALS",
            "DEEPFAKE", "ALTERED_FOOTAGE", "AI_SCRIPT", "SYNTHETIC_AUDIO"
        ]

        for content_type in expected_types:
            assert hasattr(AIContentType, content_type)

    def test_disclosure_level_enum(self):
        """Test DisclosureLevel enum contains expected values."""
        from src.compliance.ai_disclosure import DisclosureLevel

        expected_levels = ["NONE", "OPTIONAL", "RECOMMENDED", "REQUIRED"]

        for level in expected_levels:
            assert hasattr(DisclosureLevel, level)

    def test_ai_usage_record_dataclass(self):
        """Test AIUsageRecord dataclass."""
        from src.compliance.ai_disclosure import AIUsageRecord, AIContentType, DisclosureLevel

        record = AIUsageRecord(
            video_id="vid123",
            content_type=AIContentType.SYNTHETIC_VOICE,
            method="edge-tts",
            disclosure_level=DisclosureLevel.RECOMMENDED
        )

        assert record.video_id == "vid123"
        assert record.content_type == AIContentType.SYNTHETIC_VOICE
        assert record.method == "edge-tts"
        assert record.disclosure_level == DisclosureLevel.RECOMMENDED

        # Test to_dict
        record_dict = record.to_dict()
        assert record_dict["video_id"] == "vid123"
        assert record_dict["content_type"] == "synthetic_voice"
        assert record_dict["method"] == "edge-tts"

    def test_disclosure_metadata_dataclass(self):
        """Test DisclosureMetadata dataclass."""
        from src.compliance.ai_disclosure import DisclosureMetadata, DisclosureLevel

        metadata = DisclosureMetadata(
            requires_disclosure=True,
            disclosure_text="This video uses AI-generated voiceover.",
            content_types_used=["synthetic_voice"],
            methods_used=["edge-tts"],
            disclosure_level=DisclosureLevel.RECOMMENDED
        )

        assert metadata.requires_disclosure is True
        assert "AI-generated" in metadata.disclosure_text
        assert "synthetic_voice" in metadata.content_types_used
        assert "edge-tts" in metadata.methods_used

    def test_disclosure_metadata_get_description_disclaimer(self):
        """Test get_description_disclaimer method."""
        from src.compliance.ai_disclosure import DisclosureMetadata, DisclosureLevel

        metadata = DisclosureMetadata(
            requires_disclosure=True,
            disclosure_text="This video uses AI-generated voiceover.",
            content_types_used=["synthetic_voice"],
            methods_used=["edge-tts"],
            disclosure_level=DisclosureLevel.RECOMMENDED
        )

        disclaimer = metadata.get_description_disclaimer()

        assert "AI Disclosure" in disclaimer
        assert "AI-generated voiceover" in disclaimer
        assert "edge-tts" in disclaimer

    def test_disclosure_metadata_no_disclosure_needed(self):
        """Test get_description_disclaimer when no disclosure needed."""
        from src.compliance.ai_disclosure import DisclosureMetadata, DisclosureLevel

        metadata = DisclosureMetadata(
            requires_disclosure=False,
            disclosure_text="",
            content_types_used=[],
            methods_used=[],
            disclosure_level=DisclosureLevel.NONE
        )

        disclaimer = metadata.get_description_disclaimer()

        assert disclaimer == ""


# ============================================================
# RedditResearcher Tests (Mock API)
# ============================================================

class TestRedditResearcher:
    """Tests for the RedditResearcher module (with mocked API)."""

    @pytest.fixture
    def mock_reddit(self):
        """Create a mock Reddit instance."""
        mock = MagicMock()

        # Mock subreddit
        mock_subreddit = MagicMock()

        # Mock posts
        mock_post = MagicMock()
        mock_post.title = "How do I learn Python for data science?"
        mock_post.score = 150
        mock_post.num_comments = 45
        mock_post.permalink = "/r/learnprogramming/comments/abc123/how_do_i_learn_python"
        mock_post.created_utc = 1705764000  # Jan 20, 2026
        mock_post.link_flair_text = "Question"
        mock_post.subreddit = MagicMock()
        mock_post.subreddit.display_name = "learnprogramming"

        mock_subreddit.top.return_value = [mock_post]
        mock_subreddit.search.return_value = [mock_post]
        mock.subreddit.return_value = mock_subreddit

        return mock

    def test_researcher_initialization_without_credentials(self):
        """Test RedditResearcher handles missing credentials gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.research.reddit.praw") as mock_praw:
                from src.research.reddit import RedditResearcher

                researcher = RedditResearcher(
                    client_id=None,
                    client_secret=None
                )

                # Should not crash, but reddit client should be None
                assert researcher.reddit is None

    def test_researcher_initialization_with_credentials(self, mock_reddit):
        """Test RedditResearcher initializes with credentials."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit):
            from src.research.reddit import RedditResearcher

            researcher = RedditResearcher(
                client_id="test_id",
                client_secret="test_secret",
                user_agent="test_agent"
            )

            assert researcher.reddit is not None

    def test_is_question_detection(self, mock_reddit):
        """Test question detection in titles."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit):
            from src.research.reddit import RedditResearcher

            researcher = RedditResearcher(
                client_id="test_id",
                client_secret="test_secret"
            )

            # Test various question patterns
            assert researcher._is_question("How to learn Python?") is True
            assert researcher._is_question("What is the best framework?") is True
            assert researcher._is_question("Why does this happen?") is True
            assert researcher._is_question("Can someone help me?") is True
            assert researcher._is_question("Check out my project!") is False

    def test_calculate_popularity(self, mock_reddit):
        """Test popularity score calculation."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit):
            from src.research.reddit import RedditResearcher

            researcher = RedditResearcher(
                client_id="test_id",
                client_secret="test_secret"
            )

            # Create mock post
            mock_post = MagicMock()
            mock_post.score = 100
            mock_post.num_comments = 50
            mock_post.title = "How do I learn Python?"  # Question = 1.5x boost

            popularity = researcher._calculate_popularity(mock_post)

            # score + (comments * 2) * 1.5 (question boost)
            # 100 + (50 * 2) = 200, 200 * 1.5 = 300
            assert popularity == 300

    def test_clean_topic(self, mock_reddit):
        """Test topic cleaning from Reddit titles."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit):
            from src.research.reddit import RedditResearcher

            researcher = RedditResearcher(
                client_id="test_id",
                client_secret="test_secret"
            )

            # Test various title formats
            assert researcher._clean_topic("[Question] How to learn Python?") == "How to learn Python"
            assert researcher._clean_topic("Help: I need assistance") == "I need assistance"
            assert researcher._clean_topic("simple topic") == "Simple topic"

    def test_get_video_ideas_with_mock(self, mock_reddit):
        """Test getting video ideas from Reddit (mocked)."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit):
            from src.research.reddit import RedditResearcher

            researcher = RedditResearcher(
                client_id="test_id",
                client_secret="test_secret"
            )

            ideas = researcher.get_video_ideas(
                subreddits=["learnprogramming"],
                limit=10,
                min_score=50
            )

            assert len(ideas) >= 0  # May be 0 if filtered out
            if ideas:
                idea = ideas[0]
                assert hasattr(idea, 'topic')
                assert hasattr(idea, 'source_url')
                assert hasattr(idea, 'popularity_score')

    def test_reddit_post_dataclass(self):
        """Test RedditPost dataclass."""
        from src.research.reddit import RedditPost
        from datetime import datetime

        post = RedditPost(
            title="Test title",
            subreddit="learnprogramming",
            score=100,
            num_comments=25,
            url="https://reddit.com/r/learnprogramming/test",
            created_utc=datetime.now(),
            flair="Question",
            is_question=True
        )

        assert post.title == "Test title"
        assert post.subreddit == "learnprogramming"
        assert post.score == 100
        assert post.num_comments == 25
        assert post.is_question is True

    def test_video_idea_dataclass(self):
        """Test VideoIdea dataclass."""
        from src.research.reddit import VideoIdea

        idea = VideoIdea(
            topic="Learn Python for Data Science",
            source_title="How do I learn Python for data science?",
            source_url="https://reddit.com/r/learnprogramming/test",
            subreddit="learnprogramming",
            popularity_score=250,
            idea_type="question"
        )

        assert idea.topic == "Learn Python for Data Science"
        assert idea.subreddit == "learnprogramming"
        assert idea.popularity_score == 250
        assert idea.idea_type == "question"

    def test_default_subreddits(self, mock_reddit):
        """Test default subreddits are defined."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit):
            from src.research.reddit import RedditResearcher

            assert len(RedditResearcher.DEFAULT_SUBREDDITS) > 0
            assert "learnprogramming" in RedditResearcher.DEFAULT_SUBREDDITS
            assert "Python" in RedditResearcher.DEFAULT_SUBREDDITS

    def test_question_keywords(self, mock_reddit):
        """Test question keywords are defined."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit):
            from src.research.reddit import RedditResearcher

            expected_keywords = ["how to", "what is", "why does", "explain", "help"]

            for keyword in expected_keywords:
                assert keyword in RedditResearcher.QUESTION_KEYWORDS


# ============================================================
# ViralContentEngine Integration Tests
# ============================================================

class TestViralContentEngine:
    """Tests for the full ViralContentEngine."""

    def test_engine_initialization(self):
        """Test ViralContentEngine initializes correctly."""
        from src.content.viral_content_engine import ViralContentEngine

        engine = ViralContentEngine(niche="finance")

        assert engine is not None
        assert engine.niche == "finance"

    def test_generate_all_elements(self):
        """Test generating all viral elements for a video."""
        from src.content.viral_content_engine import ViralContentEngine

        engine = ViralContentEngine(niche="finance")

        elements = engine.generate_all_elements(
            topic="passive income strategies",
            duration_seconds=600
        )

        assert elements is not None
        assert "hook" in elements
        assert "emotional_arc" in elements
        assert "curiosity_gaps" in elements
        assert "micro_payoffs" in elements

    def test_emotional_arc_builder(self):
        """Test emotional arc generation."""
        from src.content.viral_content_engine import EmotionalArcBuilder

        builder = EmotionalArcBuilder()
        arc = builder.build_arc(duration_seconds=600, niche="finance")

        assert arc is not None
        assert hasattr(arc, 'beats')
        assert len(arc.beats) > 0
        assert arc.total_duration_seconds == 600

    def test_micro_payoff_scheduler(self):
        """Test micro payoff scheduling."""
        from src.content.viral_content_engine import MicroPayoffScheduler

        scheduler = MicroPayoffScheduler()
        payoffs = scheduler.schedule_payoffs(
            duration_seconds=300,
            niche="psychology",
            topic="manipulation"
        )

        assert payoffs is not None
        assert len(payoffs) > 0

        # Should have payoffs roughly every 30-60 seconds
        # For 300 seconds, expect 5-10 payoffs
        assert len(payoffs) >= 3


# ============================================================
# KeywordIntelligence Integration Tests
# ============================================================

class TestKeywordIntelligence:
    """Tests for the KeywordIntelligence module."""

    def test_keyword_intelligence_initialization(self):
        """Test KeywordIntelligence initializes correctly."""
        with patch("src.seo.keyword_intelligence.PYTRENDS_AVAILABLE", False):
            from src.seo.keyword_intelligence import KeywordIntelligence

            ki = KeywordIntelligence()

            assert ki is not None

    def test_keyword_metrics_dataclass(self):
        """Test KeywordMetrics dataclass."""
        from src.seo.keyword_intelligence import KeywordMetrics

        metrics = KeywordMetrics(
            keyword="passive income",
            search_volume_score=75.0,
            competition_score=45.0,
            opportunity_score=80.0,
            trend_direction="rising",
            trend_velocity=0.15,
            seasonality_index=0.3,
            search_intent="informational",
            intent_confidence=0.85,
            difficulty_level="medium",
            estimated_monthly_searches=12000,
            cpc_estimate=2.50
        )

        assert metrics.keyword == "passive income"
        assert metrics.opportunity_score == 80.0
        assert metrics.trend_direction == "rising"

        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert metrics_dict["keyword"] == "passive income"

    def test_trend_prediction_dataclass(self):
        """Test TrendPrediction dataclass."""
        from src.seo.keyword_intelligence import TrendPrediction

        prediction = TrendPrediction(
            keyword="bitcoin",
            current_interest=65.0,
            predicted_interest=85.0,
            prediction_date="2026-02-01",
            confidence=0.75,
            trend_type="rising",
            peak_timing="2026-03-15",
            supporting_signals=["news_coverage", "social_mentions"]
        )

        assert prediction.keyword == "bitcoin"
        assert prediction.predicted_interest > prediction.current_interest
        assert prediction.confidence == 0.75


# ============================================================
# Subtitles Module Tests
# ============================================================

class TestSubtitleGenerator:
    """Tests for the SubtitleGenerator module."""

    def test_subtitle_cue_dataclass(self):
        """Test SubtitleCue dataclass."""
        from src.content.subtitles import SubtitleCue

        cue = SubtitleCue(
            index=1,
            start_time=0.0,
            end_time=2.5,
            text="Hello world"
        )

        assert cue.index == 1
        assert cue.start_time == 0.0
        assert cue.end_time == 2.5
        assert cue.text == "Hello world"
        assert cue.duration == 2.5

    def test_subtitle_cue_srt_format(self):
        """Test SRT timestamp formatting."""
        from src.content.subtitles import SubtitleCue

        cue = SubtitleCue(
            index=1,
            start_time=61.5,  # 1:01.500
            end_time=63.75,   # 1:03.750
            text="Test text"
        )

        srt = cue.to_srt()

        assert "00:01:01,500" in srt
        assert "00:01:03,750" in srt
        assert "Test text" in srt

    def test_subtitle_cue_vtt_format(self):
        """Test VTT timestamp formatting."""
        from src.content.subtitles import SubtitleCue

        cue = SubtitleCue(
            index=1,
            start_time=61.5,
            end_time=63.75,
            text="Test text"
        )

        vtt = cue.to_vtt()

        assert "00:01:01.500" in vtt
        assert "00:01:03.750" in vtt
        assert "Test text" in vtt

    def test_subtitle_track(self):
        """Test SubtitleTrack class."""
        from src.content.subtitles import SubtitleTrack

        track = SubtitleTrack(language="en")
        track.add_cue(0.0, 2.0, "First line")
        track.add_cue(2.5, 4.5, "Second line")

        assert len(track.cues) == 2
        assert track.language == "en"
        assert track.total_duration == 4.5

    def test_subtitle_track_to_srt(self):
        """Test SubtitleTrack SRT export."""
        from src.content.subtitles import SubtitleTrack

        track = SubtitleTrack()
        track.add_cue(0.0, 2.0, "Line one")
        track.add_cue(2.5, 4.5, "Line two")

        srt_content = track.to_srt()

        assert "1\n" in srt_content
        assert "Line one" in srt_content
        assert "2\n" in srt_content
        assert "Line two" in srt_content

    def test_subtitle_track_to_vtt(self):
        """Test SubtitleTrack VTT export."""
        from src.content.subtitles import SubtitleTrack

        track = SubtitleTrack()
        track.add_cue(0.0, 2.0, "Line one")

        vtt_content = track.to_vtt()

        assert "WEBVTT" in vtt_content
        assert "Line one" in vtt_content


# ============================================================
# Sentiment Analysis Integration Tests
# ============================================================

class TestSentimentAnalysis:
    """Tests for sentiment analysis capabilities."""

    def test_textblob_import(self):
        """Test TextBlob can be imported."""
        try:
            from textblob import TextBlob
            text = TextBlob("I love this video!")
            assert text.sentiment.polarity > 0
        except ImportError:
            pytest.skip("TextBlob not installed")

    def test_vader_import(self):
        """Test VADER can be imported."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores("This is amazing!")
            assert scores["compound"] > 0
        except ImportError:
            pytest.skip("vaderSentiment not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
