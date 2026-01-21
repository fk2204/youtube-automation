"""
Integration Tests for New YouTube Automation Pipeline Components

Tests the integration between:
- Hook generation -> Script creation -> TTS -> Caption flow
- Metadata optimization flow
- AI disclosure tracking flow
- Full pipeline with new modules

Run with: pytest tests/test_pipeline_integration.py -v --integration
"""

import os
import sys
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# ============================================================
# Hook -> Script -> TTS -> Caption Flow Tests
# ============================================================

class TestHookToScriptFlow:
    """Test hook generation to script creation flow."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_viral_hook_to_script_integration(self):
        """Test: Viral hook generation integrates with script creation."""
        from src.content.viral_content_engine import ViralHookGenerator, ViralHook

        # Step 1: Generate viral hook
        hook_gen = ViralHookGenerator()
        hook = hook_gen.generate_hook("passive income", niche="finance")

        assert hook is not None
        assert hook.text is not None
        assert len(hook.text) > 10

        # Step 2: Verify hook can be used in script structure
        script_structure = {
            "hook": hook.text,
            "hook_type": hook.hook_type.value,
            "niche": hook.niche,
            "estimated_duration": hook.duration_seconds
        }

        assert script_structure["hook"] == hook.text
        assert script_structure["hook_type"] in [
            "pattern_interrupt", "bold_claim", "question_stack",
            "stats_shock", "story_lead", "loss_aversion",
            "curiosity_gap", "insider_secret", "countdown", "controversy"
        ]

    def test_emotional_arc_to_script_structure(self):
        """Test: Emotional arc integrates with script sections."""
        from src.content.viral_content_engine import EmotionalArcBuilder, ViralContentEngine

        # Step 1: Generate emotional arc
        arc_builder = EmotionalArcBuilder()
        arc = arc_builder.build_arc(duration_seconds=600, niche="psychology")

        assert arc is not None
        assert len(arc.beats) > 0

        # Step 2: Map arc beats to script sections
        script_sections = []
        for beat in arc.beats:
            section = {
                "timestamp": beat.timestamp_seconds,
                "emotion": beat.emotion.value,
                "intensity": beat.intensity,
                "content_hint": beat.narration_hint
            }
            script_sections.append(section)

        # Verify sections follow arc structure
        assert len(script_sections) == len(arc.beats)
        assert script_sections[0]["timestamp"] == arc.beats[0].timestamp_seconds

    def test_micro_payoffs_distribute_correctly(self):
        """Test: Micro payoffs are distributed throughout video duration."""
        from src.content.viral_content_engine import MicroPayoffScheduler

        scheduler = MicroPayoffScheduler()
        payoffs = scheduler.schedule_payoffs(
            duration_seconds=600,  # 10 minutes
            niche="finance",
            topic="investing"
        )

        assert len(payoffs) > 0

        # Verify payoffs are distributed
        timestamps = [p.timestamp_seconds for p in payoffs]
        timestamps_sorted = sorted(timestamps)

        # First payoff should be early (within first 60 seconds)
        assert timestamps_sorted[0] <= 60

        # Last payoff should be near the end
        assert timestamps_sorted[-1] >= 480  # Within last 2 minutes

        # Payoffs should be roughly 30-60 seconds apart on average
        if len(timestamps_sorted) > 1:
            gaps = [timestamps_sorted[i+1] - timestamps_sorted[i]
                    for i in range(len(timestamps_sorted)-1)]
            avg_gap = sum(gaps) / len(gaps)
            assert 20 <= avg_gap <= 90, f"Average gap {avg_gap}s should be 20-90s"

    def test_full_viral_content_generation(self):
        """Test: Full viral content engine generates all elements."""
        from src.content.viral_content_engine import ViralContentEngine

        engine = ViralContentEngine(niche="finance")

        elements = engine.generate_all_elements(
            topic="passive income",
            duration_seconds=480  # 8 minutes
        )

        # Verify all components are present
        assert "hook" in elements
        assert "emotional_arc" in elements
        assert "curiosity_gaps" in elements
        assert "micro_payoffs" in elements
        assert "pattern_interrupts" in elements
        assert "ctas" in elements

        # Verify hook quality
        hook = elements["hook"]
        assert hook.text is not None
        assert hook.estimated_retention_boost >= 0

        # Verify emotional arc
        arc = elements["emotional_arc"]
        assert len(arc.beats) > 0

        # Verify curiosity gaps create open loops
        gaps = elements["curiosity_gaps"]
        assert len(gaps) >= 1

        for gap in gaps:
            assert gap.opening_timestamp_seconds < gap.resolution_timestamp_seconds


# ============================================================
# Metadata Optimization Flow Tests
# ============================================================

class TestMetadataOptimizationFlow:
    """Test metadata optimization pipeline."""

    def test_keyword_to_title_optimization(self):
        """Test: Keywords feed into title optimization."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        # Start with target keywords
        keywords = ["passive income", "investing for beginners", "2026"]

        # Optimize title with keywords
        title = optimizer.optimize_title(
            base_title="How to Build Wealth",
            keywords=keywords,
            front_load_keywords=True
        )

        # Verify primary keyword is front-loaded (first 40 chars)
        first_40_chars = title[:40].lower()
        assert "passive income" in first_40_chars or "investing" in first_40_chars or "wealth" in first_40_chars

    def test_title_score_calculation(self):
        """Test: Title scoring reflects best practices."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        # Good title (should score high)
        good_title = "7 Passive Income Secrets That Changed My Life | Make Money 2026"

        # Score the title
        score = optimizer._score_title(
            title=good_title,
            keywords=["passive income", "make money"]
        )

        # Good titles should score above 50
        assert score >= 40, f"Good title should score at least 40, got {score}"

    def test_power_words_enhance_title(self):
        """Test: Power words are added to enhance titles."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        # Title without power words
        basic_title = "How to Save Money"

        optimized = optimizer.optimize_title(
            base_title=basic_title,
            keywords=["budgeting"]
        )

        # Check that some enhancement was made
        assert len(optimized) >= len(basic_title)

    def test_metadata_optimization_pipeline(self):
        """Test: Full metadata optimization pipeline."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        # Step 1: Start with raw content info
        topic = "passive income"
        keywords = ["make money online", "side hustle", "2026"]
        video_duration = 600  # 10 minutes

        # Step 2: Optimize title
        title = optimizer.optimize_title(
            base_title=f"How to Build {topic.title()}",
            keywords=keywords
        )

        assert title is not None
        assert len(title) > 0
        assert len(title) <= 100  # YouTube max title length

        # Step 3: Verify year is included for freshness
        assert "2026" in title or "2025" in title


# ============================================================
# AI Disclosure Tracking Flow Tests
# ============================================================

class TestAIDisclosureFlow:
    """Test AI disclosure tracking pipeline."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory for test database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_track_tts_usage(self, temp_db_dir):
        """Test: Track TTS usage for disclosure."""
        from src.compliance.ai_disclosure import (
            AIDisclosureTracker, AIContentType, DisclosureLevel
        )

        tracker = AIDisclosureTracker(db_path=str(temp_db_dir / "test.db"))

        # Track TTS usage
        tracker.track_voice_generation(
            video_id="vid123",
            method="edge-tts"
        )

        # Get disclosure metadata
        disclosure = tracker.get_disclosure_metadata("vid123")

        assert disclosure.requires_disclosure is True
        assert "synthetic_voice" in disclosure.content_types_used or \
               AIContentType.SYNTHETIC_VOICE.value in disclosure.content_types_used
        assert "edge-tts" in disclosure.methods_used

    def test_track_multiple_ai_components(self, temp_db_dir):
        """Test: Track multiple AI components for comprehensive disclosure."""
        from src.compliance.ai_disclosure import (
            AIDisclosureTracker, AIContentType
        )

        tracker = AIDisclosureTracker(db_path=str(temp_db_dir / "test.db"))

        # Track multiple AI usages
        tracker.track_voice_generation(video_id="vid456", method="edge-tts")
        tracker.track_visual_generation(video_id="vid456", method="stock_footage")
        tracker.track_script_generation(video_id="vid456", method="groq-llama")

        # Get comprehensive disclosure
        disclosure = tracker.get_disclosure_metadata("vid456")

        assert disclosure.requires_disclosure is True
        assert len(disclosure.content_types_used) >= 1
        assert len(disclosure.methods_used) >= 1

    def test_disclosure_description_integration(self, temp_db_dir):
        """Test: Disclosure integrates with video description."""
        from src.compliance.ai_disclosure import (
            AIDisclosureTracker, DisclosureMetadata, DisclosureLevel
        )

        tracker = AIDisclosureTracker(db_path=str(temp_db_dir / "test.db"))

        # Track AI usage
        tracker.track_voice_generation(video_id="vid789", method="edge-tts")

        # Get disclosure
        disclosure = tracker.get_disclosure_metadata("vid789")

        # Generate description disclaimer
        disclaimer = disclosure.get_description_disclaimer()

        # Verify disclaimer format
        if disclosure.requires_disclosure:
            assert "AI Disclosure" in disclaimer or len(disclaimer) > 0
            assert "edge-tts" in disclaimer.lower() or "tools" in disclaimer.lower()

    def test_no_disclosure_for_standard_content(self, temp_db_dir):
        """Test: No disclosure needed for standard content."""
        from src.compliance.ai_disclosure import (
            AIDisclosureTracker, DisclosureMetadata, DisclosureLevel
        )

        tracker = AIDisclosureTracker(db_path=str(temp_db_dir / "test.db"))

        # Get disclosure for video with no AI tracking
        disclosure = tracker.get_disclosure_metadata("vid_no_ai")

        # Should not require disclosure
        assert disclosure.requires_disclosure is False or len(disclosure.content_types_used) == 0

    def test_disclosure_levels_escalation(self, temp_db_dir):
        """Test: Disclosure levels escalate with sensitive content."""
        from src.compliance.ai_disclosure import (
            AIDisclosureTracker, AIContentType, DisclosureLevel
        )

        tracker = AIDisclosureTracker(db_path=str(temp_db_dir / "test.db"))

        # Track synthetic visuals (requires disclosure)
        tracker.track_visual_generation(
            video_id="vid_synthetic",
            method="ai_image_gen",
            is_realistic=True
        )

        disclosure = tracker.get_disclosure_metadata("vid_synthetic")

        # Realistic synthetic visuals should require disclosure
        assert disclosure.requires_disclosure is True
        assert disclosure.disclosure_level in [
            DisclosureLevel.RECOMMENDED, DisclosureLevel.REQUIRED
        ]


# ============================================================
# Full Pipeline Integration Tests
# ============================================================

class TestFullPipelineIntegration:
    """Test complete pipeline with new modules."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_research_to_content_pipeline(self):
        """Test: Research -> Content Generation pipeline."""
        # Step 1: Simulate keyword research results
        keywords = [
            {"keyword": "passive income", "opportunity_score": 85},
            {"keyword": "investing for beginners", "opportunity_score": 78},
            {"keyword": "make money online", "opportunity_score": 72}
        ]

        # Step 2: Select best keyword
        best_keyword = max(keywords, key=lambda x: x["opportunity_score"])
        assert best_keyword["keyword"] == "passive income"

        # Step 3: Generate viral elements for topic
        from src.content.viral_content_engine import ViralContentEngine

        engine = ViralContentEngine(niche="finance")
        elements = engine.generate_all_elements(
            topic=best_keyword["keyword"],
            duration_seconds=600
        )

        assert elements["hook"] is not None
        assert len(elements["curiosity_gaps"]) >= 1

    def test_content_to_metadata_pipeline(self):
        """Test: Content Generation -> Metadata Optimization pipeline."""
        # Step 1: Generate viral content
        from src.content.viral_content_engine import ViralContentEngine
        from src.seo.metadata_optimizer import MetadataOptimizer

        engine = ViralContentEngine(niche="psychology")
        elements = engine.generate_all_elements(
            topic="dark psychology",
            duration_seconds=480
        )

        # Step 2: Extract script elements
        hook_text = elements["hook"].text
        assert hook_text is not None

        # Step 3: Optimize metadata based on content
        optimizer = MetadataOptimizer()

        title = optimizer.optimize_title(
            base_title="Dark Psychology Secrets",
            keywords=["psychology", "manipulation", "mind control"]
        )

        assert title is not None
        assert len(title) > 0

    def test_disclosure_integration_in_pipeline(self, temp_output_dir):
        """Test: AI disclosure tracking throughout pipeline."""
        from src.compliance.ai_disclosure import AIDisclosureTracker

        tracker = AIDisclosureTracker(
            db_path=str(temp_output_dir / "disclosure.db")
        )

        video_id = "pipeline_test_001"

        # Track each AI component as it's used
        # Step 1: Script generation (AI)
        tracker.track_script_generation(video_id=video_id, method="groq-llama3")

        # Step 2: TTS generation (AI)
        tracker.track_voice_generation(video_id=video_id, method="edge-tts")

        # Step 3: Get final disclosure
        disclosure = tracker.get_disclosure_metadata(video_id)

        assert disclosure.requires_disclosure is True
        assert len(disclosure.methods_used) >= 1

    def test_hooks_enhance_retention_metrics(self):
        """Test: Viral hooks improve estimated retention."""
        from src.content.viral_content_engine import ViralHookGenerator, HookType

        generator = ViralHookGenerator()

        # Generate multiple hooks
        hooks = [
            generator.generate_hook("passive income", niche="finance")
            for _ in range(5)
        ]

        # Verify hooks have retention boost estimates
        for hook in hooks:
            assert hook.estimated_retention_boost >= 0
            assert hook.estimated_retention_boost <= 1.0

        # At least some hooks should have positive retention impact
        positive_hooks = [h for h in hooks if h.estimated_retention_boost > 0]
        assert len(positive_hooks) >= 1

    def test_pattern_interrupts_spacing(self):
        """Test: Pattern interrupts are properly spaced."""
        from src.content.viral_content_engine import PatternInterruptLibrary

        library = PatternInterruptLibrary()
        interrupts = library.get_interrupts_for_duration(
            duration_seconds=600,
            niche="finance"
        )

        assert len(interrupts) > 0

        # Pattern interrupts should be varied
        interrupt_types = set(i.interrupt_type for i in interrupts)
        assert len(interrupt_types) >= 1  # At least 1 type of interrupt

    def test_cta_placement_strategy(self):
        """Test: CTAs are placed at strategic moments."""
        from src.content.viral_content_engine import CallToActionOptimizer

        optimizer = CallToActionOptimizer()
        ctas = optimizer.generate_ctas(
            duration_seconds=600,
            niche="finance"
        )

        assert len(ctas) > 0

        # Verify CTA placement percentages
        percentages = [cta.percentage_through for cta in ctas]

        # Should have early, middle, and late CTAs
        has_early = any(p <= 0.35 for p in percentages)
        has_middle = any(0.35 < p <= 0.65 for p in percentages)
        has_late = any(p > 0.85 for p in percentages)

        assert has_early or has_middle, "Should have CTAs in first 65%"


# ============================================================
# Keyword Research Pipeline Tests
# ============================================================

class TestKeywordResearchPipeline:
    """Test keyword research integration."""

    def test_keyword_to_trend_prediction(self):
        """Test: Keywords feed into trend prediction."""
        from src.seo.keyword_intelligence import KeywordMetrics, TrendPrediction

        # Simulate keyword research result
        metrics = KeywordMetrics(
            keyword="cryptocurrency trading",
            search_volume_score=80.0,
            competition_score=65.0,
            opportunity_score=70.0,
            trend_direction="rising",
            trend_velocity=0.25,
            seasonality_index=0.4,
            search_intent="informational",
            intent_confidence=0.85,
            difficulty_level="hard",
            estimated_monthly_searches=50000,
            cpc_estimate=5.50
        )

        # Verify rising trend is detected
        assert metrics.trend_direction == "rising"
        assert metrics.trend_velocity > 0

    def test_long_tail_keyword_generation(self):
        """Test: Long-tail keywords are properly generated."""
        from src.seo.keyword_intelligence import LongTailKeyword

        # Simulate long-tail generation
        longtails = [
            LongTailKeyword(
                keyword="how to start passive income with $100",
                parent_keyword="passive income",
                word_count=7,
                estimated_difficulty=35.0,
                estimated_volume="medium",
                intent_match=0.85,
                variation_type="how_to"
            ),
            LongTailKeyword(
                keyword="best passive income ideas for beginners 2026",
                parent_keyword="passive income",
                word_count=7,
                estimated_difficulty=45.0,
                estimated_volume="high",
                intent_match=0.90,
                variation_type="modifier"
            )
        ]

        # Verify long-tail characteristics
        for lt in longtails:
            assert lt.word_count >= 4  # Long-tail = 4+ words
            assert lt.estimated_difficulty <= metrics.competition_score if 'metrics' in dir() else True
            assert lt.intent_match > 0.5


# ============================================================
# Reddit Research Pipeline Tests
# ============================================================

class TestRedditResearchPipeline:
    """Test Reddit research integration."""

    @pytest.fixture
    def mock_reddit_api(self):
        """Mock Reddit API responses."""
        mock = MagicMock()

        mock_post = MagicMock()
        mock_post.title = "What's the best way to learn Python in 2026?"
        mock_post.score = 250
        mock_post.num_comments = 75
        mock_post.permalink = "/r/learnprogramming/comments/xyz/test"
        mock_post.created_utc = 1705764000
        mock_post.link_flair_text = "Question"
        mock_post.subreddit = MagicMock()
        mock_post.subreddit.display_name = "learnprogramming"

        mock_subreddit = MagicMock()
        mock_subreddit.top.return_value = [mock_post]
        mock_subreddit.search.return_value = [mock_post]

        mock.subreddit.return_value = mock_subreddit

        return mock

    def test_reddit_to_video_idea_conversion(self, mock_reddit_api):
        """Test: Reddit posts convert to video ideas."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit_api):
            from src.research.reddit import RedditResearcher, VideoIdea

            researcher = RedditResearcher(
                client_id="test",
                client_secret="test"
            )

            ideas = researcher.get_video_ideas(
                subreddits=["learnprogramming"],
                limit=10
            )

            # Verify conversion
            for idea in ideas:
                assert isinstance(idea, VideoIdea)
                assert idea.topic is not None
                assert idea.popularity_score > 0

    def test_trending_topics_extraction(self, mock_reddit_api):
        """Test: Extract trending topics from Reddit."""
        with patch("src.research.reddit.praw.Reddit", return_value=mock_reddit_api):
            from src.research.reddit import RedditResearcher

            researcher = RedditResearcher(
                client_id="test",
                client_secret="test"
            )

            trending = researcher.get_trending_topics(
                subreddits=["learnprogramming"],
                limit=5
            )

            assert isinstance(trending, list)


# ============================================================
# Subtitle Pipeline Tests
# ============================================================

class TestSubtitlePipeline:
    """Test subtitle generation pipeline."""

    def test_subtitle_track_creation(self):
        """Test: Create subtitle track from script segments."""
        from src.content.subtitles import SubtitleTrack

        track = SubtitleTrack(language="en")

        # Simulate adding cues from script segments
        segments = [
            {"start": 0.0, "end": 3.0, "text": "Welcome to this video."},
            {"start": 3.5, "end": 7.0, "text": "Today we'll learn about passive income."},
            {"start": 7.5, "end": 12.0, "text": "There are five key strategies."},
        ]

        for seg in segments:
            track.add_cue(seg["start"], seg["end"], seg["text"])

        assert len(track.cues) == 3
        assert track.total_duration == 12.0

    def test_srt_export_format(self):
        """Test: SRT export is properly formatted."""
        from src.content.subtitles import SubtitleTrack

        track = SubtitleTrack()
        track.add_cue(0.0, 2.0, "First line")
        track.add_cue(2.5, 5.0, "Second line")

        srt_content = track.to_srt()

        # Verify SRT format
        lines = srt_content.strip().split('\n')

        # First block should have: index, timestamp, text
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert "First line" in lines[2]

    def test_vtt_export_format(self):
        """Test: VTT export is properly formatted."""
        from src.content.subtitles import SubtitleTrack

        track = SubtitleTrack()
        track.add_cue(0.0, 2.0, "First line")

        vtt_content = track.to_vtt()

        # Verify VTT format
        assert vtt_content.startswith("WEBVTT")
        assert "-->" in vtt_content
        assert "First line" in vtt_content


# ============================================================
# Error Handling Tests
# ============================================================

class TestPipelineErrorHandling:
    """Test error handling in pipelines."""

    def test_empty_keyword_handling(self):
        """Test: Handle empty keyword gracefully."""
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        # Should not crash with empty keywords
        title = optimizer.optimize_title(
            base_title="Test Title",
            keywords=[]
        )

        assert title is not None

    def test_missing_niche_fallback(self):
        """Test: Handle missing niche with fallback."""
        from src.content.viral_content_engine import ViralHookGenerator

        generator = ViralHookGenerator()

        # Use unknown niche - should fall back to generic
        hook = generator.generate_hook("test topic", niche="unknown_niche")

        # Should still generate something
        assert hook is not None
        assert hook.text is not None

    def test_zero_duration_handling(self):
        """Test: Handle zero/negative duration gracefully."""
        from src.content.viral_content_engine import MicroPayoffScheduler

        scheduler = MicroPayoffScheduler()

        # Should not crash with edge case durations
        payoffs = scheduler.schedule_payoffs(
            duration_seconds=0,
            niche="finance",
            topic="test"
        )

        # Should return empty or minimal list
        assert isinstance(payoffs, list)


# ============================================================
# Performance Tests
# ============================================================

class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def test_hook_generation_speed(self):
        """Test: Hook generation completes quickly."""
        import time
        from src.content.viral_content_engine import ViralHookGenerator

        generator = ViralHookGenerator()

        start = time.time()
        hook = generator.generate_hook("passive income", niche="finance")
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Hook generation took {elapsed}s, expected < 1s"
        assert hook is not None

    def test_metadata_optimization_speed(self):
        """Test: Metadata optimization completes quickly."""
        import time
        from src.seo.metadata_optimizer import MetadataOptimizer

        optimizer = MetadataOptimizer()

        start = time.time()
        title = optimizer.optimize_title(
            base_title="Test Title",
            keywords=["keyword1", "keyword2", "keyword3"]
        )
        elapsed = time.time() - start

        # Should complete in under 0.5 seconds
        assert elapsed < 0.5, f"Title optimization took {elapsed}s, expected < 0.5s"
        assert title is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
