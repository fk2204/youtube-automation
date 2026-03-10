"""Tests for src/seo/metadata_optimizer.py"""

import pytest
from src.seo.metadata_optimizer import MetadataOptimizer, OptimizedMetadata


@pytest.fixture
def optimizer():
    return MetadataOptimizer()


class TestMetadataOptimizerInit:
    def test_init(self, optimizer):
        assert optimizer is not None
        assert isinstance(optimizer, MetadataOptimizer)


class TestTitleOptimization:
    def test_optimize_title_basic(self, optimizer):
        result = optimizer.optimize_title('python tutorial', ['python', 'tutorial'])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_optimize_title_empty(self, optimizer):
        result = optimizer.optimize_title('', ['python'])
        assert isinstance(result, str)

    @pytest.mark.parametrize('title', [
        'How to Learn Python',
        'Python Programming Guide',
        'Complete Python Tutorial',
    ])
    def test_optimize_various_titles(self, optimizer, title):
        result = optimizer.optimize_title(title, ['python'])
        assert isinstance(result, str)


class TestTitleScoring:
    def test_score_title_basic(self, optimizer):
        score = optimizer.score_title('Python Tutorial', ['python'])
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_score_title_with_keywords(self, optimizer):
        score = optimizer.score_title(
            'How to Learn Python Fast',
            ['python', 'learn']
        )
        assert isinstance(score, float)
        assert 0 <= score <= 100


class TestDescriptionGeneration:
    def test_generate_description_basic(self, optimizer):
        description = optimizer.generate_description('python', ['python'])
        assert isinstance(description, str)

    def test_generate_description_with_keywords(self, optimizer):
        keywords = ['python', 'tutorial']
        description = optimizer.generate_description(
            'python', keywords
        )
        assert isinstance(description, str)


class TestTitleVariants:
    def test_generate_title_variants(self, optimizer):
        variants = optimizer.generate_title_variants('base title', ['title', 'base'])
        assert isinstance(variants, list)


class TestTagOptimization:
    def test_optimize_tags_basic(self, optimizer):
        tags = optimizer.optimize_tags(['python', 'tutorial'], 'python')
        assert isinstance(tags, list)

    @pytest.mark.parametrize('topic', [
        'python',
        'web development',
        'machine learning',
    ])
    def test_optimize_tags_various_topics(self, optimizer, topic):
        tags = optimizer.optimize_tags([topic], topic)
        assert isinstance(tags, list)


class TestMetadataCreation:
    def test_create_complete_metadata_minimal(self, optimizer):
        metadata = optimizer.create_complete_metadata(
            topic='testing',
            keywords=['test', 'video'],
            script='This is a test script.',
            video_duration=600
        )
        assert metadata is not None

    def test_create_complete_metadata_full(self, optimizer):
        metadata = optimizer.create_complete_metadata(
            topic='testing',
            keywords=['test', 'video', 'optimization'],
            script='This is a comprehensive test script with multiple sections.',
            video_duration=900
        )
        assert metadata is not None


class TestEdgeCases:
    def test_optimize_title_special_characters(self, optimizer):
        result = optimizer.optimize_title('Title with !@#$% characters', ['title'])
        assert isinstance(result, str)

    def test_score_title_very_long(self, optimizer):
        long_title = 'A' * 500
        score = optimizer.score_title(long_title)
        assert isinstance(score, float)

    def test_generate_tags_empty_input(self, optimizer):
        tags = optimizer.optimize_tags(['test'], 'test')
        assert isinstance(tags, list)
