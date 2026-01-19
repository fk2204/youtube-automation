"""
SEO Intelligence Module

World-class SEO optimization for YouTube content discovery.

Components:
- KeywordIntelligence: Advanced keyword research and trend prediction
- MetadataOptimizer: A/B test-ready title, description, and tag optimization
"""

from .keyword_intelligence import (
    KeywordResearcher,
    TrendPredictor,
    CompetitorAnalyzer,
    SearchIntentClassifier,
    LongTailGenerator,
    SeasonalityDetector,
    KeywordIntelligence,
)

from .metadata_optimizer import (
    TitleOptimizer,
    DescriptionBuilder,
    TagGenerator,
    HashtagStrategy,
    EndScreenOptimizer,
    MetadataOptimizer,
)

__all__ = [
    # Keyword Intelligence
    "KeywordResearcher",
    "TrendPredictor",
    "CompetitorAnalyzer",
    "SearchIntentClassifier",
    "LongTailGenerator",
    "SeasonalityDetector",
    "KeywordIntelligence",
    # Metadata Optimization
    "TitleOptimizer",
    "DescriptionBuilder",
    "TagGenerator",
    "HashtagStrategy",
    "EndScreenOptimizer",
    "MetadataOptimizer",
]
