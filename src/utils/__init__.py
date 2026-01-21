# Utils module

from src.utils.best_practices import (
    validate_title,
    validate_hook,
    get_best_practices,
    suggest_improvements,
    pre_publish_checklist,
    get_niche_metrics,
    get_hook_for_niche,
    get_viral_title_templates,
    ValidationResult,
    PrePublishChecklist,
    ChecklistItem,
    NICHE_METRICS,
    VIRAL_TITLE_PATTERNS,
    HOOK_FORMULAS,
    POWER_WORDS,
    SEO_PATTERNS,
    RETENTION_BEST_PRACTICES,
    IMPACT_FORMULA,
)

from src.utils.segment_cache import (
    SegmentCache,
    SegmentEntry,
    SegmentCacheStats,
    print_segment_cache_stats,
)

from src.utils.db_optimizer import (
    DatabaseOptimizer,
    ConnectionPool,
    QueryCache,
    QueryStats,
    TableInfo,
    IndexRecommendation,
    optimize_all_databases,
    print_optimization_summary,
)

from src.utils.profiler import (
    Profiler,
    ProfileResult,
    AggregatedStats,
    TimingContext,
    timed,
    profile,
    profile_func,
    get_report,
    clear,
)

__all__ = [
    # Validation functions
    "validate_title",
    "validate_hook",
    "get_best_practices",
    "suggest_improvements",
    "pre_publish_checklist",
    "get_niche_metrics",
    "get_hook_for_niche",
    "get_viral_title_templates",
    # Data classes
    "ValidationResult",
    "PrePublishChecklist",
    "ChecklistItem",
    # Constants
    "NICHE_METRICS",
    "VIRAL_TITLE_PATTERNS",
    "HOOK_FORMULAS",
    "POWER_WORDS",
    "SEO_PATTERNS",
    "RETENTION_BEST_PRACTICES",
    "IMPACT_FORMULA",
    # Segment cache
    "SegmentCache",
    "SegmentEntry",
    "SegmentCacheStats",
    "print_segment_cache_stats",
    # Database optimization
    "DatabaseOptimizer",
    "ConnectionPool",
    "QueryCache",
    "QueryStats",
    "TableInfo",
    "IndexRecommendation",
    "optimize_all_databases",
    "print_optimization_summary",
    # Profiler
    "Profiler",
    "ProfileResult",
    "AggregatedStats",
    "TimingContext",
    "timed",
    "profile",
    "profile_func",
    "get_report",
    "clear",
]
