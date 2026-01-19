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
]
