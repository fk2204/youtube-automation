# YouTube Automation Agents
#
# This module exports all agent classes for easy importing:
#
#   from src.agents import ResearchAgent, QualityAgent, AnalyticsAgent
#   from src.agents import ThumbnailAgent, RetentionOptimizerAgent, ValidatorAgent
#   from src.agents import WorkflowAgent, MonitorAgent, RecoveryAgent, SchedulerAgent
#
# Or import all:
#
#   from src.agents import *
#

# Base Agent Infrastructure
from .base_agent import BaseAgent, AgentResult, AgentMessage, AgentError

# Core Agents
from .research_agent import ResearchAgent, ResearchResult
from .quality_agent import QualityAgent, QualityResult, QualityCheckItem
from .analytics_agent import AnalyticsAgent, AnalyticsResult, PerformanceMetrics

# Production Agents (NEW - 2026-01-19)
from .thumbnail_agent import ThumbnailAgent, ThumbnailResult
from .retention_optimizer_agent import RetentionOptimizerAgent, RetentionResult
from .validator_agent import ValidatorAgent, ValidationResult, ValidationCheck

# Automation Cluster Agents (NEW - 2026-01-19)
from .workflow_agent import (
    WorkflowAgent,
    WorkflowResult,
    WorkflowState,
    WorkflowStatus,
    WorkflowStep,
)
from .monitor_agent import (
    MonitorAgent,
    HealthResult,
    HealthAlert,
    APIStatus,
    ResourceUsage,
    HealthStatus,
)
from .recovery_agent import (
    RecoveryAgent,
    RecoveryResult,
    RecoveryStrategy,
    ErrorCategory,
    FailureRecord,
)
from .scheduler_agent import (
    SchedulerAgent,
    ScheduleResult,
    ScheduledJob,
    JobStatus,
    JobType,
)

# Quality Cluster Agents (NEW - 2026-01-19)
from .compliance_agent import ComplianceAgent, ComplianceResult
from .content_safety_agent import ContentSafetyAgent, SafetyResult, RiskLevel
from .audio_quality_agent import AudioQualityAgent, AudioQualityResult
from .video_quality_agent import VideoQualityAgent, VideoQualityResult
from .accessibility_agent import AccessibilityAgent, AccessibilityResult

# SEO Agents
from .seo_agent import SEOAgent, SEOResult
from .seo_strategist import (
    SEOStrategist,
    SEOStrategyResult,
    KeywordResearcher,
    KeywordData,
    SearchIntentAnalyzer,
    SearchIntent,
    CompetitorAnalyzer,
    CompetitorReport,
    CompetitorVideo,
    PerformancePredictor,
    CTRPrediction,
    RetentionPrediction,
    ABTestManager,
    TitleVariant,
    ContentCalendar,
    TopicSuggestion,
)

# CrewAI Integration
from .crew import YouTubeCrew


__all__ = [
    # Base Infrastructure
    "BaseAgent",
    "AgentResult",
    "AgentMessage",
    "AgentError",

    # Core Agents
    "ResearchAgent",
    "ResearchResult",
    "QualityAgent",
    "QualityResult",
    "QualityCheckItem",
    "AnalyticsAgent",
    "AnalyticsResult",
    "PerformanceMetrics",

    # Production Agents (NEW - 2026-01-19)
    "ThumbnailAgent",
    "ThumbnailResult",
    "RetentionOptimizerAgent",
    "RetentionResult",
    "ValidatorAgent",
    "ValidationResult",
    "ValidationCheck",

    # Automation Cluster Agents (NEW - 2026-01-19)
    "WorkflowAgent",
    "WorkflowResult",
    "WorkflowState",
    "WorkflowStatus",
    "WorkflowStep",
    "MonitorAgent",
    "HealthResult",
    "HealthAlert",
    "APIStatus",
    "ResourceUsage",
    "HealthStatus",
    "RecoveryAgent",
    "RecoveryResult",
    "RecoveryStrategy",
    "ErrorCategory",
    "FailureRecord",
    "SchedulerAgent",
    "ScheduleResult",
    "ScheduledJob",
    "JobStatus",
    "JobType",

    # Quality Cluster Agents (NEW - 2026-01-19)
    "ComplianceAgent",
    "ComplianceResult",
    "ContentSafetyAgent",
    "SafetyResult",
    "RiskLevel",
    "AudioQualityAgent",
    "AudioQualityResult",
    "VideoQualityAgent",
    "VideoQualityResult",
    "AccessibilityAgent",
    "AccessibilityResult",

    # SEO Agents
    "SEOAgent",
    "SEOResult",
    "SEOStrategist",
    "SEOStrategyResult",

    # SEO Components (for advanced usage)
    "KeywordResearcher",
    "KeywordData",
    "SearchIntentAnalyzer",
    "SearchIntent",
    "CompetitorAnalyzer",
    "CompetitorReport",
    "CompetitorVideo",
    "PerformancePredictor",
    "CTRPrediction",
    "RetentionPrediction",
    "ABTestManager",
    "TitleVariant",
    "ContentCalendar",
    "TopicSuggestion",

    # CrewAI
    "YouTubeCrew",
]


# Version info
__version__ = "2.2.0"  # Updated with Quality Cluster Agents
__author__ = "YouTube Automation Team"
