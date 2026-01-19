# YouTube Automation AI Agent System

**23 specialized agents** for autonomous YouTube content creation, optimization, and management.

---

## Quick Reference

| # | Agent | File | Purpose | Token Cost |
|---|-------|------|---------|------------|
| **PRODUCTION** |
| 1 | ResearchAgent | `research_agent.py` | Topic discovery, trends, competitor analysis | LOW |
| 2 | ScriptAgent | `script_writer.py` | Script generation with hooks, open loops, CTAs | MEDIUM |
| 3 | ProductionAgent | `subagents.py` | Video creation (TTS, stock footage, Ken Burns) | ZERO |
| 4 | ThumbnailAgent | `thumbnail_agent.py` | AI thumbnails with A/B variants | LOW |
| 5 | UploadAgent | `subagents.py` | YouTube upload with scheduling | ZERO |
| **OPTIMIZATION** |
| 6 | SEOAgent | `seo_agent.py` | Title, description, tag optimization | LOW |
| 7 | SEOStrategist | `seo_strategist.py` | Keyword research, intent analysis, content planning | LOW |
| 8 | ABTestAgent | `ab_tester.py` | Title/thumbnail experiments with stats | ZERO |
| 9 | RetentionOptimizerAgent | `retention_optimizer_agent.py` | Open loops, micro-payoffs, hook strength | LOW |
| **ANALYTICS** |
| 10 | PerformanceAgent | `performance_monitor.py` | Views, CTR, retention tracking | ZERO |
| 11 | InsightAgent | `insight_agent.py` | AI pattern detection, recommendations | MEDIUM |
| 12 | RevenueAgent | `revenue_agent.py` | CPM tracking, revenue optimization | ZERO |
| 13 | ContentStrategyAgent | `content_strategy_agent.py` | Portfolio analysis, content calendar | LOW |
| **AUTOMATION** |
| 14 | SchedulerAgent | `scheduler_agent.py` | APScheduler wrapper, job management | ZERO |
| 15 | WorkflowAgent | `workflow_agent.py` | Pipeline state machine, coordination | ZERO |
| 16 | MonitorAgent | `monitor_agent.py` | System health, API limits, resources | ZERO |
| 17 | RecoveryAgent | `recovery_agent.py` | Error recovery, fallback providers | LOW |
| **QUALITY** |
| 18 | ValidatorAgent | `validator_agent.py` | Pre-publish checklist, quality gates | LOW |
| 19 | ComplianceAgent | `compliance_agent.py` | YouTube policy, copyright checks | LOW |
| 20 | ContentSafetyAgent | `content_safety_agent.py` | Misinformation, harmful content detection | MEDIUM |
| 21 | AudioQualityAgent | `audio_quality_agent.py` | LUFS normalization, clipping detection | ZERO |
| 22 | VideoQualityAgent | `video_quality_agent.py` | Resolution, bitrate, codec validation | ZERO |
| 23 | AccessibilityAgent | `accessibility_agent.py` | Caption generation, sync validation | LOW |

---

## Architecture

```
                    +------------------+
                    | AgentOrchestrator |
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
   Production           Optimization          Analytics
   +--------+           +-----------+         +---------+
   |Research|           |    SEO    |         |Perform- |
   | Script |           | A/B Test  |         | ance    |
   |Product-|           | Retention |         | Insight |
   |  ion   |           |           |         | Revenue |
   |Thumb-  |           |           |         |Strategy |
   | nail   |           |           |         |         |
   | Upload |           |           |         |         |
   +--------+           +-----------+         +---------+
        |                    |                    |
        +--------------------+--------------------+
                             |
                    +--------+---------+
                    |    Automation    |
                    +------------------+
                    | Scheduler        |
                    | Workflow         |
                    | Monitor          |
                    | Recovery         |
                    +--------+---------+
                             |
                    +--------+---------+
                    |     Quality      |
                    +------------------+
                    | Validator        |
                    | Compliance       |
                    | ContentSafety    |
                    | AudioQuality     |
                    | VideoQuality     |
                    | Accessibility    |
                    +------------------+
```

---

## CLI Commands

```bash
# Status
python run.py agents status              # All agents overview
python run.py agents list                # List available agents

# Research
python run.py agent research --niche finance --count 10

# SEO
python run.py agent seo-strategy research "passive income" --niche finance
python run.py agent seo-strategy ab-test "Title" --variants 5

# Quality
python run.py agent quality validate output/video.mp4
python run.py agent safety check script.txt
python run.py agent compliance check video.mp4

# Analytics
python run.py agent analytics insights --channel money_blueprints

# Automation
python run.py agent monitor health
python run.py agent workflow status
python run.py agent scheduler status

# Full Pipeline
python run.py video money_blueprints     # Create + upload video
python run.py daily-all                  # Run scheduler
```

---

## Core Infrastructure

### BaseAgent (`src/agents/base_agent.py`)

All agents inherit from BaseAgent:

```python
from src.agents.base_agent import BaseAgent, AgentResult, AgentMessage

class MyAgent(BaseAgent):
    def run(self, **kwargs) -> AgentResult:
        # Implementation
        return AgentResult(success=True, data=result)
```

**Key Classes:**
- `AgentResult` - Standard return type (success, data, tokens_used, cost, error)
- `AgentMessage` - Inter-agent communication (sender, recipient, message_type, payload)
- `AgentError`, `TokenBudgetExceeded`, `APIRateLimitError`, `QualityError`, `SafetyError`

### AgentOrchestrator (`src/agents/orchestrator.py`)

Coordinates agent workflows:

```python
from src.agents.orchestrator import get_orchestrator, VideoPipelines

orch = get_orchestrator()
orch.register_agent(ResearchAgent())
orch.register_agent(ScriptAgent())

# Run sequential pipeline
result = orch.run_pipeline(VideoPipelines.full_video_pipeline("finance", "money_blueprints"))

# Run parallel tasks
results = await orch.run_parallel([
    ("ThumbnailAgent", {"title": "My Video"}),
    ("SEOAgent", {"title": "My Video"}),
])
```

---

## Agent Details by Cluster

### Production Cluster

| Agent | Key Methods | Output |
|-------|-------------|--------|
| **ResearchAgent** | `find_topics(niche, count)`, `analyze_competitors()` | `ResearchResult` with scored ideas |
| **ScriptAgent** | `generate_script(topic, duration)` | `VideoScript` with hooks, sections, CTAs |
| **ProductionAgent** | `produce(script)` | Video file, audio, subtitles |
| **ThumbnailAgent** | `run(title, niche, variants)` | PNG files, A/B variants, predicted CTR |
| **UploadAgent** | `upload(video, metadata)` | YouTube URL, video_id |

### Optimization Cluster

| Agent | Key Methods | Output |
|-------|-------------|--------|
| **SEOAgent** | `optimize(title, description, tags)` | Optimized metadata, score |
| **SEOStrategist** | `research(keyword)`, `ab_test(title)` | Keywords, intent, variants |
| **ABTestAgent** | `create_test()`, `check_winner()` | Test results, statistical significance |
| **RetentionOptimizerAgent** | `run(script_text, duration)` | Hook strength, open loops, improvements |

### Analytics Cluster

| Agent | Key Methods | Output |
|-------|-------------|--------|
| **PerformanceAgent** | `analyze(video_id)` | Views, CTR, retention metrics |
| **InsightAgent** | `identify_patterns()`, `executive_summary()` | Patterns, recommendations |
| **RevenueAgent** | `track_revenue(channel)`, `analyze_cpm()` | Revenue data, CPM by niche |
| **ContentStrategyAgent** | `analyze_portfolio()`, `generate_calendar()` | Content plan, topic recommendations |

### Automation Cluster

| Agent | Key Methods | Output |
|-------|-------------|--------|
| **SchedulerAgent** | `run(action)`, `load_from_config()` | Job status, next_run |
| **WorkflowAgent** | `run(channel, topic)`, `run(action='status')` | Workflow state, progress |
| **MonitorAgent** | `run()` | Health status, API limits, alerts |
| **RecoveryAgent** | `run(operation, error, provider)` | Recovery strategy, fallback used |

### Quality Cluster

| Agent | Key Methods | Output |
|-------|-------------|--------|
| **ValidatorAgent** | `run(video, thumbnail, title)` | Pass/fail, score, blockers |
| **ComplianceAgent** | `run(script, title, description)` | Compliant status, issues, disclosures |
| **ContentSafetyAgent** | `run(script, niche)` | Safe status, risk level, concerns |
| **AudioQualityAgent** | `run(audio_file)` | LUFS level, peak, clipping detected |
| **VideoQualityAgent** | `run(video_file)` | Resolution, bitrate, frame rate |
| **AccessibilityAgent** | `run(video_file, script)` | Caption accuracy, accessibility score |

---

## Key Features Implemented

### Content Quality (Jan 2026)
- **Pre-publish quality gates** - Block upload if score < 75
- **RetentionOptimizer** - Auto-inject open loops (3+), micro-payoffs every 30-60s
- **NaturalPacingInjector** - Breath markers, transition pauses
- **GuaranteedHookGenerator** - Kinetic typography, never static titles
- **18 Ken Burns effects** - EffectTracker prevents repetition
- **DynamicSegmentController** - 15+ visual changes/min

### Audio Processing
- **6-band professional EQ** - 80Hz HP, 150Hz -2dB, 3kHz +3dB, 6.5kHz +2dB, 11kHz +1.5dB
- **Sidechain ducking** - Auto-duck music when voice present (-35dB threshold)
- **Broadcast compression** - Two-stage: -24dB/4:1 compressor + limiter
- **LUFS normalization** - -14 LUFS (YouTube target)

### AI Authenticity
- **NaturalVoiceVariation** - ±8% rate, ±5Hz pitch via SSML
- **Natural pacing** - Makes TTS sound more human

### Analytics & Testing
- **YouTube Analytics API** - Real performance data
- **A/B Testing** - Chi-squared statistical significance (95% confidence)
- **AI Thumbnail Generator** - Replicate API face generation
- **Performance Monitoring** - Alerts for low CTR/retention

### Automation
- **Parallel video rendering** - 3x faster batch processing
- **Stock footage caching** - Query-based, 80% time savings
- **Error recovery** - Exponential backoff, fallback providers

---

## Niche-Specific Configuration

| Niche | CPM Range | Optimal Length | Best Days |
|-------|-----------|----------------|-----------|
| Finance | $10-22 | 8-15 min | Mon, Wed, Fri |
| Psychology | $3-6 | 8-12 min | Tue, Thu, Sat |
| Storytelling | $4-15 | 12-30 min | Daily |

---

## Fallback Providers

| Service | Primary | Fallback 1 | Fallback 2 |
|---------|---------|------------|------------|
| AI | Groq (free) | Ollama (local) | Gemini |
| TTS | Fish Audio | Edge-TTS | - |
| Stock | Pexels | Pixabay | Coverr |

---

## File Locations

```
src/agents/
├── base_agent.py           # BaseAgent, AgentResult, AgentMessage
├── orchestrator.py         # AgentOrchestrator, VideoPipelines
├── research_agent.py       # ResearchAgent
├── seo_agent.py           # SEOAgent
├── seo_strategist.py      # SEOStrategist
├── quality_agent.py       # QualityAgent
├── analytics_agent.py     # AnalyticsAgent
├── thumbnail_agent.py     # ThumbnailAgent
├── retention_optimizer_agent.py  # RetentionOptimizerAgent
├── validator_agent.py     # ValidatorAgent
├── compliance_agent.py    # ComplianceAgent
├── content_safety_agent.py # ContentSafetyAgent
├── audio_quality_agent.py # AudioQualityAgent
├── video_quality_agent.py # VideoQualityAgent
├── accessibility_agent.py # AccessibilityAgent
├── revenue_agent.py       # RevenueAgent
├── content_strategy_agent.py # ContentStrategyAgent
├── insight_agent.py       # InsightAgent
├── workflow_agent.py      # WorkflowAgent
├── monitor_agent.py       # MonitorAgent
├── recovery_agent.py      # RecoveryAgent
├── scheduler_agent.py     # SchedulerAgent
├── subagents.py          # ProductionAgent, UploadAgent
└── crew.py               # CrewAI integration
```

---

## Usage Examples

### Create Video with Quality Gates

```python
from src.agents import (
    ResearchAgent, ValidatorAgent,
    ComplianceAgent, ContentSafetyAgent
)

# 1. Research topic
research = ResearchAgent()
ideas = research.run(niche="finance", count=5)

# 2. Generate and validate script
# ... script generation ...

# 3. Safety check
safety = ContentSafetyAgent()
safety_result = safety.run(script=script_text, niche="finance")
if not safety_result.data["safe"]:
    print(f"Safety concerns: {safety_result.data['concerns']}")

# 4. Compliance check
compliance = ComplianceAgent()
comp_result = compliance.run(script=script_text, title=title)
if not comp_result.data["compliant"]:
    print(f"Compliance issues: {comp_result.data['issues']}")

# 5. Final validation
validator = ValidatorAgent()
valid_result = validator.run(video_file=video_path)
if valid_result.data["passed"]:
    # Upload
    pass
```

### Monitor System Health

```python
from src.agents import MonitorAgent

monitor = MonitorAgent()
health = monitor.run()

print(f"Status: {health.status}")
print(f"Disk: {health.resource_usage['disk_percent']}%")
print(f"Alerts: {len(health.alerts)}")

for alert in health.alerts:
    print(f"  [{alert.severity}] {alert.message}")
```

### Optimize Retention

```python
from src.agents import RetentionOptimizerAgent

optimizer = RetentionOptimizerAgent()
result = optimizer.run(
    script_text="Your script here...",
    duration_seconds=600,
    niche="psychology"
)

print(f"Hook Strength: {result.data['hook_strength']:.0%}")
print(f"Open Loops: {result.data['open_loop_count']}")
print(f"Predicted Retention: {result.data['predicted_retention']:.0%}")
```

---

## Design Principles

1. **Single Responsibility** - Each agent does ONE thing well
2. **Token Efficiency** - Rule-based first, AI only when needed
3. **Fault Tolerance** - Retry with backoff, fallback providers
4. **Observability** - All operations logged and tracked

---

*Last updated: January 2026*
