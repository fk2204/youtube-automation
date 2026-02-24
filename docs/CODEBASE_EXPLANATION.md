# YouTube Automation - Codebase Explanation

A comprehensive guide to how this codebase works, covering architecture, data flow, module relationships, and key design decisions.

**Last Updated:** February 2026

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Project Structure](#3-project-structure)
4. [The Video Creation Pipeline](#4-the-video-creation-pipeline)
5. [Entry Points](#5-entry-points)
6. [Module Deep Dives](#6-module-deep-dives)
7. [Agent System](#7-agent-system)
8. [Configuration System](#8-configuration-system)
9. [Database Layer](#9-database-layer)
10. [Cost Optimization](#10-cost-optimization)
11. [Key Design Patterns](#11-key-design-patterns)
12. [Module Dependency Map](#12-module-dependency-map)

---

## 1. What This Project Does

This is a **fully automated YouTube content factory** that handles every step from finding trending topics to uploading finished videos. It manages **3 YouTube channels** across different niches (finance, psychology, storytelling) and produces both regular videos (8-15 min) and YouTube Shorts.

The system runs autonomously on a daily schedule, producing 2 videos + 4 Shorts per channel on posting days, without human intervention.

### Core Pipeline

```
Research → Script → Audio → Video → Quality Check → Upload
```

### Scale

| Metric | Value |
|--------|-------|
| Python source files | 110+ |
| Total lines of code | ~115,000 |
| Channels managed | 3 |
| AI providers supported | 5 (Ollama, Groq, Gemini, Claude, OpenAI) |
| TTS providers | 2 (Edge-TTS free, Fish Audio premium) |
| Stock footage sources | 3 (Pexels, Pixabay, Coverr) |
| Specialized agents | 23 |

---

## 2. High-Level Architecture

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│                   CLI Layer                          │
│         run.py  /  src/main.py                      │
├─────────────────────────────────────────────────────┤
│               Orchestration Layer                    │
│   MasterOrchestrator / PipelineOrchestrator / Crew  │
├──────────┬──────────┬──────────┬────────────────────┤
│ Research │ Content  │ YouTube  │   Analytics/SEO    │
│          │          │          │                    │
│ trends   │ script   │ auth     │ success_tracker    │
│ reddit   │ tts      │ uploader │ keyword_intel      │
│ idea_gen │ video    │ multi_ch │ metadata_optimizer │
│          │ audio    │          │ ab_tester          │
│          │ thumbs   │          │                    │
├──────────┴──────────┴──────────┴────────────────────┤
│              Infrastructure Layer                    │
│  Database / Token Manager / Caching / Monitoring    │
└─────────────────────────────────────────────────────┘
```

### Key Principles

1. **Provider abstraction** - Every external service (AI, TTS, stock footage) is behind an interface, allowing easy swapping
2. **Graceful degradation** - If a service fails, the system falls back (e.g., Pexels fails → tries Pixabay → tries Coverr)
3. **Cost awareness** - Token usage is tracked, prompts are optimized, and cheap/free providers are preferred
4. **Async where it matters** - TTS generation, downloads, and batch operations use async/await
5. **Quality gates** - Content passes through validation before upload

---

## 3. Project Structure

```
youtube-automation/
├── config/
│   ├── config.yaml              # Main application settings
│   ├── channels.yaml            # Channel definitions (3 channels)
│   └── credentials/             # YouTube OAuth credentials (gitignored)
│
├── src/
│   ├── main.py                  # Original entry point
│   │
│   ├── research/                # Topic discovery
│   │   ├── trends.py            # Google Trends via pytrends (642 lines)
│   │   ├── reddit.py            # Reddit API for ideas
│   │   ├── reddit_researcher.py # Extended Reddit research
│   │   ├── idea_generator.py    # AI-scored idea ranking (1,305 lines)
│   │   └── trend_predictor.py   # Statistical trend forecasting (1,228 lines)
│   │
│   ├── content/                 # Content creation (largest module)
│   │   ├── script_writer.py     # Multi-provider AI scripts (3,602 lines)
│   │   ├── tts.py               # Edge-TTS voice generation (952 lines)
│   │   ├── tts_fish.py          # Fish Audio premium TTS
│   │   ├── tts_chatterbox.py    # Chatterbox TTS provider
│   │   ├── video_assembler.py   # MoviePy video composition (1,044 lines)
│   │   ├── video_ultra.py       # High-quality video generator (2,802 lines)
│   │   ├── video_fast.py        # Quick video generation
│   │   ├── video_shorts.py      # YouTube Shorts format (1,645 lines)
│   │   ├── video_hooks.py       # Visual hook effects (2,308 lines)
│   │   ├── video_pro.py         # Professional video pipeline
│   │   ├── audio_processor.py   # Audio normalization/enhancement (1,426 lines)
│   │   ├── audio_mastering.py   # Broadcast-quality mastering
│   │   ├── pro_video_engine.py  # Cinematic effects engine (2,362 lines)
│   │   ├── viral_content_engine.py  # Viral hooks/arcs/loops (1,454 lines)
│   │   ├── viral_hooks.py       # Hook generation library
│   │   ├── stock_footage.py     # Multi-source stock video (1,903 lines)
│   │   ├── stock_cache.py       # Stock footage caching
│   │   ├── stock_prefetcher.py  # Predictive stock downloads
│   │   ├── multi_stock.py       # Multi-provider stock fallback
│   │   ├── subtitles.py         # Whisper-based captioning
│   │   ├── thumbnail_generator.py # Automated thumbnails (1,352 lines)
│   │   ├── thumbnail_ai.py      # AI-powered thumbnails
│   │   ├── template_engine.py   # Video templates (1,615 lines)
│   │   ├── channel_branding.py  # Per-channel visual identity
│   │   ├── content_cache.py     # Generated content caching
│   │   ├── quality_checker.py   # Pre-upload quality validation
│   │   ├── script_validator.py  # Script content validation
│   │   ├── shorts_hybrid.py     # AI-enhanced Shorts (Pika Labs)
│   │   ├── sponsor_manager.py   # Sponsor integration
│   │   ├── parallel_downloader.py  # Concurrent asset downloads
│   │   ├── parallel_processor.py   # Multi-core processing
│   │   ├── ai_video_providers.py   # AI video generation routing
│   │   ├── ai_video_hailuo.py      # Hailuo AI video
│   │   ├── ai_video_runway.py      # Runway ML video
│   │   └── video_pika.py           # Pika Labs video
│   │
│   ├── youtube/                 # YouTube API integration
│   │   ├── auth.py              # OAuth2 credential management
│   │   ├── uploader.py          # Upload with SEO optimization (1,436 lines)
│   │   ├── analytics_api.py     # YouTube Analytics API
│   │   └── multi_channel.py     # Multi-channel management
│   │
│   ├── agents/                  # 23-agent orchestration system
│   │   ├── master_orchestrator.py   # Central coordinator (1,418 lines)
│   │   ├── agent_communication.py   # Pub/sub messaging (1,674 lines)
│   │   ├── crew.py                  # CrewAI integration
│   │   ├── orchestrator.py          # Simplified orchestrator
│   │   ├── base_agent.py            # Agent base class
│   │   ├── base_result.py           # Result dataclass
│   │   ├── research_agent.py        # Topic research
│   │   ├── seo_agent.py             # SEO optimization
│   │   ├── seo_strategist.py        # Advanced SEO (2,007 lines)
│   │   ├── quality_agent.py         # Quality checks
│   │   ├── analytics_agent.py       # Performance analytics
│   │   ├── workflow_agent.py        # Workflow management
│   │   ├── monitor_agent.py         # System health
│   │   ├── scheduler_agent.py       # Scheduling control
│   │   ├── content_safety_agent.py  # Content moderation
│   │   ├── compliance_agent.py      # Regulatory compliance
│   │   ├── content_strategy_agent.py # Strategy planning (1,166 lines)
│   │   ├── thumbnail_agent.py       # Thumbnail generation
│   │   ├── audio_quality_agent.py   # Audio validation
│   │   ├── video_quality_agent.py   # Video validation
│   │   ├── retention_optimizer_agent.py # Retention analysis
│   │   ├── recovery_agent.py        # Error recovery
│   │   ├── revenue_agent.py         # Revenue optimization
│   │   ├── insight_agent.py         # Data insights
│   │   ├── accessibility_agent.py   # Accessibility checks
│   │   ├── validator_agent.py       # Content validation (1,182 lines)
│   │   ├── subagents.py             # Subagent definitions
│   │   ├── run_agents.py            # Agent runner
│   │   └── post_all_channels.py     # Multi-channel posting
│   │
│   ├── automation/              # Pipeline execution
│   │   ├── pipeline_orchestrator.py # Parallel pipelines (2,067 lines)
│   │   ├── runner.py                # Automation runner (1,424 lines)
│   │   ├── batch.py                 # Batch processing
│   │   ├── parallel_pipeline.py     # Parallel execution
│   │   └── unified_launcher.py      # Unified CLI
│   │
│   ├── scheduler/               # Time-based scheduling
│   │   ├── smart_scheduler.py   # Intelligent scheduling (1,414 lines)
│   │   ├── daily_scheduler.py   # Daily job runner
│   │   └── content_calendar.py  # Content planning (1,178 lines)
│   │
│   ├── seo/                     # Search engine optimization
│   │   ├── keyword_intelligence.py  # Keyword research (2,282 lines)
│   │   ├── metadata_optimizer.py    # Title/description optimization
│   │   └── free_keyword_research.py # Zero-cost keyword tools
│   │
│   ├── analytics/               # Performance tracking
│   │   ├── success_tracker.py   # KPI dashboard
│   │   ├── comment_analyzer.py  # Comment sentiment analysis
│   │   └── feedback_loop.py     # Performance → content feedback
│   │
│   ├── monitoring/              # System health
│   │   ├── performance_monitor.py   # Video performance alerts
│   │   ├── performance_alerts.py    # Alert system (1,278 lines)
│   │   └── error_monitor.py         # Error tracking
│   │
│   ├── testing/                 # A/B testing
│   │   ├── ab_tester.py         # A/B test runner
│   │   └── ab_testing.py        # Statistical testing (1,846 lines)
│   │
│   ├── compliance/              # Regulatory
│   │   └── ai_disclosure.py     # AI-generated content disclosure
│   │
│   ├── social/                  # Cross-platform
│   │   ├── multi_platform.py    # Multi-platform posting (1,158 lines)
│   │   └── social_poster.py     # Social media integration
│   │
│   ├── database/                # Data persistence
│   │   ├── db.py                # SQLAlchemy setup
│   │   └── models.py            # Video/Upload/Generation models
│   │
│   ├── captions/                # Subtitles
│   │   └── whisper_generator.py # Whisper-based captioning
│   │
│   ├── templates/               # Prompt templates
│   │   └── efficient_prompts.py # Token-optimized prompts
│   │
│   └── utils/                   # Shared utilities
│       ├── token_optimizer.py   # 50% cost reduction (2,626 lines)
│       ├── token_manager.py     # API cost tracking
│       ├── best_practices.py    # Competitor-informed validation
│       ├── youtube_optimizer.py # YouTube algorithm optimization (1,203 lines)
│       ├── smart_provider_router.py # AI provider selection
│       ├── cleanup.py           # File/cache cleanup (1,333 lines)
│       ├── gpu_utils.py         # GPU/NVENC detection
│       ├── memory_monitor.py    # Memory usage tracking
│       ├── profiler.py          # Performance profiling
│       ├── query_cache.py       # Query result caching
│       ├── segment_cache.py     # Video segment caching
│       ├── text_similarity.py   # Semantic similarity
│       └── db_optimizer.py      # Database optimization
│
├── run.py                       # Modern CLI (37,395 lines)
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # AI assistant instructions
└── docs/                        # Documentation
```

---

## 4. The Video Creation Pipeline

This is the core flow that transforms a trending topic into a published YouTube video.

### Step 1: Research (Topic Discovery)

```python
# src/research/trends.py → src/research/idea_generator.py
```

**What happens:**
1. `TrendResearcher` queries Google Trends via `pytrends` for trending topics in the channel's niche
2. `RedditResearcher` (optional) scrapes relevant subreddits for popular discussion topics
3. `IdeaGenerator` combines trend data with AI scoring to rank ideas
4. Each idea gets a composite score (0-100) based on trend momentum, competition level, and engagement potential

**Fallback behavior:** If Google Trends API fails (rate limiting), the system uses a built-in database of 70+ fallback topics categorized by niche.

**Output:** A ranked list of `ScoredIdea` objects with topic, title suggestions, and scores.

### Step 2: Script Generation

```python
# src/content/script_writer.py
```

**What happens:**
1. `ScriptWriter` selects an AI provider (Ollama/Groq/Gemini/Claude/OpenAI) based on cost routing
2. A niche-specific prompt is built using topic templates and viral title patterns (80+ templates)
3. The AI generates a structured `VideoScript` with sections, chapters, and retention points
4. `RetentionOptimizer` post-processes the script to inject:
   - Open loops (min 3 per video) to keep viewers watching
   - Micro-payoffs every 30-60 seconds
   - Pattern interrupts to prevent drop-off
5. `NaturalPacingInjector` adds breath markers and dramatic pauses for natural TTS delivery
6. The script passes through `best_practices.py` validation against competitor analysis data

**Provider priority:** Free providers (Ollama, Groq) are used first. Paid providers (Claude, OpenAI) are used when quality requirements demand it.

**Output:** A `VideoScript` dataclass with title, sections, chapter markers, tags, description, and retention points.

### Step 3: Audio Generation (Text-to-Speech)

```python
# src/content/tts.py → src/content/audio_processor.py
```

**What happens:**
1. `TextToSpeech` converts the script narration to speech using Edge-TTS (free) or Fish Audio (premium)
2. `NaturalVoiceVariation` applies SSML markup for rate variation (±8%) and pitch variation (±5Hz) to avoid robotic-sounding output
3. Each channel has a configured voice (e.g., `en-US-GuyNeural` for finance, `en-US-JennyNeural` for psychology)
4. `AudioProcessor.enhance_voice_professional()` applies broadcast-quality processing:
   - FFT noise reduction (removes background noise)
   - Presence EQ boost at 2-4kHz for voice clarity
   - High-pass filter at 80Hz to remove rumble
   - De-essing to reduce harsh sibilants
   - Two-stage compression for consistent dynamics
   - LUFS normalization to -14 (YouTube's target)

**Output:** An enhanced MP3/WAV audio file at broadcast quality.

### Step 4: Video Assembly

```python
# src/content/video_assembler.py (or video_ultra.py for high quality)
```

**What happens:**
1. `VideoAssembler` or `VideoUltra` creates the visual layer:
   - Fetches stock footage from Pexels → Pixabay → Coverr (cascading fallback)
   - Applies Ken Burns effects (18 variations) for visual movement
   - Generates title cards and text overlays
   - Adds cinematic transitions between segments
2. Audio is mixed with optional background music (12-15% volume)
3. Subtitles are burned in using Whisper transcription for accurate timing
4. Thumbnail is auto-generated with niche-specific styling
5. FFmpeg encoding at 8 Mbps video / 256k audio with H.264 High Profile

**For YouTube Shorts:** `video_shorts.py` produces 1080x1920 vertical format with larger fonts, faster pacing, and loop-friendly endings.

**Output:** A finished MP4 video file (1920x1080 or 1080x1920 for Shorts).

### Step 5: Quality Check

```python
# src/content/quality_checker.py → src/utils/best_practices.py
```

**What happens:**
1. Title is validated against niche-specific patterns (length, power words, CTR triggers)
2. Hook is checked for engagement potential
3. Audio levels are verified against YouTube's LUFS targets
4. Video duration is checked against niche-optimal ranges
5. SEO metadata is validated (tags, description keywords)
6. A composite quality score (0-100) determines if the video passes

**Configurable:** `quality_gates.block_on_fail` controls whether a failing score prevents upload or just logs a warning.

### Step 6: Upload

```python
# src/youtube/auth.py → src/youtube/uploader.py
```

**What happens:**
1. `YouTubeAuth` handles OAuth2 authentication (credential refresh, pickle-based caching)
2. `YouTubeUploader` uploads the video with optimized metadata:
   - SEO-optimized title, description, and tags (niche-specific)
   - Category selection per niche (Education, Entertainment, etc.)
   - Thumbnail attachment
   - Chapter markers from the script
   - AI content disclosure (compliance)
3. Upload timing is optimized for audience timezone activity
4. Result is tracked in the SQLite database

**Output:** A live YouTube video URL.

---

## 5. Entry Points

### `run.py` (Primary CLI - 37,395 lines)

The main command-line interface supporting all operations:

```bash
python run.py video <channel>         # Create and upload a single video
python run.py short <channel>         # Create and upload a YouTube Short
python run.py batch <count>           # Batch video creation
python run.py daily-all               # Run full daily automation
python run.py cost                    # View token/API costs
python run.py status                  # Check scheduler status
python run.py cache-stats             # View stock footage cache
python run.py analytics <video_id>    # Video analytics
python run.py monitor                 # Monitor recent videos
python run.py ab-test <vid> <t1> <t2> # Start A/B test
python run.py agent <type> <command>  # Agent operations
```

Internally, `run.py` dispatches to 9+ specialized agent types and provides JSON or human-readable output formatting.

### `src/main.py` (Original Entry Point - 348 lines)

The original CLI with argparse-based commands:

```bash
python src/main.py --pipeline          # Run single video pipeline
python src/main.py --schedule          # Start daily scheduler
python src/main.py --test-tts          # Test TTS generation
python src/main.py --test-script       # Test script generation
python src/main.py --test-research     # Test trend research
```

Configures loguru logging with daily rotation and delegates to `YouTubeCrew` for orchestration.

---

## 6. Module Deep Dives

### Research Module (`src/research/`)

**`trends.py`** - Wraps the `pytrends` library for free Google Trends data. Includes a 1-second rate limiter between requests and a fallback database of 70+ topics across 8 categories. Uses `tenacity` for retry with exponential backoff.

**`idea_generator.py`** - The brain of topic selection. Contains 75+ viral topic templates split across 3 niches. Each template uses variable substitution (e.g., `{company}`, `{number}`) for infinite variations. Scoring combines trend momentum, competition analysis, and predicted engagement.

**`trend_predictor.py`** - Statistical trend forecasting using time series analysis. Classifies trends as rising/stable/declining with confidence scores. Helps catch topics before they peak.

### Content Module (`src/content/`)

This is the largest module (~30 files, ~30,000+ lines).

**`script_writer.py`** (3,602 lines) - The most complex file. Implements 5 AI provider classes behind an `AIProvider` abstract base class. The `ScriptWriter` class builds niche-specific prompts, generates structured scripts, and applies retention optimization. Key sub-components:
- `RetentionOptimizer`: Injects open loops, micro-payoffs, and pattern interrupts
- `NaturalPacingInjector`: Adds breath markers and emphasis for natural TTS
- 80+ viral title templates with power words for high CTR

**`tts.py`** (952 lines) - Async Edge-TTS integration with SSML-based voice variation. The `NaturalVoiceVariation` class prevents AI detection by varying speech rate and pitch at sentence boundaries. Supports word-level timing for subtitle synchronization.

**`video_assembler.py`** (1,044 lines) - MoviePy-based video composition with hardware acceleration detection (NVIDIA NVENC). Implements two-pass encoding, adaptive GOP sizing, and YouTube streaming optimizations (`-movflags +faststart`).

**`video_ultra.py`** (2,802 lines) - The premium video generator. Extends the basic assembler with cinematic transitions, dynamic text animations, beat-synced visuals, and color grading. Used when maximum quality is required.

**`audio_processor.py`** (1,426 lines) - Professional audio pipeline implementing FFT noise reduction, 6-band EQ, de-essing, sidechain ducking (auto-lowers music when voice is present), and two-stage broadcast compression.

**`pro_video_engine.py`** (2,362 lines) - Modular video effects system:
- `CinematicTransitions`: 20+ FFmpeg-based transitions (film burn, whip pan, etc.)
- `DynamicTextAnimations`: 16 kinetic typography styles
- `VisualBeatSync`: Syncs visual changes to audio beats
- `ColorGradingPresets`: 10 film-look presets (teal/orange, cyberpunk, etc.)
- `MotionGraphicsLibrary`: Lower thirds, callouts, subscribe buttons

**`viral_content_engine.py`** (1,454 lines) - Psychological engagement system:
- `ViralHookGenerator`: 10+ hook formulas per niche
- `EmotionalArcBuilder`: 4 story arc templates
- `CuriosityGapCreator`: 5 open-loop types
- `MicroPayoffScheduler`: Value deliveries every 45 seconds
- `PatternInterruptLibrary`: 20+ interrupts (verbal, visual, audio)

**`stock_footage.py`** (1,903 lines) - Multi-source stock video with cascading fallback (Pexels → Pixabay → Coverr). Includes query-based caching to avoid re-downloading, and `stock_prefetcher.py` for predictive downloads based on upcoming topics.

### YouTube Module (`src/youtube/`)

**`auth.py`** - OAuth2 flow with pickle-based credential persistence and automatic token refresh. Scopes include upload and full YouTube API access.

**`uploader.py`** (1,436 lines) - Upload engine with deep SEO integration:
- `YouTubeSEOOptimizer`: Niche-specific hashtags, trending tags, category optimization
- First-hour boosting strategies
- AI disclosure compliance tracking
- Chapter marker insertion from script data

**`analytics_api.py`** - YouTube Analytics API integration for real performance data (views, watch time, CTR, retention curves).

### Agent System (`src/agents/`)

See [Section 7](#7-agent-system) for the full agent architecture.

### Scheduler (`src/scheduler/`)

**`smart_scheduler.py`** (1,414 lines) - Intelligent upload timing:
- `OptimalTimeCalculator`: Niche-specific time windows with day-of-week modifiers
- `AudienceTimezoneAnalyzer`: Models 8 major audience regions
- `CompetitorAvoidance`: Tracks competitor posting patterns
- `HolidayAwareness`: Adjusts schedule around holidays

**`content_calendar.py`** (1,178 lines) - Weekly/monthly content planning with topic rotation to prevent duplicates.

**`daily_scheduler.py`** - APScheduler-based daily job runner that executes the pipeline at configured times.

### SEO Module (`src/seo/`)

**`keyword_intelligence.py`** (2,282 lines) - Free keyword research using:
- YouTube autocomplete API
- Google Trends time series
- Long-tail keyword generation (5 variation types)
- Search intent classification
- Seasonality detection with 12-month cycle analysis
- Composite opportunity scoring: 35% volume + 45% low competition + 20% trend direction

### Analytics & Monitoring (`src/analytics/`, `src/monitoring/`)

**`success_tracker.py`** - KPI dashboard tracking subscribers, views, CTR, retention, and revenue per channel. SQLite-backed with daily snapshots.

**`performance_alerts.py`** (1,278 lines) - Real-time alerts for low CTR, poor retention, or upload failures. Supports webhook, Slack, and Discord notifications.

**`ab_testing.py`** (1,846 lines) - Statistical A/B testing for thumbnails and titles with significance calculation.

---

## 7. Agent System

The project uses a 23-agent architecture coordinated by a central `MasterOrchestrator`.

### Architecture

```
┌──────────────────────────────────────────┐
│          MasterOrchestrator              │
│  (Registry, Router, Executor, Balancer)  │
├──────────────────────────────────────────┤
│           AgentCommunication             │
│  (MessageBus, EventEmitter, SharedState) │
├─────────┬─────────┬─────────┬────────────┤
│ Content │ Quality │ Growth  │ Operations │
│ Agents  │ Agents  │ Agents  │ Agents     │
└─────────┴─────────┴─────────┴────────────┘
```

### Agent Categories

**Content Agents:**
- `ResearchAgent` - Topic research and trend analysis
- `ContentStrategyAgent` - Content planning and strategy
- `ThumbnailAgent` - Thumbnail generation
- `RetentionOptimizerAgent` - Viewer retention optimization

**Quality Agents:**
- `QualityAgent` - Overall quality scoring
- `AudioQualityAgent` - Audio level validation
- `VideoQualityAgent` - Video encoding validation
- `ValidatorAgent` - Content validation (1,182 lines)
- `AccessibilityAgent` - Accessibility compliance

**Growth Agents:**
- `SEOAgent` / `SEOStrategist` - Search optimization
- `AnalyticsAgent` - Performance analytics
- `RevenueAgent` - Revenue optimization
- `InsightAgent` - Data-driven insights

**Operations Agents:**
- `MonitorAgent` - System health monitoring
- `SchedulerAgent` - Schedule management
- `RecoveryAgent` - Error recovery and retry
- `ComplianceAgent` - Regulatory compliance
- `ContentSafetyAgent` - Content moderation
- `WorkflowAgent` - Pipeline workflow management

### Communication

Agents communicate through `agent_communication.py`:
- **MessageBus**: Topic-based pub/sub with wildcard matching and dead letter queue
- **EventEmitter**: Broadcast events with one-time listener support
- **SharedState**: Thread-safe namespaced global state with change watchers
- **TaskQueue**: Priority queue with deadlines and retry logic
- **ResultCache**: TTL-based cache with LRU eviction

### Workflows

The `MasterOrchestrator` supports pre-defined workflows:
- `full_video`: Research → Script → Audio → Video → Quality → Upload
- `short_video`: Research → Script → Audio → Short → Upload
- `research_only`: Research → Score → Report
- `quality_check`: Validate → Score → Report
- `analytics`: Fetch → Analyze → Report

---

## 8. Configuration System

### `config/config.yaml` - Application Settings

Controls all system behavior:

| Section | Purpose |
|---------|---------|
| `app` | Name, version, debug mode, log level |
| `budget` | Daily API spend limits ($10/day default) |
| `research` | Ideas per run, min score threshold, categories, subreddits |
| `content.script` | Word count targets (1,200-2,500 words), style |
| `content.voice` | Default TTS voice, rate, pitch |
| `content.audio` | Voice enhancement, noise reduction, LUFS target |
| `content.video` | Resolution, FPS, codec |
| `content.subtitles` | Whisper captioning settings |
| `youtube` | Default privacy, category, language, schedule |
| `scheduler` | Pipeline step timing (research 6am, script 8am, video 10am, upload 2pm) |
| `database` | SQLite path |
| `quality_gates` | Pre-publish checklist, score thresholds |
| `retention` | Open loops, micro-payoffs, pattern interrupts |
| `visual` | Ken Burns variety, pacing targets, effect cycling |
| `audio_advanced` | Multi-band EQ, sidechain ducking, broadcast compression |
| `ai_authenticity` | Voice variation ranges, natural pacing |

### `config/channels.yaml` - Channel Definitions

Each channel defines:
- **Identity**: ID, name, credentials file
- **Settings**: Niche, target audience, voice, posting schedule, topic templates with variable substitution
- **Branding**: Intro/outro text, colors
- **SEO**: Default tags per niche
- **Shorts**: Schedule, AI clips (Pika Labs), topic variation

### Channel Overview

| Channel | Niche | Voice | Posting Days | CPM Range |
|---------|-------|-------|-------------|-----------|
| `money_blueprints` | Finance | en-US-GuyNeural (male) | Tue-Thu | $10-25 |
| `mind_unlocked` | Psychology | en-US-JennyNeural (female) | Tue-Thu | $5-12 |
| `untold_stories` | Storytelling | en-GB-RyanNeural (British male) | Daily | $4-15 |

### Environment Variables

```bash
AI_PROVIDER=ollama|groq|gemini|claude|openai
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
YOUTUBE_CLIENT_SECRETS_FILE=config/client_secret.json
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
PEXELS_API_KEY=...
PIXABAY_API_KEY=...
FISH_AUDIO_API_KEY=...
```

---

## 9. Database Layer

### Technology

SQLite via SQLAlchemy ORM, stored at `data/youtube_automation.db`.

### Schema

**`videos` table** - Core entity tracking generated videos:
- `id`, `title`, `topic`, `niche`, `channel_id`, `file_path`, `duration`, `created_at`
- Indexed on: `(channel_id, niche)`, `created_at`, `(channel_id, created_at)`

**`uploads` table** - YouTube upload attempts:
- `id`, `video_id` (FK), `youtube_url`, `youtube_id`, `privacy`, `uploaded_at`, `status`, `error_msg`
- Statuses: pending → uploading → processing → completed/failed

**`generations` table** - Pipeline step tracking:
- `id`, `video_id` (FK), `step`, `status`, `error_msg`, `started_at`, `completed_at`
- Steps: research → script → audio → video → upload
- Statuses: pending → in_progress → completed/failed

### Additional Tables (Module-Specific)

- `daily_metrics` (success_tracker) - Per-channel daily KPIs
- `goals` (success_tracker) - Milestone tracking with deadlines
- `token_usage` (success_tracker) - Per-operation API cost tracking
- `keyword_cache` (keyword_intelligence) - Keyword research cache with 30-day TTL
- `performance_history` (smart_scheduler) - Upload performance data
- `messages`, `state`, `tasks` (agent_communication) - Agent state persistence

---

## 10. Cost Optimization

### Token Optimizer (`src/utils/token_optimizer.py`)

Claims ~50% reduction in API costs through:

1. **Prompt Caching**: Exact match + semantic similarity (85% threshold) caching
2. **Prompt Compression**:
   - Whitespace normalization
   - Example condensation (keep 2, summarize rest)
   - Instruction deduplication
   - JSON minification
   - Verbose language reduction
3. **Smart Provider Routing**: Routes to cheapest provider that meets quality needs
   - Free: Ollama (local, $0)
   - Cheap: Groq ($0.05/$0.08 per 1M tokens)
   - Mid: Gemini ($0.075/$0.30)
   - Premium: Claude ($3/$15), OpenAI ($5/$15)
4. **Token Budgets**: Per-agent daily/hourly limits with enforcement
5. **Batch Processing**: Combines multiple small requests into single API calls
6. **Task-Specific Limits**: Title generation capped at 50 tokens, scripts at 4,000

### Token Manager (`src/utils/token_manager.py`)

Tracks cumulative API costs. View with `python run.py cost`.

### Budget Configuration

```yaml
budget:
  daily_limit: 10.0        # $10/day max
  warning_threshold: 0.8   # Warn at 80%
  enforce: true             # Hard stop at limit
```

---

## 11. Key Design Patterns

### 1. Provider Abstraction (Strategy Pattern)

All AI providers implement `AIProvider` ABC:

```python
class AIProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
```

Allows seamless switching: `get_provider("ollama")` → `get_provider("groq")` → `get_provider("claude")`

### 2. Cascading Fallback

Services degrade gracefully through fallback chains:
- **Stock footage**: Pexels → Pixabay → Coverr → generated backgrounds
- **AI providers**: Preferred → cheap alternative → local (Ollama)
- **Trends API**: Google Trends → fallback topic database
- **CrewAI**: Full agent system → simplified sequential pipeline

### 3. Singleton with Lazy Init

Global instances for shared services:
```python
_instance = None
def get_master_orchestrator():
    global _instance
    if _instance is None:
        _instance = MasterOrchestrator()
    return _instance
```

### 4. Async/Await for I/O

TTS generation, file downloads, and batch operations use asyncio:
```python
async def generate(self, text: str, output: str):
    communicate = edge_tts.Communicate(text, self.voice)
    await communicate.save(output)
```

### 5. Registry + Factory (Agent System)

Agents self-register capabilities; the router matches tasks to agents:
```python
registry.register("research", ResearchAgent, capabilities=["trends", "reddit", "ideas"])
agent = router.route(task_type="trends")  # Returns ResearchAgent
```

### 6. Decorator-Based Retry

Using `tenacity` for robust API calls:
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_api(self, prompt):
    ...
```

### 7. Template Variable Substitution

Topic templates use `{variable}` placeholders filled from configured pools:
```yaml
topics:
  - "how {company} makes money"
topic_variables:
  company: ["Apple", "Tesla", "Amazon"]
```

---

## 12. Module Dependency Map

```
run.py
  └── src/automation/runner.py
       └── src/agents/master_orchestrator.py
            ├── src/agents/crew.py
            │    ├── src/research/idea_generator.py
            │    │    ├── src/research/trends.py        (pytrends)
            │    │    └── src/research/reddit.py         (PRAW)
            │    ├── src/content/script_writer.py
            │    │    ├── AI Providers (ollama/groq/gemini/claude/openai)
            │    │    ├── src/content/viral_content_engine.py
            │    │    ├── src/utils/token_optimizer.py
            │    │    └── src/utils/best_practices.py
            │    ├── src/content/tts.py                   (edge-tts)
            │    │    └── src/content/audio_processor.py   (scipy/numpy)
            │    ├── src/content/video_assembler.py        (MoviePy/FFmpeg)
            │    │    ├── src/content/stock_footage.py     (Pexels/Pixabay)
            │    │    ├── src/content/pro_video_engine.py
            │    │    └── src/content/subtitles.py         (Whisper)
            │    └── src/youtube/uploader.py               (Google API)
            │         ├── src/youtube/auth.py              (OAuth2)
            │         └── src/seo/metadata_optimizer.py
            ├── src/agents/agent_communication.py
            └── src/automation/pipeline_orchestrator.py
                 ├── src/scheduler/smart_scheduler.py
                 ├── src/analytics/success_tracker.py
                 └── src/monitoring/performance_alerts.py
```

---

## Summary

This is a large-scale (~115K lines) Python automation system that fully automates YouTube content production across 3 channels. Its key strengths are:

1. **End-to-end automation** - From trend research to published video with zero manual steps
2. **Cost efficiency** - Free-tier-first approach with token optimization and caching
3. **Quality engineering** - Retention optimization, viral hooks, and broadcast audio processing
4. **Resilience** - Cascading fallbacks, retry logic, and graceful degradation at every layer
5. **Scalability** - 23-agent orchestration with parallel pipeline execution
6. **SEO intelligence** - Free keyword research, competitor analysis, and metadata optimization

The codebase is organized around a clear pipeline (research → content → upload) with supporting systems for scheduling, analytics, monitoring, and cost management.
