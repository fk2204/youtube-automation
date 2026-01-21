# New Features Documentation

Comprehensive guide to all new features added in the January 2026 upgrade.

## Table of Contents
- [Whisper Captioning](#whisper-captioning)
- [Viral Content Engine](#viral-content-engine)
- [Free Keyword Research](#free-keyword-research)
- [Reddit Research](#reddit-research)
- [Token Optimization](#token-optimization)
- [Pro Video Engine](#pro-video-engine)
- [Smart Scheduler](#smart-scheduler)
- [Master Orchestrator](#master-orchestrator)
- [Success Tracker](#success-tracker)

---

## Whisper Captioning

Auto-generate accurate subtitles from audio using OpenAI Whisper.

### Installation

```bash
# Option 1: faster-whisper (recommended - 4x faster, lower memory)
pip install faster-whisper

# Option 2: openai-whisper (original)
pip install openai-whisper
```

### Configuration

In `config/config.yaml`:

```yaml
content:
  subtitles:
    enabled: true
    style: "regular"           # regular, shorts, minimal, cinematic
    shorts_style: "shorts"     # Style for YouTube Shorts
    use_transcription: true    # Use Whisper for accurate timing
```

### Usage Examples

```python
from src.content.subtitles import SubtitleGenerator

generator = SubtitleGenerator()

# Generate subtitles from audio file
track = generator.generate_subtitles_from_audio("audio.mp3")

# Generate from script text (fallback if Whisper unavailable)
track = generator.generate_subtitles_from_script(
    script_text="Your narration text here...",
    audio_duration=120.0  # seconds
)

# Burn subtitles into video
generator.burn_subtitles(
    video_path="input.mp4",
    subtitle_track=track,
    output_path="output.mp4",
    style="shorts"  # Large centered text for Shorts
)

# Export as SRT for YouTube upload
generator.create_srt_file(track, "subtitles.srt")

# Export as VTT
generator.create_vtt_file(track, "subtitles.vtt")
```

### API Reference

**SubtitleGenerator**
- `generate_subtitles_from_audio(audio_path, model="base")` - Transcribe audio with Whisper
- `generate_subtitles_from_script(script_text, audio_duration)` - Generate from text
- `burn_subtitles(video_path, track, output_path, style)` - Embed subtitles in video
- `create_srt_file(track, output_path)` - Export SRT format
- `create_vtt_file(track, output_path)` - Export VTT format

**SubtitleCue** (dataclass)
- `index: int` - Cue number
- `start_time: float` - Start in seconds
- `end_time: float` - End in seconds
- `text: str` - Subtitle text
- `words: List[Dict]` - Word-level timing (from Whisper)

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Whisper not found" | Install: `pip install faster-whisper` |
| Slow transcription | Use "tiny" or "base" model instead of "large" |
| Memory error | Use faster-whisper instead of openai-whisper |
| FFmpeg error | Ensure FFmpeg is installed and in PATH |

---

## Viral Content Engine

Generate viral hooks, emotional arcs, and retention elements.

### Installation

No additional dependencies required. Uses built-in templates.

### Configuration

In `config/config.yaml`:

```yaml
retention:
  enabled: true
  auto_open_loops: true
  open_loop_count: 3
  micro_payoff_interval: 60    # seconds
  pattern_interrupts: true
```

### Usage Examples

```python
from src.content.viral_content_engine import (
    ViralHookGenerator,
    EmotionalArcBuilder,
    CuriosityGapCreator,
    MicroPayoffScheduler,
    PatternInterruptLibrary,
    ViralContentEngine
)

# Generate viral hooks
hook_gen = ViralHookGenerator()
hook = hook_gen.generate_hook("passive income", niche="finance")
print(f"Hook: {hook.text}")
print(f"Type: {hook.hook_type.value}")
print(f"Est. retention boost: {hook.estimated_retention_boost:.0%}")

# Build emotional arc for video
arc_builder = EmotionalArcBuilder()
arc = arc_builder.build_arc(
    duration_seconds=600,
    niche="psychology",
    peaks=3  # Number of emotional peaks
)

# Create curiosity gaps (open loops)
gap_creator = CuriosityGapCreator()
gaps = gap_creator.create_gaps(
    topic="investment secrets",
    count=4
)

# Schedule micro-payoffs
payoff_scheduler = MicroPayoffScheduler()
payoffs = payoff_scheduler.schedule(
    duration_seconds=600,
    interval=60  # Every 60 seconds
)

# All-in-one: Generate complete viral elements
engine = ViralContentEngine(niche="finance")
elements = engine.generate_all_elements(
    topic="5 Money Mistakes",
    duration_seconds=600
)
print(f"Hook: {elements.hook.text}")
print(f"Open loops: {len(elements.curiosity_gaps)}")
print(f"Micro-payoffs: {len(elements.micro_payoffs)}")
```

### API Reference

**ViralHookGenerator**
- `generate_hook(topic, niche)` -> `ViralHook`
- `generate_multiple(topic, niche, count)` -> `List[ViralHook]`

**Hook Types (HookType enum)**
- PATTERN_INTERRUPT, BOLD_CLAIM, QUESTION_STACK
- STATS_SHOCK, STORY_LEAD, LOSS_AVERSION
- CURIOSITY_GAP, INSIDER_SECRET, COUNTDOWN, CONTROVERSY

**EmotionalArcBuilder**
- `build_arc(duration_seconds, niche, peaks)` -> `List[EmotionalBeat]`

**Emotion Types (EmotionType enum)**
- INTRIGUE, TENSION, SHOCK, RELIEF, CURIOSITY
- SATISFACTION, URGENCY, EMPATHY, EXCITEMENT, REVELATION

---

## Free Keyword Research

Research keywords without paid tools using Google Trends.

### Installation

```bash
pip install pytrends requests
```

### Configuration

No API keys required. Uses free Google Trends data.

### Usage Examples

```python
from src.seo.keyword_intelligence import KeywordIntelligence

ki = KeywordIntelligence()

# Full keyword analysis
result = ki.full_analysis("passive income", niche="finance")
print(f"Search Volume Score: {result.search_volume_score}")
print(f"Competition Score: {result.competition_score}")
print(f"Opportunity Score: {result.opportunity_score}")
print(f"Trend: {result.trend_direction}")
print(f"Difficulty: {result.difficulty_level}")

# Predict trending topics
trends = ki.predict_trends("cryptocurrency", days_ahead=14)
for trend in trends:
    print(f"{trend.keyword}: {trend.trend_type} (confidence: {trend.confidence:.0%})")

# Generate long-tail keywords
longtails = ki.generate_longtails("investing", count=50)
for kw in longtails[:10]:
    print(f"- {kw.keyword} (opportunity: {kw.opportunity_score})")

# Analyze competitor keywords
competitors = ki.analyze_competitors("finance tutorial", top_n=10)

# Detect seasonality
seasonality = ki.detect_seasonality("christmas gifts")
print(f"Seasonal strength: {seasonality.seasonality_index:.0%}")
```

### API Reference

**KeywordIntelligence**
- `full_analysis(keyword, niche)` -> `KeywordMetrics`
- `predict_trends(keyword, days_ahead)` -> `List[TrendPrediction]`
- `generate_longtails(seed, count)` -> `List[KeywordMetrics]`
- `analyze_competitors(keyword, top_n)` -> `List[CompetitorData]`
- `detect_seasonality(keyword)` -> `SeasonalityResult`
- `classify_intent(keyword)` -> `str` (informational, commercial, transactional, navigational)

**KeywordMetrics** (dataclass)
- `keyword: str`
- `search_volume_score: float` (0-100)
- `competition_score: float` (0-100, lower is better)
- `opportunity_score: float` (0-100)
- `trend_direction: str` (rising, stable, declining)
- `difficulty_level: str` (easy, medium, hard, very_hard)

---

## Reddit Research

Find trending video ideas from Reddit discussions.

### Installation

```bash
pip install praw
```

### Configuration

In `config/.env`:

```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=youtube-automation-bot/1.0
```

**Getting Reddit API Credentials:**
1. Go to https://reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" type
4. Set redirect URI to `http://localhost:8080`
5. Copy client ID and secret

### Usage Examples

```python
from src.research.reddit import RedditResearcher

researcher = RedditResearcher()

# Get video ideas from specific subreddits
ideas = researcher.get_video_ideas(
    subreddits=["learnprogramming", "webdev", "Python"],
    time_filter="week",  # hour, day, week, month, year, all
    limit=20
)

for idea in ideas:
    print(f"Topic: {idea.topic}")
    print(f"Source: r/{idea.subreddit}")
    print(f"Popularity: {idea.popularity_score}")
    print(f"Type: {idea.idea_type}")
    print(f"URL: {idea.source_url}")
    print("---")

# Use custom subreddits for niche
finance_ideas = researcher.get_video_ideas(
    subreddits=["personalfinance", "investing", "financialindependence"],
    time_filter="month",
    limit=50
)

# Filter by idea type
questions = [i for i in ideas if i.idea_type == "question"]
```

### API Reference

**RedditResearcher**
- `get_video_ideas(subreddits, time_filter, limit)` -> `List[VideoIdea]`
- `get_trending_topics(subreddit, limit)` -> `List[RedditPost]`
- `search_subreddit(subreddit, query, limit)` -> `List[RedditPost]`

**VideoIdea** (dataclass)
- `topic: str` - Cleaned topic title
- `source_title: str` - Original Reddit title
- `source_url: str` - Link to Reddit post
- `subreddit: str` - Source subreddit
- `popularity_score: int` - Combined upvotes/comments score
- `idea_type: str` - question, discussion, request, tutorial

**Default Subreddits:**
- Programming: learnprogramming, webdev, Python, javascript, reactjs
- General: explainlikeimfive, howto, techsupport

---

## Token Optimization

Reduce API costs by 50%+ through intelligent caching and routing.

### Installation

No additional dependencies. Built-in system.

### Configuration

In `config/config.yaml`:

```yaml
budget:
  daily_limit: 10.0
  warning_threshold: 0.8
  enforce: true
```

### Usage Examples

```python
from src.utils.token_optimizer import (
    TokenOptimizer,
    get_token_optimizer,
    SemanticCache
)

# Get global optimizer instance
optimizer = get_token_optimizer()

# Optimize prompt before sending (removes redundancy)
optimized = optimizer.optimize_prompt(
    prompt="Generate a detailed YouTube video script...",
    task_type="script"
)

# Smart provider routing (prefers free providers)
provider = optimizer.smart_route(
    task_type="title_generation",
    quality_required="medium"  # low, medium, high
)
print(f"Use provider: {provider}")  # e.g., "ollama" or "groq"

# Check cache before API call
cached_response = optimizer.get_cached(prompt)
if cached_response:
    print("Cache hit!")
else:
    response = call_api(prompt)
    optimizer.cache_response(prompt, response)

# Batch multiple requests into one
requests = [
    {"prompt": "Title 1...", "task": "title"},
    {"prompt": "Title 2...", "task": "title"},
    {"prompt": "Title 3...", "task": "title"},
]
results = optimizer.batch_process(requests)

# Use semantic cache for similar prompts
cache = SemanticCache(similarity_threshold=0.85)
cache.set("Generate title for Python tutorial", "5 Python Tricks...")

# This finds the cached response even though text is different
result = cache.get_similar("Create a title for Python programming video")
print(result)  # "5 Python Tricks..."
```

### API Reference

**TokenOptimizer**
- `optimize_prompt(prompt, task_type)` -> `str`
- `smart_route(task_type, quality_required)` -> `str`
- `get_cached(prompt)` -> `Optional[str]`
- `cache_response(prompt, response)` -> `None`
- `batch_process(requests)` -> `List[str]`
- `get_budget_status()` -> `Dict`

**SemanticCache**
- `set(key, value, ttl)` -> `None`
- `get_similar(query, threshold)` -> `Optional[str]`
- `get_exact(key)` -> `Optional[str]`

---

## Pro Video Engine

Create broadcast-quality videos with professional transitions and effects.

### Installation

```bash
pip install pillow moviepy
# FFmpeg must be installed and in PATH
```

### Usage Examples

```python
from src.content.pro_video_engine import ProVideoEngine

engine = ProVideoEngine()

# Apply cinematic transition between clips
final_clip = engine.transitions.apply(
    transition_type="film_burn",  # See transition types below
    clip1=first_clip,
    clip2=second_clip,
    duration=1.0
)

# Create kinetic text animation
text_clip = engine.text_animator.create(
    text="BREAKING NEWS",
    style="slide_zoom",
    duration=2.0,
    niche="finance"
)

# Sync visuals to audio beats
beat_markers = engine.beat_sync.analyze("audio.mp3")
synced_clips = engine.beat_sync.apply_visual_sync(
    clips=video_segments,
    beat_markers=beat_markers
)

# Apply film-look color grade
graded_clip = engine.color_grading.apply(
    preset="cinematic_teal_orange",
    clip=video_clip
)

# Create lower third graphic
lower_third = engine.motion_graphics.lower_third(
    name="John Smith",
    title="CEO, Tech Company"
)

# Auto-optimize pacing
paced_segments = engine.pacing.optimize(
    segments=raw_segments,
    target_duration=600,
    min_changes_per_minute=15
)
```

### Transition Types

| Category | Types |
|----------|-------|
| Classic | crossfade, fade_black, fade_white |
| Wipe | wipe_left, wipe_right, wipe_up, wipe_down, wipe_diagonal |
| Zoom | zoom_in, zoom_out, zoom_rotate |
| Creative | film_burn, glitch, shake, flash |
| Geometric | circle_reveal, rectangle_reveal, blinds_horizontal, blinds_vertical |
| Professional | push_left, push_right, slide_over, cube_rotate |

### Color Grade Presets

- `cinematic_teal_orange` - Film look with teal shadows/orange highlights
- `moody_dark` - High contrast, crushed blacks
- `bright_clean` - Light, airy, lifted shadows
- `vintage_film` - Faded, warm, film grain
- `neon_pop` - Saturated, vibrant colors

---

## Smart Scheduler

Intelligent scheduling with optimal upload times and content calendars.

### Installation

```bash
pip install apscheduler pyyaml
```

### Configuration

In `config/config.yaml`:

```yaml
scheduler:
  enabled: true
  pipeline:
    research: "06:00"
    script_generation: "08:00"
    video_creation: "10:00"
    upload: "14:00"

youtube:
  schedule:
    timezone: "UTC"
    upload_times:
      - "19:00"
      - "20:00"
      - "21:00"
```

### Usage Examples

```python
from src.scheduler.smart_scheduler import SmartScheduler

scheduler = SmartScheduler()

# Get optimal upload time for channel/niche
best_time = await scheduler.get_optimal_time(
    channel_id="money_blueprints",
    niche="finance"
)
print(f"Best time: {best_time.time_str} UTC")
print(f"Score: {best_time.score}")
print(f"Reason: {best_time.reason}")

# Plan content calendar for 4 weeks
calendar = await scheduler.plan_content_calendar(
    channel_id="money_blueprints",
    weeks=4,
    videos_per_week=3
)

for week in calendar:
    print(f"Week {week.week_number}:")
    for content in week.scheduled_content:
        print(f"  {content.scheduled_time}: {content.topic}")

# Batch schedule multiple videos
result = await scheduler.batch_schedule([
    {"channel_id": "money_blueprints", "topic": "passive income", "priority": "high"},
    {"channel_id": "mind_unlocked", "topic": "psychology tips", "priority": "normal"},
])

# Check for holiday conflicts
conflicts = scheduler.check_holiday_conflicts(
    scheduled_time=datetime(2026, 12, 25, 14, 0),
    region="US"
)
```

### API Reference

**SmartScheduler**
- `get_optimal_time(channel_id, niche)` -> `TimeSlot`
- `plan_content_calendar(channel_id, weeks, videos_per_week)` -> `ContentCalendar`
- `batch_schedule(content_list)` -> `BatchResult`
- `check_holiday_conflicts(scheduled_time, region)` -> `List[Conflict]`
- `get_competitor_schedule(niche)` -> `List[TimeSlot]`

---

## Master Orchestrator

Coordinate all 19 agents with parallel execution and load balancing.

### Usage Examples

```python
from src.agents.master_orchestrator import (
    MasterOrchestrator,
    get_master_orchestrator
)

# Get global orchestrator instance
orchestrator = get_master_orchestrator()

# Auto-register all available agents
orchestrator.auto_register_agents()

# Execute complete video workflow
result = await orchestrator.execute_workflow(
    workflow_type="full_video",
    channel_id="money_blueprints",
    topic="passive income"
)

if result.success:
    print(f"Video created: {result.video_path}")
    print(f"Duration: {result.duration_seconds}s")
    print(f"Agents used: {result.agents_used}")

# Run parallel agent tasks
results = await orchestrator.run_parallel([
    ("ResearchAgent", "find_topics", {"niche": "finance"}),
    ("AnalyticsAgent", "analyze_channel", {"channel": "money_blueprints"}),
    ("SEOAgent", "optimize_metadata", {"title": "...", "description": "..."}),
])

for agent_name, result in results.items():
    print(f"{agent_name}: {result.status}")

# Run sequential pipeline
pipeline_result = await orchestrator.run_pipeline([
    ("ResearchAgent", "find_topics", {"niche": "finance"}),
    ("ContentStrategy", "select_topic", {"topics": "$research_result"}),
    ("ScriptWriter", "write_script", {"topic": "$selected_topic"}),
])
```

### Available Agents (19 total)

| Agent | Capability | Description |
|-------|------------|-------------|
| ResearchAgent | RESEARCH | Find trending topics |
| AnalyticsAgent | ANALYTICS | Channel performance data |
| SEOAgent | SEO_OPTIMIZATION | Keyword/metadata optimization |
| ContentStrategyAgent | CONTENT_GENERATION | Topic selection strategy |
| ScriptWriter | CONTENT_GENERATION | Write video scripts |
| VideoQualityAgent | QUALITY_CHECK | Check video quality |
| AudioQualityAgent | QUALITY_CHECK | Check audio quality |
| ThumbnailAgent | THUMBNAILS | Generate thumbnails |
| SchedulerAgent | SCHEDULING | Schedule uploads |
| RetentionOptimizerAgent | CONTENT_GENERATION | Add retention elements |
| ComplianceAgent | COMPLIANCE | Check YouTube guidelines |
| RecoveryAgent | RECOVERY | Handle errors |
| ValidatorAgent | VALIDATION | Validate content |
| InsightAgent | ANALYTICS | Generate insights |
| MonitorAgent | MONITORING | Monitor performance |
| AccessibilityAgent | COMPLIANCE | Check accessibility |
| WorkflowAgent | WORKFLOW | Manage workflows |
| RevenueAgent | ANALYTICS | Track revenue |
| ContentSafetyAgent | COMPLIANCE | Check content safety |

---

## Success Tracker

Track KPIs, goals, and progress toward channel success.

### Usage Examples

```python
from src.analytics.success_tracker import get_success_tracker

tracker = get_success_tracker()

# Print dashboard
tracker.print_dashboard()

# Get channel metrics
metrics = tracker.get_channel_metrics("money_blueprints")
print(f"Subscribers: {metrics.subscribers}")
print(f"Total Views: {metrics.total_views}")
print(f"Avg CTR: {metrics.avg_ctr}%")
print(f"Engagement Score: {metrics.engagement_score}")

# Track goal progress
progress = tracker.get_goal_progress("1000_subscribers")
print(f"Progress: {progress.progress_percent}%")
print(f"On track: {progress.on_track}")

# Record daily snapshot
tracker.record_daily_snapshot(
    videos_created=3,
    shorts_created=6,
    api_cost=2.50
)

# Get trends over time
trends = tracker.get_trends(days=30)
for date, data in trends.items():
    print(f"{date}: {data.videos_created} videos, ${data.total_api_cost}")
```

### API Reference

**SuccessTracker**
- `print_dashboard()` - Display ASCII dashboard
- `get_channel_metrics(channel_id)` -> `ChannelMetrics`
- `get_goal_progress(goal_name)` -> `GoalProgress`
- `record_daily_snapshot(...)` -> `None`
- `get_trends(days)` -> `Dict[str, DailySnapshot]`
- `set_goal(name, target, deadline)` -> `None`

**ChannelMetrics** (dataclass)
- `channel_id: str`
- `subscribers: int`
- `total_views: int`
- `watch_time_hours: float`
- `avg_view_duration: float`
- `avg_ctr: float`
- `avg_retention: float`
- `revenue_estimate: float`
- `engagement_score: float` (property, 0-100)

---

## Troubleshooting

### Common Issues

| Issue | Module | Solution |
|-------|--------|----------|
| "Module not found" | All | Run `pip install -r requirements.txt` |
| API rate limit | Keyword Research | Wait 60s between requests |
| Whisper OOM | Subtitles | Use `faster-whisper` or smaller model |
| FFmpeg not found | Pro Video | Install FFmpeg and add to PATH |
| Reddit 401 | Reddit Research | Check client_id/secret in .env |
| Cache miss | Token Optimizer | Lower semantic threshold to 0.80 |

### Debug Mode

Enable detailed logging:

```python
from loguru import logger
logger.enable("src")
```

Or set in `.env`:

```bash
LOG_LEVEL=DEBUG
DEBUG=true
```
