# Quick Start Guide

Get your first video created in 5 minutes.

## Prerequisites

- Python 3.10+
- FFmpeg installed and in PATH
- At least one AI provider configured

## 5-Minute Setup

### Step 1: Install Dependencies (1 min)

```bash
cd youtube-automation
pip install -r requirements.txt
```

### Step 2: Configure AI Provider (1 min)

Copy the example environment file:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` and set ONE of these (start with free options):

**Option A: Ollama (FREE, local)**
```bash
AI_PROVIDER=ollama
# Then run: ollama pull llama3.2
```

**Option B: Groq (FREE, cloud)**
```bash
AI_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
# Get key from: https://console.groq.com/
```

### Step 3: Add Stock Footage API (1 min)

In `config/.env`, add at least one:

```bash
PEXELS_API_KEY=your_key_here    # https://www.pexels.com/api/
PIXABAY_API_KEY=your_key_here   # https://pixabay.com/api/docs/
```

### Step 4: Create Your First Video (2 min)

```bash
python src/main.py --niche "python tutorials"
```

This will:
1. Generate a video idea
2. Write a script using AI
3. Create voiceover with Edge-TTS (free)
4. Download stock footage
5. Assemble the video with subtitles
6. Output to `output/` folder

## First Video with New Features

### Create a Viral-Optimized Video

```python
from src.content.viral_content_engine import ViralContentEngine
from src.content.script_writer import ScriptWriter
from src.content.subtitles import SubtitleGenerator
from src.content.tts import TextToSpeech

# 1. Generate viral elements
engine = ViralContentEngine(niche="finance")
elements = engine.generate_all_elements(
    topic="5 Money Mistakes",
    duration_seconds=600
)

print(f"Hook: {elements.hook.text}")

# 2. Generate script with viral elements
writer = ScriptWriter(provider="groq")
script = writer.generate_script(
    topic="5 Money Mistakes",
    duration_minutes=10,
    hook=elements.hook.text,
    open_loops=elements.curiosity_gaps
)

# 3. Generate voiceover
tts = TextToSpeech()
await tts.generate_enhanced(
    text=writer.get_full_narration(script),
    output_file="output/voice.mp3",
    enhance=True  # Professional audio processing
)

# 4. Generate subtitles from audio
subtitles = SubtitleGenerator()
track = subtitles.generate_subtitles_from_audio("output/voice.mp3")
```

### Quick Video with CLI

```bash
# Create a regular video
python run.py video money_blueprints

# Create a YouTube Short
python run.py short money_blueprints

# Create videos for all channels
python run.py daily-all
```

## Common Workflows

### Workflow 1: Research to Video (Automated)

```bash
# Run the full automated pipeline
python src/main.py --niche "python tutorials" --upload
```

### Workflow 2: Manual Topic Selection

```python
from src.research.trends import TrendResearcher
from src.research.reddit import RedditResearcher

# Find trending topics
trends = TrendResearcher()
topics = trends.get_trending_topics("programming")

# Or find ideas from Reddit
reddit = RedditResearcher()
ideas = reddit.get_video_ideas(["learnprogramming", "Python"])

# Pick the best one
best_idea = max(ideas, key=lambda x: x.popularity_score)
print(f"Creating video about: {best_idea.topic}")

# Then generate the video
from src.agents.crew import YouTubeCrew
crew = YouTubeCrew()
crew.run_pipeline(niche="programming", topic=best_idea.topic)
```

### Workflow 3: Keyword-First Approach

```python
from src.seo.keyword_intelligence import KeywordIntelligence

# Research keywords first
ki = KeywordIntelligence()
result = ki.full_analysis("passive income")

if result.opportunity_score > 70:
    print(f"Good opportunity! Competition: {result.competition_score}")
    # Generate video for this keyword
else:
    # Try long-tail variations
    longtails = ki.generate_longtails("passive income", count=20)
    best = max(longtails, key=lambda x: x.opportunity_score)
    print(f"Better keyword: {best.keyword}")
```

### Workflow 4: Batch Video Creation

```python
from src.automation.unified_launcher import UnifiedLauncher, LaunchConfig

launcher = UnifiedLauncher(LaunchConfig(
    channels=["money_blueprints", "mind_unlocked"],
    parallel_videos=3,
    quality_threshold=75
))

# Create videos in parallel
result = await launcher.launch_batch(
    video_type="video",
    count_per_channel=2
)

print(f"Created {result.videos_created} videos")
print(f"Cost: ${result.cost:.2f}")
```

### Workflow 5: Schedule Content Calendar

```python
from src.scheduler.smart_scheduler import SmartScheduler

scheduler = SmartScheduler()

# Plan 4 weeks of content
calendar = await scheduler.plan_content_calendar(
    channel_id="money_blueprints",
    weeks=4,
    videos_per_week=3
)

# The scheduler will:
# - Pick optimal upload times
# - Avoid competitor uploads
# - Consider holidays
# - Balance content types
```

## Quick Commands Reference

```bash
# Video Creation
python run.py video <channel>          # Single video
python run.py short <channel>          # Single Short
python run.py daily-all                # All channels

# Research
python run.py trends <niche>           # View trends
python run.py keywords <topic>         # Keyword research

# Monitoring
python run.py status                   # Scheduler status
python run.py cost                     # API cost tracking
python run.py cache-stats              # Cache statistics

# Testing
python src/main.py --test-tts          # Test voice generation
python src/main.py --test-script       # Test script generation
```

## Next Steps

1. **Configure Channels**: Edit `config/channels.yaml` for your YouTube channels
2. **Set Up YouTube API**: Follow OAuth setup for automated uploads
3. **Enable Scheduler**: Run `python run.py daily-all` for automation
4. **Read Full Docs**: See `docs/NEW_FEATURES.md` for all features

## Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| "No module named..." | `pip install -r requirements.txt` |
| "FFmpeg not found" | Install FFmpeg, add to PATH |
| "API key invalid" | Check `.env` file format |
| "Rate limited" | Wait 60s, try again |
| "Out of memory" | Use smaller Whisper model |
