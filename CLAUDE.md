# YouTube Automation Tool

An AI-powered system for automated YouTube video creation and publishing.

## Project Overview

This tool automates the entire YouTube content pipeline:
1. **Research** trending topics using Google Trends and Reddit
2. **Generate** scripts using AI (Ollama/Groq/Claude)
3. **Create** videos with Edge-TTS voiceover and MoviePy
4. **Upload** to multiple YouTube channels automatically

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Language | Python 3.10+ | Async support required |
| AI (Free) | Ollama, Groq, Gemini | Start with Ollama locally |
| AI (Paid) | Claude, OpenAI | Better quality scripts |
| TTS | Edge-TTS | FREE, 300+ neural voices |
| Video | MoviePy + FFmpeg | Video assembly |
| Research | pytrends, PRAW | Trends and Reddit APIs |
| YouTube | google-api-python-client | Official Google library |
| Scheduling | APScheduler | Daily automation |

## Project Structure

```
youtube-automation/
├── config/
│   ├── config.yaml          # Main settings
│   ├── channels.yaml        # YouTube channel configs
│   └── .env                  # API keys (gitignored)
├── src/
│   ├── research/            # Topic research
│   │   ├── trends.py        # Google Trends
│   │   ├── reddit.py        # Reddit API
│   │   └── idea_generator.py # AI idea generation
│   ├── content/             # Content creation
│   │   ├── script_writer.py # Multi-provider AI scripts
│   │   ├── tts.py           # Edge-TTS voice generation
│   │   └── video_assembler.py # MoviePy video creation
│   ├── youtube/             # YouTube API
│   │   ├── auth.py          # OAuth2 authentication
│   │   └── uploader.py      # Video upload
│   ├── agents/              # CrewAI agents
│   └── scheduler/           # APScheduler jobs
├── output/                  # Generated content
└── requirements.txt
```

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/main.py

# Test individual modules
python -m src.content.tts          # Test TTS
python -m src.content.script_writer # Test script generation
python -m src.research.trends       # Test trend research
```

## API Keys Required

| Service | Required | How to Get |
|---------|----------|------------|
| Ollama | No (local) | Install from ollama.ai |
| Groq | Free tier | console.groq.com |
| YouTube | Yes | Google Cloud Console |
| Reddit | Optional | reddit.com/prefs/apps |

## Code Style

- Use type hints for all functions
- Use `loguru` for logging (not print statements)
- Use `async/await` for I/O operations
- Use dataclasses for structured data
- Follow PEP 8 naming conventions

## Important Patterns

### AI Provider Abstraction
All AI providers implement `AIProvider` interface:
```python
from src.content.script_writer import get_provider

# Switch providers easily
ai = get_provider("ollama")  # Free, local
ai = get_provider("groq", api_key="...")  # Free, cloud
ai = get_provider("claude", api_key="...")  # Paid, best
```

### Edge-TTS Usage (Async)
```python
import asyncio
from src.content.tts import TextToSpeech

async def main():
    tts = TextToSpeech(default_voice="en-US-GuyNeural")
    await tts.generate("Hello world", "output.mp3")

asyncio.run(main())
```

### Video Assembly
```python
from src.content.video_assembler import VideoAssembler

assembler = VideoAssembler(resolution=(1920, 1080))
assembler.create_video_from_audio(
    audio_file="narration.mp3",
    output_file="video.mp4",
    title="My Tutorial"
)
```

## Environment Variables

```bash
# AI Provider (choose one)
AI_PROVIDER=ollama          # or groq, gemini, claude, openai
GROQ_API_KEY=gsk_...        # If using Groq
ANTHROPIC_API_KEY=sk-ant... # If using Claude

# YouTube (required for upload)
YOUTUBE_CLIENT_SECRETS_FILE=config/client_secret.json

# Reddit (optional)
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```

## Common Tasks

### Generate a Video Script
```python
from src.content.script_writer import ScriptWriter

writer = ScriptWriter(provider="ollama")
script = writer.generate_script(
    topic="How to learn Python",
    duration_minutes=10
)
print(script.title)
print(writer.get_full_narration(script))
```

### Research Video Ideas
```python
from src.research.idea_generator import IdeaGenerator

gen = IdeaGenerator(provider="ollama")
ideas = gen.generate_ideas(niche="python programming", count=5)
for idea in ideas:
    print(f"{idea.title} (Score: {idea.score})")
```

### Upload to YouTube
```python
from src.youtube.uploader import YouTubeUploader

uploader = YouTubeUploader()
result = uploader.upload_video(
    video_file="output/video.mp4",
    title="Python Tutorial",
    description="Learn Python basics...",
    privacy="unlisted"
)
print(result.video_url)
```

## Testing

- Always test with `privacy="unlisted"` first
- Use Ollama for development (free, no rate limits)
- Check `/cost` before using paid APIs

## Known Issues

- MoviePy TextClip requires ImageMagick on some systems
- Edge-TTS may be rate-limited with heavy use
- YouTube API has 10,000 quota units/day limit

## Claude Code Preferences

When working on this project, Claude should:

1. **Always use subagents** for complex tasks to save tokens:
   - Use `Task` tool with `run_in_background=true` for heavy work
   - Launch multiple agents in parallel when tasks are independent
   - Examples: testing, research, implementing features, code review

2. **Parallel execution** - Run independent tasks simultaneously:
   ```
   Task 1: Research/exploration  ─┐
   Task 2: Code implementation   ─┼─> All run in parallel
   Task 3: Testing               ─┘
   ```

3. **Use automation scripts** instead of manual commands:
   - `python run.py video <channel>` for video creation
   - `python run.py short <channel>` for Shorts
   - `python run.py daily-all` for full automation

4. **Track progress** with TodoWrite for multi-step tasks

5. **Commit frequently** with descriptive messages

6. **ALWAYS use best practices from competitor analysis** when creating content:
   - Reference `docs/COMPETITOR_ANALYSIS.md` for niche-specific insights
   - Use `src/utils/best_practices.py` for validation functions
   - Run pre-publish checklist before uploading videos
   - Follow viral title patterns, hook formulas, and retention techniques

## Recent Updates (2026-01-19)

### Content Quality Improvements
- [x] **Pre-publish quality gates** - Checklist enforcement before upload
- [x] **RetentionOptimizer** - Auto-inject open loops, micro-payoffs, pattern interrupts
- [x] **NaturalPacingInjector** - Breath markers, transition pauses, emphasis markers
- [x] **GuaranteedHookGenerator** - Kinetic typography, never static titles
- [x] **Expanded Ken Burns** - 18 effects (was 6) with EffectTracker
- [x] **DynamicSegmentController** - Adaptive pacing (15+ visual changes/min)
- [x] **Multi-band EQ** - 6-band professional audio processing
- [x] **Sidechain ducking** - Auto-duck music when voice present
- [x] **Broadcast compression** - Two-stage compression for broadcast quality
- [x] **NaturalVoiceVariation** - ±8% rate, ±5Hz pitch via SSML

### Analytics & Testing
- [x] **YouTube Analytics API** - Real performance data integration
- [x] **Performance Monitoring** - Alerts for low CTR/retention
- [x] **A/B Testing System** - Thumbnail/title variant testing
- [x] **AI Thumbnail Generator** - Replicate API face generation

### Performance Optimizations
- [x] **Parallel video rendering** - 3x faster batch processing
- [x] **Database indexes** - Faster analytics queries

### New Files Added
- `src/youtube/analytics_api.py` - YouTube Analytics API integration
- `src/monitoring/performance_monitor.py` - Real-time alerts
- `src/testing/ab_tester.py` - A/B testing system
- `src/content/thumbnail_ai.py` - AI-powered thumbnails

### New Commands
```bash
python run.py analytics <video_id>       # Video analytics
python run.py monitor                    # Monitor recent videos
python run.py ab-test <vid> <t1> <t2>   # Start A/B test
python run.py ab-check <test_id>        # Check test progress
python run.py thumbnail "Title" --ai    # AI thumbnail
```

---

## Recent Updates (2026-01-18)

### Video Quality Fixes
- [x] Video bitrate: 8 Mbps (was default ~2-4 Mbps)
- [x] Encoding preset: "slow" for quality (was "medium")
- [x] Audio bitrate: 256k (was 192k)
- [x] CRF 23 for consistent quality
- [x] test_mode: false (videos now upload!)

### New Features Added
- [x] **Fish Audio TTS** - Premium quality voices (better than ElevenLabs)
- [x] **Multi-source stock footage** - Pexels → Pixabay → Coverr fallback
- [x] **Audio normalization** - -14 LUFS (YouTube's target)
- [x] **Background music mixing** - 15% volume, proper levels
- [x] **Token manager** - Track API costs (`python run.py cost`)
- [x] **Best practices scripts** - Hooks, chapters, retention points
- [x] **Scheduler posting_days** - Respects channel config now
- [x] **Professional Voice Enhancement** - Broadcast-quality audio processing (2026-01-19)
- [x] **Stock footage caching** - Query-based caching saves 80% download time (2026-01-19)

### Files Added/Modified
- `src/content/tts_fish.py` - Fish Audio TTS provider
- `src/content/audio_processor.py` - Audio normalization/enhancement
- `src/content/video_ultra.py` - High-quality video generator
- `src/content/stock_cache.py` - Query-based stock footage caching system
- `src/utils/token_manager.py` - Cost tracking system
- `src/utils/best_practices.py` - Competitor analysis validation module
- `start_scheduler.bat` - Auto-start script
- `docs/RESEARCH_REPORTS.md` - Full research findings
- `docs/COMPETITOR_ANALYSIS.md` - Competitor best practices (Jan 2026)

## API Keys Configured

| Service | Status | Purpose |
|---------|--------|---------|
| Groq | ✓ Active | AI scripts (free) |
| Pexels | ✓ Active | Stock footage |
| Pixabay | ✓ Active | Stock footage backup |
| Fish Audio | ✓ Active | Premium TTS |
| YouTube OAuth | ✓ Active | 3 channels configured |

## Channels

| Channel | Posting Days | Times (UTC) |
|---------|--------------|-------------|
| money_blueprints | Mon, Wed, Fri | 15:00, 19:00, 21:00 |
| mind_unlocked | Tue, Thu, Sat | 16:00, 19:30, 21:30 |
| untold_stories | Every day | 17:00, 20:00, 22:00 |

## Quick Commands

```bash
python run.py daily-all      # Start full scheduler
python run.py video <channel> # Single video
python run.py short <channel> # Single Short
python run.py cost           # View token usage
python run.py status         # Scheduler status
python run.py cache-stats    # View stock footage cache stats
python run.py cache-stats --cleanup  # Clean old cache (>30 days)
```

## TTS Provider Selection

```python
from src.content.tts import get_tts_provider

# Edge-TTS (free, default)
tts = get_tts_provider("edge")

# Fish Audio (premium quality)
tts = get_tts_provider("fish")
```

## Video with Audio Enhancement

```python
from src.content.video_fast import FastVideoGenerator

gen = FastVideoGenerator()
gen.create_video(
    audio_file="voice.mp3",
    output_file="video.mp4",
    normalize_audio=True,      # -14 LUFS
    background_music="bg.mp3", # Optional
    music_volume=0.15          # 15%
)
```

## YouTube Best Practices (Implemented)

### Regular Videos
- Strong hook in first 5 seconds
- Micro-payoffs every 30-60 seconds
- Open loops (min 3 per video)
- Chapter markers auto-generated
- CTAs at 30%, 50%, 95%

### Shorts
- Hook in first 1-2 seconds
- Optimal length: 20-45 seconds
- Loop-friendly endings
- Pattern interrupts every 2-3 seconds

### Audio Levels
- Voice: -12dB
- Background music: -25 to -30dB
- Overall: -14 LUFS (YouTube target)

### Professional Voice Enhancement (NEW)
The `enhance_voice_professional()` method applies broadcast-quality processing:
1. **FFT Noise Reduction** - Removes background noise and hum (nf=-20dB)
2. **Presence EQ** - Boosts 2-4kHz for voice clarity (+3dB)
3. **High-Pass Filter** - Removes rumble below 80Hz
4. **De-esser** - Reduces harsh sibilant 's' sounds
5. **Compression** - Smooths dynamics (threshold=-18dB, ratio=3:1)
6. **Loudness Normalization** - Targets -14 LUFS with -1.5dB true peak

```python
# Using professional enhancement with TTS
from src.content.tts import TextToSpeech

tts = TextToSpeech()
await tts.generate_enhanced(
    text="Your narration here...",
    output_file="output/voice.mp3",
    enhance=True,              # Enable professional enhancement
    noise_reduction=True,      # FFT-based noise removal
    normalize_lufs=-14         # YouTube's target loudness
)

# Or use the AudioProcessor directly
from src.content.audio_processor import AudioProcessor

processor = AudioProcessor()
processor.enhance_voice_professional(
    input_file="raw_voice.mp3",
    output_file="enhanced_voice.mp3",
    noise_reduction=True,
    normalize_lufs=-14
)
```

## Competitor Analysis Best Practices

The `src/utils/best_practices.py` module provides validation against competitor research:

### Validate Content Before Publishing
```python
from src.utils.best_practices import (
    validate_title,
    validate_hook,
    get_best_practices,
    suggest_improvements,
    pre_publish_checklist
)

# Validate a title
result = validate_title("5 Money Mistakes Costing You $1000/Year", "finance")
print(f"Valid: {result.is_valid}, Score: {result.score:.0%}")
for suggestion in result.suggestions:
    print(f"  - {suggestion}")

# Validate a hook
result = validate_hook("Wall Street doesn't want you to know this...", "finance")
print(f"Valid: {result.is_valid}, Score: {result.score:.0%}")

# Get best practices for a niche
practices = get_best_practices("psychology")
print(f"CPM Range: ${practices['metrics']['cpm_range'][0]}-${practices['metrics']['cpm_range'][1]}")
print(f"Optimal Length: {practices['metrics']['optimal_video_length'][0]}-{practices['metrics']['optimal_video_length'][1]} min")
```

### Run Pre-Publish Checklist
```python
from src.content.script_writer import ScriptWriter

writer = ScriptWriter(provider="groq")
script = writer.generate_script("How AI Works", niche="psychology")

# Run comprehensive checklist
checklist = writer.run_pre_publish_checklist(script, "psychology")
print(f"Ready to publish: {checklist.ready_to_publish}")
print(f"Score: {checklist.overall_score:.0%}")

for item in checklist.items:
    status = "[PASS]" if item.passed else "[FAIL]"
    print(f"{status} {item.name}: {item.details}")
```

### Niche-Specific Metrics

| Niche | CPM Range | Optimal Length | Best Posting Days |
|-------|-----------|----------------|-------------------|
| Finance | $10-22 | 8-15 min | Mon, Wed, Fri |
| Psychology | $3-6 | 8-12 min | Tue, Thu, Sat |
| Storytelling | $4-15 | 12-30 min | Daily |

## Future Improvements

- [ ] Add screen recording simulation with Playwright
- [ ] Implement CrewAI multi-agent workflow
- [ ] Add video analytics tracking
- [ ] Add A/B testing for thumbnails
- [ ] Integrate VidIQ/TubeBuddy API for SEO
- [ ] Add Reddit API for research
