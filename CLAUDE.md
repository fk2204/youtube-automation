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

## Recent Updates (2026-01)

- [x] YouTube Shorts support (1080x1920 vertical)
- [x] Research-backed optimizations (safe zones, 30s optimal duration)
- [x] Smart stock footage matching (niche-specific keywords)
- [x] SQLite database tracking
- [x] Shorts scheduler (post after regular videos)
- [x] Background music support (15% volume)

## Future Improvements

- [ ] Add screen recording simulation with Playwright
- [ ] Implement CrewAI multi-agent workflow
- [ ] Add video analytics tracking
- [ ] Add A/B testing for thumbnails
- [x] Crossfade transitions between segments (2026-01)
- [x] Animated gradient fallbacks (2026-01)
- [x] Burned-in captions support (2026-01)
- [x] Background music integration (2026-01)
