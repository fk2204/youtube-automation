# Plugin & Library Recommendations

## Test Results Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Edge-TTS | ✅ Working | 322 voices available, FREE |
| Google Trends | ✅ Working | Found rising queries |
| Thumbnail Gen | ✅ Working | 17KB PNG created |
| Audio Gen | ✅ Working | 51KB MP3 created |
| FFmpeg | ❌ Not installed | Needed for video |
| Ollama | ⚠️ No models | Need to run `ollama pull llama3.2` |

---

## Recommended Plugins by Category

### Text-to-Speech (TTS)

| Plugin | Speed | Quality | Cost | Best For |
|--------|-------|---------|------|----------|
| **Edge-TTS** ⭐ | Fast | Good | FREE | YouTube automation |
| Kokoro | Very Fast | Good | FREE | Real-time apps |
| Piper | Fast | Good | FREE | Offline/embedded |
| Coqui TTS | Slow | Excellent | FREE | Voice cloning |
| ElevenLabs | Fast | Excellent | $5+/mo | Premium quality |

**Recommendation:** Start with Edge-TTS (installed). Consider Kokoro for faster generation.

```bash
# Edge-TTS (current)
pip install edge-tts

# Kokoro (faster alternative)
pip install kokoro
```

### AI Script Generation

| Plugin | Speed | Quality | Cost | Limit |
|--------|-------|---------|------|-------|
| **Ollama** ⭐ | Medium | Good | FREE | Unlimited |
| **Groq** ⭐ | Very Fast | Good | FREE | 30 req/min |
| Gemini | Fast | Good | FREE | 15 req/min |
| Claude | Medium | Excellent | $10-30/mo | Pay per token |
| OpenAI | Medium | Excellent | $20+/mo | Pay per token |

**Recommendation:**
- Development: Ollama (local, unlimited)
- Production: Groq (fast, free tier generous)

```bash
# Ollama setup
# 1. Download from https://ollama.ai
# 2. Run:
ollama pull llama3.2

# Groq setup
# 1. Get key from https://console.groq.com
# 2. Set in .env:
GROQ_API_KEY=gsk_xxxxx
AI_PROVIDER=groq
```

### Video Generation

| Plugin | Speed | Complexity | Best For |
|--------|-------|------------|----------|
| **FFmpeg** ⭐ | Fastest | Low | Simple videos |
| MovieLite | Fast | Medium | MoviePy replacement |
| MoviePy | Slow | High | Complex editing |
| OpenCV | Fast | High | Real-time |
| Movis | Medium | High | ML integration |

**Recommendation:** Use FFmpeg for simple background+audio videos (10x faster than MoviePy).

```bash
# Windows FFmpeg install
winget install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

### Research & Trends

| Plugin | Data Source | Cost | Rate Limit |
|--------|-------------|------|------------|
| **pytrends** ⭐ | Google Trends | FREE | ~10 req/min |
| **PRAW** | Reddit | FREE | 60 req/min |
| python-youtube | YouTube API | FREE | 10k units/day |
| SerpAPI | Google Search | $50+/mo | Based on plan |

**Recommendation:** pytrends + PRAW cover most research needs for free.

### YouTube API

| Plugin | Purpose | Notes |
|--------|---------|-------|
| **google-api-python-client** ⭐ | Official API | Full access |
| python-youtube | Read-only | Simpler API |
| PyTubeFix | Download | Videos only |

### Scheduling

| Plugin | Features | Notes |
|--------|----------|-------|
| **APScheduler** ⭐ | Full featured | Cron, intervals |
| schedule | Simple | Basic only |
| Celery | Distributed | Overkill for this |

---

## Quick Install Commands

### Minimal (FREE, works now)
```bash
pip install edge-tts pytrends loguru rich python-dotenv pyyaml pillow
```

### Recommended (FREE + better features)
```bash
pip install edge-tts pytrends praw loguru rich python-dotenv pyyaml pillow aiohttp requests apscheduler google-api-python-client google-auth-oauthlib

# Then install FFmpeg separately
winget install ffmpeg
```

### Full (includes paid options)
```bash
pip install -r requirements.txt
```

---

## Performance Comparison

### Video Generation Speed
```
FFmpeg (direct):    ~5 seconds for 10-min video
MovieLite:          ~30 seconds
MoviePy:            ~2-5 minutes
```

### TTS Speed (1000 words)
```
Edge-TTS:           ~3 seconds
Kokoro:             ~1 second
Coqui XTTS:         ~30 seconds
Tortoise:           ~5 minutes
```

### AI Script Generation (10-min script)
```
Groq (Llama 3.3):   ~5 seconds
Ollama (local):     ~15-30 seconds
Claude:             ~10 seconds
OpenAI:             ~10 seconds
```

---

## Recommended Stack for Best Performance

1. **AI:** Groq (free, fastest) or Ollama (free, local)
2. **TTS:** Edge-TTS (free, good quality)
3. **Video:** FFmpeg (direct, fastest)
4. **Research:** pytrends + PRAW (free)
5. **Upload:** google-api-python-client

**Total Cost: $0/month**

---

## Next Steps

1. Install FFmpeg: `winget install ffmpeg`
2. Setup Ollama: `ollama pull llama3.2`
3. (Optional) Get Groq key: https://console.groq.com
4. Test full pipeline: `python src/main.py --niche "python tutorials"`

---

## Sources

- [MovieLite - Faster MoviePy](https://github.com/francozanardi/movielite)
- [Edge-TTS](https://github.com/rany2/edge-tts)
- [Kokoro TTS](https://smallest.ai/blog/open-source-tts-alternatives-compared)
- [Groq Free API](https://console.groq.com)
- [pytrends](https://github.com/GeneralMills/pytrends)
