# Openclaw Bot Integration Guide

## Overview

The YouTube Automation system now includes a complete plugin system for **Openclaw/Clawd bot** integration, enabling Discord and Telegram users to create, manage, and distribute videos across multiple platforms.

## What Was Found

### Existing Multi-Platform Support (Already In Place)

The codebase already had extensive multi-platform capabilities:

#### 1. **Video Repurposing Module** (`src/social/multi_platform.py` - 38KB)
- Converts long-form YouTube videos to short-form formats
- Supports: YouTube Shorts, TikTok, Instagram Reels, Pinterest
- Auto-resizes videos (16:9 → 9:16 for vertical platforms)
- Platform-specific metadata and hashtag handling
- Smart segment extraction for best moments

#### 2. **Social Media Posting** (`src/social/social_poster.py` - 35KB)
- Abstract platform interface (SocialPlatform base class)
- Implementations for: Twitter/X, Reddit, LinkedIn, Facebook, Discord
- Content validation per platform specs
- Scheduled posting with retry logic
- Image and URL support

#### 3. **Content Creation Pipeline**
- Script generation (Groq, Claude, OpenAI, Ollama)
- Text-to-speech (Edge-TTS, Fish Audio)
- Video assembly (MoviePy, FFmpeg)
- Thumbnail generation
- Professional audio enhancement (6-band EQ, noise reduction, -14 LUFS normalization)

#### 4. **YouTube Integration**
- OAuth2 authentication
- Direct video upload
- Metadata management
- Privacy controls (public, unlisted, private)

#### 5. **Analytics & Reporting**
- YouTube Analytics API integration
- Performance tracking (CTR, retention, engagement)
- KPI dashboard
- A/B testing system

## What Was Added

### New Plugin System: `/src/openclaw/`

Complete bot integration layer with 7 new modules:

#### 1. **plugin.py** - Main Plugin Interface
```python
from src.openclaw import OpenclawPlugin

plugin = OpenclawPlugin()
response = await plugin.handle_command(
    command="/video",
    args={"topic": "Passive Income Tips"},
    user_id="user123",
    username="john_doe",
    platform="discord"
)
```

**Features:**
- Command routing and execution
- Job queue management (async execution)
- Rate limiting per user
- Permission-based access control
- Real-time progress tracking
- Error handling and logging

#### 2. **models.py** - Data Models
Standardized dataclasses for:
- `CommandRequest` - Incoming bot commands
- `PluginResponse` - Structured responses
- `Job` - Async job tracking
- `JobStatus` - Enum (queued, running, completed, failed, cancelled)
- `PermissionLevel` - Enum (admin, moderator, user, public)
- Command-specific models: `VideoCommand`, `BatchCommand`, `AnalyticsCommand`, etc.

#### 3. **command_registry.py** - Command Registration
- Auto-registers 10 core commands
- Command parsing and validation
- Permission checking
- Rate limit enforcement
- Help text generation
- Extensible command system

**Registered Commands:**
```
/video          - Create single video
/short          - Create YouTube Short
/batch          - Batch video creation
/multiplatform  - Export to all platforms
/schedule       - Schedule content
/analytics      - View performance metrics
/cost           - API usage and costs
/status         - System status
/configure      - Update settings
/help           - Command help
```

#### 4. **api_bridge.py** - HTTP/WebSocket Bridge
```python
from fastapi import FastAPI
from src.openclaw import APIBridge

bridge = APIBridge()
app = bridge.setup_fastapi()
# uvicorn main:app --reload
```

**Endpoints:**
- `POST /command` - Execute command
- `GET /job/{job_id}` - Get job status
- `GET /jobs/{user_id}` - Get user's jobs
- `POST /job/{job_id}/cancel` - Cancel job
- `GET /status` - System status
- `GET /health` - Health check

#### 5. **discord_handler.py** - Discord Bot Integration
Ready-to-use Discord bot with:
- Slash command support
- Rich embeds for responses
- Job status tracking
- Real-time status updates
- Error handling
- User-friendly formatting

```bash
export DISCORD_TOKEN="your_token"
python -m src.openclaw.discord_handler
```

#### 6. **telegram_handler.py** - Telegram Bot Integration
Ready-to-use Telegram bot with:
- Command handlers
- Inline keyboards for options
- Job monitoring
- Status notifications
- Markdown formatting

```bash
export TELEGRAM_TOKEN="your_token"
python -m src.openclaw.telegram_handler
```

#### 7. **README.md** - Complete Documentation
- Quick start guides
- API examples
- Command reference
- Configuration options
- Integration examples
- Error handling patterns

### New Configuration File
**`config/openclaw_config.json`** - Plugin configuration with:
- Rate limits per command
- Permission mappings
- Platform enablement flags
- Video defaults
- AI provider settings
- Logging configuration

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Discord/Telegram Bot                       │
│          (Openclaw/Clawd Bot User)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│         Webchat/API Webhook Handler                     │
└────────────────────┬────────────────────────────────────┘
                     │
       ┌─────────────┴─────────────┐
       ↓                           ↓
  Discord Bot          Telegram Bot
  (discord.py)         (python-telegram-bot)
       │                           │
       └─────────────┬─────────────┘
                     ↓
        ┌────────────────────────┐
        │   APIBridge            │
        │ (HTTP/WebSocket)       │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │  OpenclawPlugin        │
        │  (Command Router)      │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │  CommandRegistry       │
        │  (Validation/Routing)  │
        └────────────┬───────────┘
                     ↓
  ┌──────────────────────────────────────────┐
  │  YouTube Automation System                │
  ├──────────────────────────────────────────┤
  │ • Script Generation (AI providers)        │
  │ • Text-to-Speech (Edge-TTS, Fish Audio)  │
  │ • Video Assembly (MoviePy + FFmpeg)      │
  │ • Multi-Platform Distribution             │
  │ • Social Media Posting                    │
  │ • YouTube Upload & API                    │
  │ • Analytics & Reporting                   │
  │ • Smart Scheduling                        │
  └──────────────────────────────────────────┘
                     ↓
  ┌──────────────────────────────────────────┐
  │    Platform Distribution                  │
  ├──────────────────────────────────────────┤
  │ YouTube → TikTok → Instagram → Pinterest │
  │ Twitter → Reddit → LinkedIn → Facebook    │
  │ Discord → Quora                           │
  └──────────────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │  Discord/Telegram     │
         │  Response with Links  │
         └───────────────────────┘
```

## Supported Platforms

### Video Platforms
| Platform | Format | Max Duration | Status |
|----------|--------|--------------|--------|
| YouTube | 16:9 (1920x1080) | Unlimited | Uploading |
| YouTube Shorts | 9:16 (1080x1920) | 60s | Uploading |
| TikTok | 9:16 (1080x1920) | 60s | Export ready |
| Instagram Reels | 9:16 (1080x1920) | 90s | Export ready |
| Pinterest | 1:1 or custom | - | Export ready |

### Social Platforms
| Platform | Feature | Status |
|----------|---------|--------|
| Twitter/X | Tweets + links | Posting enabled |
| Reddit | Comments + links | Posting enabled |
| LinkedIn | Professional posts | Posting enabled |
| Facebook | Community posts | Posting enabled |
| Discord | Webhooks | Posting enabled |
| Quora | Answers + links | Template-based |

## Key Features

### 1. **Job Queue System**
- Async execution of long-running tasks
- Real-time progress tracking (0-100%)
- Live logging per job
- Automatic cleanup of old jobs
- Job cancellation support

### 2. **Permission System**
```
ADMIN      → Full access, configuration changes
MODERATOR  → Video creation, batch operations, scheduling
USER       → Personal video creation, analytics, cost
PUBLIC     → Help, status only
```

### 3. **Rate Limiting**
Prevents abuse:
- `/video`: 1 per 5 minutes
- `/batch`: 1 per hour
- `/short`: 1 per minute
- `/analytics`: 10 per minute

### 4. **Command Parsing**
Flexible argument parsing with:
- Positional arguments
- Named flags (--flag value)
- Quoted strings support
- Argument validation

### 5. **Error Handling**
- Graceful error messages
- Detailed error logs
- User-friendly feedback
- Permission-based error details

## Usage Examples

### Discord
```
User: /video Passive Income Tips money_blueprints
Bot: ✅ Video creation queued: Passive Income Tips
     Job ID: `550e8400-e29b-41d4-a716-446655440000`

User: /job 550e8400-e29b-41d4-a716-446655440000
Bot: 📋 Job Status
     Status: COMPLETED (100%)
     Results: https://youtube.com/watch?v=...
```

### Telegram
```
User: /video "How to Make Money"
Bot: ⏳ Creating video...
     ✅ Video creation queued!
     Job ID: `550e8400-e29b-41d4-a716-446655440000`

User: /analytics money_blueprints
Bot: 📈 Analytics
     Views: 15,234
     CTR: 4.2%
     Retention: 68%
```

### HTTP API
```bash
curl -X POST "http://localhost:8000/command" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "/video",
    "args": {"topic": "Passive Income"},
    "user_id": "user123",
    "username": "john",
    "platform": "discord"
  }'
```

## Integration Checklist

- [x] Plugin base class and command routing
- [x] Job queue and async execution
- [x] Permission system
- [x] Rate limiting
- [x] Discord bot handler
- [x] Telegram bot handler
- [x] HTTP API bridge
- [x] Configuration management
- [x] Comprehensive documentation
- [ ] Webhook callbacks for updates (future)
- [ ] Database persistence (future)
- [ ] Auto-retry failed jobs (future)
- [ ] Job templates/presets (future)

## Files Added

```
src/openclaw/
├── __init__.py              (60 lines)
├── plugin.py                (450+ lines)
├── models.py                (280+ lines)
├── command_registry.py      (350+ lines)
├── api_bridge.py            (250+ lines)
├── discord_handler.py       (450+ lines)
├── telegram_handler.py      (380+ lines)
└── README.md                (Complete guide)

config/
└── openclaw_config.json     (Plugin configuration)

docs/
└── OPENCLAW_INTEGRATION.md  (This file)
```

**Total New Code:** ~2,500+ lines of production code

## Getting Started

### 1. Install Dependencies
```bash
pip install discord.py python-telegram-bot fastapi uvicorn
```

### 2. Run Discord Bot
```bash
export DISCORD_TOKEN="your_bot_token"
python -m src.openclaw.discord_handler
```

### 3. Run Telegram Bot
```bash
export TELEGRAM_TOKEN="your_bot_token"
python -m src.openclaw.telegram_handler
```

### 4. Run API Server
```bash
uvicorn api.server:app --reload
```

### 5. Send a Command
```bash
# Discord: /video Passive Income Tips
# Telegram: /video Passive Income Tips
# HTTP: POST /command with JSON payload
```

## Configuration

Edit `config/openclaw_config.json` to:
- Adjust rate limits
- Change permission mappings
- Enable/disable platforms
- Set default video parameters
- Configure AI providers
- Adjust logging levels

## Next Steps

1. **Production Deployment**
   - Database for job persistence
   - Redis for job queue
   - Horizontal scaling

2. **Enhanced Features**
   - Webhook callbacks for real-time updates
   - Job templates and presets
   - Auto-retry failed jobs
   - Advanced analytics dashboard
   - Direct platform uploads (TikTok, Instagram APIs)

3. **Monitoring**
   - Prometheus metrics
   - Error tracking (Sentry)
   - Performance monitoring

4. **Security**
   - OAuth2 user management
   - API key rotation
   - Audit logging
   - Rate limit by IP/user

## Support & Questions

- Review `/src/openclaw/README.md` for API docs
- Check example handlers for integration patterns
- See `config/openclaw_config.json` for configuration options
- Review job logs for troubleshooting

---

**Status:** ✅ Plugin system complete and ready for integration

**Created:** 2026-02-26
**Version:** 1.0.0
