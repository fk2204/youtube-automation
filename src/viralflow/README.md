# ViralFlow Plugin System

Integration module for [ViralFlow](https://clawd.bot) to access YouTube Automation capabilities from Discord and Telegram.

## Overview

The ViralFlow plugin allows you to:
- Create videos with `/video <topic>`
- Generate YouTube Shorts with `/short <topic>`
- Batch create videos with `/batch <count>`
- Export to TikTok, Instagram, Pinterest with `/multiplatform <video_id>`
- Schedule content with `/schedule <topic> <time>`
- View analytics with `/analytics [channel]`
- Check API costs with `/cost`
- Get system status with `/status`

## Architecture

```
Discord/Telegram Bot
         ↓
    Webhook/API
         ↓
   APIBridge (api_bridge.py)
         ↓
  ViralFlowPlugin (plugin.py)
         ↓
CommandRegistry (command_registry.py)
         ↓
YouTube Automation System
├─ Script Generation
├─ TTS Voice Synthesis
├─ Video Assembly
├─ Multi-Platform Export
├─ Social Media Posting
└─ Analytics Tracking
```

## Quick Start

### 1. Basic Plugin Usage

```python
from src.viralflow import ViralFlowPlugin

# Initialize plugin
plugin = ViralFlowPlugin()

# Handle a command
response = await plugin.handle_command(
    command="/video",
    args={"topic": "Passive Income Tips"},
    user_id="user123",
    username="john_doe",
    platform="discord"
)

print(response.to_dict())
# Output:
# {
#   "success": true,
#   "message": "Video creation queued: Passive Income Tips",
#   "job_id": "550e8400-e29b-41d4-a716-446655440000"
# }
```

### 2. With FastAPI

```python
from fastapi import FastAPI
from src.viralflow import APIBridge

# Create bridge and FastAPI app
bridge = APIBridge()
app = bridge.setup_fastapi()

# Run server
# uvicorn main:app --reload
```

Then POST to `http://localhost:8000/command`:
```bash
curl -X POST "http://localhost:8000/command" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "/video",
    "args": {"topic": "Make Money Online"},
    "user_id": "user123",
    "username": "john",
    "platform": "discord"
  }'
```

### 3. Discord Bot Integration

```python
import discord
from discord.ext import commands
from src.viralflow import ViralFlowPlugin

bot = commands.Bot(command_prefix="/")
plugin = ViralFlowPlugin()

@bot.command(name="video")
async def create_video(ctx, *, topic):
    """Create a video from Discord"""
    response = await plugin.handle_command(
        command="/video",
        args={"topic": topic},
        user_id=str(ctx.author.id),
        username=ctx.author.name,
        platform="discord"
    )

    if response.success:
        embed = discord.Embed(
            title="🎥 Video Creation Started",
            description=response.message,
            color=discord.Color.blue()
        )
        await ctx.send(embed=embed)
    else:
        await ctx.send(f"❌ Error: {response.message}")

bot.run("YOUR_TOKEN")
```

### 4. Telegram Bot Integration

```python
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from src.viralflow import ViralFlowPlugin

plugin = ViralFlowPlugin()

async def video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /video command in Telegram"""
    if not context.args:
        await update.message.reply_text("Usage: /video <topic>")
        return

    topic = " ".join(context.args)
    response = await plugin.handle_command(
        command="/video",
        args={"topic": topic},
        user_id=str(update.effective_user.id),
        username=update.effective_user.username or "Anonymous",
        platform="telegram",
        channel_id=str(update.effective_chat.id)
    )

    if response.success:
        await update.message.reply_text(
            f"✅ {response.message}\n\nJob ID: `{response.job_id}`",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(f"❌ Error: {response.message}")

# Setup application
app = Application.builder().token("YOUR_TOKEN").build()
app.add_handler(CommandHandler("video", video_command))
app.run_polling()
```

## Available Commands

### `/video <topic> [channel]`
Create a single video
```
/video Passive Income Tips money_blueprints
/video How to Make Money --upload --privacy public
```

### `/short <topic> [channel]`
Create a YouTube Short (15-60 seconds)
```
/short Quick Money Tips
```

### `/batch <count> [channel]`
Create multiple videos (requires MODERATOR permission)
```
/batch 5 money_blueprints
/batch 10 --spacing 2
```

### `/multiplatform <video_id>`
Export video to TikTok, Instagram, Pinterest, etc.
```
/multiplatform dQw4w9WgXcQ
/multiplatform abc123 --platforms tiktok,instagram
```

### `/schedule <topic> <datetime>`
Schedule video creation (requires MODERATOR permission)
```
/schedule "Passive Income Tips" "2026-03-01 10:00"
/schedule "Quick Tips" "2026-02-28 15:00" --recurring daily
```

### `/analytics [channel] [--period]`
View channel analytics
```
/analytics
/analytics money_blueprints
/analytics --period month
```

### `/cost [--period]`
View API token usage and costs
```
/cost
/cost --period week
```

### `/status [--jobs] [--detailed]`
Check system status
```
/status
/status --jobs
/status --detailed
```

### `/help [command]`
Get help on commands
```
/help
/help /video
```

## Permission Levels

| Level | Permissions | Examples |
|-------|-------------|----------|
| ADMIN | All commands, configuration changes | Bot owner |
| MODERATOR | Video/batch creation, scheduling | Channel moderators |
| USER | Basic creation, analytics | Regular users |
| PUBLIC | Help, status only | Guests |

## Job Management

### Get Job Status
```python
job = plugin.get_job_status("550e8400-e29b-41d4-a716-446655440000")
print(job.status)  # JobStatus.COMPLETED
print(job.results)  # {"video_url": "...", ...}
```

### Get User's Jobs
```python
jobs = plugin.get_user_jobs("user123", limit=5)
for job in jobs:
    print(f"{job.command}: {job.status.value}")
```

### Cancel Job
```python
job = plugin.get_job_status("550e8400-e29b-41d4-a716-446655440000")
job.status = JobStatus.CANCELLED
```

## Configuration

Create `config/viralflow_config.json`:
```json
{
  "rate_limits": {
    "/video": 300,
    "/batch": 3600,
    "/short": 60
  },
  "max_queue_size": 100,
  "job_timeout": 3600,
  "permissions": {
    "admin": ["*"],
    "moderator": ["/video", "/short", "/batch", "/schedule"],
    "user": ["/video", "/short", "/analytics", "/cost"]
  }
}
```

Load config:
```python
plugin = ViralFlowPlugin(config_path="config/viralflow_config.json")
```

## Rate Limiting

Prevents abuse by limiting commands per user:

| Command | Limit | Note |
|---------|-------|------|
| `/video` | 1 per 5 min | CPU intensive |
| `/batch` | 1 per hour | Very expensive |
| `/short` | 1 per 1 min | Light work |
| `/analytics` | 10 per min | Cheap |

## Integration with YouTube Automation

The plugin integrates with existing modules:

### Script Generation
```python
from src.content.script_writer import ScriptWriter

writer = ScriptWriter(provider="groq")
script = writer.generate_script(topic="Passive Income", duration_minutes=10)
```

### Multi-Platform Export
```python
from src.social.multi_platform import MultiPlatformDistributor

distributor = MultiPlatformDistributor()
exports = await distributor.export_all_platforms(
    video_path="output/video.mp4",
    title="My Video",
    niche="finance"
)
```

### Social Media Posting
```python
from src.social.social_poster import SocialPoster

poster = SocialPoster()
result = poster.post_to_twitter("Check out my new video!", video_url="...")
```

## Real-Time Updates

Register callbacks to get notified of job events:

```python
async def on_job_completed(data):
    print(f"Job {data['job_id']} completed!")
    print(f"Video URL: {data['results']['video_url']}")

plugin.job_queue_manager.register_callback("job_completed", on_job_completed)
```

## API Examples

### cURL
```bash
# Create video
curl -X POST "http://localhost:8000/command" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "/video",
    "args": {"topic": "Make Money Online"},
    "user_id": "user123",
    "username": "john",
    "platform": "discord"
  }'

# Check job status
curl "http://localhost:8000/job/550e8400-e29b-41d4-a716-446655440000"

# Get user's jobs
curl "http://localhost:8000/jobs/user123?limit=5"

# Check system status
curl "http://localhost:8000/status"
```

### Python Requests
```python
import requests

# Create video
response = requests.post("http://localhost:8000/command", json={
    "command": "/video",
    "args": {"topic": "Passive Income"},
    "user_id": "user123",
    "username": "john",
    "platform": "discord"
})
job_id = response.json()["data"]["job_id"]

# Poll job status
import time
while True:
    status = requests.get(f"http://localhost:8000/job/{job_id}").json()
    if status["data"]["status"] in ["completed", "failed"]:
        print(status)
        break
    time.sleep(5)
```

## Error Handling

```python
response = await plugin.handle_command(
    command="/video",
    args={"invalid": "args"},
    user_id="user123",
    username="john"
)

if not response.success:
    print(f"Error: {response.error}")
    print(f"Message: {response.message}")
```

## Future Enhancements

- [ ] Webhook callbacks for job status updates
- [ ] Database persistence for job history
- [ ] Advanced permission system with role-based access
- [ ] Job templates and presets
- [ ] Direct TikTok/Instagram API uploads
- [ ] Automated content calendar
- [ ] Performance A/B testing integration
- [ ] Cost estimation before job execution

## Support

For issues or questions:
1. Check the [main project documentation](../README.md)
2. Review example handlers in `discord_handler.py`, `telegram_handler.py`
3. Check job logs: `plugin.get_job_status(job_id).logs`
