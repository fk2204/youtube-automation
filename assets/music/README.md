# Background Music for YouTube Automation

This directory contains background music files for automated video generation.

## Expected File Names

The video generator looks for music files in this order:

1. **Niche-specific music** (highest priority):
   - `finance.mp3` - For finance/money content
   - `psychology.mp3` - For psychology/self-improvement content
   - `storytelling.mp3` - For stories/true crime content

2. **Generic fallback**:
   - `background.mp3` - Used when no niche-specific music is found

3. **Any MP3 file** - If none of the above exist, the system will use any `.mp3` file in this directory

## Recommended Music Characteristics

### Finance (`finance.mp3`)
- **Style**: Corporate, inspirational, upbeat
- **Tempo**: Medium (100-120 BPM)
- **Instruments**: Piano, light strings, subtle electronic elements
- **Mood**: Professional, motivating, confident
- **Avoid**: Heavy bass, aggressive beats, distracting melodies

### Psychology (`psychology.mp3`)
- **Style**: Ambient, contemplative, atmospheric
- **Tempo**: Slow to medium (70-100 BPM)
- **Instruments**: Soft piano, pads, light strings, subtle textures
- **Mood**: Thoughtful, calming, introspective
- **Avoid**: Upbeat rhythms, sudden changes, jarring sounds

### Storytelling (`storytelling.mp3`)
- **Style**: Cinematic, dramatic, tension-building
- **Tempo**: Variable (60-110 BPM), builds tension
- **Instruments**: Orchestral elements, deep bass, suspenseful tones
- **Mood**: Mysterious, engaging, slightly dark
- **Avoid**: Happy/uplifting themes, comedic elements

### Background (`background.mp3`)
- **Style**: Neutral, versatile, non-distracting
- **Tempo**: Medium (90-110 BPM)
- **Instruments**: Soft synths, light percussion, ambient pads
- **Mood**: Pleasant, professional, unobtrusive
- **Avoid**: Strong themes, memorable melodies, vocals

## Where to Get Royalty-Free Music

### Free Sources (No Attribution Required)

1. **YouTube Audio Library** (Recommended)
   - URL: https://studio.youtube.com/channel/UC/music
   - Access: Requires YouTube account
   - License: Free for YouTube videos
   - Quality: High-quality, professional tracks

2. **Pixabay Music**
   - URL: https://pixabay.com/music/
   - License: Free for commercial use, no attribution required
   - Quality: Varies, many good options

3. **Mixkit**
   - URL: https://mixkit.co/free-stock-music/
   - License: Free for commercial use
   - Quality: Professional quality

4. **Uppbeat**
   - URL: https://uppbeat.io/
   - License: Free tier available (with attribution)
   - Quality: High quality, curated

### Free Sources (Attribution Required)

5. **Free Music Archive (FMA)**
   - URL: https://freemusicarchive.org/
   - License: Various Creative Commons licenses
   - Quality: Varies widely

6. **ccMixter**
   - URL: http://ccmixter.org/
   - License: Creative Commons
   - Quality: Community-created content

### Paid Sources (Higher Quality)

7. **Epidemic Sound**
   - URL: https://www.epidemicsound.com/
   - Best for: Professional YouTube channels
   - License: Subscription-based, full clearance

8. **Artlist**
   - URL: https://artlist.io/
   - Best for: High production value
   - License: Subscription-based

## Volume Settings

The default music volume is set to:
- **Regular videos**: 12% (0.12)
- **YouTube Shorts**: 15% (0.15)

You can override these in `config/channels.yaml` per channel:

```yaml
settings:
  music_enabled: true
  music_volume: 0.15  # 15% volume
```

## Audio Requirements

- **Format**: MP3 (recommended) or WAV
- **Bitrate**: 128kbps minimum, 320kbps recommended
- **Duration**: At least 3 minutes (will loop automatically)
- **Loudness**: Normalized to -14 LUFS for consistency

## Tips for Selecting Music

1. **Test with voiceover**: Music should enhance, not compete with narration
2. **Avoid lyrics**: Instrumental only to prevent distraction
3. **Consider loop points**: Music will loop for long videos
4. **Match the mood**: Music sets emotional tone for viewers
5. **Stay consistent**: Use same track for series/brand recognition

## Quick Start

1. Download a suitable track from YouTube Audio Library
2. Rename it to match your niche (e.g., `finance.mp3`)
3. Place it in this directory (`assets/music/`)
4. The video generator will automatically use it

## Troubleshooting

**Music not playing in videos?**
- Check that `music_enabled: true` in channels.yaml
- Verify the file exists and is a valid MP3
- Check the logs for "Background music" messages

**Music too loud/quiet?**
- Adjust `music_volume` in channels.yaml (0.0 to 1.0)
- Re-encode the source file to normalize loudness

**Music cuts off abruptly?**
- The system auto-fades music at video end
- For smoother loops, use tracks with clean loop points
