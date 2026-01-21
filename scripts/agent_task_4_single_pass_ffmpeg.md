# Agent Task 4: Implement Single-Pass FFmpeg in video_fast.py

## Objective
Replace 4 separate FFmpeg calls with 1 complex filtergraph for 4× faster video generation.

## File Location
C:\Users\fkozi\youtube-automation\src\content\video_fast.py

## Problem
Current implementation makes 4 separate FFmpeg passes:
1. Normalize audio (-14 LUFS)
2. Mix background music
3. Create video from audio
4. Add subtitles

Each pass requires reading/writing full files. Single-pass combines all operations.

## Current Workflow (SLOW - 4 passes)
```
audio.mp3 → [normalize] → audio_norm.mp3
audio_norm.mp3 + music.mp3 → [mix] → audio_mixed.mp3
audio_mixed.mp3 → [video create] → video_no_subs.mp4
video_no_subs.mp4 + subs.srt → [burn subs] → final.mp4
```

## Target Workflow (FAST - 1 pass)
```
audio.mp3 + music.mp3 + subs.srt → [complex filtergraph] → final.mp4
```

## Implementation Strategy

### 1. Find the create_video() method
This is the main method that orchestrates video creation.

### 2. Create a new method: create_video_single_pass()
```python
def create_video_single_pass(
    self,
    audio_file: str,
    output_file: str,
    background_music: Optional[str] = None,
    subtitle_file: Optional[str] = None,
    normalize_audio: bool = True,
    music_volume: float = 0.15
) -> bool:
    """
    Single-pass FFmpeg video generation with complex filtergraph.

    Combines all operations in one FFmpeg call:
    - Audio normalization (-14 LUFS)
    - Music mixing
    - Video creation
    - Subtitle burning

    4× faster than multi-pass approach.
    """
```

### 3. Build the Complex Filtergraph

**Base filtergraph (audio + video):**
```python
# Audio chain
filters = []

# Input 0: Main audio
# Input 1: Background music (optional)
# Input 2: Background image (for video)

if background_music:
    # Mix audio with music at specified volume
    filters.append(f"[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[anorm]")
    filters.append(f"[1:a]volume={music_volume}[music]")
    filters.append(f"[anorm][music]amix=inputs=2:duration=first[audio]")
else:
    filters.append(f"[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[audio]")

# Video creation from image
filters.append(f"[2:v]scale={self.width}:{self.height}[video]")

filtergraph = ";".join(filters)
```

**With subtitles:**
```python
# Add subtitle filter to video chain
subtitle_filter = f"subtitles={subtitle_file}:force_style='FontName=Arial Bold,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Shadow=1'"

# Modify video chain
filters[-1] = f"[2:v]scale={self.width}:{self.height}[scaled]"
filters.append(f"[scaled]{subtitle_filter}[video]")
```

### 4. Build FFmpeg Command
```python
ffmpeg_cmd = [
    self.ffmpeg_path,
    "-i", audio_file,  # Input 0: audio
]

if background_music:
    ffmpeg_cmd.extend(["-i", background_music])  # Input 1: music

ffmpeg_cmd.extend([
    "-loop", "1",
    "-i", background_image,  # Input 2: background
    "-filter_complex", filtergraph,
    "-map", "[video]",
    "-map", "[audio]",
    "-c:v", self.encoder,
    "-c:a", "aac",
    "-b:a", "256k",
    "-shortest",
    "-y",
    output_file
])
```

### 5. Add GPU Encoder Support
Use the existing `get_video_encoder()` function to detect GPU.

### 6. Update create_video() to use single-pass
Add a parameter `single_pass: bool = True` and route to the new method:
```python
def create_video(self, ..., single_pass: bool = True):
    if single_pass:
        return self.create_video_single_pass(...)
    else:
        # Keep old multi-pass code for fallback
        return self._create_video_multipass(...)
```

### 7. Rename old method
Rename the current `create_video` logic to `_create_video_multipass()` for fallback.

## Testing
```bash
cd "C:\Users\fkozi\youtube-automation"
python -c "
from src.content.video_fast import FastVideoGenerator
import asyncio

gen = FastVideoGenerator()
# Test would need actual audio file
print('Single-pass FFmpeg implementation complete')
"
```

## Expected Outcome
- New `create_video_single_pass()` method
- Single FFmpeg call with complex filtergraph
- 4× faster video generation (measured by time)
- Backward compatible (single_pass parameter)
- Old multi-pass code preserved as fallback

## Deliverable
Modified C:\Users\fkozi\youtube-automation\src\content\video_fast.py with single-pass FFmpeg.
