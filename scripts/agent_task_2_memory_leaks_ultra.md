# Agent Task 2: Fix Memory Leaks in video_ultra.py

## Objective
Add explicit `.close()` calls and `gc.collect()` for all VideoClip objects to prevent OOM crashes in batch processing.

## File Location
C:\Users\fkozi\youtube-automation\src\content\video_ultra.py

## Problem
MoviePy VideoClip objects hold large memory buffers. Without explicit cleanup, batch processing (3+ videos) causes out-of-memory crashes.

## Required Changes

### 1. Import gc at the top (if not already present)
```python
import gc
```

### 2. Add try/finally blocks with cleanup for ALL video operations

**Pattern to Apply:**
```python
# Before (memory leak):
clip = VideoFileClip("video.mp4")
# ... process clip ...
return final_clip

# After (proper cleanup):
clip = None
try:
    clip = VideoFileClip("video.mp4")
    # ... process clip ...
    return final_clip
finally:
    if clip:
        clip.close()
    gc.collect()
```

### 3. Key Methods to Fix
Search for these patterns and add cleanup:
- `VideoFileClip(` - All video file loads
- `ImageClip(` - All image clip creations
- `TextClip(` - All text clip creations
- `CompositeVideoClip(` - All composite creations
- `concatenate_videoclips(` - All concatenations

### 4. Method-level cleanup
For methods that create clips, add cleanup at the end:
```python
def some_method(self):
    clips = []
    try:
        # ... create clips ...
        final = CompositeVideoClip(clips)
        return final
    finally:
        for clip in clips:
            if clip:
                clip.close()
        gc.collect()
```

### 5. Main create_video() method
Add comprehensive cleanup in the main generation method:
```python
def create_video(self, ...):
    all_clips = []
    try:
        # ... video generation ...
        final_video.write_videofile(...)
    finally:
        # Close all intermediate clips
        for clip in all_clips:
            if clip:
                clip.close()
        if 'final_video' in locals():
            final_video.close()
        gc.collect()
```

## Testing
After changes, test with batch processing:
```bash
cd "C:\Users\fkozi\youtube-automation"
python -c "
from src.content.video_ultra import UltraVideoGenerator
gen = UltraVideoGenerator()
print('Memory leak fixes applied successfully')
"
```

## Expected Outcome
- All VideoClip objects have explicit .close() calls
- try/finally blocks protect against exceptions
- gc.collect() called after major operations
- No breaking changes to API
- Batch processing can handle 5+ videos without OOM

## Deliverable
Modified C:\Users\fkozi\youtube-automation\src\content\video_ultra.py with memory leak fixes.
