# Agent Task 3: Fix Memory Leaks in pro_video_engine.py

## Objective
Add explicit `.close()` calls and `gc.collect()` for all VideoClip objects to prevent OOM crashes.

## File Location
C:\Users\fkozi\youtube-automation\src\content\pro_video_engine.py

## Problem
Same as video_ultra.py - MoviePy VideoClip objects need explicit cleanup to prevent memory leaks.

## Required Changes

### 1. Import gc at the top (if not already present)
```python
import gc
```

### 2. Add try/finally blocks with cleanup

**Apply this pattern everywhere:**
```python
# Before (memory leak):
clip = VideoFileClip("video.mp4")
processed = process_clip(clip)
return processed

# After (proper cleanup):
clip = None
processed = None
try:
    clip = VideoFileClip("video.mp4")
    processed = process_clip(clip)
    return processed
finally:
    if clip:
        clip.close()
    if processed and processed != clip:
        processed.close()
    gc.collect()
```

### 3. Focus on These Classes/Methods
This file contains multiple classes. Fix memory leaks in:

**CinematicTransitions class:**
- `apply()` method - close transition clips
- All transition effect methods

**DynamicTextAnimations class:**
- `create()` method - close text clips
- All animation methods

**VisualBeatSync class:**
- `apply_visual_sync()` - close synced clips

**ColorGradingPresets class:**
- `apply()` method - close processed clips

**MotionGraphicsLibrary class:**
- `lower_third()` - close graphics clips
- All motion graphics methods

**AdaptivePacingEngine class:**
- `optimize()` - close paced clips

### 4. Cleanup Pattern for Class Methods
```python
def apply(self, clip, ...):
    intermediate_clips = []
    try:
        # ... processing ...
        result = some_operation(clip)
        intermediate_clips.append(result)
        return final_result
    finally:
        for c in intermediate_clips:
            if c and c != clip:  # Don't close input clip
                c.close()
        gc.collect()
```

### 5. Special Attention
- Don't close clips passed as arguments (caller owns them)
- Only close clips created within the method
- Close intermediate clips even if an exception occurs

## Testing
```bash
cd "C:\Users\fkozi\youtube-automation"
python -c "
from src.content.pro_video_engine import ProVideoEngine
engine = ProVideoEngine()
print('Memory leak fixes applied successfully')
"
```

## Expected Outcome
- All created VideoClip objects have explicit .close() calls
- try/finally blocks protect against exceptions
- gc.collect() called after operations
- No breaking changes to API
- Multiple operations can run without memory exhaustion

## Deliverable
Modified C:\Users\fkozi\youtube-automation\src\content\pro_video_engine.py with memory leak fixes.
