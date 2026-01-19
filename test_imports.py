"""Quick import test for YouTube automation modules."""
import sys

print("Testing imports...")

try:
    from src.content.video_pika import PikaVideoGenerator, get_pika_generator
    print("[OK] video_pika")
except Exception as e:
    print(f"[FAIL] video_pika: {e}")

try:
    from src.content.tts import get_tts_provider
    print("[OK] tts")
except Exception as e:
    print(f"[FAIL] tts: {e}")

try:
    from src.content.video_fast import FastVideoGenerator
    print("[OK] video_fast")
except Exception as e:
    print(f"[FAIL] video_fast: {e}")

try:
    from src.content.script_writer import ScriptWriter
    print("[OK] script_writer")
except Exception as e:
    print(f"[FAIL] script_writer: {e}")

try:
    from src.content.audio_processor import AudioProcessor
    print("[OK] audio_processor")
except Exception as e:
    print(f"[FAIL] audio_processor: {e}")

try:
    from src.utils.token_manager import TokenTracker
    print("[OK] token_manager")
except Exception as e:
    print(f"[FAIL] token_manager: {e}")

try:
    from src.content.stock_footage import StockFootageManager
    print("[OK] stock_footage")
except Exception as e:
    print(f"[FAIL] stock_footage: {e}")

print("\nAll imports tested!")
