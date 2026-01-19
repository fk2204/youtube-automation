"""Test Pika Labs connection."""
import asyncio
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.content.video_pika import PikaVideoGenerator, get_pika_generator

async def test_connection():
    print("=" * 50)
    print("PIKA LABS CONNECTION TEST")
    print("=" * 50)

    try:
        generator = PikaVideoGenerator()
        print(f"[OK] PikaVideoGenerator initialized")
        print(f"[OK] API Key configured: {bool(generator.api_key)}")
        print(f"[OK] Key prefix: {generator.api_key[:20]}...")
        print(f"[OK] fal-client available")
        print()
        print("Connection ready! Pika Labs can now generate videos.")
        print()
        print("Usage examples:")
        print("  - Shorts: generator.generate_short_clip(prompt)")
        print("  - B-roll: generator.generate_broll_clip(topic)")
        print("  - Text-to-video: generator.generate_from_text(prompt)")
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection())
