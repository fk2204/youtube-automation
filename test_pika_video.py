"""Generate a test video with Pika Labs."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.content.video_pika import PikaVideoGenerator

async def generate_test():
    print("=" * 50)
    print("PIKA LABS TEST VIDEO GENERATION")
    print("=" * 50)
    print()

    generator = PikaVideoGenerator()

    # Test prompt - cinematic and simple
    prompt = "A person sitting alone, deep in thought, soft natural lighting, cinematic"

    print(f"Prompt: {prompt}")
    print(f"Duration: 5 seconds")
    print(f"Resolution: 720p")
    print(f"Estimated cost: $0.20")
    print()
    print("Generating... (this takes 1-3 minutes)")
    print()

    result = await generator.generate_from_text(
        prompt=prompt,
        output_file="output/pika_test_video.mp4",
        duration=5,
        resolution="720p",
        aspect_ratio="16:9"
    )

    if result.success:
        print()
        print("=" * 50)
        print("SUCCESS!")
        print("=" * 50)
        print(f"Video URL: {result.video_url}")
        print(f"Local file: {result.local_path}")
        print(f"Duration: {result.duration}s")
        print(f"Cost: ${result.cost_estimate:.2f}")
    else:
        print()
        print("=" * 50)
        print("FAILED")
        print("=" * 50)
        print(f"Error: {result.error}")

    return result

if __name__ == "__main__":
    asyncio.run(generate_test())
