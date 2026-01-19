"""Test hybrid short generation with Pika intro/outro."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.content.shorts_hybrid import HybridShortsGenerator

async def test_hybrid_short():
    print("=" * 60)
    print("HYBRID SHORT GENERATION TEST")
    print("=" * 60)
    print()
    print("This will generate a SHORT with:")
    print("  - Pika AI intro (5s) - $0.20")
    print("  - Stock footage middle (~20s) - FREE")
    print("  - Pika AI outro (5s) - $0.20")
    print("  - Total cost: ~$0.40")
    print()

    generator = HybridShortsGenerator()

    # Check if we have test audio, if not use existing audio or skip
    test_audio = "output/test_voice.mp3"

    if not os.path.exists(test_audio):
        # Try to find any existing audio file
        for f in os.listdir("output"):
            if f.endswith(".mp3") and "audio" in f.lower():
                test_audio = os.path.join("output", f)
                break

    if not os.path.exists(test_audio):
        print("No test audio found. Creating a simple test...")
        # Generate test audio with Edge-TTS
        try:
            from src.content.tts import TextToSpeech
            tts = TextToSpeech()
            test_text = """
            Did you know that your brain makes decisions before you're even aware of them?
            Studies show that neural activity begins up to 7 seconds before you consciously decide.
            This means your subconscious mind is running the show most of the time.
            Subscribe for more mind-blowing psychology facts!
            """
            await tts.generate(test_text.strip(), test_audio)
            print(f"Generated test audio: {test_audio}")
        except Exception as e:
            print(f"Could not create test audio: {e}")
            return

    print(f"Using audio: {test_audio}")
    print()
    print("Generating hybrid short (this takes 3-5 minutes)...")
    print()

    result = await generator.create_hybrid_short(
        audio_file=test_audio,
        output_file="output/test_hybrid_short.mp4",
        niche="psychology",
        topic="Psychology facts about decision making"
    )

    print()
    print("=" * 60)
    if result.success:
        print("SUCCESS!")
        print("=" * 60)
        print(f"Output: {result.output_path}")
        print(f"Duration: {result.duration:.1f}s")
        print(f"Pika cost: ${result.pika_cost:.2f}")
        if result.intro_path:
            print(f"Intro: {result.intro_path}")
        if result.outro_path:
            print(f"Outro: {result.outro_path}")
    else:
        print("FAILED")
        print("=" * 60)
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_hybrid_short())
