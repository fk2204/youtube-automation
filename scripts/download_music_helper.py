#!/usr/bin/env python3
"""
Music Download Helper Script
Opens royalty-free music search pages in your browser for each channel.

Usage:
    python scripts/download_music_helper.py

This script will open Pixabay Music search pages for each channel niche.
Download the tracks you like and save them to assets/music/ with the
appropriate filename (finance.mp3, psychology.mp3, storytelling.mp3, background.mp3).
"""

import webbrowser
import time
import os
from pathlib import Path

# Music search URLs for each channel
MUSIC_SEARCHES = {
    "finance": {
        "description": "Corporate/Business Background Music",
        "recommended": "Upbeat, professional, motivating tracks",
        "urls": [
            "https://pixabay.com/music/search/corporate%20ambient/",
            "https://pixabay.com/music/search/business%20background/",
            "https://pixabay.com/music/search/motivational%20corporate/",
        ]
    },
    "psychology": {
        "description": "Calm/Ambient Background Music",
        "recommended": "Calm, contemplative, atmospheric tracks",
        "urls": [
            "https://pixabay.com/music/search/calm%20ambient/",
            "https://pixabay.com/music/search/meditation%20background/",
            "https://pixabay.com/music/search/relaxing%20atmospheric/",
        ]
    },
    "storytelling": {
        "description": "Cinematic/Dramatic Background Music",
        "recommended": "Dramatic, suspenseful, engaging tracks",
        "urls": [
            "https://pixabay.com/music/search/cinematic%20tension/",
            "https://pixabay.com/music/search/documentary%20mysterious/",
            "https://pixabay.com/music/search/dark%20suspense/",
        ]
    },
    "background": {
        "description": "Generic Background Music (Fallback)",
        "recommended": "Neutral, versatile, non-distracting tracks",
        "urls": [
            "https://pixabay.com/music/search/background%20ambient/",
            "https://pixabay.com/music/search/soft%20electronic/",
        ]
    }
}

# Alternative free music sources
ALTERNATIVE_SOURCES = [
    ("YouTube Audio Library", "https://studio.youtube.com/channel/UC/music"),
    ("Mixkit Free Music", "https://mixkit.co/free-stock-music/"),
    ("Uppbeat (Free tier)", "https://uppbeat.io/browse/music"),
    ("Free Music Archive", "https://freemusicarchive.org/search?quicksearch=ambient"),
]


def get_music_directory():
    """Get the path to the music assets directory."""
    script_dir = Path(__file__).parent
    music_dir = script_dir.parent / "assets" / "music"
    return music_dir


def check_existing_music():
    """Check which music files already exist."""
    music_dir = get_music_directory()
    files = {
        "finance": music_dir / "finance.mp3",
        "psychology": music_dir / "psychology.mp3",
        "storytelling": music_dir / "storytelling.mp3",
        "background": music_dir / "background.mp3",
    }

    print("\n=== Current Music Status ===\n")
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [EXISTS] {name}.mp3 ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {name}.mp3")
    print()


def open_search_pages(channel: str = None):
    """Open music search pages in the browser."""
    channels_to_open = [channel] if channel else list(MUSIC_SEARCHES.keys())

    for ch in channels_to_open:
        if ch not in MUSIC_SEARCHES:
            print(f"Unknown channel: {ch}")
            continue

        info = MUSIC_SEARCHES[ch]
        print(f"\n=== {ch.upper()} Channel ===")
        print(f"Description: {info['description']}")
        print(f"Look for: {info['recommended']}")
        print(f"Save as: assets/music/{ch}.mp3")
        print("\nOpening search pages...")

        for url in info['urls']:
            print(f"  -> {url}")
            webbrowser.open(url)
            time.sleep(1)  # Small delay between opening tabs

        print()


def show_instructions():
    """Show download instructions."""
    print("""
=== DOWNLOAD INSTRUCTIONS ===

1. For each search page that opens:
   - Preview tracks by clicking the play button
   - Look for tracks that match the recommended style
   - Click the green "Download" button
   - Choose MP3 format

2. After downloading, rename the files:
   - Finance channel -> finance.mp3
   - Psychology channel -> psychology.mp3
   - Storytelling channel -> storytelling.mp3
   - Generic fallback -> background.mp3

3. Move the renamed files to:
   assets/music/

4. The video generator will automatically use them!

TIPS:
- Choose tracks at least 3 minutes long (they will loop)
- Avoid tracks with vocals
- Test with a sample video to ensure the mood is right
- Pixabay tracks are free, no attribution required
""")


def show_alternative_sources():
    """Show alternative music sources."""
    print("\n=== ALTERNATIVE SOURCES ===\n")
    for name, url in ALTERNATIVE_SOURCES:
        print(f"  {name}:")
        print(f"    {url}\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("   YOUTUBE AUTOMATION - MUSIC DOWNLOAD HELPER")
    print("=" * 60)

    check_existing_music()
    show_instructions()

    print("\nWhat would you like to do?\n")
    print("  1. Open ALL search pages (recommended)")
    print("  2. Open Finance channel music")
    print("  3. Open Psychology channel music")
    print("  4. Open Storytelling channel music")
    print("  5. Open Background (fallback) music")
    print("  6. Show alternative music sources")
    print("  0. Exit")

    try:
        choice = input("\nEnter choice (1-6, or 0 to exit): ").strip()

        if choice == "1":
            open_search_pages()
        elif choice == "2":
            open_search_pages("finance")
        elif choice == "3":
            open_search_pages("psychology")
        elif choice == "4":
            open_search_pages("storytelling")
        elif choice == "5":
            open_search_pages("background")
        elif choice == "6":
            show_alternative_sources()
        elif choice == "0":
            print("Goodbye!")
        else:
            print("Invalid choice. Running default (open all)...")
            open_search_pages()

    except KeyboardInterrupt:
        print("\n\nCancelled.")
    except EOFError:
        # Non-interactive mode - open all pages
        print("Running in non-interactive mode. Opening all search pages...")
        open_search_pages()


if __name__ == "__main__":
    main()
