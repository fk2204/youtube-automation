#!/usr/bin/env python3
"""
Upload video to any of your YouTube channels
"""

import sys
import os
import json

project_dir = r'C:\Users\fkozi\joe'
os.chdir(project_dir)
sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv
load_dotenv('.env')

def main():
    print("\n" + "="*70)
    print("UPLOAD TO YOUR YOUTUBE CHANNELS")
    print("="*70)

    # Load channels config
    with open('config/channels_config.json', 'r') as f:
        config = json.load(f)

    channels = config['channels']

    # Show available channels
    print("\n[AVAILABLE CHANNELS]\n")
    for i, ch in enumerate(channels, 1):
        print(f"  {i}. {ch['name']}")
        print(f"     ID: {ch['channel_id']}")
        print(f"     Niche: {ch['niche']}\n")

    # Ask user which channel
    print("[SELECT CHANNEL]\n")
    try:
        choice = input("Enter channel number (1-3): ").strip()
        channel_idx = int(choice) - 1

        if channel_idx < 0 or channel_idx >= len(channels):
            print("[ERROR] Invalid choice")
            return False

        selected = channels[channel_idx]
        print(f"\n[OK] Selected: {selected['name']}")

    except (ValueError, IndexError):
        print("[ERROR] Invalid input")
        return False

    # Check if video exists
    print("\n[CHECKING] Video file...")
    video_file = 'output/video.mp4'

    if not os.path.exists(video_file):
        print(f"[ERROR] Video not found: {video_file}")
        print(f"        Run: python3 run_full_pipeline_demo.py")
        return False

    size_mb = os.path.getsize(video_file) / 1024 / 1024
    print(f"[OK] Found: {video_file} ({size_mb:.1f} MB)")

    # Get YouTube service
    print(f"\n[AUTHENTICATING] YouTube...")
    try:
        from src.youtube.auth_oob import YouTubeAuthOOB

        auth = YouTubeAuthOOB()
        youtube = auth.get_authenticated_service()
        print(f"[OK] Authenticated!")

    except KeyboardInterrupt:
        print(f"[CANCELLED]")
        return False
    except Exception as e:
        print(f"[ERROR] Auth failed: {e}")
        return False

    # Upload
    print(f"\n[UPLOADING] to {selected['name']}...")

    try:
        request_body = {
            'snippet': {
                'title': '5 Ways to Make Passive Income with AI in 2026',
                'description': f"{selected['description']}\n\nGenerated with Joe - AI Content Automation",
                'tags': ['AI', 'money', 'tutorial', selected['niche']],
                'categoryId': '22'  # Education
            },
            'status': {
                'privacyStatus': 'unlisted'
            }
        }

        request = youtube.videos().insert(
            part='snippet,status',
            body=request_body,
            media_body=video_file
        )

        print(f"[UPLOADING...] This may take 1-2 minutes")
        response = request.execute()

        video_id = response['id']
        video_url = f'https://youtu.be/{video_id}'

        print(f"\n[OK] UPLOAD SUCCESSFUL!")
        print(f"     Channel: {selected['name']}")
        print(f"     Video URL: {video_url}")
        print(f"\n     Next: Generate another video to upload to another channel!")

        return True

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()

    print("\n" + "="*70)
    if success:
        print("Done! You can now:")
        print("  1. Run: python3 run_full_pipeline_demo.py  (to generate another video)")
        print("  2. Run: python3 upload_to_channel.py       (to choose which channel to upload to)")
    else:
        print("Upload failed")
    print("="*70 + "\n")

    sys.exit(0 if success else 1)
