#!/usr/bin/env python
"""
Quick automation launcher.

Usage:
    python run.py video money_blueprints
    python run.py video money_blueprints --no-upload  # Create only, skip upload
    python run.py video mind_unlocked -n              # Short form of --no-upload
    python run.py video untold_stories
    python run.py short money_blueprints     # Create YouTube Short
    python run.py short money_blueprints --no-upload  # Create Short only, skip upload
    python run.py batch 3
    python run.py batch-all
    python run.py schedule-shorts            # Run Shorts scheduler only
    python run.py daily-all                  # Run both videos and Shorts scheduler
"""

import sys
import os
import argparse

# Add project root
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv("config/.env")

def main():
    if len(sys.argv) < 2:
        print("""
YouTube Automation Quick Launcher
=================================

Commands:
  python run.py video <channel>     Create & upload 1 regular video
  python run.py video <channel> -n  Create video only (skip upload)
  python run.py short <channel>     Create & upload 1 YouTube Short (vertical)
  python run.py short <channel> -n  Create Short only (skip upload)
  python run.py test <channel>      Test regular video creation (no upload)
  python run.py test-short <channel> Test Short creation (no upload)
  python run.py batch <count>       Create <count> videos for all channels
  python run.py batch-all           Create 1 video for each channel
  python run.py batch --channels <list> --count <n> --parallel <workers>
                                    Parallel batch processing (3x throughput)
  python run.py validate-script <f> Validate a script file for quality
  python run.py add-subtitles <video> <script>  Add burned-in subtitles to video
  python run.py thumbnail "<title>" --niche <niche>  Generate viral thumbnail

Flags:
  -n, --no-upload                   Skip upload after video creation

Scheduler Commands:
  python run.py schedule-shorts     Run Shorts scheduler only
  python run.py schedule-videos     Run regular videos scheduler only
  python run.py daily-all           Run both videos and Shorts scheduler
  python run.py status              Show scheduler status

Disk Cleanup Commands:
  python run.py cleanup              Clean files older than 30 days
  python run.py cleanup --days 7     Clean files older than 7 days
  python run.py cleanup --dry-run    Preview what would be deleted
  python run.py disk-usage           Show disk usage stats

Stock Footage Cache:
  python run.py cache-stats          Show cache statistics & savings
  python run.py cache-stats --cleanup  Clean old cache files (> 30 days)
  python run.py cache-stats --cleanup --days 7   Clean older than 7 days
  python run.py cache-stats --clear  Clear all cached footage

Video Formats:
  video    - Regular 1920x1080 horizontal video (5-10 min)
  short    - YouTube Shorts 1080x1920 vertical video (15-60 sec)

Channels:
  money_blueprints    Finance content
  mind_unlocked       Psychology content
  untold_stories      Storytelling content

Shorts Schedule:
  - Shorts are posted 2-3 hours after each regular video
  - Configure in config/channels.yaml under shorts_schedule
  - Each channel can have custom delay_hours and standalone_times

Script Validation:
  python run.py validate-script <file> [--niche <niche>] [--clean] [--improve]
    --niche     Content niche: finance, psychology, storytelling, default
    --clean     Output cleaned script
    --improve   Output improved script
    --short     Validate as YouTube Short

Agent Commands:
  python run.py agent                         Show available agents
  python run.py agent seo-strategist <cmd>    World-class SEO strategy
  python run.py agent branding all            Generate profile pictures

Subtitles:
  python run.py add-subtitles <video> <script> [options]
    --output    Output file path (default: adds _subtitled suffix)
    --style     Subtitle style: regular, shorts, minimal, cinematic
    --niche     Content niche for styling: finance, psychology, storytelling
    --audio     Audio file for better timing (uses Whisper if available)

Thumbnails:
  python run.py thumbnail "<title>" [options]
    --niche     Content niche: finance, psychology, storytelling (required)
    --output    Output file path (default: assets/thumbnails/)
    --position  Text position: center, top, bottom (default: center)
    --variants  Generate N variants for A/B testing
    --background  Background image path (optional)
    --vignette  Vignette strength 0.0-1.0 (default: 0.7)

Examples:
  python run.py video money_blueprints
  python run.py video money_blueprints --no-upload  # Create only
  python run.py short money_blueprints
  python run.py short mind_unlocked -n              # Create Short only
  python run.py test-short mind_unlocked
  python run.py batch 3
  python run.py daily-all              # Start full scheduler
  python run.py schedule-shorts        # Shorts only
  python run.py validate-script script.txt --niche finance
  python run.py thumbnail "How I Made $10,000" --niche finance
  python run.py thumbnail "Dark Psychology Secrets" --niche psychology --variants 3
        """)
        return

    cmd = sys.argv[1]

    if cmd == "video":
        # Parse arguments for video command
        video_parser = argparse.ArgumentParser(prog="run.py video", add_help=False)
        video_parser.add_argument("channel", nargs="?", default="money_blueprints")
        video_parser.add_argument("-n", "--no-upload", action="store_true",
                                  help="Skip upload after video creation")
        video_args = video_parser.parse_args(sys.argv[2:])

        channel = video_args.channel
        no_upload = video_args.no_upload

        if no_upload:
            # Create video only, skip upload
            from src.automation.runner import task_full_pipeline
            result = task_full_pipeline(channel)
            if result["success"]:
                video_path = result['results'].get('video_file', 'N/A')
                print(f"\n[OK] Video created at: {video_path}. Skipping upload (--no-upload)")
            else:
                print(f"\n[FAIL] Failed: {result.get('error')}")
        else:
            # Default behavior: create and upload
            from src.automation.runner import task_full_with_upload
            result = task_full_with_upload(channel)
            if result["success"]:
                print(f"\n[OK] Video uploaded: {result['results'].get('video_url', 'N/A')}")
            else:
                print(f"\n[FAIL] Failed: {result.get('error')}")

    elif cmd == "short":
        # Parse arguments for short command
        short_parser = argparse.ArgumentParser(prog="run.py short", add_help=False)
        short_parser.add_argument("channel", nargs="?", default="money_blueprints")
        short_parser.add_argument("topic", nargs="?", default=None)
        short_parser.add_argument("-n", "--no-upload", action="store_true",
                                  help="Skip upload after Short creation")
        short_args = short_parser.parse_args(sys.argv[2:])

        channel = short_args.channel
        topic = short_args.topic
        no_upload = short_args.no_upload

        if no_upload:
            # Create Short only, skip upload
            from src.automation.runner import task_short_pipeline
            result = task_short_pipeline(channel, topic)
            if result["success"]:
                video_path = result['results'].get('video_file', 'N/A')
                print(f"\n[OK] YouTube Short created at: {video_path}. Skipping upload (--no-upload)")
                print(f"    Format: 1080x1920 vertical (9:16)")
                print(f"    Duration: 15-60 seconds")
            else:
                print(f"\n[FAIL] Failed: {result.get('error')}")
        else:
            # Default behavior: create and upload
            from src.automation.runner import task_short_with_upload
            result = task_short_with_upload(channel, topic)
            if result["success"]:
                print(f"\n[OK] YouTube Short uploaded: {result['results'].get('video_url', 'N/A')}")
            else:
                print(f"\n[FAIL] Failed: {result.get('error')}")

    elif cmd == "batch":
        # Parse batch arguments
        batch_parser = argparse.ArgumentParser(prog="run.py batch", add_help=False)
        batch_parser.add_argument("count", nargs="?", type=int, default=1,
                                  help="Videos per channel (positional, for backwards compat)")
        batch_parser.add_argument("--channels", help="Comma-separated list of channel IDs")
        batch_parser.add_argument("--count", "-c", type=int, dest="count_flag",
                                  help="Videos per channel")
        batch_parser.add_argument("--parallel", "-p", type=int, default=0,
                                  help="Parallel workers (0=sequential, 3=recommended)")
        batch_parser.add_argument("--no-upload", "-n", action="store_true",
                                  help="Skip upload after video creation")
        batch_parser.add_argument("--output", "-o", help="Save results to JSON file")
        batch_args = batch_parser.parse_args(sys.argv[2:])

        # Determine count (flag takes precedence over positional)
        video_count = batch_args.count_flag if batch_args.count_flag else batch_args.count

        # Determine channels
        if batch_args.channels:
            channels = [ch.strip() for ch in batch_args.channels.split(",")]
        else:
            channels = None  # All enabled channels

        # Run batch (parallel or sequential)
        if batch_args.parallel > 0:
            from src.automation.batch import run_batch_parallel
            print(f"\n[INFO] Starting PARALLEL batch processing with {batch_args.parallel} workers...")
            results = run_batch_parallel(
                channels=channels,
                videos_per_channel=video_count,
                upload=not batch_args.no_upload,
                max_workers=batch_args.parallel
            )
        else:
            from src.automation.batch import run_batch
            results = run_batch(
                channels=channels,
                videos_per_channel=video_count,
                upload=not batch_args.no_upload
            )

        # Save results if requested
        if batch_args.output:
            import json
            with open(batch_args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n[OK] Results saved: {batch_args.output}")

    elif cmd == "batch-all":
        from src.automation.batch import run_batch
        run_batch(channels=None, videos_per_channel=1, upload=True)

    elif cmd == "test":
        # Test regular video creation (no upload)
        channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
        from src.automation.runner import task_full_pipeline
        result = task_full_pipeline(channel)
        if result["success"]:
            print(f"\n[OK] Video created: {result['results'].get('video_file')}")
        else:
            print(f"\n[FAIL] Failed: {result.get('error')}")

    elif cmd == "test-short":
        # Test YouTube Short creation (no upload)
        channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
        topic = sys.argv[3] if len(sys.argv) > 3 else None
        from src.automation.runner import task_short_pipeline
        result = task_short_pipeline(channel, topic)
        if result["success"]:
            print(f"\n[OK] YouTube Short created: {result['results'].get('video_file')}")
            print(f"    Format: 1080x1920 vertical (9:16)")
            print(f"    Duration: 15-60 seconds")
        else:
            print(f"\n[FAIL] Failed: {result.get('error')}")

    elif cmd == "schedule-shorts":
        # Run Shorts scheduler only
        print("\n[INFO] Starting Shorts scheduler...")
        print("       Shorts will be posted after regular videos based on config/channels.yaml")
        from src.scheduler.daily_scheduler import run_scheduler
        run_scheduler(include_videos=False, include_shorts=True)

    elif cmd == "schedule-videos":
        # Run regular videos scheduler only (backwards compatible)
        print("\n[INFO] Starting regular videos scheduler...")
        from src.scheduler.daily_scheduler import run_scheduler
        run_scheduler(include_videos=True, include_shorts=False)

    elif cmd == "daily-all":
        # Run both regular videos and Shorts scheduler
        print("\n[INFO] Starting full scheduler (videos + Shorts)...")
        print("       Regular videos will be posted at scheduled times")
        print("       Shorts will be posted 2-3 hours after each regular video")
        from src.scheduler.daily_scheduler import run_scheduler
        run_scheduler(include_videos=True, include_shorts=True)

    elif cmd == "status":
        # Show scheduler status
        from src.scheduler.daily_scheduler import show_status
        show_status()

    elif cmd == "cost":
        # Show token usage and cost report
        from src.utils.token_manager import print_usage_report
        print_usage_report()

    elif cmd == "cache-stats":
        # Show stock footage cache statistics
        cache_parser = argparse.ArgumentParser(prog="run.py cache-stats", add_help=False)
        cache_parser.add_argument("--cleanup", action="store_true",
                                  help="Clean up cache files older than --days")
        cache_parser.add_argument("--days", type=int, default=30,
                                  help="Maximum age for cache files (default: 30)")
        cache_parser.add_argument("--clear", action="store_true",
                                  help="Clear all cached files")
        cache_args = cache_parser.parse_args(sys.argv[2:])

        from src.content.stock_cache import StockCache

        cache = StockCache()

        if cache_args.clear:
            print("\n[WARNING] This will delete ALL cached stock footage.")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                files, bytes_freed = cache.clear_cache()
                print(f"\n[OK] Cache cleared: {files} files, {bytes_freed / 1024 / 1024:.1f} MB freed")
            else:
                print("[CANCELLED] Cache not cleared.")

        elif cache_args.cleanup:
            print(f"\n[INFO] Cleaning cache files older than {cache_args.days} days...")
            files, bytes_freed = cache.cleanup_old_files(cache_args.days)
            print(f"\n[OK] Cleanup complete: {files} files removed, {bytes_freed / 1024 / 1024:.1f} MB freed")

        else:
            # Show stats
            stats = cache.get_stats()
            print(stats.summary())

            # Savings estimate
            savings = cache.estimate_savings()
            print("\n  Estimated Savings:")
            print(f"    Total cache hits: {savings['total_cache_hits']}")
            print(f"    Time saved: {savings['estimated_time_saved_minutes']:.1f} minutes")
            print(f"    Bandwidth saved: {savings['estimated_bandwidth_saved_mb']:.1f} MB")
            print()

    elif cmd == "cleanup":
        # Clean up old files to free disk space
        cleanup_parser = argparse.ArgumentParser(prog="run.py cleanup", add_help=False)
        cleanup_parser.add_argument("--days", type=int, default=30,
                                    help="Delete files older than N days (default: 30)")
        cleanup_parser.add_argument("--dry-run", action="store_true",
                                    help="Preview what would be deleted without deleting")
        cleanup_parser.add_argument("-q", "--quiet", action="store_true",
                                    help="Less verbose output")
        cleanup_args = cleanup_parser.parse_args(sys.argv[2:])

        from src.utils.cleanup import cleanup_old_files, print_cleanup_report

        print(f"\n{'='*60}")
        if cleanup_args.dry_run:
            print("  DISK CLEANUP PREVIEW (DRY RUN)")
        else:
            print("  DISK CLEANUP")
        print(f"{'='*60}")
        print(f"  Cleaning files older than {cleanup_args.days} days...")
        print()

        result = cleanup_old_files(
            max_age_days=cleanup_args.days,
            dry_run=cleanup_args.dry_run,
            verbose=not cleanup_args.quiet
        )
        print_cleanup_report(result)

    elif cmd == "disk-usage":
        # Show disk usage statistics
        from src.utils.cleanup import print_disk_usage_report
        print_disk_usage_report()

    elif cmd == "validate-script":
        # Validate a script file
        script_parser = argparse.ArgumentParser(prog="run.py validate-script", add_help=False)
        script_parser.add_argument("file", nargs="?", help="Script file to validate")
        script_parser.add_argument("--niche", default="default",
                                   choices=["finance", "psychology", "storytelling", "default"],
                                   help="Content niche for validation context")
        script_parser.add_argument("--clean", action="store_true", help="Output cleaned script")
        script_parser.add_argument("--improve", action="store_true", help="Output improved script")
        script_parser.add_argument("--short", action="store_true", help="Validate as YouTube Short")
        script_parser.add_argument("--json", action="store_true", help="Output as JSON")
        script_args = script_parser.parse_args(sys.argv[2:])

        if not script_args.file:
            print("Error: Please provide a script file to validate")
            print("Usage: python run.py validate-script <file> [--niche <niche>] [--clean] [--improve]")
            return

        import os
        if not os.path.exists(script_args.file):
            print(f"Error: File not found: {script_args.file}")
            return

        try:
            with open(script_args.file, 'r', encoding='utf-8') as f:
                script_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        from src.content.script_validator import ScriptValidator
        import json

        validator = ScriptValidator()

        print(f"\n{'='*60}")
        print(f"Script Validation: {script_args.file}")
        print(f"Niche: {script_args.niche}")
        print(f"Format: {'YouTube Short' if script_args.short else 'Regular Video'}")
        print(f"{'='*60}\n")

        # Always validate first
        result = validator.validate_script(
            script_text,
            niche=script_args.niche,
            is_short=script_args.short
        )

        if script_args.json:
            output = {"validation": result.to_dict()}
        else:
            print(result.summary())
            print()

        # Clean if requested
        if script_args.clean:
            cleaned = validator.clean_script(script_text)
            if script_args.json:
                output["cleaned_script"] = cleaned
            else:
                print(f"\n{'='*60}")
                print("CLEANED SCRIPT:")
                print(f"{'='*60}")
                print(cleaned)
                print()

        # Improve if requested
        if script_args.improve:
            improved = validator.improve_script(script_text, niche=script_args.niche)
            if script_args.json:
                output["improved_script"] = improved
            else:
                print(f"\n{'='*60}")
                print("IMPROVED SCRIPT:")
                print(f"{'='*60}")
                print(improved)
                print()

        # Output JSON if requested
        if script_args.json:
            print(json.dumps(output, indent=2))

        # Return exit code based on validation result
        if not result.is_valid:
            print(f"\n[!] Script validation FAILED (score: {result.score}/100)")
            sys.exit(1)
        else:
            print(f"\n[OK] Script validation PASSED (score: {result.score}/100)")

    elif cmd == "add-subtitles":
        # Add subtitles to a video
        sub_parser = argparse.ArgumentParser(prog="run.py add-subtitles", add_help=False)
        sub_parser.add_argument("video", nargs="?", help="Video file to add subtitles to")
        sub_parser.add_argument("script", nargs="?", help="Script/text file for subtitles")
        sub_parser.add_argument("--output", "-o", help="Output file path (default: adds _subtitled suffix)")
        sub_parser.add_argument("--style", default="regular",
                               choices=["regular", "shorts", "minimal", "cinematic"],
                               help="Subtitle style (default: regular)")
        sub_parser.add_argument("--niche", default=None,
                               choices=["finance", "psychology", "storytelling", None],
                               help="Content niche for styling")
        sub_parser.add_argument("--audio", help="Audio file for better timing (optional)")
        sub_args = sub_parser.parse_args(sys.argv[2:])

        if not sub_args.video:
            print("Error: Please provide a video file")
            print("Usage: python run.py add-subtitles <video> <script> [--output <out>] [--style <style>]")
            return

        if not sub_args.script:
            print("Error: Please provide a script/text file")
            print("Usage: python run.py add-subtitles <video> <script> [--output <out>] [--style <style>]")
            return

        import os
        if not os.path.exists(sub_args.video):
            print(f"Error: Video file not found: {sub_args.video}")
            return

        if not os.path.exists(sub_args.script):
            print(f"Error: Script file not found: {sub_args.script}")
            return

        # Read script text
        try:
            with open(sub_args.script, 'r', encoding='utf-8') as f:
                script_text = f.read()
        except Exception as e:
            print(f"Error reading script file: {e}")
            return

        # Generate output path if not provided
        output_path = sub_args.output
        if not output_path:
            from pathlib import Path
            video_path = Path(sub_args.video)
            output_path = str(video_path.with_stem(f"{video_path.stem}_subtitled"))

        print(f"\n{'='*60}")
        print(f"Adding Subtitles to Video")
        print(f"{'='*60}")
        print(f"Video: {sub_args.video}")
        print(f"Script: {sub_args.script}")
        print(f"Output: {output_path}")
        print(f"Style: {sub_args.style}")
        print(f"Niche: {sub_args.niche or 'default'}")
        print(f"{'='*60}\n")

        from src.content.subtitles import SubtitleGenerator

        generator = SubtitleGenerator()

        # Add subtitles to video
        result = generator.add_subtitles_to_video(
            video_path=sub_args.video,
            script=script_text,
            output_path=output_path,
            audio_path=sub_args.audio,
            style=sub_args.style,
            niche=sub_args.niche,
            use_transcription=True
        )

        if result:
            file_size = os.path.getsize(result) / (1024 * 1024)
            print(f"\n[OK] Subtitles added successfully!")
            print(f"     Output: {result}")
            print(f"     Size: {file_size:.1f} MB")
        else:
            print(f"\n[FAIL] Failed to add subtitles")
            sys.exit(1)

    elif cmd == "thumbnail":
        # Generate YouTube thumbnail
        thumb_parser = argparse.ArgumentParser(prog="run.py thumbnail", add_help=False)
        thumb_parser.add_argument("title", nargs="?", help="Video title for thumbnail")
        thumb_parser.add_argument("--niche", required=True,
                                 choices=["finance", "psychology", "storytelling", "default"],
                                 help="Content niche for color scheme")
        thumb_parser.add_argument("--output", "-o", help="Output file path")
        thumb_parser.add_argument("--position", default="center",
                                 choices=["center", "top", "bottom"],
                                 help="Text position (default: center)")
        thumb_parser.add_argument("--variants", type=int, help="Generate N variants for A/B testing")
        thumb_parser.add_argument("--background", help="Background image path")
        thumb_parser.add_argument("--vignette", type=float, default=0.7,
                                 help="Vignette strength 0.0-1.0 (default: 0.7)")
        thumb_args = thumb_parser.parse_args(sys.argv[2:])

        if not thumb_args.title:
            print("Error: Please provide a video title")
            print('Usage: python run.py thumbnail "My Video Title" --niche finance')
            return

        print(f"\n{'='*60}")
        print(f"YouTube Thumbnail Generator")
        print(f"{'='*60}")
        print(f"Title: {thumb_args.title}")
        print(f"Niche: {thumb_args.niche}")
        print(f"Position: {thumb_args.position}")
        if thumb_args.variants:
            print(f"Variants: {thumb_args.variants}")
        print(f"{'='*60}\n")

        from src.content.thumbnail_generator import ThumbnailGenerator

        generator = ThumbnailGenerator()

        if thumb_args.variants:
            paths = generator.generate_variants(
                title=thumb_args.title,
                niche=thumb_args.niche,
                count=thumb_args.variants
            )
            print(f"\n[OK] Generated {len(paths)} thumbnail variants:")
            for path in paths:
                print(f"     - {path}")
        else:
            path = generator.generate(
                title=thumb_args.title,
                niche=thumb_args.niche,
                output_path=thumb_args.output,
                background_image=thumb_args.background,
                text_position=thumb_args.position,
                vignette_strength=thumb_args.vignette
            )
            print(f"\n[OK] Thumbnail generated: {path}")
            print(f"     Resolution: 1280x720 (YouTube standard)")
            print(f"     Color scheme: {thumb_args.niche}")

    elif cmd == "agent":
        # Agent dispatcher for specialized tasks
        if len(sys.argv) < 3:
            print("""
Agent Dispatcher - Specialized Task Agents
==========================================

Usage:
    python run.py agent <agent-name> <command> [options]

Available Agents:
    seo-strategist    World-class SEO optimization and strategy
    seo               Basic SEO metadata optimization
    branding          Channel branding and profile pictures

SEO Strategist Commands:
    python run.py agent seo-strategist research "<keyword>" --niche <niche>
    python run.py agent seo-strategist strategy --niche <niche> --topics N
    python run.py agent seo-strategist ab-test "<title>" --variants N
    python run.py agent seo-strategist competitors "<keyword>" --top N
    python run.py agent seo-strategist calendar --niche <niche> --weeks N
    python run.py agent seo-strategist optimize --file <path>

Branding Commands:
    python run.py agent branding all                    Generate all channel assets
    python run.py agent branding <channel> <niche>      Generate assets for one channel

Examples:
    python run.py agent seo-strategist research "passive income" --niche finance
    python run.py agent seo-strategist ab-test "How to Make Money" --variants 5
    python run.py agent branding all
            """)
            return

        agent_name = sys.argv[2]

        if agent_name == "seo-strategist":
            from src.agents.seo_strategist import SEOStrategist
            import json

            strategist = SEOStrategist()

            if len(sys.argv) < 4:
                print("Usage: python run.py agent seo-strategist <command> [options]")
                return

            agent_cmd = sys.argv[3]
            kwargs = {}
            positional = None
            i = 4

            while i < len(sys.argv):
                arg = sys.argv[i]
                if arg.startswith("--"):
                    key = arg[2:]
                    if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                        value = sys.argv[i + 1]
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                        kwargs[key] = value
                        i += 2
                    else:
                        kwargs[key] = True
                        i += 1
                else:
                    positional = arg
                    i += 1

            # Map positional to command-specific key
            if positional:
                if agent_cmd in ["research", "competitors"]:
                    kwargs["keyword"] = positional
                elif agent_cmd == "ab-test":
                    kwargs["title"] = positional

            result = strategist.run(agent_cmd, **kwargs)

            print("\n" + "=" * 60)
            print(f"SEO STRATEGIST: {agent_cmd.upper()}")
            print("=" * 60)
            print(result.summary())

            if kwargs.get("json"):
                print("\nFull Output:")
                print(json.dumps(result.to_dict(), indent=2))

        elif agent_name == "branding":
            from src.content.channel_branding import ChannelBrandingGenerator

            generator = ChannelBrandingGenerator()

            if len(sys.argv) < 4:
                print("Usage: python run.py agent branding <channel|all> [niche]")
                return

            target = sys.argv[3]

            if target == "all":
                channels = [
                    ("money_blueprints", "finance"),
                    ("mind_unlocked", "psychology"),
                    ("untold_stories", "storytelling")
                ]
                for channel_id, niche in channels:
                    print(f"\nGenerating assets for {channel_id}...")
                    assets = generator.generate_all_assets(channel_id, niche)
                    print(f"  Profile: {assets['profile_picture']}")
                    print(f"  Banner: {assets['banner']}")
                print(f"\nAll assets generated in: {generator.output_dir}")
            else:
                channel_id = target
                niche = sys.argv[4] if len(sys.argv) > 4 else "default"
                assets = generator.generate_all_assets(channel_id, niche)
                print(f"\nAssets generated for {channel_id}:")
                print(f"  Profile: {assets['profile_picture']}")
                print(f"  Banner: {assets['banner']}")

        elif agent_name == "seo":
            from src.agents.seo_agent import SEOAgent
            import json

            agent = SEOAgent()
            kwargs = {}
            i = 3

            while i < len(sys.argv):
                arg = sys.argv[i]
                if arg.startswith("--"):
                    key = arg[2:]
                    if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                        kwargs[key] = sys.argv[i + 1]
                        i += 2
                    else:
                        kwargs[key] = True
                        i += 1
                else:
                    kwargs["title"] = arg
                    i += 1

            result = agent.run(**kwargs)
            print("\n" + result.summary())

        else:
            print(f"Unknown agent: {agent_name}")
            print("Available: seo-strategist, seo, branding")

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python run.py' for help")


if __name__ == "__main__":
    main()
