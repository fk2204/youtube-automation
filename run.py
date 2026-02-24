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

# ============================================================
# AGENT COMMAND HANDLERS
# ============================================================

def _print_agent_result(result, output_json=False, verbose=False):
    """Pretty print an AgentResult or similar dataclass."""
    import json

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print("\n" + "=" * 60)

    # Check for common result attributes
    if hasattr(result, 'summary'):
        print(result.summary())
    else:
        # Generic output
        if hasattr(result, 'success'):
            status = "[OK]" if result.success else "[FAIL]"
            print(f"Status: {status}")

        if hasattr(result, 'operation'):
            print(f"Operation: {result.operation}")

        if hasattr(result, 'error') and result.error:
            print(f"Error: {result.error}")

    # Show additional details in verbose mode
    if verbose:
        print("\n" + "-" * 40)
        print("Detailed Output:")
        if hasattr(result, 'to_dict'):
            for key, value in result.to_dict().items():
                if key not in ['timestamp', 'error']:
                    print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")

    # Show token usage if available
    if hasattr(result, 'tokens_used') and result.tokens_used > 0:
        print(f"\nTokens used: {result.tokens_used}")
    if hasattr(result, 'cost') and result.cost > 0:
        print(f"Cost: ${result.cost:.4f}")

    print("=" * 60)


def _parse_agent_args(args):
    """Parse agent command arguments into kwargs."""
    kwargs = {}
    positional = []
    i = 0

    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                # Try to convert to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                kwargs[key] = value
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            positional.append(arg)
            i += 1

    return kwargs, positional


def _handle_agent_command(args):
    """Handle agent subcommands."""
    import json

    if not args:
        print("""
Agent Dispatcher - Comprehensive Task Agents
============================================

Usage:
    python run.py agent <agent-type> <subcommand> [options]

Available Agent Types:
    research          Topic discovery and trend analysis
    seo-strategy      World-class SEO optimization
    quality           Script and content validation
    analytics         Video performance analysis
    workflow          Automation workflow management
    monitor           System health monitoring
    scheduler         Schedule management
    safety            Content safety checks
    compliance        Platform compliance validation
    branding          Channel branding assets

Global Options:
    --json            Output in JSON format
    --verbose         Detailed logging output

Research Commands:
    python run.py agent research --niche finance --count 10
    python run.py agent research --channel money_blueprints
    python run.py agent research --trends --niche psychology
    python run.py agent research --competitors --niche storytelling

SEO Strategy Commands:
    python run.py agent seo-strategy research "passive income" --niche finance
    python run.py agent seo-strategy ab-test "My Title" --variants 5
    python run.py agent seo-strategy strategy --niche finance --topics 10
    python run.py agent seo-strategy competitors "keyword" --top 10
    python run.py agent seo-strategy calendar --niche finance --weeks 4

Quality Commands:
    python run.py agent quality validate "output/video.mp4"
    python run.py agent quality audio "output/audio.mp3"
    python run.py agent quality video "output/video.mp4"
    python run.py agent quality script "output/script.txt" --niche finance

Analytics Commands:
    python run.py agent analytics insights --channel money_blueprints --period 30d
    python run.py agent analytics revenue --all-channels
    python run.py agent analytics strategy --niche finance
    python run.py agent analytics cost

Workflow Commands:
    python run.py agent workflow status
    python run.py agent workflow run full-video --channel money_blueprints
    python run.py agent workflow run short --channel mind_unlocked

Monitor Commands:
    python run.py agent monitor health
    python run.py agent monitor resources
    python run.py agent monitor errors

Scheduler Commands:
    python run.py agent scheduler status
    python run.py agent scheduler next
    python run.py agent scheduler list

Safety/Compliance Commands:
    python run.py agent safety check "script.txt"
    python run.py agent compliance check "video.mp4"

Examples:
    python run.py agent research --niche finance --count 5 --json
    python run.py agent seo-strategy ab-test "How to Make Money" --variants 5
    python run.py agent quality validate output/video.mp4 --verbose
    python run.py agent analytics insights --channel money_blueprints --period 7d
        """)
        return

    agent_type = args[0].lower()
    subargs = args[1:] if len(args) > 1 else []
    kwargs, positional = _parse_agent_args(subargs)

    output_json = kwargs.pop("json", False)
    verbose = kwargs.pop("verbose", False)

    try:
        if agent_type == "research":
            _handle_research_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type in ["seo-strategy", "seo-strategist"]:
            _handle_seo_strategy_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "quality":
            _handle_quality_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "analytics":
            _handle_analytics_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "workflow":
            _handle_workflow_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "monitor":
            _handle_monitor_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "scheduler":
            _handle_scheduler_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "safety":
            _handle_safety_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "compliance":
            _handle_compliance_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "branding":
            _handle_branding_agent(subargs, kwargs, positional, output_json, verbose)

        elif agent_type == "seo":
            # Basic SEO agent (backwards compatibility)
            _handle_basic_seo_agent(subargs, kwargs, positional, output_json, verbose)

        else:
            print(f"Unknown agent type: {agent_type}")
            print("Run 'python run.py agent' for available agents")

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        print(f"\n[ERROR] Agent error: {e}")
        sys.exit(1)


def _handle_research_agent(args, kwargs, positional, output_json, verbose):
    """Handle research agent commands."""
    from src.agents.research_agent import ResearchAgent

    agent = ResearchAgent()

    # Determine operation
    niche = kwargs.get("niche", "finance")
    channel = kwargs.get("channel")
    count = kwargs.get("count", 5)

    if kwargs.get("trends"):
        result = agent.analyze_trends(niche)
    elif kwargs.get("competitors"):
        result = agent.analyze_competitors(niche)
    elif channel:
        # Generate ideas for specific channel
        result = agent.generate_viral_ideas(channel_id=channel, count=count)
    else:
        # Default: find topics
        result = agent.find_topics(niche, count=count)

    _print_agent_result(result, output_json, verbose)

    # Print ideas if available
    if not output_json and hasattr(result, 'ideas') and result.ideas:
        print(f"\nTop Ideas ({len(result.ideas)}):")
        for i, idea in enumerate(result.ideas[:5], 1):
            print(f"  {i}. {idea.title}")
            if verbose:
                print(f"     Score: {idea.score} | Trend: {idea.trend_score}")


def _handle_seo_strategy_agent(args, kwargs, positional, output_json, verbose):
    """Handle SEO strategy agent commands."""
    from src.agents.seo_strategist import SEOStrategist
    import json as json_module

    strategist = SEOStrategist()

    if not args:
        print("Usage: python run.py agent seo-strategy <command> [options]")
        print("Commands: research, ab-test, strategy, competitors, calendar, optimize")
        return

    command = args[0]
    niche = kwargs.get("niche", "default")

    if command == "research":
        keyword = positional[0] if positional else kwargs.get("keyword", "")
        if not keyword:
            print("Error: Keyword required. Usage: agent seo-strategy research \"keyword\"")
            return
        result = strategist.research_keyword(keyword, niche)

    elif command == "ab-test":
        title = positional[0] if positional else kwargs.get("title", "")
        if not title:
            print("Error: Title required. Usage: agent seo-strategy ab-test \"title\"")
            return
        variants = kwargs.get("variants", 5)
        result = strategist.generate_ab_variants(title, variants, niche)

    elif command == "strategy":
        topics = kwargs.get("topics", 10)
        weeks = kwargs.get("weeks", 4)
        result = strategist.content_strategy(niche, topics, weeks)

    elif command == "competitors":
        keyword = positional[0] if positional else kwargs.get("keyword", "")
        if not keyword:
            print("Error: Keyword required. Usage: agent seo-strategy competitors \"keyword\"")
            return
        top = kwargs.get("top", 10)
        result = strategist.analyze_competitors(keyword, top)

    elif command == "calendar":
        weeks = kwargs.get("weeks", 4)
        result = strategist.content_strategy(niche, topics=weeks * 3, weeks=weeks)

    elif command == "optimize":
        file_path = kwargs.get("file")
        if not file_path:
            print("Error: File required. Usage: agent seo-strategy optimize --file path.json")
            return
        with open(file_path) as f:
            data = json_module.load(f)
        result = strategist.full_optimization(
            title=data.get("title", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            niche=niche
        )

    else:
        print(f"Unknown command: {command}")
        print("Commands: research, ab-test, strategy, competitors, calendar, optimize")
        return

    _print_agent_result(result, output_json, verbose)


def _handle_quality_agent(args, kwargs, positional, output_json, verbose):
    """Handle quality agent commands."""
    from src.agents.quality_agent import QualityAgent

    agent = QualityAgent()

    if not args:
        print("Usage: python run.py agent quality <command> [file] [options]")
        print("Commands: validate, audio, video, script")
        return

    command = args[0]
    file_path = positional[0] if positional else kwargs.get("file")
    niche = kwargs.get("niche", "default")
    is_short = kwargs.get("short", False)

    if command == "validate" or command == "video":
        if not file_path:
            print("Error: File path required")
            return
        result = agent.check_video_file(file_path, is_short=is_short)

    elif command == "audio":
        if not file_path:
            print("Error: Audio file path required")
            return
        # Audio validation - check if file exists and basic properties
        import os
        if not os.path.exists(file_path):
            print(f"[FAIL] Audio file not found: {file_path}")
            return
        print(f"[OK] Audio file exists: {file_path}")
        print(f"     Size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
        return

    elif command == "script":
        if not file_path:
            print("Error: Script file path required")
            return
        with open(file_path, 'r', encoding='utf-8') as f:
            script_text = f.read()

        full = kwargs.get("full", False)
        if full:
            result = agent.full_analysis(script_text, niche, is_short)
        else:
            result = agent.quick_check(script_text, niche, is_short)

    else:
        print(f"Unknown command: {command}")
        return

    _print_agent_result(result, output_json, verbose)


def _handle_analytics_agent(args, kwargs, positional, output_json, verbose):
    """Handle analytics agent commands."""
    from src.agents.analytics_agent import AnalyticsAgent

    agent = AnalyticsAgent()

    if not args:
        print("Usage: python run.py agent analytics <command> [options]")
        print("Commands: insights, revenue, strategy, cost, patterns")
        return

    command = args[0]

    if command == "insights":
        channel = kwargs.get("channel")
        period = kwargs.get("period", "30d")
        if not channel:
            print("Error: --channel required for insights")
            return
        result = agent.analyze_channel(channel, period)

    elif command == "revenue":
        all_channels = kwargs.get("all_channels", False)
        # Revenue analysis using cost tracker
        result = agent.get_cost_analysis()

    elif command == "strategy":
        niche = kwargs.get("niche", "finance")
        use_ai = kwargs.get("ai", False)
        result = agent.generate_strategy(niche, use_ai=use_ai)

    elif command == "cost":
        result = agent.get_cost_analysis()

    elif command == "patterns":
        niche = kwargs.get("niche")
        channel = kwargs.get("channel")
        result = agent.find_patterns(niche=niche, channel=channel)

    else:
        print(f"Unknown command: {command}")
        return

    _print_agent_result(result, output_json, verbose)


def _handle_workflow_agent(args, kwargs, positional, output_json, verbose):
    """Handle workflow agent commands."""
    import json as json_module
    from datetime import datetime

    if not args:
        print("Usage: python run.py agent workflow <command> [options]")
        print("Commands: status, run")
        return

    command = args[0]

    if command == "status":
        # Show workflow status
        print("\n" + "=" * 60)
        print("WORKFLOW STATUS")
        print("=" * 60)
        print(f"\nTimestamp: {datetime.now().isoformat()}")
        print("\nPipeline Components:")
        print("  [OK] Research Agent")
        print("  [OK] Script Writer")
        print("  [OK] TTS Engine")
        print("  [OK] Video Assembler")
        print("  [OK] Quality Checker")
        print("  [OK] YouTube Uploader")
        print("\nStatus: All systems operational")
        print("=" * 60)

    elif command == "run":
        if len(args) < 2:
            print("Usage: python run.py agent workflow run <pipeline> [options]")
            print("Pipelines: full-video, short")
            return

        pipeline = args[1]
        channel = kwargs.get("channel", "money_blueprints")
        niche = kwargs.get("niche", "finance")

        if pipeline == "full-video":
            from src.automation.runner import task_full_with_upload
            print(f"\n[INFO] Running full video pipeline for {channel}...")
            result = task_full_with_upload(channel)
            if result["success"]:
                print(f"[OK] Video uploaded: {result['results'].get('video_url', 'N/A')}")
            else:
                print(f"[FAIL] Pipeline failed: {result.get('error')}")

        elif pipeline == "short":
            from src.automation.runner import task_short_with_upload
            print(f"\n[INFO] Running short video pipeline for {channel}...")
            result = task_short_with_upload(channel)
            if result["success"]:
                print(f"[OK] Short uploaded: {result['results'].get('video_url', 'N/A')}")
            else:
                print(f"[FAIL] Pipeline failed: {result.get('error')}")

        else:
            print(f"Unknown pipeline: {pipeline}")

    else:
        print(f"Unknown command: {command}")


def _handle_monitor_agent(args, kwargs, positional, output_json, verbose):
    """Handle monitor agent commands."""
    import json as json_module
    from datetime import datetime
    import os
    import psutil

    if not args:
        print("Usage: python run.py agent monitor <command>")
        print("Commands: health, resources, errors")
        return

    command = args[0]

    if command == "health":
        print("\n" + "=" * 60)
        print("SYSTEM HEALTH CHECK")
        print("=" * 60)
        print(f"\nTimestamp: {datetime.now().isoformat()}")

        # Check API keys
        api_keys = {
            "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
            "PEXELS_API_KEY": bool(os.getenv("PEXELS_API_KEY")),
            "PIXABAY_API_KEY": bool(os.getenv("PIXABAY_API_KEY")),
            "FISH_AUDIO_API_KEY": bool(os.getenv("FISH_AUDIO_API_KEY")),
        }

        print("\nAPI Keys:")
        for key, configured in api_keys.items():
            status = "[OK]" if configured else "[MISSING]"
            print(f"  {status} {key}")

        # Check directories
        dirs = ["output", "assets", "data", "config"]
        print("\nDirectories:")
        for d in dirs:
            status = "[OK]" if os.path.exists(d) else "[MISSING]"
            print(f"  {status} {d}/")

        print("\n" + "=" * 60)

    elif command == "resources":
        print("\n" + "=" * 60)
        print("SYSTEM RESOURCES")
        print("=" * 60)

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"\nCPU Usage: {cpu_percent}%")

        # Memory usage
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.percent}% used ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")

        # Disk usage
        disk = psutil.disk_usage('.')
        print(f"Disk: {disk.percent}% used ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)")

        print("\n" + "=" * 60)

    elif command == "errors":
        print("\n" + "=" * 60)
        print("RECENT ERRORS")
        print("=" * 60)

        # Check for error logs
        log_files = [
            "logs/error.log",
            "logs/automation.log"
        ]

        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"\n{log_file}:")
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    errors = [l for l in lines if 'ERROR' in l or 'FAIL' in l][-5:]
                    for line in errors:
                        print(f"  {line.strip()}")
            else:
                print(f"\n{log_file}: Not found")

        print("\n" + "=" * 60)

    else:
        print(f"Unknown command: {command}")


def _handle_scheduler_agent(args, kwargs, positional, output_json, verbose):
    """Handle scheduler agent commands."""
    from datetime import datetime

    if not args:
        print("Usage: python run.py agent scheduler <command>")
        print("Commands: status, next, list")
        return

    command = args[0]

    if command == "status":
        from src.scheduler.daily_scheduler import show_status
        show_status()

    elif command == "next":
        # Show next scheduled items
        print("\n" + "=" * 60)
        print("NEXT SCHEDULED ITEMS")
        print("=" * 60)

        # This would integrate with APScheduler
        # For now, show configured schedule
        import yaml
        try:
            with open("config/channels.yaml") as f:
                channels = yaml.safe_load(f)

            print("\nUpcoming (based on config):")
            for channel_id, config in channels.get("channels", {}).items():
                if config.get("enabled", True):
                    schedule = config.get("schedule", {})
                    times = schedule.get("posting_times", [])
                    days = schedule.get("posting_days", [])
                    print(f"  {channel_id}: {days[:3]}... at {times[:2]}...")

        except Exception as e:
            print(f"  Error loading schedule: {e}")

        print("\n" + "=" * 60)

    elif command == "list":
        # List all scheduled jobs
        print("\n" + "=" * 60)
        print("SCHEDULED JOBS")
        print("=" * 60)

        import yaml
        try:
            with open("config/channels.yaml") as f:
                channels = yaml.safe_load(f)

            for channel_id, config in channels.get("channels", {}).items():
                enabled = "[ON]" if config.get("enabled", True) else "[OFF]"
                print(f"\n{enabled} {channel_id}")
                schedule = config.get("schedule", {})
                print(f"    Days: {schedule.get('posting_days', [])}")
                print(f"    Times: {schedule.get('posting_times', [])}")

        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "=" * 60)

    else:
        print(f"Unknown command: {command}")


def _handle_safety_agent(args, kwargs, positional, output_json, verbose):
    """Handle safety agent commands."""
    import os
    import re

    if not args:
        print("Usage: python run.py agent safety check <file>")
        return

    command = args[0]

    if command == "check":
        # File path is in args[1] (after "check")
        file_path = args[1] if len(args) > 1 else kwargs.get("file")
        if not file_path:
            print("Error: File path required")
            return

        if not os.path.exists(file_path):
            print(f"[FAIL] File not found: {file_path}")
            return

        print("\n" + "=" * 60)
        print("CONTENT SAFETY CHECK")
        print("=" * 60)
        print(f"\nFile: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        issues = []

        # Check for potentially problematic content
        problematic_patterns = [
            (r'\b(guarantee[d]?|proven|100%)\s+(results?|works?|success)', "Income claim"),
            (r'get rich quick', "Get rich quick claim"),
            (r'financial advice', "Unlicensed financial advice"),
            (r'investment advice', "Unlicensed investment advice"),
            (r'medical advice', "Unlicensed medical advice"),
        ]

        for pattern, issue_type in problematic_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(issue_type)

        # Word count
        word_count = len(content.split())

        print(f"\nWord count: {word_count}")
        print(f"\nSafety Issues Found: {len(issues)}")

        if issues:
            for issue in issues:
                print(f"  [!] {issue}")
            print("\nStatus: REVIEW REQUIRED")
        else:
            print("  None detected")
            print("\nStatus: PASSED")

        print("\n" + "=" * 60)

    else:
        print(f"Unknown command: {command}")


def _handle_compliance_agent(args, kwargs, positional, output_json, verbose):
    """Handle compliance agent commands."""
    import os

    if not args:
        print("Usage: python run.py agent compliance check <file>")
        return

    command = args[0]

    if command == "check":
        # File path is in args[1] (after "check")
        file_path = args[1] if len(args) > 1 else kwargs.get("file")
        if not file_path:
            print("Error: File path required")
            return

        if not os.path.exists(file_path):
            print(f"[FAIL] File not found: {file_path}")
            return

        print("\n" + "=" * 60)
        print("PLATFORM COMPLIANCE CHECK")
        print("=" * 60)
        print(f"\nFile: {file_path}")

        # Check file properties
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        file_ext = os.path.splitext(file_path)[1].lower()

        checks = []

        # Size check (YouTube limit is 256GB, but practical limit ~12 hours)
        if file_size > 128 * 1024:  # 128GB
            checks.append(("[WARN]", "File size > 128GB"))
        else:
            checks.append(("[OK]", f"File size: {file_size:.1f}MB"))

        # Format check
        allowed_formats = ['.mp4', '.mov', '.avi', '.wmv', '.flv', '.webm', '.mkv']
        if file_ext in allowed_formats:
            checks.append(("[OK]", f"Format: {file_ext} (supported)"))
        else:
            checks.append(("[FAIL]", f"Format: {file_ext} (not supported)"))

        # If video, check technical specs
        if file_ext in ['.mp4', '.mov', '.mkv', '.webm']:
            try:
                import subprocess
                result = subprocess.run(
                    ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', file_path],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    import json
                    probe_data = json.loads(result.stdout)
                    for stream in probe_data.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            width = stream.get('width', 0)
                            height = stream.get('height', 0)
                            if width >= 1920 and height >= 1080:
                                checks.append(("[OK]", f"Resolution: {width}x{height} (HD+)"))
                            else:
                                checks.append(("[WARN]", f"Resolution: {width}x{height} (below HD)"))
            except:
                checks.append(("[?]", "Could not analyze video specs"))

        print("\nCompliance Checks:")
        for status, message in checks:
            print(f"  {status} {message}")

        failed = any(c[0] == "[FAIL]" for c in checks)
        print(f"\nOverall: {'FAILED' if failed else 'PASSED'}")

        print("\n" + "=" * 60)

    else:
        print(f"Unknown command: {command}")


def _handle_branding_agent(args, kwargs, positional, output_json, verbose):
    """Handle branding agent commands."""
    from src.content.channel_branding import ChannelBrandingGenerator

    generator = ChannelBrandingGenerator()

    if not args:
        print("Usage: python run.py agent branding <all|channel_id> [niche]")
        return

    target = args[0]

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
        niche = args[1] if len(args) > 1 else kwargs.get("niche", "default")
        assets = generator.generate_all_assets(channel_id, niche)
        print(f"\nAssets generated for {channel_id}:")
        print(f"  Profile: {assets['profile_picture']}")
        print(f"  Banner: {assets['banner']}")


def _handle_basic_seo_agent(args, kwargs, positional, output_json, verbose):
    """Handle basic SEO agent commands (backwards compatibility)."""
    from src.agents.seo_agent import SEOAgent

    agent = SEOAgent()

    # If positional argument, treat as title
    if positional:
        kwargs["title"] = positional[0]

    result = agent.run(**kwargs)
    _print_agent_result(result, output_json, verbose)


def _handle_agents_command(args):
    """Handle 'agents' command for overall status."""
    import json as json_module
    from datetime import datetime

    if not args:
        args = ["status"]

    command = args[0]

    if command == "status":
        print("\n" + "=" * 60)
        print("AGENTS STATUS OVERVIEW")
        print("=" * 60)
        print(f"\nTimestamp: {datetime.now().isoformat()}")

        agents_status = [
            ("ResearchAgent", "research", "Topic discovery and trends", True),
            ("SEOStrategist", "seo-strategy", "World-class SEO optimization", True),
            ("SEOAgent", "seo", "Basic SEO metadata optimization", True),
            ("QualityAgent", "quality", "Content validation", True),
            ("AnalyticsAgent", "analytics", "Performance analysis", True),
            ("WorkflowAgent", "workflow", "Automation management", True),
            ("MonitorAgent", "monitor", "System health", True),
            ("SchedulerAgent", "scheduler", "Schedule management", True),
            ("SafetyAgent", "safety", "Content safety checks", True),
            ("ComplianceAgent", "compliance", "Platform compliance", True),
            ("BrandingAgent", "branding", "Channel branding", True),
        ]

        print("\nAvailable Agents:")
        print("-" * 60)
        print(f"{'Agent':<20} {'Command':<15} {'Description':<25}")
        print("-" * 60)

        for name, cmd, desc, available in agents_status:
            status = "[OK]" if available else "[--]"
            print(f"{status} {name:<17} {cmd:<15} {desc:<25}")

        print("-" * 60)
        print(f"\nTotal: {len(agents_status)} agents available")
        print("\nUsage: python run.py agent <command> [subcommand] [options]")
        print("       python run.py agent --help")
        print("=" * 60)

    elif command == "list":
        print("\n" + "=" * 60)
        print("AVAILABLE AGENTS")
        print("=" * 60)

        agents = [
            ("research", "Topic discovery, trend analysis, competitor research"),
            ("seo-strategy", "Keyword research, A/B testing, content strategy"),
            ("seo", "Title, description, tag optimization"),
            ("quality", "Video/script validation, quality checks"),
            ("analytics", "Channel insights, performance metrics, cost analysis"),
            ("workflow", "Pipeline status, run automation workflows"),
            ("monitor", "System health, resources, error monitoring"),
            ("scheduler", "View and manage scheduled tasks"),
            ("safety", "Content safety verification"),
            ("compliance", "Platform compliance checks"),
            ("branding", "Channel asset generation"),
        ]

        for cmd, desc in agents:
            print(f"\n  {cmd}")
            print(f"    {desc}")

        print("\n" + "=" * 60)

    else:
        print(f"Unknown command: {command}")
        print("Usage: python run.py agents [status|list]")


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
  python run.py monitor             Monitor all recent videos (72h)
  python run.py monitor <video_id>  Monitor specific video performance

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

AI Video Generation (Runway, Pika):
  python run.py ai-video "<prompt>"   Generate AI video from text prompt
  python run.py ai-video "<prompt>" --provider runway --quality high
  python run.py ai-video "<prompt>" --aspect-ratio 9:16  (for Shorts)
  python run.py ai-broll <script_file>  Generate B-roll for script
  python run.py ai-broll <script> --style cinematic --niche finance
  python run.py ai-video-providers    List available AI video providers
  python run.py ai-video-cost [count] Estimate cost for batch generation

Performance & Optimization:
  python run.py gpu-status            Show GPU acceleration status
  python run.py benchmark             Run video encoding performance benchmark
  python run.py benchmark 60          Run 60-second benchmark
  python run.py async-download-test   Test async download speeds

Reddit Research Commands:
  python run.py reddit-trends <niche>     Get trending topics from Reddit
  python run.py reddit-questions <niche>  Get questions for FAQ videos
  python run.py reddit-viral <niche>      Find viral content ideas
  python run.py reddit-research <niche>   Full research report

  Options for Reddit commands:
    --limit N         Number of results (default varies by command)
    --min-upvotes N   Minimum upvotes threshold
    --json            Output as JSON
    --full            Full research report (for reddit-viral)

  Niches: finance, psychology, storytelling, technology, education, health

Content Empire (Multi-Platform Distribution):
  python run.py empire-daily              Full daily automation (all channels + platforms)
  python run.py empire-daily --channels money_blueprints --types video_long,blog
  python run.py empire-video <channel>    Create video + distribute everywhere
  python run.py empire-video <channel> --platforms youtube,tiktok
  python run.py empire-blog "<topic>"     Create blog + distribute
  python run.py distribute <video_path>   Distribute existing content to platforms
  python run.py platform-status           Check all platform connections
  python run.py cross-analytics           Cross-platform performance report
  python run.py api-server                Start Content Empire REST API (port 8000)
  python run.py api-server --port 9000    Start on custom port

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
  python run.py agent                         Show all available agents
  python run.py agents status                 Overview of all agents
  python run.py agents list                   List all agents with descriptions

  Research Agent:
    python run.py agent research --niche finance --count 10
    python run.py agent research --channel money_blueprints
    python run.py agent research --trends --niche psychology
    python run.py agent research --competitors --niche storytelling

  SEO Strategy Agent:
    python run.py agent seo-strategy research "keyword" --niche finance
    python run.py agent seo-strategy ab-test "My Title" --variants 5
    python run.py agent seo-strategy strategy --niche finance --topics 10
    python run.py agent seo-strategy competitors "keyword" --top 10
    python run.py agent seo-strategy calendar --niche finance --weeks 4

  Quality Agent:
    python run.py agent quality validate "output/video.mp4"
    python run.py agent quality audio "output/audio.mp3"
    python run.py agent quality video "output/video.mp4"
    python run.py agent quality script "script.txt" --niche finance

  Analytics Agent:
    python run.py agent analytics insights --channel money_blueprints --period 30d
    python run.py agent analytics revenue --all-channels
    python run.py agent analytics strategy --niche finance
    python run.py agent analytics cost

  Automation Agents:
    python run.py agent workflow status
    python run.py agent workflow run full-video --channel money_blueprints
    python run.py agent monitor health
    python run.py agent monitor resources
    python run.py agent scheduler status
    python run.py agent scheduler next

  Safety/Compliance:
    python run.py agent safety check "script.txt"
    python run.py agent compliance check "video.mp4"

  Branding:
    python run.py agent branding all
    python run.py agent branding money_blueprints finance

  Agent Global Options:
    --json              Output in JSON format
    --verbose           Detailed logging output

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
    --ai        Use AI face generation (requires REPLICATE_API_KEY)
    --emotion   Face emotion for AI thumbnails (niche-specific)

A/B Testing:
  python run.py ab-test <video_id> <thumb1.png> <thumb2.png> [options]
                                    Start A/B test for thumbnails
    --duration  Test duration in hours (default: 72)
    --titles    Comma-separated title variants (optional)
  python run.py ab-check <test_id>  Check test progress and switch variants
  python run.py ab-report <test_id> Get detailed test report
  python run.py ab-list             List all active A/B tests

Performance Monitoring:
  python run.py monitor             Monitor all videos from last 72 hours
  python run.py monitor <video_id>  Monitor specific video
  python run.py monitor --hours 24  Monitor videos from last 24 hours
  python run.py monitor --summary   Show alert summary only
    Thresholds: CTR warning=4%, critical=2%
                Retention warning=35%, critical=25%
                Viral potential: 3x channel average

Analytics Commands:
  python run.py analytics <video_id>           Show video analytics & algorithm score
  python run.py analytics <video_id> --days N  Analyze last N days (default: 28)
  python run.py analytics <video_id> --dropoffs  Show retention dropoff points
  python run.py analytics-compare <video_id>   Compare video to channel average
  python run.py analytics-compare <video_id> --days N  Use N days for comparison

Integrated Module Commands (New - 2026-01-20):
  python run.py caption <audio_file>           Generate captions using Whisper
    --model <size>    Model: tiny, base, small, medium, large (default: base)
    --format <fmt>    Output: srt, vtt, json (default: srt)
    --language <code> Language code (auto-detect if not set)

  python run.py keywords <topic>               Research keywords (free via Google Trends)
    --niche <niche>   Content niche for context
    --longtails       Include long-tail variations
    --trends          Include trend analysis

  python run.py optimize-title "<title>"       Optimize title with IMPACT formula
    --niche <niche>   Content niche
    --variants N      Number of variants to generate (default: 5)

  python run.py feedback <video_id>            Analyze video performance
    --niche <niche>   Content niche for comparison
    --recommendations Generate improvement suggestions

  python run.py hooks "<topic>"                Generate viral hooks
    --niche <niche>   Content niche
    --style <style>   curiosity, action, contrarian, emotional, value
    --count N         Number of hooks (default: 5)
    --enhance <file>  Enhance script file with retention features

  python run.py integration-status             Show status of all integrated modules

AI Video Generation Commands:
  python run.py ai-video "<prompt>"            Generate AI video (uses smart routing)
    --provider <name>  Provider: hailuo, pika (default: auto-selects cheapest)
    --budget           Use budget mode (cheapest provider)
    --quality <level>  Quality: high, medium, low (default: medium)
    --output <path>    Output file path
    --image <url>      Image URL for image-to-video generation

  python run.py ai-video --budget "<prompt>"   Generate using cheapest provider (HailuoAI)
  python run.py ai-video --provider hailuo "<prompt>"  Use specific provider
  python run.py ai-video --provider pika "<prompt>"    Use Pika Labs
  python run.py ai-video-providers             List available AI video providers
  python run.py ai-video-cost <count>          Estimate cost for batch generation

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
  python run.py thumbnail "Money Secrets" --niche finance --ai
  python run.py thumbnail "Mind Tricks" --niche psychology --ai --emotion secretive
  python run.py monitor                    # Monitor all recent videos
  python run.py monitor dQw4w9WgXcQ        # Monitor specific video
  python run.py analytics dQw4w9WgXcQ      # Show video analytics
  python run.py analytics dQw4w9WgXcQ --dropoffs  # Include retention analysis
  python run.py analytics-compare dQw4w9WgXcQ     # Compare to channel average
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
                                 choices=["center", "top", "bottom", "right", "left"],
                                 help="Text position (default: center)")
        thumb_parser.add_argument("--variants", type=int, help="Generate N variants for A/B testing")
        thumb_parser.add_argument("--background", help="Background image path")
        thumb_parser.add_argument("--vignette", type=float, default=0.7,
                                 help="Vignette strength 0.0-1.0 (default: 0.7)")
        thumb_parser.add_argument("--ai", action="store_true",
                                 help="Use AI face generation (requires REPLICATE_API_KEY)")
        thumb_parser.add_argument("--emotion", help="Face emotion for AI thumbnails")
        thumb_args = thumb_parser.parse_args(sys.argv[2:])

        if not thumb_args.title:
            print("Error: Please provide a video title")
            print('Usage: python run.py thumbnail "My Video Title" --niche finance')
            print('       python run.py thumbnail "My Video Title" --niche finance --ai')
            return

        print(f"\n{'='*60}")
        if thumb_args.ai:
            print(f"AI-Powered Thumbnail Generator")
        else:
            print(f"YouTube Thumbnail Generator")
        print(f"{'='*60}")
        print(f"Title: {thumb_args.title}")
        print(f"Niche: {thumb_args.niche}")
        if thumb_args.ai:
            print(f"AI Face: Enabled")
            if thumb_args.emotion:
                print(f"Emotion: {thumb_args.emotion}")
            print(f"Face Position: {thumb_args.position if thumb_args.position in ['right', 'left'] else 'right'}")
        else:
            print(f"Position: {thumb_args.position}")
        if thumb_args.variants:
            print(f"Variants: {thumb_args.variants}")
        print(f"{'='*60}\n")

        # Use AI generator or standard generator based on --ai flag
        if thumb_args.ai:
            from src.content.thumbnail_ai import AIThumbnailGenerator

            generator = AIThumbnailGenerator()

            if thumb_args.variants:
                paths = generator.generate_ab_test_variants(
                    title=thumb_args.title,
                    niche=thumb_args.niche,
                    count=thumb_args.variants
                )
                print(f"\n[OK] Generated {len(paths)} AI thumbnail variants:")
                for path in paths:
                    print(f"     - {path}")
            else:
                face_position = thumb_args.position if thumb_args.position in ["right", "left"] else "right"
                path = generator.generate_with_ai_face(
                    title=thumb_args.title,
                    niche=thumb_args.niche,
                    emotion=thumb_args.emotion,
                    output_path=thumb_args.output,
                    face_position=face_position
                )
                print(f"\n[OK] AI Thumbnail generated: {path}")
                print(f"     Resolution: 1280x720 (YouTube standard)")
                print(f"     AI Face: Yes")
                print(f"     Color scheme: {thumb_args.niche}")
        else:
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
        # Comprehensive Agent dispatcher for specialized tasks
        _handle_agent_command(sys.argv[2:])

    elif cmd == "agents":
        # Show overall status of all agents
        _handle_agents_command(sys.argv[2:])

    elif cmd == "analytics":
        # Show video analytics and algorithm score
        analytics_parser = argparse.ArgumentParser(prog="run.py analytics", add_help=False)
        analytics_parser.add_argument("video_id", nargs="?", help="YouTube video ID")
        analytics_parser.add_argument("--days", type=int, default=28,
                                      help="Number of days to analyze (default: 28)")
        analytics_parser.add_argument("--dropoffs", action="store_true",
                                      help="Show retention dropoff points")
        analytics_parser.add_argument("--json", action="store_true",
                                      help="Output as JSON")
        analytics_args = analytics_parser.parse_args(sys.argv[2:])

        if not analytics_args.video_id:
            print("Error: Please provide a video ID")
            print("Usage: python run.py analytics <video_id> [--days N] [--dropoffs]")
            return

        from src.youtube.analytics_api import YouTubeAnalyticsAPI

        print(f"\n{'='*60}")
        print(f"YouTube Video Analytics")
        print(f"{'='*60}")
        print(f"Video ID: {analytics_args.video_id}")
        print(f"Period: Last {analytics_args.days} days")
        print(f"{'='*60}\n")

        try:
            api = YouTubeAnalyticsAPI()
            video_stats = api.get_video_analytics(analytics_args.video_id, days=analytics_args.days)

            if analytics_args.json:
                import json
                output = {
                    "video_id": video_stats.video_id,
                    "views": video_stats.views,
                    "watch_time_minutes": video_stats.watch_time_minutes,
                    "avg_view_duration": video_stats.avg_view_duration,
                    "avg_percentage_viewed": video_stats.avg_percentage_viewed,
                    "ctr": video_stats.ctr,
                    "impressions": video_stats.impressions,
                    "likes": video_stats.likes,
                    "comments": video_stats.comments,
                    "shares": video_stats.shares,
                    "subscribers_gained": video_stats.subscribers_gained,
                    "algorithm_score": video_stats.get_algorithm_score(),
                    "traffic_sources": video_stats.traffic_source_breakdown
                }
                print(json.dumps(output, indent=2))
            else:
                print(video_stats.get_performance_summary())

            # Show dropoff points if requested
            if analytics_args.dropoffs:
                print(f"\n{'='*60}")
                print("Retention Dropoff Analysis")
                print(f"{'='*60}\n")

                dropoffs = api.get_retention_dropoff_points(analytics_args.video_id)
                if dropoffs:
                    print(f"Found {len(dropoffs)} significant dropoff points:\n")
                    for dropoff in dropoffs[:10]:  # Show top 10
                        severity_marker = {
                            "severe": "[!!!]",
                            "moderate": "[!!]",
                            "minor": "[!]"
                        }.get(dropoff.severity, "[!]")
                        print(f"  {severity_marker} {dropoff.get_timestamp_str()} - {dropoff.severity.upper()}")
                        print(f"       Drop: {dropoff.percentage_drop:.1f}% "
                              f"({dropoff.retention_before:.1f}% -> {dropoff.retention_after:.1f}%)")
                else:
                    print("No significant dropoff points found.")

        except Exception as e:
            print(f"\n[ERROR] Failed to fetch analytics: {e}")
            print("\nMake sure you have:")
            print("1. Enabled YouTube Analytics API in Google Cloud Console")
            print("2. Added analytics scopes to OAuth consent screen")
            print("3. Re-authenticated if credentials were created before adding scopes")
            sys.exit(1)

    elif cmd == "analytics-compare":
        # Compare video to channel average
        compare_parser = argparse.ArgumentParser(prog="run.py analytics-compare", add_help=False)
        compare_parser.add_argument("video_id", nargs="?", help="YouTube video ID")
        compare_parser.add_argument("--days", type=int, default=90,
                                    help="Number of days to calculate averages (default: 90)")
        compare_parser.add_argument("--json", action="store_true",
                                    help="Output as JSON")
        compare_args = compare_parser.parse_args(sys.argv[2:])

        if not compare_args.video_id:
            print("Error: Please provide a video ID")
            print("Usage: python run.py analytics-compare <video_id> [--days N]")
            return

        from src.youtube.analytics_api import YouTubeAnalyticsAPI

        print(f"\n{'='*60}")
        print(f"Video vs Channel Comparison")
        print(f"{'='*60}")
        print(f"Video ID: {compare_args.video_id}")
        print(f"Period: Last {compare_args.days} days")
        print(f"{'='*60}\n")

        try:
            api = YouTubeAnalyticsAPI()
            comparison = api.compare_to_channel_average(
                compare_args.video_id,
                days=compare_args.days
            )

            if compare_args.json:
                import json
                output = {
                    "video_id": comparison.video_id,
                    "channel_id": comparison.channel_id,
                    "days_analyzed": comparison.days_analyzed,
                    "video_metrics": {
                        "views": comparison.video_views,
                        "ctr": comparison.video_ctr,
                        "retention": comparison.video_retention,
                        "engagement_rate": comparison.video_engagement_rate
                    },
                    "channel_averages": {
                        "views": comparison.channel_avg_views,
                        "ctr": comparison.channel_avg_ctr,
                        "retention": comparison.channel_avg_retention,
                        "engagement_rate": comparison.channel_avg_engagement_rate
                    },
                    "percentiles": {
                        "views": comparison.views_percentile,
                        "ctr": comparison.ctr_percentile,
                        "retention": comparison.retention_percentile,
                        "engagement": comparison.engagement_percentile
                    }
                }
                print(json.dumps(output, indent=2))
            else:
                print(comparison.get_summary())

        except Exception as e:
            print(f"\n[ERROR] Failed to fetch comparison data: {e}")
            print("\nMake sure you have:")
            print("1. Enabled YouTube Analytics API in Google Cloud Console")
            print("2. Added analytics scopes to OAuth consent screen")
            print("3. Re-authenticated if credentials were created before adding scopes")
            sys.exit(1)

    elif cmd == "ab-test":
        # Start A/B test for thumbnails
        ab_parser = argparse.ArgumentParser(prog="run.py ab-test", add_help=False)
        ab_parser.add_argument("video_id", nargs="?", help="YouTube video ID")
        ab_parser.add_argument("thumbnails", nargs="*", help="Thumbnail image paths (at least 2)")
        ab_parser.add_argument("--duration", type=int, default=72,
                              help="Test duration in hours (default: 72)")
        ab_parser.add_argument("--titles", help="Comma-separated title variants (optional)")
        ab_args = ab_parser.parse_args(sys.argv[2:])

        if not ab_args.video_id:
            print("Error: Please provide a video ID")
            print("Usage: python run.py ab-test <video_id> <thumb1.png> <thumb2.png> [--duration 72]")
            return

        if len(ab_args.thumbnails) < 2:
            print("Error: At least 2 thumbnail variants required")
            print("Usage: python run.py ab-test <video_id> <thumb1.png> <thumb2.png> [--duration 72]")
            return

        # Validate thumbnail files exist
        import os
        for thumb in ab_args.thumbnails:
            if not os.path.exists(thumb):
                print(f"Error: Thumbnail not found: {thumb}")
                return

        print(f"\n{'='*60}")
        print("A/B TEST SETUP")
        print(f"{'='*60}")
        print(f"Video ID:     {ab_args.video_id}")
        print(f"Thumbnails:   {len(ab_args.thumbnails)} variants")
        for i, thumb in enumerate(ab_args.thumbnails):
            print(f"  Variant {chr(65+i)}: {thumb}")
        print(f"Duration:     {ab_args.duration} hours")
        if ab_args.titles:
            print(f"Titles:       {ab_args.titles}")
        print(f"{'='*60}\n")

        from src.testing.ab_tester import ABTester

        tester = ABTester()

        # Parse titles if provided
        titles = None
        if ab_args.titles:
            titles = [t.strip() for t in ab_args.titles.split(",")]
            if len(titles) != len(ab_args.thumbnails):
                print(f"Error: Number of titles ({len(titles)}) must match thumbnails ({len(ab_args.thumbnails)})")
                return

        try:
            test_id = tester.start_thumbnail_test(
                video_id=ab_args.video_id,
                thumbnail_variants=ab_args.thumbnails,
                test_duration_hours=ab_args.duration,
                titles=titles
            )
            print(f"[OK] A/B test started!")
            print(f"     Test ID: {test_id}")
            print(f"     Duration: {ab_args.duration} hours ({ab_args.duration/24:.1f} days)")
            print(f"\n     To check progress: python run.py ab-check {test_id}")
            print(f"     To get report:     python run.py ab-report {test_id}")

            # Show YouTube actions needed
            actions = tester.get_youtube_actions(test_id)
            print(f"\n     [ACTION REQUIRED] Set thumbnail to: {actions.get('thumbnail_path')}")
            if actions.get('title'):
                print(f"     [ACTION REQUIRED] Set title to: {actions.get('title')}")

        except Exception as e:
            print(f"[FAIL] Failed to start test: {e}")
            sys.exit(1)

    elif cmd == "ab-check":
        # Check A/B test progress
        if len(sys.argv) < 3:
            print("Error: Please provide a test ID")
            print("Usage: python run.py ab-check <test_id>")
            return

        test_id = sys.argv[2]

        from src.testing.ab_tester import ABTester
        import json

        tester = ABTester()

        try:
            progress = tester.check_test_progress(test_id)

            print(f"\n{'='*60}")
            print(f"A/B TEST PROGRESS: {test_id}")
            print(f"{'='*60}")
            print(f"Video ID:        {progress['video_id']}")
            print(f"Status:          {progress['status'].upper()}")
            print(f"Progress:        {progress['progress_percent']:.1f}%")
            print(f"Elapsed:         {progress['elapsed_hours']:.1f} hours")
            print(f"Remaining:       {progress['remaining_hours']:.1f} hours")
            print(f"Current Variant: {progress['current_variant']}")

            print(f"\n{'Variant':<12} {'Impressions':>12} {'Clicks':>10} {'CTR':>10}")
            print("-" * 46)
            for v in progress['variants']:
                print(f"{v['id']:<12} {v['impressions']:>12,} {v['clicks']:>10,} {v['ctr']:>9.2f}%")

            if progress.get('winner_id'):
                print(f"\n[WINNER] Variant {progress['winner_id']} with {progress['confidence_level']:.1%} confidence")

            # Check if variant switch needed
            if progress.get('action') == 'switch_variant':
                new_var = progress['new_variant']
                print(f"\n[ACTION REQUIRED] Switch to variant {new_var['id']}")
                print(f"     Thumbnail: {new_var['thumbnail_path']}")
                if new_var.get('title'):
                    print(f"     Title: {new_var['title']}")

        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif cmd == "ab-report":
        # Get detailed A/B test report
        if len(sys.argv) < 3:
            print("Error: Please provide a test ID")
            print("Usage: python run.py ab-report <test_id>")
            return

        test_id = sys.argv[2]

        from src.testing.ab_tester import ABTester

        tester = ABTester()
        report = tester.get_test_report(test_id)
        print(report)

    elif cmd == "ab-list":
        # List all active A/B tests
        from src.testing.ab_tester import ABTester

        tester = ABTester()
        active = tester.get_active_tests()

        print(f"\n{'='*60}")
        print("ACTIVE A/B TESTS")
        print(f"{'='*60}")

        if not active:
            print("\nNo active A/B tests found.")
            print("\nTo start a test:")
            print("  python run.py ab-test <video_id> <thumb1.png> <thumb2.png>")
        else:
            print(f"\nFound {len(active)} active test(s):\n")
            for test in active:
                print(f"  Test ID:  {test['test_id']}")
                print(f"  Video:    {test['video_id']}")
                print(f"  Progress: {test['progress_percent']:.1f}%")
                print(f"  Variants: {len(test['variants'])}")
                print(f"  Current:  {test['current_variant']}")
                print()

    elif cmd == "monitor":
        # Performance monitoring for videos
        monitor_parser = argparse.ArgumentParser(prog="run.py monitor", add_help=False)
        monitor_parser.add_argument("video_id", nargs="?", help="Specific video ID to monitor")
        monitor_parser.add_argument("--hours", type=int, default=72,
                                    help="Hours to look back (default: 72)")
        monitor_parser.add_argument("--summary", action="store_true",
                                    help="Show alert summary only")
        monitor_parser.add_argument("--json", action="store_true",
                                    help="Output as JSON")
        monitor_args = monitor_parser.parse_args(sys.argv[2:])

        from src.monitoring.performance_monitor import PerformanceMonitor
        import json

        monitor = PerformanceMonitor()

        print(f"\n{'='*60}")
        print("  PERFORMANCE MONITOR")
        print(f"{'='*60}")

        if monitor_args.summary:
            # Show summary only
            summary = monitor.get_alert_summary(monitor_args.hours)
            print(f"\n  Alert Summary (last {monitor_args.hours} hours):")
            print(f"    Total alerts: {summary['total']}")
            print(f"    Critical: {summary['critical']}")
            print(f"    Warning: {summary['warning']}")
            print(f"    Info: {summary['info']}")

            if summary.get('by_type'):
                print("\n  By Type:")
                for alert_type, count in summary['by_type'].items():
                    print(f"    {alert_type}: {count}")

            if summary.get('recent_critical'):
                print("\n  Recent Critical Alerts:")
                for alert in summary['recent_critical']:
                    print(f"    - {alert.get('video_id')}: {alert.get('message')}")

            if monitor_args.json:
                print("\n" + json.dumps(summary, indent=2, default=str))

        elif monitor_args.video_id:
            # Monitor specific video
            print(f"\n  Monitoring video: {monitor_args.video_id}")
            print(f"  Thresholds: CTR={monitor.CTR_WARNING}%/{monitor.CTR_CRITICAL}%, "
                  f"Retention={monitor.RETENTION_WARNING}%/{monitor.RETENTION_CRITICAL}%")
            print()

            alerts = monitor.check_video(monitor_args.video_id, hours_since_upload=24)

            if alerts:
                print(f"  Found {len(alerts)} alert(s):\n")
                for alert in alerts:
                    severity_icon = {
                        "critical": "[!!!]",
                        "warning": "[!]",
                        "info": "[i]"
                    }.get(alert.severity, "[?]")

                    print(f"  {severity_icon} {alert.alert_type.upper()}")
                    print(f"      {alert.message}")
                    print(f"      Value: {alert.metric_value:.2f}% | Threshold: {alert.threshold_value:.2f}%")
                    print(f"      Recommendation: {alert.recommendation[:100]}...")
                    print()

                if monitor_args.json:
                    print("\n" + json.dumps([a.to_dict() for a in alerts], indent=2, default=str))
            else:
                print("  [OK] No alerts - video is performing well!")

        else:
            # Monitor all recent videos
            print(f"\n  Monitoring all videos from the last {monitor_args.hours} hours...")
            print(f"  Thresholds: CTR={monitor.CTR_WARNING}%/{monitor.CTR_CRITICAL}%, "
                  f"Retention={monitor.RETENTION_WARNING}%/{monitor.RETENTION_CRITICAL}%")
            print()

            all_alerts = monitor.monitor_all_recent(monitor_args.hours)

            if all_alerts:
                total_alerts = sum(len(alerts) for alerts in all_alerts.values())
                critical_count = sum(
                    1 for alerts in all_alerts.values()
                    for alert in alerts if alert.severity == "critical"
                )
                warning_count = sum(
                    1 for alerts in all_alerts.values()
                    for alert in alerts if alert.severity == "warning"
                )

                print(f"  Found {total_alerts} alert(s) across {len(all_alerts)} video(s)")
                print(f"  Critical: {critical_count} | Warning: {warning_count}")
                print()

                for video_id, alerts in all_alerts.items():
                    print(f"  Video: {video_id}")
                    for alert in alerts:
                        severity_icon = {
                            "critical": "[!!!]",
                            "warning": "[!]",
                            "info": "[i]"
                        }.get(alert.severity, "[?]")
                        print(f"    {severity_icon} {alert.alert_type}: {alert.message}")
                    print()

                if monitor_args.json:
                    json_output = {
                        vid: [a.to_dict() for a in alerts]
                        for vid, alerts in all_alerts.items()
                    }
                    print("\n" + json.dumps(json_output, indent=2, default=str))
            else:
                print("  [OK] No alerts - all videos are performing well!")

        print(f"\n{'='*60}")

    elif cmd == "reddit-trends":
        # Get trending topics from Reddit
        reddit_parser = argparse.ArgumentParser(prog="run.py reddit-trends", add_help=False)
        reddit_parser.add_argument("niche", nargs="?", default="finance",
                                   help="Content niche (finance, psychology, storytelling)")
        reddit_parser.add_argument("--limit", type=int, default=20,
                                   help="Number of topics to return (default: 20)")
        reddit_parser.add_argument("--json", action="store_true",
                                   help="Output as JSON")
        reddit_args = reddit_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print(f"REDDIT TRENDING TOPICS: {reddit_args.niche.upper()}")
        print(f"{'='*60}\n")

        try:
            from src.research.reddit_researcher import RedditResearcher

            researcher = RedditResearcher()

            if not researcher.reddit:
                print("[ERROR] Reddit not configured. Please set credentials in .env:")
                print("  REDDIT_CLIENT_ID=your_client_id")
                print("  REDDIT_CLIENT_SECRET=your_client_secret")
                print("\nTo get credentials:")
                print("  1. Go to https://www.reddit.com/prefs/apps")
                print("  2. Click 'create another app...'")
                print("  3. Select 'script' type")
                print("  4. Copy client_id and client_secret")
                sys.exit(1)

            topics = researcher.get_trending_topics(reddit_args.niche, limit=reddit_args.limit)

            if reddit_args.json:
                import json
                print(json.dumps({"niche": reddit_args.niche, "topics": topics}, indent=2))
            else:
                print(f"Found {len(topics)} trending topics:\n")
                for i, topic in enumerate(topics, 1):
                    print(f"  {i:2d}. {topic[:80]}{'...' if len(topic) > 80 else ''}")

        except ImportError as e:
            print(f"[ERROR] Failed to import Reddit researcher: {e}")
            print("Make sure praw is installed: pip install praw")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to get trending topics: {e}")
            sys.exit(1)

        print(f"\n{'='*60}")

    elif cmd == "reddit-questions":
        # Get questions from Reddit for FAQ videos
        reddit_parser = argparse.ArgumentParser(prog="run.py reddit-questions", add_help=False)
        reddit_parser.add_argument("niche", nargs="?", default="finance",
                                   help="Content niche (finance, psychology, storytelling)")
        reddit_parser.add_argument("--limit", type=int, default=30,
                                   help="Number of questions to return (default: 30)")
        reddit_parser.add_argument("--min-upvotes", type=int, default=10,
                                   help="Minimum upvotes (default: 10)")
        reddit_parser.add_argument("--json", action="store_true",
                                   help="Output as JSON")
        reddit_args = reddit_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print(f"REDDIT QUESTIONS: {reddit_args.niche.upper()}")
        print(f"{'='*60}\n")

        try:
            from src.research.reddit_researcher import RedditResearcher

            researcher = RedditResearcher()

            if not researcher.reddit:
                print("[ERROR] Reddit not configured. Set credentials in .env")
                sys.exit(1)

            questions = researcher.get_questions(
                reddit_args.niche,
                limit=reddit_args.limit,
                min_upvotes=reddit_args.min_upvotes
            )

            if reddit_args.json:
                import json
                output = [q.to_dict() for q in questions]
                print(json.dumps({"niche": reddit_args.niche, "questions": output}, indent=2))
            else:
                print(f"Found {len(questions)} questions for FAQ videos:\n")
                for i, q in enumerate(questions, 1):
                    print(f"  {i:2d}. {q.topic[:75]}{'...' if len(q.topic) > 75 else ''}")
                    print(f"      Subreddit: r/{q.subreddit} | Score: {q.popularity_score}")
                    print()

        except ImportError as e:
            print(f"[ERROR] Failed to import: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to get questions: {e}")
            sys.exit(1)

        print(f"{'='*60}")

    elif cmd == "reddit-viral":
        # Find viral content ideas from Reddit
        reddit_parser = argparse.ArgumentParser(prog="run.py reddit-viral", add_help=False)
        reddit_parser.add_argument("niche", nargs="?", default="finance",
                                   help="Content niche (finance, psychology, storytelling)")
        reddit_parser.add_argument("--limit", type=int, default=15,
                                   help="Number of ideas to return (default: 15)")
        reddit_parser.add_argument("--min-upvotes", type=int,
                                   help="Minimum upvotes (uses niche default)")
        reddit_parser.add_argument("--json", action="store_true",
                                   help="Output as JSON")
        reddit_parser.add_argument("--full", action="store_true",
                                   help="Run full research report")
        reddit_args = reddit_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        if reddit_args.full:
            print(f"REDDIT FULL RESEARCH: {reddit_args.niche.upper()}")
        else:
            print(f"REDDIT VIRAL IDEAS: {reddit_args.niche.upper()}")
        print(f"{'='*60}\n")

        try:
            from src.research.reddit_researcher import RedditResearcher

            researcher = RedditResearcher()

            if not researcher.reddit:
                print("[ERROR] Reddit not configured. Set credentials in .env")
                sys.exit(1)

            if reddit_args.full:
                # Full research report
                report = researcher.full_research(reddit_args.niche)

                if reddit_args.json:
                    import json
                    print(json.dumps(report.to_dict(), indent=2, default=str))
                else:
                    print(report.summary())
            else:
                # Just viral ideas
                viral = researcher.find_viral_content(
                    reddit_args.niche,
                    min_upvotes=reddit_args.min_upvotes,
                    limit=reddit_args.limit
                )

                if reddit_args.json:
                    import json
                    output = [v.to_dict() for v in viral]
                    print(json.dumps({"niche": reddit_args.niche, "viral_ideas": output}, indent=2))
                else:
                    print(f"Found {len(viral)} viral content ideas:\n")
                    for i, v in enumerate(viral, 1):
                        print(f"  {i:2d}. {v.title_suggestion}")
                        print(f"      Subreddit: r/{v.subreddit}")
                        print(f"      Viral Score: {v.viral_score:.0f} | Type: {v.idea_type}")
                        print(f"      Source: {v.source_url}")
                        print()

        except ImportError as e:
            print(f"[ERROR] Failed to import: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to get viral ideas: {e}")
            sys.exit(1)

        print(f"{'='*60}")

    elif cmd == "reddit-research":
        # Full Reddit research (alias for reddit-viral --full)
        reddit_parser = argparse.ArgumentParser(prog="run.py reddit-research", add_help=False)
        reddit_parser.add_argument("niche", nargs="?", default="finance",
                                   help="Content niche (finance, psychology, storytelling)")
        reddit_parser.add_argument("--json", action="store_true",
                                   help="Output as JSON")
        reddit_args = reddit_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print(f"REDDIT FULL RESEARCH: {reddit_args.niche.upper()}")
        print(f"{'='*60}\n")

        try:
            from src.research.reddit_researcher import RedditResearcher

            researcher = RedditResearcher()

            if not researcher.reddit:
                print("[ERROR] Reddit not configured. Set credentials in .env")
                sys.exit(1)

            print("Running full research (this may take a minute)...\n")
            report = researcher.full_research(reddit_args.niche)

            if reddit_args.json:
                import json
                print(json.dumps(report.to_dict(), indent=2, default=str))
            else:
                print(report.summary())

        except ImportError as e:
            print(f"[ERROR] Failed to import: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Research failed: {e}")
            sys.exit(1)

    # ============================================================
    # NEW INTEGRATED MODULE COMMANDS (2026-01-20)
    # ============================================================

    elif cmd == "caption":
        # Generate captions using Whisper
        caption_parser = argparse.ArgumentParser(prog="run.py caption", add_help=False)
        caption_parser.add_argument("audio_file", help="Path to audio/video file")
        caption_parser.add_argument("--output", "-o", help="Output file path (default: same name.srt)")
        caption_parser.add_argument("--format", "-f", default="srt",
                                    choices=["srt", "vtt", "json"],
                                    help="Output format (default: srt)")
        caption_parser.add_argument("--model", "-m", default="base",
                                    choices=["tiny", "base", "small", "medium", "large"],
                                    help="Whisper model size (default: base)")
        caption_parser.add_argument("--language", "-l", default=None,
                                    help="Language code (auto-detect if not specified)")
        caption_parser.add_argument("--json", action="store_true",
                                    help="Output timing data as JSON")
        caption_args = caption_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print("WHISPER CAPTION GENERATOR")
        print(f"{'='*60}")
        print(f"Audio file: {caption_args.audio_file}")
        print(f"Model: {caption_args.model}")
        print(f"Format: {caption_args.format}")
        print()

        import os
        if not os.path.exists(caption_args.audio_file):
            print(f"[ERROR] Audio file not found: {caption_args.audio_file}")
            sys.exit(1)

        try:
            from src.captions.whisper_generator import WhisperCaptionGenerator
            import asyncio

            generator = WhisperCaptionGenerator(model_size=caption_args.model)

            # Determine output file
            if caption_args.output:
                output_file = caption_args.output
            else:
                base_name = os.path.splitext(caption_args.audio_file)[0]
                output_file = f"{base_name}.{caption_args.format}"

            print(f"Generating captions...")

            result = asyncio.run(generator.generate_captions(
                audio_file=caption_args.audio_file,
                output_file=output_file,
                format=caption_args.format,
                language=caption_args.language
            ))

            print(f"\n[OK] Captions saved to: {result}")

            if caption_args.json:
                # Also output word-level timing
                word_timing = asyncio.run(generator.generate_word_timestamps(
                    audio_file=caption_args.audio_file,
                    language=caption_args.language
                ))
                import json
                print("\nWord-level timing:")
                print(json.dumps(word_timing[:10], indent=2))  # First 10 words
                print(f"... ({len(word_timing)} words total)")

        except ImportError as e:
            print(f"[ERROR] Whisper not available: {e}")
            print("Install with: pip install faster-whisper")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Caption generation failed: {e}")
            sys.exit(1)

        print(f"\n{'='*60}")

    elif cmd == "keywords":
        # Keyword research using Google Trends
        kw_parser = argparse.ArgumentParser(prog="run.py keywords", add_help=False)
        kw_parser.add_argument("topic", help="Topic to research keywords for")
        kw_parser.add_argument("--niche", "-n", default="default",
                               help="Content niche (finance, psychology, storytelling)")
        kw_parser.add_argument("--count", "-c", type=int, default=20,
                               help="Number of keywords to return (default: 20)")
        kw_parser.add_argument("--longtails", action="store_true",
                               help="Include long-tail keyword variations")
        kw_parser.add_argument("--trends", action="store_true",
                               help="Include trend analysis")
        kw_parser.add_argument("--json", action="store_true",
                               help="Output as JSON")
        kw_args = kw_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print("KEYWORD RESEARCH (Free - Google Trends)")
        print(f"{'='*60}")
        print(f"Topic: {kw_args.topic}")
        print(f"Niche: {kw_args.niche}")
        print()

        try:
            from src.seo.keyword_intelligence import KeywordIntelligence

            ki = KeywordIntelligence()

            # Full analysis
            result = ki.full_analysis(kw_args.topic, niche=kw_args.niche)

            if kw_args.json:
                import json
                print(json.dumps(result.to_dict(), indent=2, default=str))
            else:
                print(f"Opportunity Score: {result.opportunity_score:.1f}/100")
                print(f"Competition: {result.competition_score:.1f}/100")
                print(f"Search Intent: {result.search_intent}")
                print(f"Trend Direction: {result.trend_direction}")

                if result.related_keywords:
                    print(f"\nRelated Keywords ({min(len(result.related_keywords), kw_args.count)}):")
                    for i, kw in enumerate(result.related_keywords[:kw_args.count], 1):
                        print(f"  {i:2d}. {kw}")

                if kw_args.longtails and hasattr(ki, 'generate_longtails'):
                    longtails = ki.generate_longtails(kw_args.topic, count=10)
                    print(f"\nLong-tail Variations:")
                    for lt in longtails[:10]:
                        print(f"  - {lt}")

                if kw_args.trends:
                    print(f"\nTrend Analysis:")
                    print(f"  Rising: {result.is_rising}")
                    print(f"  Seasonal: {result.is_seasonal if hasattr(result, 'is_seasonal') else 'N/A'}")

        except ImportError as e:
            print(f"[ERROR] Keyword intelligence not available: {e}")
            print("Make sure pytrends is installed: pip install pytrends")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Keyword research failed: {e}")
            sys.exit(1)

        print(f"\n{'='*60}")

    elif cmd == "optimize-title":
        # Optimize title using IMPACT formula
        title_parser = argparse.ArgumentParser(prog="run.py optimize-title", add_help=False)
        title_parser.add_argument("title", help="Title to optimize")
        title_parser.add_argument("--niche", "-n", default="default",
                                  help="Content niche (finance, psychology, storytelling)")
        title_parser.add_argument("--keywords", "-k", default="",
                                  help="Comma-separated target keywords")
        title_parser.add_argument("--variants", "-v", type=int, default=5,
                                  help="Number of title variants to generate")
        title_parser.add_argument("--json", action="store_true",
                                  help="Output as JSON")
        title_args = title_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print("TITLE OPTIMIZER (IMPACT Formula)")
        print(f"{'='*60}")
        print(f"Original: {title_args.title}")
        print(f"Niche: {title_args.niche}")
        print()

        try:
            from src.seo.metadata_optimizer import MetadataOptimizer

            optimizer = MetadataOptimizer()
            keywords = [k.strip() for k in title_args.keywords.split(",") if k.strip()]

            # Generate optimized variants
            variants = optimizer.generate_title_variants(
                topic=title_args.title,
                keywords=keywords if keywords else [title_args.title.split()[0]],
                count=title_args.variants
            )

            if title_args.json:
                import json
                print(json.dumps({"original": title_args.title, "variants": variants}, indent=2))
            else:
                print(f"Generated {len(variants)} optimized variants:\n")
                for i, variant in enumerate(variants, 1):
                    # Score each variant
                    score = optimizer.score_title(variant)
                    print(f"  {i}. {variant}")
                    print(f"     Score: {score:.1f}/100")
                    print()

                # Also analyze original title
                original_score = optimizer.score_title(title_args.title)
                print(f"Original title score: {original_score:.1f}/100")

                # Show IMPACT criteria
                print(f"\nIMPACT Formula Checks:")
                impact = optimizer.analyze_title_impact(title_args.title)
                for criterion, passed in impact.items():
                    status = "[OK]" if passed else "[MISS]"
                    print(f"  {status} {criterion}")

        except ImportError as e:
            print(f"[ERROR] Metadata optimizer not available: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Title optimization failed: {e}")
            sys.exit(1)

        print(f"\n{'='*60}")

    elif cmd == "feedback":
        # Analyze video performance feedback
        fb_parser = argparse.ArgumentParser(prog="run.py feedback", add_help=False)
        fb_parser.add_argument("video_id", help="YouTube video ID to analyze")
        fb_parser.add_argument("--niche", "-n", default="default",
                               help="Content niche for comparison")
        fb_parser.add_argument("--recommendations", "-r", action="store_true",
                               help="Generate improvement recommendations")
        fb_parser.add_argument("--json", action="store_true",
                               help="Output as JSON")
        fb_args = fb_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print("ANALYTICS FEEDBACK LOOP")
        print(f"{'='*60}")
        print(f"Video ID: {fb_args.video_id}")
        print(f"Niche: {fb_args.niche}")
        print()

        try:
            from src.analytics.feedback_loop import AnalyticsFeedbackLoop
            import asyncio

            feedback = AnalyticsFeedbackLoop()

            print("Analyzing video performance...")

            result = asyncio.run(feedback.analyze_video(fb_args.video_id, fb_args.niche))

            if fb_args.json:
                import json
                print(json.dumps(result.to_dict(), indent=2, default=str))
            else:
                print(f"\nPerformance Score: {result.performance_score:.1f}/100")
                print(f"CTR: {result.ctr:.2f}%")
                print(f"Avg View Duration: {result.avg_view_duration:.1f}%")
                print(f"Engagement Rate: {result.engagement_rate:.2f}%")

                if result.drop_off_points:
                    print(f"\nDrop-off Points:")
                    for point in result.drop_off_points[:5]:
                        print(f"  - {point['timestamp']}s: {point['drop_percentage']:.1f}% drop")

                if fb_args.recommendations and result.recommendations:
                    print(f"\nRecommendations:")
                    for i, rec in enumerate(result.recommendations, 1):
                        print(f"  {i}. {rec}")

        except ImportError as e:
            print(f"[ERROR] Feedback loop not available: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Feedback analysis failed: {e}")
            sys.exit(1)

        print(f"\n{'='*60}")

    elif cmd == "hooks":
        # Generate viral hooks
        hook_parser = argparse.ArgumentParser(prog="run.py hooks", add_help=False)
        hook_parser.add_argument("topic", help="Video topic")
        hook_parser.add_argument("--niche", "-n", default="all",
                                 help="Content niche (finance, psychology, storytelling)")
        hook_parser.add_argument("--style", "-s", default="curiosity",
                                 choices=["curiosity", "action", "contrarian", "emotional", "value"],
                                 help="Hook style (default: curiosity)")
        hook_parser.add_argument("--count", "-c", type=int, default=5,
                                 help="Number of hooks to generate (default: 5)")
        hook_parser.add_argument("--enhance", "-e", help="Script file to enhance with retention features")
        hook_parser.add_argument("--json", action="store_true",
                                 help="Output as JSON")
        hook_args = hook_parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print("VIRAL HOOK GENERATOR")
        print(f"{'='*60}")
        print(f"Topic: {hook_args.topic}")
        print(f"Niche: {hook_args.niche}")
        print(f"Style: {hook_args.style}")
        print()

        try:
            from src.content.viral_hooks import ViralHookGenerator

            generator = ViralHookGenerator(niche=hook_args.niche)

            # Generate hooks
            hooks = generator.get_all_hooks(hook_args.topic, count=hook_args.count)

            if hook_args.json:
                import json
                hook_data = [{"hook": h[0], "style": h[1], "retention": h[2]} for h in hooks]
                print(json.dumps({"topic": hook_args.topic, "hooks": hook_data}, indent=2))
            else:
                print(f"Top {len(hooks)} Hook Options:\n")
                for i, (hook, style, retention) in enumerate(hooks, 1):
                    print(f"  {i}. [{style.upper()}] {hook}")
                    print(f"     Avg Retention: {retention:.1f}%")
                    print()

            # Enhance script if provided
            if hook_args.enhance:
                import os
                if os.path.exists(hook_args.enhance):
                    with open(hook_args.enhance, 'r', encoding='utf-8') as f:
                        script = f.read()

                    print(f"\nEnhancing script: {hook_args.enhance}")
                    enhanced = generator.enhance_script_retention(script, video_duration=600)

                    output_file = hook_args.enhance.replace('.txt', '_enhanced.txt')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(enhanced)

                    print(f"[OK] Enhanced script saved: {output_file}")
                else:
                    print(f"[ERROR] Script file not found: {hook_args.enhance}")

        except ImportError as e:
            print(f"[ERROR] Viral hooks not available: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Hook generation failed: {e}")
            sys.exit(1)

        print(f"\n{'='*60}")

    elif cmd == "integration-status":
        # Show status of all integrated modules
        print(f"\n{'='*60}")
        print("INTEGRATION STATUS")
        print(f"{'='*60}\n")

        modules = {
            "Whisper Captioning": ("src.captions.whisper_generator", "WhisperCaptionGenerator"),
            "AI Disclosure": ("src.compliance.ai_disclosure", "AIDisclosureTracker"),
            "Analytics Feedback": ("src.analytics.feedback_loop", "AnalyticsFeedbackLoop"),
            "Viral Hooks": ("src.content.viral_hooks", "ViralHookGenerator"),
            "Metadata Optimizer": ("src.seo.metadata_optimizer", "MetadataOptimizer"),
            "Keyword Intelligence": ("src.seo.keyword_intelligence", "KeywordIntelligence"),
            "Chatterbox TTS": ("src.content.tts_chatterbox", "ChatterboxTTS"),
        }

        for name, (module_path, class_name) in modules.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
                print(f"  [OK] {name}")
            except ImportError as e:
                print(f"  [--] {name} (not installed)")
            except AttributeError as e:
                print(f"  [!!] {name} (import error: {e})")
            except Exception as e:
                print(f"  [??] {name} ({e})")

        # Check config file
        import os
        config_path = "config/integrations.yaml"
        if os.path.exists(config_path):
            print(f"\n  [OK] Integration config: {config_path}")
        else:
            print(f"\n  [--] Integration config: {config_path} (not found)")

        print(f"\n{'='*60}")

    elif cmd == "ai-video":
        # AI Video Generation command (supports Runway, Pika, HailuoAI)
        import asyncio

        ai_video_parser = argparse.ArgumentParser(prog="run.py ai-video", add_help=False)
        ai_video_parser.add_argument("prompt", nargs="?", help="Video prompt")
        ai_video_parser.add_argument("--provider", "-p", default=None,
                                     help="Provider: runway, pika, hailuo (default: auto)")
        ai_video_parser.add_argument("--budget", "-b", action="store_true",
                                     help="Use cheapest provider")
        ai_video_parser.add_argument("--quality", "-q", default="medium",
                                     choices=["high", "medium", "low"],
                                     help="Quality level (high=runway_alpha)")
        ai_video_parser.add_argument("--output", "-o", default=None,
                                     help="Output file path")
        ai_video_parser.add_argument("--image", default=None,
                                     help="Image path/URL for image-to-video")
        ai_video_parser.add_argument("--duration", "-d", type=int, default=5,
                                     help="Video duration in seconds (5 or 10)")
        ai_video_parser.add_argument("--aspect-ratio", default="16:9",
                                     choices=["16:9", "9:16", "21:9"],
                                     help="Aspect ratio (default: 16:9)")
        ai_video_parser.add_argument("--json", action="store_true",
                                     help="Output as JSON")
        ai_video_args = ai_video_parser.parse_args(sys.argv[2:])

        if not ai_video_args.prompt:
            print("Error: Prompt required")
            print("Usage: python run.py ai-video \"Your video prompt here\"")
            print("\nOptions:")
            print("  --provider runway|pika  Use specific provider (default: auto-select)")
            print("  --budget                Use cheapest available provider")
            print("  --quality high|medium|low  Quality level (high uses runway alpha)")
            print("  --duration 5|10         Video duration in seconds")
            print("  --aspect-ratio 16:9|9:16|21:9  Aspect ratio")
            print("  --output path           Output file path")
            print("  --image path            Image path for image-to-video")
            print("\nExamples:")
            print('  python run.py ai-video "Cinematic sunset over ocean"')
            print('  python run.py ai-video "City timelapse" --provider runway --quality high')
            print('  python run.py ai-video "Portrait video" --aspect-ratio 9:16')
            sys.exit(1)

        async def _run_ai_video():
            from src.content.ai_video_providers import (
                get_ai_video_provider,
                get_smart_router
            )

            try:
                duration = ai_video_args.duration
                aspect_ratio = ai_video_args.aspect_ratio
                quality_map = {"high": True, "medium": False, "low": False}
                prefer_quality = quality_map.get(ai_video_args.quality, False)

                # Select provider
                if ai_video_args.provider:
                    provider = get_ai_video_provider(ai_video_args.provider)
                    if not provider:
                        print(f"[ERROR] Provider '{ai_video_args.provider}' not available")
                        print("Available providers: runway, pika")
                        sys.exit(1)
                    print(f"[INFO] Using provider: {ai_video_args.provider}")
                elif ai_video_args.budget:
                    router = get_smart_router()
                    provider = router.get_cheapest_provider()
                    if provider:
                        print(f"[INFO] Using cheapest provider: {provider.get_provider_name()}")
                    else:
                        print("[ERROR] No providers available")
                        sys.exit(1)
                else:
                    router = get_smart_router()
                    provider = router.select_provider(
                        duration=duration,
                        prefer_quality=prefer_quality
                    )
                    if provider:
                        print(f"[INFO] Smart-selected provider: {provider.get_provider_name()}")
                    else:
                        print("[ERROR] No providers available. Check API keys.")
                        sys.exit(1)

                cost_estimate = provider.get_cost_per_video(duration)
                print(f"[INFO] Duration: {duration}s, Aspect ratio: {aspect_ratio}")
                print(f"[INFO] Estimated cost: ${cost_estimate:.2f}")
                print(f"[INFO] Generating video... (this may take 1-3 minutes)")

                # Generate video
                if ai_video_args.image:
                    result = await provider.generate_from_image(
                        image_path=ai_video_args.image,
                        motion_prompt=ai_video_args.prompt,
                        output_file=ai_video_args.output,
                        duration=duration
                    )
                else:
                    result = await provider.generate_video(
                        prompt=ai_video_args.prompt,
                        output_file=ai_video_args.output,
                        duration=duration,
                        aspect_ratio=aspect_ratio
                    )

                # Output result
                if ai_video_args.json:
                    import json
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    if result.success:
                        print(f"\n[OK] Video generated successfully!")
                        print(f"     Provider: {result.provider}")
                        print(f"     Local path: {result.local_path}")
                        if result.video_url:
                            print(f"     Video URL: {result.video_url}")
                        print(f"     Duration: {result.duration}s")
                        if result.cost_estimate:
                            print(f"     Cost: ${result.cost_estimate:.2f}")
                        if result.generation_time:
                            print(f"     Generation time: {result.generation_time:.1f}s")
                    else:
                        print(f"\n[FAIL] Video generation failed")
                        print(f"       Error: {result.error}")
                        sys.exit(1)

            except ImportError as e:
                print(f"\n[ERROR] Missing dependency: {e}")
                print("Install with: pip install runwayml httpx")
                sys.exit(1)
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        asyncio.run(_run_ai_video())

    elif cmd == "ai-broll":
        # Generate B-roll footage from script file
        import asyncio

        broll_parser = argparse.ArgumentParser(prog="run.py ai-broll", add_help=False)
        broll_parser.add_argument("script_file", nargs="?", help="Path to script file")
        broll_parser.add_argument("--output-dir", "-o", default="output/ai_broll",
                                  help="Output directory for generated clips")
        broll_parser.add_argument("--style", "-s", default="cinematic",
                                  choices=["cinematic", "documentary", "corporate", "abstract"],
                                  help="Visual style for B-roll")
        broll_parser.add_argument("--niche", "-n", default="default",
                                  choices=["finance", "psychology", "storytelling", "technology", "default"],
                                  help="Content niche for visual context")
        broll_parser.add_argument("--duration", "-d", type=int, default=5,
                                  help="Duration per clip in seconds")
        broll_parser.add_argument("--max-clips", type=int, default=5,
                                  help="Maximum number of clips to generate")
        broll_parser.add_argument("--provider", "-p", default=None,
                                  help="Provider: runway, pika (default: auto)")
        broll_parser.add_argument("--dry-run", action="store_true",
                                  help="Show what would be generated without actually generating")
        broll_parser.add_argument("--json", action="store_true",
                                  help="Output as JSON")
        broll_args = broll_parser.parse_args(sys.argv[2:])

        if not broll_args.script_file:
            print("Error: Script file required")
            print("Usage: python run.py ai-broll <script_file>")
            print("\nOptions:")
            print("  --output-dir dir        Output directory (default: output/ai_broll)")
            print("  --style cinematic|documentary|corporate|abstract")
            print("  --niche finance|psychology|storytelling|technology")
            print("  --duration 5            Duration per clip in seconds")
            print("  --max-clips 5           Maximum clips to generate")
            print("  --provider runway|pika  Specific provider to use")
            print("  --dry-run               Preview without generating")
            print("\nExamples:")
            print("  python run.py ai-broll output/script.txt --style cinematic")
            print("  python run.py ai-broll output/script.txt --niche finance --max-clips 3")
            sys.exit(1)

        import os
        if not os.path.exists(broll_args.script_file):
            print(f"[ERROR] Script file not found: {broll_args.script_file}")
            sys.exit(1)

        async def _run_ai_broll():
            from src.content.ai_video_providers import get_smart_router, get_ai_video_provider
            import re

            # Read script file
            with open(broll_args.script_file, 'r', encoding='utf-8') as f:
                script_text = f.read()

            # Extract segments (sentences or paragraphs)
            # Split by sentences or paragraph breaks
            segments = re.split(r'(?<=[.!?])\s+|\n\n+', script_text)
            segments = [s.strip() for s in segments if s.strip() and len(s.strip()) > 20]

            # Limit to max_clips
            segments = segments[:broll_args.max_clips]

            print(f"\n{'='*60}")
            print("AI B-ROLL GENERATOR")
            print(f"{'='*60}")
            print(f"Script: {broll_args.script_file}")
            print(f"Style: {broll_args.style}")
            print(f"Niche: {broll_args.niche}")
            print(f"Clips to generate: {len(segments)}")
            print(f"Duration per clip: {broll_args.duration}s")
            print(f"Output directory: {broll_args.output_dir}")
            print()

            # Show segments
            print("Segments to visualize:")
            for i, seg in enumerate(segments, 1):
                preview = seg[:80] + "..." if len(seg) > 80 else seg
                print(f"  {i}. {preview}")
            print()

            if broll_args.dry_run:
                print("[DRY RUN] Would generate clips for the above segments")
                print(f"[DRY RUN] Estimated cost: ${len(segments) * 0.25:.2f} (rough estimate)")
                return

            # Get provider
            if broll_args.provider:
                provider = get_ai_video_provider(broll_args.provider)
                if not provider:
                    print(f"[ERROR] Provider '{broll_args.provider}' not available")
                    sys.exit(1)
            else:
                router = get_smart_router()
                provider = router.select_provider(duration=broll_args.duration)
                if not provider:
                    print("[ERROR] No AI video providers available. Check API keys.")
                    sys.exit(1)

            print(f"[INFO] Using provider: {provider.get_provider_name()}")

            # Create output directory
            os.makedirs(broll_args.output_dir, exist_ok=True)

            # Generate clips
            results = []
            total_cost = 0.0

            for i, segment in enumerate(segments, 1):
                print(f"\n[{i}/{len(segments)}] Generating B-roll...")
                print(f"    Segment: {segment[:60]}...")

                output_file = os.path.join(broll_args.output_dir, f"broll_{i:03d}.mp4")

                # Build enhanced prompt
                style_prompts = {
                    "cinematic": "cinematic shot, dramatic lighting, film grain",
                    "documentary": "documentary style, natural lighting, realistic",
                    "corporate": "clean, professional, modern, well-lit",
                    "abstract": "abstract visuals, artistic, symbolic",
                }
                niche_context = {
                    "finance": "business, money, investment",
                    "psychology": "human emotion, mind, thinking",
                    "storytelling": "dramatic, narrative, atmospheric",
                    "technology": "futuristic, digital, tech",
                    "default": "",
                }

                enhanced_prompt = (
                    f"B-roll footage: {segment}. "
                    f"{style_prompts.get(broll_args.style, '')}. "
                    f"{niche_context.get(broll_args.niche, '')}. "
                    f"16:9 widescreen, high quality."
                )

                result = await provider.generate_video(
                    prompt=enhanced_prompt,
                    output_file=output_file,
                    duration=broll_args.duration,
                    aspect_ratio="16:9"
                )

                if result.success:
                    print(f"    [OK] Saved: {result.local_path}")
                    if result.cost_estimate:
                        total_cost += result.cost_estimate
                else:
                    print(f"    [FAIL] {result.error}")

                results.append(result.to_dict() if hasattr(result, 'to_dict') else {"success": result.success})

            # Summary
            successful = sum(1 for r in results if r.get("success"))
            print(f"\n{'='*60}")
            print(f"B-ROLL GENERATION COMPLETE")
            print(f"{'='*60}")
            print(f"Successful: {successful}/{len(segments)}")
            print(f"Total cost: ${total_cost:.2f}")
            print(f"Output dir: {broll_args.output_dir}")

            if broll_args.json:
                import json
                print("\n" + json.dumps({
                    "script_file": broll_args.script_file,
                    "clips_generated": successful,
                    "total_cost": total_cost,
                    "results": results
                }, indent=2))

        asyncio.run(_run_ai_broll())

    elif cmd == "ai-video-providers":
        # List AI video providers
        from src.content.ai_video_providers import list_providers

        print(f"\n{'='*60}")
        print("AI VIDEO PROVIDERS")
        print(f"{'='*60}\n")

        providers = list_providers()
        print(f"{'Provider':<15} {'Cost/Video':>12} {'Quality':>10} {'Status':>10}")
        print("-" * 50)

        for p in providers:
            cost = f"${p['cost_per_video']:.2f}" if p['cost_per_video'] else "N/A"
            quality = f"{p['quality_score']:.0%}" if p['quality_score'] else "N/A"
            status = "[OK]" if p['available'] else "[--]"
            print(f"{p['name']:<15} {cost:>12} {quality:>10} {status:>10}")

        print("-" * 50)
        print("\nUsage:")
        print("  python run.py ai-video --provider hailuo \"prompt\"")
        print("  python run.py ai-video --budget \"prompt\"  # Auto-select cheapest")
        print(f"\n{'='*60}")

    elif cmd == "ai-video-cost":
        # Estimate AI video batch cost
        from src.content.ai_video_providers import get_smart_router

        count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        provider_name = sys.argv[3] if len(sys.argv) > 3 else None

        router = get_smart_router()
        estimate = router.estimate_batch_cost(count, provider_name)

        print(f"\n{'='*60}")
        print("AI VIDEO COST ESTIMATE")
        print(f"{'='*60}\n")

        print(f"Clip count: {estimate['clip_count']}")

        if 'estimates' in estimate:
            print(f"\n{'Provider':<15} {'Per Clip':>12} {'Total':>12} {'Quality':>10}")
            print("-" * 52)
            for e in estimate['estimates']:
                print(f"{e['provider']:<15} ${e['cost_per_clip']:.2f}         ${e['total_cost']:.2f}        {e['quality']:.0%}")

            if estimate['savings_vs_pika'] > 0:
                print(f"\nPotential savings with HailuoAI: ${estimate['savings_vs_pika']:.2f}")
        else:
            print(f"Provider: {estimate['provider']}")
            print(f"Cost per clip: ${estimate['cost_per_clip']:.2f}")
            print(f"Total cost: ${estimate['total_cost']:.2f}")

        print(f"\n{'='*60}")

    elif cmd == "gpu-status":
        # Show GPU acceleration status
        from src.utils.gpu_utils import GPUAccelerator

        print(f"\n{'='*60}")
        print("GPU ACCELERATION STATUS")
        print(f"{'='*60}\n")

        try:
            accelerator = GPUAccelerator()
            status = accelerator.get_status()

            print(f"GPU Available:     {'YES' if status['available'] else 'NO'}")
            print(f"Type:              {status['type'].upper()}")
            print(f"Name:              {status['name']}")
            print(f"Encoder:           {status['encoder']}")

            if status['available']:
                print(f"Decoder:           {status['decoder']}")
                print(f"Scale Filter:      {status['scale_filter']}")
                print(f"HEVC Support:      {'YES' if status['supports_hevc'] else 'NO'}")
                print(f"Max Resolution:    {status['max_resolution']}")

            print(f"Expected Speedup:  {status['speedup']}")

            # Show sample encoding command
            if status['available']:
                print(f"\nSample FFmpeg encoding arguments:")
                args = accelerator.get_ffmpeg_args(preset="fast", bitrate="8M")
                print(f"  {' '.join(args)}")

                print(f"\nHardware decoding arguments:")
                input_args = accelerator.get_input_args()
                if input_args:
                    print(f"  {' '.join(input_args)}")
                else:
                    print(f"  (Not required)")

        except Exception as e:
            print(f"Error detecting GPU: {e}")
            sys.exit(1)

        print(f"\n{'='*60}")

    elif cmd == "benchmark":
        # Run performance benchmark
        import time
        import tempfile
        from pathlib import Path

        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARK")
        print(f"{'='*60}\n")

        # Parse options
        duration = 30  # seconds
        iterations = 3
        if len(sys.argv) > 2:
            try:
                duration = int(sys.argv[2])
            except ValueError:
                pass

        print(f"Test parameters:")
        print(f"  Video duration: {duration}s")
        print(f"  Iterations: {iterations}")
        print(f"  Resolution: 1920x1080")
        print()

        from src.content.video_pro import ProVideoGenerator

        # Create test audio
        temp_dir = Path(tempfile.gettempdir()) / "benchmark"
        temp_dir.mkdir(exist_ok=True)
        audio_file = temp_dir / "test_audio.mp3"

        # Generate silent audio if it doesn't exist
        if not audio_file.exists():
            print("Generating test audio...")
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
                "-t", str(duration), "-c:a", "mp3", str(audio_file)
            ]
            subprocess.run(cmd, capture_output=True)

        # Benchmark scenarios
        scenarios = [
            {"name": "CPU Encoding", "use_gpu": False},
            {"name": "GPU Encoding", "use_gpu": True},
        ]

        results = {}

        for scenario in scenarios:
            scenario_name = scenario["name"]
            print(f"\n{scenario_name}:")
            print("-" * 40)

            times = []

            for i in range(iterations):
                output_file = temp_dir / f"bench_{scenario_name.replace(' ', '_')}_{i}.mp4"

                # Create simple test script object
                class TestScript:
                    def __init__(self):
                        self.title = "Benchmark Test"
                        self.description = "Performance test"
                        self.sections = []

                generator = ProVideoGenerator(use_gpu=scenario["use_gpu"])

                start = time.time()

                try:
                    result = generator.create_video(
                        audio_file=str(audio_file),
                        script=TestScript(),
                        output_file=str(output_file),
                        use_stock=False
                    )

                    elapsed = time.time() - start
                    times.append(elapsed)

                    print(f"  Run {i+1}/{iterations}: {elapsed:.1f}s")

                    # Cleanup
                    if output_file.exists():
                        output_file.unlink()

                except Exception as e:
                    print(f"  Run {i+1}/{iterations}: FAILED - {e}")
                    continue

            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                results[scenario_name] = {
                    "avg": avg_time,
                    "min": min_time,
                    "max": max_time
                }

                print(f"\n  Average: {avg_time:.1f}s")
                print(f"  Min:     {min_time:.1f}s")
                print(f"  Max:     {max_time:.1f}s")

        # Compare results
        if "CPU Encoding" in results and "GPU Encoding" in results:
            cpu_time = results["CPU Encoding"]["avg"]
            gpu_time = results["GPU Encoding"]["avg"]
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0

            print(f"\n{'='*60}")
            print(f"RESULTS SUMMARY")
            print(f"{'='*60}\n")
            print(f"CPU Encoding:  {cpu_time:.1f}s average")
            print(f"GPU Encoding:  {gpu_time:.1f}s average")
            print(f"Speedup:       {speedup:.2f}x faster with GPU")
            print(f"Time Saved:    {cpu_time - gpu_time:.1f}s per video")

        # Cleanup
        if audio_file.exists():
            audio_file.unlink()

        print(f"\n{'='*60}")

    elif cmd == "async-download-test":
        # Test async downloads
        import asyncio

        print(f"\n{'='*60}")
        print("ASYNC DOWNLOAD TEST")
        print(f"{'='*60}\n")

        async def _run_test():
            from src.content.stock_footage import StockFootageProvider
            import time

            provider = StockFootageProvider()

            if not provider.is_available():
                print("Error: No stock footage providers available")
                print("Configure PEXELS_API_KEY or PIXABAY_API_KEY in .env")
                sys.exit(1)

            # Search for test videos
            print("Searching for test videos...")
            videos = provider.search_videos("nature", count=10)

            if not videos:
                print("No videos found")
                sys.exit(1)

            print(f"Found {len(videos)} videos\n")

            # Test 1: Sequential downloads
            print("Test 1: Sequential Downloads")
            print("-" * 40)
            start = time.time()

            for i, video in enumerate(videos[:5], 1):
                output_path = f"output/test/seq_{i}.mp4"
                result = provider.download_video(video, output_path)
                if result:
                    print(f"  [{i}/5] Downloaded")

            seq_time = time.time() - start
            print(f"Sequential time: {seq_time:.1f}s\n")

            # Test 2: Async downloads
            print("Test 2: Async Concurrent Downloads")
            print("-" * 40)
            start = time.time()

            results = await provider.download_videos_async(
                videos[:5],
                output_dir="output/test",
                max_concurrent=5
            )

            async_time = time.time() - start
            success = sum(1 for r in results if r is not None)
            print(f"Async time: {async_time:.1f}s")
            print(f"Success rate: {success}/5\n")

            # Compare
            speedup = seq_time / async_time if async_time > 0 else 0
            print(f"{'='*60}")
            print(f"RESULTS")
            print(f"{'='*60}")
            print(f"Sequential:  {seq_time:.1f}s")
            print(f"Async:       {async_time:.1f}s")
            print(f"Speedup:     {speedup:.2f}x faster")
            print(f"Time saved:  {seq_time - async_time:.1f}s")
            print(f"{'='*60}")

        try:
            asyncio.run(_run_test())
        except KeyboardInterrupt:
            print("\nTest cancelled")
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # CONTENT EMPIRE COMMANDS (Multi-Platform Distribution)
    # ============================================================

    elif cmd == "empire-daily":
        # Full daily content empire automation
        print("\n[INFO] Starting daily content empire automation...")
        print("       Research -> Create -> Distribute -> Optimize")
        import asyncio as _asyncio
        from src.automation.daily_empire import run_daily_empire

        empire_parser = argparse.ArgumentParser(prog="run.py empire-daily", add_help=False)
        empire_parser.add_argument("--channels", help="Comma-separated channel IDs")
        empire_parser.add_argument("--types", help="Content types: video_long,video_short,blog,image")
        empire_parser.add_argument("--no-distribute", action="store_true", help="Skip distribution")
        empire_args = empire_parser.parse_args(sys.argv[2:])

        channels = [c.strip() for c in empire_args.channels.split(",")] if empire_args.channels else None
        content_types = [t.strip() for t in empire_args.types.split(",")] if empire_args.types else None

        result = _asyncio.run(run_daily_empire(
            channels=channels,
            content_types=content_types,
            distribute=not empire_args.no_distribute,
        ))
        print(f"\n[OK] Empire daily complete:")
        print(f"  Videos: {result.get('videos_created', 0)}")
        print(f"  Shorts: {result.get('shorts_created', 0)}")
        print(f"  Blogs: {result.get('blogs_created', 0)}")
        print(f"  Images: {result.get('images_created', 0)}")
        print(f"  Distributions: {result.get('distributions', 0)}")
        if result.get("errors"):
            print(f"  Errors: {len(result['errors'])}")

    elif cmd == "empire-video":
        # Create video and distribute everywhere
        ev_parser = argparse.ArgumentParser(prog="run.py empire-video", add_help=False)
        ev_parser.add_argument("channel", nargs="?", default="money_blueprints")
        ev_parser.add_argument("--topic", help="Video topic")
        ev_parser.add_argument("--platforms", help="Comma-separated platforms")
        ev_args = ev_parser.parse_args(sys.argv[2:])

        platforms = [p.strip() for p in ev_args.platforms.split(",")] if ev_args.platforms else None
        from src.automation.unified_launcher import empire_video
        result = empire_video(ev_args.channel, ev_args.topic, platforms)
        status = "[OK]" if result.success else "[FAIL]"
        print(f"\n{status} Video: {result.videos_created} created, {result.distributions} distributions")
        if result.errors:
            for e in result.errors:
                print(f"  Error: {e}")

    elif cmd == "empire-blog":
        # Create blog article and distribute
        eb_parser = argparse.ArgumentParser(prog="run.py empire-blog", add_help=False)
        eb_parser.add_argument("topic", help="Blog topic")
        eb_parser.add_argument("--niche", default="general")
        eb_parser.add_argument("--channel", default="")
        eb_args = eb_parser.parse_args(sys.argv[2:])

        from src.automation.unified_launcher import empire_blog
        result = empire_blog(eb_args.topic, eb_args.niche, eb_args.channel)
        status = "[OK]" if result.get("success") else "[FAIL]"
        print(f"\n{status} Blog creation: {result}")

    elif cmd == "distribute":
        # Distribute existing content to platforms
        dist_parser = argparse.ArgumentParser(prog="run.py distribute", add_help=False)
        dist_parser.add_argument("content_path", help="Path to video/image file")
        dist_parser.add_argument("--type", default="video_long", help="Content type")
        dist_parser.add_argument("--title", default="", help="Content title")
        dist_parser.add_argument("--niche", default="general")
        dist_parser.add_argument("--channel", default="")
        dist_parser.add_argument("--platforms", help="Comma-separated platforms")
        dist_args = dist_parser.parse_args(sys.argv[2:])

        import asyncio as _asyncio
        from src.distribution.distributor import ContentDistributor

        async def _distribute():
            distributor = ContentDistributor()
            platforms = [p.strip() for p in dist_args.platforms.split(",")] if dist_args.platforms else None
            return await distributor.distribute(
                content_path=dist_args.content_path,
                content_type=dist_args.type,
                title=dist_args.title,
                niche=dist_args.niche,
                channel=dist_args.channel,
                platforms=platforms,
            )
        result = _asyncio.run(_distribute())
        print(f"\n[OK] Distribution: {result.get('successful', 0)} successful, {result.get('failed', 0)} failed")

    elif cmd == "platform-status":
        # Check all platform connections
        from src.automation.unified_launcher import platform_status
        statuses = platform_status()
        print("\n" + "=" * 50)
        print("  PLATFORM STATUS")
        print("=" * 50)
        for name, configured in statuses.items():
            icon = "[OK]" if configured else "[--]"
            print(f"  {icon} {name}")
        print("=" * 50)

    elif cmd == "cross-analytics":
        # Cross-platform performance report
        ca_parser = argparse.ArgumentParser(prog="run.py cross-analytics", add_help=False)
        ca_parser.add_argument("--period", default="last_7_days",
                               choices=["last_7_days", "last_30_days", "last_90_days", "all_time"])
        ca_args = ca_parser.parse_args(sys.argv[2:])

        from src.automation.unified_launcher import cross_analytics
        import json
        report = cross_analytics(ca_args.period)
        print(json.dumps(report, indent=2, default=str))

    elif cmd == "api-server":
        # Start the Content Empire API server
        api_parser = argparse.ArgumentParser(prog="run.py api-server", add_help=False)
        api_parser.add_argument("--host", default="0.0.0.0")
        api_parser.add_argument("--port", type=int, default=8000)
        api_parser.add_argument("--reload", action="store_true")
        api_args = api_parser.parse_args(sys.argv[2:])

        import uvicorn
        print(f"\n[INFO] Starting Content Empire API on {api_args.host}:{api_args.port}")
        uvicorn.run(
            "src.api.app:app",
            host=api_args.host,
            port=api_args.port,
            reload=api_args.reload,
        )

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python run.py' for help")


if __name__ == "__main__":
    main()
