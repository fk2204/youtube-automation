#!/usr/bin/env python3
"""
One-Click Setup for New YouTube Automation Features

This script:
1. Installs all required dependencies
2. Downloads Whisper model (optional)
3. Verifies API credentials
4. Runs basic tests to ensure everything works

Usage:
    python scripts/setup_new_features.py
    python scripts/setup_new_features.py --skip-whisper
    python scripts/setup_new_features.py --test-only
    python scripts/setup_new_features.py --install-only

Requirements:
    - Python 3.10+
    - pip
    - Internet connection (for downloads)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_step(step: int, total: int, text: str):
    """Print a step indicator."""
    print(f"{Colors.CYAN}[{step}/{total}]{Colors.ENDC} {text}")


def print_success(text: str):
    """Print success message."""
    print(f"  {Colors.GREEN}[OK]{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"  {Colors.WARNING}[WARN]{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message."""
    print(f"  {Colors.FAIL}[ERROR]{Colors.ENDC} {text}")


def run_command(cmd: List[str], description: str = "") -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False


def install_dependencies(dev: bool = True, extras: List[str] = None) -> bool:
    """Install project dependencies."""
    print_step(1, 6, "Installing core dependencies...")

    # Install main requirements
    success, output = run_command([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])

    if not success:
        print_error("Failed to install core dependencies")
        print(output)
        return False
    print_success("Core dependencies installed")

    # Install dev dependencies if requested
    if dev:
        success, output = run_command([
            sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"
        ])
        if success:
            print_success("Dev dependencies installed")
        else:
            print_warning("Dev dependencies installation had issues")

    # Install extras if specified
    if extras:
        for extra in extras:
            print(f"  Installing {extra}...")
            success, output = run_command([
                sys.executable, "-m", "pip", "install", f".[{extra}]"
            ])
            if success:
                print_success(f"{extra} installed")
            else:
                print_warning(f"Could not install {extra}")

    return True


def setup_whisper(model_size: str = "base") -> bool:
    """Download and verify Whisper model."""
    print_step(2, 6, f"Setting up Whisper model ({model_size})...")

    try:
        import whisper
        print_success("Whisper package is available")

        # Check if model is already downloaded
        print(f"  Downloading/verifying {model_size} model...")
        model = whisper.load_model(model_size)
        print_success(f"Whisper {model_size} model ready")

        # Clean up to free memory
        del model

        return True

    except ImportError:
        print_warning("Whisper not installed - skipping model setup")
        print("  To install: pip install openai-whisper")
        return False
    except Exception as e:
        print_error(f"Whisper setup failed: {e}")
        return False


def verify_api_credentials() -> Dict[str, bool]:
    """Verify that required API credentials are configured."""
    print_step(3, 6, "Verifying API credentials...")

    # Load environment variables
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try root .env
        load_dotenv(PROJECT_ROOT / ".env")

    credentials = {
        "GROQ_API_KEY": ("Groq AI", True),  # (name, required)
        "PEXELS_API_KEY": ("Pexels Stock", True),
        "PIXABAY_API_KEY": ("Pixabay Stock", False),
        "REDDIT_CLIENT_ID": ("Reddit API", False),
        "REDDIT_CLIENT_SECRET": ("Reddit API", False),
        "ANTHROPIC_API_KEY": ("Claude AI", False),
        "FISH_AUDIO_API_KEY": ("Fish Audio TTS", False),
        "DISCORD_WEBHOOK_URL": ("Discord Notifications", False),
    }

    results = {}

    for key, (name, required) in credentials.items():
        value = os.getenv(key)
        if value and len(value) > 5:
            print_success(f"{name} ({key}) configured")
            results[key] = True
        else:
            if required:
                print_warning(f"{name} ({key}) NOT configured (recommended)")
            else:
                print(f"  [--] {name} ({key}) not configured (optional)")
            results[key] = False

    return results


def verify_youtube_auth() -> bool:
    """Verify YouTube OAuth credentials."""
    print_step(4, 6, "Verifying YouTube authentication...")

    credentials_path = PROJECT_ROOT / "config" / "youtube_credentials.pickle"
    client_secrets = PROJECT_ROOT / "config" / "client_secret.json"

    if credentials_path.exists():
        print_success("YouTube credentials found")
        return True
    elif client_secrets.exists():
        print_warning("YouTube not authenticated - run auth flow first")
        print("  To authenticate: python -c \"from src.youtube.auth import get_authenticated_service; get_authenticated_service()\"")
        return False
    else:
        print_warning("YouTube client_secret.json not found")
        print("  Download from Google Cloud Console -> APIs -> Credentials")
        return False


def test_new_modules() -> Dict[str, bool]:
    """Run basic tests for new modules."""
    print_step(5, 6, "Testing new modules...")

    results = {}

    # Test ViralHookGenerator
    try:
        from src.content.viral_content_engine import ViralHookGenerator
        generator = ViralHookGenerator()
        hook = generator.generate_hook("test topic", niche="finance")
        if hook and hook.text:
            print_success("ViralHookGenerator working")
            results["ViralHookGenerator"] = True
        else:
            print_error("ViralHookGenerator returned empty hook")
            results["ViralHookGenerator"] = False
    except Exception as e:
        print_error(f"ViralHookGenerator: {e}")
        results["ViralHookGenerator"] = False

    # Test MetadataOptimizer
    try:
        from src.seo.metadata_optimizer import MetadataOptimizer
        optimizer = MetadataOptimizer()
        title = optimizer.optimize_title("Test Title", keywords=["test"])
        if title:
            print_success("MetadataOptimizer working")
            results["MetadataOptimizer"] = True
        else:
            print_error("MetadataOptimizer returned empty title")
            results["MetadataOptimizer"] = False
    except Exception as e:
        print_error(f"MetadataOptimizer: {e}")
        results["MetadataOptimizer"] = False

    # Test FreeKeywordResearch
    try:
        from src.seo.free_keyword_research import FreeKeywordResearch
        researcher = FreeKeywordResearch()
        print_success("FreeKeywordResearch initialized")
        results["FreeKeywordResearch"] = True
    except Exception as e:
        print_error(f"FreeKeywordResearch: {e}")
        results["FreeKeywordResearch"] = False

    # Test AIDisclosureTracker
    try:
        import tempfile
        from src.compliance.ai_disclosure import AIDisclosureTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AIDisclosureTracker(db_path=f"{tmpdir}/test.db")
            tracker.track_voice_generation(video_id="test", method="test")
            disclosure = tracker.get_disclosure_metadata("test")
            if disclosure:
                print_success("AIDisclosureTracker working")
                results["AIDisclosureTracker"] = True
            else:
                print_error("AIDisclosureTracker returned empty disclosure")
                results["AIDisclosureTracker"] = False
    except Exception as e:
        print_error(f"AIDisclosureTracker: {e}")
        results["AIDisclosureTracker"] = False

    # Test WhisperCaptionGenerator (optional)
    try:
        from src.captions.whisper_generator import WhisperCaptionGenerator, WHISPER_AVAILABLE
        if WHISPER_AVAILABLE:
            generator = WhisperCaptionGenerator(model_size="tiny")
            print_success("WhisperCaptionGenerator available")
            results["WhisperCaptionGenerator"] = True
        else:
            print_warning("WhisperCaptionGenerator - Whisper not installed")
            results["WhisperCaptionGenerator"] = False
    except Exception as e:
        print_warning(f"WhisperCaptionGenerator: {e}")
        results["WhisperCaptionGenerator"] = False

    # Test RedditResearcher (without credentials)
    try:
        from src.research.reddit import RedditResearcher
        researcher = RedditResearcher(client_id="test", client_secret="test")
        # Just check initialization
        print_success("RedditResearcher initialized (credentials not verified)")
        results["RedditResearcher"] = True
    except Exception as e:
        print_error(f"RedditResearcher: {e}")
        results["RedditResearcher"] = False

    return results


def run_pytest() -> bool:
    """Run pytest on new module tests."""
    print_step(6, 6, "Running pytest on new modules...")

    success, output = run_command([
        sys.executable, "-m", "pytest",
        "tests/test_new_modules.py",
        "-v", "--tb=short", "-q"
    ])

    if success:
        print_success("All tests passed!")
    else:
        print_warning("Some tests failed - check output above")
        print(output[-2000:] if len(output) > 2000 else output)

    return success


def print_summary(
    credentials: Dict[str, bool],
    module_tests: Dict[str, bool],
    pytest_passed: bool
):
    """Print setup summary."""
    print_header("Setup Summary")

    # Credentials
    print(f"{Colors.BOLD}API Credentials:{Colors.ENDC}")
    configured = sum(1 for v in credentials.values() if v)
    total = len(credentials)
    print(f"  {configured}/{total} credentials configured")

    # Module Tests
    print(f"\n{Colors.BOLD}Module Tests:{Colors.ENDC}")
    passed = sum(1 for v in module_tests.values() if v)
    total = len(module_tests)
    for name, success in module_tests.items():
        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if success else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"  [{status}] {name}")

    print(f"\n  {passed}/{total} modules working")

    # Pytest
    print(f"\n{Colors.BOLD}Pytest:{Colors.ENDC}")
    if pytest_passed:
        print(f"  {Colors.GREEN}All tests passed{Colors.ENDC}")
    else:
        print(f"  {Colors.WARNING}Some tests failed{Colors.ENDC}")

    # Next steps
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    if not credentials.get("GROQ_API_KEY"):
        print("  1. Get a free Groq API key at https://console.groq.com")
    if not credentials.get("PEXELS_API_KEY"):
        print("  2. Get a free Pexels API key at https://www.pexels.com/api/")
    if not credentials.get("REDDIT_CLIENT_ID"):
        print("  3. (Optional) Set up Reddit API at https://reddit.com/prefs/apps")

    print(f"\n{Colors.BOLD}Quick Start:{Colors.ENDC}")
    print("  python run.py video money_blueprints  # Create a video")
    print("  python run.py daily-all              # Start scheduler")
    print("  pytest tests/test_new_modules.py -v  # Run tests")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup new YouTube automation features"
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper model download"
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to download (default: base)"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run tests, skip installation"
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Only install dependencies, skip tests"
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip running pytest"
    )

    args = parser.parse_args()

    print_header("YouTube Automation - New Features Setup")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    credentials = {}
    module_tests = {}
    pytest_passed = False

    if not args.test_only:
        # Install dependencies
        if not install_dependencies(dev=True):
            print_error("Dependency installation failed")
            sys.exit(1)

        # Setup Whisper
        if not args.skip_whisper:
            setup_whisper(args.whisper_model)

    if not args.install_only:
        # Verify credentials
        credentials = verify_api_credentials()

        # Verify YouTube auth
        verify_youtube_auth()

        # Test modules
        module_tests = test_new_modules()

        # Run pytest
        if not args.skip_pytest:
            pytest_passed = run_pytest()

    # Print summary
    print_summary(credentials, module_tests, pytest_passed)

    print(f"\n{Colors.GREEN}{Colors.BOLD}Setup complete!{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
