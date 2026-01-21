.PHONY: install install-new install-all test test-new test-integration lint format clean run help coverage type-check pre-commit setup-whisper setup-features verify-setup

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "  Installation:"
	@echo "    make install       - Install all dependencies and pre-commit hooks"
	@echo "    make install-new   - Install new feature dependencies only"
	@echo "    make install-all   - Install all dependencies including optional features"
	@echo ""
	@echo "  Testing:"
	@echo "    make test          - Run all tests with pytest"
	@echo "    make test-new      - Run tests for new modules only"
	@echo "    make test-integration - Run integration tests"
	@echo "    make coverage      - Run tests with coverage report"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup-whisper - Download and setup Whisper model"
	@echo "    make setup-features- Run full new features setup script"
	@echo "    make verify-setup  - Verify all new features are working"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint          - Check code style (black, isort, flake8)"
	@echo "    make format        - Auto-format code (black, isort)"
	@echo "    make type-check    - Run mypy type checking"
	@echo "    make pre-commit    - Run pre-commit hooks on all files"
	@echo ""
	@echo "  Other:"
	@echo "    make run           - Run the main application"
	@echo "    make clean         - Remove cache files and build artifacts"

# ============================================================
# Installation Targets
# ============================================================

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

install-new:
	@echo "Installing new feature dependencies..."
	pip install -r requirements.txt
	pip install openai-whisper>=20231117
	pip install praw>=7.7.0
	pip install textblob>=0.17.1
	pip install vaderSentiment>=3.3.2
	pip install prophet>=1.1.4
	pip install statsmodels>=0.14.0
	pip install face-recognition>=1.3.0
	pip install opencv-python>=4.8.0
	pip install discord-webhook>=1.3.0
	@echo "New dependencies installed!"

install-all: install install-new
	@echo "All dependencies installed!"

# ============================================================
# Testing Targets
# ============================================================

test:
	pytest tests/ -v

test-new:
	@echo "Running tests for new modules..."
	pytest tests/test_new_modules.py -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/test_pipeline_integration.py -v -m integration

test-fast:
	@echo "Running fast tests only (excluding slow/integration)..."
	pytest tests/ -v -m "not slow and not integration"

coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

coverage-new:
	pytest tests/test_new_modules.py tests/test_pipeline_integration.py -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report for new modules generated in htmlcov/index.html"

# ============================================================
# Setup Targets
# ============================================================

setup-whisper:
	@echo "Downloading Whisper base model..."
	python -c "import whisper; whisper.load_model('base')"
	@echo "Whisper base model ready!"

setup-whisper-small:
	@echo "Downloading Whisper small model..."
	python -c "import whisper; whisper.load_model('small')"
	@echo "Whisper small model ready!"

setup-features:
	@echo "Running full new features setup..."
	python scripts/setup_new_features.py

setup-features-no-whisper:
	@echo "Running new features setup (skipping Whisper)..."
	python scripts/setup_new_features.py --skip-whisper

verify-setup:
	@echo "Verifying new features installation..."
	python scripts/setup_new_features.py --test-only --skip-pytest

# ============================================================
# Code Quality Targets
# ============================================================

lint:
	black --check src/
	isort --check src/
	flake8 src/ --max-line-length=100 --ignore=E203,W503

format:
	black src/
	isort src/

type-check:
	mypy src/ --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

# ============================================================
# Clean Target
# ============================================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

clean-cache:
	@echo "Cleaning video/audio cache..."
	find cache/ -type f -mtime +30 -delete 2>/dev/null || true
	@echo "Cache cleaned!"

# ============================================================
# Run Targets
# ============================================================

run:
	python run.py

run-daily:
	python run.py daily-all

run-video:
	@echo "Usage: python run.py video <channel_name>"
	@echo "Example: python run.py video money_blueprints"

run-short:
	@echo "Usage: python run.py short <channel_name>"
	@echo "Example: python run.py short money_blueprints"

# ============================================================
# Quick Access Targets
# ============================================================

dashboard:
	python -c "from src.analytics.success_tracker import get_success_tracker; get_success_tracker().print_dashboard()"

cost:
	python run.py cost

status:
	python run.py status

cache-stats:
	python run.py cache-stats
