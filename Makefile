# DevGuard Environment, Testing and Quality Makefile
# Provides convenient commands for environment setup, dependency sync, testing, linting, and quality assurance

.PHONY: help install install-dev env sync clean test test-unit test-integration test-performance test-security
.PHONY: lint format type-check security-check coverage quality pre-commit setup-hooks
.PHONY: build docs serve-docs clean-build clean-test clean-all

# Default target
help:
	@echo "DevGuard Testing and Quality Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup-hooks      Install pre-commit hooks"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  test-security    Run security tests only"
	@echo "  coverage         Generate coverage report"
	@echo ""
	@echo "Quality Commands:"
	@echo "  lint             Run all linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run mypy type checking"
	@echo "  security-check   Run security analysis"
	@echo "  quality          Run all quality checks"
	@echo "  pre-commit       Run pre-commit hooks"
	@echo ""
	@echo "Build Commands:"
	@echo "  build            Build package"
	@echo "  docs             Generate documentation"
	@echo "  serve-docs       Serve documentation locally"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  clean            Clean all generated files"
	@echo "  clean-build      Clean build artifacts"
	@echo "  clean-test       Clean test artifacts"

# Installation commands (legacy pip)
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Environment management with uv
# Requires: uv (https://github.com/astral-sh/uv)
PYTHON ?= python3
UV ?= uv

env:
	@echo "🔧 Creating/using virtualenv with uv..."
	$(UV) venv --python $(PYTHON)
	@echo "✅ Virtualenv ready (.venv)"

sync:
	@echo "📦 Syncing dependencies with uv..."
	$(UV) pip sync --python $(PYTHON) requirements.uv || $(UV) pip install -e .
	@echo "✅ Dependencies synced"

setup-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Testing commands
test: test-unit test-integration
	@echo "✅ All tests completed"

# Run tests via uv when available
UV_RUN ?= $(UV) run --python $(PYTHON)

test-unit:
	@echo "🧪 Running unit tests..."
	$(UV_RUN) pytest tests/unit \
		--cov=src/dev_guard \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml \
		--cov-fail-under=95 \
		--junitxml=junit-results.xml \
		-v

test-integration:
	@echo "🔗 Running integration tests..."
	$(UV_RUN) pytest tests/integration \
		--cov=src/dev_guard \
		--cov-append \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml \
		--junitxml=integration-results.xml \
		--timeout=300 \
		-v

test-performance:
	@echo "⚡ Running performance tests..."
	$(UV_RUN) pytest tests/performance \
		--benchmark-only \
		--benchmark-json=benchmark-results.json \
		--benchmark-sort=mean \
		-v

test-security:
	@echo "🔒 Running security tests..."
	$(UV_RUN) pytest tests/security \
		--junitxml=security-test-results.xml \
		-v

test-fast:
	@echo "🚀 Running fast tests only..."
	$(UV_RUN) pytest tests/unit tests/integration \
		-m "not slow" \
		--cov=src/dev_guard \
		--cov-report=term-missing \
		--maxfail=5 \
		-x

test-watch:
	@echo "👀 Running tests in watch mode..."
	pytest-watch tests/unit \
		--cov=src/dev_guard \
		--cov-report=term-missing

coverage:
	@echo "📊 Generating coverage report..."
	coverage report --show-missing
	coverage html
	@echo "Coverage report generated in htmlcov/"

# Quality commands
lint:
	@echo "🔍 Running linting checks..."
	$(UV_RUN) ruff check src tests
	@echo "✅ Linting completed"

format:
	@echo "🎨 Formatting code..."
	$(UV_RUN) black src tests
	$(UV_RUN) isort src tests
	@echo "✅ Code formatting completed"

type-check:
	@echo "🔍 Running type checking..."
	$(UV_RUN) mypy src tests
	@echo "✅ Type checking completed"

security-check:
	@echo "🔒 Running security analysis..."
	$(UV_RUN) bandit -r src -f json -o bandit-report.json || true
	$(UV_RUN) safety check --json --output safety-report.json || true
	@echo "✅ Security analysis completed"

quality: lint type-check security-check
	@echo "✅ All quality checks completed"

pre-commit:
	@echo "🪝 Running pre-commit hooks..."
	pre-commit run --all-files

# Build commands
build:
	@echo "📦 Building package..."
	python -m build
	twine check dist/*
	@echo "✅ Package built successfully"

docs:
	@echo "📚 Generating documentation..."
	# Add documentation generation commands here
	@echo "✅ Documentation generated"

serve-docs:
	@echo "🌐 Serving documentation..."
	# Add documentation serving commands here
	@echo "Documentation server started"

# Cleanup commands
clean: clean-build clean-test
	@echo "🧹 Cleaning all generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	@echo "✅ Cleanup completed"

clean-build:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

clean-test:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf junit-results.xml
	rm -rf integration-results.xml
	rm -rf security-test-results.xml
	rm -rf benchmark-results.json
	rm -rf bandit-report.json
	rm -rf safety-report.json

# Development workflow commands
dev-setup: install-dev setup-hooks
	@echo "🚀 Development environment setup completed"

dev-test: format lint type-check test-unit
	@echo "🧪 Development testing cycle completed"

ci-test: quality test coverage
	@echo "🤖 CI testing cycle completed"

# Docker commands (if needed)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t dev-guard:latest .

docker-test:
	@echo "🐳 Running tests in Docker..."
	docker run --rm -v $(PWD):/app dev-guard:latest make test

# Utility commands
check-deps:
	@echo "📋 Checking dependencies..."
	pip list --outdated
	safety check

update-deps:
	@echo "⬆️ Updating dependencies..."
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

install-tools:
	@echo "🔧 Installing development tools..."
	pip install pip-tools pre-commit pytest-watch

# Performance profiling
profile:
	@echo "📈 Running performance profiling..."
	python -m cProfile -o profile.stats -m pytest tests/performance
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	@echo "🧠 Running memory profiling..."
	python -m memory_profiler tests/performance/test_memory.py

# Benchmark comparison
benchmark-compare:
	@echo "📊 Comparing benchmarks..."
	pytest-benchmark compare benchmark-results.json

# Git hooks validation
validate-hooks:
	@echo "✅ Validating git hooks..."
	pre-commit run --all-files --verbose

# Environment info
env-info:
	@echo "ℹ️ Environment Information:"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git status: $(shell git status --porcelain 2>/dev/null | wc -l || echo 'N/A') files changed"