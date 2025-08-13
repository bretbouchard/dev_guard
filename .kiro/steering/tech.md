# DevGuard Technology Stack

## Core Technologies
- **Python 3.11**: Primary language (strict version requirement)
- **LangGraph**: Multi-agent orchestration and workflow management
- **FastAPI**: HTTP API server with OpenAPI documentation
- **ChromaDB**: Vector database for embeddings and knowledge storage
- **SQLite**: Primary database (configurable to other backends)

## Package Management
- **uv**: Primary package manager for fast dependency resolution
- **pip**: Fallback package manager
- **Requirements**: Locked dependencies in `requirements.uv`

## Development Tools
- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checking
- **Black**: Code formatting (100 char line length)
- **isort**: Import sorting
- **pytest**: Testing framework with coverage reporting
- **pre-commit**: Git hooks for code quality

## Key Dependencies
- **Pydantic**: Data validation and settings management
- **Sentence Transformers**: Text embeddings for vector search
- **GitPython**: Git repository operations
- **OpenTelemetry**: Observability and tracing
- **Typer**: CLI framework
- **PyYAML**: Configuration file parsing

## External Integrations
- **Ollama**: Local LLM provider (primary)
- **OpenRouter**: Cloud LLM fallback
- **Goose CLI**: Code generation and refactoring tool
- **MCP Protocol**: Model Context Protocol server

## Common Commands

### Environment Setup
```bash
# Create and activate virtual environment
uv venv --python python3
uv sync

# Install development dependencies
make install-dev
make setup-hooks
```

### Testing
```bash
# Run all tests
make test

# Run specific test suites
make test-unit
make test-integration
make test-performance
make test-security

# Generate coverage report
make coverage
```

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# All quality checks
make quality

# Pre-commit hooks
make pre-commit
```

### Development Workflow
```bash
# Start DevGuard swarm
dev-guard start

# Inject tasks
dev-guard inject testing "Run unit tests"

# Check status
dev-guard status

# Start API server
uvicorn dev_guard.api.app:app --reload --port 8000
```

## Configuration
- **Main config**: `config/config.yaml`
- **Agent settings**: Individual agent configuration in config
- **Environment variables**: Override config values
- **LLM providers**: Ollama (local) + OpenRouter (fallback)