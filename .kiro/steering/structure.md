# DevGuard Project Structure

## Source Code Organization

### Core Package (`src/dev_guard/`)
```
src/dev_guard/
├── __init__.py              # Main API surface (DevGuardSwarm, Config)
├── cli.py                   # Command-line interface
├── py.typed                 # Type checking marker
├── agents/                  # Multi-agent system
│   ├── base_agent.py        # Base agent class
│   ├── commander.py         # System coordination
│   ├── planner.py           # Task planning and assignment
│   ├── code_agent.py        # Code generation/refactoring
│   ├── qa_agent.py          # Quality assurance
│   ├── qa_test.py           # Test execution
│   ├── docs.py              # Documentation management
│   ├── git_watcher.py       # Repository monitoring
│   ├── impact_mapper.py     # Cross-repo impact analysis
│   ├── repo_auditor.py      # Missing file detection
│   ├── dep_manager.py       # Dependency management
│   └── red_team.py          # Security analysis
├── api/                     # HTTP API layer
│   ├── config.py            # API configuration
│   └── swarm.py             # Swarm API endpoints
├── core/                    # Core business logic
│   ├── config.py            # Configuration management
│   └── swarm.py             # Swarm orchestration
├── domain/                  # Domain models
│   └── models.py            # Core data models
├── llm/                     # LLM provider abstraction
│   ├── factory.py           # Provider factory
│   ├── provider.py          # Base provider interface
│   ├── ollama.py            # Ollama integration
│   ├── openrouter.py        # OpenRouter integration
│   └── smart.py             # Smart LLM with loop mitigation
├── memory/                  # Persistent storage
│   ├── models.py            # Memory data models
│   ├── shared_memory.py     # Shared agent memory
│   └── vector_db.py         # Vector database operations
├── mcp/                     # Model Context Protocol
│   ├── models.py            # MCP data models
│   ├── server.py            # MCP server implementation
│   └── tools.py             # MCP tools and capabilities
├── notifications/           # Notification system
│   ├── base.py              # Base notification interface
│   ├── notification_manager.py  # Notification orchestration
│   ├── templates.py         # Message templates
│   ├── email_provider.py    # Email notifications
│   ├── slack_provider.py    # Slack integration
│   ├── discord_provider.py  # Discord integration
│   └── telegram_provider.py # Telegram integration
└── adapters/                # External system adapters
    ├── repositories.py      # Repository pattern implementations
    ├── tracing.py           # OpenTelemetry tracing
    └── postgres/            # PostgreSQL adapter
        └── repositories.py  # PostgreSQL-specific repos
```

## Configuration and Data

### Configuration (`config/`)
- `config.yaml` - Main configuration file with agent settings, LLM providers, database config

### Data Storage (`data/`)
- `shared_memory.db` - SQLite database for agent coordination
- `vector_db/` - ChromaDB vector database for embeddings

## Testing Structure (`tests/`)
```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests (isolated components)
├── integration/             # Integration tests (system interactions)
├── performance/             # Performance and benchmark tests
├── security/                # Security-focused tests
└── utils/                   # Test utilities and helpers
    ├── assertions.py        # Custom test assertions
    └── helpers.py           # Test helper functions
```

## Development and Build

### Scripts (`scripts/`)
- `setup-dev.sh` - Development environment setup
- `run-tests.sh` - Test execution script
- `generate_tests.py` - Automated test generation
- `validate-setup.py` - Environment validation

### Documentation (`docs/`)
- Task completion documentation
- Architecture Decision Records (ADRs)
- API documentation

## Key Conventions

### Import Structure
- **Absolute imports only**: No relative imports allowed (enforced by Ruff TID252)
- **First-party imports**: Use `from dev_guard.module import Class`
- **Import sorting**: Managed by isort with Black compatibility

### Code Organization
- **Domain-driven structure**: Core business logic in `domain/` and `core/`
- **Adapter pattern**: External integrations in `adapters/`
- **Agent pattern**: All agents inherit from `base_agent.py`
- **Factory pattern**: LLM providers use factory for instantiation

### Configuration
- **YAML-based**: Main config in `config/config.yaml`
- **Environment overrides**: Support for env var configuration
- **Pydantic validation**: All config validated with Pydantic models

### Testing
- **Comprehensive coverage**: 95%+ coverage requirement
- **Layered testing**: Unit, integration, performance, security
- **Fixtures**: Shared test fixtures in `conftest.py`