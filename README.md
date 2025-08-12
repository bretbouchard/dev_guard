# 🤖 DevGuard - Autonomous Multi-Agent Developer Swarm

`dev_guard` is a **LangGraph-powered, local-first, autonomous development swarm** composed of intelligent agents designed to continuously monitor, enhance, and refactor codebases based on user-defined requirements, code changes, and best practices.

## ✨ Features

- **Multi-Agent Swarm**: Coordinated agents via LangGraph with shared memory
- **Local-First**: Operates offline using Ollama with cloud LLM fallback
- **Continuous Monitoring**: Watches Git repositories for changes and impacts
- **Autonomous Enhancement**: Code generation, testing, documentation updates
- **Cross-Repository Impact Analysis**: Detects changes affecting multiple repos
- **Dependency Management**: Automated upgrades with justification tracking
- **Audit Trail**: Complete replay and decision logging

- **Smart Loop Mitigation**: SmartLLM reduces repetitive/looping responses when using local LLMs

## 🧠 Agent Architecture

| Agent | Responsibility |
|-------|----------------|
| **Commander** | System oversight, user communication, swarm coordination |
| **Planner** | Task breakdown and agent assignment |
| **Code** | Code generation/refactoring via Goose/LLM |
| **QA/Test** | Linting, type checking, test execution |
| **Docs** | Documentation and docstring updates |
| **Git Watcher** | Repository monitoring and change detection |
| **Impact Mapper** | Cross-repository impact analysis |
| **Repo Auditor** | Missing file detection and ingestion |
| **Dep Manager** | Dependency tracking and upgrades |

## 🚀 Quick Start

### Prerequisites

- Python 3.11.x (see .python-version)
- uv package manager (see .tool-versions)

### Installation (with uv)

```bash
# Clone the repository
git clone https://github.com/bretbouchard/dev_guard.git
cd dev_guard

# Create virtualenv and sync deps using uv
uv venv --python python3
uv sync

# Optional: install pre-commit hooks
uv run pre-commit install
```

### Configuration

```bash
# Copy example configuration
cp config/config.example.yaml config/config.yaml

# Edit configuration file
nano config/config.yaml
```

### Usage

```bash
# Standard workflow
make env       # create virtualenv with uv
make sync      # sync dependencies from requirements.uv
make test      # run unit and integration tests
```

```bash
# Start the dev_guard swarm (requires LangGraph installed)
dev-guard start

# Inject a task via CLI
dev-guard inject testing "Run unit tests"

# View swarm status
dev-guard status

# Stop the swarm
dev-guard stop
```

### API Server

Run the HTTP API for programmatic control:

```bash
uvicorn dev_guard.api.app:app --reload --host 0.0.0.0 --port 8000
```

- Health: curl <http://localhost:8000/health>
- Docs (OpenAPI): <http://localhost:8000/docs>
- Inject a task:

```bash
curl -X POST http://localhost:8000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"description":"Refactor API module","task_type":"code_refactor","priority":"high"}'
```

### Documentation

MkDocs Material documentation lives under docs/source.

- Serve docs locally:

```bash
pip install mkdocs-material
mkdocs serve
```

- Entry page: <http://127.0.0.1:8000> (from mkdocs serve)
- See also examples/example_api_usage.ipynb for a runnable demo.

## 🛠️ Development

### Dev Prerequisites

- Python 3.10+
- Git
- Ollama (for local LLM)
- Goose CLI (Block, for LLM code generation)

#### Install Goose CLI (Block)

Run the following command to install the latest version of Goose on macOS:

```bash
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
```

To update Goose:

```bash
goose update
```

See [Goose CLI Quickstart](https://block.github.io/goose/docs/quickstart/) for more details.

### Setup Development Environment

```bash
# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy src/
```

## 📊 Architecture

```mermaid
flowchart TD
  U[User / CI / Tools]
  subgraph Interface
    CLI[dev-guard CLI]
    API[FastAPI HTTP API]
  end
  U -->|commands| CLI
  U -->|HTTP| API

  subgraph Core
    SW[Swarm (LangGraph)]
    CFG[Config]
    SM[Shared Memory]
    VDB[Vector DB]
  end

  subgraph Agents
    CMD[Commander]
    PLN[Planner]
    COD[Code Agent]
    QA[QA/Test]
    DOC[Docs]
    GW[Git Watcher]
    IMP[Impact Mapper]
    AUD[Repo Auditor]
    DEP[Dep Manager]
  end

  CLI --> SW
  API --> SW
  CFG --> SW
  CFG --> Agents
  SW <--> SM
  SW <--> VDB
  SW <--> Agents

  subgraph Adapters
    LLM[LLM Providers]
    NOTIF[Notification Providers]
  end

  Agents <--> LLM
  Agents <--> NOTIF
  VDB -. embeddings .-> LLM
```

## 🔧 Configuration

See `config/config.example.yaml` for full configuration options including:

- Repository paths to monitor
- Agent behavior settings
- LLM provider configuration
- Notification settings
- Vector database configuration

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/bretbouchard/dev_guard/issues)
- 💬 [Discussions](https://github.com/bretbouchard/dev_guard/discussions)
