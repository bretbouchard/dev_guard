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
git clone https://github.com/bretbouchard/dev-guard.git
cd dev-guard

# Create virtualenv and sync deps using uv
uv venv --python python3
uv pip sync requirements.uv

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
# Start the dev_guard swarm
dev-guard start

# Add repositories to monitor
dev-guard repo add /path/to/repo

# View swarm status
dev-guard status

# Stop the swarm
dev-guard stop
```

## 🛠️ Development


### Prerequisites

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

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Commander     │    │    Planner      │    │   Code Agent    │
│     Agent       │◄──►│     Agent       │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Shared Memory & Vector DB                     │
│                        (Chroma + SQLite)                       │
└─────────────────────────────────────────────────────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Git Watcher    │    │ Impact Mapper   │    │  Dep Manager    │
│     Agent       │    │     Agent       │    │     Agent       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
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
- 🐛 [Issue Tracker](https://github.com/bretbouchard/dev-guard/issues)
- 💬 [Discussions](https://github.com/bretbouchard/dev-guard/discussions)
