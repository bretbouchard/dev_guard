# ADR 0001: Bootstrap DevGuard - Repository Audit and Requirements Capture

Date: 2025-08-08
Status: Proposed

## Context
A repository audit was performed on agent-shell.local (macOS) to inventory the current state of the DevGuard project and capture references to the dev-guard package, with the goal of bootstrapping ongoing work. This ADR will also serve as a living tracker for decisions made in subsequent steps.

Repository root: /Users/bretbouchard/apps/dev_guard

## Findings

1. Packaging and Build
- Project metadata: pyproject.toml
  - project.name: dev-guard
  - project.scripts: dev-guard -> dev_guard.cli:main
  - Source layout: src/ with setuptools.find where=["src"]
- Public top-level package: dev_guard
- __init__.py exports:
  - from dev_guard.core.swarm import DevGuardSwarm
  - from dev_guard.core.config import Config
  - __all__ = ["DevGuardSwarm", "Config"]

2. Source Packages (src/dev_guard)
- dev_guard/__init__.py
- dev_guard/cli.py (Typer/Rich CLI with multiple commands)
- dev_guard/agents/
  - base_agent.py, code_agent.py, commander.py, dep_manager.py, docs.py,
    git_watcher.py, impact_mapper.py, planner.py, qa_agent.py, red_team.py,
    repo_auditor.py
- dev_guard/core/
  - config.py (Config and related models, load/save/validate helpers)
  - swarm.py (DevGuardSwarm)
- dev_guard/llm/
  - __init__.py, provider.py, ollama.py, openrouter.py
- dev_guard/mcp/
  - __init__.py, models.py, server.py, tools.py
- dev_guard/memory/
  - shared_memory.py, vector_db.py
- dev_guard/notifications/
  - __init__.py, base.py, discord_provider.py, email_provider.py,
    notification_manager.py, slack_provider.py, telegram_provider.py,
    templates.py

3. Tests
- Test root: tests/
- Types:
  - Unit: tests/unit/*.py (e.g., test_base_agent.py, test_code_agent.py,
    test_config.py, test_vector_db.py, test_enhanced_qa_agent.py, etc.)
  - Integration: tests/integration/*.py (e.g., test_end_to_end_workflows.py,
    test_environment_setup.py, test_vector_db_integration.py,
    test_system_resilience.py, test_swarm_coordination.py,
    test_performance_integration.py)
  - System-level: tests/test_*.py (e.g., test_mcp_system.py,
    test_vector_database_system.py, test_shared_memory_system.py,
    test_notifications.py, test_cli.py, test_llm_providers.py,
    test_mcp_components.py)
  - Performance and Security directories exist (with __init__.py) for
    classification and future tests.
- Pytest configuration:
  - pytest.ini present and pyproject.toml has [tool.pytest.ini_options]
  - Coverage target: --cov=src/dev_guard and --cov-fail-under=95
  - Markers: slow, integration, unit, security

4. CI Workflows
- .github/workflows/ci.yml present
  - Notes: content not fully reproduced here; exists and should run tests
    and quality checks per project configuration (ruff/black/mypy/pytest).

5. Documentation
- docs/
  - Task-10.2-Code-Quality-Integration-Complete.md
  - Task-10.3-AST-Goose-Integration-Complete.md
  - Task-10.4-Enhanced-Goose-Format-Complete.md
  - Task-11.1-Automated-Testing-QA-Complete.md
  - Task-11.2-TDD-Support-Complete.md
  - Task-11.3-Goose-QA-Integration-Complete.md
  - Task-12.1-Git-Watcher-Agent-Complete.md
  - Task-13.1-Impact-Analysis-Complete.md
  - Task-14.2-Incremental-Updates-Cleanup-Complete.md
  - Task-15-Dependency-Manager-Agent-Complete.md
  - Task-15.1-Dependency-Tracking-Version-Management-Complete.md
  - Task-16-Red-Team-Agent-Complete.md
  - Task-17-Docs-Agent-Complete.md
  - Task-18-MCP-Server-Complete.md
  - Task-19.2-User-Override-Manual-Intervention-Complete.md
  - Task-20-Notification-System-Complete.md
  - Task-21-Integration-Testing-Complete.md
  - TODO.md
  - dev_guard_rfp.md
- No docs/adr/ directory existed prior to this ADR.

## References to dev-guard/dev_guard and Expected Public APIs

- Packaging
  - pyproject.toml: project.name = "dev-guard"
  - Console script: dev-guard = dev_guard.cli:main
  - URLs and README reference GitHub repository dev-guard

- Imports/Usage across codebase
  - Source code imports use the Python package name dev_guard (e.g.,
    from dev_guard.core.config import Config)
  - Tests reference dev_guard modules widely across unit, integration, and
    system tests (e.g., config, agents, memory, notifications, CLI)

- CLI Public Interface (from dev_guard/cli.py)
  - Commands: start, stop, status, config, agents, models, mcp_server, version,
    interactive, pause_agent, resume_agent, inject_task, cancel_task,
    task_details, agent_details, list_tasks, notify, test-notifications,
    notification-status
  - Entry point: main() (Typer app)

- Python Public API Surface
  - dev_guard (package)
    - __all__: DevGuardSwarm, Config
    - dev_guard.core.config
      - Classes: Config (+ LLMConfig, VectorDBConfig, DatabaseConfig,
        NotificationConfig, AgentConfig, RepositoryConfig)
      - Helpers: get_default_config(), load_config(), validate_config_file(),
        create_example_config()
      - Exceptions: ConfigError, ConfigValidationError, ConfigLoadError
    - dev_guard.core.swarm
      - DevGuardSwarm (swarm lifecycle and agent orchestration)
    - dev_guard.notifications.NotificationManager and related provider classes
    - dev_guard.memory.shared_memory and vector_db (shared memory, vector DB)
    - dev_guard.llm.* provider clients (Ollama, OpenRouter) and base provider
    - dev_guard.mcp.* (MCP server interfaces)
  - Note: __all__ currently exports only DevGuardSwarm and Config at the
    package top-level, but submodules provide additional useful APIs.

- Observations regarding "missing dev-guard package"
  - The package is present under src/dev_guard and is importable as dev_guard.
  - If any environment reports a missing package, ensure installation via
    pip install -e . or that the src/ layout is discoverable by the runner.
  - The console script name uses hyphen dev-guard, while Python module imports
    use underscore dev_guard (expected).

## Decisions / Next Steps to Track in This ADR

- Confirm and document the supported public API surface
  - Keep DevGuardSwarm and Config as top-level exports (__all__).
  - Optionally expose additional stable APIs at the top level in a future
    version (e.g., NotificationManager), or explicitly document import paths.

- Validate CI configuration and align with local test markers and coverage
  - Ensure .github/workflows/ci.yml runs pytest with the same addopts as
    pyproject.toml and enforces the 95% coverage threshold.

- Documentation structure
  - Establish docs/adr/ as the place for architectural decisions going
    forward; index ADRs in a docs/adr/README.md in a later change.

- Developer experience
  - Verify dev-guard console script installation path in local dev and CI.
  - Provide a quickstart for installing and running the CLI locally.

- Open Questions
  - Background/daemon modes in CLI commands are placeholders; define runtime
    model and supervision strategy (e.g., process manager) in a later ADR.
  - Define stable versioning and deprecation policy for public APIs.

## Appendices

- Key configuration defaults and environment overrides are implemented in
  dev_guard/core/config.py (see validators and apply_environment_overrides()).
- Extensive test coverage exists across units, integration, and system tests
  with a high coverage bar (95%).

