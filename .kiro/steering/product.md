# DevGuard Product Overview

DevGuard is an autonomous multi-agent developer swarm powered by LangGraph that continuously monitors, enhances, and refactors codebases. It operates as a local-first system using Ollama with cloud LLM fallback capabilities.

## Core Purpose
- **Autonomous Development**: Coordinated agents work together to improve code quality, generate tests, update documentation, and manage dependencies
- **Continuous Monitoring**: Watches Git repositories for changes and analyzes cross-repository impacts
- **Local-First Architecture**: Operates offline using local LLMs with intelligent cloud fallback

## Agent Ecosystem
The system consists of 9 specialized agents:
- **Commander**: System oversight and coordination
- **Planner**: Task breakdown and assignment
- **Code Agent**: Code generation/refactoring via Goose integration
- **QA/Test**: Automated testing and quality assurance
- **Docs**: Documentation maintenance
- **Git Watcher**: Repository monitoring
- **Impact Mapper**: Cross-repository analysis
- **Repo Auditor**: Missing file detection
- **Dep Manager**: Dependency tracking and upgrades

## Key Features
- Multi-agent coordination via LangGraph
- Smart loop mitigation for local LLMs
- Complete audit trail and decision logging
- HTTP API for programmatic control
- MCP (Model Context Protocol) server integration