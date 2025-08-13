# DevGuard Quick Reference Guide

## 🚀 Starting & Stopping

```bash
# Start DevGuard
source .venv/bin/activate && python -m src.dev_guard.cli start

# Stop DevGuard
Ctrl+C (graceful shutdown)

# Start in background (using screen)
screen -S devguard
source .venv/bin/activate && python -m src.dev_guard.cli start
# Ctrl+A then D to detach, screen -r devguard to reattach
```

## 📊 Status & Monitoring

```bash
# Check overall status
python -m src.dev_guard.cli status

# List all agents and their status
python -m src.dev_guard.cli agents

# View current configuration
python -m src.dev_guard.cli config --show

# List recent tasks
python -m src.dev_guard.cli list-tasks --limit 10

# Check specific agent details
python -m src.dev_guard.cli agent-details commander
```

## 🎯 Task Management

```bash
# Inject a new task (high priority)
python -m src.dev_guard.cli inject-task "code_generation" "Add unit tests for the API module"

# Inject task to specific agent
python -m src.dev_guard.cli inject-task "testing" "Run security tests" --agent qa_test

# Cancel a running task
python -m src.dev_guard.cli cancel-task <task_id>

# Get task details
python -m src.dev_guard.cli task-details <task_id>

# Filter tasks by status
python -m src.dev_guard.cli list-tasks --status running
python -m src.dev_guard.cli list-tasks --status completed
```

## 🤖 Agent Control

```bash
# Pause an agent
python -m src.dev_guard.cli pause-agent planner

# Resume a paused agent
python -m src.dev_guard.cli resume-agent planner

# Interactive control mode
python -m src.dev_guard.cli interactive
```

## 🧠 LLM & Models

```bash
# List available Ollama models
python -m src.dev_guard.cli models --provider ollama

# Pull a new model
python -m src.dev_guard.cli models --pull llama3.1:8b

# Check OpenRouter models (if configured)
python -m src.dev_guard.cli models --provider openrouter
```

## ⚙️ Configuration

```bash
# Validate config
python -m src.dev_guard.cli config --validate

# Show config with specific file
python -m src.dev_guard.cli config --show --file config/config.yaml
```

## 🔔 Notifications

```bash
# Test notifications
python -m src.dev_guard.cli test-notifications

# Send test notification
python -m src.dev_guard.cli notify "Test message" --level INFO

# Check notification status
python -m src.dev_guard.cli notification-status
```

## 📁 Key File Locations

```
config/config.yaml          # Main configuration
data/dev_guard.db           # SQLite database (agent states, tasks)
data/vector_db/             # ChromaDB vector database
devguard.log               # Log file (if running with nohup)
.env                       # Environment variables (API keys)
```

## 🎮 Interactive Mode Commands

Once in interactive mode (`python -m src.dev_guard.cli interactive`):

```
status                     # Show swarm status
agents                     # List all agents
tasks [N]                  # Show recent N tasks
pause <agent>              # Pause specific agent
resume <agent>             # Resume specific agent
inject <type> <desc>       # Inject new task
cancel <task_id>           # Cancel task
task <task_id>             # Show task details
agent <agent_id>           # Show agent details
help                       # Show help
exit                       # Exit interactive mode
```

## 🚨 Common Task Types

```bash
# Code-related tasks
inject-task "code_generation" "Create API endpoint for user management"
inject-task "code_refactor" "Refactor the authentication module"
inject-task "code_review" "Review the new payment processing code"

# Testing tasks
inject-task "testing" "Run unit tests for the database layer"
inject-task "security_testing" "Perform security scan on API endpoints"
inject-task "performance_testing" "Run load tests on the main API"

# Documentation tasks
inject-task "documentation" "Update README with new installation steps"
inject-task "api_documentation" "Generate OpenAPI docs for new endpoints"

# Maintenance tasks
inject-task "dependency_update" "Update Python dependencies"
inject-task "security_audit" "Run security audit on dependencies"
```

## 🔧 Troubleshooting

```bash
# Check if Ollama is running
ollama ps

# Load/warm up the model
ollama run gpt-oss:20b "Hello"

# Check DevGuard logs
tail -f devguard.log

# Restart with clean state (if needed)
rm -rf data/vector_db data/dev_guard.db
python -m src.dev_guard.cli start
```

## 💡 Tips & Best Practices

### First Time Setup

1. Let the vector database complete initial ingestion before stopping
2. Keep the gpt-oss:20b model loaded in Ollama for faster responses
3. Monitor the initial logs to ensure all agents start successfully

### Daily Usage

- Use `status` and `agents` commands to check system health
- Inject specific tasks when you need immediate action
- Use interactive mode for hands-on control and debugging
- Check `list-tasks` regularly to see what DevGuard has been working on

### Performance

- The first startup takes longer due to vector database population
- Subsequent startups are much faster
- Keep Ollama running to avoid model loading delays
- Use screen/tmux for persistent sessions

### Monitoring

- Watch for agent timeout errors (may need config adjustment)
- Check notification settings to stay informed of important events
- Review task completion status regularly
- Monitor vector database size growth over time

## 🆘 Emergency Commands

```bash
# Force stop all processes
pkill -f "dev_guard"

# Clean restart (loses all state)
rm -rf data/
python -m src.dev_guard.cli start

# Check what's using resources
ps aux | grep dev_guard
ps aux | grep ollama

# View recent logs
tail -100 devguard.log
```

---

**Remember**: DevGuard is designed to run continuously and learn from your codebase. The longer it runs, the better it understands your project patterns and can provide more targeted assistance.
