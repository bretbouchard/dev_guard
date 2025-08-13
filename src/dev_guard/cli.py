"""CLI interface for DevGuard."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.dev_guard.core.config import Config, load_config
from src.dev_guard.core.swarm import DevGuardSwarm
from src.dev_guard.notifications import NotificationLevel, NotificationManager, NotificationMessage

# Lightweight .env loader (no external deps)
import os

def _load_env_from_dotenv() -> None:
    """Load environment variables from a .env file if present.
    Also map openrouter_key -> OPENROUTER_API_KEY for convenience.
    """
    try:
        candidates = []
        try:
            candidates.append(Path.cwd() / ".env")
        except Exception:
            pass
        try:
            # Attempt repo root: root/src/dev_guard/cli.py -> parents[3] is root
            repo_root = Path(__file__).resolve()
            for _ in range(4):
                if repo_root.parent == repo_root:
                    break
                repo_root = repo_root.parent
            candidates.append(repo_root / ".env")
        except Exception:
            pass

        for dotenv_path in candidates:
            if not dotenv_path or not dotenv_path.exists():
                continue
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                elif ":" in line:
                    key, val = line.split(":", 1)
                else:
                    continue
                key = key.strip()
                val = val.strip().strip("\"").strip("'")
                if not key:
                    continue
                os.environ.setdefault(key, val)
                os.environ.setdefault(key.upper(), val)
        # Map common alias for OpenRouter
        if not os.getenv("OPENROUTER_API_KEY"):
            alias = os.getenv("openrouter_key") or os.getenv("OPENROUTER_KEY")
            if alias:
                os.environ["OPENROUTER_API_KEY"] = alias
    except Exception:
        # Fail silently; CLI should still work
        pass

# Load .env early
_load_env_from_dotenv()

# Initialize rich console
console = Console()

# Create the main CLI app
app = typer.Typer(
    name="dev-guard",
    help="Autonomous Multi-Agent Developer Swarm System",
    rich_markup_mode="rich"
)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging with rich handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.command()
def start(
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level"
    ),
    background: bool = typer.Option(
        False,
        "--background",
        "-b",
        help="Run in background"
    )
):
    """Start the DevGuard swarm."""
    setup_logging(log_level)

    try:
        # Load configuration
        config = load_config(str(config_file)) if config_file else load_config()

        console.print("[bold green]Starting DevGuard Swarm...[/bold green]")

        # Create and start the swarm
        swarm = DevGuardSwarm(config)

        if background:
            console.print("Starting swarm in background mode...")
            # TODO: Implement background daemon mode
            console.print(
                "[yellow]Background mode not yet implemented[/yellow]"
            )
        else:
            # Run in foreground
            asyncio.run(swarm.start())

    except Exception as e:
        console.print(f"[bold red]Error starting swarm: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def stop():
    """Stop the DevGuard swarm."""
    console.print("[bold red]Stopping DevGuard Swarm...[/bold red]")
    # TODO: Implement graceful shutdown
    console.print("[yellow]Stop command not yet implemented[/yellow]")


@app.command()
def status():
    """Show the status of the DevGuard swarm."""
    console.print("[bold blue]DevGuard Swarm Status[/bold blue]")

    # Create status table
    table = Table(title="Agent Status")
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Current Task", style="yellow")
    table.add_column("Last Activity", style="magenta")

    # TODO: Get actual status from swarm
    agents = [
        ("Commander", "idle", "none", "2 minutes ago"),
        ("Planner", "idle", "none", "2 minutes ago"),
        ("Code Agent", "idle", "none", "2 minutes ago"),
        ("QA Agent", "idle", "none", "2 minutes ago"),
        ("Git Watcher", "monitoring", "watch repositories", "1 minute ago"),
    ]

    for agent, status, task, activity in agents:
        table.add_row(agent, status, task, activity)

    console.print(table)
    console.print(
        "[yellow]Status command showing mock data - "
        "not yet implemented[/yellow]"
    )


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration"
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Validate configuration file"
    ),
    init: bool = typer.Option(
        False,
        "--init",
        help="Initialize default configuration"
    ),
    config_file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Configuration file path"
    )
):
    """Manage DevGuard configuration."""
    if init:
        console.print("[bold blue]Initializing default configuration...[/bold blue]")
        # TODO: Create default config file
        console.print("[yellow]Config init not yet implemented[/yellow]")
    elif validate:
        console.print("[bold blue]Validating configuration...[/bold blue]")
        try:
            config = load_config(str(config_file)) if config_file else load_config()
            console.print("[bold green]Configuration is valid![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Configuration error: {e}[/bold red]")
            raise typer.Exit(1)
    elif show:
        console.print("[bold blue]Current Configuration[/bold blue]")
        try:
            config = load_config(str(config_file)) if config_file else load_config()

            # Display configuration in a nice format
            table = Table(title="Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("LLM Provider", config.llm.provider.value)
            table.add_row("LLM Model", config.llm.model)
            table.add_row(
                "Vector DB Provider", config.vector_db.provider.value
            )
            table.add_row("Vector DB Path", str(config.vector_db.path))
            table.add_row("Database Type", config.database.type.value)
            table.add_row("Database URL", config.database.url)

            console.print(table)
        except Exception as e:
            console.print(
                f"[bold red]Error loading configuration: {e}[/bold red]"
            )
            raise typer.Exit(1)


@app.command()
def agents():
    """List and manage agents."""
    console.print("[bold blue]DevGuard Agents[/bold blue]")

    table = Table(title="Available Agents")
    table.add_column("Agent", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", style="yellow")

    agents_info = [
        ("Commander", "System oversight and coordination", "Available"),
        ("Planner", "Task breakdown and assignment", "Available"),
        ("Code Agent", "Code generation and refactoring", "Available"),
        ("QA/Test Agent", "Testing and quality assurance", "Available"),
        ("Docs Agent", "Documentation generation", "Available"),
        ("Git Watcher", "Repository monitoring", "Available"),
        ("Impact Mapper", "Cross-repository analysis", "Available"),
        ("Repo Auditor", "Repository scanning", "Available"),
        ("Dependency Manager", "Dependency tracking", "Available"),
        ("Red Team Agent", "Security testing", "Available"),
    ]

    for name, desc, status in agents_info:
        table.add_row(name, desc, status)

    console.print(table)


@app.command()
def models(
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help="LLM provider (ollama, openrouter)"
    ),
    pull: str | None = typer.Option(
        None,
        "--pull",
        help="Pull/download a specific model"
    ),
    available: bool = typer.Option(
        False,
        "--available",
        "-a",
        help="List available models"
    )
):
    """Manage LLM models."""
    setup_logging()

    if pull:
        console.print(
            f"[bold blue]Pulling model {pull} from {provider}...[/bold blue]"
        )

        if provider == "ollama":
            # Pull model using Ollama
            asyncio.run(_pull_ollama_model(pull))
        else:
            console.print(
                f"[red]Model pulling not supported for {provider}[/red]"
            )
            raise typer.Exit(1)

    elif available or not pull:
        console.print(
            f"[bold blue]Available models for {provider}[/bold blue]"
        )

        if provider == "ollama":
            asyncio.run(_list_ollama_models())
        elif provider == "openrouter":
            asyncio.run(_list_openrouter_models())
        else:
            console.print(f"[red]Unknown provider: {provider}[/red]")
            raise typer.Exit(1)


async def _list_ollama_models():
    """List available Ollama models."""
    try:
        from src.dev_guard.llm.ollama import OllamaClient

        # Use the updated Ollama server with correct port
        client = OllamaClient({
            "base_url": "http://localhost:11434",
            "model": "gpt-oss:20b"
        })

        # Check if Ollama is available
        if not await client.is_available():
            console.print("[red]Ollama is not running or not available[/red]")
            console.print("Make sure Ollama is installed and running:")
            console.print("  1. Install: https://ollama.ai")
            console.print("  2. Start: ollama serve")
            return

        models = await client.get_available_models()

        if not models:
            console.print(
                "[yellow]No models found. Pull a model first:[/yellow]"
            )
            console.print("  dev-guard models --pull llama2")
            return

        table = Table(title="Available Ollama Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")

        for model in models:
            name = model.get("name", "Unknown")
            size = model.get("size", 0)
            modified = model.get("modified_at", "Unknown")

            # Format size
            if isinstance(size, int):
                if size > 1024**3:
                    size_str = f"{size / (1024**3):.1f} GB"
                elif size > 1024**2:
                    size_str = f"{size / (1024**2):.1f} MB"
                else:
                    size_str = f"{size} bytes"
            else:
                size_str = str(size)

            # Format modified date
            if modified and modified != "Unknown":
                try:
                    from datetime import datetime
                    if "T" in modified:
                        dt = datetime.fromisoformat(
                            modified.replace("Z", "+00:00")
                        )
                        modified_str = dt.strftime("%Y-%m-%d %H:%M")
                    else:
                        modified_str = modified
                except Exception:
                    modified_str = modified
            else:
                modified_str = "Unknown"

            table.add_row(name, size_str, modified_str)

        console.print(table)
        console.print(f"\nFound {len(models)} models")

    except Exception as e:
        console.print(f"[red]Error listing Ollama models: {e}[/red]")


async def _pull_ollama_model(model_name: str):
    """Pull an Ollama model."""
    try:
        from src.dev_guard.llm.ollama import OllamaClient

        # Use the updated Ollama server with correct port
        client = OllamaClient({
            "base_url": "http://localhost:11434",
            "model": "gpt-oss:20b"
        })

        # Check if Ollama is available
        if not await client.is_available():
            console.print("[red]Ollama is not running or not available[/red]")
            console.print("[yellow]Try starting Ollama server first[/yellow]")
            return

        console.print(f"Pulling model '{model_name}'...")
        console.print(
            "[yellow]This may take several minutes "
            "depending on model size[/yellow]"
        )

        success = await client.pull_model(model_name)

        if success:
            console.print(
                f"[green]Successfully pulled model '{model_name}'[/green]"
            )
        else:
            console.print(f"[red]Failed to pull model '{model_name}'[/red]")

    except Exception as e:
        console.print(f"[red]Error pulling model: {e}[/red]")


async def _list_openrouter_models():
    """List available OpenRouter models."""
    try:
        from src.dev_guard.llm.openrouter import OpenRouterClient

        # Try to load config to get API key
        try:
            config = Config()
            api_key = config.llm.get_api_key()
        except Exception:
            api_key = None

        if not api_key:
            console.print("[red]OpenRouter API key not found[/red]")
            console.print("Set OPENROUTER_API_KEY environment variable")
            return

        client = OpenRouterClient({
            "api_key": api_key,
            "model": "openai/gpt-3.5-turbo"
        })

        models = await client.get_available_models()

        if not models:
            console.print("[yellow]No models found or API error[/yellow]")
            return

        table = Table(title="Available OpenRouter Models")
        table.add_column("Model", style="cyan")
        table.add_column("Context", style="green")
        table.add_column("Pricing", style="yellow")

        for model in models[:20]:  # Show first 20 models
            name = model.get("id", "Unknown")
            context = model.get("context_length", "Unknown")
            pricing = model.get("pricing", {})

            prompt_price = pricing.get("prompt", "Unknown")
            completion_price = pricing.get("completion", "Unknown")

            price_str = f"${prompt_price}/${completion_price}" if prompt_price != "Unknown" and completion_price != "Unknown" else "Unknown"

            table.add_row(name, str(context), price_str)

        console.print(table)
        console.print(f"\nShowing 20 of {len(models)} models")

    except Exception as e:
        console.print(f"[red]Error listing OpenRouter models: {e}[/red]")


@app.command()
def mcp_server(
    host: str = typer.Option(
        "localhost",
        "--host",
        "-h",
        help="MCP server host"
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="MCP server port"
    ),
    background: bool = typer.Option(
        False,
        "--background",
        "-b",
        help="Run MCP server in background"
    )
):
    """Start the Model Context Protocol server."""
    setup_logging()

    console.print(
        f"[bold blue]Starting MCP server on {host}:{port}...[/bold blue]"
    )

    try:
        from src.dev_guard.mcp import MCPServer

        # Initialize MCP server
        server = MCPServer(host=host, port=port)

        if background:
            console.print(
                "[green]Starting MCP server in background...[/green]"
            )
            # For background mode, we would typically use a process manager
            # For now, just show the command
            console.print(
                f"[yellow]Background mode not fully implemented. "
                f"Start with: uvicorn dev_guard.mcp.server:app "
                f"--host {host} --port {port}[/yellow]"
            )
        else:
            console.print("[green]MCP server starting...[/green]")
            console.print(
                f"[dim]WebSocket endpoint: ws://{host}:{port}/mcp[/dim]"
            )
            console.print(
                f"[dim]HTTP endpoint: http://{host}:{port}/mcp/invoke[/dim]"
            )
            console.print("[dim]Press Ctrl+C to stop[/dim]")

            # Start server synchronously
            server.start_sync()

    except KeyboardInterrupt:
        console.print("\n[yellow]MCP server stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to start MCP server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show DevGuard version information."""
    console.print(
        "[bold blue]DevGuard[/bold blue] version "
        "[bold green]0.1.0[/bold green]"
    )
    console.print("Autonomous Multi-Agent Developer Swarm System")


# Interactive control commands
@app.command()
def interactive():
    """Start interactive control mode for manual intervention."""
    console.print("[bold blue]DevGuard Interactive Control Mode[/bold blue]")
    console.print("Type 'help' for available commands or 'exit' to quit\n")

    # Try to load the swarm
    try:
        config = load_config()
        swarm = DevGuardSwarm(config)
    except Exception as e:
        console.print(f"[red]Failed to initialize swarm: {e}[/red]")
        raise typer.Exit(1)

    while True:
        try:
            command = console.input("[bold]devguard> [/bold]").strip()

            if not command:
                continue

            parts = command.split()
            cmd = parts[0].lower()

            if cmd == "exit" or cmd == "quit":
                console.print("[yellow]Exiting interactive mode...[/yellow]")
                break
            elif cmd == "help":
                _show_interactive_help()
            elif cmd == "status":
                _show_swarm_status(swarm)
            elif cmd == "agents":
                _show_agents_status(swarm)
            elif cmd == "tasks":
                limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
                _show_tasks(swarm, limit)
            elif cmd == "pause" and len(parts) > 1:
                agent_id = parts[1]
                _pause_agent(swarm, agent_id)
            elif cmd == "resume" and len(parts) > 1:
                agent_id = parts[1]
                _resume_agent(swarm, agent_id)
            elif cmd == "inject":
                if len(parts) < 3:
                    console.print("[red]Usage: inject <type> <description>[/red]")
                    continue
                task_type = parts[1]
                description = " ".join(parts[2:])
                _inject_task(swarm, task_type, description)
            elif cmd == "cancel" and len(parts) > 1:
                task_id = parts[1]
                _cancel_task(swarm, task_id)
            elif cmd == "task" and len(parts) > 1:
                task_id = parts[1]
                _show_task_details(swarm, task_id)
            elif cmd == "agent" and len(parts) > 1:
                agent_id = parts[1]
                _show_agent_details(swarm, agent_id)
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type 'help' for available commands")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def pause_agent(agent_id: str):
    """Pause a specific agent."""
    try:
        config = load_config()
        swarm = DevGuardSwarm(config)

        if swarm.pause_agent(agent_id):
            console.print(f"[green]Agent {agent_id} paused successfully[/green]")
        else:
            console.print(f"[red]Failed to pause agent {agent_id}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def resume_agent(agent_id: str):
    """Resume a paused agent."""
    try:
        config = load_config()
        swarm = DevGuardSwarm(config)

        if swarm.resume_agent(agent_id):
            console.print(f"[green]Agent {agent_id} resumed successfully[/green]")
        else:
            console.print(f"[red]Failed to resume agent {agent_id}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def inject_task(
    task_type: str = typer.Argument(..., help="Type of task (code_generation, testing, etc.)"),
    description: str = typer.Argument(..., help="Task description"),
    agent_id: str | None = typer.Option(None, "--agent", "-a", help="Target agent ID"),
    priority: str = typer.Option("high", "--priority", "-p", help="Task priority (low, normal, high, critical)")
):
    """Inject a high-priority task into the system."""
    try:
        config = load_config()
        swarm = DevGuardSwarm(config)

        task_id = swarm.inject_task(
            description=description,
            task_type=task_type,
            agent_id=agent_id,
            priority=priority
        )

        console.print(f"[green]Task {task_id} injected successfully[/green]")
        console.print(f"[dim]Type: {task_type}, Priority: {priority}[/dim]")
        if agent_id:
            console.print(f"[dim]Assigned to: {agent_id}[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def cancel_task(task_id: str):
    """Cancel a pending or running task."""
    try:
        config = load_config()
        swarm = DevGuardSwarm(config)

        if swarm.cancel_task(task_id):
            console.print(f"[green]Task {task_id} cancelled successfully[/green]")
        else:
            console.print(f"[red]Failed to cancel task {task_id}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def task_details(task_id: str):
    """Show detailed information about a specific task."""
    try:
        config = load_config()
        swarm = DevGuardSwarm(config)

        details = swarm.get_task_details(task_id)
        if not details:
            console.print(f"[red]Task {task_id} not found[/red]")
            raise typer.Exit(1)

        table = Table(title=f"Task {task_id} Details")
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("ID", details["id"])
        table.add_row("Status", f"[bold {_get_status_color(details['status'])}]{details['status'].upper()}[/bold {_get_status_color(details['status'])}]")
        table.add_row("Description", details["description"])
        table.add_row("Agent", details["agent_id"])
        table.add_row("Created", details["created_at"])
        table.add_row("Updated", details["updated_at"])

        if details.get("metadata"):
            metadata_str = "\n".join([f"{k}: {v}" for k, v in details["metadata"].items()])
            table.add_row("Metadata", metadata_str)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def agent_details(agent_id: str):
    """Show detailed information about a specific agent."""
    try:
        config = load_config()
        swarm = DevGuardSwarm(config)

        details = swarm.get_agent_details(agent_id)
        if not details:
            console.print(f"[red]Agent {agent_id} not found[/red]")
            raise typer.Exit(1)

        table = Table(title=f"Agent {agent_id} Details")
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("ID", details["id"])
        table.add_row("Status", f"[bold {_get_status_color(details['status'])}]{details['status'].upper()}[/bold {_get_status_color(details['status'])}]")
        table.add_row("Enabled", "✅" if details["enabled"] else "❌")
        table.add_row("Current Task", details["current_task"] or "None")
        table.add_row("Last Heartbeat", details["last_heartbeat"])

        if details.get("capabilities"):
            capabilities_str = ", ".join(details["capabilities"][:5])
            if len(details["capabilities"]) > 5:
                capabilities_str += f" ... and {len(details['capabilities']) - 5} more"
            table.add_row("Capabilities", capabilities_str)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_tasks(
    status: str | None = typer.Option(None, "--status", "-s", help="Filter by status"),
    agent_id: str | None = typer.Option(None, "--agent", "-a", help="Filter by agent"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of tasks to show")
):
    """List tasks with optional filtering."""
    try:
        config = load_config()
        swarm = DevGuardSwarm(config)

        tasks = swarm.list_tasks(status=status, agent_id=agent_id, limit=limit)

        if not tasks:
            console.print("[yellow]No tasks found[/yellow]")
            return

        table = Table(title="Tasks")
        table.add_column("ID", style="dim")
        table.add_column("Status", min_width=10)
        table.add_column("Agent", min_width=12)
        table.add_column("Description", max_width=50)
        table.add_column("Created", style="dim")

        for task in tasks:
            status_color = _get_status_color(task["status"])
            table.add_row(
                task["id"][:8] + "...",
                f"[{status_color}]{task['status'].upper()}[/{status_color}]",
                task["agent_id"],
                task["description"][:47] + "..." if len(task["description"]) > 50 else task["description"],
                task["created_at"][:16].replace("T", " ")
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _get_status_color(status: str) -> str:
    """Get color for task/agent status."""
    colors = {
        "pending": "yellow",
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "cancelled": "bright_black",
        "paused": "magenta",
        "active": "green",
        "inactive": "bright_black",
        "error": "red"
    }
    return colors.get(status.lower(), "white")


def _show_interactive_help():
    """Show help for interactive mode commands."""
    help_table = Table(title="Interactive Commands")
    help_table.add_column("Command", style="bold blue")
    help_table.add_column("Description")
    help_table.add_column("Usage", style="dim")

    commands = [
        ("status", "Show swarm status", "status"),
        ("agents", "Show all agents", "agents"),
        ("tasks [N]", "Show recent tasks", "tasks [limit]"),
        ("pause <agent>", "Pause an agent", "pause commander"),
        ("resume <agent>", "Resume an agent", "resume commander"),
        ("inject <type> <desc>", "Inject a task", "inject testing 'Run unit tests'"),
        ("cancel <task_id>", "Cancel a task", "cancel abc123"),
        ("task <task_id>", "Show task details", "task abc123"),
        ("agent <agent_id>", "Show agent details", "agent commander"),
        ("help", "Show this help", "help"),
        ("exit", "Exit interactive mode", "exit")
    ]

    for cmd, desc, usage in commands:
        help_table.add_row(cmd, desc, usage)

    console.print(help_table)


def _show_swarm_status(swarm):
    """Show overall swarm status."""
    status = swarm.get_status()

    console.print(f"[bold]Swarm Status:[/bold] {'🟢 Running' if status['is_running'] else '🔴 Stopped'}")
    console.print(f"[bold]Agents:[/bold] {len(status['agents'])} active")
    console.print(f"[bold]Recent Tasks:[/bold] {len(status['recent_tasks'])}")
    console.print(f"[bold]Repositories:[/bold] {len(status['repositories'])}")
    console.print(f"[bold]Vector DB Documents:[/bold] {status['vector_db_documents']}")


def _show_agents_status(swarm):
    """Show status of all agents."""
    agents = swarm.list_agents()

    table = Table(title="Agents Status")
    table.add_column("Agent", style="bold")
    table.add_column("Status", min_width=10)
    table.add_column("Enabled")
    table.add_column("Current Task")
    table.add_column("Last Heartbeat", style="dim")

    for agent in agents:
        if not agent:
            continue

        status_color = _get_status_color(agent["status"])
        table.add_row(
            agent["id"],
            f"[{status_color}]{agent['status'].upper()}[/{status_color}]",
            "✅" if agent["enabled"] else "❌",
            agent["current_task"] or "None",
            agent["last_heartbeat"][:16].replace("T", " ") if agent["last_heartbeat"] != "never" else "Never"
        )

    console.print(table)


def _show_tasks(swarm, limit: int = 10):
    """Show recent tasks."""
    tasks = swarm.list_tasks(limit=limit)

    if not tasks:
        console.print("[yellow]No tasks found[/yellow]")
        return

    table = Table(title=f"Recent {len(tasks)} Tasks")
    table.add_column("ID", style="dim")
    table.add_column("Status", min_width=10)
    table.add_column("Agent", min_width=12)
    table.add_column("Description", max_width=40)

    for task in tasks:
        status_color = _get_status_color(task["status"])
        table.add_row(
            task["id"][:8] + "...",
            f"[{status_color}]{task['status'].upper()}[/{status_color}]",
            task["agent_id"],
            task["description"][:37] + "..." if len(task["description"]) > 40 else task["description"]
        )

    console.print(table)


def _pause_agent(swarm, agent_id: str):
    """Pause an agent interactively."""
    if swarm.pause_agent(agent_id):
        console.print(f"[green]✅ Agent {agent_id} paused[/green]")
    else:
        console.print(f"[red]❌ Failed to pause agent {agent_id}[/red]")


def _resume_agent(swarm, agent_id: str):
    """Resume an agent interactively."""
    if swarm.resume_agent(agent_id):
        console.print(f"[green]✅ Agent {agent_id} resumed[/green]")
    else:
        console.print(f"[red]❌ Failed to resume agent {agent_id}[/red]")


def _inject_task(swarm, task_type: str, description: str):
    """Inject a task interactively."""
    try:
        task_id = swarm.inject_task(description, task_type, priority="high")
        console.print(f"[green]✅ Task {task_id[:8]}... injected[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to inject task: {e}[/red]")


def _cancel_task(swarm, task_id: str):
    """Cancel a task interactively."""
    if swarm.cancel_task(task_id):
        console.print(f"[green]✅ Task {task_id} cancelled[/green]")
    else:
        console.print(f"[red]❌ Failed to cancel task {task_id}[/red]")


def _show_task_details(swarm, task_id: str):
    """Show task details interactively."""
    details = swarm.get_task_details(task_id)
    if not details:
        console.print(f"[red]❌ Task {task_id} not found[/red]")
        return

    console.print(f"[bold]Task {task_id}:[/bold]")
    console.print(f"  Status: [{_get_status_color(details['status'])}]{details['status'].upper()}[/{_get_status_color(details['status'])}]")
    console.print(f"  Agent: {details['agent_id']}")
    console.print(f"  Description: {details['description']}")
    console.print(f"  Created: {details['created_at'][:16].replace('T', ' ')}")


def _show_agent_details(swarm, agent_id: str):
    """Show agent details interactively."""
    details = swarm.get_agent_details(agent_id)
    if not details:
        console.print(f"[red]❌ Agent {agent_id} not found[/red]")
        return

    console.print(f"[bold]Agent {agent_id}:[/bold]")
    console.print(f"  Status: [{_get_status_color(details['status'])}]{details['status'].upper()}[/{_get_status_color(details['status'])}]")
    console.print(f"  Enabled: {'✅' if details['enabled'] else '❌'}")
    console.print(f"  Current Task: {details['current_task'] or 'None'}")
    console.print(f"  Capabilities: {len(details.get('capabilities', []))}")


@app.command("notify")
def send_notification(
    message: str = typer.Argument(..., help="Notification message"),
    title: str | None = typer.Option(
        None, "--title", "-t", help="Notification title"
    ),
    level: str = typer.Option(
        "INFO", "--level", "-l", help="Notification level"
    ),
    source: str = typer.Option(
        "cli", "--source", "-s", help="Notification source"
    ),
    template: str | None = typer.Option(
        None, "--template", help="Use template for formatting"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    )
):
    """Send a test notification through configured providers."""
    setup_logging()

    try:
        # Load configuration
        config = load_config(str(config_file) if config_file else None)
        notification_manager = NotificationManager(config.notifications)

        # Parse notification level
        try:
            notification_level = NotificationLevel[level.upper()]
        except KeyError:
            console.print(f"[red]❌ Invalid level: {level}[/red]")
            console.print(
                f"Valid levels: {', '.join([level_item.value for level_item in NotificationLevel])}"
            )
            return

        if template:
            # Use template
            context = {
                "message": message,
                "timestamp": "now",
                "source": source
            }

            async def send_templated():
                results = await notification_manager.send_templated_notification(
                    template_name=template,
                    context=context,
                    level=notification_level,
                    source=source
                )
                return results

            results = asyncio.run(send_templated())
        else:
            # Direct message
            notification_message = NotificationMessage(
                title=title or "DevGuard CLI Notification",
                content=message,
                level=notification_level,
                source=source
            )

            async def send_direct():
                results = await notification_manager.send_notification(notification_message)
                return results

            results = asyncio.run(send_direct())

        # Display results
        successful = sum(1 for r in results if r.success)
        total = len(results)

        if successful > 0:
            console.print(f"[green]✅ Notification sent to {successful}/{total} providers[/green]")
        else:
            console.print("[red]❌ Failed to send notification to all providers[/red]")

        # Show provider results
        for result in results:
            status_color = "green" if result.success else "red"
            status_text = "✅ SUCCESS" if result.success else "❌ FAILED"
            console.print(f"  {result.provider}: [{status_color}]{status_text}[/{status_color}]")
            if result.error:
                console.print(f"    Error: {result.error}")

    except Exception as e:
        console.print(f"[red]❌ Failed to send notification: {e}[/red]")


@app.command("test-notifications")
def test_notification_providers(
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    )
):
    """Test all configured notification providers."""
    setup_logging()

    try:
        # Load configuration
        config = load_config(str(config_file) if config_file else None)
        notification_manager = NotificationManager(config.notifications)

        console.print("[blue]🔍 Testing notification providers...[/blue]")

        async def test_all():
            return await notification_manager.test_providers()

        results = asyncio.run(test_all())

        # Display results
        table = Table(title="Notification Provider Test Results")
        table.add_column("Provider", style="bold")
        table.add_column("Status")
        table.add_column("Result")

        for provider, success in results.items():
            status_color = "green" if success else "red"
            status_text = "✅ PASS" if success else "❌ FAIL"
            result_text = "Connection successful" if success else "Connection failed"

            table.add_row(
                provider.title(),
                f"[{status_color}]{status_text}[/{status_color}]",
                result_text
            )

        console.print(table)

        # Summary
        passed = sum(1 for success in results.values() if success)
        total = len(results)

        if passed == total:
            console.print(f"[green]🎉 All {total} providers passed tests![/green]")
        elif passed > 0:
            console.print(f"[yellow]⚠️  {passed}/{total} providers passed tests[/yellow]")
        else:
            console.print("[red]❌ All providers failed tests[/red]")

    except Exception as e:
        console.print(f"[red]❌ Failed to test providers: {e}[/red]")


@app.command("notification-status")
def show_notification_status(
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    )
):
    """Show status of notification system and providers."""
    setup_logging()

    try:
        # Load configuration
        config = load_config(str(config_file) if config_file else None)
        notification_manager = NotificationManager(config.notifications)

        # Show general configuration
        console.print("[bold]📬 Notification System Status[/bold]")
        console.print(f"  Enabled: {'✅' if config.notifications.enabled else '❌'}")
        console.print(f"  Global Levels: {', '.join(config.notifications.notification_levels)}")

        # Show provider status
        provider_status = notification_manager.get_provider_status()

        if provider_status:
            console.print("\n[bold]🔌 Configured Providers:[/bold]")

            table = Table()
            table.add_column("Provider", style="bold")
            table.add_column("Status")
            table.add_column("Supported Levels")
            table.add_column("Type")

            for provider_name, status in provider_status.items():
                status_color = "green" if status["enabled"] else "red"
                status_text = "✅ Enabled" if status["enabled"] else "❌ Disabled"

                table.add_row(
                    provider_name.title(),
                    f"[{status_color}]{status_text}[/{status_color}]",
                    ", ".join(status["supported_levels"]),
                    status["type"]
                )

            console.print(table)
        else:
            console.print("\n[yellow]⚠️  No notification providers configured[/yellow]")

        # Show available templates
        templates = notification_manager.list_templates()

        if templates:
            console.print(f"\n[bold]📝 Available Templates ({len(templates)}):[/bold]")

            # Show in columns
            import math
            cols = 3
            rows = math.ceil(len(templates) / cols)

            for row in range(rows):
                row_templates = []
                for col in range(cols):
                    idx = row + col * rows
                    if idx < len(templates):
                        row_templates.append(templates[idx])

                console.print(f"  {' | '.join(f'{t:<20}' for t in row_templates)}")

    except Exception as e:
        console.print(f"[red]❌ Failed to show notification status: {e}[/red]")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
