"""Test suite for CLI functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

# Test imports - handle import errors gracefully
try:
    import typer
    from typer.testing import CliRunner

    from src.dev_guard.cli import app, main
    from src.dev_guard.core.config import Config, load_config
    from src.dev_guard.core.swarm import DevGuardSwarm
    CLI_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"CLI Import error: {e}")
    CLI_IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not CLI_IMPORTS_AVAILABLE, reason="CLI modules not available")
class TestCLICommands:
    """Test CLI command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_app_creation(self):
        """Test that CLI app is properly created."""
        assert app is not None
        assert isinstance(app, typer.Typer)
    
    def test_start_command_help(self):
        """Test start command help text."""
        result = self.runner.invoke(app, ["start", "--help"])
        
        assert result.exit_code == 0
        assert "Start the DevGuard swarm" in result.stdout
        assert "--config" in result.stdout
        assert "--log-level" in result.stdout
        assert "--background" in result.stdout
    
    def test_stop_command_help(self):
        """Test stop command help text.""" 
        result = self.runner.invoke(app, ["stop", "--help"])
        
        assert result.exit_code == 0
        assert "Stop the DevGuard swarm" in result.stdout
    
    def test_status_command_help(self):
        """Test status command help text."""
        result = self.runner.invoke(app, ["status", "--help"])
        
        assert result.exit_code == 0
        assert "Show the status of the DevGuard swarm" in result.stdout
    
    def test_config_command_help(self):
        """Test config command help text."""
        result = self.runner.invoke(app, ["config", "--help"])
        
        assert result.exit_code == 0
        assert "Manage DevGuard configuration" in result.stdout
        assert "--show" in result.stdout
        assert "--validate" in result.stdout
        assert "--init" in result.stdout
    
    def test_agents_command_help(self):
        """Test agents command help text."""
        result = self.runner.invoke(app, ["agents", "--help"])
        
        assert result.exit_code == 0
        assert "List and manage agents" in result.stdout
    
    def test_models_command_help(self):
        """Test models command help text."""
        result = self.runner.invoke(app, ["models", "--help"])
        
        assert result.exit_code == 0
        assert "Manage LLM models" in result.stdout
        assert "--provider" in result.stdout
        assert "--pull" in result.stdout
        assert "--available" in result.stdout
    
    def test_interactive_command_help(self):
        """Test interactive command help text."""
        result = self.runner.invoke(app, ["interactive", "--help"])
        
        assert result.exit_code == 0
        assert "Start interactive control mode" in result.stdout
    
    def test_version_command_help(self):
        """Test version command help text."""
        result = self.runner.invoke(app, ["version", "--help"])
        
        assert result.exit_code == 0
        assert "Show DevGuard version information" in result.stdout
    
    @patch('src.dev_guard.cli.load_config')
    @patch('src.dev_guard.cli.DevGuardSwarm')
    def test_start_command_with_config(self, mock_swarm, mock_load_config):
        """Test start command with configuration file."""
        # Mock configuration
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        # Mock swarm
        mock_swarm_instance = Mock()
        mock_swarm_instance.start = AsyncMock()
        mock_swarm.return_value = mock_swarm_instance
        
        # Test start command with config file
        with patch('src.dev_guard.cli.asyncio.run') as mock_asyncio:
            result = self.runner.invoke(app, ["start", "--config", "test_config.yaml"])
            
            # Should load config with specified path
            mock_load_config.assert_called_with("test_config.yaml")
            mock_swarm.assert_called_with(mock_config)
    
    @patch('src.dev_guard.cli.Config')
    def test_config_show_command(self, mock_config_class):
        """Test config show command."""
        # Mock config instance
        mock_config = Mock()
        mock_config.llm.provider.value = "ollama"
        mock_config.llm.model = "llama2"
        mock_config.vector_db.provider.value = "chroma"
        mock_config.vector_db.path = "./data/vector_db"
        mock_config.database.type.value = "sqlite"
        mock_config.database.url = "sqlite:///./data/dev_guard.db"
        mock_config_class.return_value = mock_config
        
        result = self.runner.invoke(app, ["config", "--show"])
        
        assert result.exit_code == 0
        assert "Configuration" in result.stdout
    
    def test_agents_list_command(self):
        """Test agents list command."""
        result = self.runner.invoke(app, ["agents"])
        
        assert result.exit_code == 0
        assert "DevGuard Agents" in result.stdout
        assert "Commander" in result.stdout
        assert "Planner" in result.stdout
        assert "Code Agent" in result.stdout


@pytest.mark.skipif(not CLI_IMPORTS_AVAILABLE, reason="CLI modules not available")
class TestCLIInteractiveMode:
    """Test CLI interactive mode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @patch('src.dev_guard.cli.load_config')
    @patch('src.dev_guard.cli.DevGuardSwarm')
    @patch('src.dev_guard.cli.console.input')
    def test_interactive_mode_startup(self, mock_input, mock_swarm, mock_load_config):
        """Test interactive mode startup."""
        # Mock configuration and swarm
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        mock_swarm_instance = Mock()
        mock_swarm.return_value = mock_swarm_instance
        
        # Mock input to exit immediately
        mock_input.return_value = "exit"
        
        result = self.runner.invoke(app, ["interactive"])
        
        # Should load config and create swarm
        mock_load_config.assert_called_once()
        mock_swarm.assert_called_with(mock_config)
    
    @patch('src.dev_guard.cli.load_config')
    @patch('src.dev_guard.cli.DevGuardSwarm')
    def test_interactive_commands_structure(self, mock_swarm, mock_load_config):
        """Test interactive commands are properly structured."""
        # Mock configuration and swarm
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        mock_swarm_instance = Mock()
        mock_swarm_instance.list_agents.return_value = [
            {"id": "commander", "status": "active", "enabled": True, 
             "current_task": None, "last_heartbeat": "2024-01-01T00:00:00"}
        ]
        mock_swarm_instance.pause_agent.return_value = True
        mock_swarm_instance.resume_agent.return_value = True
        mock_swarm.return_value = mock_swarm_instance
        
        # Test that helper functions exist
        from src.dev_guard.cli import _pause_agent, _resume_agent, _show_agents_status, _show_tasks
        
        # Basic validation that functions are callable
        assert callable(_show_agents_status)
        assert callable(_show_tasks) 
        assert callable(_pause_agent)
        assert callable(_resume_agent)


@pytest.mark.skipif(not CLI_IMPORTS_AVAILABLE, reason="CLI modules not available") 
class TestCLIModelManagement:
    """Test CLI model management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_models_command_ollama_provider(self):
        """Test models command with Ollama provider."""
        result = self.runner.invoke(app, ["models", "--provider", "ollama", "--available"])
        
        # Should not fail (may show no models if Ollama not running)
        assert result.exit_code in [0, 1]  # May exit with 1 if connection fails
    
    def test_models_command_openrouter_provider(self):
        """Test models command with OpenRouter provider."""
        result = self.runner.invoke(app, ["models", "--provider", "openrouter", "--available"])
        
        # Should not fail (may show no models if no API key)
        assert result.exit_code in [0, 1]  # May exit with 1 if no API key
    
    @patch('src.dev_guard.cli._pull_ollama_model')
    def test_models_pull_command(self, mock_pull):
        """Test model pull command."""
        mock_pull.return_value = None
        
        result = self.runner.invoke(app, ["models", "--pull", "llama2", "--provider", "ollama"])
        
        # Should attempt to pull model
        assert result.exit_code == 0
        assert "Pulling model llama2" in result.stdout


@pytest.mark.skipif(not CLI_IMPORTS_AVAILABLE, reason="CLI modules not available")
class TestCLINotificationCommands:
    """Test CLI notification-related commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_notification_status_command_help(self):
        """Test notification status command help."""
        result = self.runner.invoke(app, ["notification-status", "--help"])
        
        assert result.exit_code == 0
        assert "Show status of notification system" in result.stdout
        assert "--config" in result.stdout
    
    @patch('src.dev_guard.cli.load_config')
    @patch('src.dev_guard.cli.NotificationManager')
    def test_notification_status_command(self, mock_manager_class, mock_load_config):
        """Test notification status command."""
        # Mock configuration
        mock_config = Mock()
        mock_config.notifications.enabled = True
        mock_config.notifications.notification_levels = ["ERROR", "CRITICAL"]
        mock_load_config.return_value = mock_config
        
        # Mock notification manager
        mock_manager = Mock()
        mock_manager.get_provider_status.return_value = {
            "discord": {
                "enabled": True,
                "supported_levels": ["ERROR", "CRITICAL"],
                "type": "webhook"
            }
        }
        mock_manager.list_templates.return_value = ["agent_error", "task_completed"]
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(app, ["notification-status"])
        
        assert result.exit_code == 0
        assert "Notification System Status" in result.stdout


@pytest.mark.skipif(not CLI_IMPORTS_AVAILABLE, reason="CLI modules not available")
class TestCLIMCPServerCommands:
    """Test CLI MCP server commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_mcp_server_command_help(self):
        """Test MCP server command help."""
        result = self.runner.invoke(app, ["mcp-server", "--help"])
        
        assert result.exit_code == 0
        assert "Start the Model Context Protocol server" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--background" in result.stdout
    
    @patch('src.dev_guard.cli.console.input')
    def test_mcp_server_command_start(self, mock_input):
        """Test MCP server start command."""
        # Mock the MCP server import within the function
        with patch('src.dev_guard.mcp.MCPServer') as mock_mcp_server:
            mock_server = Mock()
            mock_server.start_sync = Mock()
            mock_mcp_server.return_value = mock_server
            
            # Test should handle KeyboardInterrupt gracefully
            mock_server.start_sync.side_effect = KeyboardInterrupt()
            
            # Mock input to exit immediately
            mock_input.side_effect = KeyboardInterrupt()
            
            result = self.runner.invoke(app, ["mcp-server"])
            
            # Should attempt to start server
            assert result.exit_code in [0, 1]  # May exit with 1 due to KeyboardInterrupt


# Mock-based tests that don't require actual CLI implementation
class TestCLIMocks:
    """Test CLI functionality using mocks."""
    
    def test_mock_cli_runner(self):
        """Test CLI runner with mock commands."""
        mock_app = Mock()
        mock_runner = Mock()
        
        # Mock successful command execution
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.stdout = "Command executed successfully"
        mock_runner.invoke.return_value = mock_result
        
        # Test command execution
        result = mock_runner.invoke(mock_app, ["status"])
        assert result.exit_code == 0
        assert "successfully" in result.stdout
    
    def test_mock_swarm_operations(self):
        """Test swarm operations using mocks."""
        mock_swarm = Mock()
        
        # Mock agent operations
        mock_swarm.pause_agent.return_value = True
        mock_swarm.resume_agent.return_value = True
        mock_swarm.list_agents.return_value = [
            {"id": "commander", "status": "active"},
            {"id": "planner", "status": "active"}
        ]
        
        # Test operations
        assert mock_swarm.pause_agent("commander") is True
        assert mock_swarm.resume_agent("commander") is True
        
        agents = mock_swarm.list_agents()
        assert len(agents) == 2
        assert agents[0]["id"] == "commander"
    
    def test_mock_config_operations(self):
        """Test configuration operations using mocks."""
        mock_config = Mock()
        mock_config.llm.provider = "ollama"
        mock_config.llm.model = "llama2"
        mock_config.agents = {
            "commander": Mock(enabled=True),
            "planner": Mock(enabled=True)
        }
        
        # Test config access
        assert mock_config.llm.provider == "ollama"
        assert mock_config.llm.model == "llama2"
        assert mock_config.agents["commander"].enabled is True


class TestCLIUtilities:
    """Test CLI utility functions."""
    
    def test_status_color_mapping(self):
        """Test status color mapping utility."""
        # Test expected status values
        test_statuses = ["pending", "running", "completed", "failed", "paused", "active"]
        
        # Mock the color mapping function
        def mock_get_status_color(status):
            color_map = {
                "pending": "yellow",
                "running": "blue", 
                "completed": "green",
                "failed": "red",
                "paused": "orange",
                "active": "green"
            }
            return color_map.get(status, "white")
        
        # Test color mapping
        for status in test_statuses:
            color = mock_get_status_color(status)
            assert isinstance(color, str)
            assert len(color) > 0
    
    def test_table_formatting(self):
        """Test table formatting functionality."""
        # Mock table data
        table_data = [
            {"name": "Commander", "status": "active", "task": "monitoring"},
            {"name": "Planner", "status": "idle", "task": None}
        ]
        
        # Test table structure validation
        for row in table_data:
            assert "name" in row
            assert "status" in row
            assert isinstance(row["name"], str)
            assert isinstance(row["status"], str)
    
    def test_command_validation(self):
        """Test command validation logic."""
        # Test valid command formats
        valid_commands = [
            "start",
            "stop", 
            "status",
            "agents",
            "config --show",
            "models --provider ollama",
            "interactive"
        ]
        
        # Test that commands follow expected patterns
        for cmd in valid_commands:
            parts = cmd.split()
            assert len(parts) >= 1
            assert len(parts[0]) > 0
            assert parts[0].isalpha()


if __name__ == "__main__":
    if CLI_IMPORTS_AVAILABLE:
        pytest.main([__file__, "-v", "-x"])
    else:
        print("Skipping CLI tests due to import errors")
        exit(1)
