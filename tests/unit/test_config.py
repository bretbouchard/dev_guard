"""Unit tests for configuration management system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from src.dev_guard.core.config import (
    AgentConfig,
    Config,
    ConfigLoadError,
    ConfigValidationError,
    DatabaseConfig,
    DatabaseType,
    LLMConfig,
    LLMProvider,
    LogLevel,
    NotificationConfig,
    RepositoryConfig,
    VectorDBConfig,
    VectorDBProvider,
    create_example_config,
    get_default_config,
    load_config,
    validate_config_file,
)


class TestLLMConfig:
    """Test LLM configuration validation."""
    
    def test_default_config(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "qwen/qwen3-235b-a22b:free"
        assert config.temperature == 0.1
        assert config.max_tokens == 4096
        assert config.timeout == 30.0
        assert config.max_retries == 3
    
    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)
        
        # Invalid temperatures
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid values
        LLMConfig(max_tokens=1)
        LLMConfig(max_tokens=100000)
        
        # Invalid values
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=100001)
    
    def test_base_url_validation(self):
        """Test base URL validation."""
        # Valid URLs
        LLMConfig(base_url="http://localhost:8080")
        LLMConfig(base_url="https://api.example.com")
        
        # Invalid URLs
        with pytest.raises(ValidationError):
            LLMConfig(base_url="ftp://example.com")
        with pytest.raises(ValidationError):
            LLMConfig(base_url="not-a-url")
    
    def test_api_key_environment_override(self):
        """Test API key retrieval from environment."""
        config = LLMConfig(provider=LLMProvider.OPENAI)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            assert config.get_api_key() == "test-key"
        
        # Config value takes precedence
        config.api_key = "config-key"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            assert config.get_api_key() == "config-key"
    
    def test_effective_base_url(self):
        """Test effective base URL with environment override."""
        config = LLMConfig(base_url="http://localhost:8080")
        
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env-url:8080"}):
            assert config.get_effective_base_url() == "http://env-url:8080"
        
        # Without environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert config.get_effective_base_url() == "http://localhost:8080"


class TestVectorDBConfig:
    """Test Vector DB configuration validation."""
    
    def test_default_config(self):
        """Test default vector DB configuration."""
        config = VectorDBConfig()
        assert config.provider == VectorDBProvider.CHROMA
        assert config.path == "./data/vector_db"
        assert config.collection_name == "dev_guard_knowledge"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
    
    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        # Valid overlap
        VectorDBConfig(chunk_size=1000, chunk_overlap=500)
        
        # Invalid overlap (>= chunk_size)
        with pytest.raises(ValidationError):
            VectorDBConfig(chunk_size=1000, chunk_overlap=1000)
        with pytest.raises(ValidationError):
            VectorDBConfig(chunk_size=1000, chunk_overlap=1001)
    
    def test_collection_name_validation(self):
        """Test collection name validation."""
        # Valid names
        VectorDBConfig(collection_name="valid_name")
        VectorDBConfig(collection_name="valid-name")
        VectorDBConfig(collection_name="valid123")
        
        # Invalid names
        with pytest.raises(ValidationError):
            VectorDBConfig(collection_name="invalid name")  # space
        with pytest.raises(ValidationError):
            VectorDBConfig(collection_name="invalid@name")  # special char
        with pytest.raises(ValidationError):
            VectorDBConfig(collection_name="")  # empty
    
    def test_path_creation(self):
        """Test path validation and creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_vector_db"
            config = VectorDBConfig(path=str(test_path))
            assert config.path == str(test_path)
            assert test_path.parent.exists()


class TestDatabaseConfig:
    """Test Database configuration validation."""
    
    def test_default_config(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        assert config.type == DatabaseType.SQLITE
        assert config.url == "sqlite:///./data/dev_guard.db"
        assert config.echo is False
        assert config.pool_size == 5
    
    def test_sqlite_url_validation(self):
        """Test SQLite URL validation."""
        # Valid SQLite URLs
        DatabaseConfig(type=DatabaseType.SQLITE, url="sqlite:///./test.db")
        DatabaseConfig(type=DatabaseType.SQLITE, url="sqlite:///memory:")
        
        # Invalid SQLite URLs
        with pytest.raises(ValidationError):
            DatabaseConfig(type=DatabaseType.SQLITE, url="postgresql://test")
    
    def test_postgresql_url_validation(self):
        """Test PostgreSQL URL validation."""
        # Valid PostgreSQL URLs
        DatabaseConfig(type=DatabaseType.POSTGRESQL, url="postgresql://user:pass@localhost/db")
        
        # Invalid PostgreSQL URLs
        with pytest.raises(ValidationError):
            DatabaseConfig(type=DatabaseType.POSTGRESQL, url="sqlite:///test.db")
    
    def test_effective_url_environment_override(self):
        """Test effective URL with environment override."""
        config = DatabaseConfig(url="sqlite:///./test.db")
        
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://env-url"}):
            assert config.get_effective_url() == "postgresql://env-url"
        
        # Without environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert config.get_effective_url() == "sqlite:///./test.db"


class TestNotificationConfig:
    """Test Notification configuration validation."""
    
    def test_default_config(self):
        """Test default notification configuration."""
        config = NotificationConfig()
        assert config.enabled is True
        assert config.email_smtp_port == 587
        assert config.email_use_tls is True
        assert config.notification_levels == ["ERROR", "CRITICAL"]
    
    def test_webhook_url_validation(self):
        """Test webhook URL validation."""
        # Valid URLs
        NotificationConfig(discord_webhook="https://discord.com/webhook")
        NotificationConfig(slack_webhook="http://slack.com/webhook")
        
        # Invalid URLs
        with pytest.raises(ValidationError):
            NotificationConfig(discord_webhook="ftp://invalid")
        with pytest.raises(ValidationError):
            NotificationConfig(slack_webhook="not-a-url")
    
    def test_email_validation(self):
        """Test email address validation."""
        # Valid emails
        NotificationConfig(email_to=["test@example.com", "user@domain.org"])
        
        # Invalid emails
        with pytest.raises(ValidationError):
            NotificationConfig(email_to=["invalid-email"])
        with pytest.raises(ValidationError):
            NotificationConfig(email_to=["test@", "@domain.com"])
    
    def test_notification_levels_validation(self):
        """Test notification levels validation."""
        # Valid levels
        NotificationConfig(notification_levels=["DEBUG", "INFO", "WARNING"])
        
        # Invalid levels
        with pytest.raises(ValidationError):
            NotificationConfig(notification_levels=["INVALID"])
    
    def test_telegram_token_environment(self):
        """Test Telegram token from environment."""
        config = NotificationConfig()
        
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token"}):
            assert config.get_telegram_bot_token() == "env-token"
        
        # Config value takes precedence
        config.telegram_bot_token = "config-token"
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token"}):
            assert config.get_telegram_bot_token() == "config-token"


class TestAgentConfig:
    """Test Agent configuration validation."""
    
    def test_default_config(self):
        """Test default agent configuration."""
        config = AgentConfig()
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout == 300.0
        assert config.priority == 5
        assert config.max_concurrent_tasks == 1
    
    def test_validation_ranges(self):
        """Test validation ranges for numeric fields."""
        # Valid values
        AgentConfig(max_retries=0, priority=1, max_concurrent_tasks=10)
        AgentConfig(max_retries=10, priority=10, timeout=3600.0)
        
        # Invalid values
        with pytest.raises(ValidationError):
            AgentConfig(max_retries=11)
        with pytest.raises(ValidationError):
            AgentConfig(priority=0)
        with pytest.raises(ValidationError):
            AgentConfig(priority=11)
        with pytest.raises(ValidationError):
            AgentConfig(timeout=0.0)
        with pytest.raises(ValidationError):
            AgentConfig(timeout=3601.0)
    
    def test_custom_instructions_validation(self):
        """Test custom instructions validation."""
        # Valid instructions
        AgentConfig(custom_instructions="Valid instructions")
        AgentConfig(custom_instructions=None)
        
        # Empty string should become None
        config = AgentConfig(custom_instructions="   ")
        assert config.custom_instructions is None
        
        # Too long instructions
        with pytest.raises(ValidationError):
            AgentConfig(custom_instructions="x" * 10001)
    
    def test_effective_custom_instructions(self):
        """Test custom instructions with environment override."""
        config = AgentConfig(custom_instructions="config instructions")
        
        with patch.dict(os.environ, {"TEST_AGENT_CUSTOM_INSTRUCTIONS": "env instructions"}):
            assert config.get_effective_custom_instructions("test_agent") == "env instructions"
        
        # Without environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert config.get_effective_custom_instructions("test_agent") == "config instructions"


class TestRepositoryConfig:
    """Test Repository configuration validation."""
    
    def test_path_validation(self):
        """Test repository path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid directory
            RepositoryConfig(path=temp_dir)
            
            # Create a git directory to test git repo detection
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()
            RepositoryConfig(path=temp_dir)  # Should not raise warning
        
        # Non-existent path
        with pytest.raises(ValidationError):
            RepositoryConfig(path="/non/existent/path")
        
        # File instead of directory
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValidationError):
                RepositoryConfig(path=temp_file.name)
    
    def test_branch_validation(self):
        """Test branch name validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid branch names
            RepositoryConfig(path=temp_dir, branch="main")
            RepositoryConfig(path=temp_dir, branch="feature/test")
            RepositoryConfig(path=temp_dir, branch="v1.0.0")
            
            # Invalid branch names
            with pytest.raises(ValidationError):
                RepositoryConfig(path=temp_dir, branch="")
            with pytest.raises(ValidationError):
                RepositoryConfig(path=temp_dir, branch="branch with spaces")
            with pytest.raises(ValidationError):
                RepositoryConfig(path=temp_dir, branch="branch..with..dots")
    
    def test_patterns_validation(self):
        """Test file patterns validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid patterns
            config = RepositoryConfig(
                path=temp_dir,
                ignore_patterns=["*.pyc", "__pycache__", ""],  # Empty should be removed
                watch_files=["*.py", "*.js"]
            )
            assert "" not in config.ignore_patterns
            
            # Too long pattern
            with pytest.raises(ValidationError):
                RepositoryConfig(path=temp_dir, ignore_patterns=["x" * 256])


class TestMainConfig:
    """Test main Config class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.data_dir == "./data"
        assert config.log_level == LogLevel.INFO
        assert config.debug is False
        assert config.swarm_interval == 30.0
        assert config.max_concurrent_agents == 5
        assert len(config.agents) == 10  # All required agents
    
    def test_data_dir_validation(self):
        """Test data directory validation and creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_data"
            config = Config(data_dir=str(test_dir))
            assert config.data_dir == str(test_dir.resolve())
            assert test_dir.exists()
    
    def test_agents_validation(self):
        """Test agents validation and missing agent handling."""
        # Missing agents should be added with defaults
        config = Config(agents={"commander": AgentConfig()})
        assert len(config.agents) == 10
        assert "planner" in config.agents
        assert "red_team" in config.agents
    
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        config = Config()
        
        with patch.dict(os.environ, {
            "DEV_GUARD_DATA_DIR": "/tmp/test",
            "DEV_GUARD_LOG_LEVEL": "DEBUG",
            "DEV_GUARD_DEBUG": "true",
            "DEV_GUARD_SWARM_INTERVAL": "60.0",
            "DEV_GUARD_MAX_CONCURRENT_AGENTS": "10"
        }):
            config.apply_environment_overrides()
            assert config.data_dir == "/tmp/test"
            assert config.log_level == LogLevel.DEBUG
            assert config.debug is True
            assert config.swarm_interval == 60.0
            assert config.max_concurrent_agents == 10
    
    def test_agent_config_management(self):
        """Test agent configuration management."""
        config = Config()
        
        # Get existing agent
        agent_config = config.get_agent_config("commander")
        assert isinstance(agent_config, AgentConfig)
        
        # Get non-existent agent
        agent_config = config.get_agent_config("non_existent")
        assert isinstance(agent_config, AgentConfig)
        
        # Update agent config
        config.update_agent_config("commander", priority=10, timeout=600.0)
        assert config.agents["commander"].priority == 10
        assert config.agents["commander"].timeout == 600.0
        
        # Invalid update
        with pytest.raises(ConfigValidationError):
            config.update_agent_config("commander", priority=11)  # Out of range
    
    def test_repository_management(self):
        """Test repository management."""
        config = Config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add repository
            config.add_repository(temp_dir, branch="develop")
            assert len(config.repositories) == 1
            assert config.repositories[0].path == str(Path(temp_dir).resolve())
            assert config.repositories[0].branch == "develop"
            
            # Update existing repository
            config.add_repository(temp_dir, branch="main")
            assert len(config.repositories) == 1
            assert config.repositories[0].branch == "main"
            
            # Get repository config
            repo_config = config.get_repository_config(temp_dir)
            assert repo_config is not None
            assert repo_config.branch == "main"
            
            # Remove repository
            assert config.remove_repository(temp_dir) is True
            assert len(config.repositories) == 0
            assert config.remove_repository(temp_dir) is False  # Already removed
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = Config()
        warnings = config.validate_configuration()
        assert isinstance(warnings, list)
        # Should have warnings about missing API keys and notification methods


class TestConfigLoading:
    """Test configuration loading and saving."""
    
    def test_load_from_file_success(self):
        """Test successful configuration loading."""
        config_data = {
            "data_dir": "./test_data",
            "log_level": "DEBUG",
            "llm": {"provider": "openai", "model": "gpt-4"},
            "agents": {"commander": {"priority": 10}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = Config.load_from_file(config_path)
            assert config.data_dir.endswith("test_data")
            assert config.log_level == LogLevel.DEBUG
            assert config.llm.provider == LLMProvider.OPENAI
            assert config.agents["commander"].priority == 10
        finally:
            os.unlink(config_path)
    
    def test_load_from_file_not_exists(self):
        """Test loading from non-existent file creates default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config = Config.load_from_file(str(config_path))
            
            assert isinstance(config, Config)
            assert config_path.exists()  # Should be created
    
    def test_load_from_file_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with pytest.raises(ConfigLoadError):
                Config.load_from_file(config_path)
        finally:
            os.unlink(config_path)
    
    def test_load_from_file_validation_error(self):
        """Test loading with validation errors."""
        config_data = {
            "llm": {"temperature": 3.0},  # Invalid temperature
            "agents": {"commander": {"priority": 11}}  # Invalid priority
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError):
                Config.load_from_file(config_path)
        finally:
            os.unlink(config_path)
    
    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = Config(data_dir="./test_data", debug=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            config.save_to_file(config_path)
            
            # Verify file was created and contains expected content
            with open(config_path) as f:
                content = f.read()
                assert "DevGuard Configuration File" in content
                assert "debug: true" in content
        finally:
            os.unlink(config_path)
    
    def test_load_from_dict(self):
        """Test loading from dictionary."""
        config_data = {
            "data_dir": "./test_data",
            "debug": True,
            "llm": {"provider": "anthropic"}
        }
        
        config = Config.load_from_dict(config_data)
        assert config.debug is True
        assert config.llm.provider == LLMProvider.ANTHROPIC
    
    def test_load_from_dict_validation_error(self):
        """Test loading from dictionary with validation error."""
        config_data = {
            "llm": {"temperature": -1.0}  # Invalid
        }
        
        with pytest.raises(ConfigValidationError):
            Config.load_from_dict(config_data)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        assert isinstance(config, Config)
        assert config.log_level == LogLevel.INFO
    
    def test_load_config(self):
        """Test load_config function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config = load_config(str(config_path))
            assert isinstance(config, Config)
    
    def test_validate_config_file(self):
        """Test config file validation."""
        # Valid config
        config_data = {"data_dir": "./test"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            warnings = validate_config_file(config_path)
            assert isinstance(warnings, list)
        finally:
            os.unlink(config_path)
        
        # Invalid config
        config_data = {"llm": {"temperature": 3.0}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            errors = validate_config_file(config_path)
            assert len(errors) > 0
            assert any("Validation error" in error for error in errors)
        finally:
            os.unlink(config_path)
    
    def test_create_example_config(self):
        """Test creating example configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "example.yaml"
            create_example_config(str(config_path))
            
            assert config_path.exists()
            
            # Verify it can be loaded
            config = Config.load_from_file(str(config_path))
            assert config.agents["code"].custom_instructions is not None
            assert config.agents["qa_test"].custom_instructions is not None


if __name__ == "__main__":
    pytest.main([__file__])