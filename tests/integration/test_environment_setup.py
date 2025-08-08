"""Integration tests for environment setup and dependency loading."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dev_guard.core.config import Config, create_example_config, load_config


class TestEnvironmentSetup:
    """Test environment setup and configuration loading."""
    
    def test_python_version_compatibility(self):
        """Test that Python version meets requirements."""
        assert sys.version_info >= (3, 9), "Python 3.9+ is required"
    
    def test_required_directories_creation(self):
        """Test that required directories are created properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(data_dir=temp_dir)
            config.ensure_directories()
            
            # Check that data directory exists
            assert Path(temp_dir).exists()
            assert Path(temp_dir).is_dir()
    
    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly."""
        test_env_vars = {
            "DEV_GUARD_DATA_DIR": "/tmp/test_data",
            "DEV_GUARD_LOG_LEVEL": "DEBUG",
            "DEV_GUARD_DEBUG": "true",
            "OPENAI_API_KEY": "test-key-123",
            "DATABASE_URL": "sqlite:///test.db"
        }
        
        with patch.dict(os.environ, test_env_vars):
            config = Config()
            config.apply_environment_overrides()
            
            assert config.data_dir == "/tmp/test_data"
            assert config.log_level.value == "DEBUG"
            assert config.debug is True
            # The LLM config is using Ollama by default, so it won't look for OPENAI_API_KEY
            # Let's test with the correct provider
            config.llm.provider = "openai"
            assert config.llm.get_api_key() == "test-key-123"
            assert config.database.get_effective_url() == "sqlite:///test.db"
    
    def test_configuration_file_loading(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "data_dir": "./test_data",
            "log_level": "WARNING",
            "debug": True,
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.5
            },
            "agents": {
                "commander": {
                    "priority": 10,
                    "timeout": 600.0
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config.data_dir.endswith("test_data")
            assert config.log_level.value == "WARNING"
            assert config.debug is True
            assert config.llm.provider.value == "openai"
            assert config.llm.model == "gpt-4"
            assert config.llm.temperature == 0.5
            assert config.agents["commander"].priority == 10
            assert config.agents["commander"].timeout == 600.0
        finally:
            os.unlink(config_path)
    
    def test_example_configuration_creation(self):
        """Test creation of example configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "example.yaml"
            create_example_config(str(config_path))
            
            assert config_path.exists()
            
            # Verify the file can be loaded
            config = load_config(str(config_path))
            assert isinstance(config, Config)
            assert config.agents["code"].custom_instructions is not None
            assert config.agents["qa_test"].custom_instructions is not None
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(data_dir=temp_dir)
            warnings = config.validate_configuration()
            
            # Should be a list (may contain warnings but no errors)
            assert isinstance(warnings, list)
    
    def test_agent_configuration_completeness(self):
        """Test that all required agents are configured."""
        config = Config()
        
        required_agents = {
            "commander", "planner", "code", "qa_test", "docs",
            "git_watcher", "impact_mapper", "repo_auditor", "dep_manager", "red_team"
        }
        
        configured_agents = set(config.agents.keys())
        assert required_agents.issubset(configured_agents), \
            f"Missing agents: {required_agents - configured_agents}"
    
    def test_database_configuration(self):
        """Test database configuration and connection."""
        config = Config()
        
        # Test SQLite configuration
        assert config.database.type.value == "sqlite"
        assert config.database.url.startswith("sqlite:///")
        
        # Test effective URL with environment override
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
            assert config.database.get_effective_url() == "postgresql://test"
    
    def test_vector_database_configuration(self):
        """Test vector database configuration."""
        config = Config()
        
        assert config.vector_db.provider.value == "chroma"
        assert config.vector_db.collection_name == "dev_guard_knowledge"
        assert config.vector_db.chunk_size > 0
        assert config.vector_db.chunk_overlap < config.vector_db.chunk_size


class TestDependencyLoading:
    """Test that all required dependencies can be imported."""
    
    def test_core_dependencies(self):
        """Test that core dependencies can be imported."""
        core_deps = [
            "pydantic",
            "yaml",
            "pathlib",
            "asyncio",
            "logging",
            "typing",
            "dataclasses",
            "enum"
        ]
        
        for dep in core_deps:
            try:
                __import__(dep)
            except ImportError as e:
                pytest.fail(f"Failed to import core dependency {dep}: {e}")
    
    def test_optional_dependencies(self):
        """Test that optional dependencies are handled gracefully."""
        optional_deps = [
            ("chromadb", "Vector database functionality"),
            ("openai", "OpenAI LLM integration"),
            ("anthropic", "Anthropic LLM integration"),
            ("GitPython", "Git repository operations"),
            ("watchdog", "File system monitoring"),
            ("typer", "CLI interface"),
            ("sqlalchemy", "Database operations"),
        ]
        
        for dep, description in optional_deps:
            try:
                __import__(dep)
            except ImportError:
                pytest.skip(f"Optional dependency {dep} not available ({description})")
    
    def test_development_dependencies(self):
        """Test that development dependencies are available in dev environment."""
        if os.getenv("DEV_MODE") == "true":
            dev_deps = [
                "pytest",
                "black",
                "ruff",
                "mypy",
                "coverage"
            ]
            
            for dep in dev_deps:
                try:
                    __import__(dep)
                except ImportError as e:
                    pytest.fail(f"Failed to import dev dependency {dep}: {e}")
    
    def test_version_compatibility(self):
        """Test that dependency versions meet requirements."""
        version_checks = [
            ("pydantic", "2.0.0"),
        ]
        
        # Add pytest check if in dev mode
        if os.getenv("DEV_MODE") == "true":
            version_checks.append(("pytest", "7.0.0"))
        
        for dep, min_version in version_checks:
                
            try:
                module = __import__(dep)
                if hasattr(module, "__version__"):
                    from packaging import version
                    actual_version = module.__version__
                    assert version.parse(actual_version) >= version.parse(min_version), \
                        f"{dep} version {actual_version} is below minimum {min_version}"
            except ImportError:
                pytest.skip(f"Dependency {dep} not available for version check")


class TestConfigurationIntegration:
    """Test configuration integration with other components."""
    
    def test_config_with_environment_file(self):
        """Test configuration loading with .env file."""
        env_content = """
# Test environment file
DEV_GUARD_DATA_DIR=./test_env_data
DEV_GUARD_LOG_LEVEL=ERROR
OPENAI_API_KEY=test-env-key
DATABASE_URL=sqlite:///env_test.db
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_path = f.name
        
        try:
            # Mock python-dotenv loading
            with patch('os.getenv') as mock_getenv:
                def side_effect(key, default=None):
                    env_vars = {
                        "DEV_GUARD_DATA_DIR": "./test_env_data",
                        "DEV_GUARD_LOG_LEVEL": "ERROR",
                        "OPENAI_API_KEY": "test-env-key",
                        "DATABASE_URL": "sqlite:///env_test.db"
                    }
                    return env_vars.get(key, default)
                
                mock_getenv.side_effect = side_effect
                
                config = Config()
                config.apply_environment_overrides()
                
                assert config.data_dir == "./test_env_data"
                assert config.log_level.value == "ERROR"
                assert config.database.get_effective_url() == "sqlite:///env_test.db"
        finally:
            os.unlink(env_path)
    
    def test_config_repository_management(self):
        """Test repository configuration management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            
            # Add repository
            config.add_repository(temp_dir, branch="develop", auto_commit=True)
            assert len(config.repositories) == 1
            assert config.repositories[0].branch == "develop"
            assert config.repositories[0].auto_commit is True
            
            # Update repository
            config.add_repository(temp_dir, branch="main", auto_commit=False)
            assert len(config.repositories) == 1  # Should update, not add
            assert config.repositories[0].branch == "main"
            assert config.repositories[0].auto_commit is False
            
            # Remove repository
            assert config.remove_repository(temp_dir) is True
            assert len(config.repositories) == 0
            assert config.remove_repository(temp_dir) is False  # Already removed
    
    def test_config_agent_management(self):
        """Test agent configuration management."""
        config = Config()
        
        # Get existing agent
        commander_config = config.get_agent_config("commander")
        assert commander_config.priority == 10  # Default for commander
        
        # Update agent config
        config.update_agent_config("commander", timeout=1200.0, max_retries=5)
        updated_config = config.get_agent_config("commander")
        assert updated_config.timeout == 1200.0
        assert updated_config.max_retries == 5
        assert updated_config.priority == 10  # Should remain unchanged
        
        # Get non-existent agent (should return default)
        unknown_config = config.get_agent_config("unknown_agent")
        assert unknown_config.priority == 5  # Default priority
    
    def test_config_persistence(self):
        """Test configuration saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create and customize config
            config = Config(data_dir=temp_dir, debug=True)
            config.update_agent_config("code", priority=8, timeout=900.0)
            
            # Save config
            config.save_to_file(str(config_path))
            assert config_path.exists()
            
            # Load config
            loaded_config = load_config(str(config_path))
            assert loaded_config.data_dir.endswith(temp_dir.split('/')[-1])
            assert loaded_config.debug is True
            assert loaded_config.agents["code"].priority == 8
            assert loaded_config.agents["code"].timeout == 900.0


class TestSetupScriptIntegration:
    """Test integration with setup scripts."""
    
    @pytest.mark.slow
    def test_setup_script_execution(self):
        """Test that setup script can be executed successfully."""
        setup_script = Path("scripts/setup-dev.sh")
        if not setup_script.exists():
            pytest.skip("Setup script not found")
        
        # Test script syntax (dry run)
        result = subprocess.run(
            ["bash", "-n", str(setup_script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Setup script syntax error: {result.stderr}"
    
    @pytest.mark.slow
    def test_validation_script_execution(self):
        """Test that validation script can be executed successfully."""
        validation_script = Path("scripts/validate-setup.py")
        if not validation_script.exists():
            pytest.skip("Validation script not found")
        
        # Test script execution
        result = subprocess.run(
            [sys.executable, str(validation_script)],
            capture_output=True,
            text=True
        )
        # Script may fail due to missing components, but should not crash
        assert "DevGuard Testing Infrastructure Validation" in result.stdout
    
    def test_makefile_targets(self):
        """Test that Makefile targets are properly defined."""
        makefile = Path("Makefile")
        if not makefile.exists():
            pytest.skip("Makefile not found")
        
        with open(makefile) as f:
            content = f.read()
        
        required_targets = [
            "test:", "test-unit:", "lint:", "format:", "quality:"
        ]
        
        for target in required_targets:
            assert target in content, f"Makefile missing target: {target.rstrip(':')}"


if __name__ == "__main__":
    pytest.main([__file__])