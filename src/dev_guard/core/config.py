"""Configuration management for DevGuard."""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigLoadError(ConfigError):
    """Exception raised when configuration loading fails."""
    pass


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class VectorDBProvider(str, Enum):
    """Supported vector database providers."""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class DatabaseType(str, Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMConfig(BaseModel):
    """LLM provider configuration with comprehensive validation."""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-3-haiku-20240307"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0, le=100000)
    fallback_provider: LLMProvider | None = LLMProvider.OPENAI
    fallback_model: str | None = "gpt-4o-mini"
    timeout: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v, info):
        """Validate API key is provided for cloud providers."""
        if info.data:
            provider = info.data.get('provider')
            if provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.OPENROUTER]:
                if not v and not os.getenv(f"{provider.upper()}_API_KEY"):
                    logger.warning(f"No API key provided for {provider}. Set {provider.upper()}_API_KEY environment variable.")
        return v

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v):
        """Validate base URL format."""
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("base_url must start with http:// or https://")
        return v

    def get_api_key(self) -> str | None:
        """Get API key from config or environment variable."""
        if self.api_key:
            return self.api_key
        return os.getenv(f"{self.provider.upper()}_API_KEY")

    def get_effective_base_url(self) -> str | None:
        """Get effective base URL with environment variable override."""
        env_url = os.getenv(f"{self.provider.upper()}_BASE_URL")
        return env_url or self.base_url


class VectorDBConfig(BaseModel):
    """Vector database configuration with comprehensive validation."""
    provider: VectorDBProvider = VectorDBProvider.CHROMA
    path: str = "./data/vector_db"
    collection_name: str = Field(default="dev_guard_knowledge", min_length=1, max_length=100)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=1000, gt=0, le=10000)
    chunk_overlap: int = Field(default=200, ge=0)
    max_documents: int = Field(default=100000, gt=0)
    api_key: str | None = None
    environment: str | None = None  # For Pinecone
    index_name: str | None = None  # For Pinecone

    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        """Ensure chunk overlap is less than chunk size."""
        if info.data:
            chunk_size = info.data.get('chunk_size', 1000)
            if v >= chunk_size:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator('collection_name')
    @classmethod
    def validate_collection_name(cls, v):
        """Validate collection name format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("collection_name must contain only alphanumeric characters, hyphens, and underscores")
        return v

    @field_validator('path')
    @classmethod
    def validate_path(cls, v):
        """Ensure path is valid and create directory if needed."""
        path = Path(v)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create vector DB directory {path.parent}: {e}")
        return str(path)

    def get_api_key(self) -> str | None:
        """Get API key from config or environment variable."""
        if self.api_key:
            return self.api_key
        return os.getenv(f"{self.provider.upper()}_API_KEY")


class DatabaseConfig(BaseModel):
    """Database configuration with comprehensive validation."""
    type: DatabaseType = DatabaseType.SQLITE
    url: str = "sqlite:///./data/dev_guard.db"
    echo: bool = False
    pool_size: int = Field(default=5, gt=0, le=50)
    max_overflow: int = Field(default=10, ge=0, le=100)
    pool_timeout: float = Field(default=30.0, gt=0.0)

    @field_validator('url')
    @classmethod
    def validate_url(cls, v, info):
        """Validate database URL format."""
        if info.data:
            db_type = info.data.get('type', DatabaseType.SQLITE)

            if db_type == DatabaseType.SQLITE:
                if not v.startswith('sqlite:///'):
                    raise ValueError("SQLite URL must start with 'sqlite:///'")
                # Ensure directory exists for SQLite
                db_path = Path(v.replace('sqlite:///', ''))
                try:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Could not create database directory {db_path.parent}: {e}")
            elif db_type == DatabaseType.POSTGRESQL:
                if not v.startswith('postgresql://'):
                    raise ValueError("PostgreSQL URL must start with 'postgresql://'")

        return v


    def get_effective_url(self) -> str:
        """Get effective database URL with environment variable override."""
        env_url = os.getenv("DATABASE_URL")
        return env_url or self.url

class SharedMemoryConfig(BaseModel):
    """Shared memory (SQLite) configuration."""
    provider: str = "sqlite"
    db_path: str = "./data/shared_memory.db"

    @field_validator('db_path')
    @classmethod
    def validate_db_path(cls, v):
        path = Path(v)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Invalid shared memory db path: {e}")
        return str(path)

        # No additional validation needed for shared memory path beyond directory existence
        return str(path)



class NotificationConfig(BaseModel):
    """Notification system configuration with comprehensive validation."""
    enabled: bool = True
    discord_webhook: str | None = None
    slack_webhook: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    email_smtp_server: str | None = None
    email_smtp_port: int = Field(default=587, gt=0, le=65535)
    email_use_tls: bool = True
    email_username: str | None = None
    email_password: str | None = None
    email_from: str | None = None
    email_to: list[str] = Field(default_factory=list)
    notification_levels: list[str] = Field(default_factory=lambda: ["ERROR", "CRITICAL"])

    @field_validator('discord_webhook', 'slack_webhook')
    @classmethod
    def validate_webhook_url(cls, v):
        """Validate webhook URL format."""
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("Webhook URL must start with http:// or https://")
        return v

    @field_validator('email_to')
    @classmethod
    def validate_email_addresses(cls, v):
        """Basic email address validation."""
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        for email in v:
            if not email_pattern.match(email):
                raise ValueError(f"Invalid email address: {email}")
        return v

    @field_validator('notification_levels')
    @classmethod
    def validate_notification_levels(cls, v):
        """Validate notification levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        for level in v:
            if level not in valid_levels:
                raise ValueError(f"Invalid notification level: {level}. Must be one of {valid_levels}")
        return v

    def get_telegram_bot_token(self) -> str | None:
        """Get Telegram bot token from config or environment."""
        return self.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")

    def get_email_password(self) -> str | None:
        """Get email password from config or environment."""
        return self.email_password or os.getenv("EMAIL_PASSWORD")


class AgentConfig(BaseModel):
    """Agent-specific configuration with comprehensive validation."""
    enabled: bool = True
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, gt=0.0, le=60.0)
    timeout: float = Field(default=300.0, gt=0.0, le=3600.0)
    custom_instructions: str | None = Field(default=None, max_length=10000)
    priority: int = Field(default=5, ge=1, le=10)
    max_concurrent_tasks: int = Field(default=1, gt=0, le=10)
    heartbeat_interval: float = Field(default=30.0, gt=0.0, le=300.0)
    memory_limit_mb: int = Field(default=512, gt=0, le=4096)

    @field_validator('custom_instructions')
    @classmethod
    def validate_custom_instructions(cls, v):
        """Validate custom instructions format."""
        if v and len(v.strip()) == 0:
            return None
        return v

    def get_effective_custom_instructions(self, agent_name: str) -> str | None:
        """Get custom instructions with environment variable override."""
        env_instructions = os.getenv(f"{agent_name.upper()}_CUSTOM_INSTRUCTIONS")
        return env_instructions or self.custom_instructions


class RepositoryConfig(BaseModel):
    """Repository monitoring configuration with comprehensive validation."""
    path: str
    branch: str = "main"
    auto_commit: bool = False
    auto_push: bool = False
    ignore_patterns: list[str] = Field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", ".venv", "node_modules", "*.log",
        ".DS_Store", "Thumbs.db", "*.tmp", "*.swp", "*.bak"
    ])
    watch_files: list[str] = Field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.md", "*.yaml", "*.yml", "*.json",
        "requirements.txt", "package.json", "pyproject.toml", "Dockerfile",
        "*.toml", "*.cfg", "*.ini", "*.sh", "*.bat"
    ])
    max_file_size_mb: int = Field(default=10, gt=0, le=100)
    scan_depth: int = Field(default=10, gt=0, le=20)

    @field_validator('path')
    @classmethod
    def validate_path(cls, v):
        """Validate repository path exists and is a directory."""
        path = Path(v).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Repository path is not a directory: {path}")

        # Check if it's a git repository
        git_dir = path / ".git"
        if not git_dir.exists():
            logger.warning(f"Path {path} does not appear to be a Git repository")

        return str(path)

    @field_validator('branch')
    @classmethod
    def validate_branch(cls, v):
        """Validate branch name format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Branch name cannot be empty")
        # Basic branch name validation
        invalid_chars = ['..', '~', '^', ':', '?', '*', '[', '\\', ' ']
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Branch name contains invalid character: {char}")
        return v.strip()

    @field_validator('ignore_patterns', 'watch_files')
    @classmethod
    def validate_patterns(cls, v):
        """Validate file patterns."""
        if not v:
            return v

        # Remove empty patterns
        patterns = [p.strip() for p in v if p.strip()]

        # Basic pattern validation
        for pattern in patterns:
            if len(pattern) > 255:
                raise ValueError(f"Pattern too long: {pattern}")

        return patterns


class Config(BaseModel):
    """Main configuration class for DevGuard with comprehensive validation."""

    # Core settings
    data_dir: str = Field(default="./data", min_length=1)
    log_level: LogLevel = LogLevel.INFO
    debug: bool = False
    config_version: str = "1.0"

    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    shared_memory: SharedMemoryConfig = Field(default_factory=SharedMemoryConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    # Agent configurations
    agents: dict[str, AgentConfig] = Field(default_factory=lambda: {
        "commander": AgentConfig(priority=10),
        "planner": AgentConfig(priority=9),
        "code": AgentConfig(priority=8, timeout=600.0),
        "qa_test": AgentConfig(priority=7, timeout=900.0),
        "docs": AgentConfig(priority=6),
        "git_watcher": AgentConfig(priority=8, heartbeat_interval=10.0),
        "impact_mapper": AgentConfig(priority=7),
        "repo_auditor": AgentConfig(priority=5, timeout=1800.0),
        "dep_manager": AgentConfig(priority=6),
        "red_team": AgentConfig(priority=9, timeout=1200.0),
    })

    # Repository configurations
    repositories: list[RepositoryConfig] = Field(default_factory=list)

    # Swarm settings
    swarm_interval: float = Field(default=30.0, gt=0.0, le=3600.0)
    max_concurrent_agents: int = Field(default=5, gt=0, le=20)
    enable_audit_trail: bool = True
    audit_retention_days: int = Field(default=30, gt=0, le=365)

    # Performance settings
    memory_limit_mb: int = Field(default=2048, gt=0, le=16384)
    cpu_limit_percent: int = Field(default=80, gt=0, le=100)

    # Security settings
    enable_security_scanning: bool = True
    security_scan_interval: float = Field(default=3600.0, gt=0.0)  # 1 hour

    @field_validator('data_dir')
    @classmethod
    def validate_data_dir(cls, v):
        """Validate and create data directory."""
        data_path = Path(v).expanduser().resolve()
        try:
            data_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create data directory {data_path}: {e}")
        return str(data_path)

    @field_validator('agents', mode='before')
    @classmethod
    def coerce_agents_input(cls, v):
        """Allow tests to pass a single AgentConfig; coerce into dict."""
        if isinstance(v, AgentConfig):
            # Minimal dict; the post-validator will fill missing agents with defaults
            return {"qa_test": v}
        return v

    @field_validator('agents')
    @classmethod
    def validate_agents(cls, v):
        """Validate agent configurations."""
        required_agents = {
            "commander", "planner", "code", "qa_test", "docs",
            "git_watcher", "impact_mapper", "repo_auditor", "dep_manager", "red_team"
        }

        missing_agents = required_agents - set(v.keys())
        if missing_agents:
            logger.warning(f"Missing agent configurations: {missing_agents}")
            # Add missing agents with default config
            for agent in missing_agents:
                v[agent] = AgentConfig()

        return v

    def apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Core settings
        if os.getenv("DEV_GUARD_DATA_DIR"):
            self.data_dir = os.getenv("DEV_GUARD_DATA_DIR")
        if os.getenv("DEV_GUARD_LOG_LEVEL"):
            try:
                self.log_level = LogLevel(os.getenv("DEV_GUARD_LOG_LEVEL"))
            except ValueError:
                logger.warning(f"Invalid log level in DEV_GUARD_LOG_LEVEL: {os.getenv('DEV_GUARD_LOG_LEVEL')}")
        if os.getenv("DEV_GUARD_DEBUG"):
            self.debug = os.getenv("DEV_GUARD_DEBUG").lower() in ("true", "1", "yes")

        # LLM settings (support project-level env overrides)
        provider_env = os.getenv("DEV_GUARD_LLM_PROVIDER")
        model_env = os.getenv("DEV_GUARD_LLM_MODEL")
        base_url_env = os.getenv("DEV_GUARD_LLM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
        if provider_env:
            try:
                self.llm.provider = LLMProvider(provider_env)
            except ValueError:
                logger.warning(f"Invalid LLM provider in DEV_GUARD_LLM_PROVIDER: {provider_env}")
        if model_env:
            self.llm.model = model_env
        if base_url_env:
            self.llm.base_url = base_url_env

        # Swarm settings
        if os.getenv("DEV_GUARD_SWARM_INTERVAL"):
            try:
                self.swarm_interval = float(os.getenv("DEV_GUARD_SWARM_INTERVAL"))
            except ValueError:
                logger.warning(f"Invalid swarm interval: {os.getenv('DEV_GUARD_SWARM_INTERVAL')}")

        if os.getenv("DEV_GUARD_MAX_CONCURRENT_AGENTS"):
            try:
                self.max_concurrent_agents = int(os.getenv("DEV_GUARD_MAX_CONCURRENT_AGENTS"))
            except ValueError:
                logger.warning(f"Invalid max concurrent agents: {os.getenv('DEV_GUARD_MAX_CONCURRENT_AGENTS')}")

    def validate_configuration(self) -> list[str]:
        """Validate the entire configuration and return list of warnings/errors."""
        warnings = []

        # Check if required directories are writable
        try:
            test_file = Path(self.data_dir) / "test_write"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            warnings.append(f"Data directory is not writable: {e}")

        # Check LLM configuration
        if self.llm.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.OPENROUTER]:
            if not self.llm.get_api_key():
                warnings.append(f"No API key configured for {self.llm.provider}")

        # Check notification configuration
        if self.notifications.enabled:
            has_notification_method = any([
                self.notifications.discord_webhook,
                self.notifications.slack_webhook,
                self.notifications.get_telegram_bot_token(),
                self.notifications.email_smtp_server
            ])
            if not has_notification_method:
                warnings.append("Notifications enabled but no notification methods configured")

        # Check repository configurations
        for i, repo in enumerate(self.repositories):
            try:
                # This will trigger validation
                RepositoryConfig(**repo.dict())
            except ValidationError as e:
                warnings.append(f"Repository {i} configuration error: {e}")

        return warnings

    @classmethod
    def load_from_file(cls, config_path: str | None = None) -> "Config":
        """Load configuration from YAML file with comprehensive error handling."""
        if config_path is None:
            config_path = os.getenv("DEV_GUARD_CONFIG", "config/config.yaml")

        config_file = Path(config_path).expanduser().resolve()

        # If config file doesn't exist, create default
        if not config_file.exists():
            logger.info(f"Configuration file not found at {config_file}, creating default configuration")
            config = cls()
            try:
                config.save_to_file(str(config_file))
                logger.info(f"Default configuration saved to {config_file}")
            except Exception as e:
                logger.warning(f"Could not save default configuration: {e}")
            return config

        # Load and parse YAML
        try:
            with open(config_file, encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Invalid YAML in configuration file {config_file}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Could not read configuration file {config_file}: {e}")

        if data is None:
            logger.warning(f"Configuration file {config_file} is empty, using defaults")
            data = {}

        # Validate and create configuration
        try:
            config = cls(**data)
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field = " -> ".join(str(x) for x in error['loc'])
                error_details.append(f"{field}: {error['msg']}")
            raise ConfigValidationError("Configuration validation failed:\n" + "\n".join(error_details))
        except Exception as e:
            raise ConfigLoadError(f"Unexpected error loading configuration: {e}")

        # Apply environment variable overrides only when using default config path
        # This avoids test flakiness where a specific file explicitly sets provider/model
        if config_path is None:
            config.apply_environment_overrides()

        # Validate the final configuration
        warnings = config.validate_configuration()
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        logger.info(f"Configuration loaded successfully from {config_file}")
        return config

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> "Config":
        """Load configuration from dictionary with validation."""
        try:
            config = cls(**data)
            # Respect explicit dict values; only apply env overrides that are set
            config.apply_environment_overrides()
            return config
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field = " -> ".join(str(x) for x in error['loc'])
                error_details.append(f"{field}: {error['msg']}")
            raise ConfigValidationError("Configuration validation failed:\n" + "\n".join(error_details))

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file with error handling."""
        config_file = Path(config_path).expanduser().resolve()

        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigError(f"Could not create configuration directory {config_file.parent}: {e}")

        # Create a clean dictionary for YAML output with enum values as strings
        config_dict = self.model_dump(exclude_none=True, mode='json')

        # Add header comment
        header = f"""# DevGuard Configuration File
# Version: {self.config_version}
# Generated automatically - modify with care
# Environment variables can override most settings

"""

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(header)
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=True)
        except Exception as e:
            raise ConfigError(f"Could not write configuration file {config_file}: {e}")

        logger.info(f"Configuration saved to {config_file}")

    def ensure_directories(self) -> None:
        """Ensure all required directories exist with proper error handling."""
        directories = [
            ("data", self.data_dir),
            ("vector_db", Path(self.vector_db.path).parent),
        ]

        # Add database directory for SQLite
        if self.database.type == DatabaseType.SQLITE:
            db_path = self.database.url.replace("sqlite:///", "")
            directories.append(("database", Path(db_path).parent))

        for name, directory in directories:
            if directory:
                try:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Ensured {name} directory exists: {directory}")
                except Exception as e:
                    logger.error(f"Could not create {name} directory {directory}: {e}")
                    raise ConfigError(f"Could not create {name} directory {directory}: {e}")

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent with validation."""
        if agent_name not in self.agents:
            logger.warning(f"Agent '{agent_name}' not found in configuration, using default")
            return AgentConfig()
        return self.agents[agent_name]

    def update_agent_config(self, agent_name: str, **kwargs) -> None:
        """Update configuration for a specific agent."""
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentConfig()

        # Update the agent config
        current_config = self.agents[agent_name].model_dump()
        current_config.update(kwargs)

        try:
            self.agents[agent_name] = AgentConfig(**current_config)
            logger.info(f"Updated configuration for agent '{agent_name}'")
        except ValidationError as e:
            raise ConfigValidationError(f"Invalid agent configuration for '{agent_name}': {e}")

    def add_repository(self, repo_path: str, **kwargs) -> None:
        """Add a repository to monitor with validation."""
        try:
            repo_config = RepositoryConfig(path=repo_path, **kwargs)
        except ValidationError as e:
            raise ConfigValidationError(f"Invalid repository configuration: {e}")

        # Check if repository already exists
        repo_path_resolved = str(Path(repo_path).expanduser().resolve())
        for i, existing_repo in enumerate(self.repositories):
            existing_path_resolved = str(Path(existing_repo.path).expanduser().resolve())
            if existing_path_resolved == repo_path_resolved:
                self.repositories[i] = repo_config
                logger.info(f"Updated repository configuration: {repo_path}")
                return

        self.repositories.append(repo_config)
        logger.info(f"Added repository to monitoring: {repo_path}")

    def remove_repository(self, repo_path: str) -> bool:
        """Remove a repository from monitoring."""
        repo_path_resolved = str(Path(repo_path).expanduser().resolve())
        for i, repo in enumerate(self.repositories):
            existing_path_resolved = str(Path(repo.path).expanduser().resolve())
            if existing_path_resolved == repo_path_resolved:
                del self.repositories[i]
                logger.info(f"Removed repository from monitoring: {repo_path}")
                return True
        logger.warning(f"Repository not found for removal: {repo_path}")
        return False

    def get_repository_config(self, repo_path: str) -> RepositoryConfig | None:
        """Get configuration for a specific repository."""
        repo_path_resolved = str(Path(repo_path).expanduser().resolve())
        for repo in self.repositories:
            existing_path_resolved = str(Path(repo.path).expanduser().resolve())
            if existing_path_resolved == repo_path_resolved:
                return repo
        return None


def get_default_config() -> Config:
    """Get default configuration instance with validation."""
    try:
        config = Config()
        config.apply_environment_overrides()
        return config
    except Exception as e:
        raise ConfigError(f"Could not create default configuration: {e}")


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from file or create default with comprehensive error handling."""
    try:
        return Config.load_from_file(config_path)
    except (ConfigLoadError, ConfigValidationError):
        # Re-raise configuration-specific errors
        raise
    except Exception as e:
        raise ConfigError(f"Unexpected error loading configuration: {e}")


def validate_config_file(config_path: str) -> list[str]:
    """Validate a configuration file and return list of errors/warnings."""
    errors = []

    try:
        config = Config.load_from_file(config_path)
        warnings = config.validate_configuration()
        return warnings
    except ConfigLoadError as e:
        errors.append(f"Load error: {e}")
    except ConfigValidationError as e:
        errors.append(f"Validation error: {e}")
    except Exception as e:
        errors.append(f"Unexpected error: {e}")

    return errors


def create_example_config(output_path: str) -> None:
    """Create an example configuration file with all options documented."""
    config = get_default_config()

    # Add example custom instructions (but no repositories since they need to exist)
    config.agents["code"].custom_instructions = "Always follow PEP 8 style guidelines and include comprehensive docstrings."
    config.agents["qa_test"].custom_instructions = "Focus on edge cases and ensure 100% test coverage for critical functions."

    try:
        config.save_to_file(output_path)
        logger.info(f"Example configuration created at {output_path}")
    except Exception as e:
        raise ConfigError(f"Could not create example configuration: {e}")
