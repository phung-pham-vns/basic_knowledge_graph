import enum
from abc import ABC
from pathlib import Path
from tempfile import gettempdir

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class ProjectBaseSettings(BaseSettings, ABC):
    """Base settings for the project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMSettings(ProjectBaseSettings):
    """Settings for the LLM-related options."""

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_endpoint: str = ""
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 16384
    llm_small_model: str | None = None
    llm_thinking_budget: int | None = None


class GraphDBSettings(ProjectBaseSettings):
    """Settings for graph database connection."""

    graph_db_provider: str = "neo4j"
    graph_db_url: str = "neo4j://localhost:7687"
    graph_db_user: str = "neo4j"
    graph_db_password: str = "aisac_kg"


class ProjectSettings(GraphDBSettings, LLMSettings):
    """Application settings.

    These parameters can be configured with environment variables.
    """

    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # quantity of workers for uvicorn
    api_workers_count: int = 1
    # Enable uvicorn reloading
    api_reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    # This variable is used to define
    # multiproc_dir. It's required for [uvi|guni]corn projects.
    prometheus_dir_path: Path = TEMP_DIR / "prom"

    # Sentry's configuration.
    sentry_dsn: str | None = None
    sentry_sample_rate: float = 1.0

    available_kg_group_ids: list[str] = Field(default_factory=list)

    @property
    def graph_db(self) -> GraphDBSettings:
        """Get the graph database settings."""
        return GraphDBSettings(**self.model_dump())

    @property
    def llm(self) -> LLMSettings:
        """Get the LLM settings."""
        return LLMSettings(**self.model_dump())


settings = ProjectSettings()
