import enum
from abc import ABC
from pathlib import Path
from tempfile import gettempdir

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class ProjectBaseSettings(BaseSettings, ABC):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMSettings(ProjectBaseSettings):
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-pro"
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 16384


class EmbeddingSettings(ProjectBaseSettings):
    embedding_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embedding_provider: str | None = "gemini"
    embedding_model: str | None = "embedding-001"
    embedding_dimensions: int | None = 3072
    embedding_api_key: str | None = None
    embedding_max_tokens: int | None = 8191
    embedding_rate_limit_enabled: bool = False
    embedding_rate_limit_requests: int = 60
    embedding_rate_limit_interval: int = 60


class RerankerSettings(ProjectBaseSettings):
    reranker_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    reranker_provider: str | None = "gemini"
    reranker_model: str | None = "gemini-2.5-flash-lite-preview-06-17"
    reranker_dimensions: int | None = 3072
    reranker_api_key: str | None = None
    reranker_max_tokens: int | None = 8191


class GraphDBSettings(ProjectBaseSettings):
    graph_db_provider: str = "neo4j"
    graph_db_url: str = "bolt://localhost:7687"
    graph_db_host: str = "localhost"
    graph_db_port: int = 7687
    graph_db_username: str | None = None
    graph_db_password: str | None = None
    graph_db_database: str = "default_db"


class VectorDBSettings(ProjectBaseSettings):
    vector_db_provider: str = "milvus"
    vector_db_url: str = "http://localhost:19530"
    vector_db_key: str = ""
    vector_db_username: str | None = None
    vector_db_password: str | None = None
    vector_db_db_name: str = "aisac_durian_pest_and_disease"


class ProjectSettings(GraphDBSettings, LLMSettings):
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

    @property
    def graph_db(self) -> GraphDBSettings:
        """Get the graph database settings."""
        return GraphDBSettings(**self.model_dump())

    @property
    def llm(self) -> LLMSettings:
        """Get the LLM settings."""
        return LLMSettings(**self.model_dump())

    @property
    def embedding(self) -> EmbeddingSettings:
        """Get the embedding settings."""
        return EmbeddingSettings(**self.model_dump())

    @property
    def vector_db(self) -> VectorDBSettings:
        """Get the vector database settings."""
        return VectorDBSettings(**self.model_dump())

    @property
    def reranker(self) -> RerankerSettings:
        """Get the reranker settings."""
        return RerankerSettings(**self.model_dump())


settings = ProjectSettings()
