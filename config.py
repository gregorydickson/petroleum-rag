"""Configuration management using Pydantic Settings.

This module provides centralized configuration for all components of the
petroleum RAG benchmark system, loading values from environment variables.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    See .env.example for all available configuration options.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys - Document Parsers
    llama_cloud_api_key: str = Field(default="", description="LlamaParse API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Google Cloud - Vertex AI
    vertex_api_key: str = Field(
        default="",
        description="Vertex AI Studio API key (simpler authentication)",
    )
    google_application_credentials: str | None = Field(
        default=None,
        description="Path to Google Cloud service account key (alternative to API key)",
    )
    google_cloud_project: str | None = Field(
        default=None,
        description="Google Cloud project ID",
    )
    vertex_docai_processor_id: str | None = Field(
        default=None,
        description="Vertex Document AI processor ID",
    )
    vertex_docai_location: str = Field(
        default="us",
        description="Vertex Document AI location",
    )

    # PageIndex (if requires API key)
    pageindex_api_key: str = Field(default="", description="PageIndex API key")

    # Storage Backend Configurations
    # ChromaDB
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, description="ChromaDB port")
    chroma_collection_name: str = Field(
        default="petroleum_docs",
        description="ChromaDB collection name",
    )

    # Weaviate
    weaviate_host: str = Field(default="localhost", description="Weaviate host")
    weaviate_port: int = Field(default=8080, description="Weaviate HTTP port")
    weaviate_grpc_port: int = Field(default=50051, description="Weaviate gRPC port")
    weaviate_class_name: str = Field(
        default="PetroleumDocument",
        description="Weaviate class name",
    )

    # FalkorDB (Redis-based)
    falkordb_host: str = Field(default="localhost", description="FalkorDB host")
    falkordb_port: int = Field(default=6379, description="FalkorDB port")
    falkordb_graph_name: str = Field(
        default="petroleum_rag",
        description="FalkorDB graph name",
    )

    # Parser Settings
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Default chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Default chunk overlap in characters",
    )
    min_chunk_size: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Minimum chunk size",
    )
    max_chunk_size: int = Field(
        default=2000,
        ge=500,
        le=8000,
        description="Maximum chunk size",
    )

    # Embedding Model Settings
    embedding_provider: Literal["openai", "vertex"] = Field(
        default="openai",
        description="Embedding provider (openai or vertex)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model or Vertex model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
    )
    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Batch size for embedding generation",
    )

    # Vertex AI Embedding Settings
    vertex_embedding_model: str = Field(
        default="textembedding-gecko@003",
        description="Vertex AI embedding model name",
    )

    # Evaluation Settings
    eval_llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="LLM model for evaluation and answer generation",
    )
    eval_llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for evaluation LLM",
    )
    eval_llm_max_tokens: int = Field(
        default=4096,
        ge=100,
        le=200000,
        description="Max tokens for LLM responses",
    )

    # Retrieval Settings
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of chunks to retrieve",
    )
    retrieval_min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold",
    )

    # Benchmark Settings
    benchmark_parallel_parsers: bool = Field(
        default=True,
        description="Run parsers in parallel",
    )
    benchmark_parallel_storage: bool = Field(
        default=True,
        description="Store in backends in parallel",
    )
    benchmark_save_intermediate_results: bool = Field(
        default=True,
        description="Save intermediate parsing/storage results",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_file: str | None = Field(
        default="benchmark.log",
        description="Log file path",
    )

    # Caching Settings
    enable_cache: bool = Field(
        default=True,
        description="Enable caching for embeddings and LLM responses",
    )
    cache_dir: Path = Field(
        default=Path("data/cache"),
        description="Base directory for cache storage",
    )
    cache_max_memory_items: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum items in memory cache per cache instance",
    )
    cache_embedding_enabled: bool = Field(
        default=True,
        description="Enable caching for embeddings",
    )
    cache_llm_enabled: bool = Field(
        default=True,
        description="Enable caching for LLM responses",
    )

    # Rate Limits (requests per minute)
    openai_rate_limit: int = Field(
        default=3000,
        ge=1,
        le=100000,
        description="OpenAI API rate limit (requests per minute)",
    )
    anthropic_rate_limit: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Anthropic API rate limit (requests per minute)",
    )
    llamaparse_rate_limit: int = Field(
        default=600,
        ge=1,
        le=10000,
        description="LlamaParse API rate limit (requests per minute)",
    )
    vertex_rate_limit: int = Field(
        default=600,
        ge=1,
        le=10000,
        description="Vertex Document AI rate limit (requests per minute)",
    )

    # Development
    debug: bool = Field(default=False, description="Enable debug mode")
    enable_profiling: bool = Field(default=False, description="Enable profiling")

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Validate chunk overlap is less than chunk size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @property
    def chroma_url(self) -> str:
        """Get ChromaDB connection URL."""
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @property
    def weaviate_url(self) -> str:
        """Get Weaviate connection URL."""
        return f"http://{self.weaviate_host}:{self.weaviate_port}"

    @property
    def falkordb_connection_string(self) -> str:
        """Get FalkorDB connection string."""
        return f"redis://{self.falkordb_host}:{self.falkordb_port}"

    def get_parser_config(self) -> dict[str, int]:
        """Get parser configuration dictionary.

        Returns:
            Dictionary with parser configuration
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }

    def get_storage_config(self, backend: Literal["chroma", "weaviate", "falkordb"]) -> dict:
        """Get storage backend configuration.

        Args:
            backend: Storage backend name

        Returns:
            Configuration dictionary for the specified backend
        """
        if backend == "chroma":
            return {
                "host": self.chroma_host,
                "port": self.chroma_port,
                "collection_name": self.chroma_collection_name,
                "top_k": self.retrieval_top_k,
                "min_score": self.retrieval_min_score,
            }
        elif backend == "weaviate":
            return {
                "host": self.weaviate_host,
                "port": self.weaviate_port,
                "grpc_port": self.weaviate_grpc_port,
                "class_name": self.weaviate_class_name,
                "top_k": self.retrieval_top_k,
                "min_score": self.retrieval_min_score,
            }
        elif backend == "falkordb":
            return {
                "host": self.falkordb_host,
                "port": self.falkordb_port,
                "graph_name": self.falkordb_graph_name,
                "top_k": self.retrieval_top_k,
                "min_score": self.retrieval_min_score,
            }
        else:
            raise ValueError(f"Unknown storage backend: {backend}")

    def validate_required_keys(self) -> list[str]:
        """Validate that required API keys are set.

        Returns:
            List of missing API key names
        """
        missing = []

        if not self.anthropic_api_key:
            missing.append("anthropic_api_key")

        if not self.openai_api_key:
            missing.append("openai_api_key")

        return missing

    def get_cache_config(self) -> dict[str, Any]:
        """Get cache configuration dictionary.

        Returns:
            Dictionary with cache configuration
        """
        return {
            "enabled": self.enable_cache,
            "base_dir": self.cache_dir,
            "embedding_dir": self.cache_dir / "embeddings",
            "llm_dir": self.cache_dir / "llm",
            "max_memory_items": self.cache_max_memory_items,
            "embedding_enabled": self.cache_embedding_enabled and self.enable_cache,
            "llm_enabled": self.cache_llm_enabled and self.enable_cache,
        }


# Global settings instance
settings = Settings()

# Initialize global caches
from utils.cache import initialize_caches

cache_config = settings.get_cache_config()
initialize_caches(
    embedding_dir=cache_config["embedding_dir"],
    llm_dir=cache_config["llm_dir"],
    max_memory_items=cache_config["max_memory_items"],
    embedding_enabled=cache_config["embedding_enabled"],
    llm_enabled=cache_config["llm_enabled"],
)
