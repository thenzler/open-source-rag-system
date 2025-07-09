"""
Configuration management for the RAG System API Gateway.
"""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "RAG System API Gateway"
    debug: bool = Field(default=False, env="DEBUG")
    version: str = "1.0.0"
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    access_log: bool = Field(default=True, env="ACCESS_LOG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/ragdb",
        env="DATABASE_URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    enable_authentication: bool = Field(default=False, env="ENABLE_AUTH")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # File Upload
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    allowed_mime_types: List[str] = Field(
        default=[
            "application/pdf",
            "text/plain",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/markdown"
        ]
    )
    
    # Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Search
    max_query_length: int = Field(default=1000, env="MAX_QUERY_LENGTH")
    max_search_results: int = Field(default=50, env="MAX_SEARCH_RESULTS")
    
    # Models
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    llm_model_name: str = Field(
        default="gpt-3.5-turbo",
        env="LLM_MODEL_NAME"
    )
    
    # Vector Database
    vector_db_url: str = Field(
        default="http://localhost:6333",
        env="VECTOR_DB_URL"
    )
    vector_collection_name: str = Field(
        default="documents",
        env="VECTOR_COLLECTION_NAME"
    )
    
    # External Services
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Features
    enable_query_expansion: bool = Field(default=True, env="ENABLE_QUERY_EXPANSION")
    enable_reranking: bool = Field(default=True, env="ENABLE_RERANKING")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
