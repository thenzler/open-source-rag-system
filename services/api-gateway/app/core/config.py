"""
Configuration settings for the RAG System
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "RAG System API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://raguser:password@localhost:5432/ragdb",
        env="DATABASE_URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Service URLs
    document_processor_url: str = Field(
        default="http://localhost:8001",
        env="DOCUMENT_PROCESSOR_URL"
    )
    vector_engine_url: str = Field(
        default="http://localhost:8002",
        env="VECTOR_ENGINE_URL"
    )
    llm_service_url: str = Field(
        default="http://localhost:11434",
        env="LLM_SERVICE_URL"
    )
    
    # LLM Configuration
    llm_model_name: str = Field(default="llama3.1:8b", env="LLM_MODEL_NAME")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    
    # Document Processing Configuration
    upload_directory: str = Field(default="./storage/uploads", env="UPLOAD_DIRECTORY")
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    supported_file_types: List[str] = Field(
        default=["pdf", "docx", "doc", "txt", "csv", "xlsx", "xml", "html"],
        env="SUPPORTED_FILE_TYPES"
    )
    
    # Text Processing Configuration
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    
    # Query Configuration
    max_query_length: int = Field(default=1000, env="MAX_QUERY_LENGTH")
    max_search_results: int = Field(default=50, env="MAX_SEARCH_RESULTS")
    default_top_k: int = Field(default=5, env="DEFAULT_TOP_K")
    
    # Feature Flags
    enable_query_expansion: bool = Field(default=False, env="ENABLE_QUERY_EXPANSION")
    enable_reranking: bool = Field(default=False, env="ENABLE_RERANKING")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="your-jwt-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Monitoring Configuration
    enable_prometheus: bool = Field(default=True, env="ENABLE_PROMETHEUS")
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # Health Check Configuration
    health_check_timeout: int = Field(default=5, env="HEALTH_CHECK_TIMEOUT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Background Tasks
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Vector Database Configuration
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_collection_name: str = Field(default="documents", env="QDRANT_COLLECTION_NAME")
    vector_dimension: int = Field(default=768, env="VECTOR_DIMENSION")
    
    # Cache Configuration
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    cache_max_entries: int = Field(default=10000, env="CACHE_MAX_ENTRIES")
    
    # Development Configuration
    auto_reload: bool = Field(default=True, env="AUTO_RELOAD")
    show_sql: bool = Field(default=False, env="SHOW_SQL")
    
    # Testing Configuration
    test_database_url: str = Field(
        default="postgresql+asyncpg://raguser:password@localhost:5432/ragdb_test",
        env="TEST_DATABASE_URL"
    )
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL for migrations."""
        return self.database_url.replace("+asyncpg", "")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment.lower() == "testing"
    
    def get_upload_path(self) -> Path:
        """Get upload directory path."""
        path = Path(self.upload_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get log file path if configured."""
        if self.log_file:
            return Path(self.log_file)
        return None
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Field validation
        validate_assignment = True
        use_enum_values = True
        
        # Example values for documentation
        schema_extra = {
            "example": {
                "app_name": "RAG System API",
                "environment": "development",
                "debug": True,
                "database_url": "postgresql+asyncpg://raguser:password@localhost:5432/ragdb",
                "redis_url": "redis://localhost:6379/0",
                "llm_model_name": "llama3.1:8b",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "max_file_size_mb": 100,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "enable_query_expansion": True,
                "enable_reranking": True,
                "enable_caching": True
            }
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(**kwargs) -> None:
    """Override settings (for testing)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    
    for key, value in kwargs.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)


def reset_settings() -> None:
    """Reset settings to default (for testing)."""
    global _settings
    _settings = None


# Environment-specific configurations
def get_development_settings() -> Settings:
    """Get development-specific settings."""
    return Settings(
        environment="development",
        debug=True,
        log_level="DEBUG",
        auto_reload=True,
        show_sql=True,
        enable_metrics=True,
        cors_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"]
    )


def get_production_settings() -> Settings:
    """Get production-specific settings."""
    return Settings(
        environment="production",
        debug=False,
        log_level="INFO",
        auto_reload=False,
        show_sql=False,
        enable_metrics=True,
        cors_origins=["https://your-domain.com"]
    )


def get_testing_settings() -> Settings:
    """Get testing-specific settings."""
    return Settings(
        environment="testing",
        debug=False,
        log_level="WARNING",
        auto_reload=False,
        show_sql=False,
        enable_metrics=False,
        database_url="postgresql+asyncpg://raguser:password@localhost:5432/ragdb_test",
        redis_url="redis://localhost:6379/1",  # Different Redis DB for testing
        enable_caching=False,
        enable_query_expansion=False,
        enable_reranking=False
    )


# Configuration validation
def validate_settings(settings: Settings) -> None:
    """Validate settings configuration."""
    errors = []
    
    # Required settings
    if settings.secret_key == "your-secret-key-here":
        errors.append("SECRET_KEY must be set to a secure value")
    
    if settings.jwt_secret_key == "your-jwt-secret-key":
        errors.append("JWT_SECRET_KEY must be set to a secure value")
    
    # File size limits
    if settings.max_file_size_mb <= 0:
        errors.append("MAX_FILE_SIZE_MB must be greater than 0")
    
    if settings.max_file_size_mb > 1000:  # 1GB limit
        errors.append("MAX_FILE_SIZE_MB should not exceed 1000 MB")
    
    # Chunk configuration
    if settings.chunk_size <= 0:
        errors.append("CHUNK_SIZE must be greater than 0")
    
    if settings.chunk_overlap >= settings.chunk_size:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    
    # Query limits
    if settings.max_query_length <= 0:
        errors.append("MAX_QUERY_LENGTH must be greater than 0")
    
    if settings.max_search_results <= 0:
        errors.append("MAX_SEARCH_RESULTS must be greater than 0")
    
    # URL validation
    required_urls = [
        "database_url",
        "redis_url",
        "document_processor_url",
        "vector_engine_url",
        "llm_service_url",
        "qdrant_url"
    ]
    
    for url_field in required_urls:
        url_value = getattr(settings, url_field)
        if not url_value or not url_value.strip():
            errors.append(f"{url_field.upper()} must be set")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")


# Initialize settings based on environment
def initialize_settings() -> Settings:
    """Initialize settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        settings = get_production_settings()
    elif env == "testing":
        settings = get_testing_settings()
    else:
        settings = get_development_settings()
    
    # Validate settings
    if not settings.is_testing:
        validate_settings(settings)
    
    return settings
