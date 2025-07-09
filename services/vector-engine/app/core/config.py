"""
Configuration settings for Vector Engine Service.
"""
from functools import lru_cache
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Service settings
    service_name: str = "vector-engine"
    service_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8002, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Redis settings (for caching)
    redis_url: str = Field(
        default="redis://redis:6379/1",
        env="REDIS_URL"
    )
    
    # Qdrant settings
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        env="QDRANT_URL"
    )
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="documents",
        env="QDRANT_COLLECTION_NAME"
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    vector_dimension: int = Field(default=768, env="VECTOR_DIMENSION")
    
    # Performance settings
    enable_quantization: bool = Field(default=False, env="ENABLE_QUANTIZATION")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    max_batch_size: int = Field(default=100, env="MAX_BATCH_SIZE")
    
    # Search settings
    default_top_k: int = Field(default=10, env="DEFAULT_TOP_K")
    max_top_k: int = Field(default=100, env="MAX_TOP_K")
    default_score_threshold: float = Field(default=0.0, env="DEFAULT_SCORE_THRESHOLD")
    
    # Security settings
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9092, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()