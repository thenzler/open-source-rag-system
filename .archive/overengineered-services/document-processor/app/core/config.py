"""
Configuration settings for Document Processor Service.
"""
from functools import lru_cache
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Service settings
    service_name: str = "document-processor"
    service_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8001, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Database settings
    database_url: str = Field(
        default="postgresql://raguser:password@postgres:5432/ragdb",
        env="DATABASE_URL"
    )
    
    # Redis settings
    redis_url: str = Field(
        default="redis://redis:6379/0",
        env="REDIS_URL"
    )
    
    # Celery settings
    celery_broker: str = Field(
        default="redis://redis:6379/0",
        env="CELERY_BROKER"
    )
    celery_result_backend: str = Field(
        default="redis://redis:6379/0",
        env="CELERY_RESULT_BACKEND"
    )
    
    # Storage settings
    upload_directory: str = Field(
        default="/app/storage/uploads",
        env="UPLOAD_DIRECTORY"
    )
    processed_directory: str = Field(
        default="/app/storage/processed",
        env="PROCESSED_DIRECTORY"
    )
    max_file_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        env="MAX_FILE_SIZE"
    )
    
    # Processing settings
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=100, env="MIN_CHUNK_SIZE")
    
    # Embedding settings
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    
    # OCR settings
    enable_ocr: bool = Field(default=True, env="ENABLE_OCR")
    ocr_language: str = Field(default="eng", env="OCR_LANGUAGE")
    tesseract_config: str = Field(
        default="--oem 3 --psm 6",
        env="TESSERACT_CONFIG"
    )
    
    # Vector database settings
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        env="QDRANT_URL"
    )
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="documents",
        env="QDRANT_COLLECTION_NAME"
    )
    vector_dimension: int = Field(default=768, env="VECTOR_DIMENSION")
    enable_quantization: bool = Field(default=False, env="ENABLE_QUANTIZATION")
    
    # Cache settings
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    
    # Security settings
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()