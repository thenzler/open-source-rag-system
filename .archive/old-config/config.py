"""
Central configuration management for the RAG system.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Central configuration for the RAG system"""
    
    # Base paths
    BASE_DIR = Path(os.getenv('RAG_SYSTEM_BASE_DIR', Path.cwd()))
    DATA_DIR = BASE_DIR / os.getenv('RAG_DATA_DIR', 'data')
    CONFIG_DIR = BASE_DIR / os.getenv('RAG_CONFIG_DIR', 'config')
    
    # Storage paths
    UPLOAD_DIR = BASE_DIR / os.getenv('RAG_UPLOAD_DIR', 'data/storage/uploads')
    PROCESSED_DIR = BASE_DIR / os.getenv('RAG_PROCESSED_DIR', 'data/storage/processed')
    OUTPUT_DIR = BASE_DIR / os.getenv('RAG_OUTPUT_DIR', 'output')
    
    # Ollama configuration
    OLLAMA_EXECUTABLE = os.getenv('OLLAMA_EXECUTABLE', 'ollama')
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    
    # API configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key-change-in-production')
    
    # Development settings
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # File handling
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(50 * 1024 * 1024)))  # 50MB default
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.json', '.md'}
    
    @classmethod
    def get_llm_config_path(cls) -> Path:
        """Get the path to the LLM configuration file"""
        return cls.CONFIG_DIR / "llm_config.yaml"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        for directory in [cls.DATA_DIR, cls.UPLOAD_DIR, cls.PROCESSED_DIR, 
                         cls.OUTPUT_DIR, cls.CONFIG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate the configuration"""
        if not cls.BASE_DIR.exists():
            raise ValueError(f"Base directory does not exist: {cls.BASE_DIR}")
        
        if cls.SECRET_KEY == 'default-secret-key-change-in-production' and not cls.DEBUG:
            raise ValueError("SECRET_KEY must be changed in production")

# Singleton instance
config = Config()

# Ensure directories exist on import
config.ensure_directories()