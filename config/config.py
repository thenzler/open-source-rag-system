"""
Central configuration for RAG system
"""
import os
from pathlib import Path

class Config:
    """Central configuration for the RAG system"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CONFIG_DIR = BASE_DIR / "config"
    
    # Storage paths
    UPLOAD_DIR = DATA_DIR / "storage" / "uploads"  
    PROCESSED_DIR = DATA_DIR / "storage" / "processed"
    OUTPUT_DIR = BASE_DIR / "output"
    
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

# Global config instance
config = Config()
