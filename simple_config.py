"""
Simple configuration for RAG system
"""
import os
from pathlib import Path

class Config:
    """Simple configuration"""
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    
# Copy to config directory
config_dir = Path(__file__).parent / "config"
config_dir.mkdir(exist_ok=True)

config_content = '''"""
Central configuration for RAG system
"""
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    @classmethod
    def get_llm_config_path(cls):
        return cls.BASE_DIR / "config" / "llm_config.yaml"

config = Config()
'''

with open(config_dir / "config.py", "w") as f:
    f.write(config_content)

print("Config created successfully")