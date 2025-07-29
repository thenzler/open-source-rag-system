#!/usr/bin/env python3
"""
Startup script for the modular RAG system
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    
    # Import the app from core module
    from core.main import app
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )