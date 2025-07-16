#!/usr/bin/env python3
"""
Quick Start Script - Just start the server directly
"""
import sys
import os
from pathlib import Path

# Set the working directory to the project directory
project_dir = Path(__file__).parent
os.chdir(project_dir)

# Add project directory to Python path
sys.path.insert(0, str(project_dir))

print("[RAG] Starting RAG System...")
print(f"[RAG] Working directory: {os.getcwd()}")
print(f"[RAG] Server will be available at: http://localhost:8001")
print("[RAG] Frontend will be available at: http://localhost:8001/simple_frontend.html")
print("[RAG] API docs will be available at: http://localhost:8001/docs")
print()
print("[RAG] Press Ctrl+C to stop the server")
print("=" * 60)

# Import and run the FastAPI app directly
try:
    from simple_api import app
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False
    )
except KeyboardInterrupt:
    print("\n[RAG] Server stopped by user")
except Exception as e:
    print(f"[RAG] Error starting server: {e}")
    input("Press Enter to exit...")
    sys.exit(1)