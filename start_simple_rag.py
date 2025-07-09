#!/usr/bin/env python3
"""
Simple RAG System Startup Script
"""
import os
import sys
import subprocess
import platform

def is_windows():
    return platform.system() == "Windows"

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "simple_requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Requirements installed successfully")
        else:
            print("✗ Failed to install requirements:")
            print(result.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error installing requirements: {e}")
        sys.exit(1)

def create_directories():
    """Create required directories"""
    dirs = ["storage", "storage/uploads", "storage/processed"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✓ Directories created")

def start_api():
    """Start the FastAPI server"""
    print("Starting Simple RAG API...")
    print("=" * 50)
    
    try:
        # Start the API server
        subprocess.run([
            sys.executable, "simple_api.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 Simple RAG API stopped")
    except Exception as e:
        print(f"✗ Error starting API: {e}")
        sys.exit(1)

def main():
    print("Simple RAG System Startup")
    print("=" * 30)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Start API
    start_api()

if __name__ == "__main__":
    main()