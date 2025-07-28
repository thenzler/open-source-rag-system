#!/usr/bin/env python3
"""
Simple RAG System Startup Script with Dependency Verification
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def is_windows():
    return platform.system() == "Windows"

def run_dependency_checks():
    """Run comprehensive dependency checks before starting"""
    print("Running dependency checks...")
    print("=" * 40)
    
    try:
        # Import and run the startup checks
        sys.path.insert(0, str(Path(__file__).parent))
        from startup_checks import StartupChecker
        
        checker = StartupChecker()
        results = checker.run_all_checks()
        
        # Check if critical components are ready
        critical_checks = ['python_version', 'required_packages', 'storage_directories']
        all_critical_passed = all(results[key] for key in critical_checks)
        
        if not all_critical_passed:
            print("\n[FAIL] Critical dependency checks failed!")
            print("Please resolve the issues above before starting the system.")
            sys.exit(1)
            
        # Check if Ollama is available (non-critical)
        if not results.get('ollama_service', False):
            print("\n[WARN] Ollama service is not running.")
            print("The system will start but AI features will not be available.")
            print("Start Ollama with: ollama serve")
            
        print("\n[OK] Dependency checks passed! Starting system...")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Could not import startup checks: {e}")
        print("Continuing with basic checks...")
        return False
    except Exception as e:
        print(f"[FAIL] Error during dependency checks: {e}")
        print("Continuing with basic checks...")
        return False

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
    dirs = ["data/storage", "data/storage/uploads", "data/storage/processed"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✓ Directories created")

def start_api():
    """Start the FastAPI server"""
    print("Starting Simple RAG API...")
    print("=" * 50)
    
    try:
        # Get the correct path to simple_api.py
        api_path = Path(__file__).parent / "simple_api.py"
        project_dir = Path(__file__).parent
        
        # Start the API server with correct working directory
        subprocess.run([
            sys.executable, str(api_path)
        ], cwd=str(project_dir))
    except KeyboardInterrupt:
        print("\n[INFO] Simple RAG API stopped")
    except Exception as e:
        print(f"[FAIL] Error starting API: {e}")
        sys.exit(1)

def main():
    print("Simple RAG System Startup")
    print("=" * 30)
    
    # Run comprehensive dependency checks first
    dependency_checks_passed = run_dependency_checks()
    
    # If dependency checks failed, fall back to basic checks
    if not dependency_checks_passed:
        print("\nRunning basic checks...")
        check_python_version()
        install_requirements()
        create_directories()
    
    # Start API
    start_api()

if __name__ == "__main__":
    main()