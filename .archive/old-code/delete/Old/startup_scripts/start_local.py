#!/usr/bin/env python3
"""
Start all RAG system services locally
"""
import os
import sys
import time
import subprocess
import platform
import signal
from pathlib import Path
import threading
import webbrowser


# Store process handles
processes = []


def is_windows():
    return platform.system() == "Windows"


def get_venv_python(service_path):
    """Get the python executable from venv"""
    if is_windows():
        return os.path.join(service_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(service_path, "venv", "bin", "python")


def start_service(name, command, cwd=None, env=None):
    """Start a service in the background"""
    print(f"Starting {name}...")
    
    try:
        if env is None:
            env = os.environ.copy()
        
        # Add Python path to environment
        if cwd:
            env['PYTHONPATH'] = cwd
        
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            shell=is_windows(),
            creationflags=subprocess.CREATE_NEW_CONSOLE if is_windows() else 0
        )
        
        processes.append((name, process))
        print(f"✓ {name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"✗ Failed to start {name}: {e}")
        return None


def check_service_health(url, name, max_retries=10):
    """Check if a service is responding"""
    import urllib.request
    import urllib.error
    
    for i in range(max_retries):
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"✓ {name} is responding")
            return True
        except:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                print(f"✗ {name} is not responding after {max_retries} attempts")
                return False


def stop_all_services():
    """Stop all running services"""
    print("\nStopping all services...")
    for name, process in processes:
        try:
            if is_windows():
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
            else:
                process.terminate()
            print(f"Stopped {name}")
        except:
            pass


def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    print("\nReceived interrupt signal...")
    stop_all_services()
    sys.exit(0)


def main():
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting RAG System Services Locally")
    print("=" * 50)
    
    # Check if .env exists
    if not os.path.exists(".env"):
        if os.path.exists(".env.local"):
            print("Copying .env.local to .env...")
            import shutil
            shutil.copy(".env.local", ".env")
        else:
            print("ERROR: No .env file found!")
            print("Please copy .env.example to .env and configure it.")
            return 1
    
    # Start Qdrant
    qdrant_path = Path("qdrant")
    if qdrant_path.exists():
        qdrant_exe = "qdrant.exe" if is_windows() else "./qdrant"
        start_service("Qdrant", [os.path.join(qdrant_path, qdrant_exe)], cwd=str(qdrant_path))
    else:
        print("⚠ Qdrant not found. Please download it first.")
    
    time.sleep(3)
    
    # Start Redis
    redis_cmd = "redis-server" if not is_windows() else "redis-server.exe"
    start_service("Redis", [redis_cmd])
    
    time.sleep(2)
    
    # Start API Gateway
    api_path = Path("services/api-gateway")
    api_python = get_venv_python(api_path)
    start_service(
        "API Gateway",
        [api_python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(api_path)
    )
    
    time.sleep(3)
    
    # Start Document Processor
    doc_path = Path("services/document-processor")
    doc_python = get_venv_python(doc_path)
    start_service(
        "Document Processor",
        [doc_python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"],
        cwd=str(doc_path)
    )
    
    # Start Celery Worker
    start_service(
        "Celery Worker",
        [doc_python, "-m", "celery", "-A", "app.processor", "worker", "--loglevel=info"],
        cwd=str(doc_path)
    )
    
    time.sleep(2)
    
    # Start Vector Engine
    vec_path = Path("services/vector-engine")
    vec_python = get_venv_python(vec_path)
    start_service(
        "Vector Engine",
        [vec_python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"],
        cwd=str(vec_path)
    )
    
    time.sleep(2)
    
    # Start Web Interface
    web_path = Path("services/web-interface")
    npm_cmd = "npm.cmd" if is_windows() else "npm"
    start_service(
        "Web Interface",
        [npm_cmd, "start"],
        cwd=str(web_path),
        env={**os.environ, "BROWSER": "none"}  # Prevent auto-opening browser
    )
    
    print("\n" + "=" * 50)
    print("Waiting for services to be ready...")
    
    # Check service health
    time.sleep(5)
    all_healthy = True
    all_healthy &= check_service_health("http://localhost:8000/health", "API Gateway")
    all_healthy &= check_service_health("http://localhost:6333/", "Qdrant")
    
    if all_healthy:
        print("\n✓ All services are running!")
        print("\nAccess points:")
        print("- Web Interface: http://localhost:3000")
        print("- API Documentation: http://localhost:8000/docs")
        print("- Qdrant Dashboard: http://localhost:6333/dashboard")
        print("\nDefault login: admin / admin123")
        
        # Optionally open browser
        time.sleep(2)
        webbrowser.open("http://localhost:3000")
    else:
        print("\n⚠ Some services failed to start properly.")
        print("Check the console output for errors.")
    
    print("\nPress Ctrl+C to stop all services...")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())