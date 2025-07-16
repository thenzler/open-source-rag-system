#!/usr/bin/env python3
"""
Check system requirements for running RAG system locally
"""
import sys
import subprocess
import os
import platform
from pathlib import Path


def check_command(command, name, install_url=None):
    """Check if a command is available"""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        print(f"[OK] {name} is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"[MISSING] {name} is NOT installed")
        if install_url:
            print(f"  Install from: {install_url}")
        return False


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro} is installed")
        return True
    else:
        print(f"[MISSING] Python 3.9+ required, found {version.major}.{version.minor}")
        return False


def check_port(port):
    """Check if a port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result != 0:
        print(f"[OK] Port {port} is available")
        return True
    else:
        print(f"[BUSY] Port {port} is already in use")
        return False


def create_directories():
    """Create necessary directories"""
    dirs = [
        "storage/uploads",
        "storage/processed",
        "logs",
        "models"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("[OK] Created necessary directories")


def main():
    print("RAG System Requirements Check")
    print("=" * 50)
    
    all_good = True
    
    # Check Python
    print("\n1. Checking Python...")
    all_good &= check_python_version()
    
    # Check Node.js
    print("\n2. Checking Node.js...")
    all_good &= check_command("node", "Node.js", "https://nodejs.org/")
    
    # Check Git
    print("\n3. Checking Git...")
    all_good &= check_command("git", "Git", "https://git-scm.com/")
    
    # Check PostgreSQL
    print("\n4. Checking PostgreSQL...")
    if platform.system() == "Windows":
        pg_installed = check_command("psql", "PostgreSQL", "https://www.postgresql.org/download/windows/")
    else:
        pg_installed = check_command("psql", "PostgreSQL", "https://www.postgresql.org/download/")
    all_good &= pg_installed
    
    # Check Redis
    print("\n5. Checking Redis...")
    if platform.system() == "Windows":
        redis_installed = check_command("redis-cli", "Redis", "https://github.com/tporadowski/redis/releases")
    else:
        redis_installed = check_command("redis-cli", "Redis", "https://redis.io/download")
    all_good &= redis_installed
    
    # Check ports
    print("\n6. Checking required ports...")
    ports = {
        8000: "API Gateway",
        8001: "Document Processor",
        8002: "Vector Engine",
        3000: "Web Interface",
        6333: "Qdrant",
        5432: "PostgreSQL",
        6379: "Redis"
    }
    
    for port, service in ports.items():
        if check_port(port):
            all_good &= True
        else:
            print(f"  {service} typically uses this port")
            all_good &= False
    
    # Create directories
    print("\n7. Creating directories...")
    create_directories()
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("[OK] All requirements met! You can proceed with setup.")
        print("\nNext steps:")
        print("1. Install any missing software")
        print("2. Run: pip install -r requirements.txt in each service directory")
        print("3. Configure .env file")
        print("4. Run: python start_local.py")
    else:
        print("[MISSING] Some requirements are missing. Please install them first.")
        print("\nFor detailed instructions, see LOCAL_SETUP.md")
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())