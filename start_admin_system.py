#!/usr/bin/env python3
"""
Start RAG system with admin interface
"""
import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("[SETUP] Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jinja2==3.1.2"])
        print("[SUCCESS] Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False

def start_server():
    """Start the server with admin interface"""
    print("[STARTING] RAG System with Admin Interface...")
    print("=" * 60)
    
    # Check if jinja2 is available
    try:
        import jinja2
        print("[OK] Jinja2 is available")
    except ImportError:
        print("[INSTALLING] Jinja2 dependency...")
        if not install_dependencies():
            return False
    
    # Test admin system
    try:
        from core.routers import admin
        print("[OK] Admin router ready")
    except ImportError as e:
        print(f"[ERROR] Admin router failed: {e}")
        return False
    
    # Show access information
    print("\n[READY] Starting server...")
    print("🔗 Main RAG System: http://localhost:8000/ui")
    print("🛠️ Admin Interface: http://localhost:8000/admin")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\n💡 Click the 'Settings' button in the main UI to access admin")
    print("\n[SERVER] Starting on port 8000...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Start the server
    try:
        os.system("python -m uvicorn core.main:app --host 0.0.0.0 --port 8000")
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped")
    except Exception as e:
        print(f"[ERROR] Server failed to start: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = start_server()
    if not success:
        print("\n[FAILED] Could not start admin system")
        sys.exit(1)