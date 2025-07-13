#!/usr/bin/env python3
"""
🤖 Project SUSI Launcher
Beautiful startup script for Smart Universal Search Intelligence
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def print_banner():
    """Display beautiful Project SUSI banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║    🤖 PROJECT SUSI - Smart Universal Search Intelligence         ║
    ║                                                                  ║
    ║    Advanced AI-powered document processing and search system     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version.split()[0])
    
    # Check required files
    required_files = [
        "simple_api.py",
        "project_susi_frontend.html",
        "ollama_client.py"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file missing: {file}")
            return False
        print(f"✅ Found: {file}")
    
    return True

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing requirements...")
    
    requirements = [
        "fastapi",
        "uvicorn[standard]",
        "sentence-transformers",
        "numpy",
        "python-multipart",
        "requests",
        "PyYAML",
        "rank-bm25"
    ]
    
    try:
        for req in requirements:
            print(f"  Installing {req}...")
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                         check=True, capture_output=True)
        print("✅ All requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_ollama():
    """Check if Ollama is available"""
    print("\n🤖 Checking Ollama AI engine...")
    
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Ollama found: {result.stdout.strip()}")
            
            # Check for available models
            try:
                models_result = subprocess.run(["ollama", "list"], 
                                             capture_output=True, text=True, timeout=10)
                if models_result.returncode == 0 and len(models_result.stdout.strip().split('\n')) > 1:
                    print("✅ AI models available")
                    return True
                else:
                    print("⚠️  No AI models found. Consider running:")
                    print("   ollama pull command-r:latest")
                    print("   ollama pull mistral:latest")
                    return True
            except:
                return True
        else:
            print("❌ Ollama not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not found. Please install from: https://ollama.ai/download")
        print("   Or run: python install_ollama.py")
        return False

def start_server():
    """Start the Project SUSI server"""
    print("\n🚀 Starting Project SUSI server...")
    
    try:
        # Start server in background
        process = subprocess.Popen([
            sys.executable, "simple_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server to initialize...")
        for i in range(10):
            time.sleep(1)
            try:
                import requests
                response = requests.get("http://localhost:8001/api/v1/status", timeout=2)
                if response.status_code == 200:
                    print("✅ Server is running!")
                    return process
            except:
                pass
            print(f"   Starting... ({i+1}/10)")
        
        print("⚠️  Server started but may still be initializing")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def open_interface():
    """Open the beautiful Project SUSI interface"""
    print("\n🎨 Opening Project SUSI interface...")
    
    interface_path = Path("project_susi_frontend.html").resolve()
    if interface_path.exists():
        try:
            webbrowser.open(f"file://{interface_path}")
            print("✅ Interface opened in your default browser")
            return True
        except Exception as e:
            print(f"❌ Could not open interface: {e}")
            print(f"   Please manually open: {interface_path}")
            return False
    else:
        print("❌ Interface file not found")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check system requirements
    if not check_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        input("Press Enter to exit...")
        return
    
    # Install requirements if needed
    try:
        import fastapi
        import uvicorn
        import sentence_transformers
        print("✅ Core requirements already installed")
    except ImportError:
        if not install_requirements():
            print("\n❌ Failed to install requirements.")
            input("Press Enter to exit...")
            return
    
    # Check Ollama
    ollama_available = check_ollama()
    
    # Start server
    server_process = start_server()
    if not server_process:
        print("\n❌ Failed to start server.")
        input("Press Enter to exit...")
        return
    
    # Open interface
    open_interface()
    
    # Display status
    print("\n🎉 Project SUSI is now running!")
    print("=" * 50)
    print("🌐 Server URL: http://localhost:8001")
    print("📚 API Docs: http://localhost:8001/docs")
    print("💻 Interface: project_susi_frontend.html")
    
    if ollama_available:
        print("🤖 AI Engine: Ready")
    else:
        print("🤖 AI Engine: Not available (install Ollama for AI features)")
    
    print("\n📋 Useful commands:")
    print("• python manage_llm.py list      - List AI models")
    print("• python manage_llm.py status    - Check system status")
    print("• python manage_llm.py switch    - Switch AI models")
    print("\n⚠️  Press Ctrl+C to stop the server")
    
    try:
        # Keep script running and monitor server
        while True:
            time.sleep(1)
            if server_process.poll() is not None:
                print("\n❌ Server stopped unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping Project SUSI...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("✅ Project SUSI stopped successfully")

if __name__ == "__main__":
    main()