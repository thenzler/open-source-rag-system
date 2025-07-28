#!/usr/bin/env python3
"""
Ollama Installer for Windows
"""

import os
import sys
import subprocess
import urllib.request
import tempfile
from pathlib import Path

def download_ollama():
    """Download Ollama installer"""
    print("📥 Downloading Ollama installer...")
    
    url = "https://ollama.ai/download/OllamaSetup.exe"
    
    try:
        # Download to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as tmp_file:
            urllib.request.urlretrieve(url, tmp_file.name)
            installer_path = tmp_file.name
        
        print(f"✅ Downloaded to: {installer_path}")
        return installer_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def install_ollama(installer_path):
    """Run Ollama installer"""
    print("🚀 Installing Ollama...")
    
    try:
        # Run installer silently
        result = subprocess.run([installer_path, '/S'], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)
        
        if result.returncode == 0:
            print("✅ Ollama installed successfully!")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def setup_path():
    """Add Ollama to PATH"""
    print("🔧 Setting up PATH...")
    
    # Common Ollama installation paths
    possible_paths = [
        Path(os.environ['USERPROFILE']) / 'AppData' / 'Local' / 'Programs' / 'Ollama',
        Path('C:') / 'Program Files' / 'Ollama',
        Path('C:') / 'Users' / os.environ['USERNAME'] / 'AppData' / 'Local' / 'Programs' / 'Ollama'
    ]
    
    ollama_path = None
    for path in possible_paths:
        if (path / 'ollama.exe').exists():
            ollama_path = str(path)
            break
    
    if ollama_path:
        print(f"✅ Found Ollama at: {ollama_path}")
        
        # Add to PATH for current session
        current_path = os.environ.get('PATH', '')
        if ollama_path not in current_path:
            os.environ['PATH'] = f"{current_path};{ollama_path}"
            print("✅ Added to PATH for current session")
        
        # Add to PATH permanently (Windows)
        try:
            subprocess.run([
                'powershell', '-Command',
                f'[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";{ollama_path}", "User")'
            ], check=True, capture_output=True)
            print("✅ Added to PATH permanently")
        except:
            print("⚠️  Could not set PATH permanently - you may need to restart terminal")
        
        return ollama_path
    else:
        print("❌ Could not find Ollama installation")
        return None

def test_ollama():
    """Test if Ollama works"""
    print("🧪 Testing Ollama...")
    
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode == 0:
            print(f"✅ Ollama is working! Version: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Ollama test failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Ollama test timed out")
        return False
    except Exception as e:
        print(f"❌ Ollama test error: {e}")
        return False

def start_ollama_service():
    """Start Ollama service"""
    print("🔄 Starting Ollama service...")
    
    try:
        # Start ollama serve in background
        process = subprocess.Popen(['ollama', 'serve'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
        
        print(f"✅ Ollama service started (PID: {process.pid})")
        print("🌐 Ollama is now running on http://localhost:11434")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start Ollama service: {e}")
        return False

def main():
    print("🚀 Ollama Installation Script")
    print("=" * 40)
    
    # Step 1: Download installer
    installer_path = download_ollama()
    if not installer_path:
        print("❌ Could not download Ollama")
        return False
    
    # Step 2: Install Ollama
    if not install_ollama(installer_path):
        print("❌ Installation failed")
        return False
    
    # Step 3: Setup PATH
    ollama_path = setup_path()
    if not ollama_path:
        print("❌ Could not setup PATH")
        return False
    
    # Step 4: Test installation
    if not test_ollama():
        print("❌ Ollama test failed")
        return False
    
    # Step 5: Start service
    start_ollama_service()
    
    print("\n🎉 Ollama installation complete!")
    print("\n📋 Next steps:")
    print("1. Restart your terminal/PowerShell")
    print("2. Run: ollama pull command-r:latest")
    print("3. Run: python simple_api.py")
    
    # Cleanup
    try:
        os.unlink(installer_path)
    except:
        pass
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)