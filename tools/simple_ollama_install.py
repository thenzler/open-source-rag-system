#!/usr/bin/env python3
"""
Simple Ollama Installer for Windows
"""

import os
import sys
import subprocess
import urllib.request
import tempfile
from pathlib import Path

def download_ollama():
    """Download Ollama installer"""
    print("Downloading Ollama installer...")
    
    url = "https://ollama.ai/download/OllamaSetup.exe"
    
    try:
        # Download to Downloads folder
        downloads_path = Path.home() / "Downloads" / "OllamaSetup.exe"
        urllib.request.urlretrieve(url, str(downloads_path))
        
        print(f"Downloaded to: {downloads_path}")
        return str(downloads_path)
        
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def install_ollama(installer_path):
    """Run Ollama installer"""
    print("Installing Ollama...")
    print("Please run the installer manually and then restart your terminal")
    print(f"Installer location: {installer_path}")
    
    # Open the installer
    try:
        os.startfile(installer_path)
        print("Installer opened. Please complete the installation.")
        return True
    except Exception as e:
        print(f"Could not open installer: {e}")
        return False

def check_ollama_installed():
    """Check if Ollama is installed"""
    print("Checking if Ollama is installed...")
    
    # Common paths
    possible_paths = [
        Path(os.environ.get('USERPROFILE', '')) / 'AppData' / 'Local' / 'Programs' / 'Ollama' / 'ollama.exe',
        Path('C:') / 'Program Files' / 'Ollama' / 'ollama.exe',
        Path('C:') / 'Users' / os.environ.get('USERNAME', '') / 'AppData' / 'Local' / 'Programs' / 'Ollama' / 'ollama.exe'
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found Ollama at: {path}")
            return str(path.parent)
    
    print("Ollama not found in common locations")
    return None

def add_to_path(ollama_dir):
    """Add Ollama to PATH"""
    print(f"Adding {ollama_dir} to PATH...")
    
    try:
        # Add to current session
        current_path = os.environ.get('PATH', '')
        if ollama_dir not in current_path:
            os.environ['PATH'] = f"{current_path};{ollama_dir}"
        
        # Add permanently
        cmd = f'setx PATH "%PATH%;{ollama_dir}"'
        subprocess.run(cmd, shell=True, check=True)
        
        print("Added to PATH successfully")
        return True
        
    except Exception as e:
        print(f"Failed to add to PATH: {e}")
        return False

def test_ollama():
    """Test Ollama"""
    print("Testing Ollama...")
    
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode == 0:
            print(f"Ollama is working! {result.stdout.strip()}")
            return True
        else:
            print("Ollama test failed")
            return False
            
    except Exception as e:
        print(f"Ollama test error: {e}")
        return False

def main():
    print("Ollama Installation Helper")
    print("=" * 30)
    
    # Check if already installed
    ollama_dir = check_ollama_installed()
    
    if ollama_dir:
        print("Ollama is already installed!")
        
        # Add to PATH if needed
        if add_to_path(ollama_dir):
            print("PATH updated")
        
        # Test
        if test_ollama():
            print("\nOllama is ready to use!")
            print("You can now run: ollama pull command-r:latest")
            return True
    
    # Download and install
    print("Ollama not found. Downloading installer...")
    installer_path = download_ollama()
    
    if installer_path:
        install_ollama(installer_path)
        print("\nAfter installation completes:")
        print("1. Restart your terminal")
        print("2. Run this script again")
        print("3. Then run: ollama pull command-r:latest")
    
    return False

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)