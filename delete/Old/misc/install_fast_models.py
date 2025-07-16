#!/usr/bin/env python3
"""
Install Fast Models for Laptop Performance
Run this to download smaller, faster models that work better on laptops
"""

import os
import subprocess
import sys

def install_fast_models():
    """Install fast models for laptop use"""
    
    print("🚀 Installing Fast Models for Better Laptop Performance")
    print("=" * 60)
    
    fast_models = [
        {
            "name": "tinyllama:latest",
            "size": "637MB",
            "speed": "10-30 seconds",
            "description": "Smallest and fastest - Good for basic queries"
        },
        {
            "name": "phi3:mini",
            "size": "2.3GB", 
            "speed": "30-60 seconds",
            "description": "Fast with better quality - Recommended for laptops"
        },
        {
            "name": "orca-mini:latest",
            "size": "1.9GB",
            "speed": "20-40 seconds", 
            "description": "Good balance of speed and quality"
        }
    ]
    
    print("\nRecommended fast models for laptops:")
    print("-" * 60)
    for i, model in enumerate(fast_models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Size: {model['size']}")
        print(f"   Speed: {model['speed']}")
        print(f"   Description: {model['description']}")
    
    print("\n" + "-" * 60)
    choice = input("\nWhich model would you like to install? (1-3, or 'all' for all models): ").strip()
    
    models_to_install = []
    if choice.lower() == 'all':
        models_to_install = fast_models
    elif choice in ['1', '2', '3']:
        models_to_install = [fast_models[int(choice) - 1]]
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n🔄 Installing selected models...")
    for model in models_to_install:
        model_name = model['name']
        print(f"\n📥 Downloading {model_name}...")
        print(f"   This will download ~{model['size']} of data")
        
        try:
            # Run ollama pull command
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully installed {model_name}")
            else:
                print(f"❌ Failed to install {model_name}: {result.stderr}")
                
        except FileNotFoundError:
            print("❌ Ollama not found. Please ensure Ollama is installed and in your PATH")
            print("   Install from: https://ollama.ai/download")
            return
        except Exception as e:
            print(f"❌ Error installing {model_name}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Installation complete!")
    print("\nTo use these models in Project SUSI:")
    print("1. Start the server: python simple_api.py")
    print("2. Open the web interface")
    print("3. Click the Settings button")
    print("4. Select a fast model from the dropdown")
    print("5. Click 'Switch Model'")
    print("\n💡 Tip: tinyllama is the fastest for quick responses!")

if __name__ == "__main__":
    install_fast_models()