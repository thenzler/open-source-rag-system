#!/usr/bin/env python3
"""
LLM Management CLI Tool
Easy command-line interface for switching models
"""

import argparse
import requests
import json
import sys
from typing import Dict, Any

class LLMManager:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/llm/models")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ Error: {e}")
            return {}
    
    def switch_model(self, model_key: str) -> bool:
        """Switch to a different model"""
        try:
            response = requests.post(f"{self.base_url}/api/v1/llm/switch/{model_key}")
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                print(f"✅ {data.get('message')}")
                if not data.get('available'):
                    print(f"⚠️  {data.get('warning')}")
                return True
            else:
                print(f"❌ Failed to switch model")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Error: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM status"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/llm/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ Error: {e}")
            return {}
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model in Ollama"""
        try:
            print(f"📥 Pulling model: {model_name}...")
            response = requests.post(f"{self.base_url}/api/v1/llm/pull/{model_name}")
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                print(f"✅ {data.get('message')}")
                return True
            else:
                print(f"❌ Failed to pull model")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Error: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="LLM Management Tool")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List models command
    subparsers.add_parser("list", help="List available models")
    
    # Switch model command
    switch_parser = subparsers.add_parser("switch", help="Switch to a different model")
    switch_parser.add_argument("model", help="Model key to switch to")
    
    # Status command
    subparsers.add_parser("status", help="Show current LLM status")
    
    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull a model in Ollama")
    pull_parser.add_argument("model", help="Model name to pull")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    manager = LLMManager(args.url)
    
    if args.command == "list":
        print("📋 Available LLM Models:")
        print("=" * 50)
        
        models_data = manager.list_models()
        if models_data:
            current = models_data.get('current_model', 'Unknown')
            print(f"Current Model: {current}")
            print()
            
            for key, info in models_data.get('available_models', {}).items():
                status = "🔥 ACTIVE" if info.get('current') else "   "
                print(f"{status} {key:15} - {info.get('description', 'No description')}")
                print(f"     Model: {info.get('name', 'Unknown')}")
                print()
    
    elif args.command == "switch":
        print(f"🔄 Switching to model: {args.model}")
        success = manager.switch_model(args.model)
        if not success:
            sys.exit(1)
    
    elif args.command == "status":
        print("📊 LLM Status:")
        print("=" * 30)
        
        status = manager.get_status()
        if status:
            print(f"LLM Manager: {'✅' if status.get('llm_manager') else '❌'}")
            print(f"Ollama: {'✅' if status.get('ollama') else '❌'}")
            print(f"Ollama Available: {'✅' if status.get('ollama_available') else '❌'}")
            print(f"Current Model: {status.get('current_model', 'Unknown')}")
            
            config = status.get('model_config', {})
            if config:
                print(f"Temperature: {config.get('temperature', 'Unknown')}")
                print(f"Max Tokens: {config.get('max_tokens', 'Unknown')}")
                print(f"Context Length: {config.get('context_length', 'Unknown')}")
    
    elif args.command == "pull":
        print(f"📥 Pulling model: {args.model}")
        success = manager.pull_model(args.model)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()