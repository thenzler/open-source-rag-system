#!/usr/bin/env python3
"""
Model Switcher - Easily switch between fast and quality models
"""
import yaml
import sys
from pathlib import Path

def switch_model(model_name):
    """Switch to specified model"""
    config_path = Path(__file__).parent / "config" / "llm_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    old_model = config.get('default_model')
    config['default_model'] = model_name
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Switched model: {old_model} → {model_name}")
    print("Restart system: python run_core.py")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("🚀 Available RAG-Optimized Models (2024):")
        print("=" * 50)
        print("  qwen2.5        - 🏆 BEST for German RAG (20-40s)")
        print("  mistral-small  - ⚡ Ultra-fast RAG expert (15-30s)")
        print("  deepseek-r1    - 🧠 Advanced reasoning (30-50s)")
        print("  phi3-mini-fast - 💻 Laptop-friendly (30-60s)")
        print("  tinyllama      - 🚨 Emergency fallback (10-30s)")
        print("  arlesheim-german - 🏛️ Municipality-specific (60s+)")
        print()
        print("💡 Recommended: qwen2.5 (best balance of speed & quality)")
        print()
        print("Usage: python switch_model.py <model_name>")
        print("Example: python switch_model.py qwen2.5")
    else:
        switch_model(sys.argv[1])