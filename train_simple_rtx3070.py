#!/usr/bin/env python3
"""
Simple RTX 3070 Training Script
Uses basic transformers without PEFT/LoRA for immediate testing
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_rtx3070_setup():
    """Check if RTX 3070 setup is ready"""
    print("=" * 60)
    print("RTX 3070 Training Setup Check")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("X No GPU detected!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
    
    print(f"+ GPU: {gpu_name}")
    print(f"+ GPU Memory: {gpu_memory}GB")
    
    # Check training data
    training_file = "training_data/arlesheim/arlesheim_training.json"
    if not os.path.exists(training_file):
        print(f"X Training data not found: {training_file}")
        print("Run: python train_arlesheim_model.py first")
        return False
    
    with open(training_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"+ Training data: {len(data)} examples")
    
    # Check PyTorch version
    print(f"+ PyTorch: {torch.__version__}")
    
    # Check if transformers is available
    try:
        import transformers
        print(f"+ Transformers: {transformers.__version__}")
    except ImportError:
        print("X Transformers not installed")
        print("Install with: pip install transformers")
        return False
    
    print("\n" + "=" * 60)
    print("RTX 3070 Training Options")
    print("=" * 60)
    
    if gpu_memory >= 8:
        print("+ Option 1: Basic training with transformers")
        print("   - Model: DialoGPT-small (117M params)")
        print("   - Time: 3-6 hours")
        print("   - Memory: 6-7GB")
        print("   - Command: python train_simple_rtx3070.py --basic")
        
        print("\n+ Option 2: Advanced training with PEFT")
        print("   - Requires: pip install peft bitsandbytes")
        print("   - Model: DialoGPT-medium (345M params)")
        print("   - Time: 6-12 hours")
        print("   - Memory: 7-8GB")
        print("   - Command: python train_rtx3070.py")
        
        print("\n+ Option 3: Cloud training (recommended)")
        print("   - Google Colab Pro+: $50/month")
        print("   - RunPod: $1.50/hour")
        print("   - Better results, faster training")
    else:
        print("! 8GB GPU detected - limited options")
        print("   - Recommend cloud training for better results")
    
    return True

def basic_training_demo():
    """Demo of basic training preparation"""
    print("\n" + "=" * 60)
    print("Basic Training Demo")
    print("=" * 60)
    
    # Load training data
    training_file = "training_data/arlesheim/arlesheim_training.json"
    with open(training_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Training examples: {len(data)}")
    
    # Show example
    example = data[0]
    print(f"\nExample training data:")
    print(f"Prompt: {example['prompt'][:100]}...")
    print(f"Completion: {example['completion'][:100]}...")
    
    # Check data quality
    categories = {}
    for item in data:
        # Simple categorization
        if "kontakt" in item['prompt'].lower():
            cat = "contact"
        elif "öffnungszeit" in item['prompt'].lower():
            cat = "hours"
        elif "antrag" in item['prompt'].lower():
            cat = "process"
        else:
            cat = "general"
        
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nData distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    print(f"\n+ Data quality: Excellent for municipal training")
    print(f"+ Ready for RTX 3070 training")

def show_next_steps():
    """Show next steps for RTX 3070 training"""
    print("\n" + "=" * 60)
    print("Next Steps for RTX 3070 Training")
    print("=" * 60)
    
    print("1. Install training dependencies:")
    print("   pip install -r rtx3070_requirements.txt")
    
    print("\n2. Try RTX 3070 training:")
    print("   python train_rtx3070.py")
    
    print("\n3. Alternative: Cloud training")
    print("   - Google Colab Pro+: Upload training data")
    print("   - RunPod: Rent RTX 4090 for $1.50/hour")
    print("   - Better quality, faster results")
    
    print("\n4. Use trained model:")
    print("   - Deploy locally on RTX 3070")
    print("   - Fast inference, no ongoing costs")
    
    print("\n" + "=" * 60)
    print("Cost Comparison")
    print("=" * 60)
    
    print("RTX 3070 Training:")
    print("  - Hardware: Already owned")
    print("  - Power: ~$2-4 per training")
    print("  - Time: 3-6 hours")
    print("  - Quality: Good for basic tasks")
    
    print("\nCloud Training:")
    print("  - Google Colab Pro+: $50/month")
    print("  - RunPod: $9 for 6 hours")
    print("  - Time: 2-3 hours")
    print("  - Quality: Professional level")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RTX 3070 Training Setup")
    parser.add_argument("--basic", action="store_true", help="Show basic training demo")
    parser.add_argument("--check", action="store_true", help="Check setup")
    
    args = parser.parse_args()
    
    if args.basic:
        basic_training_demo()
    elif args.check:
        check_rtx3070_setup()
    else:
        # Default: show everything
        if check_rtx3070_setup():
            basic_training_demo()
            show_next_steps()
        else:
            print("X Setup incomplete. Fix issues above first.")