#!/usr/bin/env python3
"""
Train Arlesheim Model Script
Actually trains/fine-tunes a model on Arlesheim municipality data
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.municipal_web_scraper import MunicipalWebScraper
from tools.municipal_model_trainer import MunicipalModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_arlesheim_model():
    """Complete training pipeline for Arlesheim"""
    
    print("=" * 60)
    print("Arlesheim Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Scrape Arlesheim website
    print("\n1. Scraping Arlesheim website...")
    scraper = MunicipalWebScraper('arlesheim', 'https://www.arlesheim.ch')
    documents = scraper.scrape_municipality(max_pages=50)
    scraper.save_documents(documents)
    print(f"   Scraped {len(documents)} documents")
    
    # Step 2: Initialize model trainer
    print("\n2. Initializing model trainer...")
    trainer = MunicipalModelTrainer('arlesheim')
    
    # Step 3: Prepare training data
    print("\n3. Preparing training data...")
    training_examples = trainer.prepare_training_data("municipal_data")
    print(f"   Generated {len(training_examples)} training examples")
    
    # Step 4: Create training datasets
    print("\n4. Creating training datasets...")
    
    # Create Alpaca format for fine-tuning
    alpaca_file = trainer.create_training_dataset(training_examples, format_type='alpaca')
    print(f"   Created Alpaca dataset: {alpaca_file}")
    
    # Create Ollama format
    ollama_file = trainer.create_training_dataset(training_examples, format_type='ollama')
    print(f"   Created Ollama dataset: {ollama_file}")
    
    # Step 5: Export for external training tools
    print("\n5. Exporting for external training tools...")
    export_paths = trainer.export_for_external_training(training_examples)
    for format_name, path in export_paths.items():
        if path:
            print(f"   Exported {format_name}: {path}")
    
    # Step 6: Create custom Ollama model with system prompt
    print("\n6. Creating custom Ollama model...")
    model_name = trainer.fine_tune_with_ollama(alpaca_file, base_model="mistral:latest")
    
    if model_name:
        print(f"\n✓ Successfully created model: {model_name}")
        print(f"\nTo use the model:")
        print(f"   ollama run {model_name}")
    
    # Step 7: Create LoRA configuration for external fine-tuning
    print("\n7. Creating LoRA adapter configuration...")
    lora_config = trainer.create_lora_adapter(training_examples)
    print(f"   Created LoRA config: {lora_config}")
    
    # Display training statistics
    print("\n" + "=" * 60)
    print("Training Statistics")
    print("=" * 60)
    
    # Category distribution
    categories = {}
    for example in training_examples:
        categories[example.category] = categories.get(example.category, 0) + 1
    
    print("\nTraining examples by category:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   {category}: {count}")
    
    # Importance distribution
    importance_levels = {'high': 0, 'medium': 0, 'low': 0}
    for example in training_examples:
        if example.importance >= 0.9:
            importance_levels['high'] += 1
        elif example.importance >= 0.7:
            importance_levels['medium'] += 1
        else:
            importance_levels['low'] += 1
    
    print("\nImportance distribution:")
    for level, count in importance_levels.items():
        print(f"   {level}: {count}")
    
    print("\n" + "=" * 60)
    print("Next Steps for Full Fine-Tuning")
    print("=" * 60)
    print("\n1. For Ollama (when fine-tuning is supported):")
    print(f"   - Model created: {model_name}")
    print("   - Uses specialized system prompt for Arlesheim")
    print("   - Run: ollama run arlesheim-assistant:latest")
    
    print("\n2. For HuggingFace Transformers:")
    print(f"   - Training data: {export_paths.get('jsonl', 'N/A')}")
    print(f"   - LoRA config: {lora_config}")
    print("   - Use autotrain-advanced or transformers library")
    
    print("\n3. For llama.cpp/llama-cpp-python:")
    print(f"   - Training data: {export_paths.get('jsonl', 'N/A')}")
    print("   - Convert to GGUF format for quantization")
    
    print("\n4. For OpenAI-style fine-tuning APIs:")
    print(f"   - JSONL format: {export_paths.get('jsonl', 'N/A')}")
    print("   - Can be used with various fine-tuning services")
    
    return model_name, training_examples

if __name__ == "__main__":
    model_name, examples = train_arlesheim_model()