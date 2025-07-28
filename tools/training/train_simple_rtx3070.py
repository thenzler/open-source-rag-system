#!/usr/bin/env python3
"""
Simple RTX 3070 Fine-tuning for Arlesheim
Optimized for RTX 3070 with 8GB VRAM
"""

import json
import os
import torch
from pathlib import Path

def check_gpu():
    """Check GPU availability and memory"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("No GPU detected - using CPU")
        return False

def install_unsloth():
    """Install Unsloth for fast fine-tuning"""
    import subprocess
    
    try:
        # Check if unsloth is installed
        import unsloth
        print("✓ Unsloth is already installed")
        return True
    except ImportError:
        print("Installing Unsloth for fast fine-tuning...")
        
        commands = [
            "pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git --quiet",
            "pip install --no-deps trl peft accelerate bitsandbytes --quiet"
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"✓ {cmd.split()[2] if len(cmd.split()) > 2 else 'Package'} installed")
            except subprocess.CalledProcessError:
                print(f"✗ Failed: {cmd}")
                return False
        
        return True

def load_training_data():
    """Load German/Swiss German training data"""
    training_file = "training_data/arlesheim_german/arlesheim_german_training.jsonl"
    
    if not os.path.exists(training_file):
        print(f"Training data not found: {training_file}")
        print("Please run: python create_german_training_data.py first")
        return None
    
    training_data = []
    with open(training_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            training_data.append(data)
    
    print(f"Loaded {len(training_data)} training examples")
    return training_data

def format_for_unsloth(data):
    """Format data for Unsloth training"""
    formatted_data = []
    
    for item in data:
        messages = item['messages']
        
        # Find system, user, and assistant messages
        system_msg = ""
        user_msg = ""
        assistant_msg = ""
        
        for msg in messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            elif msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'assistant':
                assistant_msg = msg['content']
        
        # Create instruction format
        instruction = f"{system_msg}\n\nUser: {user_msg}\nAssistant:"
        
        formatted_data.append({
            "instruction": instruction,
            "input": "",
            "output": assistant_msg
        })
    
    return formatted_data

def fine_tune_with_unsloth():
    """Fine-tune using Unsloth for RTX 3070"""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import torch
        
        # Load training data
        training_data = load_training_data()
        if not training_data:
            return None
        
        # Format data
        formatted_data = format_for_unsloth(training_data)
        
        # Model configuration for RTX 3070
        max_seq_length = 512  # Smaller for RTX 3070
        dtype = torch.float16  # Use half precision
        load_in_4bit = True  # Use 4-bit quantization
        
        # Load model
        print("Loading model for RTX 3070...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="microsoft/DialoGPT-medium",  # Smaller model
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
        )
        
        # Training arguments for RTX 3070
        training_args = TrainingArguments(
            per_device_train_batch_size=2,  # Small batch size
            gradient_accumulation_steps=8,  # Accumulate gradients
            warmup_steps=10,
            max_steps=100,  # Quick training
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            output_dir="models/arlesheim_unsloth",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=formatted_data,
            dataset_text_field="instruction",
            max_seq_length=max_seq_length,
            args=training_args,
            packing=False,
        )
        
        # Start training
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save model
        print("Saving model...")
        model.save_pretrained("models/arlesheim_unsloth")
        tokenizer.save_pretrained("models/arlesheim_unsloth")
        
        print("✓ Fine-tuning completed!")
        return "models/arlesheim_unsloth"
        
    except ImportError:
        print("Unsloth not available. Using alternative approach...")
        return fine_tune_simple()
    except Exception as e:
        print(f"Error in fine-tuning: {e}")
        return None

def fine_tune_simple():
    """Simple fine-tuning without Unsloth"""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, 
            TrainingArguments, Trainer, DataCollatorForLanguageModeling
        )
        from datasets import Dataset
        import torch
        
        # Load training data
        training_data = load_training_data()
        if not training_data:
            return None
        
        # Format data
        formatted_data = format_for_unsloth(training_data)
        
        # Load model and tokenizer
        model_name = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations for RTX 3070
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Tokenize data
        def tokenize_function(examples):
            texts = [f"{ex['instruction']}\n{ex['output']}" for ex in examples]
            return tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments for RTX 3070
        training_args = TrainingArguments(
            output_dir="models/arlesheim_simple",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            max_steps=100,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=10,
            save_steps=50,
            dataloader_num_workers=0
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        # Start training
        print("Starting simple fine-tuning...")
        trainer.train()
        
        # Save model
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained("models/arlesheim_simple")
        
        print("✓ Simple fine-tuning completed!")
        return "models/arlesheim_simple"
        
    except Exception as e:
        print(f"Error in simple fine-tuning: {e}")
        return None

def main():
    """Main training process optimized for RTX 3070"""
    print("=== ARLESHEIM RTX 3070 TRAINING ===")
    print("German/Swiss German Municipal Assistant")
    print("Optimized for RTX 3070 8GB VRAM")
    print("===================================")
    
    # Check GPU
    has_gpu = check_gpu()
    
    if not has_gpu:
        print("Warning: No GPU detected. Training will be slow.")
        proceed = input("Continue with CPU training? (y/n): ").lower()
        if proceed != 'y':
            return
    
    # Install Unsloth if requested
    use_unsloth = input("\nUse Unsloth for faster training? (y/n): ").lower()
    if use_unsloth == 'y':
        if not install_unsloth():
            print("Failed to install Unsloth. Using simple approach.")
            use_unsloth = False
    
    # Start training
    if use_unsloth:
        print("\nStarting Unsloth fine-tuning...")
        result = fine_tune_with_unsloth()
    else:
        print("\nStarting simple fine-tuning...")
        result = fine_tune_simple()
    
    if result:
        print(f"\n✓ Training completed successfully!")
        print(f"Model saved to: {result}")
        print(f"\nYour Arlesheim model is ready!")
        print(f"Trained on German/Swiss German municipal data")
    else:
        print("\n✗ Training failed. Check the logs for details.")

if __name__ == "__main__":
    main()