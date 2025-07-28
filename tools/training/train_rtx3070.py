#!/usr/bin/env python3
"""
RTX 3070 Model Training Script
Optimized for 8GB VRAM gaming GPU
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os
import gc
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class RTX3070Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            self.gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU: {self.gpu_name}")
            print(f"GPU Memory: {self.gpu_memory}GB")
        else:
            print("No GPU detected!")
            return
        
        # RTX 3070 optimized configuration
        self.config = {
            "model_name": "microsoft/DialoGPT-small",  # 117M parameters
            "max_length": 256,                          # Short sequences
            "batch_size": 1,                           # Very small batch
            "use_lora": True,                          # Essential for 8GB
            "use_4bit": True,                          # Essential for 8GB
            "gradient_checkpointing": True,            # Essential for 8GB
            "lora_r": 4,                              # Very low rank
            "gradient_accumulation_steps": 32,         # Simulate larger batch
            "expected_time": "3-6 hours"
        }
        
        print("RTX 3070 Optimized Configuration:")
        print(f"  Model: {self.config['model_name']}")
        print(f"  Max Length: {self.config['max_length']}")
        print(f"  Batch Size: {self.config['batch_size']}")
        print(f"  LoRA Rank: {self.config['lora_r']}")
        print(f"  Expected Time: {self.config['expected_time']}")
        print("\nMemory Optimizations:")
        print("  ✓ 4-bit quantization")
        print("  ✓ LoRA (trains only 0.5% of parameters)")
        print("  ✓ Gradient checkpointing")
        print("  ✓ Small batch with accumulation")
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_optimized_model(self):
        """Load model with maximum memory optimization"""
        print("Loading model with maximum memory optimization...")
        
        # 4-bit quantization config (essential for 8GB)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Enable gradient checkpointing (saves memory)
        model.gradient_checkpointing_enable()
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA with minimal rank (essential for 8GB)
        print(f"Applying LoRA with rank {self.config['lora_r']}")
        lora_config = LoraConfig(
            r=self.config["lora_r"],          # Very low rank for 8GB
            lora_alpha=16,                    # Lower alpha
            target_modules=["q_proj", "v_proj"],  # Fewer modules
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def prepare_dataset(self, dataset_path, tokenizer):
        """Prepare dataset with memory optimization"""
        print(f"Loading dataset from: {dataset_path}")
        
        # Load training data
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to training format and filter by length
        texts = []
        for item in data:
            if "instruction" in item and "output" in item:
                text = f"### Instruction: {item['instruction']}\n### Response: {item['output']}"
            elif "prompt" in item and "completion" in item:
                text = f"{item['prompt']}\n{item['completion']}"
            else:
                continue
            
            # Filter out very long texts to save memory
            if len(text) < 1000:  # Rough character limit
                texts.append(text)
        
        print(f"Loaded {len(texts)} training examples (filtered for length)")
        
        # Tokenize with strict length limits
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Don't pad to save memory
                max_length=self.config["max_length"],
                return_tensors="pt"
            )
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]  # Remove original text to save memory
        )
        
        return tokenized_dataset
    
    def train_municipal_model(self, dataset_path, output_dir="./arlesheim-rtx3070-model", epochs=3):
        """Train model optimized for RTX 3070"""
        
        print("=" * 60)
        print("RTX 3070 Municipal Model Training")
        print("=" * 60)
        
        # Clear memory before starting
        self.clear_memory()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load optimized model
        model = self.load_optimized_model()
        
        # Check memory usage
        print(f"GPU Memory after model loading: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB")
        
        # Prepare dataset
        dataset = self.prepare_dataset(dataset_path, tokenizer)
        
        # Training arguments optimized for RTX 3070
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            gradient_checkpointing=True,
            fp16=True,                        # Essential for memory
            dataloader_pin_memory=False,      # Save system RAM
            dataloader_num_workers=0,         # Avoid multiprocessing overhead
            remove_unused_columns=True,
            logging_steps=25,
            save_steps=1000,
            num_train_epochs=epochs,
            learning_rate=5e-5,               # Slightly higher for small model
            warmup_steps=50,
            weight_decay=0.01,
            report_to=None,
            save_total_limit=1,               # Save disk space
            prediction_loss_only=True,       # Save memory
            max_grad_norm=1.0,               # Gradient clipping
        )
        
        # Simple data collator to save memory
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,            # Efficient padding
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Monitor memory usage
        def print_memory_usage():
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")
        
        print("Starting training...")
        print_memory_usage()
        
        try:
            # Clear memory before training
            self.clear_memory()
            
            # Train with memory monitoring
            trainer.train()
            
            print("Training completed successfully!")
            print_memory_usage()
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Save LoRA adapters
            model.save_pretrained(f"{output_dir}/lora_adapters")
            
            print(f"Model saved to: {output_dir}")
            print(f"LoRA adapters saved to: {output_dir}/lora_adapters")
            
            return output_dir
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of GPU memory: {e}")
            print("\nTroubleshooting for RTX 3070:")
            print("1. Reduce max_length to 128")
            print("2. Reduce LoRA rank to 2")
            print("3. Use even smaller model (GPT-2)")
            print("4. Reduce gradient_accumulation_steps")
            print("5. Consider cloud training instead")
            return None
        except Exception as e:
            print(f"Training failed: {e}")
            return None
    
    def test_model(self, model_path):
        """Test the trained model"""
        print("Testing trained model...")
        
        # Clear memory
        self.clear_memory()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load base model with quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Load LoRA adapters
            model = get_peft_model(base_model, LoraConfig.from_pretrained(f"{model_path}/lora_adapters"))
            
            # Test queries
            test_queries = [
                "Was sind die Öffnungszeiten?",
                "Wie kann ich einen Antrag stellen?",
                "Wo finde ich Informationen?",
                "Kontakt zur Verwaltung?"
            ]
            
            print("\nTesting with sample queries:")
            for query in test_queries:
                prompt = f"### Instruction: {query}\n### Response:"
                inputs = tokenizer(prompt, return_tensors="pt", max_length=200, truncation=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 50,  # Short responses
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(prompt, "").strip()
                
                print(f"\nQuery: {query}")
                print(f"Response: {response[:150]}...")
                
        except Exception as e:
            print(f"Testing failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train municipal model on RTX 3070")
    parser.add_argument("--dataset", default="training_data/arlesheim/arlesheim_training.json",
                       help="Path to training dataset")
    parser.add_argument("--output", default="./arlesheim-rtx3070-model",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--test", action="store_true",
                       help="Test the model after training")
    
    args = parser.parse_args()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("No GPU detected! Training will be very slow.")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    if "3070" not in gpu_name and "RTX 30" not in gpu_name:
        print(f"Detected GPU: {gpu_name}")
        print("This script is optimized for RTX 3070 (8GB). Your GPU may have different limits.")
    
    # Check if training data exists
    if not os.path.exists(args.dataset):
        print(f"Training data not found: {args.dataset}")
        print("Please run: python train_arlesheim_model.py first")
        return
    
    # Initialize trainer
    trainer = RTX3070Trainer()
    
    # Train model
    model_path = trainer.train_municipal_model(
        args.dataset, 
        args.output, 
        args.epochs
    )
    
    # Test model if requested
    if args.test and model_path:
        trainer.test_model(model_path)

if __name__ == "__main__":
    main()