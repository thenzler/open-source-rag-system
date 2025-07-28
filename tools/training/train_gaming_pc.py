#!/usr/bin/env python3
"""
Gaming PC Model Training Script
Optimized for consumer hardware with automatic GPU detection
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
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class GamingPCTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            self.gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU: {self.gpu_name}")
            print(f"GPU Memory: {self.gpu_memory}GB")
        else:
            print("No GPU detected! This will be very slow.")
            self.gpu_memory = 0
        
        self.optimize_for_hardware()
    
    def optimize_for_hardware(self):
        """Optimize settings based on available hardware"""
        if self.gpu_memory <= 8:
            print("Optimizing for 8GB GPU (RTX 3070, RTX 4060)")
            self.config = {
                "model_name": "microsoft/DialoGPT-small",
                "max_length": 256,
                "batch_size": 2,
                "use_lora": True,
                "use_4bit": True,
                "gradient_checkpointing": True,
                "lora_r": 8,
                "expected_time": "2-4 hours"
            }
        elif self.gpu_memory <= 12:
            print("Optimizing for 12GB GPU (RTX 3080 Ti, RTX 4070)")
            self.config = {
                "model_name": "microsoft/DialoGPT-medium",
                "max_length": 512,
                "batch_size": 1,
                "use_lora": True,
                "use_4bit": False,
                "gradient_checkpointing": True,
                "lora_r": 16,
                "expected_time": "6-12 hours"
            }
        elif self.gpu_memory <= 16:
            print("Optimizing for 16GB GPU (RTX 4080, RTX 4060 Ti)")
            self.config = {
                "model_name": "microsoft/DialoGPT-medium",
                "max_length": 1024,
                "batch_size": 1,
                "use_lora": True,
                "use_4bit": True,
                "gradient_checkpointing": True,
                "lora_r": 16,
                "expected_time": "8-16 hours"
            }
        else:  # 20GB+ (RTX 4090, RTX 3090)
            print("Optimizing for 20GB+ GPU (RTX 4090, RTX 3090)")
            self.config = {
                "model_name": "mistralai/Mistral-7B-v0.1",
                "max_length": 1024,
                "batch_size": 1,
                "use_lora": True,
                "use_4bit": True,
                "gradient_checkpointing": True,
                "lora_r": 16,
                "expected_time": "12-24 hours"
            }
        
        print(f"Using model: {self.config['model_name']}")
        print(f"Expected training time: {self.config['expected_time']}")
    
    def load_optimized_model(self, model_name=None):
        """Load model with optimizations for gaming PC"""
        if model_name is None:
            model_name = self.config["model_name"]
        
        print(f"Loading model: {model_name}")
        
        if self.config["use_4bit"]:
            print("Using 4-bit quantization to save memory")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Enable gradient checkpointing to save memory
        if self.config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        
        # Apply LoRA for efficient training
        if self.config["use_lora"]:
            print(f"Applying LoRA with r={self.config['lora_r']}")
            
            if self.config["use_4bit"]:
                model = prepare_model_for_kbit_training(model)
            
            lora_config = LoraConfig(
                r=self.config["lora_r"],
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model
    
    def prepare_dataset(self, dataset_path, tokenizer):
        """Prepare dataset for training"""
        print(f"Loading dataset from: {dataset_path}")
        
        # Load training data
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to training format
        texts = []
        for item in data:
            if "instruction" in item and "output" in item:
                # Alpaca format
                text = f"### Instruction: {item['instruction']}\n### Response: {item['output']}"
            elif "prompt" in item and "completion" in item:
                # OpenAI format
                text = f"{item['prompt']}\n{item['completion']}"
            else:
                continue
            
            texts.append(text)
        
        print(f"Loaded {len(texts)} training examples")
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config["max_length"],
                return_tensors="pt"
            )
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train_municipal_model(self, dataset_path, output_dir="./arlesheim-gaming-model", epochs=3):
        """Train model optimized for gaming PC"""
        
        print("=" * 60)
        print("Gaming PC Municipal Model Training")
        print("=" * 60)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load optimized model
        model = self.load_optimized_model()
        
        # Prepare dataset
        dataset = self.prepare_dataset(dataset_path, tokenizer)
        
        # Training arguments optimized for gaming PC
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=16 // self.config["batch_size"],
            gradient_checkpointing=self.config["gradient_checkpointing"],
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            num_train_epochs=epochs,
            learning_rate=2e-5,
            warmup_steps=100,
            report_to=None,  # Disable wandb for gaming PC
            save_total_limit=2,  # Save disk space
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Monitor GPU usage
        print("Starting training...")
        print(f"GPU Memory before training: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB")
        
        try:
            trainer.train()
            print("Training completed successfully!")
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            print(f"Model saved to: {output_dir}")
            
            # Save LoRA adapters separately if used
            if self.config["use_lora"]:
                model.save_pretrained(f"{output_dir}/lora_adapters")
                print(f"LoRA adapters saved to: {output_dir}/lora_adapters")
            
            return output_dir
            
        except torch.cuda.OutOfMemoryError:
            print("Out of GPU memory! Try:")
            print("1. Reduce batch size")
            print("2. Enable 4-bit quantization")
            print("3. Use smaller model")
            print("4. Reduce max_length")
            return None
        except Exception as e:
            print(f"Training failed: {e}")
            return None
    
    def test_model(self, model_path):
        """Test the trained model"""
        print("Testing trained model...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.config["use_lora"]:
            # Load base model and LoRA adapters
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = get_peft_model(base_model, LoraConfig.from_pretrained(f"{model_path}/lora_adapters"))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Test queries
        test_queries = [
            "Was sind die Öffnungszeiten der Gemeindeverwaltung?",
            "Wie kann ich einen Bauantrag stellen?",
            "Wo finde ich Informationen über Steuern?",
            "Wie kontaktiere ich die Gemeindeverwaltung?"
        ]
        
        print("\nTesting with sample queries:")
        for query in test_queries:
            prompt = f"### Instruction: {query}\n### Response:"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            print(f"\nQuery: {query}")
            print(f"Response: {response[:200]}...")

def main():
    parser = argparse.ArgumentParser(description="Train municipal model on gaming PC")
    parser.add_argument("--dataset", default="training_data/arlesheim/arlesheim_training.json",
                       help="Path to training dataset")
    parser.add_argument("--output", default="./arlesheim-gaming-model",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--test", action="store_true",
                       help="Test the model after training")
    
    args = parser.parse_args()
    
    # Check if training data exists
    if not os.path.exists(args.dataset):
        print(f"Training data not found: {args.dataset}")
        print("Please run: python train_arlesheim_model.py first")
        return
    
    # Initialize trainer
    trainer = GamingPCTrainer()
    
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