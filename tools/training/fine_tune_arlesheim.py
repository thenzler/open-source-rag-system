#!/usr/bin/env python3
"""
Fine-tune Arlesheim Model with German/Swiss German Data
Real fine-tuning using transformers library with LoRA
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List
import logging
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import configuration
try:
    from config.config import config
    CONFIG_AVAILABLE = True
except ImportError:
    config = None
    CONFIG_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArlesheimFineTuner:
    """Fine-tune model for Arlesheim municipal assistant"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.output_dir = "models/arlesheim_finetuned"
        self.training_data_path = "training_data/arlesheim_german/arlesheim_german_training.jsonl"
        
        # Check if required packages are available
        self.check_dependencies()
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        required_packages = [
            'transformers', 'torch', 'datasets', 'peft', 'accelerate', 'bitsandbytes'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing packages for fine-tuning: {missing_packages}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        return True
    
    def prepare_training_data(self) -> List[Dict]:
        """Load and prepare training data"""
        logger.info(f"Loading training data from {self.training_data_path}")
        
        training_data = []
        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                training_data.append(data)
        
        logger.info(f"Loaded {len(training_data)} training examples")
        return training_data
    
    def format_for_training(self, data: List[Dict]) -> List[str]:
        """Format data for instruction tuning"""
        formatted_texts = []
        
        for item in data:
            messages = item['messages']
            
            # Create conversation format
            conversation = ""
            for msg in messages:
                if msg['role'] == 'system':
                    conversation += f"<|system|>\n{msg['content']}\n"
                elif msg['role'] == 'user':
                    conversation += f"<|user|>\n{msg['content']}\n"
                elif msg['role'] == 'assistant':
                    conversation += f"<|assistant|>\n{msg['content']}\n"
            
            conversation += "<|endoftext|>"
            formatted_texts.append(conversation)
        
        return formatted_texts
    
    def create_lora_config(self):
        """Create LoRA configuration for efficient fine-tuning"""
        try:
            from peft import LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,  # Rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj']
            )
            
            return lora_config
        except ImportError:
            logger.warning("PEFT not available, using regular fine-tuning")
            return None
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        try:
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM, 
                TrainingArguments, Trainer, DataCollatorForLanguageModeling
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Apply LoRA if available
            lora_config = self.create_lora_config()
            if lora_config:
                from peft import get_peft_model
                model = get_peft_model(model, lora_config)
                logger.info("Applied LoRA configuration")
            
            return model, tokenizer
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            return None, None
    
    def tokenize_data(self, texts: List[str], tokenizer):
        """Tokenize training data"""
        try:
            from datasets import Dataset
            
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
            
            dataset = Dataset.from_dict({"text": texts})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            return tokenized_dataset
            
        except ImportError:
            logger.error("Datasets library not available")
            return None
    
    def fine_tune_model(self):
        """Execute fine-tuning process"""
        logger.info("Starting fine-tuning process...")
        
        # Prepare data
        training_data = self.prepare_training_data()
        formatted_texts = self.format_for_training(training_data)
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()
        if model is None:
            logger.error("Failed to setup model and tokenizer")
            return None
        
        # Tokenize data
        tokenized_dataset = self.tokenize_data(formatted_texts, tokenizer)
        if tokenized_dataset is None:
            logger.error("Failed to tokenize data")
            return None
        
        # Setup training arguments
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                max_steps=100,  # Small for testing
                logging_dir='./logs',
                logging_steps=10,
                save_steps=50,
                eval_steps=50,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                report_to=None,  # Disable wandb
                dataloader_drop_last=False,
                fp16=torch.cuda.is_available(),
                learning_rate=5e-5,
                weight_decay=0.01,
                adam_epsilon=1e-8,
                max_grad_norm=1.0,
                lr_scheduler_type="cosine"
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                return_tensors="pt",
                pad_to_multiple_of=8
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
            logger.info("Starting training...")
            trainer.train()
            
            # Save model
            logger.info(f"Saving model to {self.output_dir}")
            trainer.save_model()
            tokenizer.save_pretrained(self.output_dir)
            
            logger.info("Fine-tuning completed successfully!")
            return self.output_dir
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return None
    
    def create_ollama_model(self, finetuned_model_path: str):
        """Create Ollama model from fine-tuned model"""
        try:
            # Create Modelfile for Ollama
            modelfile_content = f"""FROM {self.model_name}

# Fine-tuned parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# System prompt for Arlesheim
SYSTEM \"\"\"Du bist ein spezialisierter Assistent für die Gemeinde Arlesheim in der Schweiz. 
Du verstehst sowohl Hochdeutsch als auch Schweizerdeutsch und antwortest in der passenden Sprache.
Du kennst alle Dienstleistungen, Öffnungszeiten und Verfahren der Gemeinde Arlesheim.

Wichtige Informationen:
- Öffnungszeiten: Montag-Freitag 08:00-12:00 und 14:00-17:00, Donnerstag bis 18:00
- Telefon: 061 705 20 20
- Email: info@arlesheim.ch
- Website: www.arlesheim.ch

Antworte hilfreich und präzise auf Fragen zu:
- Gemeindeverwaltung und Dienstleistungen
- Baugesuche und Baubewilligungen
- Zivilstand (Heirat, Geburt, Tod)
- Steuern und Abgaben
- Schulen und Bildung
- Kehrichtabfuhr und Entsorgung
- Veranstaltungen und Kultur
\"\"\"

# Add adapter if available
# ADAPTER {finetuned_model_path}/adapter_model.bin
"""
            
            # Save Modelfile
            modelfile_path = "training_data/arlesheim_german/Modelfile_finetuned"
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            logger.info(f"Created Modelfile at {modelfile_path}")
            
            # Create Ollama model
            import subprocess
            # Get Ollama executable path
            if CONFIG_AVAILABLE and config:
                ollama_exe = config.OLLAMA_EXECUTABLE
            else:
                ollama_exe = "C:/Users/THE/AppData/Local/Programs/Ollama/ollama.exe"
                
            ollama_cmd = [
                ollama_exe,
                "create",
                "arlesheim-finetuned:latest",
                "-f",
                modelfile_path
            ]
            
            result = subprocess.run(ollama_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Successfully created Ollama model: arlesheim-finetuned:latest")
                return "arlesheim-finetuned:latest"
            else:
                logger.error(f"Failed to create Ollama model: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating Ollama model: {e}")
            return None

def install_requirements():
    """Install required packages for fine-tuning"""
    packages = [
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "datasets>=2.10.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0"
    ]
    
    print("Installing required packages for fine-tuning...")
    import subprocess
    
    for package in packages:
        try:
            subprocess.check_call([
                "pip", "install", package, "--quiet"
            ])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\nPackage installation completed.")

def main():
    """Main fine-tuning process"""
    print("=== ARLESHEIM MODEL FINE-TUNING ===")
    print("German/Swiss German Municipal Assistant")
    print("=====================================")
    
    # Check if we want to install packages
    response = input("\nInstall required packages? (y/n): ").lower()
    if response == 'y':
        install_requirements()
    
    # Initialize fine-tuner
    fine_tuner = ArlesheimFineTuner()
    
    # Check dependencies
    if not fine_tuner.check_dependencies():
        print("\nMissing dependencies. Please install required packages first.")
        return
    
    # Start fine-tuning
    print(f"\nStarting fine-tuning with {fine_tuner.training_data_path}")
    result = fine_tuner.fine_tune_model()
    
    if result:
        print(f"\n✓ Fine-tuning completed successfully!")
        print(f"Model saved to: {result}")
        
        # Create Ollama model
        ollama_model = fine_tuner.create_ollama_model(result)
        if ollama_model:
            print(f"✓ Ollama model created: {ollama_model}")
            print(f"\nTo use the model:")
            print(f"  ollama run {ollama_model}")
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Your specialized Arlesheim model is ready!")
    else:
        print("\n✗ Fine-tuning failed. Check the logs for details.")

if __name__ == "__main__":
    main()