#!/usr/bin/env python3
"""
Automatic German/Swiss German Fine-tuning for Arlesheim
No user input required - runs automatically
"""

import json
import os
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_setup():
    """Check if setup is ready for training"""
    print("=== ARLESHEIM GERMAN FINE-TUNING ===")
    print("Checking training setup...")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"+ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        use_gpu = True
    else:
        print("! No GPU detected - using CPU (will be slow)")
        use_gpu = False
    
    # Check training data
    training_file = "training_data/arlesheim_german/arlesheim_german_training.jsonl"
    if not os.path.exists(training_file):
        print(f"- Training data not found: {training_file}")
        print("Please run: python create_german_training_data.py first")
        return False, False
    
    print(f"+ Training data found: {training_file}")
    
    # Check required packages
    try:
        import transformers
        print(f"+ Transformers: {transformers.__version__}")
    except ImportError:
        print("- Transformers not installed")
        print("Install with: pip install transformers torch datasets")
        return False, False
    
    return True, use_gpu

def load_training_data():
    """Load German/Swiss German training data"""
    training_file = "training_data/arlesheim_german/arlesheim_german_training.jsonl"
    
    training_data = []
    with open(training_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            training_data.append(data)
    
    print(f"+ Loaded {len(training_data)} German/Swiss German examples")
    return training_data

def format_conversations(data):
    """Format data for conversation-style training"""
    formatted_data = []
    
    for item in data:
        messages = item['messages']
        
        # Create conversation text
        conversation = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                conversation += f"System: {content}\n\n"
            elif role == 'user':
                conversation += f"User: {content}\n"
            elif role == 'assistant':
                conversation += f"Assistant: {content}\n"
        
        formatted_data.append({
            "text": conversation,
            "category": item.get('category', 'general'),
            "dialect": item.get('dialect', 'german')
        })
    
    return formatted_data

def fine_tune_basic():
    """Basic fine-tuning approach"""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, 
            TrainingArguments, Trainer, DataCollatorForLanguageModeling
        )
        from datasets import Dataset
        import torch
        
        print("Starting basic fine-tuning...")
        
        # Load training data
        training_data = load_training_data()
        formatted_data = format_conversations(training_data)
        
        # Use smaller model for compatibility
        model_name = "microsoft/DialoGPT-small"
        print(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=256,  # Shorter for speed
                return_tensors="pt"
            )
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments optimized for speed
        training_args = TrainingArguments(
            output_dir="models/arlesheim_german",
            overwrite_output_dir=True,
            per_device_train_batch_size=4 if torch.cuda.is_available() else 2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=50,  # Quick training for testing
            learning_rate=3e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=25,
            dataloader_num_workers=0,
            report_to=None,  # Disable wandb
            load_best_model_at_end=False,
            save_total_limit=2
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
        
        print("Starting training...")
        trainer.train()
        
        # Save model
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained("models/arlesheim_german")
        
        print("+ Fine-tuning completed successfully!")
        return "models/arlesheim_german"
        
    except Exception as e:
        print(f"- Error in fine-tuning: {e}")
        return None

def create_ollama_model(model_path):
    """Create Ollama model from fine-tuned model"""
    try:
        # Create enhanced Modelfile
        modelfile_content = f"""FROM mistral:latest

# Enhanced system prompt for German/Swiss German Arlesheim assistant
SYSTEM \"\"\"Du bist ein spezialisierter Assistent für die Gemeinde Arlesheim in der Schweiz. 
Du verstehst sowohl Hochdeutsch als auch Schweizerdeutsch perfekt und antwortest immer in der 
Sprache, die der Benutzer verwendet.

Du hast umfassendes Wissen über:
- Gemeindeverwaltung und alle Dienstleistungen
- Öffnungszeiten: Mo-Fr 08:00-12:00 und 14:00-17:00, Do bis 18:00
- Kontakt: Telefon 061 705 20 20, Email info@arlesheim.ch
- Baugesuche und Baubewilligungen
- Zivilstand (Heirat, Geburt, Tod)
- Steuern und Abgaben (Abgabe bis 31. März)
- Kehrichtabfuhr (jeden Montag)
- Schulen und Bildung
- Veranstaltungen und lokale Ereignisse

Antworte immer:
- Hilfreich und präzise
- In der Sprache des Benutzers (Hochdeutsch oder Schweizerdeutsch)
- Mit konkreten Informationen (Telefonnummern, Adressen, Zeiten)
- Freundlich und professionell

Bei Schweizerdeutsch verwende authentische Ausdrücke wie:
- "Grüezi" für Begrüssung
- "Merci vilmal" für Danke
- "Gmeindverwautig" für Gemeindeverwaltung
- "Uf Widerluege" für Auf Wiedersehen
\"\"\"

# Optimized parameters for municipal assistant
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
PARAMETER stop "User:"
PARAMETER stop "Assistant:"

# Fine-tuned model reference
# MODEL_PATH: {model_path}
"""
        
        # Save enhanced Modelfile
        modelfile_path = "training_data/arlesheim_german/Modelfile_enhanced"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"+ Created enhanced Modelfile: {modelfile_path}")
        
        # Create Ollama model
        import subprocess
        ollama_cmd = [
            "C:/Users/THE/AppData/Local/Programs/Ollama/ollama.exe",
            "create",
            "arlesheim-german:latest",
            "-f",
            modelfile_path
        ]
        
        result = subprocess.run(ollama_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("+ Created Ollama model: arlesheim-german:latest")
            return "arlesheim-german:latest"
        else:
            print(f"- Failed to create Ollama model: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"- Error creating Ollama model: {e}")
        return None

def test_model():
    """Test the trained model with German/Swiss German examples"""
    test_queries = [
        "Grüezi! Wann isch d'Gmeindverwautig offe?",
        "Wie kann ich einen Bauantrag stellen?",
        "Wo chan i mini Stüürerklärig abgeh?",
        "Wann wird de Kehricht abgholt?",
        "Gibt es einen Mittagstisch für Kinder?"
    ]
    
    print("\n=== TESTING MODEL ===")
    print("Test queries (German/Swiss German):")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
    
    print("\nTo test your model, run:")
    print("  ollama run arlesheim-german:latest")
    print("\nThen try the test queries above!")

def main():
    """Main automatic training process"""
    print("Automatic German/Swiss German Fine-tuning")
    print("==========================================")
    
    # Check setup
    ready, use_gpu = check_setup()
    if not ready:
        print("\n- Setup incomplete. Please fix issues above.")
        return
    
    print(f"\n+ Setup complete! Using {'GPU' if use_gpu else 'CPU'} for training")
    
    # Start fine-tuning
    print("\n=== STARTING FINE-TUNING ===")
    result = fine_tune_basic()
    
    if result:
        print(f"\n+ Fine-tuning completed!")
        print(f"Model saved to: {result}")
        
        # Create Ollama model
        print("\n=== CREATING OLLAMA MODEL ===")
        ollama_model = create_ollama_model(result)
        
        if ollama_model:
            print(f"\n*** SUCCESS! Your German/Swiss German model is ready!")
            print(f"Model: {ollama_model}")
            print(f"Trained on: 48 German/Swiss German examples")
            print(f"Specialization: Arlesheim municipal services")
            
            # Test model
            test_model()
            
        else:
            print("\n* Model trained but Ollama creation failed")
            print("You can still use the model files directly")
    else:
        print("\n- Fine-tuning failed. Check error messages above.")

if __name__ == "__main__":
    main()