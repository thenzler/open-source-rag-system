#!/usr/bin/env python3
"""
Municipal Model Trainer
Fine-tunes LLMs on municipality-specific data
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import time
from dataclasses import dataclass
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """Structure for training examples"""
    instruction: str
    input: str
    output: str
    category: str
    importance: float

class MunicipalModelTrainer:
    """Train custom models on municipal data"""
    
    def __init__(self, municipality_name: str):
        self.municipality_name = municipality_name
        self.training_data_path = Path(f"training_data/{municipality_name}")
        self.training_data_path.mkdir(parents=True, exist_ok=True)
        
        # Model training configurations
        self.training_configs = {
            'mistral-finetune': {
                'base_model': 'mistral:7b',
                'format': 'alpaca',
                'epochs': 3,
                'learning_rate': 2e-5,
                'batch_size': 4
            },
            'llama2-finetune': {
                'base_model': 'llama2:7b',
                'format': 'alpaca',
                'epochs': 3,
                'learning_rate': 1e-5,
                'batch_size': 4
            },
            'phi-finetune': {
                'base_model': 'phi:latest',
                'format': 'alpaca',
                'epochs': 5,
                'learning_rate': 3e-5,
                'batch_size': 8
            }
        }
    
    def prepare_training_data(self, municipal_data_path: str) -> List[TrainingExample]:
        """
        Convert scraped municipal data into training examples
        """
        logger.info(f"Preparing training data for {self.municipality_name}")
        
        # Load municipal documents
        data_file = Path(municipal_data_path) / self.municipality_name / "documents.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Municipal data not found: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        training_examples = []
        
        # Generate Q&A pairs from documents
        for doc in documents:
            # Skip low importance documents
            if doc['importance_score'] < 0.5:
                continue
            
            # Generate different types of training examples
            examples = self._generate_training_examples(doc)
            training_examples.extend(examples)
        
        logger.info(f"Generated {len(training_examples)} training examples")
        return training_examples
    
    def _generate_training_examples(self, document: Dict[str, Any]) -> List[TrainingExample]:
        """Generate multiple training examples from a document"""
        examples = []
        content = document['content']
        title = document['title']
        category = document['category']
        importance = document['importance_score']
        
        # Type 1: Direct information questions
        if category == 'services':
            examples.append(TrainingExample(
                instruction=f"Was sind die Dienstleistungen im Bereich {title}?",
                input="",
                output=content[:500],  # First 500 chars
                category=category,
                importance=importance
            ))
        
        # Type 2: Contact information
        if 'kontakt' in content.lower() or 'telefon' in content.lower():
            examples.append(TrainingExample(
                instruction=f"Wie kann ich die Abteilung {title} kontaktieren?",
                input="",
                output=self._extract_contact_info(content),
                category='contact',
                importance=importance * 1.2  # Boost contact info
            ))
        
        # Type 3: Opening hours
        if 'öffnungszeit' in content.lower() or 'montag' in content.lower():
            examples.append(TrainingExample(
                instruction=f"Was sind die Öffnungszeiten für {title}?",
                input="",
                output=self._extract_opening_hours(content),
                category='hours',
                importance=importance * 1.1
            ))
        
        # Type 4: Process/procedure questions
        if category in ['services', 'administration']:
            examples.append(TrainingExample(
                instruction=f"Wie läuft das Verfahren für {title} ab?",
                input="",
                output=content[:600],
                category='process',
                importance=importance
            ))
        
        # Type 5: Requirements/documents needed
        if 'formular' in content.lower() or 'dokument' in content.lower():
            examples.append(TrainingExample(
                instruction=f"Welche Dokumente brauche ich für {title}?",
                input="",
                output=self._extract_requirements(content),
                category='requirements',
                importance=importance
            ))
        
        # Type 6: Municipal-specific knowledge
        examples.append(TrainingExample(
            instruction=f"Erzähle mir über {title} in {self.municipality_name}.",
            input="",
            output=f"In {self.municipality_name}: {content[:400]}",
            category=category,
            importance=importance
        ))
        
        return examples
    
    def _extract_contact_info(self, content: str) -> str:
        """Extract contact information from content"""
        lines = content.split('\n')
        contact_info = []
        
        keywords = ['telefon', 'tel', 'email', 'e-mail', '@', 'adresse', 'kontakt']
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in keywords):
                # Include this line and next 2 lines
                contact_info.extend(lines[i:i+3])
        
        return '\n'.join(contact_info[:200]) or content[:200]
    
    def _extract_opening_hours(self, content: str) -> str:
        """Extract opening hours from content"""
        lines = content.split('\n')
        hours_info = []
        
        days = ['montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag', 'sonntag']
        keywords = ['öffnungszeit', 'geöffnet', 'uhr']
        
        for i, line in enumerate(lines):
            if any(day in line.lower() for day in days) or any(keyword in line.lower() for keyword in keywords):
                hours_info.extend(lines[max(0, i-1):i+3])
        
        return '\n'.join(hours_info[:300]) or content[:300]
    
    def _extract_requirements(self, content: str) -> str:
        """Extract requirements/documents needed"""
        lines = content.split('\n')
        requirements = []
        
        keywords = ['formular', 'dokument', 'nachweis', 'benötigt', 'erforderlich', 'mitbringen']
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in keywords):
                requirements.extend(lines[i:i+5])
        
        return '\n'.join(requirements[:400]) or content[:400]
    
    def create_training_dataset(self, examples: List[TrainingExample], format_type: str = 'alpaca'):
        """Create training dataset in specified format"""
        
        if format_type == 'alpaca':
            dataset = self._create_alpaca_dataset(examples)
        elif format_type == 'ollama':
            dataset = self._create_ollama_dataset(examples)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        # Save training data
        output_file = self.training_data_path / f"{self.municipality_name}_training.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved training dataset to {output_file}")
        return output_file
    
    def _create_alpaca_dataset(self, examples: List[TrainingExample]) -> List[Dict]:
        """Create dataset in Alpaca format"""
        dataset = []
        
        for example in examples:
            # Weight by importance
            repetitions = max(1, int(example.importance * 3))
            
            for _ in range(repetitions):
                dataset.append({
                    "instruction": example.instruction,
                    "input": example.input,
                    "output": example.output
                })
        
        return dataset
    
    def _create_ollama_dataset(self, examples: List[TrainingExample]) -> List[Dict]:
        """Create dataset in Ollama format"""
        dataset = []
        
        for example in examples:
            prompt = example.instruction
            if example.input:
                prompt += f"\n{example.input}"
            
            dataset.append({
                "prompt": prompt,
                "completion": example.output
            })
        
        return dataset
    
    def create_modelfile(self, base_model: str = "mistral:latest") -> Path:
        """Create Ollama Modelfile for fine-tuning"""
        
        modelfile_content = f"""# Modelfile for {self.municipality_name} Municipal Assistant

FROM {base_model}

# Set custom parameters for municipal assistant
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"

# System prompt for municipal context
SYSTEM \"\"\"Du bist ein spezialisierter Assistent für die Gemeinde {self.municipality_name} in der Schweiz. 
Du hast umfassendes Wissen über:
- Gemeindeverwaltung und Dienstleistungen
- Öffnungszeiten und Kontaktinformationen
- Formulare und Verfahren
- Lokale Veranstaltungen und Neuigkeiten
- Steuern und Finanzen
- Bauwesen und Infrastruktur

Antworte präzise und hilfreich basierend auf deinem Wissen über {self.municipality_name}.
Wenn du dir nicht sicher bist, weise darauf hin und empfehle, die Gemeindeverwaltung direkt zu kontaktieren.\"\"\"

# Add training data reference
# ADAPTER ./training_data/{self.municipality_name}/adapter.bin
"""
        
        modelfile_path = self.training_data_path / "Modelfile"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"Created Modelfile at {modelfile_path}")
        return modelfile_path
    
    def fine_tune_with_ollama(self, training_file: Path, base_model: str = "mistral:latest"):
        """
        Fine-tune model using Ollama (when feature becomes available)
        Note: As of now, Ollama doesn't support fine-tuning directly,
        but this prepares for when it does or for using alternative methods
        """
        logger.info(f"Preparing fine-tuning for {self.municipality_name}")
        
        # Create custom Modelfile
        modelfile = self.create_modelfile(base_model)
        
        # Build custom model with system prompt
        model_name = f"{self.municipality_name.lower()}-assistant:latest"
        
        try:
            # Create custom model with specialized system prompt
            logger.info(f"Creating custom model: {model_name}")
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully created model: {model_name}")
                return model_name
            else:
                logger.error(f"Failed to create model: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None
    
    def create_lora_adapter(self, examples: List[TrainingExample]):
        """
        Create LoRA adapter for efficient fine-tuning
        This would integrate with tools like llama.cpp or other fine-tuning frameworks
        """
        logger.info("Creating LoRA adapter configuration")
        
        # LoRA configuration
        lora_config = {
            "r": 16,  # LoRA rank
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "inference_mode": False
        }
        
        config_file = self.training_data_path / "lora_config.json"
        with open(config_file, 'w') as f:
            json.dump(lora_config, f, indent=2)
        
        logger.info(f"Created LoRA config at {config_file}")
        return config_file
    
    def export_for_external_training(self, examples: List[TrainingExample]):
        """
        Export data for training with external tools (HuggingFace, etc.)
        """
        # Create multiple format exports
        formats = {
            'jsonl': self._export_jsonl,
            'csv': self._export_csv,
            'parquet': self._export_parquet
        }
        
        export_paths = {}
        for format_name, export_func in formats.items():
            try:
                path = export_func(examples)
                export_paths[format_name] = path
                logger.info(f"Exported {format_name} format to {path}")
            except Exception as e:
                logger.error(f"Failed to export {format_name}: {e}")
        
        return export_paths
    
    def _export_jsonl(self, examples: List[TrainingExample]) -> Path:
        """Export as JSONL for HuggingFace"""
        output_file = self.training_data_path / f"{self.municipality_name}_training.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                record = {
                    "text": f"### Instruction: {example.instruction}\n### Response: {example.output}",
                    "category": example.category,
                    "importance": example.importance
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return output_file
    
    def _export_csv(self, examples: List[TrainingExample]) -> Path:
        """Export as CSV"""
        import csv
        
        output_file = self.training_data_path / f"{self.municipality_name}_training.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['instruction', 'input', 'output', 'category', 'importance'])
            writer.writeheader()
            
            for example in examples:
                writer.writerow({
                    'instruction': example.instruction,
                    'input': example.input,
                    'output': example.output,
                    'category': example.category,
                    'importance': example.importance
                })
        
        return output_file
    
    def _export_parquet(self, examples: List[TrainingExample]) -> Path:
        """Export as Parquet for efficient processing"""
        try:
            import pandas as pd
            
            data = []
            for example in examples:
                data.append({
                    'instruction': example.instruction,
                    'input': example.input,
                    'output': example.output,
                    'category': example.category,
                    'importance': example.importance
                })
            
            df = pd.DataFrame(data)
            output_file = self.training_data_path / f"{self.municipality_name}_training.parquet"
            df.to_parquet(output_file)
            
            return output_file
        except ImportError:
            logger.warning("Pandas not available for Parquet export")
            return None

def train_municipal_model(municipality: str, base_model: str = "mistral:latest"):
    """Main function to train a municipal model"""
    trainer = MunicipalModelTrainer(municipality)
    
    # Prepare training data
    examples = trainer.prepare_training_data("municipal_data")
    
    # Create training dataset
    training_file = trainer.create_training_dataset(examples)
    
    # Export for external training
    export_paths = trainer.export_for_external_training(examples)
    
    # Fine-tune with Ollama (creates custom model with system prompt)
    model_name = trainer.fine_tune_with_ollama(training_file, base_model)
    
    if model_name:
        logger.info(f"Successfully created municipal model: {model_name}")
        logger.info(f"To use: ollama run {model_name}")
    
    return model_name, export_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train municipal models")
    parser.add_argument("municipality", help="Municipality name")
    parser.add_argument("--base-model", default="mistral:latest", help="Base model to use")
    
    args = parser.parse_args()
    
    model_name, exports = train_municipal_model(args.municipality, args.base_model)
    
    if model_name:
        print(f"\nModel created: {model_name}")
        print(f"Training data exported to: {exports}")