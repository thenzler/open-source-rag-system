#!/usr/bin/env python3
"""
Demo: RAG vs Model Training
Shows the difference between RAG retrieval and actual model training
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from municipal_setup import MunicipalRagSetup
from tools.municipal_model_trainer import MunicipalModelTrainer

def demo_rag_vs_training():
    """Demonstrate the difference between RAG and model training"""
    
    print("=" * 80)
    print("RAG vs Model Training Demo - Arlesheim Municipality")
    print("=" * 80)
    
    # Part 1: RAG System (Retrieval-Augmented Generation)
    print("\n" + "=" * 50)
    print("PART 1: RAG SYSTEM (Retrieval-Augmented Generation)")
    print("=" * 50)
    
    print("\nWhat RAG does:")
    print("- Stores documents in a vector database")
    print("- Retrieves relevant documents for each query")
    print("- Sends documents + query to existing LLM")
    print("- LLM generates answer based on retrieved context")
    print("- NO model training/fine-tuning involved")
    
    print("\nSetting up RAG system...")
    rag_setup = MunicipalRagSetup('arlesheim')
    municipal_rag = rag_setup.setup_complete_system(scrape_fresh=False, max_pages=20)
    
    print("\nTesting RAG system with query:")
    test_query = "Was sind die Öffnungszeiten der Gemeindeverwaltung?"
    print(f"Query: {test_query}")
    
    start_time = time.time()
    rag_result = rag_setup.query_municipal_rag(test_query)
    rag_time = time.time() - start_time
    
    print(f"\nRAG Response ({rag_time:.2f}s):")
    print(f"Answer: {rag_result['answer'][:200]}...")
    print(f"Sources: {len(rag_result['sources'])} documents retrieved")
    print(f"Confidence: {rag_result['confidence']:.2f}")
    
    # Part 2: Model Training/Fine-tuning
    print("\n" + "=" * 50)
    print("PART 2: MODEL TRAINING/FINE-TUNING")
    print("=" * 50)
    
    print("\nWhat Model Training does:")
    print("- Creates training examples from municipal data")
    print("- Trains/fine-tunes model weights on this data")
    print("- Model learns municipal knowledge internally")
    print("- Can answer questions without document retrieval")
    print("- Model becomes specialized for municipality")
    
    print("\nPreparing training data...")
    trainer = MunicipalModelTrainer('arlesheim')
    
    # Load training examples
    try:
        training_examples = trainer.prepare_training_data("municipal_data")
        print(f"Generated {len(training_examples)} training examples")
        
        # Show example training data
        print("\nExample training data:")
        for i, example in enumerate(training_examples[:3]):
            print(f"\n{i+1}. Category: {example.category}")
            print(f"   Instruction: {example.instruction}")
            print(f"   Output: {example.output[:100]}...")
            print(f"   Importance: {example.importance:.2f}")
        
        # Create training datasets
        print("\nCreating training datasets...")
        alpaca_file = trainer.create_training_dataset(training_examples, format_type='alpaca')
        print(f"Created Alpaca format: {alpaca_file}")
        
        # Export for external training
        export_paths = trainer.export_for_external_training(training_examples)
        print(f"Exported formats: {list(export_paths.keys())}")
        
        # Create custom Ollama model
        print("\nCreating custom Ollama model...")
        model_name = trainer.fine_tune_with_ollama(alpaca_file, base_model="mistral:latest")
        
        if model_name:
            print(f"✓ Created custom model: {model_name}")
            print(f"This model has specialized knowledge about Arlesheim")
        
    except Exception as e:
        print(f"Error in training preparation: {e}")
        print("Make sure to scrape Arlesheim data first!")
    
    # Part 3: Comparison
    print("\n" + "=" * 50)
    print("PART 3: COMPARISON")
    print("=" * 50)
    
    comparison_table = """
    ┌─────────────────────┬─────────────────────┬─────────────────────┐
    │      Aspect         │      RAG System     │   Model Training    │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Knowledge Storage   │ External vectors    │ Internal weights    │
    │ Query Processing    │ Retrieve + Generate │ Direct generation   │
    │ Setup Time          │ Minutes             │ Hours/Days          │
    │ Customization       │ Document-based      │ Weight-based        │
    │ Updates             │ Add/remove docs     │ Retrain model       │
    │ Computational Cost  │ Low                 │ High                │
    │ Response Quality    │ Document-dependent  │ Model-dependent     │
    │ Scalability         │ High                │ Medium              │
    └─────────────────────┴─────────────────────┴─────────────────────┘
    """
    print(comparison_table)
    
    print("\n" + "=" * 50)
    print("WHEN TO USE EACH APPROACH")
    print("=" * 50)
    
    print("\nUse RAG when:")
    print("- You need quick setup and deployment")
    print("- Documents change frequently")
    print("- You want to cite specific sources")
    print("- You have limited computational resources")
    print("- You need to update knowledge easily")
    
    print("\nUse Model Training when:")
    print("- You want specialized model behavior")
    print("- You need consistent response style")
    print("- You want to embed domain expertise")
    print("- You have training resources available")
    print("- You need the model to 'understand' the domain")
    
    print("\n" + "=" * 50)
    print("HYBRID APPROACH (RECOMMENDED)")
    print("=" * 50)
    
    print("\nFor Arlesheim municipality:")
    print("1. Start with RAG system (fast setup, immediate results)")
    print("2. Collect user queries and feedback")
    print("3. Create training data from successful RAG responses")
    print("4. Fine-tune model on municipal-specific patterns")
    print("5. Use fine-tuned model + RAG for best results")
    
    print("\nThis gives you:")
    print("- Immediate deployment capability")
    print("- Continuous improvement through training")
    print("- Best of both worlds: speed + specialization")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    demo_rag_vs_training()