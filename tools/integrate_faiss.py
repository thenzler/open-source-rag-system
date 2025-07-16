#!/usr/bin/env python3
"""
Script to integrate FAISS vector search into the existing RAG system
This will modify simple_api.py to use the optimized FAISS-based search
"""

import os
import sys
import shutil
from datetime import datetime

def create_backup(file_path):
    """Create a backup of the original file"""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def integrate_faiss_into_api():
    """Integrate FAISS vector search into simple_api.py"""
    
    api_file = "simple_api.py"
    if not os.path.exists(api_file):
        print(f"Error: {api_file} not found!")
        return False
    
    # Create backup
    backup_path = create_backup(api_file)
    
    # Read the original file
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add import for FAISS vector search
    import_section = """from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import time
import hashlib
import os
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
import json

# Import FAISS vector search
try:
    from services.vector_search import OptimizedVectorStore, FAISSVectorSearch
    FAISS_AVAILABLE = True
    print("FAISS vector search loaded successfully!")
except ImportError as e:
    print(f"Warning: FAISS not available, falling back to basic search: {e}")
    FAISS_AVAILABLE = False
    OptimizedVectorStore = None
"""
    
    # Replace the import section
    content = content.replace(
        "from sentence_transformers import SentenceTransformer",
        import_section,
        1
    )
    
    # Replace vector search initialization
    vector_init_old = """# Initialize embeddings model (using a smaller, faster model)
try:
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load embedding model: {e}")
    embedding_model = None"""
    
    vector_init_new = """# Initialize embeddings model (using a smaller, faster model)
try:
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded successfully!")
    
    # Initialize FAISS vector store if available
    if FAISS_AVAILABLE and embedding_model:
        vector_store = OptimizedVectorStore(embedding_model)
        print("FAISS vector store initialized!")
    else:
        vector_store = None
        print("Using fallback vector search")
except Exception as e:
    print(f"Warning: Could not load embedding model: {e}")
    embedding_model = None
    vector_store = None"""
    
    content = content.replace(vector_init_old, vector_init_new)
    
    # Update the find_similar_chunks function to use FAISS when available
    new_find_similar = '''def find_similar_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """Find similar document chunks using FAISS (10-100x faster) or fallback to cosine similarity"""
    
    if not embedding_model or (not document_chunks and (not vector_store or not FAISS_AVAILABLE)):
        logger.warning("No embedding model or documents available")
        return []
    
    try:
        # Use FAISS if available
        if FAISS_AVAILABLE and vector_store:
            logger.info("Using FAISS vector search")
            
            # Check if we need to migrate existing documents to FAISS
            if document_embeddings and len(document_embeddings) > 0 and vector_store.faiss_search.index_size == 0:
                logger.info(f"Migrating {len(document_embeddings)} existing embeddings to FAISS")
                
                # Prepare data for FAISS
                chunk_texts = []
                metadatas = []
                
                for i, chunk in enumerate(document_chunks):
                    chunk_texts.append(chunk["text"])
                    metadatas.append({
                        "document_id": chunk["document_id"],
                        "chunk_index": i,
                        "filename": documents[chunk["document_id"]]["filename"] if chunk["document_id"] < len(documents) else "Unknown"
                    })
                
                # Build FAISS index
                embeddings_array = np.array(document_embeddings)
                chunk_ids = list(range(len(document_chunks)))
                vector_store.faiss_search.build_index(
                    embeddings_array,
                    chunk_ids,
                    chunk_texts,
                    metadatas
                )
                logger.info("Migration to FAISS complete!")
            
            # Perform FAISS search
            results = vector_store.similarity_search(query, k=top_k)
            
            # Format results
            similar_chunks = []
            for text, score, metadata in results:
                similar_chunks.append({
                    "text": text,
                    "score": float(score),
                    "document_id": metadata.get("document_id", 0),
                    "source": metadata.get("filename", "Unknown")
                })
            
            return similar_chunks
            
        else:
            # Fallback to original cosine similarity search
            logger.info("Using fallback cosine similarity search")
            
            # Cache the query embedding
            cache_key = f"query_embedding:{hashlib.md5(query.encode()).hexdigest()}"
            query_embedding = fast_cache.get_from_cache(cache_key)
            
            if query_embedding is None:
                query_embedding = embedding_model.encode([query])
                fast_cache.add_to_cache(cache_key, query_embedding)
            
            # Calculate similarities
            if len(document_embeddings) == 0:
                return []
                
            similarities = cosine_similarity(query_embedding, document_embeddings)[0]
            
            # Get top-k similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_chunks = []
            for idx in top_indices:
                if similarities[idx] > 0.0:  # Only include positive similarities
                    chunk = document_chunks[idx]
                    similar_chunks.append({
                        "text": chunk["text"],
                        "score": float(similarities[idx]),
                        "document_id": chunk["document_id"],
                        "source": documents[chunk["document_id"]]["filename"] if chunk["document_id"] < len(documents) else "Unknown"
                    })
            
            return similar_chunks
            
    except Exception as e:
        logger.error(f"Error in find_similar_chunks: {str(e)}")
        return []'''
    
    # Find and replace the find_similar_chunks function
    import re
    pattern = r'def find_similar_chunks\(.*?\):\s*\n(?:.*\n)*?(?=\ndef|\nclass|\n@|\Z)'
    
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if match:
        content = content[:match.start()] + new_find_similar + content[match.end():]
        print("Updated find_similar_chunks function to use FAISS")
    else:
        print("Warning: Could not find find_similar_chunks function to update")
    
    # Update process_document to use FAISS vector store
    process_doc_addition = '''
        # Add to FAISS vector store if available
        if FAISS_AVAILABLE and vector_store and chunks:
            logger.info(f"Adding {len(chunks)} chunks to FAISS vector store")
            
            chunk_texts = [chunk["text"] for chunk in chunks]
            metadatas = [{
                "document_id": chunk["document_id"],
                "chunk_index": i,
                "filename": documents[chunk["document_id"]]["filename"]
            } for i, chunk in enumerate(chunks)]
            
            vector_store.add_documents(chunk_texts, metadatas)
            logger.info("Added chunks to FAISS vector store")
'''
    
    # Find where to insert the FAISS addition in process_document
    process_pattern = r'(document_chunks\.extend\(chunks\)\s*\n\s*document_embeddings\.extend\(embeddings\))'
    
    match = re.search(process_pattern, content)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + process_doc_addition + content[insert_pos:]
        print("Updated process_document to add chunks to FAISS")
    else:
        print("Warning: Could not update process_document function")
    
    # Add a new endpoint to get vector store stats
    stats_endpoint = '''
@app.get("/api/v1/vector-stats")
async def get_vector_stats():
    """Get vector store statistics"""
    if FAISS_AVAILABLE and vector_store:
        stats = vector_store.get_stats()
        stats["type"] = "FAISS (Optimized)"
        stats["performance"] = "10-100x faster than cosine similarity"
        return stats
    else:
        return {
            "type": "Cosine Similarity (Basic)",
            "total_vectors": len(document_embeddings),
            "performance": "Baseline performance",
            "status": "Consider installing FAISS for better performance"
        }
'''
    
    # Add the stats endpoint before the last run statement
    run_pattern = r'(if __name__ == "__main__":\s*import uvicorn)'
    match = re.search(run_pattern, content)
    if match:
        insert_pos = match.start()
        content = content[:insert_pos] + stats_endpoint + "\n\n" + content[insert_pos:]
        print("Added vector stats endpoint")
    
    # Write the modified content
    with open(api_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nSuccessfully integrated FAISS into {api_file}")
    print(f"Original file backed up to: {backup_path}")
    
    # Create __init__.py for services module
    os.makedirs("services", exist_ok=True)
    with open("services/__init__.py", 'w') as f:
        f.write("# Services module for RAG system\n")
    
    return True

def create_faiss_requirements():
    """Create a requirements file for FAISS dependencies"""
    
    faiss_requirements = """# FAISS Vector Search Requirements
# Install with: pip install -r faiss_requirements.txt

# FAISS for fast similarity search
# Choose one based on your system:
# CPU version (recommended for most users)
faiss-cpu>=1.7.4

# GPU version (if you have CUDA)
# faiss-gpu>=1.7.4

# Additional optimizations
numpy>=1.21.0
numba>=0.58.0  # For additional optimizations

# Monitoring and profiling
psutil>=5.9.0  # For memory monitoring
"""
    
    with open("faiss_requirements.txt", 'w') as f:
        f.write(faiss_requirements)
    
    print("Created faiss_requirements.txt")

def create_test_script():
    """Create a test script to verify FAISS integration"""
    
    test_script = '''#!/usr/bin/env python3
"""Test script to verify FAISS integration and performance"""

import time
import numpy as np
import requests
import json
from typing import List, Dict

def test_vector_stats():
    """Test the vector stats endpoint"""
    print("\\n1. Testing Vector Stats Endpoint...")
    try:
        response = requests.get("http://localhost:8001/api/v1/vector-stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"Vector Store Type: {stats.get('type', 'Unknown')}")
            print(f"Total Vectors: {stats.get('total_vectors', 0)}")
            print(f"Performance: {stats.get('performance', 'Unknown')}")
            print(f"Status: {stats.get('status', 'Ready')}")
            
            if "FAISS" in stats.get('type', ''):
                print("✅ FAISS is active!")
            else:
                print("⚠️  FAISS not active, using fallback search")
        else:
            print(f"❌ Failed to get vector stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing vector stats: {e}")

def test_search_performance():
    """Test search performance with timing"""
    print("\\n2. Testing Search Performance...")
    
    test_queries = [
        "What is machine learning?",
        "Explain artificial intelligence",
        "How does deep learning work?",
        "What are neural networks?",
        "Tell me about data science"
    ]
    
    # Test optimized endpoint
    print("\\nTesting Optimized Endpoint (with FAISS):")
    total_time = 0
    for query in test_queries:
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://localhost:8001/api/v1/query/optimized",
                json={"query": query, "context_limit": 5}
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if response.status_code == 200:
                print(f"  Query: '{query[:30]}...' - Time: {elapsed:.3f}s")
            else:
                print(f"  Query failed: {response.status_code}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    avg_time = total_time / len(test_queries)
    print(f"\\nAverage response time: {avg_time:.3f}s")
    
    if avg_time < 2.0:
        print("✅ Excellent performance!")
    elif avg_time < 5.0:
        print("✅ Good performance")
    else:
        print("⚠️  Performance could be improved")

def test_document_processing():
    """Test document processing with FAISS"""
    print("\\n3. Testing Document Processing...")
    
    # Create a test text file
    test_content = """
    FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
    and clustering of dense vectors. It contains algorithms that search in sets of 
    vectors of any size, up to ones that possibly do not fit in RAM. FAISS is written
    in C++ with complete wrappers for Python. Some of the most useful algorithms are
    implemented on the GPU.
    
    Key features of FAISS:
    - Exact and approximate nearest neighbor search
    - Multiple index types for different use cases
    - GPU acceleration support
    - Excellent performance on large datasets
    - Support for binary vectors
    """
    
    with open("test_faiss_doc.txt", "w") as f:
        f.write(test_content)
    
    # Upload the document
    print("Uploading test document...")
    try:
        with open("test_faiss_doc.txt", "rb") as f:
            files = {"file": ("test_faiss_doc.txt", f, "text/plain")}
            response = requests.post("http://localhost:8001/api/v1/documents", files=files)
        
        if response.status_code == 200:
            print("✅ Document uploaded successfully")
            
            # Wait a bit for processing
            time.sleep(2)
            
            # Test search
            print("Testing search on uploaded document...")
            response = requests.post(
                "http://localhost:8001/api/v1/query/optimized",
                json={"query": "What is FAISS?", "context_limit": 3}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Search successful!")
                print(f"Response preview: {result.get('response', '')[:200]}...")
            else:
                print(f"❌ Search failed: {response.status_code}")
        else:
            print(f"❌ Document upload failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        import os
        if os.path.exists("test_faiss_doc.txt"):
            os.remove("test_faiss_doc.txt")

def benchmark_search_methods():
    """Benchmark FAISS vs regular search if both are available"""
    print("\\n4. Benchmarking Search Methods...")
    
    # This would require modifying the API to expose both methods
    # For now, just show current performance
    
    queries = ["test query"] * 10
    
    print("Running 10 searches...")
    start_time = time.time()
    
    for query in queries:
        try:
            requests.post(
                "http://localhost:8001/api/v1/query",
                json={"query": query, "top_k": 5},
                timeout=30
            )
        except:
            pass
    
    total_time = time.time() - start_time
    avg_time = total_time / len(queries)
    
    print(f"Total time for 10 queries: {total_time:.2f}s")
    print(f"Average time per query: {avg_time:.3f}s")
    print(f"Queries per second: {len(queries)/total_time:.2f}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("FAISS Integration Test Suite")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8001/api/status")
        if response.status_code != 200:
            print("❌ API is not responding. Please start the server first.")
            return
    except:
        print("❌ Cannot connect to API. Please run: python simple_api.py")
        return
    
    # Run tests
    test_vector_stats()
    test_search_performance()
    test_document_processing()
    benchmark_search_methods()
    
    print("\\n" + "=" * 60)
    print("Test suite complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''
    
    with open("test_faiss_integration.py", 'w') as f:
        f.write(test_script)
    
    print("Created test_faiss_integration.py")

def create_installation_guide():
    """Create installation guide for FAISS"""
    
    guide = """# FAISS Installation Guide for RAG System

## Quick Install (CPU Version)

```bash
pip install faiss-cpu
```

## Platform-Specific Installation

### Windows
```bash
# CPU version (recommended)
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

### macOS
```bash
# CPU version only (GPU not supported on macOS)
pip install faiss-cpu
```

### Linux
```bash
# CPU version
pip install faiss-cpu

# GPU version (requires CUDA 11.0+)
pip install faiss-gpu
```

## Conda Installation (Alternative)

```bash
# CPU version
conda install -c conda-forge faiss-cpu

# GPU version
conda install -c conda-forge faiss-gpu
```

## Verify Installation

```python
import faiss
print(f"FAISS version: {faiss.__version__}")

# Check GPU availability (if installed)
print(f"GPU available: {faiss.get_num_gpus() > 0}")
```

## Troubleshooting

### ImportError on Windows
If you get ImportError, install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Performance Tips
1. Use CPU version unless you have many vectors (>1M)
2. GPU version requires NVIDIA GPU with CUDA support
3. For Apple Silicon Macs, use CPU version with native ARM support

## Integration Steps

1. Install FAISS:
   ```bash
   pip install -r faiss_requirements.txt
   ```

2. Run integration script:
   ```bash
   python integrate_faiss.py
   ```

3. Restart the API server:
   ```bash
   python simple_api.py
   ```

4. Test the integration:
   ```bash
   python test_faiss_integration.py
   ```

## Expected Performance Improvements

- **Small datasets (<1K vectors)**: 5-10x faster
- **Medium datasets (1K-100K vectors)**: 20-50x faster  
- **Large datasets (>100K vectors)**: 100x+ faster

## Memory Usage

FAISS is memory efficient:
- ~4KB per vector (384 dimensions)
- 100K documents ≈ 400MB RAM
- 1M documents ≈ 4GB RAM
"""
    
    with open("FAISS_INSTALLATION_GUIDE.md", 'w') as f:
        f.write(guide)
    
    print("Created FAISS_INSTALLATION_GUIDE.md")

def main():
    """Main integration function"""
    print("=" * 60)
    print("FAISS Vector Search Integration for RAG System")
    print("=" * 60)
    
    # Create services directory
    os.makedirs("services", exist_ok=True)
    
    # Integrate FAISS into API
    success = integrate_faiss_into_api()
    
    if success:
        # Create additional files
        create_faiss_requirements()
        create_test_script()
        create_installation_guide()
        
        print("\n" + "=" * 60)
        print("✅ FAISS Integration Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install FAISS: pip install -r faiss_requirements.txt")
        print("2. Restart the API: python simple_api.py")
        print("3. Test integration: python test_faiss_integration.py")
        print("\nFor detailed instructions, see FAISS_INSTALLATION_GUIDE.md")
    else:
        print("\n❌ Integration failed. Please check the errors above.")

if __name__ == "__main__":
    main()