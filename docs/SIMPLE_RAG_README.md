# Simple RAG System with Ollama LLM - No Docker Required

A complete RAG (Retrieval-Augmented Generation) system with AI answer generation that runs directly on your machine without Docker or virtual machines.

## ✨ Features

- **🤖 AI Answer Generation**: Uses Ollama LLM to generate coherent answers from your documents
- **📄 Document Upload**: Support for PDF, Word (.docx), Text (.txt), and CSV files
- **🔍 Smart Search**: Semantic vector search with sentence transformers
- **🎯 Source Attribution**: Every answer includes source document references
- **🔄 Graceful Fallback**: Falls back to vector search if LLM unavailable
- **🌐 Web Interface**: Modern interface with AI/vector search modes
- **🛠️ RESTful API**: Clean API endpoints for integration
- **🚀 No Docker**: Runs directly on your machine

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- [Ollama](https://ollama.ai) (for AI answer generation - optional)

### Installation & Setup

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/thenzler/open-source-rag-system.git
   cd open-source-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r simple_requirements.txt
   ```

3. **Install Ollama (Optional but Recommended)**
   ```bash
   # Download from: https://ollama.ai/download
   # Install for your operating system
   
   # After installation, pull a model:
   ollama pull llama3.1:8b    # Recommended (needs 8GB RAM)
   # or
   ollama pull phi-3:mini     # Faster option (needs 4GB RAM)
   ```

4. **Start the system**
   ```bash
   # Option 1: Use the batch file (Windows)
   start_simple_rag.bat
   
   # Option 2: Use Python script
   python start_simple_rag.py
   
   # Option 3: Start directly
   python simple_api.py
   ```

5. **Access the system**
   - Web Interface: http://localhost:8001/simple_frontend.html
   - API Documentation: http://localhost:8001/docs
   - API Base URL: http://localhost:8001

## 📁 File Structure

```
├── simple_api.py              # Main API server with LLM integration
├── ollama_client.py           # Ollama LLM client
├── simple_requirements.txt    # Python dependencies
├── simple_frontend.html       # Web interface with AI modes
├── start_simple_rag.py       # Startup script
├── start_simple_rag.bat      # Windows batch file
├── test_simple_rag.py        # Basic test script
├── test_ollama_integration.py # Comprehensive LLM tests
├── storage/                  # Document storage
│   ├── uploads/             # Uploaded files
│   └── processed/           # Processed files
├── SIMPLE_RAG_README.md     # This file
└── DEVELOPMENT_NOTES.md     # Development tracking
```

## 🔧 API Endpoints

### Upload Document
```bash
POST /api/v1/documents
Content-Type: multipart/form-data

curl -X POST "http://localhost:8001/api/v1/documents" \
  -F "file=@document.pdf"
```

### Enhanced Query (AI + Vector Search)
```bash
POST /api/v1/query/enhanced
Content-Type: application/json

curl -X POST "http://localhost:8001/api/v1/query/enhanced" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5, "use_llm": true}'
```

### Vector Search Only
```bash
POST /api/v1/query
Content-Type: application/json

curl -X POST "http://localhost:8001/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'
```

### System Status
```bash
GET /api/v1/status

curl "http://localhost:8001/api/v1/status"
```

### List Documents
```bash
GET /api/v1/documents

curl "http://localhost:8001/api/v1/documents"
```

### Delete Document
```bash
DELETE /api/v1/documents/{document_id}

curl -X DELETE "http://localhost:8001/api/v1/documents/1"
```

## 🧪 Testing

### Quick Test
```bash
python test_simple_rag.py
```

### Comprehensive LLM Test
```bash
python test_ollama_integration.py
```

This will:
1. Test API health and system status
2. Check Ollama availability
3. Upload a comprehensive test document
4. Test vector search functionality
5. Test AI answer generation
6. Test fallback mechanisms
7. Provide setup instructions if needed

## 📊 How It Works

### Document Processing
1. **Document Upload**: Files are uploaded and saved to `storage/uploads/`
2. **Text Extraction**: Text is extracted using appropriate libraries (PyPDF2, python-docx, etc.)
3. **Text Chunking**: Documents are split into manageable chunks (512 words with 50 word overlap)
4. **Embedding Creation**: Each chunk is converted to vectors using sentence-transformers

### Query Processing
**Mode 1: AI Generated Answers (Recommended)**
1. **Vector Search**: Query is embedded and compared with document chunks using cosine similarity
2. **Context Preparation**: Top relevant chunks are combined into context for LLM
3. **LLM Generation**: Ollama generates a coherent answer based on the context
4. **Source Attribution**: Answer includes references to source documents
5. **Fallback**: If LLM fails, falls back to vector search results

**Mode 2: Vector Search Only**
1. **Vector Search**: Query is embedded and compared using cosine similarity
2. **Results**: Most similar chunks are returned with similarity scores

## 🔍 Supported File Types

- **PDF** (.pdf) - Requires PyPDF2
- **Word** (.docx) - Requires python-docx
- **Text** (.txt) - Native support
- **CSV** (.csv) - Requires pandas

## 🤖 Ollama Setup Guide

### Installation
1. **Download Ollama**: Visit [ollama.ai/download](https://ollama.ai/download)
2. **Install**: Follow the installation instructions for your OS
3. **Start Ollama**: Run `ollama serve` in terminal (or it starts automatically)

### Model Selection
```bash
# Recommended models:
ollama pull llama3.1:8b      # Best quality (8GB RAM)
ollama pull phi-3:mini       # Fastest (4GB RAM)
ollama pull mistral:7b       # Balanced (7GB RAM)

# List installed models:
ollama list

# Test a model:
ollama run llama3.1:8b "Hello world"
```

### Memory Requirements
- **llama3.1:8b**: 8GB RAM, best quality answers
- **phi-3:mini**: 4GB RAM, fast responses
- **mistral:7b**: 7GB RAM, good balance

### Troubleshooting
- **Connection failed**: Check if `ollama serve` is running
- **Model not found**: Run `ollama pull <model-name>`
- **Out of memory**: Try a smaller model like `phi-3:mini`

## ⚙️ Configuration

You can modify these settings in `simple_api.py`:

```python
# Text chunking
chunk_size = 512        # Words per chunk
chunk_overlap = 50      # Overlapping words

# Search
top_k = 5              # Default number of results
embedding_model = 'all-MiniLM-L6-v2'  # Sentence transformer model

# LLM settings
USE_LLM_DEFAULT = True          # Try LLM by default
MAX_CONTEXT_LENGTH = 4000       # Max context for LLM

# Ollama configuration in ollama_client.py:
base_url = "http://localhost:11434"  # Ollama API URL
model = "llama3.1:8b"               # Default model
timeout = 30                        # Request timeout
```

## 🎯 Use Cases

- **📚 Document Q&A**: Get AI-generated answers from your document collection
- **🏢 Knowledge Base**: Transform company docs into an intelligent assistant
- **🔬 Research Assistant**: Ask complex questions about academic papers
- **📖 Content Discovery**: Explore and understand large document collections
- **💼 Legal/Medical**: Extract insights from professional documents
- **📝 Learning**: Create study aids from textbooks and materials

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Install dependencies with `pip install -r simple_requirements.txt`
2. **Port already in use**: Change port in `simple_api.py` (line with `port=8001`)
3. **File upload fails**: Check file permissions and supported formats
4. **No search results**: Ensure documents are uploaded and processed

### Debug Mode

Enable debug logging by modifying `simple_api.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔗 Integration

### Python Client Example

```python
import requests

# Upload document
with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    response = requests.post('http://localhost:8001/api/v1/documents', files=files)

# Get AI-generated answer
query_data = {
    "query": "What are the main conclusions of this document?", 
    "top_k": 5, 
    "use_llm": True
}
response = requests.post('http://localhost:8001/api/v1/query/enhanced', json=query_data)
result = response.json()

print("AI Answer:", result['answer'])
print("Sources:", [src['source_document'] for src in result['sources']])
```

### JavaScript Client Example

```javascript
// Upload document
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8001/api/v1/documents', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log('Uploaded:', data.filename));

// Get AI answer
fetch('http://localhost:8001/api/v1/query/enhanced', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        query: "Summarize the key points", 
        top_k: 5, 
        use_llm: true
    })
})
.then(response => response.json())
.then(data => {
    console.log('Answer:', data.answer);
    console.log('Method:', data.method);
    console.log('Sources:', data.sources.length);
});
```

## 📈 Performance

### Vector Search Mode
- **Embedding Model**: Uses `all-MiniLM-L6-v2` (small, fast model)
- **Memory Usage**: Stores embeddings in memory (RAM)
- **Processing Speed**: ~1-2 seconds per document
- **Search Speed**: ~100-200ms per query

### AI Answer Mode
- **LLM Processing**: 2-10 seconds depending on model and query complexity
- **Context Preparation**: ~50-100ms
- **Total Response Time**: 2-10 seconds for AI answers
- **Fallback Speed**: <500ms if LLM unavailable

### Resource Usage
- **RAM**: 2-4GB for embeddings + LLM model size (4-8GB)
- **CPU**: Moderate usage during processing
- **Storage**: Document files + model files (4-8GB per LLM model)

## 🛡️ Security

- No authentication required (for simplicity)
- Files stored locally
- No external API calls
- CORS enabled for localhost

## 🎨 Customization

### Change Embedding Model

```python
# In simple_api.py
embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Better quality
# or
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')  # Balanced
```

### Add New File Types

```python
# In simple_api.py, add to extract_text_from_file function
elif content_type == "application/new-type":
    return extract_text_from_new_type(file_path)
```

## 📝 License

MIT License - feel free to use and modify as needed.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test script
3. Check the API documentation at `/docs`
4. Create an issue on GitHub

## 🚀 Next Steps

To enhance the system:
1. Add authentication
2. Use a proper database
3. Add more file type support
4. Implement caching
5. Add LLM integration for answer generation
6. Deploy to cloud services

## 🎉 Summary

This RAG system provides:

**✅ Complete AI-powered document question answering**
- Upload documents → Ask questions → Get intelligent answers
- Full source attribution and transparency
- Works entirely offline and locally

**✅ Two modes for different needs**
- **AI Mode**: Coherent, synthesized answers from Ollama LLM
- **Search Mode**: Raw document chunks for manual review

**✅ Production-ready features**
- Robust error handling and fallback mechanisms
- RESTful API for integration
- Comprehensive testing and monitoring
- Easy setup without Docker complexity

**✅ Privacy and control**
- No external API calls
- All processing happens locally
- You own your data and AI models

**🚀 Ready to transform your documents into an intelligent assistant!**

---

**Happy RAG-ing!** 🤖📚