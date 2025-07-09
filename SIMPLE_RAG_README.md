# Simple RAG System - No Docker Required

A simplified version of the RAG (Retrieval-Augmented Generation) system that runs directly on your machine without Docker or virtual machines.

## ✨ Features

- **Document Upload**: Support for PDF, Word (.docx), Text (.txt), and CSV files
- **Text Extraction**: Automatically extracts text from uploaded documents
- **Vector Search**: Uses sentence transformers for semantic search
- **Web Interface**: Simple HTML interface for testing
- **RESTful API**: Clean API endpoints for integration
- **No Dependencies**: Runs without Docker, databases, or external services

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

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

3. **Start the system**
   ```bash
   # Option 1: Use the batch file (Windows)
   start_simple_rag.bat
   
   # Option 2: Use Python script
   python start_simple_rag.py
   
   # Option 3: Start directly
   python simple_api.py
   ```

4. **Access the system**
   - Web Interface: http://localhost:8001/simple_frontend.html
   - API Documentation: http://localhost:8001/docs
   - API Base URL: http://localhost:8001

## 📁 File Structure

```
├── simple_api.py              # Main API server
├── simple_requirements.txt    # Python dependencies
├── simple_frontend.html       # Web interface
├── start_simple_rag.py       # Startup script
├── start_simple_rag.bat      # Windows batch file
├── test_simple_rag.py        # Test script
├── storage/                  # Document storage
│   ├── uploads/             # Uploaded files
│   └── processed/           # Processed files
└── SIMPLE_RAG_README.md     # This file
```

## 🔧 API Endpoints

### Upload Document
```bash
POST /api/v1/documents
Content-Type: multipart/form-data

curl -X POST "http://localhost:8001/api/v1/documents" \
  -F "file=@document.pdf"
```

### Query Documents
```bash
POST /api/v1/query
Content-Type: application/json

curl -X POST "http://localhost:8001/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'
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

Run the test script to verify everything works:

```bash
python test_simple_rag.py
```

This will:
1. Test API health
2. Upload a test document
3. Query the document
4. List documents

## 📊 How It Works

1. **Document Upload**: Files are uploaded and saved to `storage/uploads/`
2. **Text Extraction**: Text is extracted using appropriate libraries (PyPDF2, python-docx, etc.)
3. **Text Chunking**: Documents are split into manageable chunks (512 words with 50 word overlap)
4. **Embedding Creation**: Each chunk is converted to vectors using sentence-transformers
5. **Vector Search**: Queries are embedded and compared using cosine similarity
6. **Results**: Most similar chunks are returned with similarity scores

## 🔍 Supported File Types

- **PDF** (.pdf) - Requires PyPDF2
- **Word** (.docx) - Requires python-docx
- **Text** (.txt) - Native support
- **CSV** (.csv) - Requires pandas

## ⚙️ Configuration

You can modify these settings in `simple_api.py`:

```python
# Text chunking
chunk_size = 512        # Words per chunk
chunk_overlap = 50      # Overlapping words

# Search
top_k = 5              # Default number of results
embedding_model = 'all-MiniLM-L6-v2'  # Sentence transformer model
```

## 🎯 Use Cases

- **Document Q&A**: Ask questions about your documents
- **Knowledge Base**: Search through company documentation
- **Research**: Find relevant information in academic papers
- **Content Discovery**: Explore large document collections

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

# Query document
query_data = {"query": "What is the main topic?", "top_k": 5}
response = requests.post('http://localhost:8001/api/v1/query', json=query_data)
results = response.json()
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
.then(data => console.log(data));

// Query document
fetch('http://localhost:8001/api/v1/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: "What is machine learning?", top_k: 5})
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📈 Performance

- **Embedding Model**: Uses `all-MiniLM-L6-v2` (small, fast model)
- **Memory Usage**: Stores embeddings in memory (RAM)
- **Processing Speed**: ~1-2 seconds per document
- **Search Speed**: ~100-200ms per query

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

---

**Happy RAG-ing!** 🤖📚