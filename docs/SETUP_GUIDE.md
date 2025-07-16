# Setup Guide

This guide will help you set up and run the Open Source RAG System on your local machine.

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **Internet**: Required for initial setup and model downloads

### Required Software

1. **Python 3.8+** - Download from [python.org](https://python.org)
2. **Git** - Download from [git-scm.com](https://git-scm.com)
3. **Ollama** - We'll install this during setup

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/open-source-rag-system.git
cd open-source-rag-system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r simple_requirements.txt
```

If you encounter any issues, try upgrading pip first:
```bash
pip install --upgrade pip
```

### Step 4: Install Ollama

#### Windows
```bash
python install_ollama.py
```

#### macOS/Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 5: Download an LLM Model

Start with a smaller model for testing:
```bash
# Start Ollama service first
ollama serve

# In a new terminal, download a model
ollama pull tinyllama:latest
```

For better performance, you can also download larger models:
```bash
ollama pull mistral:latest
ollama pull phi3:mini
```

### Step 6: Configure the System

The system should work with default settings, but you can customize:

1. **Edit LLM Configuration** (optional):
   ```bash
   # Edit config/llm_config.yaml
   notepad config/llm_config.yaml  # Windows
   nano config/llm_config.yaml    # Linux/macOS
   ```

2. **Check Database Configuration**:
   - The system uses SQLite by default
   - Database will be created automatically at first run

### Step 7: Run the System

1. **Start Ollama** (in one terminal):
   ```bash
   ollama serve
   ```

2. **Start the RAG System** (in another terminal):
   ```bash
   python start_simple_rag.py
   ```

3. **Access the Web Interface**:
   - Open your browser to: http://localhost:8001
   - You should see the RAG system interface

## Verification

### Test the Installation

1. **Check System Status**:
   ```bash
   curl http://localhost:8001/api/health
   ```

2. **Upload a Test Document**:
   - Use the web interface to upload a PDF or text file
   - Or use the API:
   ```bash
   curl -X POST -F "file=@test_document.pdf" http://localhost:8001/api/upload_document
   ```

3. **Test Query**:
   - Ask a question through the web interface
   - Or use the API:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "What is this document about?"}' \
     http://localhost:8001/api/query_enhanced
   ```

## Troubleshooting

### Common Issues

#### 1. Ollama Not Found
**Error**: `ollama: command not found`

**Solution**:
- Ensure Ollama is installed correctly
- Check if it's in your PATH
- Try reinstalling with the installation script

#### 2. Python Dependencies Issues
**Error**: `ModuleNotFoundError` or package installation failures

**Solution**:
```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Clear pip cache
pip cache purge

# Reinstall requirements
pip install -r simple_requirements.txt --force-reinstall
```

#### 3. Port Already in Use
**Error**: `Port 8001 is already in use`

**Solution**:
- Stop any existing instances
- Change the port in `simple_api.py` (look for `port=8001`)
- Or kill the process using the port

#### 4. Memory Issues
**Error**: Out of memory errors during document processing

**Solution**:
- Use smaller models (tinyllama instead of mistral)
- Process fewer documents at once
- Increase system RAM if possible

#### 5. Database Issues
**Error**: Database connection or SQLite errors

**Solution**:
```bash
# Delete the database file to reset
rm rag_database.db

# Restart the system
python start_simple_rag.py
```

### Getting Help

If you encounter issues:

1. **Check the logs** in the console output
2. **Enable debug mode** by setting `DEBUG=True` in the code
3. **Check system requirements** are met
4. **Consult the troubleshooting section** in the main README
5. **Open an issue** on GitHub with:
   - Your operating system
   - Python version
   - Error messages
   - Steps to reproduce

## Advanced Configuration

### Custom Models

To use different models:

1. **Edit `config/llm_config.yaml`**:
   ```yaml
   default_model: my_custom_model
   
   models:
     my_custom_model:
       name: "custom-model:latest"
       context_length: 4096
       temperature: 0.7
   ```

2. **Pull the model with Ollama**:
   ```bash
   ollama pull custom-model:latest
   ```

### Environment Variables

You can configure the system using environment variables:

```bash
# Set custom port
export RAG_PORT=8002

# Set custom database path
export RAG_DATABASE_PATH="/path/to/database.db"

# Set debug mode
export RAG_DEBUG=true
```

### Production Deployment

For production use:

1. **Use a production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn simple_api:app --host 0.0.0.0 --port 8001
   ```

2. **Set up reverse proxy** (nginx, Apache)

3. **Configure SSL/TLS** for HTTPS

4. **Set up monitoring** and logging

5. **Configure firewall** rules

## Performance Optimization

### Hardware Recommendations

- **CPU**: 4+ cores recommended
- **RAM**: 16GB+ for large document sets
- **Storage**: SSD for better performance
- **GPU**: Optional, can accelerate some operations

### Software Optimizations

1. **Use faster models** for development (tinyllama)
2. **Enable caching** (enabled by default)
3. **Limit concurrent uploads** to prevent memory issues
4. **Regular cleanup** of old documents

## Next Steps

After successful setup:

1. **Upload your documents** through the web interface
2. **Test different query types** to understand capabilities
3. **Explore the API** using the API documentation
4. **Integrate with your applications** using the provided endpoints
5. **Consider production deployment** if needed

## Support

For additional support:

- **Documentation**: Check the `docs/` folder
- **API Reference**: See `docs/API_DOCUMENTATION.md`
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions and share experiences

---

Congratulations! You now have a fully functional RAG system running locally. Start by uploading some documents and asking questions to see the system in action.