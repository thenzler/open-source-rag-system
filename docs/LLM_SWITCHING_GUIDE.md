# 🚀 LLM Switching Guide

## Quick Setup

### 1. Install Dependencies
```bash
pip install PyYAML==6.0.1
pip install rank-bm25  # Optional: for better search
```

### 2. Create Config Directory
Create the directory `config/` in your project root if it doesn't exist.

### 3. Start Server  
```bash
python simple_api.py
```

## 📋 Available Commands

### List All Models
```bash
python manage_llm.py list
```

### Switch Models
```bash
# Switch to Command-R (best for RAG)
python manage_llm.py switch command-r

# Switch to Llama 3.2
python manage_llm.py switch llama3.2

# Switch to Mistral  
python manage_llm.py switch mistral
```

### Check Status
```bash
python manage_llm.py status
```

### Pull New Models
```bash
# Pull Command-R (best for RAG)
python manage_llm.py pull command-r:latest

# Pull Llama 3.2:3b
python manage_llm.py pull llama3.2:3b
```

## 🌐 API Endpoints

### List Models
```bash
curl http://localhost:8001/api/v1/llm/models
```

### Switch Model
```bash
curl -X POST http://localhost:8001/api/v1/llm/switch/command-r
```

### Get Status
```bash
curl http://localhost:8001/api/v1/llm/status
```

### Pull Model
```bash
curl -X POST http://localhost:8001/api/v1/llm/pull/command-r:latest
```

## ⚙️ Configuration

Edit `config/llm_config.yaml` to:
- Add new models
- Change default model
- Customize prompts  
- Adjust parameters

Example:
```yaml
default_model: "command-r:latest"

models:
  my-model:
    name: "my-custom-model:latest"
    description: "My custom model"
    temperature: 0.3
    max_tokens: 2048
    prompt_template: "default"
```

## 🎯 Recommended Models for RAG

### Best: Command-R
```bash
python manage_llm.py pull command-r:latest
python manage_llm.py switch command-r
```

### Good: Llama 3.2:3b
```bash  
python manage_llm.py pull llama3.2:3b
python manage_llm.py switch llama3.2
```

### Fast: Mistral 7B
```bash
python manage_llm.py pull mistral:7b-instruct-v0.2  
python manage_llm.py switch mistral
```

## 🔧 Advanced Usage

### Custom Prompts
Each model can have its own optimized prompt template in `llm_config.yaml`.

### Dynamic Switching
Switch models without restarting the server:
```bash
# Test with Mistral (fast)
python manage_llm.py switch mistral

# Switch to Command-R for better quality
python manage_llm.py switch command-r
```

### Monitor Performance
```bash
# Check current setup
python manage_llm.py status

# Test query
curl -X POST "http://localhost:8001/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Was gehört in den Biomüll", "top_k": 3}'
```

This system makes it **super easy** to switch between models and find the best one for your specific use case!