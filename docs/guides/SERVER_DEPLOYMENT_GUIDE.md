# 🚀 Server Deployment Guide for Production RAG System

## 📋 Server Requirements for Fast, Quality Responses

### **Minimum Server Specs (Budget)**
- **CPU**: 8+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 16-32 GB
- **GPU**: Optional but recommended (NVIDIA T4 or better)
- **Storage**: 100GB SSD
- **Models**: qwen2.5:7b, mistral:7b
- **Expected performance**: 15-30 seconds per query

### **Recommended Server Specs (Production)**
- **CPU**: 16+ cores
- **RAM**: 32-64 GB
- **GPU**: NVIDIA A10G, RTX 4090, or better
- **Storage**: 200GB NVMe SSD
- **Models**: qwen2.5:7b, command-r, mixtral:8x7b
- **Expected performance**: 5-15 seconds per query

### **High-Performance Server Specs**
- **CPU**: 32+ cores
- **RAM**: 64-128 GB
- **GPU**: NVIDIA A100 or H100
- **Storage**: 500GB NVMe SSD
- **Models**: command-r-plus, llama3.3:70b
- **Expected performance**: 2-10 seconds per query

## 🛠️ Server Deployment Steps

### 1. **Environment Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
sudo apt install python3.11 python3.11-venv python3-pip

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install NVIDIA drivers (if GPU available)
sudo apt install nvidia-driver-530
```

### 2. **Deploy Application**
```bash
# Clone repository
git clone <your-repo-url> /opt/rag-system
cd /opt/rag-system

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download production models
ollama pull qwen2.5:7b
ollama pull command-r  # If 128GB RAM available
```

### 3. **Configure for Production**
```python
# config/llm_config.yaml
default_model: qwen2.5  # or command-r for better quality

# Update model settings for server
qwen2.5:
  context_length: 8192    # Increase from 2048
  max_tokens: 2048        # Increase from 512
  temperature: 0.3        # Slightly higher for better responses
  
# .env file
API_HOST=0.0.0.0
API_PORT=8000
OLLAMA_HOST=http://localhost:11434
RAG_MAX_RESULTS=10      # Increase from 3
MAX_CONTEXT_LENGTH=8000  # Increase from 500
```

### 4. **Production Deployment with Docker**
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy application
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8000 11434

# Start script
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
```

```bash
# docker-entrypoint.sh
#!/bin/bash
# Start Ollama in background
ollama serve &

# Wait for Ollama to be ready
sleep 10

# Pull models if not exists
ollama pull qwen2.5:7b

# Start application
python -m uvicorn core.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. **Use SystemD Service (Recommended)**
```ini
# /etc/systemd/system/rag-system.service
[Unit]
Description=RAG System API
After=network.target

[Service]
Type=simple
User=rag-user
WorkingDirectory=/opt/rag-system
Environment="PATH=/opt/rag-system/venv/bin"
ExecStart=/opt/rag-system/venv/bin/python -m uvicorn core.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable rag-system
sudo systemctl start rag-system
```

### 6. **Nginx Reverse Proxy**
```nginx
# /etc/nginx/sites-available/rag-system
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

## 🚀 Performance Optimization for Servers

### 1. **Model Selection by Server Specs**
```python
# Auto-select model based on available RAM
import psutil

def select_best_model():
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if ram_gb >= 128:
        return "command-r"      # Best for RAG
    elif ram_gb >= 64:
        return "mixtral:8x7b"   # High quality
    elif ram_gb >= 32:
        return "qwen2.5:7b"     # Good balance
    elif ram_gb >= 16:
        return "mistral:7b"     # Fast & efficient
    else:
        return "tinyllama"      # Fallback
```

### 2. **GPU Acceleration**
```bash
# Check GPU availability
nvidia-smi

# Ollama will automatically use GPU if available
# Monitor GPU usage
watch -n 1 nvidia-smi
```

### 3. **Multi-Worker Configuration**
```python
# run_production.py
import multiprocessing

workers = multiprocessing.cpu_count() * 2
```

### 4. **Caching and Performance**
```python
# Enable Redis for production caching
# docker-compose.yml
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 📊 Expected Performance on Server

| Model | RAM Needed | GPU | Response Time | Quality |
|-------|------------|-----|---------------|---------|
| tinyllama | 4GB | No | 5-10s | Basic |
| mistral:7b | 8GB | Optional | 10-20s | Good |
| qwen2.5:7b | 16GB | Optional | 15-30s | Very Good |
| mixtral:8x7b | 48GB | Recommended | 20-40s | Excellent |
| command-r | 64GB | Recommended | 15-25s | Best for RAG |

## 🔧 Monitoring and Logging

```bash
# Install monitoring
pip install prometheus-client

# Add health endpoint
@app.get("/metrics")
def metrics():
    return {
        "model": current_model,
        "avg_response_time": avg_time,
        "requests_per_minute": rpm,
        "active_connections": connections
    }
```

## 🌐 Cloud Deployment Options

### **AWS EC2**
- Instance: g4dn.xlarge (GPU) or m6i.2xlarge (CPU)
- ~$0.50-1.00/hour

### **Google Cloud**
- Instance: n1-standard-8 with T4 GPU
- ~$0.40-0.80/hour

### **Azure**
- Instance: Standard_NC6s_v3
- ~$0.60-1.20/hour

### **Dedicated Servers**
- Hetzner: AX101 (AMD EPYC, 128GB RAM)
- ~€150/month

## 🎯 Quick Start for Server

```bash
# 1. SSH to server
ssh user@your-server

# 2. Quick install script
curl -fsSL https://your-repo/install.sh | bash

# 3. Configure model
echo "default_model: qwen2.5" > config/llm_config.yaml

# 4. Start service
systemctl start rag-system

# 5. Test
curl http://localhost:8000/api/v1/status
```

## 📈 Scaling Considerations

1. **Horizontal Scaling**: Use load balancer with multiple instances
2. **Model Caching**: Share model files via NFS/EFS
3. **Queue System**: Add Celery for async processing
4. **Database**: Move from SQLite to PostgreSQL
5. **Vector DB**: Consider Qdrant or Weaviate for large scale

---

**Key Takeaway**: On a proper server with 32GB+ RAM, you can run qwen2.5:7b or better models with 5-20 second response times and much higher quality than tinyllama!