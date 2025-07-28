# 🛠️ Admin System Guide - Model Management Interface

## Overview

The RAG System now includes a comprehensive admin interface that allows you to:
- **Switch between LLM models** in real-time
- **Install new models** directly from the interface
- **Monitor system status** and performance
- **Test all available models** in production

## 🎯 Key Features

### ✅ **Complete Model Library Available**
All these models are now available for testing in your production system:

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **command-r7b** | 7B | 10-20s | Excellent | German RAG (Recommended) |
| **qwen2.5** | 32B | 15-30s | Excellent | Multilingual, Complex Queries |
| **mistral** | 7B | 15-25s | Very Good | General Purpose |
| **deepseek-r1** | 8B | 30-50s | Excellent | Advanced Reasoning |
| **phi4** | 14B | 20-30s | Good | Laptops, Efficiency |
| **gemma2** | 9B | 15-25s | Good | Google's Fast Model |
| **tinyllama** | 1.1B | 10-30s | Basic | Emergency Fallback |

### ✅ **Real-Time Model Switching**
- Switch models without restarting the server
- See immediate availability status
- Install missing models with one click
- Track why models were changed (audit trail)

### ✅ **Production-Safe Testing**
- All models available for live testing
- Easy rollback to previous model
- System health monitoring
- Graceful fallback handling

## 🚀 How to Access

### **From Main UI:**
1. Go to your RAG system: `http://localhost:8000/ui`
2. Click the **"Settings"** button (top-right corner)
3. Admin interface opens: `http://localhost:8000/admin`

### **Direct Access:**
- Admin Dashboard: `http://localhost:8000/admin`
- Models API: `http://localhost:8000/admin/models`
- System Stats: `http://localhost:8000/admin/system/stats`

## 🎛️ Admin Interface Features

### **Dashboard Overview**
- **Current Model**: Shows active LLM
- **Available Models**: Total models configured
- **System Status**: Health indicator
- **Memory Usage**: Resource monitoring

### **Model Cards**
Each model shows:
- **Name & Description**: What it's optimized for
- **Status**: Available/Not Downloaded/Error/Currently Active
- **Specifications**: Context length, max tokens, temperature
- **Actions**: Switch/Install buttons

### **Admin Actions**
- **🔄 Refresh Status**: Update all model statuses
- **💾 Download Config**: Backup current configuration
- **🔄 Restart System**: Apply configuration changes
- **📋 View Logs**: System activity logs

## 📋 Step-by-Step Usage

### **1. Install a New Model**
```
1. Go to /admin
2. Find model with "NOT INSTALLED" status
3. Click "Install Model" button
4. Wait for download (may take 5-15 minutes)
5. Model status changes to "READY"
```

### **2. Switch Active Model**
```
1. Find model with "READY" status
2. Click "Switch to This Model"
3. Enter reason for switching (e.g., "Testing German performance")
4. System switches immediately
5. Model status changes to "ACTIVE"
```

### **3. Test Different Models**
```
1. Switch to command-r7b for German RAG testing
2. Go back to main UI (/ui)
3. Ask a German question about your documents
4. Note response time and quality
5. Return to admin and try qwen2.5 for comparison
```

## 🧪 Recommended Testing Sequence

### **Phase 1: Speed Testing**
1. **tinyllama** - Baseline speed (should be ~10s)
2. **command-r7b** - Target model (should be ~15s)
3. **mistral** - Alternative (should be ~20s)

### **Phase 2: Quality Testing** 
1. **command-r7b** - Best for German
2. **qwen2.5** - Best overall quality
3. **deepseek-r1** - Advanced reasoning

### **Phase 3: Production Decision**
Based on your tests, choose between:
- **Speed Priority**: command-r7b or mistral
- **Quality Priority**: qwen2.5 or deepseek-r1
- **Balance**: command-r7b (recommended)

## 🔧 Technical Details

### **Configuration File**
Models are defined in `/config/llm_config.yaml`:
```yaml
default_model: command-r7b  # Currently active

models:
  command-r7b:
    name: command-r7b:latest
    context_length: 8192
    max_tokens: 2048
    temperature: 0.3
    description: "Best for German RAG"
```

### **Model Installation**
When you click "Install Model":
1. System runs `ollama pull <model_name>`
2. Downloads model files (1-10GB depending on model)
3. Verifies installation
4. Updates status to "Available"

### **Model Switching**
When you switch models:
1. Updates `default_model` in config
2. Logs the change with reason
3. System picks up new model on next query
4. Previous model remains installed

### **API Endpoints**
- `GET /admin/` - Dashboard interface
- `GET /admin/models` - List all models with status
- `POST /admin/models/switch` - Switch active model
- `POST /admin/models/{model}/install` - Install model
- `GET /admin/system/stats` - System statistics

## 🚨 Important Notes

### **Production Safety**
- ✅ **Safe to use in production** - no data loss risk
- ✅ **Immediate rollback** - switch back anytime
- ✅ **Audit trail** - all changes logged with reasons
- ✅ **Graceful degradation** - system handles errors

### **Resource Management**
- **Disk Space**: Each model takes 1-10GB
- **Memory**: Only active model loaded in RAM
- **Network**: Initial download requires internet
- **Time**: Downloads take 5-15 minutes per model

### **Model Availability**
- **Green "READY"**: Model installed and working
- **Yellow "NOT INSTALLED"**: Click to install
- **Red "ERROR"**: Check logs or try manual installation
- **Blue "ACTIVE"**: Currently in use

## 🎉 Benefits

### **For Development**
- Test multiple models without code changes
- Quick A/B comparisons
- Easy rollback when issues occur
- Comprehensive model library

### **For Production**
- Zero-downtime model switching
- Real-time performance monitoring
- Easy deployment of model updates
- Professional admin interface

### **For Decision Making**
- Data-driven model selection
- Performance benchmarking
- Quality vs speed trade-offs
- Cost optimization

## 🔗 Quick Links

- **Main System**: http://localhost:8000/ui
- **Admin Dashboard**: http://localhost:8000/admin
- **API Documentation**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/health

---

**Result**: You now have a professional admin interface that lets you test all available LLM models in production with easy switching, monitoring, and management capabilities!