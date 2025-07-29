# 🚀 RAG System - Cursor IDE Setup Guide

## Quick Start in Cursor IDE

### 1. Open Project
- Open Cursor IDE
- `File` → `Open Folder` 
- Select: `C:\Users\THE\open-source-rag-system`

### 2. Install Recommended Extensions
Cursor will automatically prompt to install recommended extensions from `.vscode/extensions.json`:
- **Python** (Microsoft)
- **Pylance** (Python language server)
- **Black Formatter** (Code formatting)
- **REST Client** (API testing)

### 3. Start the Server

#### Option A: Debug Mode (Recommended)
1. Press `F5` or go to `Run and Debug` panel
2. Select "RAG System Server"
3. Click ▶️ or press `F5`

#### Option B: Task Runner
1. Press `Ctrl + Shift + P`
2. Type "Tasks: Run Task"
3. Select "Start RAG Server"

#### Option C: Terminal
1. Press `Ctrl + Shift + ` (backtick) to open terminal
2. Run: `python run_core.py`

### 4. Access the Application
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Simple UI**: http://localhost:8000/ui

## 🔧 Development Features

### Debugging
- Set breakpoints in Python files
- Use `F5` to start debugging
- Variable inspection and step-through debugging

### API Testing
- Open `api_test.http`
- Click "Send Request" on any endpoint
- Test all API functionality directly in Cursor

### Code Formatting
- Auto-format on save (Black formatter)
- Import organization with isort
- Linting with flake8

### Tasks Available
- **Start RAG Server**: Production server
- **Start RAG Server (Development)**: With auto-reload
- **Test RAG System**: Run pytest tests
- **Install Dependencies**: pip install requirements

## 📊 Project Structure in Cursor

```
open-source-rag-system/
├── .vscode/                 # Cursor IDE configuration
│   ├── launch.json         # Debug configurations
│   ├── tasks.json          # Build and run tasks
│   ├── settings.json       # Editor settings
│   └── extensions.json     # Recommended extensions
├── core/                   # Main application code
│   ├── main.py            # FastAPI application
│   ├── services/          # Business logic
│   ├── repositories/      # Data access
│   ├── routers/           # API endpoints
│   └── di/                # Dependency injection
├── run_core.py            # Startup script
├── api_test.http          # API test requests
└── requirements.txt       # Python dependencies
```

## 🎯 Keyboard Shortcuts

- `F5` - Start debugging
- `Ctrl + F5` - Run without debugging
- `Ctrl + Shift + P` - Command palette
- `Ctrl + Shift + ` - Toggle terminal
- `Ctrl + Shift + E` - Explorer panel
- `Ctrl + Shift + D` - Debug panel

## 🐛 Troubleshooting

### Python Interpreter Not Found
1. `Ctrl + Shift + P`
2. "Python: Select Interpreter"
3. Choose your Python installation

### Port Already in Use
- Change port in `run_core.py` line 23: `port=8001`
- Or kill existing process: `taskkill /f /im python.exe`

### Dependencies Missing
- Run task: "Install Dependencies"
- Or in terminal: `pip install -r requirements.txt`

## 🚀 Ready to Code!

Your RAG system is now fully configured for development in Cursor IDE. Start the server and begin building! 🎉