#!/usr/bin/env python3
"""
Simple startup script for the RAG system
"""

if __name__ == "__main__":
    from core.main import app
    import uvicorn
    
    print("🚀 Starting RAG System...")
    print("📍 Server will be available at: http://localhost:8002")
    print("📖 API docs at: http://localhost:8002/docs")
    print("🌐 Web UI at: http://localhost:8002/ui")
    print("❤️ Health check: http://localhost:8002/health")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )