#!/usr/bin/env python3
"""
Modular RAG System API Server
Main application entry point with router-based architecture
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import routers
from .routers import documents, query, system, llm, admin, document_manager

# Import DI system
from .di.services import ServiceConfiguration, initialize_services, shutdown_services

# Import configuration
try:
    from .config.config import config
    CONFIG_AVAILABLE = True
except ImportError:
    config = None
    CONFIG_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    logger.info("Starting modular RAG API server...")
    
    try:
        # Configure and initialize all services via DI
        logger.info("Configuring dependency injection...")
        ServiceConfiguration.configure_all()
        
        logger.info("Initializing services...")
        success = await initialize_services()
        
        if not success:
            logger.error("Service initialization failed!")
            raise RuntimeError("Failed to initialize services")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down modular RAG API server...")
    try:
        await shutdown_services()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Create FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Modular Retrieval-Augmented Generation System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(system.router)
app.include_router(llm.router)
app.include_router(admin.router)
app.include_router(document_manager.router)

# Root endpoint - redirect to UI
@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect root to the web interface"""
    return HTMLResponse("""
    <script>window.location.href = '/ui'</script>
    <p>Redirecting to <a href="/ui">web interface</a>...</p>
    """)

# API info endpoint
@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ui": "/ui"
    }

# Modern frontend
@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Modern web interface"""
    try:
        static_path = Path("static/index.html")
        if static_path.exists():
            with open(static_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            # Fallback to simple interface
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>RAG System</title>
                <style>body { font-family: Arial, sans-serif; margin: 40px; }</style>
            </head>
            <body>
                <h1>RAG System</h1>
                <p>Frontend file not found. Please ensure static/index.html exists.</p>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/api">API Info</a></li>
                </ul>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(f"""
        <h1>Error loading frontend</h1>
        <p>Error: {str(e)}</p>
        <p><a href="/docs">Go to API Documentation</a></p>
        """)

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    host = config.API_HOST if CONFIG_AVAILABLE and config else "0.0.0.0"
    port = config.API_PORT if CONFIG_AVAILABLE and config else 8002
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )