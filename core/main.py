#!/usr/bin/env python3
"""
Modular RAG System API Server
Main application entry point with router-based architecture
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
import secrets
import hmac
import hashlib
import time

# Import routers
from .routers import documents, query, system, llm, admin, document_manager, metrics, async_processing, compliance, progress, cache

# Import DI system
from .di.services import ServiceConfiguration, initialize_services, shutdown_services

# Import multi-tenancy
from .middleware import tenant_middleware, initialize_tenant_resolver
from .repositories.tenant_repository import TenantRepository

# Import security
from .utils.security import initialize_id_obfuscator
from .utils.encryption import setup_encryption_from_config

# Import metrics
from .services.metrics_service import init_metrics_service
from .middleware.metrics_middleware import MetricsMiddleware

# Import async processing
from .services.async_processing_service import initialize_async_processor, shutdown_async_processor
from .processors import register_document_processors

# Import compliance service
from .services.compliance_service import initialize_compliance_service

# Import progress tracking
from .services.progress_tracking_service import initialize_progress_tracker

# Import cache service
from .services.redis_cache_service import initialize_cache_service, shutdown_cache_service

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
        
        # Initialize security
        logger.info("Initializing security systems...")
        secret_key = config.SECRET_KEY if CONFIG_AVAILABLE and config and hasattr(config, 'SECRET_KEY') else 'default-secret-key'
        initialize_id_obfuscator(secret_key)
        
        # Initialize encryption if enabled
        if CONFIG_AVAILABLE and config:
            encryption_setup = setup_encryption_from_config(config)
            if encryption_setup:
                logger.info("Encryption enabled and configured")
            else:
                logger.info("Encryption disabled or not configured")
        else:
            logger.info("No config available, encryption disabled")
        
        logger.info("Security systems initialized successfully")
        
        # Initialize multi-tenancy
        logger.info("Initializing multi-tenancy...")
        db_path = config.DATABASE_PATH if CONFIG_AVAILABLE and config and hasattr(config, 'DATABASE_PATH') else 'data/rag_database.db'
        tenant_repo = TenantRepository(db_path)
        initialize_tenant_resolver(tenant_repo)
        logger.info("Multi-tenancy initialized successfully")
        
        # Initialize metrics service
        logger.info("Initializing metrics service...")
        init_metrics_service()
        logger.info("Metrics service initialized successfully")
        
        # Initialize async processing service
        logger.info("Initializing async document processing...")
        await initialize_async_processor(max_workers=4)
        await register_document_processors()
        logger.info("Async document processing initialized successfully")
        
        # Initialize compliance service
        logger.info("Initializing compliance service...")
        initialize_compliance_service(
            storage_path="data/compliance",
            enable_audit_logging=True,
            data_residency_region="CH"
        )
        logger.info("Compliance service initialized successfully")
        
        # Initialize progress tracking service
        logger.info("Initializing progress tracking service...")
        await initialize_progress_tracker(persistence_file="data/progress_operations.json")
        logger.info("Progress tracking service initialized successfully")
        
        # Initialize cache service (Redis)
        logger.info("Initializing cache service...")
        redis_url = config.REDIS_URL if CONFIG_AVAILABLE and config and hasattr(config, 'REDIS_URL') else "redis://localhost:6379"
        redis_db = config.REDIS_DB if CONFIG_AVAILABLE and config and hasattr(config, 'REDIS_DB') else 0
        await initialize_cache_service(
            redis_url=redis_url,
            redis_db=redis_db,
            enable_compression=True
        )
        logger.info("Cache service initialization completed")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down modular RAG API server...")
    try:
        # Shutdown services in reverse order
        await shutdown_cache_service()
        await shutdown_async_processor()
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

# CSRF Token Management
CSRF_SECRET_KEY = getattr(config, 'SECRET_KEY', 'default-secret-key') if CONFIG_AVAILABLE and config else 'default-secret-key'

def generate_csrf_token() -> str:
    """Generate a secure CSRF token"""
    token = secrets.token_urlsafe(32)
    timestamp = str(int(time.time()))
    message = f"{token}:{timestamp}"
    signature = hmac.new(
        CSRF_SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{token}:{timestamp}:{signature}"

def validate_csrf_token(token: str) -> bool:
    """Validate CSRF token"""
    try:
        parts = token.split(':')
        if len(parts) != 3:
            return False
        
        token_part, timestamp, signature = parts
        message = f"{token_part}:{timestamp}"
        expected_signature = hmac.new(
            CSRF_SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Check signature
        if not hmac.compare_digest(signature, expected_signature):
            return False
        
        # Check if token is not too old (24 hours)
        token_age = time.time() - int(timestamp)
        if token_age > 86400:  # 24 hours
            return False
        
        return True
    except (ValueError, TypeError):
        return False

# Security Headers Middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # HSTS (only add if HTTPS)
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Remove server information
    response.headers.pop("server", None)
    
    return response

# CSRF Middleware
@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """CSRF protection middleware"""
    # Skip CSRF for GET, HEAD, OPTIONS requests
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        response = await call_next(request)
        return response
    
    # Skip CSRF for health checks and system endpoints
    if request.url.path in ["/health", "/api/v1/health", "/api/v1/status"]:
        response = await call_next(request)
        return response
    
    # For state-changing requests, check CSRF token
    csrf_token = request.headers.get("X-CSRF-Token")
    if not csrf_token:
        # Also check in form data for HTML forms
        if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
            try:
                form = await request.form()
                csrf_token = form.get("csrf_token")
            except:
                pass
    
    if not csrf_token or not validate_csrf_token(csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")
    
    response = await call_next(request)
    return response

# Add security middleware
app.add_middleware(SessionMiddleware, secret_key=CSRF_SECRET_KEY)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production

# Add tenant middleware
app.middleware("http")(tenant_middleware)

# Add metrics middleware
app.add_middleware(MetricsMiddleware, collect_detailed_metrics=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-CSRF-Token"],
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
app.include_router(metrics.router)
app.include_router(async_processing.router)
app.include_router(compliance.router)
app.include_router(progress.router)
app.include_router(cache.router)

# Include tenant router
from .routers import tenants
app.include_router(tenants.router)

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

# CSRF token endpoint
@app.get("/api/v1/csrf-token")
async def get_csrf_token():
    """Get CSRF token for form submissions"""
    return {
        "csrf_token": generate_csrf_token(),
        "expires_in": 86400  # 24 hours
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
        app,
        host=host,
        port=port,
        reload=False,  # Disable reload to avoid import issues
        log_level="info"
    )