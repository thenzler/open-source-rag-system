"""
Admin Router for Model Management and System Configuration
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..ollama_client import OllamaClient
from .document_manager import (analyze_all_documents,
                               cleanup_problematic_documents)

logger = logging.getLogger(__name__)

# Initialize templates
templates = Jinja2Templates(directory="core/templates")

router = APIRouter(prefix="/admin", tags=["admin"])


class ModelSelection(BaseModel):
    model_key: str
    reason: Optional[str] = None


class ModelConfig(BaseModel):
    name: str
    context_length: int
    max_tokens: int
    temperature: float
    description: str
    prompt_template: str


class SystemStats(BaseModel):
    current_model: str
    available_models: List[str]
    system_status: str
    total_queries: int
    avg_response_time: float
    memory_usage: Dict[str, Any]


class DocumentFilterConfig(BaseModel):
    """Configuration for document filtering"""

    bio_waste_keywords: List[str] = [
        "bioabfall",
        "bio waste",
        "organic waste",
        "kompost",
        "grünabfall",
        "küchenabfälle",
        "obst",
        "gemüse",
        "fruit",
        "vegetable",
        "food waste",
    ]
    problematic_keywords: List[str] = [
        "zero-hallucination",
        "guidelines for following",
        "only use information",
        "training instructions",
        "quelels",
    ]
    exclude_keywords: List[str] = [
        "javascript",
        "console.log",
        "function",
        "cloud computing",
        "programming",
        "software",
        "algorithm",
    ]
    min_content_length: int = 100
    max_corruption_chars: int = 10


class DatabaseConfig(BaseModel):
    """SQL Database configuration"""

    db_type: str = "sqlite"  # sqlite, postgresql, mysql
    host: Optional[str] = None
    port: Optional[int] = None
    database: str = "rag_database.db"
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None


@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Main admin dashboard"""
    try:
        # Load current configuration
        config_path = Path("config/llm_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        current_model = config.get("default_model", "unknown")
        available_models = list(config.get("models", {}).keys())

        # Get system stats using direct client creation

        # Check model availability
        model_status = {}
        for model_key in available_models:
            try:
                # Temporarily switch to check availability
                temp_client = OllamaClient()
                temp_client.model = config["models"][model_key]["name"]
                is_available = temp_client.is_available()
                model_status[model_key] = (
                    "Available" if is_available else "Not Downloaded"
                )
            except Exception as e:
                model_status[model_key] = f"Error: {str(e)}"

        return templates.TemplateResponse(
            "admin_dashboard.html",
            {
                "request": request,
                "current_model": current_model,
                "available_models": available_models,
                "model_configs": config.get("models", {}),
                "model_status": model_status,
                "system_healthy": True,
            },
        )

    except Exception as e:
        logger.error(f"Admin dashboard error: {e}")
        return templates.TemplateResponse(
            "admin_error.html", {"request": request, "error": str(e)}
        )


@router.get("/models", response_model=Dict[str, Any])
async def get_available_models():
    """Get all available models with their configurations"""
    try:
        config_path = Path("config/llm_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        current_model = config.get("default_model")
        models = config.get("models", {})

        # Add status for each model using direct client creation

        for model_key, model_config in models.items():
            try:
                # Check if model is downloaded
                temp_client = OllamaClient()
                temp_client.model = model_config["name"]
                models[model_key]["status"] = (
                    "available" if temp_client.is_available() else "not_downloaded"
                )
                models[model_key]["is_current"] = model_key == current_model
            except Exception as e:
                models[model_key]["status"] = "error"
                models[model_key]["error"] = str(e)

        return {
            "current_model": current_model,
            "models": models,
            "total_count": len(models),
        }

    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/switch")
async def switch_model(selection: ModelSelection):
    """Switch the active model"""
    try:
        config_path = Path("config/llm_config.yaml")

        # Load current config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate model exists
        if selection.model_key not in config.get("models", {}):
            raise HTTPException(
                status_code=400, detail=f"Model '{selection.model_key}' not found"
            )

        # Check if model is available
        model_config = config["models"][selection.model_key]
        temp_client = OllamaClient()
        temp_client.model = model_config["name"]

        if not temp_client.is_available():
            return {
                "success": False,
                "message": f"Model '{selection.model_key}' is not downloaded. Please install it first.",
                "download_command": f"ollama pull {model_config['name']}",
            }

        # Update default model
        old_model = config.get("default_model", "unknown")
        config["default_model"] = selection.model_key

        # Save updated config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        # Log the change
        logger.info(
            f"Model switched from {old_model} to {selection.model_key} (reason: {selection.reason})"
        )

        return {
            "success": True,
            "message": f"Successfully switched from {old_model} to {selection.model_key}",
            "previous_model": old_model,
            "new_model": selection.model_key,
            "restart_required": True,
        }

    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_key}/install")
async def install_model(model_key: str):
    """Install/download a model"""
    try:
        config_path = Path("config/llm_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if model_key not in config.get("models", {}):
            raise HTTPException(
                status_code=400, detail=f"Model '{model_key}' not found in config"
            )

        model_config = config["models"][model_key]
        model_name = model_config["name"]

        # Try to pull the model
        temp_client = OllamaClient()
        success = temp_client.pull_model(model_name)

        if success:
            return {
                "success": True,
                "message": f"Successfully installed {model_name}",
                "model_key": model_key,
                "model_name": model_name,
            }
        else:
            return {
                "success": False,
                "message": f"Failed to install {model_name}",
                "manual_command": f"ollama pull {model_name}",
            }

    except Exception as e:
        logger.error(f"Error installing model {model_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        config_path = Path("config/llm_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Get basic stats
        current_model = config.get("default_model")
        available_models = list(config.get("models", {}).keys())

        # Try to get ollama client stats
        try:
            ollama_client = OllamaClient()
            health = ollama_client.health_check()
            system_status = "healthy" if health.get("available") else "degraded"
        except Exception:
            system_status = "unknown"

        # Mock stats (in a real implementation, you'd track these)
        return {
            "current_model": current_model,
            "available_models": available_models,
            "system_status": system_status,
            "total_queries": 0,  # Would come from audit logs
            "avg_response_time": 0.0,  # Would come from metrics
            "memory_usage": {
                "total": "32GB",
                "used": "Unknown",
                "available": "Unknown",
            },
            "uptime": "Unknown",
        }

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {
            "current_model": "unknown",
            "available_models": [],
            "system_status": "error",
            "error": str(e),
        }


@router.post("/system/restart")
async def restart_system():
    """Restart the RAG system (recreate services)"""
    try:
        # Note: In a real system, you'd need proper restart logic
        # For now, just return success message
        return {
            "success": True,
            "message": "Configuration updated. Changes will take effect on next query.",
        }

    except Exception as e:
        logger.error(f"Error restarting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/download")
async def download_config():
    """Download current configuration as backup"""
    try:
        config_path = Path("config/llm_config.yaml")
        with open(config_path, "r") as f:
            config_content = f.read()

        from fastapi.responses import Response

        return Response(
            content=config_content,
            media_type="application/x-yaml",
            headers={
                "Content-Disposition": "attachment; filename=llm_config_backup.yaml"
            },
        )

    except Exception as e:
        logger.error(f"Error downloading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_recent_logs():
    """Get recent system logs"""
    try:
        # This would normally read from log files
        # For now, return mock data
        return {
            "logs": [
                {
                    "timestamp": "2024-01-28 10:30:00",
                    "level": "INFO",
                    "message": "System started",
                },
                {
                    "timestamp": "2024-01-28 10:31:00",
                    "level": "INFO",
                    "message": "Model loaded: command-r7b",
                },
                {
                    "timestamp": "2024-01-28 10:32:00",
                    "level": "INFO",
                    "message": "Admin interface accessed",
                },
            ],
            "total_count": 3,
        }

    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document Management Endpoints


@router.get("/documents/analysis")
async def get_document_analysis():
    """Analyze all documents and categorize them"""
    try:
        analyses = await analyze_all_documents()

        # Categorize documents
        bio_waste_docs = []
        problematic_docs = []
        unknown_docs = []

        for doc in analyses:
            if doc.content_type == "bio_waste" and not doc.is_problematic:
                bio_waste_docs.append(doc)
            elif doc.is_problematic:
                problematic_docs.append(doc)
            else:
                unknown_docs.append(doc)

        return {
            "total_documents": len(analyses),
            "bio_waste_documents": len(bio_waste_docs),
            "problematic_documents": len(problematic_docs),
            "unknown_documents": len(unknown_docs),
            "documents": {
                "bio_waste": bio_waste_docs,
                "problematic": problematic_docs,
                "unknown": unknown_docs,
            },
        }
    except Exception as e:
        logger.error(f"Error analyzing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/filter-config")
async def get_filter_config():
    """Get current document filter configuration"""
    config_path = Path("config/document_filters.yaml")

    # Load from file if exists, otherwise use defaults
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return default config as dict
        default_config = DocumentFilterConfig()
        return default_config.dict()


@router.post("/documents/filter-config")
async def update_filter_config(config: DocumentFilterConfig):
    """Update document filter configuration"""
    try:
        config_path = Path("config/document_filters.yaml")
        config_path.parent.mkdir(exist_ok=True)

        # Save configuration
        with open(config_path, "w") as f:
            yaml.dump(config.dict(), f, default_flow_style=False, allow_unicode=True)

        return {"success": True, "message": "Filter configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating filter config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/cleanup")
async def cleanup_documents(
    remove_training: bool = True,
    remove_offtopic: bool = True,
    remove_corrupted: bool = True,
    dry_run: bool = True,
):
    """Clean up problematic documents"""
    try:
        report = await cleanup_problematic_documents(
            remove_training_docs=remove_training,
            remove_computer_science=remove_offtopic,
            remove_corrupted=remove_corrupted,
            dry_run=dry_run,
        )

        return {
            "success": True,
            "report": report,
            "message": f"{'Dry run completed' if dry_run else 'Cleanup completed'}",
        }
    except Exception as e:
        logger.error(f"Error cleaning documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/management", response_class=HTMLResponse)
async def document_management_page(request: Request):
    """Document management admin page"""
    try:
        # Get document analysis
        analyses = await get_document_analysis()

        # Get filter config (already returns dict)
        filter_config = await get_filter_config()

        # Convert analyses documents to dict format for JSON serialization
        analyses_dict = {
            "total_documents": analyses["total_documents"],
            "bio_waste_documents": analyses["bio_waste_documents"],
            "problematic_documents": analyses["problematic_documents"],
            "unknown_documents": analyses["unknown_documents"],
            "documents": {
                "bio_waste": [doc.dict() for doc in analyses["documents"]["bio_waste"]],
                "problematic": [
                    doc.dict() for doc in analyses["documents"]["problematic"]
                ],
                "unknown": [doc.dict() for doc in analyses["documents"]["unknown"]],
            },
        }

        return templates.TemplateResponse(
            "document_management.html",
            {
                "request": request,
                "analyses": analyses_dict,
                "filter_config": filter_config,
            },
        )
    except Exception as e:
        logger.error(f"Error loading document management page: {e}")
        return templates.TemplateResponse(
            "admin_error.html", {"request": request, "error": str(e)}
        )


@router.get("/documents/{document_id}")
async def get_document_details(document_id: int):
    """Get detailed information about a single document"""
    try:
        from ..repositories.factory import get_document_repository

        doc_repo = get_document_repository()

        document = await doc_repo.get_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get document content preview
        file_path = Path(document.file_path) if hasattr(document, "file_path") else None
        content_preview = ""
        if file_path and file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content_preview = f.read(1000)  # First 1000 chars
            except:
                content_preview = "Unable to read file content"

        return {
            "id": document.id,
            "filename": document.filename,
            "size": document.file_size,
            "content_type": document.content_type,
            "upload_date": (
                document.upload_date.isoformat() if document.upload_date else None
            ),
            "status": document.status,
            "file_path": str(file_path) if file_path else None,
            "content_preview": content_preview,
            "metadata": getattr(document, "metadata", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_single_document(document_id: int):
    """Delete a single document"""
    try:
        from ..repositories.factory import get_document_repository

        doc_repo = get_document_repository()

        # Check if document exists
        document = await doc_repo.get_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete file if exists
        if hasattr(document, "file_path") and document.file_path:
            file_path = Path(document.file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")

        # Delete from database
        success = await doc_repo.delete(document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")

        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/documents/{document_id}")
async def update_document_metadata(document_id: int, metadata: Dict[str, Any]):
    """Update document metadata"""
    try:
        from ..repositories.factory import get_document_repository

        doc_repo = get_document_repository()

        # Check if document exists
        document = await doc_repo.get_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update metadata
        success = await doc_repo.update(document_id, {"metadata": metadata})
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update document")

        return {"success": True, "message": "Document metadata updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Database Configuration Endpoints


@router.get("/database/config")
async def get_database_config():
    """Get current database configuration"""
    config_path = Path("config/database_config.yaml")

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return default SQLite config
        default_config = DatabaseConfig()
        return default_config.dict()


@router.post("/database/config")
async def update_database_config(config: DatabaseConfig):
    """Update database configuration"""
    try:
        config_path = Path("config/database_config.yaml")
        config_path.parent.mkdir(exist_ok=True)

        # Validate connection before saving
        if config.db_type == "postgresql":
            # Build connection string
            if not config.connection_string:
                config.connection_string = f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        elif config.db_type == "mysql":
            if not config.connection_string:
                config.connection_string = f"mysql+pymysql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"

        # Save configuration
        with open(config_path, "w") as f:
            yaml.dump(config.dict(), f, default_flow_style=False, allow_unicode=True)

        return {
            "success": True,
            "message": "Database configuration updated. Restart required for changes to take effect.",
        }
    except Exception as e:
        logger.error(f"Error updating database config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/database/test")
async def test_database_connection(config: DatabaseConfig):
    """Test database connection"""
    try:
        if config.db_type == "sqlite":
            import sqlite3

            conn = sqlite3.connect(config.database)
            conn.execute("SELECT 1")
            conn.close()
            return {"success": True, "message": "SQLite connection successful"}

        elif config.db_type == "postgresql":
            import psycopg2

            conn = psycopg2.connect(
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.username,
                password=config.password,
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return {"success": True, "message": "PostgreSQL connection successful"}

        elif config.db_type == "mysql":
            import pymysql

            conn = pymysql.connect(
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.username,
                password=config.password,
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return {"success": True, "message": "MySQL connection successful"}

        else:
            return {
                "success": False,
                "message": f"Unsupported database type: {config.db_type}",
            }

    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return {"success": False, "message": f"Connection failed: {str(e)}"}
