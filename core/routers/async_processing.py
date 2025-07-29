"""
Async Processing API Router
Provides endpoints for managing document processing queue
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..middleware.metrics_middleware import get_doc_metrics
from ..services.async_processing_service import (
    ProcessingTask,
    TaskPriority,
    TaskStatus,
    get_async_processor,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/processing", tags=["async-processing"])


# Request/Response Models
class TaskSubmissionRequest(BaseModel):
    tenant_id: str
    task_type: str
    file_path: str
    document_id: Optional[int] = None
    priority: str = "normal"  # low, normal, high, urgent
    metadata: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    id: str
    tenant_id: str
    document_id: Optional[int]
    task_type: str
    file_path: str
    priority: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: float
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    metadata: Optional[Dict[str, Any]]


class QueueStatsResponse(BaseModel):
    pending_tasks: int
    processing_tasks: int
    total_tasks: int
    workers: int
    workers_active: int
    status_breakdown: Dict[str, int]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    processing_time_total: float
    average_processing_time: float


def priority_from_string(priority_str: str) -> TaskPriority:
    """Convert string priority to TaskPriority enum"""
    priority_map = {
        "low": TaskPriority.LOW,
        "normal": TaskPriority.NORMAL,
        "high": TaskPriority.HIGH,
        "urgent": TaskPriority.URGENT,
    }
    return priority_map.get(priority_str.lower(), TaskPriority.NORMAL)


def task_to_response(task: ProcessingTask) -> TaskResponse:
    """Convert ProcessingTask to TaskResponse"""
    return TaskResponse(
        id=task.id,
        tenant_id=task.tenant_id,
        document_id=task.document_id,
        task_type=task.task_type,
        file_path=task.file_path,
        priority=task.priority.name.lower(),
        status=task.status.value,
        created_at=task.created_at.isoformat(),
        started_at=task.started_at.isoformat() if task.started_at else None,
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
        progress=task.progress,
        result=task.result,
        error_message=task.error_message,
        retry_count=task.retry_count,
        max_retries=task.max_retries,
        metadata=task.metadata,
    )


@router.post("/tasks", response_model=Dict[str, str])
async def submit_task(request: TaskSubmissionRequest):
    """
    Submit a new document processing task

    The task will be added to the processing queue and executed asynchronously.
    Returns the task ID for tracking progress.
    """
    try:
        processor = get_async_processor()

        # Convert priority string to enum
        priority = priority_from_string(request.priority)

        # Submit task
        task_id = await processor.add_task(
            tenant_id=request.tenant_id,
            task_type=request.task_type,
            file_path=request.file_path,
            document_id=request.document_id,
            priority=priority,
            metadata=request.metadata,
        )

        # Record metrics
        doc_metrics = get_doc_metrics()
        doc_metrics.record_processing(request.task_type, "queued", 0)

        logger.info(f"Task {task_id} submitted for tenant {request.tenant_id}")

        return {
            "task_id": task_id,
            "status": "queued",
            "message": f"Task submitted successfully. ID: {task_id}",
        }

    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a specific processing task

    Returns detailed information about the task including progress,
    status, and any results or error messages.
    """
    try:
        processor = get_async_processor()
        task = await processor.get_task_status(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return task_to_response(task)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get task status: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/tasks", response_model=List[TaskResponse])
async def get_tenant_tasks(tenant_id: str, status: Optional[str] = None):
    """
    Get all tasks for a specific tenant

    Optionally filter by task status (pending, processing, completed, failed, cancelled).
    """
    try:
        processor = get_async_processor()
        tasks = await processor.get_tasks_by_tenant(tenant_id)

        # Filter by status if provided
        if status:
            try:
                status_enum = TaskStatus(status.lower())
                tasks = [task for task in tasks if task.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        return [task_to_response(task) for task in tasks]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tenant tasks: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get tenant tasks: {str(e)}"
        )


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    Cancel a pending or processing task

    Tasks that are already completed or failed cannot be cancelled.
    """
    try:
        processor = get_async_processor()
        success = await processor.cancel_task(task_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Task cannot be cancelled (not found, already completed, or already cancelled)",
            )

        logger.info(f"Task {task_id} cancelled")

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancelled successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.post("/tasks/{task_id}/retry")
async def retry_task(task_id: str):
    """
    Retry a failed task

    Only failed tasks that haven't exceeded their retry limit can be retried.
    """
    try:
        processor = get_async_processor()
        success = await processor.retry_failed_task(task_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Task cannot be retried (not found, not failed, or exceeded retry limit)",
            )

        logger.info(f"Task {task_id} scheduled for retry")

        return {
            "task_id": task_id,
            "status": "retry_scheduled",
            "message": "Task scheduled for retry",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry task: {str(e)}")


@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """
    Get processing queue statistics

    Returns information about queue size, worker status, and processing statistics.
    """
    try:
        processor = get_async_processor()
        stats = await processor.get_queue_stats()

        return QueueStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get queue stats: {str(e)}"
        )


@router.post("/queue/cleanup")
async def cleanup_old_tasks(max_age_days: int = 30):
    """
    Clean up old completed/failed tasks

    Removes tasks older than the specified number of days to prevent
    the task history from growing indefinitely.
    """
    try:
        processor = get_async_processor()
        await processor.cleanup_old_tasks(max_age_days)

        return {
            "status": "success",
            "message": f"Cleaned up tasks older than {max_age_days} days",
        }

    except Exception as e:
        logger.error(f"Failed to cleanup tasks: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cleanup tasks: {str(e)}"
        )


@router.get("/health")
async def processing_health_check():
    """
    Health check for the async processing service

    Returns the current status of the processing service and basic statistics.
    """
    try:
        processor = get_async_processor()

        if not processor.is_running:
            raise HTTPException(
                status_code=503, detail="Processing service is not running"
            )

        stats = await processor.get_queue_stats()

        return {
            "status": "healthy",
            "service": "async_document_processor",
            "workers": stats["workers"],
            "workers_active": stats["workers_active"],
            "pending_tasks": stats["pending_tasks"],
            "processing_tasks": stats["processing_tasks"],
            "total_tasks": stats["total_tasks"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "async_document_processor",
            "error": str(e),
        }
