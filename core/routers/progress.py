#!/usr/bin/env python3
"""
Progress Tracking API Router
WebSocket and REST endpoints for real-time progress tracking
"""
import json
import logging
from typing import Optional, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

try:
    from ..services.progress_tracking_service import (
        get_progress_tracker, 
        ProgressTracker, 
        ProgressStatus,
        ProgressOperation
    )
    PROGRESS_TRACKING_AVAILABLE = True
except ImportError:
    # Fallback when progress tracking is not available
    PROGRESS_TRACKING_AVAILABLE = False
    def get_progress_tracker():
        return None
    
    class ProgressStatus:
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    class ProgressTracker:
        pass
    
    class ProgressOperation:
        pass
from ..middleware.tenant_middleware import get_current_tenant
from ..utils.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/progress", tags=["progress"])

def get_tracker() -> ProgressTracker:
    """Dependency to get progress tracker"""
    return get_progress_tracker()

@router.websocket("/ws/{operation_id}")
async def websocket_progress_updates(
    websocket: WebSocket, 
    operation_id: str,
    tracker: ProgressTracker = Depends(get_tracker)
):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    
    try:
        # Add websocket to operation listeners
        await tracker.add_websocket_connection(operation_id, websocket)
        
        # Send current operation state immediately
        operation = tracker.get_operation(operation_id)
        if operation:
            await websocket.send_text(json.dumps({
                'type': 'current_state',
                'operation_id': operation_id,
                'operation': operation.__dict__,
                'timestamp': operation.started_at
            }, default=str))
        else:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': f'Operation {operation_id} not found'
            }))
            await websocket.close()
            return
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message (could be ping/pong)
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get('type') == 'ping':
                    await websocket.send_text(json.dumps({
                        'type': 'pong',
                        'timestamp': operation.started_at
                    }))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                # Ignore invalid JSON
                continue
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Remove websocket from listeners
        await tracker.remove_websocket_connection(operation_id, websocket)

@router.post("/operations", response_model=dict)
async def create_operation(
    name: str,
    description: str,
    steps: List[dict],
    tenant_id: Optional[str] = Depends(get_current_tenant),
    user_id: Optional[str] = Depends(get_current_user),
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Create a new progress operation"""
    try:
        operation_id = await tracker.create_operation(
            name=name,
            description=description,
            steps=steps,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return {
            'operation_id': operation_id,
            'message': 'Operation created successfully',
            'websocket_url': f'/api/v1/progress/ws/{operation_id}'
        }
        
    except Exception as e:
        logger.error(f"Failed to create operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/operations/{operation_id}", response_model=dict)
async def get_operation(
    operation_id: str,
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Get operation details"""
    operation = tracker.get_operation(operation_id)
    if not operation:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    return {
        'operation': operation.__dict__,
        'websocket_url': f'/api/v1/progress/ws/{operation_id}'
    }

@router.get("/operations", response_model=dict)
async def list_operations(
    tenant_id: Optional[str] = Depends(get_current_tenant),
    user_id: Optional[str] = Depends(get_current_user),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Limit number of results"),
    tracker: ProgressTracker = Depends(get_tracker)
):
    """List operations with optional filtering"""
    try:
        status_filter = None
        if status:
            try:
                status_filter = ProgressStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        operations = tracker.list_operations(
            tenant_id=tenant_id,
            user_id=user_id,
            status=status_filter
        )
        
        # Apply limit
        operations = operations[:limit]
        
        return {
            'operations': [op.__dict__ for op in operations],
            'total': len(operations),
            'filters': {
                'tenant_id': tenant_id,
                'user_id': user_id,
                'status': status
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/operations/{operation_id}/start", response_model=dict)
async def start_operation(
    operation_id: str,
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Start a progress operation"""
    success = await tracker.start_operation(operation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    return {
        'message': 'Operation started successfully',
        'operation_id': operation_id,
        'websocket_url': f'/api/v1/progress/ws/{operation_id}'
    }

@router.post("/operations/{operation_id}/cancel", response_model=dict)
async def cancel_operation(
    operation_id: str,
    reason: str = "Cancelled by user",
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Cancel a progress operation"""
    success = await tracker.cancel_operation(operation_id, reason)
    if not success:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    return {
        'message': 'Operation cancelled successfully',
        'operation_id': operation_id,
        'reason': reason
    }

@router.put("/operations/{operation_id}/steps/{step_index}", response_model=dict)
async def update_step_progress(
    operation_id: str,
    step_index: int,
    progress: float,
    status: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Update progress for a specific step"""
    try:
        status_enum = None
        if status:
            try:
                status_enum = ProgressStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        success = await tracker.update_step_progress(
            operation_id=operation_id,
            step_index=step_index,
            progress=progress,
            status=status_enum,
            error=error,
            metadata=metadata
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Operation or step not found")
        
        return {
            'message': 'Step progress updated successfully',
            'operation_id': operation_id,
            'step_index': step_index,
            'progress': progress
        }
        
    except Exception as e:
        logger.error(f"Failed to update step progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/operations/{operation_id}/steps/{step_index}/complete", response_model=dict)
async def complete_step(
    operation_id: str,
    step_index: int,
    metadata: Optional[dict] = None,
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Mark a step as completed"""
    success = await tracker.complete_step(operation_id, step_index, metadata)
    if not success:
        raise HTTPException(status_code=404, detail="Operation or step not found")
    
    return {
        'message': 'Step completed successfully',
        'operation_id': operation_id,
        'step_index': step_index
    }

@router.post("/operations/{operation_id}/steps/{step_index}/fail", response_model=dict)
async def fail_step(
    operation_id: str,
    step_index: int,
    error: str,
    metadata: Optional[dict] = None,
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Mark a step as failed"""
    success = await tracker.fail_step(operation_id, step_index, error, metadata)
    if not success:
        raise HTTPException(status_code=404, detail="Operation or step not found")
    
    return {
        'message': 'Step marked as failed',
        'operation_id': operation_id,
        'step_index': step_index,
        'error': error
    }

@router.delete("/operations/cleanup", response_model=dict)
async def cleanup_old_operations(
    max_age_hours: int = Query(168, description="Maximum age in hours (default: 7 days)"),
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Clean up old completed/failed/cancelled operations"""
    try:
        await tracker.cleanup_old_operations(max_age_hours)
        return {
            'message': f'Cleaned up operations older than {max_age_hours} hours',
            'max_age_hours': max_age_hours
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=dict)
async def progress_health_check(tracker: ProgressTracker = Depends(get_tracker)):
    """Health check for progress tracking service"""
    try:
        total_operations = len(tracker.operations)
        active_operations = len([
            op for op in tracker.operations.values() 
            if op.status == ProgressStatus.RUNNING
        ])
        
        return {
            'status': 'healthy',
            'total_operations': total_operations,
            'active_operations': active_operations,
            'service': 'progress_tracking'
        }
        
    except Exception as e:
        logger.error(f"Progress health check failed: {e}")
        raise HTTPException(status_code=503, detail="Progress tracking service unavailable")

@router.get("/stats", response_model=dict)
async def progress_statistics(
    tracker: ProgressTracker = Depends(get_tracker)
):
    """Get progress tracking statistics"""
    try:
        operations = list(tracker.operations.values())
        
        stats = {
            'total_operations': len(operations),
            'by_status': {},
            'active_websockets': sum(len(conns) for conns in tracker.websocket_connections.values()),
            'operations_with_listeners': len([
                op_id for op_id, conns in tracker.websocket_connections.items() 
                if len(conns) > 0
            ])
        }
        
        # Count by status
        for status in ProgressStatus:
            count = len([op for op in operations if op.status == status])
            stats['by_status'][status.value] = count
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get progress statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))