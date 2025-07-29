#!/usr/bin/env python3
"""
Progress Tracking Service for Long Operations
Provides real-time progress updates via WebSocket connections
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ProgressStatus(Enum):
    """Progress status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressStep:
    """Individual progress step"""

    id: str
    name: str
    description: str
    status: ProgressStatus = ProgressStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProgressOperation:
    """Complete progress operation containing multiple steps"""

    id: str
    name: str
    description: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    status: ProgressStatus = ProgressStatus.PENDING
    current_step: int = 0
    total_steps: int = 0
    overall_progress: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    estimated_completion: Optional[float] = None
    steps: List[ProgressStep] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.metadata is None:
            self.metadata = {}


class ProgressTracker:
    """Progress tracking with WebSocket notifications"""

    def __init__(self, persistence_file: str = "data/progress_operations.json"):
        self.operations: Dict[str, ProgressOperation] = {}
        self.websocket_connections: Dict[str, Set] = (
            {}
        )  # operation_id -> set of websockets
        self.operation_callbacks: Dict[str, List[Callable]] = (
            {}
        )  # operation_id -> callbacks
        self.persistence_file = Path(persistence_file)
        self._lock = asyncio.Lock()

        # Ensure data directory exists
        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)

        # Load persisted operations
        self._load_operations()

    def _load_operations(self):
        """Load operations from persistence file"""
        try:
            if self.persistence_file.exists():
                with open(self.persistence_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for op_data in data.get("operations", []):
                        operation = self._dict_to_operation(op_data)
                        self.operations[operation.id] = operation
                logger.info(f"Loaded {len(self.operations)} persisted operations")
        except Exception as e:
            logger.error(f"Failed to load persisted operations: {e}")

    def _save_operations(self):
        """Save operations to persistence file"""
        try:
            data = {
                "operations": [asdict(op) for op in self.operations.values()],
                "last_updated": time.time(),
            }
            with open(self.persistence_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save operations: {e}")

    def _dict_to_operation(self, data: Dict) -> ProgressOperation:
        """Convert dictionary to ProgressOperation"""
        steps_data = data.get("steps", [])
        steps = []
        for step_data in steps_data:
            step = ProgressStep(
                id=step_data["id"],
                name=step_data["name"],
                description=step_data["description"],
                status=ProgressStatus(step_data["status"]),
                progress=step_data["progress"],
                started_at=step_data.get("started_at"),
                completed_at=step_data.get("completed_at"),
                error=step_data.get("error"),
                metadata=step_data.get("metadata", {}),
            )
            steps.append(step)

        return ProgressOperation(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            tenant_id=data.get("tenant_id"),
            user_id=data.get("user_id"),
            status=ProgressStatus(data["status"]),
            current_step=data["current_step"],
            total_steps=data["total_steps"],
            overall_progress=data["overall_progress"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            estimated_completion=data.get("estimated_completion"),
            steps=steps,
            metadata=data.get("metadata", {}),
        )

    async def create_operation(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, str]],
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new progress operation"""
        async with self._lock:
            operation_id = str(uuid.uuid4())

            # Create progress steps
            progress_steps = []
            for i, step_info in enumerate(steps):
                step = ProgressStep(
                    id=f"{operation_id}-step-{i}",
                    name=step_info["name"],
                    description=step_info["description"],
                    metadata=step_info.get("metadata", {}),
                )
                progress_steps.append(step)

            # Create operation
            operation = ProgressOperation(
                id=operation_id,
                name=name,
                description=description,
                tenant_id=tenant_id,
                user_id=user_id,
                total_steps=len(progress_steps),
                steps=progress_steps,
                metadata=metadata or {},
            )

            self.operations[operation_id] = operation
            self.websocket_connections[operation_id] = set()
            self.operation_callbacks[operation_id] = []

            self._save_operations()

            logger.info(f"Created progress operation: {operation_id} - {name}")
            return operation_id

    async def start_operation(self, operation_id: str) -> bool:
        """Start a progress operation"""
        async with self._lock:
            if operation_id not in self.operations:
                return False

            operation = self.operations[operation_id]
            operation.status = ProgressStatus.RUNNING
            operation.started_at = time.time()

            await self._notify_progress_update(operation_id)
            self._save_operations()

            logger.info(f"Started operation: {operation_id}")
            return True

    async def update_step_progress(
        self,
        operation_id: str,
        step_index: int,
        progress: float,
        status: Optional[ProgressStatus] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update progress for a specific step"""
        async with self._lock:
            if operation_id not in self.operations:
                return False

            operation = self.operations[operation_id]
            if step_index >= len(operation.steps):
                return False

            step = operation.steps[step_index]

            # Update step
            step.progress = max(0.0, min(1.0, progress))
            if status:
                step.status = status
                if status == ProgressStatus.RUNNING and not step.started_at:
                    step.started_at = time.time()
                elif status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED]:
                    step.completed_at = time.time()

            if error:
                step.error = error
                step.status = ProgressStatus.FAILED

            if metadata:
                step.metadata.update(metadata)

            # Update operation overall progress
            operation.current_step = step_index
            total_progress = sum(s.progress for s in operation.steps)
            operation.overall_progress = (
                total_progress / operation.total_steps
                if operation.total_steps > 0
                else 0.0
            )

            # Check if operation is complete
            if all(s.status == ProgressStatus.COMPLETED for s in operation.steps):
                operation.status = ProgressStatus.COMPLETED
                operation.completed_at = time.time()
                operation.overall_progress = 1.0
            elif any(s.status == ProgressStatus.FAILED for s in operation.steps):
                operation.status = ProgressStatus.FAILED
                operation.completed_at = time.time()

            # Estimate completion time
            if operation.status == ProgressStatus.RUNNING and operation.started_at:
                elapsed = time.time() - operation.started_at
                if operation.overall_progress > 0:
                    estimated_total = elapsed / operation.overall_progress
                    operation.estimated_completion = (
                        operation.started_at + estimated_total
                    )

            await self._notify_progress_update(operation_id)
            self._save_operations()

            return True

    async def complete_step(
        self,
        operation_id: str,
        step_index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a step as completed"""
        return await self.update_step_progress(
            operation_id, step_index, 1.0, ProgressStatus.COMPLETED, metadata=metadata
        )

    async def fail_step(
        self,
        operation_id: str,
        step_index: int,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a step as failed"""
        return await self.update_step_progress(
            operation_id,
            step_index,
            None,
            ProgressStatus.FAILED,
            error=error,
            metadata=metadata,
        )

    async def cancel_operation(
        self, operation_id: str, reason: str = "Cancelled by user"
    ) -> bool:
        """Cancel a progress operation"""
        async with self._lock:
            if operation_id not in self.operations:
                return False

            operation = self.operations[operation_id]
            operation.status = ProgressStatus.CANCELLED
            operation.completed_at = time.time()
            operation.metadata["cancellation_reason"] = reason

            # Cancel all pending/running steps
            for step in operation.steps:
                if step.status in [ProgressStatus.PENDING, ProgressStatus.RUNNING]:
                    step.status = ProgressStatus.CANCELLED
                    step.completed_at = time.time()
                    step.error = reason

            await self._notify_progress_update(operation_id)
            self._save_operations()

            logger.info(f"Cancelled operation: {operation_id} - {reason}")
            return True

    def get_operation(self, operation_id: str) -> Optional[ProgressOperation]:
        """Get operation by ID"""
        return self.operations.get(operation_id)

    def list_operations(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[ProgressStatus] = None,
    ) -> List[ProgressOperation]:
        """List operations with optional filtering"""
        operations = list(self.operations.values())

        if tenant_id:
            operations = [op for op in operations if op.tenant_id == tenant_id]
        if user_id:
            operations = [op for op in operations if op.user_id == user_id]
        if status:
            operations = [op for op in operations if op.status == status]

        # Sort by creation time (newest first)
        operations.sort(key=lambda op: op.started_at or 0, reverse=True)
        return operations

    async def cleanup_old_operations(self, max_age_hours: int = 168):  # 7 days default
        """Clean up old completed/failed/cancelled operations"""
        async with self._lock:
            cutoff_time = time.time() - (max_age_hours * 3600)
            to_remove = []

            for operation_id, operation in self.operations.items():
                if (
                    operation.status
                    in [
                        ProgressStatus.COMPLETED,
                        ProgressStatus.FAILED,
                        ProgressStatus.CANCELLED,
                    ]
                    and operation.completed_at
                    and operation.completed_at < cutoff_time
                ):
                    to_remove.append(operation_id)

            for operation_id in to_remove:
                del self.operations[operation_id]
                if operation_id in self.websocket_connections:
                    del self.websocket_connections[operation_id]
                if operation_id in self.operation_callbacks:
                    del self.operation_callbacks[operation_id]

            if to_remove:
                self._save_operations()
                logger.info(f"Cleaned up {len(to_remove)} old operations")

    async def add_websocket_connection(self, operation_id: str, websocket):
        """Add WebSocket connection for operation updates"""
        if operation_id in self.websocket_connections:
            self.websocket_connections[operation_id].add(websocket)
            logger.debug(f"Added WebSocket connection for operation: {operation_id}")

    async def remove_websocket_connection(self, operation_id: str, websocket):
        """Remove WebSocket connection"""
        if operation_id in self.websocket_connections:
            self.websocket_connections[operation_id].discard(websocket)
            logger.debug(f"Removed WebSocket connection for operation: {operation_id}")

    def add_callback(self, operation_id: str, callback: Callable):
        """Add callback for operation updates"""
        if operation_id in self.operation_callbacks:
            self.operation_callbacks[operation_id].append(callback)

    async def _notify_progress_update(self, operation_id: str):
        """Notify all listeners of progress update"""
        if operation_id not in self.operations:
            return

        operation = self.operations[operation_id]
        update_data = {
            "type": "progress_update",
            "operation_id": operation_id,
            "operation": asdict(operation),
            "timestamp": time.time(),
        }

        # Notify WebSocket connections
        if operation_id in self.websocket_connections:
            disconnected_websockets = set()
            for websocket in self.websocket_connections[operation_id]:
                try:
                    await websocket.send_text(json.dumps(update_data, default=str))
                except Exception as e:
                    logger.debug(f"WebSocket send failed: {e}")
                    disconnected_websockets.add(websocket)

            # Remove disconnected websockets
            for websocket in disconnected_websockets:
                self.websocket_connections[operation_id].discard(websocket)

        # Notify callbacks
        if operation_id in self.operation_callbacks:
            for callback in self.operation_callbacks[operation_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(operation)
                    else:
                        callback(operation)
                except Exception as e:
                    logger.error(f"Callback error: {e}")


# Global progress tracker instance
_progress_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get global progress tracker instance"""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker


async def initialize_progress_tracker(
    persistence_file: str = "data/progress_operations.json",
) -> ProgressTracker:
    """Initialize global progress tracker"""
    global _progress_tracker
    _progress_tracker = ProgressTracker(persistence_file)

    # Start cleanup task
    asyncio.create_task(_cleanup_task())

    logger.info("Progress tracking service initialized")
    return _progress_tracker


async def _cleanup_task():
    """Background task to clean up old operations"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            tracker = get_progress_tracker()
            await tracker.cleanup_old_operations()
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")


# Context manager for easy progress tracking
class ProgressContext:
    """Context manager for tracking operation progress"""

    def __init__(
        self,
        operation_id: str,
        step_index: int,
        tracker: Optional[ProgressTracker] = None,
    ):
        self.operation_id = operation_id
        self.step_index = step_index
        self.tracker = tracker or get_progress_tracker()

    async def __aenter__(self):
        await self.tracker.update_step_progress(
            self.operation_id, self.step_index, 0.0, ProgressStatus.RUNNING
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.tracker.fail_step(
                self.operation_id, self.step_index, str(exc_val)
            )
            return False
        else:
            await self.tracker.complete_step(self.operation_id, self.step_index)
            return True

    async def update_progress(
        self, progress: float, metadata: Optional[Dict[str, Any]] = None
    ):
        """Update progress within the context"""
        await self.tracker.update_step_progress(
            self.operation_id, self.step_index, progress, metadata=metadata
        )
