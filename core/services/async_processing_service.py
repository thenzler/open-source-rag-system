"""
Async Document Processing Service
Handles background document processing with queue management
"""
import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class ProcessingTask:
    """Document processing task"""
    id: str
    tenant_id: str
    document_id: Optional[int]
    task_type: str  # 'upload', 'reprocess', 'analyze', 'index'
    file_path: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'started_at', 'completed_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        # Convert enums to values
        data['priority'] = data['priority'].value
        data['status'] = data['status'].value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingTask':
        """Create task from dictionary"""
        # Convert datetime strings back to datetime objects
        for field in ['created_at', 'started_at', 'completed_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        # Convert enum values back to enums
        data['priority'] = TaskPriority(data['priority'])
        data['status'] = TaskStatus(data['status'])
        return cls(**data)

class AsyncDocumentProcessor:
    """Async document processing queue manager"""
    
    def __init__(
        self,
        max_workers: int = 4,
        queue_persistence_file: str = "data/processing_queue.json",
        enable_persistence: bool = True
    ):
        self.max_workers = max_workers
        self.queue_persistence_file = Path(queue_persistence_file)
        self.enable_persistence = enable_persistence
        
        # Task storage
        self.tasks: Dict[str, ProcessingTask] = {}
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'processing_time_total': 0.0,
            'average_processing_time': 0.0
        }
        
        # Task processors - will be registered by other services
        self.task_processors: Dict[str, Callable] = {}
        
        logger.info(f"AsyncDocumentProcessor initialized with {max_workers} workers")
    
    def register_processor(self, task_type: str, processor: Callable):
        """Register a processor function for a specific task type"""
        self.task_processors[task_type] = processor
        logger.info(f"Registered processor for task type: {task_type}")
    
    async def start(self):
        """Start the async processing service"""
        if self.is_running:
            logger.warning("AsyncDocumentProcessor is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Load persisted tasks
        if self.enable_persistence:
            await self._load_tasks_from_persistence()
        
        # Start worker tasks
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
        
        logger.info(f"AsyncDocumentProcessor started with {len(self.workers)} workers")
    
    async def stop(self):
        """Stop the async processing service"""
        if not self.is_running:
            return
        
        logger.info("Stopping AsyncDocumentProcessor...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.is_running = False
        
        # Cancel all processing tasks
        for task_id, task in self.processing_tasks.items():
            if not task.done():
                task.cancel()
                await self._update_task_status(task_id, TaskStatus.CANCELLED)
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Persist remaining tasks
        if self.enable_persistence:
            await self._persist_tasks()
        
        logger.info("AsyncDocumentProcessor stopped")
    
    async def add_task(
        self,
        tenant_id: str,
        task_type: str,
        file_path: str,
        document_id: Optional[int] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new processing task to the queue"""
        
        task_id = str(uuid.uuid4())
        task = ProcessingTask(
            id=task_id,
            tenant_id=tenant_id,
            document_id=document_id,
            task_type=task_type,
            file_path=file_path,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to queue (priority queue uses tuple: (priority, task_id))
        # Lower priority values have higher priority
        await self.pending_queue.put((5 - priority.value, task_id))
        
        self.stats['total_tasks'] += 1
        
        logger.info(f"Added task {task_id} ({task_type}) for tenant {tenant_id}")
        
        # Persist if enabled
        if self.enable_persistence:
            await self._persist_tasks()
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get the status of a specific task"""
        return self.tasks.get(task_id)
    
    async def get_tasks_by_tenant(self, tenant_id: str) -> List[ProcessingTask]:
        """Get all tasks for a specific tenant"""
        return [
            task for task in self.tasks.values()
            if task.tenant_id == tenant_id
        ]
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        pending_count = self.pending_queue.qsize()
        processing_count = len(self.processing_tasks)
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for task in self.tasks.values()
                if task.status == status
            )
        
        return {
            'pending_tasks': pending_count,
            'processing_tasks': processing_count,
            'total_tasks': len(self.tasks),
            'workers': len(self.workers),
            'workers_active': len(self.processing_tasks),
            'status_breakdown': status_counts,
            **self.stats
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or processing task"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            await self._update_task_status(task_id, TaskStatus.CANCELLED)
            self.stats['cancelled_tasks'] += 1
            return True
        
        elif task.status == TaskStatus.PROCESSING:
            # Cancel the processing task
            processing_task = self.processing_tasks.get(task_id)
            if processing_task:
                processing_task.cancel()
                await self._update_task_status(task_id, TaskStatus.CANCELLED)
                self.stats['cancelled_tasks'] += 1
                return True
        
        return False
    
    async def retry_failed_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.FAILED:
            return False
        
        if task.retry_count >= task.max_retries:
            logger.warning(f"Task {task_id} has exceeded max retries")
            return False
        
        # Reset task for retry
        task.status = TaskStatus.PENDING
        task.retry_count += 1
        task.started_at = None
        task.completed_at = None
        task.error_message = None
        
        # Add back to queue
        await self.pending_queue.put((5 - task.priority.value, task_id))
        
        logger.info(f"Retrying task {task_id} (attempt {task.retry_count}/{task.max_retries})")
        return True
    
    async def _worker(self, worker_name: str):
        """Worker coroutine that processes tasks from the queue"""
        logger.info(f"Worker {worker_name} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next task with timeout
                try:
                    priority, task_id = await asyncio.wait_for(
                        self.pending_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue
                
                # Start processing
                await self._process_task(worker_name, task)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                continue
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _process_task(self, worker_name: str, task: ProcessingTask):
        """Process a single task"""
        task_id = task.id
        
        try:
            # Update task status
            await self._update_task_status(task_id, TaskStatus.PROCESSING)
            task.started_at = datetime.now()
            
            logger.info(f"Worker {worker_name} processing task {task_id} ({task.task_type})")
            
            # Get processor for task type
            processor = self.task_processors.get(task.task_type)
            if not processor:
                raise ValueError(f"No processor registered for task type: {task.task_type}")
            
            # Create processing task
            processing_task = asyncio.create_task(
                self._run_processor(processor, task)
            )
            self.processing_tasks[task_id] = processing_task
            
            # Wait for completion
            result = await processing_task
            
            # Update task with result
            task.result = result
            task.completed_at = datetime.now()
            task.progress = 100.0
            
            await self._update_task_status(task_id, TaskStatus.COMPLETED)
            
            # Update statistics
            self.stats['completed_tasks'] += 1
            processing_time = (task.completed_at - task.started_at).total_seconds()
            self.stats['processing_time_total'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['processing_time_total'] / self.stats['completed_tasks']
            )
            
            logger.info(f"Task {task_id} completed in {processing_time:.2f}s")
            
        except asyncio.CancelledError:
            await self._update_task_status(task_id, TaskStatus.CANCELLED)
            logger.info(f"Task {task_id} cancelled")
            
        except Exception as e:
            # Update task with error
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            await self._update_task_status(task_id, TaskStatus.FAILED)
            self.stats['failed_tasks'] += 1
            
            logger.error(f"Task {task_id} failed: {e}")
            
            # Auto-retry if under limit
            if task.retry_count < task.max_retries:
                logger.info(f"Auto-retrying task {task_id} in 30 seconds")
                asyncio.create_task(self._schedule_retry(task_id, 30))
        
        finally:
            # Clean up processing task
            self.processing_tasks.pop(task_id, None)
            
            # Persist state
            if self.enable_persistence:
                asyncio.create_task(self._persist_tasks())
    
    async def _run_processor(self, processor: Callable, task: ProcessingTask) -> Any:
        """Run the processor function with progress tracking"""
        
        # Create progress callback
        async def update_progress(progress: float):
            task.progress = min(100.0, max(0.0, progress))
        
        # Call processor with task and progress callback
        if asyncio.iscoroutinefunction(processor):
            return await processor(task, update_progress)
        else:
            # Run sync processor in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, processor, task, update_progress)
    
    async def _schedule_retry(self, task_id: str, delay_seconds: int):
        """Schedule a task retry after delay"""
        await asyncio.sleep(delay_seconds)
        await self.retry_failed_task(task_id)
    
    async def _update_task_status(self, task_id: str, status: TaskStatus):
        """Update task status"""
        task = self.tasks.get(task_id)
        if task:
            task.status = status
    
    async def _persist_tasks(self):
        """Persist tasks to file"""
        if not self.enable_persistence:
            return
        
        try:
            # Ensure directory exists
            self.queue_persistence_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert tasks to serializable format
            tasks_data = {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            }
            
            # Write to file
            with open(self.queue_persistence_file, 'w') as f:
                json.dump({
                    'tasks': tasks_data,
                    'stats': self.stats,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to persist tasks: {e}")
    
    async def _load_tasks_from_persistence(self):
        """Load tasks from persistence file"""
        if not self.queue_persistence_file.exists():
            return
        
        try:
            with open(self.queue_persistence_file, 'r') as f:
                data = json.load(f)
            
            # Load tasks
            for task_id, task_data in data.get('tasks', {}).items():
                try:
                    task = ProcessingTask.from_dict(task_data)
                    self.tasks[task_id] = task
                    
                    # Re-queue pending tasks
                    if task.status == TaskStatus.PENDING:
                        await self.pending_queue.put((5 - task.priority.value, task_id))
                    
                    # Reset processing tasks to pending (they were interrupted)
                    elif task.status == TaskStatus.PROCESSING:
                        task.status = TaskStatus.PENDING
                        await self.pending_queue.put((5 - task.priority.value, task_id))
                    
                except Exception as e:
                    logger.error(f"Failed to load task {task_id}: {e}")
            
            # Load stats
            self.stats.update(data.get('stats', {}))
            
            logger.info(f"Loaded {len(self.tasks)} tasks from persistence")
            
        except Exception as e:
            logger.error(f"Failed to load tasks from persistence: {e}")
    
    async def cleanup_old_tasks(self, max_age_days: int = 30):
        """Clean up old completed/failed tasks"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.completed_at and task.completed_at < cutoff_date):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
        
        if self.enable_persistence:
            await self._persist_tasks()

# Global processor instance
_processor: Optional[AsyncDocumentProcessor] = None

def get_async_processor() -> AsyncDocumentProcessor:
    """Get global async document processor"""
    global _processor
    if _processor is None:
        _processor = AsyncDocumentProcessor()
    return _processor

async def initialize_async_processor(
    max_workers: int = 4,
    enable_persistence: bool = True
) -> AsyncDocumentProcessor:
    """Initialize and start the global async processor"""
    global _processor
    _processor = AsyncDocumentProcessor(
        max_workers=max_workers,
        enable_persistence=enable_persistence
    )
    await _processor.start()
    return _processor

async def shutdown_async_processor():
    """Shutdown the global async processor"""
    global _processor
    if _processor:
        await _processor.stop()
        _processor = None