#!/usr/bin/env python3
"""
Async Document Processing Service
Provides background document processing to prevent API blocking
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingJob:
    """Represents a document processing job"""
    job_id: str
    filename: str
    file_path: str
    content_type: str
    file_size: int
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    document_id: Optional[int] = None
    chunks_created: int = 0
    processing_time: float = 0.0

class AsyncDocumentProcessor:
    """
    Async document processor with queue management and progress tracking
    Processes documents in background without blocking API responses
    """
    
    def __init__(self, max_workers: int = 3, max_queue_size: int = 100):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.jobs: Dict[str, ProcessingJob] = {}
        self.processing_queue = asyncio.Queue(maxsize=max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.workers_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self._lock = threading.Lock()
        
        # Callbacks for processing steps
        self.extract_text_callback: Optional[Callable] = None
        self.chunk_text_callback: Optional[Callable] = None
        self.create_embeddings_callback: Optional[Callable] = None
        self.store_document_callback: Optional[Callable] = None
        
        logger.info(f"AsyncDocumentProcessor initialized with {max_workers} workers")
    
    def set_callbacks(self, extract_text_fn, chunk_text_fn, create_embeddings_fn, store_document_fn):
        """Set the processing callback functions"""
        self.extract_text_callback = extract_text_fn
        self.chunk_text_callback = chunk_text_fn
        self.create_embeddings_callback = create_embeddings_fn
        self.store_document_callback = store_document_fn
    
    async def start_workers(self):
        """Start the background worker tasks"""
        if self.workers_running:
            return
        
        self.workers_running = True
        self.worker_tasks = []
        
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Started {len(self.worker_tasks)} async processing workers")
    
    async def stop_workers(self):
        """Stop all background workers gracefully"""
        if not self.workers_running:
            return
        
        self.workers_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("Stopped all async processing workers")
    
    async def queue_document(self, filename: str, file_path: str, content_type: str, 
                           file_size: int) -> str:
        """
        Queue a document for async processing
        Returns job_id for tracking
        """
        # Check queue capacity
        if self.processing_queue.qsize() >= self.max_queue_size:
            raise Exception(f"Processing queue is full ({self.max_queue_size} jobs)")
        
        # Create job
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            filename=filename,
            file_path=file_path,
            content_type=content_type,
            file_size=file_size,
            status=ProcessingStatus.QUEUED,
            created_at=datetime.now()
        )
        
        # Store job
        with self._lock:
            self.jobs[job_id] = job
        
        # Add to queue
        await self.processing_queue.put(job)
        
        logger.info(f"Queued document for processing: {filename} (job_id: {job_id})")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get the current status of a processing job"""
        with self._lock:
            return self.jobs.get(job_id)
    
    async def get_all_jobs(self, limit: int = 50) -> List[ProcessingJob]:
        """Get all processing jobs (most recent first)"""
        with self._lock:
            jobs = list(self.jobs.values())
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            return jobs[:limit]
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or processing job"""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            if job.status in [ProcessingStatus.QUEUED, ProcessingStatus.PROCESSING]:
                job.status = ProcessingStatus.CANCELLED
                job.completed_at = datetime.now()
                logger.info(f"Cancelled job: {job_id}")
                return True
            
            return False
    
    def get_queue_stats(self) -> Dict:
        """Get queue and processing statistics"""
        with self._lock:
            job_counts = {}
            for status in ProcessingStatus:
                job_counts[status.value] = sum(1 for job in self.jobs.values() if job.status == status)
        
        return {
            "queue_size": self.processing_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_workers": len(self.worker_tasks),
            "max_workers": self.max_workers,
            "total_jobs": len(self.jobs),
            "job_counts": job_counts,
            "workers_running": self.workers_running
        }
    
    async def _worker(self, worker_name: str):
        """Background worker that processes jobs from the queue"""
        logger.info(f"Started async worker: {worker_name}")
        
        while self.workers_running:
            try:
                # Get next job from queue (with timeout to allow graceful shutdown)
                try:
                    job = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check if job was cancelled
                if job.status == ProcessingStatus.CANCELLED:
                    continue
                
                # Process the job
                await self._process_job(job, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
                continue
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _process_job(self, job: ProcessingJob, worker_name: str):
        """Process a single document job"""
        job_start_time = time.time()
        
        try:
            # Update job status
            with self._lock:
                job.status = ProcessingStatus.PROCESSING
                job.started_at = datetime.now()
                job.progress = 0.1
            
            logger.info(f"[{worker_name}] Processing: {job.filename} (job_id: {job.job_id})")
            
            # Step 1: Extract text
            if not self.extract_text_callback:
                raise Exception("Text extraction callback not set")
            
            loop = asyncio.get_event_loop()
            extracted_text = await loop.run_in_executor(
                self.executor,
                self.extract_text_callback,
                job.file_path,
                job.content_type
            )
            
            with self._lock:
                job.progress = 0.3
            
            # Step 2: Chunk text
            if not self.chunk_text_callback:
                raise Exception("Text chunking callback not set")
            
            chunks = await loop.run_in_executor(
                self.executor,
                self.chunk_text_callback,
                extracted_text
            )
            
            with self._lock:
                job.progress = 0.5
                job.chunks_created = len(chunks)
            
            # Step 3: Create embeddings
            if not self.create_embeddings_callback:
                raise Exception("Embedding creation callback not set")
            
            embeddings = await loop.run_in_executor(
                self.executor,
                self.create_embeddings_callback,
                chunks
            )
            
            with self._lock:
                job.progress = 0.8
            
            # Step 4: Store document
            if not self.store_document_callback:
                raise Exception("Document storage callback not set")
            
            document_id = await loop.run_in_executor(
                self.executor,
                self.store_document_callback,
                job.filename,
                job.content_type,
                job.file_size,
                extracted_text,
                chunks,
                embeddings
            )
            
            # Job completed successfully
            processing_time = time.time() - job_start_time
            
            with self._lock:
                job.status = ProcessingStatus.COMPLETED
                job.completed_at = datetime.now()
                job.progress = 1.0
                job.document_id = document_id
                job.processing_time = processing_time
            
            logger.info(f"[{worker_name}] Completed: {job.filename} "
                       f"(document_id: {document_id}, chunks: {len(chunks)}, "
                       f"time: {processing_time:.1f}s)")
        
        except Exception as e:
            # Job failed
            processing_time = time.time() - job_start_time
            
            with self._lock:
                job.status = ProcessingStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = str(e)
                job.processing_time = processing_time
            
            logger.error(f"[{worker_name}] Failed: {job.filename} - {str(e)}")

# Global instance (will be initialized in simple_api.py)
async_processor: Optional[AsyncDocumentProcessor] = None

def get_async_processor() -> AsyncDocumentProcessor:
    """Get the global async processor instance"""
    global async_processor
    if async_processor is None:
        async_processor = AsyncDocumentProcessor()
    return async_processor

async def init_async_processor(extract_text_fn, chunk_text_fn, create_embeddings_fn, store_document_fn):
    """Initialize the async processor with callback functions"""
    global async_processor
    async_processor = AsyncDocumentProcessor()
    async_processor.set_callbacks(extract_text_fn, chunk_text_fn, create_embeddings_fn, store_document_fn)
    await async_processor.start_workers()
    return async_processor

async def shutdown_async_processor():
    """Shutdown the async processor gracefully"""
    global async_processor
    if async_processor:
        await async_processor.stop_workers()