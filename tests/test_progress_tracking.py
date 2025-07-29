#!/usr/bin/env python3
"""
Tests for Progress Tracking Service
"""
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from core.services.progress_tracking_service import (
        ProgressContext,
        ProgressOperation,
        ProgressStatus,
        ProgressStep,
        ProgressTracker,
        get_progress_tracker,
        initialize_progress_tracker,
    )

    PROGRESS_TRACKING_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKING_AVAILABLE = False
    pytest.skip("Progress tracking service not available", allow_module_level=True)


@pytest.fixture
async def temp_tracker():
    """Create a temporary progress tracker"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tracker = ProgressTracker(persistence_file=tmp.name)
        yield tracker
        # Cleanup
        Path(tmp.name).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_create_operation(temp_tracker):
    """Test creating a progress operation"""
    steps = [
        {"name": "Step 1", "description": "First step"},
        {"name": "Step 2", "description": "Second step"},
        {"name": "Step 3", "description": "Third step"},
    ]

    operation_id = await temp_tracker.create_operation(
        name="Test Operation",
        description="Test operation description",
        steps=steps,
        tenant_id="test_tenant",
        user_id="test_user",
    )

    assert operation_id is not None
    assert operation_id in temp_tracker.operations

    operation = temp_tracker.get_operation(operation_id)
    assert operation.name == "Test Operation"
    assert operation.tenant_id == "test_tenant"
    assert operation.user_id == "test_user"
    assert operation.total_steps == 3
    assert len(operation.steps) == 3
    assert operation.status == ProgressStatus.PENDING


@pytest.mark.asyncio
async def test_start_operation(temp_tracker):
    """Test starting an operation"""
    operation_id = await temp_tracker.create_operation(
        name="Test Operation",
        description="Test description",
        steps=[{"name": "Step 1", "description": "Test step"}],
    )

    success = await temp_tracker.start_operation(operation_id)
    assert success is True

    operation = temp_tracker.get_operation(operation_id)
    assert operation.status == ProgressStatus.RUNNING
    assert operation.started_at is not None


@pytest.mark.asyncio
async def test_update_step_progress(temp_tracker):
    """Test updating step progress"""
    operation_id = await temp_tracker.create_operation(
        name="Test Operation",
        description="Test description",
        steps=[
            {"name": "Step 1", "description": "First step"},
            {"name": "Step 2", "description": "Second step"},
        ],
    )

    await temp_tracker.start_operation(operation_id)

    # Update first step
    success = await temp_tracker.update_step_progress(
        operation_id, 0, 0.5, ProgressStatus.RUNNING
    )
    assert success is True

    operation = temp_tracker.get_operation(operation_id)
    assert operation.steps[0].progress == 0.5
    assert operation.steps[0].status == ProgressStatus.RUNNING
    assert operation.current_step == 0
    assert operation.overall_progress == 0.25  # 0.5 / 2 steps


@pytest.mark.asyncio
async def test_complete_step(temp_tracker):
    """Test completing a step"""
    operation_id = await temp_tracker.create_operation(
        name="Test Operation",
        description="Test description",
        steps=[{"name": "Step 1", "description": "Test step"}],
    )

    await temp_tracker.start_operation(operation_id)

    success = await temp_tracker.complete_step(
        operation_id, 0, metadata={"result": "success"}
    )
    assert success is True

    operation = temp_tracker.get_operation(operation_id)
    assert operation.steps[0].status == ProgressStatus.COMPLETED
    assert operation.steps[0].progress == 1.0
    assert operation.status == ProgressStatus.COMPLETED
    assert operation.overall_progress == 1.0
    assert operation.steps[0].metadata["result"] == "success"


@pytest.mark.asyncio
async def test_fail_step(temp_tracker):
    """Test failing a step"""
    operation_id = await temp_tracker.create_operation(
        name="Test Operation",
        description="Test description",
        steps=[{"name": "Step 1", "description": "Test step"}],
    )

    await temp_tracker.start_operation(operation_id)

    success = await temp_tracker.fail_step(
        operation_id, 0, "Test error", metadata={"error_code": 500}
    )
    assert success is True

    operation = temp_tracker.get_operation(operation_id)
    assert operation.steps[0].status == ProgressStatus.FAILED
    assert operation.steps[0].error == "Test error"
    assert operation.status == ProgressStatus.FAILED
    assert operation.steps[0].metadata["error_code"] == 500


@pytest.mark.asyncio
async def test_cancel_operation(temp_tracker):
    """Test cancelling an operation"""
    operation_id = await temp_tracker.create_operation(
        name="Test Operation",
        description="Test description",
        steps=[
            {"name": "Step 1", "description": "First step"},
            {"name": "Step 2", "description": "Second step"},
        ],
    )

    await temp_tracker.start_operation(operation_id)

    success = await temp_tracker.cancel_operation(operation_id, "User cancelled")
    assert success is True

    operation = temp_tracker.get_operation(operation_id)
    assert operation.status == ProgressStatus.CANCELLED
    assert operation.metadata["cancellation_reason"] == "User cancelled"

    # All steps should be cancelled
    for step in operation.steps:
        assert step.status == ProgressStatus.CANCELLED


@pytest.mark.asyncio
async def test_list_operations(temp_tracker):
    """Test listing operations with filters"""
    # Create operations for different tenants and users
    op1 = await temp_tracker.create_operation(
        "Op 1",
        "Description 1",
        [{"name": "Step", "description": "Step"}],
        tenant_id="tenant1",
        user_id="user1",
    )
    op2 = await temp_tracker.create_operation(
        "Op 2",
        "Description 2",
        [{"name": "Step", "description": "Step"}],
        tenant_id="tenant2",
        user_id="user1",
    )
    op3 = await temp_tracker.create_operation(
        "Op 3",
        "Description 3",
        [{"name": "Step", "description": "Step"}],
        tenant_id="tenant1",
        user_id="user2",
    )

    await temp_tracker.start_operation(op1)
    await temp_tracker.complete_step(op1, 0)

    # Test listing all operations
    all_ops = temp_tracker.list_operations()
    assert len(all_ops) == 3

    # Test filtering by tenant
    tenant1_ops = temp_tracker.list_operations(tenant_id="tenant1")
    assert len(tenant1_ops) == 2

    # Test filtering by user
    user1_ops = temp_tracker.list_operations(user_id="user1")
    assert len(user1_ops) == 2

    # Test filtering by status
    completed_ops = temp_tracker.list_operations(status=ProgressStatus.COMPLETED)
    assert len(completed_ops) == 1
    assert completed_ops[0].id == op1


@pytest.mark.asyncio
async def test_websocket_connections(temp_tracker):
    """Test WebSocket connection management"""
    operation_id = await temp_tracker.create_operation(
        "Test Operation", "Description", [{"name": "Step", "description": "Step"}]
    )

    # Mock WebSocket
    mock_websocket = AsyncMock()

    # Add WebSocket connection
    await temp_tracker.add_websocket_connection(operation_id, mock_websocket)
    assert mock_websocket in temp_tracker.websocket_connections[operation_id]

    # Update progress (should trigger WebSocket notification)
    await temp_tracker.start_operation(operation_id)
    await temp_tracker.update_step_progress(operation_id, 0, 0.5)

    # Verify WebSocket was called
    assert mock_websocket.send_text.called

    # Remove WebSocket connection
    await temp_tracker.remove_websocket_connection(operation_id, mock_websocket)
    assert mock_websocket not in temp_tracker.websocket_connections[operation_id]


@pytest.mark.asyncio
async def test_callbacks(temp_tracker):
    """Test operation callbacks"""
    operation_id = await temp_tracker.create_operation(
        "Test Operation", "Description", [{"name": "Step", "description": "Step"}]
    )

    # Mock callback
    callback_called = False

    def test_callback(operation):
        nonlocal callback_called
        callback_called = True
        assert operation.id == operation_id

    # Add callback
    temp_tracker.add_callback(operation_id, test_callback)

    # Update progress (should trigger callback)
    await temp_tracker.start_operation(operation_id)

    assert callback_called is True


@pytest.mark.asyncio
async def test_progress_context():
    """Test progress context manager"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tracker = ProgressTracker(persistence_file=tmp.name)

        operation_id = await tracker.create_operation(
            "Test Operation", "Description", [{"name": "Step", "description": "Step"}]
        )

        await tracker.start_operation(operation_id)

        # Test successful context
        async with ProgressContext(operation_id, 0, tracker) as ctx:
            await ctx.update_progress(0.5, {"intermediate": "result"})

        operation = tracker.get_operation(operation_id)
        assert operation.steps[0].status == ProgressStatus.COMPLETED
        assert operation.steps[0].progress == 1.0
        assert operation.steps[0].metadata["intermediate"] == "result"

        # Cleanup
        Path(tmp.name).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_progress_context_error():
    """Test progress context manager with error"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tracker = ProgressTracker(persistence_file=tmp.name)

        operation_id = await tracker.create_operation(
            "Test Operation", "Description", [{"name": "Step", "description": "Step"}]
        )

        await tracker.start_operation(operation_id)

        # Test error in context
        try:
            async with ProgressContext(operation_id, 0, tracker):
                raise ValueError("Test error")
        except ValueError:
            pass

        operation = tracker.get_operation(operation_id)
        assert operation.steps[0].status == ProgressStatus.FAILED
        assert "Test error" in operation.steps[0].error

        # Cleanup
        Path(tmp.name).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_persistence(temp_tracker):
    """Test operation persistence"""
    operation_id = await temp_tracker.create_operation(
        "Test Operation", "Description", [{"name": "Step", "description": "Step"}]
    )

    await temp_tracker.start_operation(operation_id)
    await temp_tracker.update_step_progress(operation_id, 0, 0.7)

    # Create new tracker with same persistence file
    new_tracker = ProgressTracker(persistence_file=temp_tracker.persistence_file)

    # Verify operation was loaded
    assert operation_id in new_tracker.operations
    restored_operation = new_tracker.get_operation(operation_id)
    assert restored_operation.name == "Test Operation"
    assert restored_operation.status == ProgressStatus.RUNNING
    assert restored_operation.steps[0].progress == 0.7


@pytest.mark.asyncio
async def test_cleanup_old_operations(temp_tracker):
    """Test cleaning up old operations"""
    # Create completed operation
    operation_id = await temp_tracker.create_operation(
        "Old Operation", "Description", [{"name": "Step", "description": "Step"}]
    )

    await temp_tracker.start_operation(operation_id)
    await temp_tracker.complete_step(operation_id, 0)

    # Manually set completion time to be old
    operation = temp_tracker.get_operation(operation_id)
    operation.completed_at = time.time() - (200 * 3600)  # 200 hours ago

    # Run cleanup with 168 hour threshold (7 days)
    await temp_tracker.cleanup_old_operations(max_age_hours=168)

    # Operation should be removed
    assert operation_id not in temp_tracker.operations


@pytest.mark.asyncio
async def test_estimated_completion():
    """Test estimated completion time calculation"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tracker = ProgressTracker(persistence_file=tmp.name)

        operation_id = await tracker.create_operation(
            "Test Operation",
            "Description",
            [{"name": f"Step {i}", "description": f"Step {i}"} for i in range(4)],
        )

        start_time = time.time()
        await tracker.start_operation(operation_id)

        # Simulate some progress
        await asyncio.sleep(0.1)  # Small delay to ensure time difference
        await tracker.update_step_progress(
            operation_id, 0, 1.0, ProgressStatus.COMPLETED
        )
        await tracker.update_step_progress(operation_id, 1, 0.5, ProgressStatus.RUNNING)

        operation = tracker.get_operation(operation_id)

        # Should have estimated completion time
        assert operation.estimated_completion is not None
        assert operation.estimated_completion > start_time

        # Progress should be calculated correctly
        expected_progress = (1.0 + 0.5) / 4  # 1.5 out of 4 steps
        assert abs(operation.overall_progress - expected_progress) < 0.01

        # Cleanup
        Path(tmp.name).unlink(missing_ok=True)


def test_progress_status_enum():
    """Test ProgressStatus enum values"""
    assert ProgressStatus.PENDING.value == "pending"
    assert ProgressStatus.RUNNING.value == "running"
    assert ProgressStatus.COMPLETED.value == "completed"
    assert ProgressStatus.FAILED.value == "failed"
    assert ProgressStatus.CANCELLED.value == "cancelled"


@pytest.mark.asyncio
async def test_invalid_operations(temp_tracker):
    """Test handling of invalid operations"""
    # Test with non-existent operation
    success = await temp_tracker.start_operation("non_existent_id")
    assert success is False

    success = await temp_tracker.update_step_progress("non_existent_id", 0, 0.5)
    assert success is False

    success = await temp_tracker.complete_step("non_existent_id", 0)
    assert success is False

    success = await temp_tracker.cancel_operation("non_existent_id")
    assert success is False

    # Test with invalid step index
    operation_id = await temp_tracker.create_operation(
        "Test Operation", "Description", [{"name": "Step", "description": "Step"}]
    )

    success = await temp_tracker.update_step_progress(
        operation_id, 5, 0.5
    )  # Invalid index
    assert success is False


if __name__ == "__main__":
    pytest.main([__file__])
