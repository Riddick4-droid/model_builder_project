"""
FastAPI entry point for the MLOps Agent System.
Provides endpoints to trigger pipelines, approve human-in-the-loop requests, and resume from checkpoints.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import pickle
import uuid
import os

from model_builder_project.graph.orchestrator import Orchestrator
from model_builder_project.utils.checkpoint import CheckpointManager

# Global queues and orchestrator instances (per pipeline)
# In production, you might want to use a distributed queue (e.g., Redis pub/sub)
approval_queues: Dict[str, asyncio.Queue] = {}
running_pipelines: Dict[str, asyncio.Task] = {}

app = FastAPI(title="MLOps Agent Orchestrator", version="1.0.0")

# Configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
S3_BUCKET = os.getenv("S3_BUCKET", None)  # optional, set for production

# Request/Response models
class RunPipelineRequest(BaseModel):
    data_path: str
    model_type: str = "random_forest"
    metric_threshold: float = 0.75
    target_column: Optional[str] = None
    pipeline_id: Optional[str] = None  # allow resuming previously saved

class ApprovalRequest(BaseModel):
    decision: str  # "yes" or "no"

class PipelineResponse(BaseModel):
    pipeline_id: str
    status: str  # "running", "completed", "failed", "waiting_approval"
    final_state: Optional[Dict[str, Any]] = None


async def run_pipeline_background(pipeline_id: str, initial_state: dict, approval_queue: asyncio.Queue):
    """
    Background task that runs the pipeline and cleans up afterwards.
    """
    checkpoint_mgr = CheckpointManager(redis_host=REDIS_HOST, s3_bucket=S3_BUCKET)
    orch = Orchestrator(approval_queue=approval_queue, checkpoint_manager=checkpoint_mgr, verbose=True)
    
    try:
        final_state = await orch.run(initial_state)
        # Store final state somewhere (e.g., Redis) for later retrieval
        checkpoint_mgr.redis_client.setex(
            f"pipeline_result:{pipeline_id}",
            3600,
            pickle.dumps(final_state)
        )
    except Exception as e:
        print(f"Pipeline {pipeline_id} failed: {e}")
        # Optionally store error
    finally:
        # Cleanup queue reference
        if pipeline_id in approval_queues:
            del approval_queues[pipeline_id]
        if pipeline_id in running_pipelines:
            del running_pipelines[pipeline_id]


@app.post("/run", response_model=PipelineResponse)
async def run_pipeline(request: RunPipelineRequest, background_tasks: BackgroundTasks):
    """
    Start a new pipeline or resume an existing one (if pipeline_id provided).
    """
    pipeline_id = request.pipeline_id or str(uuid.uuid4())
    
    if pipeline_id in running_pipelines:
        raise HTTPException(status_code=409, detail="Pipeline already running")
    
    # Build initial state
    initial_state = {
        "pipeline_id": pipeline_id,
        "data_path": request.data_path,
        "model_type": request.model_type,
        "metric_threshold": request.metric_threshold,
        "target_column": request.target_column,
        "retries": 0
    }
    
    # Create a dedicated queue for this pipeline's HITL approvals
    approval_queue = asyncio.Queue()
    approval_queues[pipeline_id] = approval_queue
    
    # Start background task
    task = asyncio.create_task(
        run_pipeline_background(pipeline_id, initial_state, approval_queue)
    )
    running_pipelines[pipeline_id] = task
    
    return PipelineResponse(pipeline_id=pipeline_id, status="running")


@app.post("/approve/{pipeline_id}")
async def approve_pipeline(pipeline_id: str, request: ApprovalRequest):
    """
    Human provides decision for a pipeline waiting for approval.
    """
    if pipeline_id not in approval_queues:
        raise HTTPException(status_code=404, detail="Pipeline not found or not waiting for approval")
    
    if request.decision.lower() not in ["yes", "no"]:
        raise HTTPException(status_code=400, detail="Decision must be 'yes' or 'no'")
    
    queue = approval_queues[pipeline_id]
    await queue.put(request.decision.lower())
    return {"status": "decision received", "pipeline_id": pipeline_id}


@app.get("/status/{pipeline_id}")
async def get_status(pipeline_id: str):
    """
    Check if pipeline is still running and get final result if completed.
    """
    import pickle
    checkpoint_mgr = CheckpointManager(redis_host=REDIS_HOST)
    # Check if result exists
    result_key = f"pipeline_result:{pipeline_id}"
    result_data = await checkpoint_mgr.redis_client.get(result_key)
    if result_data:
        final_state = pickle.loads(result_data)
        return {"pipeline_id": pipeline_id, "status": "completed", "final_state": final_state}
    
    # Check if still running
    if pipeline_id in running_pipelines:
        # Optionally get last checkpoint step
        steps = await checkpoint_mgr.list_checkpoints(pipeline_id)
        return {"pipeline_id": pipeline_id, "status": "running", "completed_steps": steps}
    else:
        return {"pipeline_id": pipeline_id, "status": "not_found"}


@app.get("/")
async def root():
    return {"message": "MLOps Agent System with LangGraph, HITL, Checkpointing"}

# Optional: graceful shutdown to cancel tasks
@app.on_event("shutdown")
async def shutdown():
    for task in running_pipelines.values():
        task.cancel()
    await asyncio.gather(*running_pipelines.values(), return_exceptions=True)