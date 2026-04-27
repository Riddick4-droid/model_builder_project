"""
Shared state schema for the pipeline.
All agents read from and write to this state.
"""

from typing import TypedDict, Dict, Any, Optional, List

class PipelineState(TypedDict):
    # Input / user provided
    data_path: str
    model_type: str
    metric_threshold: float
    target_column: Optional[str]  # can be guessed by DataAgent
    
    # Unique ID for checkpointing (can be generated if not provided)
    pipeline_id: str
    
    # Data agent outputs
    data_loaded: bool
    shape: List[int]
    columns: List[str]
    missing_percentage: float
    imputation_suggestion: str
    target_column_guess: str
    
    # Preprocess agent outputs
    processed_paths: Dict[str, str]  # X_train, X_test, y_train, y_test paths
    input_shape: int
    num_classes: int
    
    # Train agent outputs
    accuracy: float
    run_id: str
    model_path: str
    
    # Eval agent outputs
    decision: str  # "deploy", "retrain", "fail"
    reason: str
    should_stop: bool
    
    # HITL agent outputs
    human_approved: bool
    human_feedback: str
    
    # Deploy agent outputs
    deployment_status: str
    deployment_url: str
    deployment_timestamp: str
    
    # Retry tracking
    retries: int