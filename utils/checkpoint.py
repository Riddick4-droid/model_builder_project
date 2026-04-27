"""
Checkpointing utilities for saving and loading pipeline state.
Supports Redis (fast, ephemeral) and S3 (durable, cross‑instance).
"""

import os
import pickle
import asyncio
from typing import Dict, Any, Optional
import redis.asyncio as redis
import boto3
from botocore.exceptions import ClientError

class CheckpointManager:
    """
    Async checkpoint manager. Saves state to Redis with TTL, and optionally to S3.
    """
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_ttl_seconds: int = 3600,  # 1 hour
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "checkpoints/"
    ):
        self.redis_ttl = redis_ttl_seconds
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        
        # Redis client (async)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        
        # S3 client (sync – we'll run in thread pool)
        if s3_bucket:
            self.s3_client = boto3.client("s3")
        else:
            self.s3_client = None

    async def save(self, pipeline_id: str, step_name: str, state: Dict[str, Any]) -> None:
        """
        Save checkpoint for a specific pipeline step.
        """
        key = f"checkpoint:{pipeline_id}:{step_name}"
        data = pickle.dumps(state)
        
        # Redis
        await self.redis_client.setex(key, self.redis_ttl, data)
        
        # S3 (optional)
        if self.s3_client and self.s3_bucket:
            s3_key = f"{self.s3_prefix}{pipeline_id}/{step_name}.pkl"
            # Run sync S3 upload in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.put_object,
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=data
            )
    
    async def load(self, pipeline_id: str, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from Redis (fallback to S3). Returns None if not found.
        """
        key = f"checkpoint:{pipeline_id}:{step_name}"
        data = await self.redis_client.get(key)
        
        if data is not None:
            return pickle.loads(data)
        
        # Fallback to S3
        if self.s3_client and self.s3_bucket:
            s3_key = f"{self.s3_prefix}{pipeline_id}/{step_name}.pkl"
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self.s3_client.get_object,
                    Bucket=self.s3_bucket,
                    Key=s3_key
                )
                data = response["Body"].read()
                return pickle.loads(data)
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    return None
                raise
        return None
    
    async def list_checkpoints(self, pipeline_id: str) -> list:
        """
        List all available step names for a pipeline (Redis only for simplicity).
        """
        pattern = f"checkpoint:{pipeline_id}:*"
        keys = await self.redis_client.keys(pattern)
        # Extract step names
        return [k.decode().split(":")[-1] for k in keys]

    async def delete(self, pipeline_id: str, step_name: str) -> None:
        """
        Delete checkpoint for a specific step.
        """
        key = f"checkpoint:{pipeline_id}:{step_name}"
        await self.redis_client.delete(key)
        # S3 deletion optional – could implement with boto3 if needed