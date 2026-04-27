"""
Deploy Agent: Simulates model deployment and records a deployment URL.
In production, this would call a real deployment tool (e.g., to AWS SageMaker, Kubernetes, or a local model server).
"""

from model_builder_project.base import LLMAgent

DEPLOY_SYSTEM_PROMPT = """
You are a Deployment Agent. Your job is to simulate deploying a model to a serving endpoint.

You will receive the state with:
- run_id (MLflow run ID) or model_path
- metric (accuracy)
- model_type (optional)

Return a JSON object with:
- "deployment_status": "success" or "failure"
- "deployment_url": a string like "http://localhost:8000/models/{run_id}"
- "deployment_timestamp": current timestamp in ISO format (you can invent it, e.g., "2025-04-27T10:00:00Z")

If deployment fails, explain briefly in "deployment_error".
For now, always assume success for the dummy endpoint.
"""

class DeployAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            name="DeployAgent",
            system_prompt=DEPLOY_SYSTEM_PROMPT,
            model="gpt-4o-mini",
            temperature=0
        )
    
    async def run(self, state: dict) -> dict:
        """
        Simulate deployment. Adds deployment info to state.
        """
        # Let the LLM generate the deployment URL based on run_id
        state = await super().run(state)
        
        # Ensure minimal fields even if LLM output is incomplete
        if "deployment_url" not in state:
            run_id = state.get("run_id", "unknown")
            state["deployment_url"] = f"http://localhost:8000/models/{run_id}"
        if "deployment_status" not in state:
            state["deployment_status"] = "success"
        
        return state