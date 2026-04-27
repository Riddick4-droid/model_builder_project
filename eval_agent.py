"""
Evaluation Agent: LLM-based judge that compares model metric against threshold
and decides next action (deploy, retrain, or fail).
"""

from model_builder_project.base import LLMAgent

EVAL_SYSTEM_PROMPT = """
You are an Evaluation Agent. You decide if a trained model is ready for deployment.

You will receive the state with:
- metric (float, e.g., 0.82)
- metric_threshold (float, e.g., 0.75)
- retries (int, number of retraining attempts so far)
- model_type (optional)
- any other relevant info.

Your decision rules:
- If metric >= threshold → "deploy"
- If metric < threshold and retries < 3 → "retrain"
- If metric < threshold and retries >= 3 → "fail"

Return a JSON object with:
- "decision": one of "deploy", "retrain", "fail"
- "reason": short explanation (e.g., "Accuracy 0.72 below threshold 0.75, retry 2/3")
- "should_stop": boolean (true only if decision is "fail")

Be concise. Do not add extra text outside the JSON.
"""

class EvalAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            name="EvalAgent",
            system_prompt=EVAL_SYSTEM_PROMPT,
            model="gpt-4o-mini",
            temperature=0
        )
    
    async def run(self, state: dict) -> dict:
        """
        Run the evaluation agent. Ensures retries counter exists and applies decision.
        """
        # Initialize retries if not present
        if "retries" not in state:
            state["retries"] = 0
        
        # Call base LLM agent to get decision
        state = await super().run(state)
        
        # Increment retries if decision is retrain, or reset if deploy/fail?
        # Actually we increment after retrain, but the orchestrator will handle retry loop.
        # For now just store decision; the orchestrator will increment retries when it goes back to train.
        return state