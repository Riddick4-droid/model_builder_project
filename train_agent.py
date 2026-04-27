"""
Train Agent: uses the train_sklearn_model tool to train a model.
The LLM decides hyperparameters based on dataset size and problem type.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from model_builder_project.tools.train_tools import train_sklearn_model

TRAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a training agent. Your job is to train a RandomForest model using the provided training data.
You will receive the state with keys:
- processed_paths: dict containing "X_train", "y_train" file paths (from PreprocessAgent)
- input_shape (number of features)
- num_classes (for classification)
- model_type (maybe specified by user, e.g., "random_forest")
- metric_threshold (optional, not used by tool but for later evaluation)

Use the `train_sklearn_model` tool. Choose reasonable hyperparameters:
- For small datasets (< 1k rows): n_estimators=50, max_depth=5
- For medium (1k-10k): n_estimators=100, max_depth=10
- For large (>10k): n_estimators=200, max_depth=15
Always set random_state=42 for reproducibility.

Return the tool's output exactly (accuracy and run_id).
"""),
    ("human", "Current state: {state}")
])

class TrainAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        llm = ChatOpenAI(model=model, temperature=0)
        tools = [train_sklearn_model]
        agent = create_openai_tools_agent(llm, tools, TRAIN_PROMPT)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    async def run(self, state: dict) -> dict:
        """Execute training agent. Expects state to have processed_paths."""
        # Ensure the tool gets the file paths
        result = await self.executor.ainvoke({"state": state})
        # result['output'] is dict with accuracy, run_id
        state.update(result['output'])
        state["model_path"] = f"mlflow:///{state['run_id']}"
        return state