"""
Preprocess Agent: uses the preprocessing tool to transform raw data.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from model_builder_project.tools.preprocess_tools import run_preprocessing

PREPROCESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a preprocessing agent. Your goal is to prepare raw data for training.
You will receive the state with keys:
- data_path (path to raw CSV)
- target_column_guess (from DataAgent, or explicit target_column from user)
- imputation_suggestion (from DataAgent)

Use the `run_preprocessing` tool with appropriate arguments.
- target_column: use the explicitly provided one, or the guess.
- imputation_strategy: use the suggestion, or default to 'mean'.
- test_size: 0.2 unless the dataset is tiny (then use 0.3).

After calling the tool, return the same keys as the tool's output.
"""),
    ("human", "Current state: {state}")
])

class PreprocessAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        llm = ChatOpenAI(model=model, temperature=0)
        tools = [run_preprocessing]
        agent = create_openai_tools_agent(llm, tools, PREPROCESS_PROMPT)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    async def run(self, state: dict) -> dict:
        """Call the agent with current state, get updates."""
        result = await self.executor.ainvoke({"state": state})
        # result['output'] is the tool's returned dict
        # Merge into state
        state.update(result['output'])
        return state