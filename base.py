"""
Base class for all LLM-powered agents.
Handles prompt construction, LLM invocation (async), and JSON response parsing.
"""

import json
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class LLMAgent:
    def __init__(self, name: str, system_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0):
        """
        Args:
            name: Agent identifier (used for logging)
            system_prompt: Instructions for the LLM on what to do.
            model: OpenAI model name.
            temperature: 0 for deterministic, higher for creativity.
        """
        self.name = name
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.system_prompt = system_prompt

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent asynchronously.
        Expects state to be a serializable dict.
        The agent will update the state based on LLM's JSON response.
        """
        # Prepare the user message with current state
        user_msg = (
            f"Current pipeline state:\n{json.dumps(state, indent=2)}\n\n"
            f"Perform the {self.name} step. Return **only** a JSON object with the fields you want to update in the state. "
            "Do not include any extra text or markdown formatting."
        )

        # Call LLM asynchronously
        response = await self.llm.ainvoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_msg)
        ])

        # Parse JSON response (strip potential markdown code fences)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        updates = json.loads(content)

        # Merge updates into state
        state.update(updates)
        return state
