"""
LangGraph orchestrator that wires all agents together with:
- Async execution
- Conditional edges based on evaluation decision
- Human-in-the-loop approval for borderline metrics
- Checkpointing via Redis/S3
"""

from langgraph.graph import StateGraph, END
from model_builder_project.graph.state import PipelineState
from model_builder_project.data_agent import DataAgent
from model_builder_project.preprocess_agent import PreprocessAgent
from model_builder_project.train_agent import TrainAgent
from model_builder_project.eval_agent import EvalAgent
from model_builder_project.hitl_agent import HumanApprovalAgent
from model_builder_project.deploy_agent import DeployAgent
from model_builder_project.utils.checkpoint import CheckpointManager

import asyncio
from typing import Literal, Dict, Any

class Orchestrator:
    def __init__(
        self,
        approval_queue: asyncio.Queue,
        checkpoint_manager: CheckpointManager,
        verbose: bool = True
    ):
        self.approval_queue = approval_queue
        self.checkpointer = checkpoint_manager
        self.verbose = verbose
        
        # Instantiate agents
        self.data_agent = DataAgent()
        self.preprocess_agent = PreprocessAgent()
        self.train_agent = TrainAgent()
        self.eval_agent = EvalAgent()
        self.hitl_agent = HumanApprovalAgent(approval_queue)
        self.deploy_agent = DeployAgent()
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"[Orchestrator] {msg}")
    
    async def _run_with_checkpoint(self, step_name: str, state: Dict[str, Any], agent_func) -> Dict[str, Any]:
        """
        Wrapper that loads checkpoint before running agent, and saves after.
        """
        pipeline_id = state.get("pipeline_id")
        if not pipeline_id:
            raise ValueError("State missing 'pipeline_id' for checkpointing")
        
        # Try load
        saved = await self.checkpointer.load(pipeline_id, step_name)
        if saved is not None:
            self._log(f"Loaded checkpoint for {step_name}, resuming.")
            # Merge saved into state (keeps any new fields from previous nodes)
            state.update(saved)
            return state
        
        # Execute agent
        self._log(f"Running {step_name}...")
        new_state = await agent_func(state)
        
        # Save checkpoint
        await self.checkpointer.save(pipeline_id, step_name, new_state)
        return new_state
    
    async def data_node(self, state: PipelineState) -> PipelineState:
        return await self._run_with_checkpoint("data", state, self.data_agent.run)
    
    async def preprocess_node(self, state: PipelineState) -> PipelineState:
        return await self._run_with_checkpoint("preprocess", state, self.preprocess_agent.run)
    
    async def train_node(self, state: PipelineState) -> PipelineState:
        return await self._run_with_checkpoint("train", state, self.train_agent.run)
    
    async def eval_node(self, state: PipelineState) -> PipelineState:
        return await self._run_with_checkpoint("eval", state, self.eval_agent.run)
    
    async def hitl_node(self, state: PipelineState) -> PipelineState:
        # HITL might wait for human input; we also checkpoint before waiting?
        # For simplicity, we checkpoint after approval.
        new_state = await self.hitl_agent.run(state)
        await self.checkpointer.save(state["pipeline_id"], "human_check", new_state)
        return new_state
    
    async def deploy_node(self, state: PipelineState) -> PipelineState:
        return await self._run_with_checkpoint("deploy", state, self.deploy_agent.run)
    
    def _router(self, state: PipelineState) -> Literal["deploy", "train", "fail"]:
        """
        Conditional edge after evaluation + human approval.
        """
        decision = state.get("decision", "fail")
        human_approved = state.get("human_approved", False)
        retries = state.get("retries", 0)
        
        self._log(f"Router: decision={decision}, approved={human_approved}, retries={retries}")
        
        if decision == "deploy" and human_approved:
            return "deploy"
        elif decision == "retrain" and retries < 3:
            # Increment retries before going back to train
            state["retries"] = retries + 1
            return "train"
        else:
            return "fail"
    
    async def build_graph(self):
        """
        Construct the LangGraph state graph with all nodes and edges.
        """
        builder = StateGraph(PipelineState)
        
        # Add nodes
        builder.add_node("data", self.data_node)
        builder.add_node("preprocess", self.preprocess_node)
        builder.add_node("train", self.train_node)
        builder.add_node("eval", self.eval_node)
        builder.add_node("human_check", self.hitl_node)
        builder.add_node("deploy", self.deploy_node)
        builder.add_node("fail", lambda s: s)  # terminal fail node (no change)
        
        # Set entry point
        builder.set_entry_point("data")
        
        # Linear edges until eval
        builder.add_edge("data", "preprocess")
        builder.add_edge("preprocess", "train")
        builder.add_edge("train", "eval")
        builder.add_edge("eval", "human_check")
        
        # Conditional edges after human_check
        builder.add_conditional_edges(
            "human_check",
            self._router,
            {
                "deploy": "deploy",
                "train": "train",
                "fail": "fail"
            }
        )
        
        # Terminal edges
        builder.add_edge("deploy", END)
        builder.add_edge("fail", END)
        
        # Compile (LangGraph doesn't natively support async checkpointer yet, 
        # but we use our own wrapper so it's fine)
        graph = builder.compile()
        return graph
    
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline from the given initial state.
        """
        graph = await self.build_graph()
        final_state = await graph.ainvoke(initial_state)
        return final_state