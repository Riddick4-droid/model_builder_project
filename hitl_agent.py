"""
Human-in-the-Loop Agent: Interrupts pipeline when metric is borderline and asks for human approval.
Approval decisions can be submitted via an external API endpoint.
"""

import asyncio
from typing import Dict, Any

class HumanApprovalAgent:
    def __init__(self, approval_queue: asyncio.Queue, timeout_seconds: float = 60.0):
        """
        Args:
            approval_queue: Async queue that receives human decisions ("yes" or "no").
            timeout_seconds: How long to wait for a human response before auto-failing.
        """
        self.queue = approval_queue
        self.timeout = timeout_seconds

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if metric is borderline. If yes, request human approval.
        Updates state with 'human_approved' (bool) and optionally 'human_feedback'.
        """
        metric = state.get("metric", 0.0)
        threshold = state.get("metric_threshold", 0.75)
        tolerance = 0.05  # borderline range: ±0.05

        # Decide if human intervention is needed
        if abs(metric - threshold) <= tolerance:
            # Borderline case – ask human
            request_id = f"pipeline_{state.get('run_id', 'unknown')}"
            print(f"\n[HITL] Request ID: {request_id}")
            print(f"[HITL] Model metric = {metric:.3f}, threshold = {threshold:.3f}")
            print(f"[HITL] Waiting for human decision (yes/no) via /approve/{request_id} ...")

            try:
                # Wait for a decision from the queue
                decision = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
                if decision.lower() == "yes":
                    state["human_approved"] = True
                    state["human_feedback"] = "Approved by user"
                else:
                    state["human_approved"] = False
                    state["human_feedback"] = "Rejected by user"
                    # Override evaluation decision to force retrain or fail
                    state["decision"] = "retrain"  # orchestrator will see this
            except asyncio.TimeoutError:
                print("[HITL] Timeout – no human response. Failing deployment.")
                state["human_approved"] = False
                state["human_feedback"] = "Timeout – no response"
                state["decision"] = "fail"
        else:
            # Not borderline – auto-approve
            state["human_approved"] = True
            state["human_feedback"] = "Auto-approved (metric well above/below threshold)"

        return state