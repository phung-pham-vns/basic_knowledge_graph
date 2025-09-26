"""
Refinement node for handling query refinement and retry logic
"""

from langchain_core.messages import HumanMessage
from src.core.types import AgentState


async def refinement_node(state: AgentState) -> AgentState:
    """
    Refinement node that triggers new searches with refined queries.

    Args:
        state: Current agent state

    Returns:
        Updated agent state with refinement message
    """
    current_query = state.get("current_query")
    reflection_feedback = state.get("reflection_feedback", "")

    if current_query:
        # Create a message that will trigger the agent to search again
        refinement_message = HumanMessage(
            content=f"Please search for: {current_query}\n\n[Refinement based on: {reflection_feedback}]"
        )

        # Reset refinement flags for the next iteration
        return {"messages": [refinement_message], "should_refine": False, "is_sufficient": False}

    # If no refined query, mark as sufficient to end the loop
    return {"is_sufficient": True, "should_refine": False}
