"""
Generation node for creating final answers
"""

from langchain_core.messages import AIMessage
from src.core.types import AgentState
from src.deps.models import ModelConfig


async def generation_node(state: AgentState, model_config: ModelConfig = None) -> AgentState:
    """
    Generation node that creates comprehensive final answers.

    Args:
        state: Current agent state
        model_config: Optional model configuration

    Returns:
        Updated agent state with final generated response
    """
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]

    if tool_name == "generate_final_answer":
        # Lazy import to avoid circular dependency
        from src.core.tools import generate_final_answer

        # Get required information from state
        original_query = state.get("original_query", "")
        context = state.get("context", "")
        attempts = state.get("retrieval_attempts", 1)
        reflection_feedback = state.get("reflection_feedback", "")

        # Extract query from tool args if not in state
        if not original_query:
            original_query = tool_call["args"].get("query", "")

        # Extract context from tool args if not in state
        if not context:
            context = tool_call["args"].get("context", "")

        # Generate the final answer
        final_response = await generate_final_answer.ainvoke(
            {
                "query": original_query,
                "context": context,
                "attempts": attempts,
                "reflection_feedback": reflection_feedback,
                "llm_config": model_config,
            }
        )

        # Update state with the final response
        new_state = {
            "messages": state["messages"] + [AIMessage(content=final_response, id=tool_call["id"])],
            "final_response": final_response,
            "is_complete": True,
        }

        return new_state

    return state
