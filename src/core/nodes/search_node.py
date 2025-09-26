"""
Search node for handling knowledge graph searches with reflection
"""

from langchain_core.messages import HumanMessage, AIMessage
from src.core.types import AgentState
from src.deps.models import ModelConfig


async def search_node(state: AgentState, model_config: ModelConfig = None) -> AgentState:
    """
    Search node that handles knowledge graph searches with reflection and refinement.

    Args:
        state: Current agent state
        model_config: Optional model configuration

    Returns:
        Updated agent state with search results and reflection
    """
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]

    if tool_name == "search_knowledge_graph":
        # Lazy imports to avoid circular dependency
        from src.core.tools import search_knowledge_graph
        from src.core.tools.reflection_tools import reflect_on_retrieval_quality, refine_query_based_on_feedback

        # Initialize state values if not present
        original_query = state.get("original_query")
        current_query = state.get("current_query")
        retrieval_attempts = state.get("retrieval_attempts", 0)
        max_attempts = state.get("max_retrieval_attempts", 3)

        # Extract query from the first human message if not set
        if not original_query:
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    current_query = msg.content
                    break

        # Use current query or fall back to tool args
        query_to_use = current_query or tool_call["args"].get("question", original_query)

        # Perform the search
        tool_args = tool_call["args"].copy()
        tool_args["question"] = query_to_use
        result = await search_knowledge_graph.ainvoke(tool_args)

        # Increment attempts
        retrieval_attempts += 1

        # Perform reflection on the results
        reflection_result = await reflect_on_retrieval_quality(
            original_query=original_query,
            retrieved_context=result,
            current_attempt=retrieval_attempts,
            max_attempts=max_attempts,
            llm_config=model_config,
        )

        # Determine if we should refine and retry
        should_refine = (
            not reflection_result["is_sufficient"]
            and retrieval_attempts < max_attempts
            and reflection_result["missing_aspects"]
        )

        refined_query = None
        if should_refine:
            refined_query = await refine_query_based_on_feedback(
                original_query=original_query,
                missing_aspects=reflection_result["missing_aspects"],
                previous_context=result,
                llm_config=model_config,
            )

        # Update state with reflection results
        new_state = {
            "messages": state["messages"] + [AIMessage(content=result, id=tool_call["id"])],
            "context": result,
            "original_query": original_query,
            "current_query": refined_query if should_refine else current_query,
            "retrieval_attempts": retrieval_attempts,
            "max_retrieval_attempts": max_attempts,
            "reflection_feedback": reflection_result["reasoning"],
            "should_refine": should_refine,
            "is_sufficient": reflection_result["is_sufficient"],
        }

        return new_state

    return state
