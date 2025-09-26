from src.core.types import AgentState
from src.prompts.generation_prompts import get_system_prompt
from src.deps.models import get_llm, ModelConfig


async def agent_node(state: AgentState, model_config: ModelConfig = None) -> AgentState:
    """
    Main agent node that processes messages and orchestrates tool calls.

    Args:
        state: Current agent state
        model_config: Optional model configuration

    Returns:
        Updated agent state
    """
    messages = state["messages"]

    # Lazy import to avoid circular dependency
    from src.core.tools import search_knowledge_graph, generate_final_answer

    # Get the LLM and tools
    llm = get_llm(model_config)
    tools = [search_knowledge_graph, generate_final_answer]
    llm_with_tools = llm.bind_tools(tools)

    # Get the system prompt
    prompt = get_system_prompt()

    # Create the chain
    chain = prompt | llm_with_tools

    # Invoke the chain
    response = await chain.ainvoke({"messages": messages})

    return {"messages": [response], "context": state.get("context")}
