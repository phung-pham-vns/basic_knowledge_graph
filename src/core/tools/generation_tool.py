"""
Generation tool for creating final answers in Agentic Graph RAG
"""

from langchain_core.tools import tool
from src.prompts.generation_prompts import get_generation_prompt
from src.deps.models import get_generation_llm, ModelConfig


@tool
async def generate_final_answer(
    query: str,
    context: str,
    attempts: int = 1,
    reflection_feedback: str = "",
    llm_config: ModelConfig = None,
) -> str:
    """
    Generate a comprehensive final answer based on the user query and retrieved context.

    Args:
        query: The original user question
        context: The retrieved context from knowledge graph searches
        attempts: Number of retrieval attempts made
        reflection_feedback: Feedback from reflection analysis
        llm_config: Optional model configuration

    Returns:
        Comprehensive final answer
    """

    # Get the generation LLM
    llm = get_generation_llm(llm_config)

    # Get the generation prompt
    prompt = get_generation_prompt()

    # Create the chain
    chain = prompt | llm

    # Generate the response
    response = await chain.ainvoke(
        {
            "query": query,
            "context": context,
            "attempts": attempts,
            "reflection_feedback": reflection_feedback or "No reflection feedback available",
        }
    )

    return response.content.strip()
