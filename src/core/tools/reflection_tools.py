"""
Reflection tools for quality assessment and query refinement
"""

from typing import List, Dict, Any
from src.prompts.reflection_prompts import get_reflection_prompt, get_query_refinement_prompt
from src.deps.models import get_reflection_llm, get_refinement_llm, ModelConfig


async def reflect_on_retrieval_quality(
    original_query: str,
    retrieved_context: str,
    current_attempt: int,
    max_attempts: int,
    llm_config: ModelConfig = None,
) -> Dict[str, Any]:
    """Reflect on the quality of retrieved results and determine if refinement is needed."""

    # Get the reflection LLM
    reflection_llm = get_reflection_llm(llm_config)

    # Get the reflection prompt
    reflection_prompt = get_reflection_prompt()

    chain = reflection_prompt | reflection_llm
    response = await chain.ainvoke(
        {
            "original_query": original_query,
            "retrieved_context": retrieved_context,
            "current_attempt": current_attempt,
            "max_attempts": max_attempts,
        }
    )

    # Parse the structured response
    content = response.content
    result = {
        "is_sufficient": False,
        "quality_score": 0.5,
        "missing_aspects": [],
        "refined_query": original_query,
        "reasoning": "Unable to parse reflection",
    }

    try:
        lines = content.strip().split("\n")
        for line in lines:
            if line.startswith("SUFFICIENT:"):
                result["is_sufficient"] = "YES" in line.upper()
            elif line.startswith("QUALITY_SCORE:"):
                try:
                    result["quality_score"] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("MISSING:"):
                missing_text = line.split(":", 1)[1].strip()
                if missing_text and missing_text != "None":
                    result["missing_aspects"] = [missing_text]
            elif line.startswith("REFINED_QUERY:"):
                refined = line.split(":", 1)[1].strip()
                if refined and refined != "N/A":
                    result["refined_query"] = refined
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
    except Exception as e:
        result["reasoning"] = f"Parsing error: {str(e)}"

    # Force sufficient if we've reached max attempts
    if current_attempt >= max_attempts:
        result["is_sufficient"] = True

    return result


async def refine_query_based_on_feedback(
    original_query: str, missing_aspects: List[str], previous_context: str, llm_config: ModelConfig = None
) -> str:
    """Refine the query based on reflection feedback."""

    # Get the refinement LLM
    refinement_llm = get_refinement_llm(llm_config)

    # Get the refinement prompt
    refinement_prompt = get_query_refinement_prompt()

    chain = refinement_prompt | refinement_llm
    response = await chain.ainvoke(
        {
            "original_query": original_query,
            "missing_aspects": ", ".join(missing_aspects),
            "previous_context": previous_context[:500],  # Limit context length
        }
    )

    return response.content.strip()
