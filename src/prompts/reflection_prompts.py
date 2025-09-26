"""
Prompts for reflection and quality assessment in Agentic Graph RAG
"""

from langchain_core.prompts import ChatPromptTemplate


def get_reflection_prompt() -> ChatPromptTemplate:
    """Get the reflection prompt for evaluating retrieval quality."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert evaluator of information retrieval results. Analyze whether the retrieved context adequately addresses the original query.

Evaluation criteria:
- Relevance: Does the context directly address the query?
- Completeness: Are all aspects of the query covered?
- Quality: Is the information detailed and actionable?
- Specificity: Does it provide concrete answers rather than generic information?

Based on your analysis, determine:
1. Whether the results are sufficient to answer the query
2. What specific information is missing (if any)
3. How to refine the query for better results (if needed)

Respond in this exact format:
SUFFICIENT: [YES/NO]
QUALITY_SCORE: [0.0-1.0]
MISSING: [list specific missing aspects]
REFINED_QUERY: [improved query if refinement needed, otherwise 'N/A']
REASONING: [brief explanation of your assessment]""",
            ),
            (
                "user",
                """Original Query: {original_query}

Retrieved Context: {retrieved_context}

Current Attempt: {current_attempt}/{max_attempts}

Please evaluate the retrieved results:""",
            ),
        ]
    )


def get_query_refinement_prompt() -> ChatPromptTemplate:
    """Get the prompt for refining queries based on reflection feedback."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at refining search queries to get better results. 
Given an original query, missing information aspects, and previous results, create an improved query that:
1. Addresses the missing aspects specifically
2. Uses different keywords or synonyms
3. Is more specific or focused
4. Avoids repeating what was already found

Respond with only the refined query, nothing else.""",
            ),
            (
                "user",
                """Original Query: {original_query}

Missing Aspects: {missing_aspects}

Previous Context: {previous_context}

Please provide a refined query:""",
            ),
        ]
    )
