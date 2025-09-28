from typing import Literal
from pydantic import BaseModel, Field
from src.core.tools import Tools


class RouteQuery(BaseModel):
    """Route a user query to the most relevant data_source."""

    data_source: Literal[Tools.KG_RETRIEVAL.value, Tools.WEB_SEARCH.value] = Field(
        description="Route to web search or knowledge graph retrieval."
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant: 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination check in generation."""

    binary_score: str = Field(description="Answer is grounded: 'yes' or 'no'")


class GenerateAnswer(BaseModel):
    """Generate an answer to a question and given context."""

    answer: str = Field(description="Answer to the question and the given context")


class GradeAnswer(BaseModel):
    """Binary score to assess if answer addresses question."""

    binary_score: str = Field(description="Answer addresses question: 'yes' or 'no'")


class QueryRefinement(BaseModel):
    """Refined question for knowledge graph retrieval."""

    refined_question: str = Field(description="Optimized question for retrieval")
