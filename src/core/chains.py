from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from src.core.prompts import (
    question_routing_prompt,
    retrieval_grading_prompt,
    hallucination_grading_prompt,
    answer_grading_prompt,
    question_rewriting_prompt,
    anwser_generation_prompt,
)
from src.core.schema import (
    RouteQuery,
    GradeDocuments,
    GradeHallucinations,
    GradeAnswer,
    QueryRefinement,
    GenerateAnswer,
)
from src.settings import settings

# Centralized LLM configuration
llm = ChatGoogleGenerativeAI(
    model=settings.llm.llm_model,
    temperature=settings.llm.llm_temperature,
    google_api_key=settings.llm.llm_api_key,
)

# Chain definitions
question_router = question_routing_prompt | llm.with_structured_output(RouteQuery)
retrieval_grader = retrieval_grading_prompt | llm.with_structured_output(GradeDocuments)
hallucination_grader = hallucination_grading_prompt | llm.with_structured_output(GradeHallucinations)
answer_grader = answer_grading_prompt | llm.with_structured_output(GradeAnswer)
question_rewriter = question_rewriting_prompt | llm.with_structured_output(QueryRefinement)
answer_generator = anwser_generation_prompt | llm.with_structured_output(GenerateAnswer)
