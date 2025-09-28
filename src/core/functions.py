import asyncio
from typing import Optional, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults

from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER
from graphiti_core.search.search_config import SearchResults

from src.core.chains import (
    answer_generator,
    question_rewriter,
    question_router,
    retrieval_grader,
    hallucination_grader,
    answer_grader,
)


class GraphState(TypedDict):
    question: str
    generation: Optional[str]
    node_contents: list[str]
    edge_contents: list[str]
    n_documents: int
    graphiti: Graphiti


async def get_node_and_edge_contents(result: SearchResults) -> tuple[list[str], list[str]]:
    """Extract node and edge contents from search results."""
    node_contents = [node.summary for node in getattr(result, "nodes", []) if hasattr(node, "summary")]
    edge_contents = [edge.fact for edge in getattr(result, "edges", []) if hasattr(edge, "fact")]
    return node_contents, edge_contents


async def search_durian_pest_and_disease_knowledge(
    graphiti: Graphiti,
    question: str,
    limit: int = 3,
) -> tuple[list[str], list[str]]:
    """Search knowledge graph for durian pest and disease information."""
    try:
        search_type = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
        search_type.limit = limit
        results = await graphiti.search_(query=question, config=search_type)
        return await get_node_and_edge_contents(results)
    except Exception as e:
        print(f"---ERROR IN KNOWLEDGE GRAPH SEARCH: {e}---")
        return [], []


def format_context(node_contents: list[str], edge_contents: list[str]) -> str:
    """Format node and edge contents into a single string."""
    nodes = "\n".join(f"{i + 1}. {node}" for i, node in enumerate(node_contents)) if node_contents else "None"
    edges = "\n".join(f"{i + 1}. {edge}" for i, edge in enumerate(edge_contents)) if edge_contents else "None"
    return f"Node Information:\n{nodes}\n\nRelationship Information:\n{edges}"


async def knowledge_graph_retrieval(state: GraphState) -> dict:
    """Retrieve nodes and edges from the knowledge graph."""
    print("---KNOWLEDGE GRAPH RETRIEVAL---")
    graphiti = state.get("graphiti")
    if not graphiti:
        print("---ERROR: Graphiti client not initialized---")
        return {"node_contents": [], "edge_contents": []}
    node_contents, edge_contents = await search_durian_pest_and_disease_knowledge(
        graphiti=graphiti, question=state["question"], limit=state.get("n_documents", 3)
    )
    return {
        "node_contents": node_contents,
        "edge_contents": edge_contents,
        "context": format_context(node_contents, edge_contents),
    }


async def answer_generation(state: GraphState) -> dict:
    """Generate an answer using the provided context and question."""
    print("---ANSWER GENERATION---")
    context = state.get("context", format_context(state["node_contents"], state["edge_contents"]))
    try:
        generation = await answer_generator.ainvoke({"context": context, "question": state["question"]})
        return {"generation": generation.answer}
    except Exception as e:
        print(f"---ERROR IN ANSWER GENERATION: {e}---")
        return {"generation": ""}


async def nodes_and_edges_grading(state: GraphState) -> dict:
    """Grade the relevance of node and edge contents."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]

    async def filter_contents(contents: list[str], content_type: str) -> list[str]:
        tasks = [retrieval_grader.ainvoke({"question": question, "document": content}) for content in contents]
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        filtered = []
        for content, score in zip(contents, scores):
            if isinstance(score, Exception):
                print(f"---ERROR GRADING {content_type.upper()} CONTENT: {score}---")
                continue
            grade = score.binary_score
            print(f"---GRADE: {content_type.upper()} CONTENT {'RELEVANT' if grade == 'yes' else 'NOT RELEVANT'}---")
            if grade == "yes":
                filtered.append(content)
        return filtered

    node_contents = await filter_contents(state["node_contents"], "NODE")
    edge_contents = await filter_contents(state["edge_contents"], "EDGE")

    return {
        "node_contents": node_contents,
        "edge_contents": edge_contents,
        "context": format_context(node_contents, edge_contents),
    }


async def query_transformation(state: GraphState) -> dict:
    """Transform the query for better retrieval."""
    print("---QUERY TRANSFORMATION---")
    try:
        refined = await question_rewriter.ainvoke({"question": state["question"]})
        return {"question": refined.refined_question}
    except Exception as e:
        print(f"---ERROR IN QUERY TRANSFORMATION: {e}---")
        return {"question": state["question"]}


async def web_search(state: GraphState) -> dict:
    """Perform a web search to gather relevant content."""
    print("---WEB SEARCH---")
    web_search_tool = TavilySearchResults(k=3)
    try:
        docs = await web_search_tool.ainvoke({"query": state["question"]})
        web_results = "\n".join([d["content"] for d in docs])
        node_contents = [web_results] if web_results else []
        return {
            "node_contents": node_contents,
            "edge_contents": [],
            "context": format_context(node_contents, []),
        }
    except Exception as e:
        print(f"---ERROR IN WEB SEARCH: {e}---")
        return {"node_contents": [], "edge_contents": [], "context": "None"}
