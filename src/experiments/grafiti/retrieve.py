import os
import json
import time
import asyncio
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER


load_dotenv("/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/.env")

graphdb_provider = os.environ.get("GRAPHDB_PROVIDER", "neo4j")
graphdb_uri = os.environ.get("GRAPHDB_URI", "bolt://localhost:7687")
graphdb_user = os.environ.get("GRAPHDB_USER", "neo4j")
graphdb_password = os.environ.get("GRAPHDB_PASSWORD", "aisac_kg")

gemini_api_key = os.environ.get("LLM_API_KEY", "AIzaSyAMLUw2MSUQbDGFKDNoOBIriFKqB6MKUQI")


@dataclass
class GraphitiDependencies:
    graphiti_client: Graphiti


class GraphitiSearchResult(BaseModel):
    uuid: str = Field(description="The unique identifier for this fact")
    fact: str = Field(description="The factual statement retrieved from the knowledge graph")
    valid_at: Optional[str] = Field(None, description="When this fact became valid (if known)")
    invalid_at: Optional[str] = Field(None, description="When this fact became invalid (if known)")
    source_node_uuid: Optional[str] = Field(None, description="UUID of the source node")


graphiti_agent = Agent(
    model=GoogleModel(
        model_name="gemini-2.5-flash",
        provider=GoogleProvider(api_key=gemini_api_key),
    ),
    system_prompt="""You are an agricultural expert specializing in crops, soil, livestock, climate, pests, diseases, and farming practices.
    When the user asks a question, use your search tool to query the agricultural knowledge graph and provide a clear, accurate, and practical answer grounded in agricultural science and real-world practices.""",
    deps_type=GraphitiDependencies,
)


@graphiti_agent.tool
async def search_graphiti(ctx: RunContext[GraphitiDependencies], query: str) -> list[GraphitiSearchResult]:
    """Search the agricultural knowledge graph with the given query.

    Args:
        ctx: The run context containing dependencies
        query: The search query to find information in the knowledge graph

    Returns:
        A list of search results containing facts that match the query
    """
    # Access the Graphiti client from dependencies
    graphiti = ctx.deps.graphiti_client

    try:
        # Perform the search
        search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
        search_config.limit = 5

        results = await graphiti._search(query=query, config=search_config)

        # Format the results
        formatted_results = []
        for result in results.edges:
            formatted_result = GraphitiSearchResult(
                uuid=result.uuid,
                fact=result.fact,
                source_node_uuid=result.source_node_uuid if hasattr(result, "source_node_uuid") else None,
            )

            # Add temporal information if available
            if hasattr(result, "valid_at") and result.valid_at:
                formatted_result.valid_at = str(result.valid_at)
            if hasattr(result, "invalid_at") and result.invalid_at:
                formatted_result.invalid_at = str(result.invalid_at)

            formatted_results.append(formatted_result)
        return formatted_results
    except Exception as e:
        # Log the error but don't close the connection since it's managed by the dependency
        print(f"Error searching Graphiti: {str(e)}")
        raise


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def print_facts(edges):
    print("\n".join([edge.fact for edge in edges]))


async def main(input_json_path, output_json_path):
    driver = Neo4jDriver(
        uri=graphdb_uri,
        user=graphdb_user,
        password=graphdb_password,
    )
    llm_client = GeminiClient(
        config=LLMConfig(
            api_key=gemini_api_key,
            model="gemini-2.5-flash",
        )
    )
    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=gemini_api_key,
            embedding_model="embedding-001",
        )
    )
    cross_encoder = GeminiRerankerClient(
        config=LLMConfig(
            api_key=gemini_api_key,
            model="gemini-2.5-flash-lite-preview-06-17",
        )
    )

    graphiti_client = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    # Initialize the graph database with graphiti's indices if needed
    try:
        await graphiti_client.build_indices_and_constraints()
        print("Graphiti indices built successfully.")
    except Exception as e:
        print(f"Note: {str(e)}/nContinuing with existing indices...")

    deps = GraphitiDependencies(graphiti_client=graphiti_client)

    try:
        with open(file=input_json_path, mode="r", encoding="utf-8") as file:
            data = json.load(file)

        for i, sample in enumerate(data):
            print("--" * 30)
            print(f"Sample {i + 1} / {len(data)}")
            question = sample["question"]
            print(f"Question: {question}")

            if "successed" in sample and sample["successed"] == True:
                print(f"Answer: {sample['predict']}")
                print("\n")
                continue

            try:
                t1 = time.time()
                result = await graphiti_agent.run(question, deps=deps)
                t2 = time.time()
                sample["predict"] = result.output
                sample["tool_used"] = len(result.all_messages()) >= 4
                sample["response_time_sec"] = t2 - t1
                sample["successed"] = True
                print(f"Answer: {result.output}")
                print(f"Response Time: {t2 - t1} seconds")
                print("\n")
            except Exception as e:
                sample["predict"] = f"Error: {e}"
                sample["successed"] = False
                print(f"Error: {e}")
                print("\n")

            save_json(data, output_json_path)

    finally:
        await graphiti_client.close()
        print("\nConnection closed")


if __name__ == "__main__":
    asyncio.run(
        main(
            input_json_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/examples/data/QA_17_pest_disease_predict_with_graphiti.json",
            output_json_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/examples/data/QA_17_pest_disease_predict_with_graphiti.json",
        )
    )
