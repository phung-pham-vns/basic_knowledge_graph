import os
import json
import time
import asyncio
from dotenv import load_dotenv

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
            model="gemini-2.0-flash",
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

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    search_config.limit = 5

    try:
        with open(file=input_json_path, mode="r", encoding="utf-8") as file:
            data = json.load(file)

        for i, sample in enumerate(data):
            print("--" * 30)
            print(f"Sample {i + 1} / {len(data)}")
            question = sample["question"]
            print(f"Question: {question}")
            try:
                t1 = time.time()
                response = await graphiti._search(
                    query=question,
                    config=search_config,
                )
                t2 = time.time()
                print(f"Response Time: {t2 - t1} seconds")
                print_facts(response.edges)
                print("\n")
            except Exception as e:
                sample["predict"] = f"Error: {e}"
                sample["successed"] = False
                print(f"Error: {e}")
                print("\n")

            save_json(data, output_json_path)

    finally:
        await graphiti.close()
        print("\nConnection closed")


if __name__ == "__main__":
    asyncio.run(
        main(
            input_json_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/examples/data/QA_17_pest_disease.json",
            output_json_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/examples/data/QA_17_pest_disease_predict_with_graphiti.json",
        )
    )
