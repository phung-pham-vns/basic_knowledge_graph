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
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
)
from llm_client import LLMClient


load_dotenv("/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/.env")

graphdb_provider = os.environ.get("GRAPHDB_PROVIDER", "neo4j")
graphdb_uri = os.environ.get("GRAPHDB_URI", "bolt://localhost:7687")
graphdb_user = os.environ.get("GRAPHDB_USER", "neo4j")
graphdb_password = os.environ.get("GRAPHDB_PASSWORD", "aisac_kg")

gemini_api_key = os.environ.get("LLM_API_KEY", "AIzaSyAMLUw2MSUQbDGFKDNoOBIriFKqB6MKUQI")


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_edge_facts(edges):
    text = ""
    for i, edge in enumerate(edges):
        text += f"{i + 1} - {edge.fact}\n"
    return text


def get_node_summaries(nodes):
    text = ""
    for i, node in enumerate(nodes):
        text += f"{i + 1} - {node.summary}\n"
    return text


prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

llm_generation = LLMClient(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_keys=[
        "AIzaSyCUXk6UFETEYO_4QWlof-x6F-fHW1O1aBg",
        "AIzaSyDuE2meWEZ22quM3ef84DjHDmByYOpvb3Y",
        "AIzaSyB6H_Q3BAn4sau0cKcCWi306cfUp5c2o2U",
        # "AIzaSyAMLUw2MSUQbDGFKDNoOBIriFKqB6MKUQI",
    ],
    model_id="gemini-2.5-flash",
)


def generate_answer(question: str, context: str, temperature: float = 0.7, max_tokens: int = 5000):
    prompt = prompt_template.format(question=question, context=context)
    parts = [
        {"type": "text", "text": prompt},
    ]
    return llm_generation.generate_content(
        contents=[{"role": "user", "content": parts}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


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

    search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    search_config.limit = 10
    # search_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    # search_config.limit = 5

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
                result = await graphiti_client._search(query=question, config=search_config)
                retrieved_time = time.time() - t1
                retrieved_context = get_node_summaries(result.nodes)
                t1 = time.time()
                output = generate_answer(question, retrieved_context)
                generation_time = time.time() - t1
                sample["predict"] = output
                sample["context"] = retrieved_context
                sample["retrieved_time_second"] = retrieved_time
                sample["generation_time_second"] = generation_time
                sample["successed"] = True
                print(f"Groundtruth: {sample['answer']}")
                print(f"Prediction: {output}")
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
            input_json_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/examples/data/QA_17_pest_disease_predict_with_graphiti_v3.json",
            output_json_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/examples/data/QA_17_pest_disease_predict_with_graphiti_v3.json",
        )
    )
