from graphiti_core import Graphiti

# LLM Client
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.openai_client import OpenAIClient

# Embedder
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

# Cross Encoder
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# GraphDB
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.driver.falkordb_driver import FalkorDriver

from graphiti_core.nodes import EpisodeType

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# -----------------------
# ENV / CONFIG
# -----------------------
load_dotenv("/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/.env")

# Safer defaults
llm_provider = os.environ.get("LLM_PROVIDER", "openai")
llm_model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")

embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "openai")
embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

graphdb_provider = os.environ.get("GRAPHDB_PROVIDER", "neo4j")
graphdb_uri = os.environ.get("GRAPHDB_URI", "bolt://localhost:7687")
graphdb_user = os.environ.get("GRAPHDB_USER", "neo4j")
graphdb_password = os.environ.get("GRAPHDB_PASSWORD", "aisac_kg")

api_key = os.environ.get("LLM_API_KEY", "AIzaSyAMLUw2MSUQbDGFKDNoOBIriFKqB6MKUQI")

# Cleanup behavior flags
CLEAN_GRAPH_DATA = os.environ.get("CLEAN_GRAPH_DATA", "true").lower() == "true"
RESET_SCHEMA = os.environ.get("RESET_SCHEMA", "false").lower() == "true"  # drops constraints & indexes

json_paths = list(
    Path("/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/examples/data/pest_and_disease").glob("*.json")
)

# -----------------------
# DRIVER INIT
# -----------------------
if graphdb_provider == "neo4j":
    driver = Neo4jDriver(uri=graphdb_uri, user=graphdb_user, password=graphdb_password)
elif graphdb_provider == "falkordb":
    driver = FalkorDriver(uri=graphdb_uri, user=graphdb_user, password=graphdb_password)
else:
    raise ValueError(f"Invalid graphdb provider: {graphdb_provider}")

# -----------------------
# LLM / EMBEDDERS INIT
# -----------------------
if llm_provider == "gemini":
    llm_client = GeminiClient(config=LLMConfig(api_key=api_key, model="gemini-2.0-flash"))
    embedder = GeminiEmbedder(config=GeminiEmbedderConfig(api_key=api_key, embedding_model="embedding-001"))
    cross_encoder = GeminiRerankerClient(config=LLMConfig(api_key=api_key, model="gemini-2.5-flash-lite-preview-06-17"))
elif llm_provider == "openai":
    llm_client = OpenAIClient(config=LLMConfig(api_key=api_key, model=llm_model))
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key=api_key, embedding_model=embedding_model))
    cross_encoder = OpenAIRerankerClient(config=LLMConfig(api_key=api_key, model=llm_model))
else:
    raise ValueError(f"Invalid llm provider: {llm_provider}")


# -----------------------
# GRAPH CLEANUP HELPERS
# -----------------------
async def clear_graph_data_only(driver):
    """
    Deletes all nodes and relationships from the graph.
    Works for Neo4j and FalkorDB (openCypher-compatible).
    """
    print("Clearing graph data (nodes & relationships)...")
    async with driver.session() as session:
        # Most robust approach for large graphs:
        # delete in batches to avoid transaction timeouts / memory spikes
        while True:
            # Delete up to N nodes per batch
            result = await session.run(
                """
                CALL {
                    MATCH (n)
                    WITH n LIMIT 10000
                    DETACH DELETE n
                    RETURN count(*) AS deleted
                }
                RETURN deleted
            """
            )
            records = [r async for r in result]
            deleted = records[0]["deleted"] if records else 0
            print(f" - Deleted batch size: {deleted}")
            if deleted == 0:
                break
    print("Graph data cleared.")


async def reset_schema(driver, provider: str):
    """
    Drops all constraints and indexes.
    Currently implemented for Neo4j.
    For FalkorDB, schema objects are minimal; this is a no-op.
    """
    if provider != "neo4j":
        print("Schema reset skipped (only implemented for Neo4j).")
        return

    print("Resetting schema (dropping constraints & indexes) on Neo4j...")
    async with driver.session() as session:
        # Drop constraints
        constraints = []
        result = await session.run("SHOW CONSTRAINTS YIELD name RETURN name")
        async for rec in result:
            constraints.append(rec["name"])
        for name in constraints:
            print(f" - Dropping constraint: {name}")
            await session.run(f"DROP CONSTRAINT {name} IF EXISTS")

        # Drop indexes
        indexes = []
        result = await session.run("SHOW INDEXES YIELD name RETURN name")
        async for rec in result:
            indexes.append(rec["name"])
        for name in indexes:
            print(f" - Dropping index: {name}")
            await session.run(f"DROP INDEX {name} IF EXISTS")

    print("Schema reset complete.")


# -----------------------
# MAIN INGESTION
# -----------------------
async def main(json_paths, llm_client, embedder, cross_encoder):
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        # 1) Optional cleanup
        if CLEAN_GRAPH_DATA:
            await clear_graph_data_only(driver)

        if RESET_SCHEMA:
            await reset_schema(driver, graphdb_provider)

        # 2) Rebuild indices & constraints required by Graphiti
        await graphiti.build_indices_and_constraints()
        print("Database prepared for new ingestion.")

        # 3) Load and normalize samples
        total_chunks = 0
        samples = defaultdict(list)
        for json_path in json_paths:
            file_name = json_path.name
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                total_chunks += len(data)
                print(f"Loading {json_path} - {len(data)} items")
                for item in data:
                    if item.get("text") is not None:
                        # keep original if present, fallback to translated if that's your intent
                        content = item["text"].get("text_translated") or item["text"].get("content") or ""
                        if content:
                            samples[file_name].append({"text": content})
                    if item.get("image") is not None:
                        caption = item["image"].get("image_caption")
                        if caption:
                            samples[file_name].append({"text": caption})
                    if item.get("table") is not None:
                        table_txt = item["table"].get("content")
                        if table_txt:
                            samples[file_name].append({"text": table_txt})

        # 4) Ingest episodes
        successful_chunks = 0
        failed_chunks = 0

        count_chunk = 0
        for i, (file_name, chunks) in enumerate(samples.items()):
            for j, chunk in enumerate(chunks):
                count_chunk += 1
                try:
                    await graphiti.add_episode(
                        name=f"{file_name}_chunk_{j}",
                        episode_body=chunk["text"],
                        source_description=f"{file_name}_chunk_{j}",
                        reference_time=datetime.now(),
                        source=EpisodeType.text,
                        # Optional domain-specific types:
                        # entity_types=ENTITY_TYPES,
                        # edge_types=EDGE_TYPES,
                        # edge_type_map=EDGE_TYPE_MAP,
                    )
                    print(
                        f"Chunk #{count_chunk} / {total_chunks} ✓ Successfully processed chunk {j} in document {file_name}"
                    )
                    successful_chunks += 1
                except Exception as e:
                    print(
                        f"Chunk #{count_chunk} / {total_chunks} ✗ Error processing chunk {j} in document {file_name}: {str(e)}"
                    )
                    failed_chunks += 1
                    continue

        total = successful_chunks + failed_chunks
        rate = (successful_chunks / total * 100.0) if total else 0.0
        print(f"Successful chunks: {successful_chunks}")
        print(f"Failed chunks: {failed_chunks}")
        print(f"Total chunks: {total}")
        print(f"Chunk success rate: {rate:.2f}%")

    finally:
        # Ensure resources are closed
        await graphiti.close()
        print("\nConnection closed")


if __name__ == "__main__":
    asyncio.run(main(json_paths, llm_client, embedder, cross_encoder))
