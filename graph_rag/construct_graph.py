import os
import asyncio

from ms_graphrag import MsGraphRAG
from qdrant_client import QdrantClient, models
from openai import OpenAI
import uuid
from neo4j import GraphDatabase
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/.env")

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
openai_api_key = os.environ.get("OPENAI_API_KEY")


driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(
        os.environ["NEO4J_USERNAME"],
        os.environ["NEO4J_PASSWORD"],
    ),
)


async def main():
    async with MsGraphRAG(
        driver=driver,
        openai_model_name="gpt-4.1-mini",
        openai_api_key=openai_api_key,
        max_workers=10,
    ) as ms_graph:
        # Load data
        import pandas as pd

        # Login using e.g. `huggingface-cli login` to access this dataset
        df = pd.read_parquet(
            "hf://datasets/weaviate/agents/query-agent/financial-contracts/0001.parquet"
        )
        df.head()

        texts = [el["contract_text"] for el in df["properties"]]

        # Extract Relevant Entities
        allowed_entities = ["Person", "Organization", "Location"]
        await ms_graph.extract_nodes_and_relationships(texts[:2], allowed_entities)

        # Summarize Nodes and Communities
        await ms_graph.summarize_nodes_and_relationships()
        await ms_graph.summarize_communities()

        #         # Entities
        #         entities = ms_graph.query(
        #             """
        #  MATCH (e:__Entity__)
        #  RETURN e.name AS entity_id, e.summary AS entity_summary
        #  """
        #         )

        #         print(entities)

        # Ingest entities into Qdrant vector DB
        qdrant_key = os.getenv("QDRANT_KEY")
        qdrant_uri = os.getenv("QDRANT_URI")
        collection_name = "graphrag_entities"
        vector_dimension = 1536  # text-embedding-3-small

        qdrant_client = QdrantClient(url=qdrant_uri, api_key=qdrant_key)
        openai_client = OpenAI(api_key=openai_api_key)

        # Ensure collection exists
        try:
            qdrant_client.get_collection(collection_name)
        except Exception as e:
            if "Not found: Collection" in str(e):
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_dimension, distance=models.Distance.COSINE
                    ),
                )
            else:
                raise

        # Build vectors and upsert
        points = []
        for row in entities:
            name = row.get("entity_id")
            if not name:
                continue
            # Qdrant requires point ids to be integer or UUID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"graphrag:{name}"))
            emb = (
                openai_client.embeddings.create(
                    input=name, model="text-embedding-3-small"
                )
                .data[0]
                .embedding
            )
            points.append(
                {
                    "id": point_id,
                    "vector": emb,
                    "payload": {
                        "id": point_id,
                        "name": name,
                        "summary": row.get("entity_summary"),
                    },
                }
            )
        if points:
            qdrant_client.upsert(collection_name=collection_name, points=points)


if __name__ == "__main__":
    asyncio.run(main())
