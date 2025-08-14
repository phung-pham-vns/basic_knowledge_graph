import os

from ms_graphrag import MsGraphRAG
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

ms_graph = MsGraphRAG(
    driver=driver,
    openai_model_name="gpt-4o",
    openai_api_key=openai_api_key,
    max_workers=10,
)

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
ms_graph.extract_nodes_and_relationships(texts, allowed_entities)

# Summarize Nodes and Communities
ms_graph.summarize_nodes_and_relationships()
ms_graph.summarize_communities()

# Entities
entities = ms_graph.query(
    """
MATCH (e:__Entity__)
RETURN e.name AS entity_id, e.summary AS entity_summary
"""
)

print(entities)
