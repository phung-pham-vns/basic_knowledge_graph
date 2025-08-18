import os
import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from dotenv import load_dotenv

load_dotenv(".env")

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Connect to the Neo4j database
driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(
        neo4j_username,
        neo4j_password,
    ),
)

# List the entities and relations the LLM should look for in the text
node_types = ["Person", "House", "Planet"]
relationship_types = ["PARENT_OF", "HEIR_OF", "RULES"]
patterns = [
    ("Person", "PARENT_OF", "Person"),
    ("Person", "HEIR_OF", "House"),
    ("House", "RULES", "Planet"),
]

# Create an Embedder object
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openai_api_key,
)

# Instantiate the LLM
llm = OpenAILLM(
    api_key=openai_api_key,
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)

# Instantiate the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    schema={
        "node_types": node_types,
        "relationship_types": relationship_types,
        "patterns": patterns,
    },
    on_error="IGNORE",
    from_pdf=False,
)

# Run the pipeline on a piece of text
text = "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House Atreides, an aristocratic family that rules the planet Caladan."

asyncio.run(kg_builder.run_async(text=text))
driver.close()
