import asyncio
from pathlib import Path

import neo4j
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface

from src.deps.llms import get_llm_client
from src.deps.embeddings import get_embedding_client, TextEmbedder
from src.settings import settings

root_dir = Path(__file__).parents[1]
file_path = root_dir / "scripts" / "data" / "Harry Potter and the Chamber of Secrets Summary.pdf"


NODE_TYPES = ["Person", "Organization", "Location"]
RELATIONSHIP_TYPES = ["SITUATED_AT", "INTERACTS", "LED_BY"]
PATTERNS = [
    ("Person", "SITUATED_AT", "Location"),
    ("Person", "INTERACTS", "Person"),
    ("Organization", "LED_BY", "Person"),
]


async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
    text_embedder: TextEmbedder,
) -> PipelineResult:
    # Create an instance of the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=text_embedder,
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
        },
        neo4j_database="neo4j",
    )
    return await kg_builder.run_async(file_path=str(file_path))


async def main() -> PipelineResult:
    # Get LLM and embedding clients based on settings
    llm = get_llm_client(settings)
    embedder = get_embedding_client(settings)

    with neo4j.GraphDatabase.driver(
        settings.graph_db_url, auth=(settings.graph_db_user, settings.graph_db_password)
    ) as driver:
        res = await define_and_run_pipeline(driver, llm, embedder)

    return res


if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)
