from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from schema.disease_schema import (
    node_types as disease_node_types,
    allowed_relationships as disease_allowed_relationships,
)
from utils import load_document_from_excel
from schema.disease_schema import node_types, relation_types, allowed_relationships
from prompts.graph_schema_prompt import graph_schema_prompt
from prompts.entity_and_relation_extraction_prompt import entities_and_relationships_extraction_prompt
from deps.llm_client import get_llm_client
from settings import settings


graph_client = Neo4jGraph(
    url=settings.graph_db.graph_db_url,
    username=settings.graph_db.graph_db_user,
    password=settings.graph_db.graph_db_password,
)

llm_client = get_llm_client(settings)

llm_transformer = LLMGraphTransformer(
    llm=llm_client,
    allowed_nodes=[node_type["label"] for node_type in disease_node_types],
    allowed_relationships=disease_allowed_relationships,
)

disease_graph_schema = graph_schema_prompt(
    node_types,
    relation_types,
    allowed_relationships,
)


async def construct_knowledge_graph(
    data_path: str,
    sheet_name: str,
    ignored_column_names: list[str] = None,
    clear_existing_graph: bool = True,
):
    try:
        documents = load_document_from_excel(
            file_path=data_path,
            sheet_name=sheet_name,
            ignored_column_names=ignored_column_names,
        )

        # Convert documents to graph format with enhanced context
        enhanced_documents = []
        for document in documents:
            content = entities_and_relationships_extraction_prompt.invoke(
                input={
                    "graph_schema": disease_graph_schema,
                    "category": "disease",
                    "document": document,
                }
            ).text
            enhanced_documents.append(Document(page_content=content))

        # Convert to graph documents
        graph_documents = await llm_transformer.aconvert_to_graph_documents(enhanced_documents)

        # Print graph information
        total_nodes, total_relations = 0, 0
        for i, graph_document in enumerate(graph_documents):
            n_nodes = len(graph_document.nodes)
            n_relations = len(graph_document.relationships)
            print(f"\nDocument #{i+1}:")
            print(f"  Nodes ({n_nodes}): {graph_document.nodes}")
            print(f"  Relations ({n_relations}): {graph_document.relationships}")
            total_nodes += n_nodes
            total_relations += n_relations

        print(f"\nTotal nodes created: {total_nodes}")
        print(f"Total relations created: {total_relations}")

        if clear_existing_graph:
            print(f"Clearing existing data from {settings.graph_db_provider}...")
            graph_client.query("MATCH (n) DETACH DELETE n")

        print(f"Adding graph documents to {settings.graph_db_provider}...")
        graph_client.add_graph_documents(graph_documents)
        print("Knowledge graph construction completed successfully!")

    except Exception as e:
        print(f"Error constructing knowledge graph: {e}")
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        construct_knowledge_graph(
            data_path="docs/data/durian_pest_and_disease_data.xlsx",
            sheet_name="(3) Diseases Information",
            ignored_column_names=["No.", "References"],
            clear_existing_graph=True,
        )
    )
