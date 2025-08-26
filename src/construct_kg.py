import os
import getpass
import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from schema.disease_schema import node_types, allowed_relationships
from utils import load_excel
from create_prompt import create_detailed_schema_prompt


def load_environment_variables():
    """Load environment variables, prompting user if not found."""
    load_dotenv("/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/.env")

    if "LLM_PROVIDER" not in os.environ:
        os.environ["LLM_PROVIDER"] = getpass.getpass("Enter your LLM provider: ")
    if "LLM_API_KEY" not in os.environ:
        os.environ["LLM_API_KEY"] = getpass.getpass("Enter your LLM API key: ")
    if "LLM_MODEL_NAME" not in os.environ:
        os.environ["LLM_MODEL_NAME"] = getpass.getpass("Enter your LLM model name: ")

    if "NEO4J_URI" not in os.environ:
        os.environ["NEO4J_URI"] = getpass.getpass("Enter your Neo4j URI: ")
    if "NEO4J_USERNAME" not in os.environ:
        os.environ["NEO4J_USERNAME"] = getpass.getpass("Enter your Neo4j username: ")
    if "NEO4J_PASSWORD" not in os.environ:
        os.environ["NEO4J_PASSWORD"] = getpass.getpass("Enter your Neo4j password: ")

    if os.environ["LLM_PROVIDER"] == "openai":
        os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]


def initialize_neo4j_graph():
    """Initialize and return Neo4j graph connection."""
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )


async def construct_knowledge_graph(
    data_path: str, sheet_name: str, ignored_column_names: list[str] = None
):
    """Main function to construct the knowledge graph."""
    try:
        # Load environment variables
        load_environment_variables()

        # Initialize Neo4j graph
        graph = initialize_neo4j_graph()

        # Load Excel data
        documents = load_excel(
            file_path=data_path,
            sheet_name=sheet_name,
            ignored_column_names=ignored_column_names,
        )

        # Prepare schema information
        NODE_TYPES = [node_type["label"] for node_type in node_types]
        ALLOWED_RELATIONSHIPS = allowed_relationships

        # Create detailed schema prompt
        schema_prompt = create_detailed_schema_prompt()
        print("Schema prompt created successfully!")
        print(f"Schema prompt length: {len(schema_prompt)} characters")

        # Initialize LLM and transformer with enhanced configuration
        llm = ChatOpenAI(model_name=os.environ["LLM_MODEL_NAME"], temperature=0)

        llm_transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=NODE_TYPES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
        )

        # Convert documents to graph format with enhanced context
        enhanced_documents = []
        for i, doc in enumerate(documents):
            # Add schema context to each document
            enhanced_content = f"""
{schema_prompt}

## SPECIFIC INSTRUCTIONS FOR THIS DATA:
- Extract entities that match the defined node types from the agricultural disease data below
- Create relationships following the allowed patterns shown above
- Pay special attention to disease names, symptoms, pathogens, and treatments
- Ensure all extracted entities have appropriate node types and properties

## DATA TO PROCESS:
{doc}

## REMINDER:
Follow the schema exactly. Only create nodes and relationships that match the defined types and patterns.
"""
            enhanced_documents.append(Document(page_content=enhanced_content))

        print(f"Processing {len(enhanced_documents)} documents with enhanced schema...")

        # Convert to graph documents
        graph_documents = await llm_transformer.aconvert_to_graph_documents(
            enhanced_documents
        )

        # Print graph information
        total_nodes = 0
        total_relationships = 0
        for i, graph_document in enumerate(graph_documents):
            print(f"\nDocument {i+1}:")
            print(f"  Nodes ({len(graph_document.nodes)}): {graph_document.nodes}")
            print(
                f"  Relationships ({len(graph_document.relationships)}): {graph_document.relationships}"
            )
            total_nodes += len(graph_document.nodes)
            total_relationships += len(graph_document.relationships)

        print(f"\nTotal nodes created: {total_nodes}")
        print(f"Total relationships created: {total_relationships}")

        # Clear existing data if needed
        DELETE_EXISTING_DATA = True
        if DELETE_EXISTING_DATA:
            print("Clearing existing data from Neo4j...")
            graph.query("MATCH (n) DETACH DELETE n")

        # Add graph documents to Neo4j
        print("Adding graph documents to Neo4j...")
        graph.add_graph_documents(graph_documents)
        print("Knowledge graph construction completed successfully!")

    except Exception as e:
        print(f"Error constructing knowledge graph: {e}")
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        construct_knowledge_graph(
            data_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/data/durian_pest_and_disease_data.xlsx",
            sheet_name="(3) Diseases Information",
            ignored_column_names=["No.", "References"],
        )
    )
