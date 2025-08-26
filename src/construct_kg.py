import os
import getpass
import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from schema.disease_schema import node_types, allowed_relationships


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


def load_excel(
    file_path: str, sheet_name: str, ignored_column_names: list[str] = None
) -> list[str]:
    """Load Excel file and convert rows to text format."""
    if ignored_column_names is None:
        ignored_column_names = []

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    row_infos = []
    for _, row in df.iterrows():
        text_parts = []

        column_id = 0
        for column in df.columns:
            if column in ignored_column_names:
                continue

            if pd.notna(row[column]) and str(row[column]).strip():
                column_id += 1
                text_parts.append(f"{column_id}. `{column}`: {row[column]}")

        full_text = "\n".join(text_parts)
        row_infos.append(full_text)

    return row_infos


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

        # Initialize LLM and transformer
        llm = ChatOpenAI(model_name=os.environ["LLM_MODEL_NAME"], temperature=0)
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=NODE_TYPES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
        )

        # Convert documents to graph format
        documents = [Document(page_content=document) for document in documents]
        graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)

        # Print graph information
        for graph_document in graph_documents:
            print(f"Nodes ({len(graph_document.nodes)}): {graph_document.nodes}")
            print(
                f"Relationships ({len(graph_document.relationships)}): {graph_document.relationships}"
            )

        # Clear existing data if needed
        DELETE_EXISTING_DATA = True
        if DELETE_EXISTING_DATA:
            graph.query("MATCH (n) DETACH DELETE n")

        # Add graph documents to Neo4j
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
