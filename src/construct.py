from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from schema.disease_schema import (
    node_types as disease_node_types,
    allowed_relationships as disease_allowed_relationships,
    relation_types as disease_relation_types,
)
from utils import load_document_from_excel
from schema.disease_schema import node_types, relation_types, allowed_relationships
from prompts.graph_schema_prompt import graph_schema_prompt
from prompts.entity_and_relation_extraction_prompt import entities_and_relationships_extraction_prompt
from deps.llm_client import get_llm_client, get_embedding_client
from settings import settings


graph_client = Neo4jGraph(
    url=settings.graph_db.graph_db_url,
    username=settings.graph_db.graph_db_user,
    password=settings.graph_db.graph_db_password,
)

llm_client = get_llm_client(settings)
embedding_client = get_embedding_client(settings)

llm_transformer = LLMGraphTransformer(
    llm=llm_client,
    allowed_nodes=[node_type["label"] for node_type in disease_node_types],
    allowed_relationships=disease_allowed_relationships,
    node_properties=[
        property["name"]
        for node_type in disease_node_types
        if len(node_type.get("properties", [])) > 0
        for property in node_type["properties"]
    ],
    # relationship_properties=[
    #     property["name"]
    #     for relationship_type in disease_relation_types
    #     if len(relationship_type.get("properties", [])) > 0
    #     for property in relationship_type["properties"]
    # ],
)

disease_graph_schema = graph_schema_prompt(
    node_types,
    relation_types,
    allowed_relationships,
)


def create_vector_index(
    graph_client: Neo4jGraph,
    index_name: str,
    node_label: str,
    property_name: str,
    dimensions: int,
):
    """Create a vector index for a specific node label and property."""
    try:
        # Check if index already exists
        result = graph_client.query(
            "SHOW INDEXES YIELD name WHERE name = $index_name", params={"index_name": index_name}
        )

        if result:
            print(f"Vector index '{index_name}' already exists, skipping creation.")
            return

        # Create vector index
        create_index_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{node_label})
        ON n.{property_name}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """

        graph_client.query(create_index_query)
        print(f"Created vector index '{index_name}' for {node_label}.{property_name}")

    except Exception as e:
        print(f"Error creating vector index '{index_name}': {e}")


def get_nodes_with_descriptions(graph_client: Neo4jGraph):
    """Get all nodes that have description properties."""
    query = """
    MATCH (n)
    WHERE n.description IS NOT NULL AND n.description <> ''
    RETURN DISTINCT labels(n) as labels, n.description as description, elementId(n) as id
    """

    result = graph_client.query(query)
    return result


async def embed_node_descriptions(
    graph_client: Neo4jGraph,
    embedding_client,
    batch_size: int = 50,
):
    """Embed all node descriptions and store them as vector properties."""
    try:
        # Get all nodes with descriptions
        nodes_with_descriptions = get_nodes_with_descriptions(graph_client)

        if not nodes_with_descriptions:
            print("No nodes with descriptions found.")
            return

        print(f"Found {len(nodes_with_descriptions)} nodes with descriptions to embed.")

        # Process nodes in batches
        for i in range(0, len(nodes_with_descriptions), batch_size):
            batch = nodes_with_descriptions[i : i + batch_size]
            descriptions = [node["description"] for node in batch]

            # Generate embeddings for the batch
            embeddings = await embedding_client.aembed_documents(descriptions)

            # Update nodes with embeddings
            for j, node in enumerate(batch):
                embedding = embeddings[j]
                node_id = node["id"]

                update_query = """
                MATCH (n)
                WHERE elementId(n) = $node_id
                SET n.description_embedding = $embedding
                """

                graph_client.query(update_query, params={"node_id": node_id, "embedding": embedding})

            print(
                f"Embedded descriptions for batch {i//batch_size + 1}/{(len(nodes_with_descriptions) + batch_size - 1)//batch_size}"
            )

        print("Successfully embedded all node descriptions.")

    except Exception as e:
        print(f"Error embedding node descriptions: {e}")
        raise


def setup_vector_indices(graph_client: Neo4jGraph, dimensions: int = 1536):
    """Set up vector indices for all node types that have descriptions."""
    # Node types that have description properties
    node_types_with_descriptions = ["Crop", "Variety", "Disease", "Symptom"]

    for node_label in node_types_with_descriptions:
        index_name = f"{node_label.lower()}_description_vector_index"
        create_vector_index(
            graph_client=graph_client,
            index_name=index_name,
            node_label=node_label,
            property_name="description_embedding",
            dimensions=dimensions,
        )


async def construct_knowledge_graph(
    data_path: str,
    sheet_name: str,
    ignored_column_names: list[str] = None,
    clear_existing_graph: bool = True,
    embed_descriptions: bool = True,
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

        # Embed node descriptions if requested
        if embed_descriptions:
            print("\n--- Starting node description embedding ---")

            # Set up vector indices
            print("Setting up vector indices...")
            setup_vector_indices(graph_client, dimensions=settings.llm.embedding_dimensions)

            # Embed all node descriptions
            print("Embedding node descriptions...")
            await embed_node_descriptions(graph_client, embedding_client)

            print("Node description embedding completed successfully!")

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
            embed_descriptions=True,
        )
    )
