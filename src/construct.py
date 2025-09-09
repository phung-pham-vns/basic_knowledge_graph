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


def create_fulltext_index(
    graph_client: Neo4jGraph,
    index_name: str,
    node_labels: list[str],
    property_names: list[str],
):
    """Create a full-text index for specified node labels and properties."""
    try:
        # Check if index already exists
        result = graph_client.query(
            "SHOW INDEXES YIELD name WHERE name = $index_name", params={"index_name": index_name}
        )

        if result:
            print(f"Full-text index '{index_name}' already exists, skipping creation.")
            return

        # Create full-text index
        node_labels_str = "|".join(node_labels)
        property_names_str = ", ".join([f"n.{prop}" for prop in property_names])

        create_index_query = f"""
        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
        FOR (n:{node_labels_str})
        ON EACH [{property_names_str}]
        """

        graph_client.query(create_index_query)
        print(f"Created full-text index '{index_name}' for labels {node_labels} on properties {property_names}")

    except Exception as e:
        print(f"Error creating full-text index '{index_name}': {e}")


def setup_fulltext_indices(graph_client: Neo4jGraph):
    """Set up full-text indices for searchable node properties."""
    # Create a comprehensive full-text index for all nodes with descriptions
    description_nodes = ["Crop", "Variety", "Disease", "Symptom"]
    create_fulltext_index(
        graph_client=graph_client,
        index_name="nodes_description_fulltext_index",
        node_labels=description_nodes,
        property_names=["description"],
    )

    # Create additional indices for other searchable properties
    all_searchable_nodes = [
        "Crop",
        "Variety",
        "Disease",
        "Symptom",
        "Crop_part",
        "Pathogen",
        "Condition",
        "Seasonality",
        "Location",
        "Treatment",
        "Prevention_method",
        "Spread_method",
        "Risk_factor",
    ]

    # Index for node names/IDs across all node types
    create_fulltext_index(
        graph_client=graph_client,
        index_name="nodes_name_fulltext_index",
        node_labels=all_searchable_nodes,
        property_names=["id", "name"],
    )

    # Special index for pathogen types
    create_fulltext_index(
        graph_client=graph_client,
        index_name="pathogen_type_fulltext_index",
        node_labels=["Pathogen"],
        property_names=["type"],
    )


def fulltext_search(
    graph_client: Neo4jGraph,
    query: str,
    index_name: str = "nodes_description_fulltext_index",
    limit: int = 10,
) -> list[dict]:
    """
    Perform full-text search on Neo4j graph using specified index.

    Args:
        graph_client: Neo4j graph client
        query: Search query string
        index_name: Name of the full-text index to use
        limit: Maximum number of results to return

    Returns:
        List of matching nodes with their properties and relevance scores
    """
    try:
        # Use CALL db.index.fulltext.queryNodes for full-text search
        search_query = """
        CALL db.index.fulltext.queryNodes($index_name, $query)
        YIELD node, score
        RETURN 
            labels(node) as labels,
            properties(node) as properties,
            score,
            elementId(node) as id
        ORDER BY score DESC
        LIMIT $limit
        """

        result = graph_client.query(search_query, params={"index_name": index_name, "query": query, "limit": limit})

        return result

    except Exception as e:
        print(f"Error performing full-text search: {e}")
        return []


def search_by_keywords(
    graph_client: Neo4jGraph,
    keywords: str,
    search_descriptions: bool = True,
    search_names: bool = True,
    limit: int = 20,
) -> dict:
    """
    Comprehensive keyword search across different node properties.

    Args:
        graph_client: Neo4j graph client
        keywords: Search keywords
        search_descriptions: Whether to search in description properties
        search_names: Whether to search in name/id properties
        limit: Maximum results per search type

    Returns:
        Dictionary with search results categorized by type
    """
    results = {"description_matches": [], "name_matches": [], "pathogen_type_matches": []}

    if search_descriptions:
        results["description_matches"] = fulltext_search(
            graph_client, keywords, "nodes_description_fulltext_index", limit
        )

    if search_names:
        results["name_matches"] = fulltext_search(graph_client, keywords, "nodes_name_fulltext_index", limit)

    # Also search pathogen types specifically
    results["pathogen_type_matches"] = fulltext_search(graph_client, keywords, "pathogen_type_fulltext_index", limit)

    return results


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

        # Set up search indices
        print("\n--- Setting up search indices ---")

        # Set up full-text indices for keyword search
        print("Setting up full-text indices...")
        setup_fulltext_indices(graph_client)

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
