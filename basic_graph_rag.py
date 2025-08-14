from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import uuid
import os
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

# Get credentials and config from environment variables
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_uri = os.getenv("QDRANT_URI")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
openai_key = os.getenv("OPENAI_API_KEY")

# Model config (override via env if desired)
OPENAI_JSON_MODEL = os.getenv("OPENAI_JSON_MODEL", "gpt-4o-2024-08-06")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-2024-08-06")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_username, neo4j_password),
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_uri,
    api_key=qdrant_key,
)


class RelationshipTriple(BaseModel):
    node: str
    target_node: str
    relationship: str


class GraphComponents(BaseModel):
    graph: List[RelationshipTriple]


client = OpenAI(api_key=openai_key)


def openai_llm_parser(prompt: str) -> GraphComponents:
    completion = client.chat.completions.create(
        model=OPENAI_JSON_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """ You are a precise graph relationship extractor. Extract all
                    relationships from the text and format them as a JSON object
                    with this exact structure:
                    {
                        "graph": [
                            {"node": "Person/Entity",
                             "target_node": "Related Entity",
                             "relationship": "Type of Relationship"}
                        ]
                    }
                    Include ALL relationships mentioned in the text, including
                    implicit ones. Be thorough and precise. """,
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = completion.choices[0].message.content or "{}"
    return GraphComponents.model_validate_json(content)


def extract_graph_components(
    raw_data: str,
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

    parsed = openai_llm_parser(prompt)
    triples = parsed.graph

    nodes: Dict[str, str] = {}
    relationships: List[Dict[str, str]] = []

    for entry in triples:
        node = entry.node.strip()
        target_node = entry.target_node.strip() if entry.target_node else None
        relationship = entry.relationship.strip() if entry.relationship else None

        if not node or not target_node or not relationship:
            continue

        if node not in nodes:
            nodes[node] = str(uuid.uuid4())
        if target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        relationships.append(
            {
                "source": nodes[node],
                "target": nodes[target_node],
                "type": relationship,
            }
        )

    return nodes, relationships


def ingest_to_neo4j(
    nodes: Dict[str, str], relationships: List[Dict[str, str]]
) -> Dict[str, str]:
    with neo4j_driver.session() as session:
        # Ensure unique constraint for Entity.id
        session.run(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )

        # Upsert nodes
        for name, node_id in nodes.items():
            session.run(
                "MERGE (n:Entity {id: $id})\n"
                "ON CREATE SET n.name = $name\n"
                "ON MATCH SET n.name = coalesce(n.name, $name)",
                id=node_id,
                name=name,
            )

        # Upsert relationships
        for relationship in relationships:
            session.run(
                "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})\n"
                "MERGE (a)-[r:RELATIONSHIP {type: $type}]->(b)",
                source_id=relationship["source"],
                target_id=relationship["target"],
                type=relationship["type"],
            )

    return nodes


def create_collection(client, collection_name, vector_dimension):
    # Try to fetch the collection status
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        # If collection does not exist, an error will be thrown, so we create the collection
        if "Not found: Collection" in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension, distance=models.Distance.COSINE
                ),
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")


def openai_embeddings(text: str) -> List[float]:
    response = client.embeddings.create(input=text, model=OPENAI_EMBEDDING_MODEL)
    return response.data[0].embedding


def ingest_to_qdrant(collection_name: str, node_id_mapping: Dict[str, str]) -> None:
    # Create one vector per node using the node's name for retrieval
    points = []
    for node_name, node_id in node_id_mapping.items():
        vector = openai_embeddings(node_name)
        points.append(
            {
                "id": node_id,
                "vector": vector,
                "payload": {"id": node_id, "name": node_name},
            }
        )

    qdrant_client.upsert(collection_name=collection_name, points=points)


def retriever_search(collection_name: str, query: str, top_k: int = 5):
    # Directly query Qdrant and return scored points with payloads
    return qdrant_client.search(
        collection_name=collection_name,
        query_vector=openai_embeddings(query),
        limit=top_k,
        with_payload=True,
    )


def fetch_related_graph(neo4j_client, entity_ids: List[str]):
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        for record in result:
            subgraph.append(
                {
                    "entity": record["e"],
                    "relationship": record["r"],
                    "related_node": record["related"],
                }
            )
            if record["r2"] and record["n2"]:
                subgraph.append(
                    {
                        "entity": record["related"],
                        "relationship": record["r2"],
                        "related_node": record["n2"],
                    }
                )
    return subgraph


def format_graph_context(subgraph: List[Dict]) -> Dict[str, List[str]]:
    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    return {"nodes": list(nodes), "edges": edges}


def graphRAG_run(graph_context: Dict[str, List[str]], user_query: str) -> str:
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """

    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Provide the answer for the following question:",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    except Exception as e:
        return f"Error querying LLM: {str(e)}"


if __name__ == "__main__":
    print("Script started")
    print("Loading environment variables...")
    load_dotenv(".env.local")
    print("Environment variables loaded")

    print("Initializing clients...")
    # Clients are already initialized globally using env vars
    print("Clients initialized")

    print("Creating collection...")
    collection_name = "graphRAGstoreds"
    vector_dimension = 1536
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified")

    print("Extracting graph components...")

    raw_data = """Alice is a data scientist at TechCorp's Seattle office.
    Bob and Carol collaborate on the Alpha project.
    Carol transferred to the New York office last year.
    Dave mentors both Alice and Bob.
    TechCorp's headquarters is in Seattle.
    Carol leads the East Coast team.
    Dave started his career in Seattle.
    The Alpha project is managed from New York.
    Alice previously worked with Carol at DataCo.
    Bob joined the team after Dave's recommendation.
    Eve runs the West Coast operations from Seattle.
    Frank works with Carol on client relations.
    The New York office expanded under Carol's leadership.
    Dave's team spans multiple locations.
    Alice visits Seattle monthly for team meetings.
    Bob's expertise is crucial for the Alpha project.
    Carol implemented new processes in New York.
    Eve and Dave collaborated on previous projects.
    Frank reports to the New York office.
    TechCorp's main AI research is in Seattle.
    The Alpha project revolutionized East Coast operations.
    Dave oversees projects in both offices.
    Bob's contributions are mainly remote.
    Carol's team grew significantly after moving to New York.
    Seattle remains the technology hub for TechCorp."""

    nodes, relationships = extract_graph_components(raw_data)
    print("Nodes:", nodes)
    print("Relationships:", relationships)

    print("Ingesting to Neo4j...")
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Neo4j ingestion complete")

    print("Ingesting to Qdrant...")
    ingest_to_qdrant(collection_name, node_id_mapping)
    print("Qdrant ingestion complete")

    query = "How is Bob connected to New York?"
    print("Starting retriever search...")
    retriever_result = retriever_search(collection_name, query)
    print("Retriever results:", retriever_result)

    print("Extracting entity IDs...")
    entity_ids = []
    for item in retriever_result:
        payload = getattr(item, "payload", {}) or {}
        id_from_payload = payload.get("id")
        if id_from_payload:
            entity_ids.append(id_from_payload)
        else:
            # Fallback to the point id if payload missing
            entity_ids.append(str(getattr(item, "id", "")))
    print("Entity IDs:", entity_ids)

    print("Fetching related graph...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Subgraph:", subgraph)

    print("Formatting graph context...")
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)

    print("Running GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)
