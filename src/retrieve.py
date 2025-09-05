from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain

from prompts.query_enhancement_prompt import query_enhancement_prompt
from deps.llm_client import get_llm_client
from settings import settings

graph_client = Neo4jGraph(
    url=settings.graph_db.graph_db_url,
    username=settings.graph_db.graph_db_user,
    password=settings.graph_db.graph_db_password,
    enhanced_schema=True,  # Add for enhanced schema
)

llm_client = get_llm_client(settings)

chain = GraphCypherQAChain.from_llm(
    graph=graph_client,
    qa_llm=llm_client,
    cypher_llm=llm_client,
    cypher_prompt=query_enhancement_prompt,
    verbose=True,
    validate_cypher=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)


if __name__ == "__main__":
    print("GRAPH SCHEMA:")
    print(graph_client.schema)
    print("-" * 100 + "\n")

    examples = [
        # "Which disease appears in more than two seasons in a year, and which seasons are they?",
        # "Which disease usually appears on durian trees during the rainy season?",
        # "If my tree has yellowing leaves, what disease could it be?",
        # "Which disease appears on the most parts of the durian tree, and which parts are they?",
        "Which disease in Thailand affects the most durian varieties?",
        "Which diseases on durian caused by Phytophthora Palmivora?",
    ]

    for query in examples:
        print(f"Question: {query}")
        response = chain.invoke({"query": query})
        print(f"Answer: {response['result']}")
        print("-" * 100)

    # MATCH p=(symptom:Symptom {id: "Unilateral Yellowing"})<-[:HAS_SYMPTOM]-(crop_part:Crop_part)-[:HAS_DISEASE]->(disease:Disease)
    # RETURN p;

    # MATCH p=(d:Disease)<-[:HAS_DISEASE]-(cp:Crop_part)
    # RETURN p;
