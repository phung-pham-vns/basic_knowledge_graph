from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain

from settings import settings


graph_client = Neo4jGraph(
    url=settings.graph_db.graph_db_url,
    username=settings.graph_db.graph_db_user,
    password=settings.graph_db.graph_db_password,
    enhanced_schema=True,  # Add for enhanced schema
)

llm_client = ChatOpenAI(
    api_key=settings.llm.llm_api_key,
    model_name=settings.llm.llm_model,
    temperature=settings.llm.llm_temperature,
)

chain = GraphCypherQAChain.from_llm(
    graph=graph_client,
    llm=llm_client,
    verbose=True,
    allow_dangerous_requests=True,
)

print("GRAPH SCHEMA:")
print(graph_client.schema)
print("-" * 100 + "\n")

examples = [
    "Which disease in Thailand affects the most durian varieties?",
    "Which disease appears in more than two seasons in a year, and which seasons are they?",
    "Which disease usually appears on durian trees during the rainy season?",
    "If my tree has yellowing leaves, what disease could it be?",
    "Which disease appears on the most parts of the durian tree, and which parts are they?",
    "Bệnh nào ảnh hưởng đến nhiều loại giống cây trồng nhất?",
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
