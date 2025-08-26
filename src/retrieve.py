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

response = chain.invoke({"query": "Bệnh nào ở Thailand ảnh hưởng tới nhiều giống cây sầu riêng nhất?"})
print(response)

# # Generated Cypher:
# MATCH (loc:Location {id: "Thailand"})<-[:OCCURS_IN]-(d:Disease)<-[:SUSCEPTIBLE_TO]-(v:Variety)
# RETURN d.id AS Disease, COUNT(DISTINCT v) AS VarietyCount
# ORDER BY VarietyCount DESC
# LIMIT 1
# Full Context:
# [{'Disease': 'Fusarium Wilt Of Durian', 'VarietyCount': 14}]
