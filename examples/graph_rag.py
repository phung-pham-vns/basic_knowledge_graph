import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j import GraphDatabase
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv("/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/.env")


graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

loader = TextLoader(
    file_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/examples/dummytext.txt"
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=24,
)
documents = text_splitter.split_documents(documents=docs)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)
vector_retriever = vector_index.as_retriever()

driver = GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(
        os.environ["NEO4J_USERNAME"],
        os.environ["NEO4J_PASSWORD"],
    ),
)


def create_fulltext_index(tx):
    query = """
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    """
    tx.run(query)


# Function to execute the query
def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")


# Call the function to create the index
try:
    create_index()
except:
    pass

# Close the driver connection
driver.close()
