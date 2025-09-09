import logging
from retry import retry
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from src.deps.llm_client import get_llm_client, get_embedding_client
from src.settings import settings

llm_client = get_llm_client(settings)
embedding_client = get_embedding_client(settings)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True,
)


vector_store = Neo4jVector.from_existing_index(
    embedding=embedding_client,
    url=settings.graph_db.graph_db_url,
    username=settings.graph_db.graph_db_user,
    password=settings.graph_db.graph_db_password,
    index_name="nodes_description_vector_index",
    embedding_node_property="description_embedding",
)

vector_retriever = vector_store.as_retriever()

vector_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm_client,
    chain_type="stuff",
    retriever=vector_retriever,
    memory=memory,
    reduce_k_below_max_tokens=True,
    max_tokens_limit=3000,
)


@retry(tries=2, delay=5)
def get_results(question) -> str:
    """Generate response using Neo4jVector using vector index only

    Args:
        question (str): User query

    Returns:
        str: Formatted string answer with citations, if available.
    """

    logging.info(f"Using Neo4j url: {url}")

    # Returns a dict with keys: answer, sources
    chain_result = vector_chain.invoke(
        {"question": question},
        prompt=VECTOR_PROMPT,
        return_only_outputs=True,
    )

    logging.debug(f"chain_result: {chain_result}")

    result = chain_result["answer"]

    # Cite sources, if any
    sources = chain_result["sources"]
    sources_split = sources.split(", ")
    for source in sources_split:
        if source != "" and source != "N/A" and source != "None":
            result += f"\n - [{source}]({source})"

    return result
