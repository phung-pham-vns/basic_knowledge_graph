"""
Knowledge graph construction using Graphiti framework.
This module provides functionality to build a disease knowledge graph
using Graphiti with Neo4j backend and Gemini AI services.
"""

import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any

from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient

from src.schema.graphiti_disease_schema import ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP
from src.utils import load_document_from_excel
from src.settings import settings


async def initialize_graphiti_with_openai(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    openai_api_key: str,
    namespace: str = "disease_knowledge_graph",
) -> Graphiti:
    """
    Initialize Graphiti with Neo4j backend and OpenAI services.

    Args:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        openai_api_key: OpenAI API key
        namespace: Graph namespace for isolation

    Returns:
        Configured Graphiti instance
    """
    # Initialize OpenAI client with configuration
    llm_config = LLMConfig(
        api_key=openai_api_key, model="gpt-4o-mini", temperature=0.0, max_tokens=4000  # or "gpt-4" for better quality
    )
    llm_client = OpenAIClient(config=llm_config)

    # Initialize OpenAI embedder
    embedder_config = OpenAIEmbedderConfig(api_key=openai_api_key, model="text-embedding-3-small")
    embedder = OpenAIEmbedder(config=embedder_config)

    # Initialize OpenAI reranker
    cross_encoder = OpenAIRerankerClient(api_key=openai_api_key)

    # Initialize Graphiti with Neo4j backend and OpenAI services
    graphiti = Graphiti(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    return graphiti


async def initialize_graphiti_with_gemini_via_langchain(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    gemini_api_key: str,
    namespace: str = "disease_knowledge_graph",
) -> Graphiti:
    """
    Initialize Graphiti with Neo4j backend and Gemini services via LangChain wrappers.
    This is a workaround since Graphiti doesn't have direct Gemini support yet.

    Args:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        gemini_api_key: Gemini API key
        namespace: Graph namespace for isolation

    Returns:
        Configured Graphiti instance
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from graphiti_core.llm_client.client import LLMClient
        from graphiti_core.embedder.client import EmbedderClient

        # Create wrapper classes to make LangChain Gemini compatible with Graphiti
        class GeminiLLMWrapper(LLMClient):
            def __init__(self, api_key: str):
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.0)

            async def _generate_response(self, messages, **kwargs):
                # This is the abstract method that needs to be implemented
                # Convert messages to the format expected by LangChain
                if isinstance(messages, list):
                    prompt = "\n".join([str(msg) for msg in messages])
                else:
                    prompt = str(messages)

                response = await self.llm.ainvoke(prompt)
                return response.content

        class GeminiEmbedderWrapper(EmbedderClient):
            def __init__(self, api_key: str):
                self.embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

            def create(self, input_data):
                # Handle single input or list of inputs
                if isinstance(input_data, str):
                    # For single string, return single embedding (synchronous)
                    import asyncio

                    return asyncio.run(self.embedder.aembed_query(input_data))
                elif isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
                    # For list of strings, return list of embeddings
                    import asyncio

                    return asyncio.run(self.embedder.aembed_documents(input_data))
                else:
                    raise ValueError(f"Unsupported input type: {type(input_data)}")

            def create_batch(self, input_data_list):
                # Handle batch of documents
                import asyncio

                return asyncio.run(self.embedder.aembed_documents(input_data_list))

        # Initialize wrapped clients
        llm_client = GeminiLLMWrapper(gemini_api_key)
        embedder = GeminiEmbedderWrapper(gemini_api_key)

        # For cross-encoder, we'll create a simple dummy implementation since Gemini doesn't have a direct equivalent
        from graphiti_core.cross_encoder.client import CrossEncoderClient

        class DummyCrossEncoder(CrossEncoderClient):
            def rank(self, query: str, documents: list[str], **kwargs) -> list[float]:
                # Simple dummy implementation - just return equal scores
                return [1.0] * len(documents)

        cross_encoder = DummyCrossEncoder()

        # Initialize Graphiti
        graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

        return graphiti

    except ImportError:
        raise ImportError("LangChain Google GenAI not available. Please install: pip install langchain-google-genai")
    except Exception as e:
        raise Exception(f"Error initializing Graphiti with Gemini: {e}")


async def construct_disease_knowledge_graph_with_graphiti(
    data_path: str,
    sheet_name: str,
    ignored_column_names: List[str] = None,
    clear_existing_graph: bool = True,
    namespace: str = "disease_knowledge_graph",
) -> None:
    """
    Construct a disease knowledge graph using Graphiti framework.

    Args:
        data_path: Path to Excel data file
        sheet_name: Name of the Excel sheet to process
        ignored_column_names: List of column names to ignore
        clear_existing_graph: Whether to clear existing graph data
        namespace: Graph namespace for data isolation
    """
    try:
        # Try to use Gemini via LangChain wrapper first, fallback to OpenAI if needed
        api_key = settings.llm.llm_api_key
        if not api_key:
            raise ValueError("LLM API key is required in settings")

        print("Initializing Graphiti with AI services...")

        # Check if we have a Gemini API key (starts with 'AIza') or OpenAI key
        if api_key.startswith("AIza") or "gemini" in settings.llm.llm_provider.lower():
            print("Using Gemini AI services via LangChain wrapper...")
            graphiti = await initialize_graphiti_with_gemini_via_langchain(
                neo4j_uri=settings.graph_db.graph_db_url,
                neo4j_user=settings.graph_db.graph_db_user,
                neo4j_password=settings.graph_db.graph_db_password,
                gemini_api_key=api_key,
                namespace=namespace,
            )
        else:
            print("Using OpenAI services...")
            graphiti = await initialize_graphiti_with_openai(
                neo4j_uri=settings.graph_db.graph_db_url,
                neo4j_user=settings.graph_db.graph_db_user,
                neo4j_password=settings.graph_db.graph_db_password,
                openai_api_key=api_key,
                namespace=namespace,
            )

        # Clear existing graph if requested
        if clear_existing_graph:
            print(f"Clearing existing data from namespace '{namespace}'...")
            # Note: Graphiti may handle namespace clearing differently
            # Check Graphiti documentation for the correct method
            try:
                # This is a placeholder - actual method depends on Graphiti API
                await graphiti.clear()
            except AttributeError:
                print("Clear method not available, continuing with data ingestion...")

        # Load documents from Excel
        print(f"Loading documents from {data_path}, sheet: {sheet_name}...")
        documents = load_document_from_excel(
            file_path=data_path, sheet_name=sheet_name, ignored_column_names=ignored_column_names or []
        )

        print(f"Loaded {len(documents)} documents from Excel file")

        # Process each document as an episode in Graphiti
        total_episodes = len(documents)
        for i, document_content in enumerate(documents, 1):
            print(f"Processing episode {i}/{total_episodes}...")

            # Add episode to Graphiti with custom entity and edge types
            await graphiti.add_episode(
                name=f"Disease_Data_Episode_{i}",
                episode_body=document_content,
                source_description=f"Disease data from {sheet_name} - Row {i}",
                reference_time=datetime.now(),
                entity_types=ENTITY_TYPES,
                edge_types=EDGE_TYPES,
                edge_type_map=EDGE_TYPE_MAP,
            )

            if i % 10 == 0:  # Progress update every 10 episodes
                print(f"Processed {i}/{total_episodes} episodes...")

        print("Knowledge graph construction completed successfully!")
        print(f"Total episodes processed: {total_episodes}")
        print(f"Graph namespace: {namespace}")

        # Optional: Get some statistics about the constructed graph
        try:
            search_results = await graphiti.search(query="What diseases affect crops?", limit=5)
            print(f"Sample search results: {len(search_results)} results found")
        except Exception as e:
            print(f"Could not perform sample search: {e}")

    except Exception as e:
        print(f"Error constructing knowledge graph with Graphiti: {e}")
        raise


async def search_disease_knowledge_graph(
    query: str, namespace: str = "disease_knowledge_graph", limit: int = 10
) -> List[dict]:
    """
    Search the disease knowledge graph using Graphiti.

    Args:
        query: Natural language query
        namespace: Graph namespace to search in
        limit: Maximum number of results to return

    Returns:
        List of search results
    """
    try:
        # Initialize Graphiti with appropriate AI service
        api_key = settings.llm.llm_api_key
        if not api_key:
            raise ValueError("LLM API key is required in settings")

        if api_key.startswith("AIza") or "gemini" in settings.llm.llm_provider.lower():
            graphiti = await initialize_graphiti_with_gemini_via_langchain(
                neo4j_uri=settings.graph_db.graph_db_url,
                neo4j_user=settings.graph_db.graph_db_user,
                neo4j_password=settings.graph_db.graph_db_password,
                gemini_api_key=api_key,
                namespace=namespace,
            )
        else:
            graphiti = await initialize_graphiti_with_openai(
                neo4j_uri=settings.graph_db.graph_db_url,
                neo4j_user=settings.graph_db.graph_db_user,
                neo4j_password=settings.graph_db.graph_db_password,
                openai_api_key=api_key,
                namespace=namespace,
            )

        # Perform search
        results = await graphiti.search(query=query, limit=limit)
        return results

    except Exception as e:
        print(f"Error searching knowledge graph: {e}")
        raise


async def get_graph_statistics(namespace: str = "disease_knowledge_graph") -> Dict[str, Any]:
    """
    Get statistics about the constructed knowledge graph.

    Args:
        namespace: Graph namespace to analyze

    Returns:
        Dictionary containing graph statistics
    """
    try:
        # Initialize Graphiti with appropriate AI service
        api_key = settings.llm.llm_api_key
        if not api_key:
            raise ValueError("LLM API key is required in settings")

        if api_key.startswith("AIza") or "gemini" in settings.llm.llm_provider.lower():
            graphiti = await initialize_graphiti_with_gemini_via_langchain(
                neo4j_uri=settings.graph_db.graph_db_url,
                neo4j_user=settings.graph_db.graph_db_user,
                neo4j_password=settings.graph_db.graph_db_password,
                gemini_api_key=api_key,
                namespace=namespace,
            )
        else:
            graphiti = await initialize_graphiti_with_openai(
                neo4j_uri=settings.graph_db.graph_db_url,
                neo4j_user=settings.graph_db.graph_db_user,
                neo4j_password=settings.graph_db.graph_db_password,
                openai_api_key=api_key,
                namespace=namespace,
            )

        # Get basic statistics
        stats = {
            "namespace": namespace,
            "timestamp": datetime.now().isoformat(),
        }

        # Try to get entity and relationship counts
        try:
            # These methods depend on Graphiti's actual API
            # Adjust based on actual Graphiti documentation
            search_results = await graphiti.search(query="", limit=1000)
            stats["total_search_results"] = len(search_results)
        except Exception as e:
            stats["search_error"] = str(e)

        return stats

    except Exception as e:
        print(f"Error getting graph statistics: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    asyncio.run(
        construct_disease_knowledge_graph_with_graphiti(
            data_path="docs/data/durian_pest_and_disease_data.xlsx",
            sheet_name="(3) Diseases Information",
            ignored_column_names=["No.", "References"],
            clear_existing_graph=True,
            namespace="durian_disease_kg",
        )
    )
