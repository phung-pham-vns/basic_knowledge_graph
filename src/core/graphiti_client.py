from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.openai_client import OpenAIClient

from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

# from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver

# from graphiti_core.driver.falkordb_driver import FalkorDriver

from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from src.settings import settings
from src.models.providers import LLMProviders, GraphDBProviders, EmbeddingProviders

from src.deps.embedder.cloud_hosted_embedding import CloudHostedEmbedder, CloudHostedEmbedderConfig
from src.deps.reranker.gemini import GeminiRerankerClient


class GraphitiClient:
    def __init__(self):
        if settings.llm_provider == LLMProviders.Gemini.value:
            self.llm_client = GeminiClient(
                config=LLMConfig(
                    base_url=settings.llm_base_url,
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                )
            )
        elif settings.llm_provider == LLMProviders.OpenAI.value:
            self.llm_client = OpenAIClient(
                config=LLMConfig(
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                )
            )
        else:
            raise ValueError(f"Invalid llm provider: {settings.llm_provider}")

        if settings.reranker.reranker_provider == LLMProviders.Gemini.value:
            self.cross_encoder = GeminiRerankerClient(
                config=LLMConfig(
                    base_url=settings.reranker.reranker_base_url,
                    api_key=settings.reranker.reranker_api_key,
                    model=settings.reranker.reranker_model,
                )
            )
        elif settings.reranker.reranker_provider == LLMProviders.OpenAI.value:
            self.cross_encoder = OpenAIRerankerClient(
                config=LLMConfig(
                    api_key=settings.reranker.reranker_api_key,
                    model=settings.reranker.reranker_model,
                )
            )
        else:
            raise ValueError(f"Invalid reranker provider: {settings.reranker.reranker_provider}")

        if settings.embedding.embedding_provider == EmbeddingProviders.Gemini.value:
            self.embedder = GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=settings.embedding.embedding_api_key,
                    embedding_model=settings.embedding.embedding_model,
                    embedding_dim=settings.embedding.embedding_dimensions,
                )
            )
        elif settings.embedding.embedding_provider == EmbeddingProviders.OpenAI.value:
            self.embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=settings.embedding.embedding_api_key,
                    embedding_model=settings.embedding.embedding_model,
                    embedding_dim=settings.embedding.embedding_dimensions,
                )
            )
        else:
            raise ValueError(f"Invalid embedder provider: {settings.embedding.embedding_provider}")
        # self.embedder = CloudHostedEmbedder(
        #     config=CloudHostedEmbedderConfig(
        #         api_key=settings.embedding.embedder_api_key,
        #         embedding_model=settings.embedding.embedding_model,
        #         embedding_dim=settings.embedding.embedding_dimensions,
        #     )
        # )

        if settings.graph_db_provider == GraphDBProviders.Neo4j.value:
            self.driver = Neo4jDriver(
                uri=settings.graph_db_url,
                user=settings.graph_db_username,
                password=settings.graph_db_password,
            )
        # elif settings.graph_db_provider == GraphDBProviders.FalkorDB.value:
        #     self.driver = FalkorDriver(
        #         host=settings.graph_db_host,
        #         port=settings.graph_db_port,
        #         username=settings.graph_db_username,
        #         password=settings.graph_db_password,
        #         database=settings.graph_db_database,
        #     )
        else:
            raise ValueError(f"Invalid graphdb provider: {settings.graph_db_provider}")

    async def create_client(
        self,
        clear_existing_graphdb_data: bool = False,
        max_coroutines: int = 1,
    ):
        graphiti = Graphiti(
            graph_driver=self.driver,
            llm_client=self.llm_client,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
            max_coroutines=max_coroutines,
        )

        # Initialize the graph database with graphiti's indices
        await graphiti.build_indices_and_constraints()

        if clear_existing_graphdb_data:
            print("[GraphitiClient] Clearing existing graph data...")
            await clear_data(graphiti.driver)
            print("[GraphitiClient] Graph data cleared successfully.")

        return graphiti
