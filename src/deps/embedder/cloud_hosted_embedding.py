import logging
from pydantic import Field
from openai import AsyncOpenAI
from typing import Union, List, Iterable

from graphiti_core.embedder import EmbedderClient
from graphiti_core.embedder.client import EmbedderConfig

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_BATCH_SIZE = 100
GEMINI_BATCH_LIMIT = 1


class CloudHostedEmbedderConfig(EmbedderConfig):
    embedding_base_url: str = Field(default=DEFAULT_BASE_URL, frozen=True)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    embedding_max_tokens: int | None = Field(default=None, description="Context Windown Size")
    api_key: str | None = Field(default=None)


class CloudHostedEmbedder(EmbedderClient):
    def __init__(self, config: CloudHostedEmbedderConfig, batch_size: int | None = None):
        if not config.api_key:
            raise ValueError("API key is required for GeminiEmbedder")

        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.embedding_base_url,
            api_key=config.api_key,
        )
        self.batch_size = self._determine_batch_size(batch_size, config.embedding_model)

    def _determine_batch_size(self, batch_size: int | None, model: str) -> int:
        """Determine the appropriate batch size based on model and input."""
        if batch_size is not None:
            return batch_size
        return GEMINI_BATCH_LIMIT if model == "gemini-embedding-001" else DEFAULT_BATCH_SIZE

    async def create(self, input_data: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]) -> List[float]:
        """Create embeddings for a single input."""
        try:
            response = await self.client.embeddings.create(
                input=input_data,
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dim,
            )
            if not response.data or not response.data[0].embedding:
                raise ValueError("No valid embedding returned from Gemini API")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of inputs with fallback to individual processing."""
        if not input_data_list:
            return []

        all_embeddings = []
        for i in range(0, len(input_data_list), self.batch_size):
            batch = input_data_list[i : i + self.batch_size]
            embeddings = await self._process_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a single batch of inputs, with fallback to individual processing on failure."""
        try:
            response = await self.client.embeddings.create(
                input=batch,
                model=self.config.embedding_model,
                # dimensions=self.config.embedding_dim,
            )
            if not response.data:
                raise ValueError("No embeddings returned from Gemini API")

            embeddings = [data.embedding for data in response.data]
            if any(not emb for emb in embeddings):
                raise ValueError("Empty embedding values returned")
            return embeddings

        except Exception as e:
            logger.warning(f"Batch processing failed: {e}. Falling back to individual processing")
            return await self._process_individual_items(batch)

    async def _process_individual_items(self, batch: List[str]) -> List[List[float]]:
        """Process each item individually as a fallback."""
        embeddings = []
        for item in batch:
            try:
                embedding = await self.create(item)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed individual item: {e}")
                raise
        return embeddings
