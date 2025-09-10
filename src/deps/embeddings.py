import os
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.settings import ProjectSettings, EmbeddingProviders
from abc import ABC, abstractmethod


class TextEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text (str): Text to convert to vector embedding

        Returns:
            list[float]: A vector embedding.
        """


class OpenAIEmbeddingWrapper(TextEmbedder):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ):
        self.model = OpenAIEmbeddings(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        """Generate embeddings using OpenAI embedding model."""
        try:
            return self.model.embed_query(text)
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings with OpenAI: {str(e)}")


class GeminiEmbeddingWrapper(TextEmbedder):
    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004",
    ):
        # Ensure key is present for google genai SDK
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        self.model = GoogleGenerativeAIEmbeddings(model=model)

    def embed_query(self, text: str) -> list[float]:
        """Generate embeddings using Gemini embedding model."""
        try:
            return self.model.embed_query(text)
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings with Gemini: {str(e)}")


class OllamaEmbeddingWrapper(TextEmbedder):
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )

    def embed_query(self, text: str) -> list[float]:
        """Generate embeddings using Ollama embedding model."""
        try:
            return self.model.embed_query(text)
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings with Ollama: {str(e)}")


class HuggingFaceEmbeddingWrapper(TextEmbedder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs: Optional[dict] = None,
    ):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs or {},
        )

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.model.embed_query(text)
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings with HuggingFace: {str(e)}")


def get_embedding_client(settings: ProjectSettings) -> TextEmbedder:
    provider = settings.llm.embedding_provider.lower().strip()

    if provider == EmbeddingProviders.OPENAI.value:
        return OpenAIEmbeddingWrapper(
            api_key=settings.llm.llm_api_key,
            model=settings.llm.embedding_model,
            dimensions=settings.llm.embedding_dimensions,
        )
    elif provider == EmbeddingProviders.GEMINI.value:
        return GeminiEmbeddingWrapper(
            api_key=settings.llm.llm_api_key,
            model=settings.llm.embedding_model,
        )
    elif provider == EmbeddingProviders.OLLAMA.value:
        return OllamaEmbeddingWrapper(
            model=settings.llm.embedding_model,
            base_url=settings.llm.llm_endpoint or "http://localhost:11434",
        )
    elif provider == EmbeddingProviders.HUGGINGFACE.value:
        return HuggingFaceEmbeddingWrapper(
            model_name=settings.llm.embedding_model,
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: {settings.llm.embedding_provider!r}. "
            "Expected 'openai', 'gemini', 'ollama', 'huggingface'."
        )
