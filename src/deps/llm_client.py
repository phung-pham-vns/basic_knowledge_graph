import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from settings import ProjectSettings


def get_llm_client(settings: ProjectSettings):
    provider = settings.llm.llm_provider.lower().strip()

    if provider == "openai":
        return ChatOpenAI(
            api_key=settings.llm.llm_api_key,
            model_name=settings.llm.llm_model,
            temperature=settings.llm.llm_temperature,
        )
    elif provider == "gemini":
        # Ensure key is present for google genai SDK
        os.environ.setdefault("GOOGLE_API_KEY", settings.llm.llm_api_key)
        return ChatGoogleGenerativeAI(
            model=settings.llm.llm_model,
            temperature=settings.llm.llm_temperature,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm.llm_provider!r}. " "Expected 'openai' or 'gemini'.")


def get_embedding_client(settings: ProjectSettings):
    """Get embedding client based on provider settings."""
    provider = settings.llm.embedding_provider.lower().strip()

    if provider == "openai":
        return OpenAIEmbeddings(
            api_key=settings.llm.llm_api_key,
            model=settings.llm.embedding_model,
            dimensions=settings.llm.embedding_dimensions,
        )
    elif provider == "gemini":
        # Ensure key is present for google genai SDK
        os.environ.setdefault("GOOGLE_API_KEY", settings.llm.llm_api_key)
        return GoogleGenerativeAIEmbeddings(
            model=settings.llm.embedding_model,
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: {settings.llm.embedding_provider!r}. " "Expected 'openai' or 'gemini'."
        )
