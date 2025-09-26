"""
LLM factory for creating and managing language models
"""

import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from .model_config import ModelConfig, LLMType
from src.settings import settings


def _get_model_name(llm_type: LLMType) -> str:
    """Get the actual model name for the LLM type."""
    model_mapping = {
        LLMType.GEMINI_FLASH: "gemini-2.0-flash",
        LLMType.GEMINI_PRO: "gemini-1.5-pro",
        LLMType.GPT4: "gpt-4",
        LLMType.GPT4_TURBO: "gpt-4-turbo",
    }
    return model_mapping[llm_type]


def _get_provider(llm_type: LLMType) -> str:
    """Get the provider for the LLM type."""
    if llm_type in [LLMType.GEMINI_FLASH, LLMType.GEMINI_PRO]:
        return "google_genai"
    elif llm_type in [LLMType.GPT4, LLMType.GPT4_TURBO]:
        return "openai"
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")


def _setup_environment(llm_type: LLMType):
    """Setup environment variables for the LLM type."""
    if llm_type in [LLMType.GEMINI_FLASH, LLMType.GEMINI_PRO]:
        os.environ["GOOGLE_API_KEY"] = settings.llm_api_key
    elif llm_type in [LLMType.GPT4, LLMType.GPT4_TURBO]:
        os.environ["OPENAI_API_KEY"] = settings.llm_api_key


def create_llm(
    llm_type: LLMType, temperature: float = 0.0, max_tokens: int = None, timeout: int = 60, max_retries: int = 3
) -> BaseChatModel:
    """Create an LLM instance with the specified configuration."""

    _setup_environment(llm_type)

    model_name = _get_model_name(llm_type)
    provider = _get_provider(llm_type)

    kwargs = {
        "model": model_name,
        "model_provider": provider,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }

    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    return init_chat_model(**kwargs)


def get_llm(config: ModelConfig = None) -> BaseChatModel:
    """Get the main agent LLM."""
    if config is None:
        config = ModelConfig.get_default_config()

    return create_llm(
        llm_type=config.agent_model,
        temperature=config.agent_temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def get_reflection_llm(config: ModelConfig = None) -> BaseChatModel:
    """Get the reflection LLM for quality assessment."""
    if config is None:
        config = ModelConfig.get_default_config()

    return create_llm(
        llm_type=config.reflection_model,
        temperature=config.reflection_temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def get_generation_llm(config: ModelConfig = None) -> BaseChatModel:
    """Get the generation LLM for final answer creation."""
    if config is None:
        config = ModelConfig.get_default_config()

    return create_llm(
        llm_type=config.generation_model,
        temperature=config.generation_temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def get_refinement_llm(config: ModelConfig = None) -> BaseChatModel:
    """Get the refinement LLM for query improvement."""
    if config is None:
        config = ModelConfig.get_default_config()

    return create_llm(
        llm_type=config.refinement_model,
        temperature=config.refinement_temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )
