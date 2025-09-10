import os
from typing import Any, List, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.exceptions import LLMGenerationError
from src.settings import ProjectSettings, LLMProviders


class OpenAILLMWrapper(LLMInterface):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.client = ChatOpenAI(
            api_key=api_key,
            model_name=model_name,
            temperature=self.model_params.get("temperature", 0.0),
            max_tokens=self.model_params.get("max_tokens", 16384),
        )

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Send text input to OpenAI and get response."""
        try:
            # Prepare the prompt with system instruction if provided
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {input}"
            else:
                full_prompt = input

            # Generate response using langchain
            response = self.client.invoke(full_prompt)
            return LLMResponse(content=response.content)

        except Exception as e:
            raise LLMGenerationError(f"Error generating response from OpenAI: {str(e)}")

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Async version - for now just call the sync version."""
        return self.invoke(input, message_history, system_instruction)


class GeminiLLMWrapper(LLMInterface):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs: Any):
        super().__init__(model_name, **kwargs)
        # Ensure key is present for google genai SDK
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        self.client = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=self.model_params.get("temperature", 0.0),
            max_tokens=self.model_params.get("max_tokens", 2000),
        )

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Send text input to Gemini and get response."""
        try:
            # Prepare the prompt with system instruction if provided
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {input}"
            else:
                full_prompt = input

            # Generate response using langchain
            response = self.client.invoke(full_prompt)
            return LLMResponse(content=response.content)

        except Exception as e:
            raise LLMGenerationError(f"Error generating response from Gemini: {str(e)}")

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Async version - for now just call the sync version."""
        return self.invoke(input, message_history, system_instruction)


class OllamaLLMWrapper(LLMInterface):
    """Ollama LLM wrapper for neo4j_graphrag."""

    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434", **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.client = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=self.model_params.get("temperature", 0.0),
        )

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Send text input to Ollama and get response."""
        try:
            # Prepare the prompt with system instruction if provided
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {input}"
            else:
                full_prompt = input

            # Generate response using langchain
            response = self.client.invoke(full_prompt)
            return LLMResponse(content=response.content)

        except Exception as e:
            raise LLMGenerationError(f"Error generating response from Ollama: {str(e)}")

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Async version - for now just call the sync version."""
        return self.invoke(input, message_history, system_instruction)


class HuggingFaceLLMWrapper(LLMInterface):
    """HuggingFace LLM wrapper for neo4j_graphrag (for self-hosted models)."""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.client = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            model_kwargs={
                "temperature": self.model_params.get("temperature", 0.0),
                "max_length": self.model_params.get("max_tokens", 2000),
            },
        )

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Send text input to HuggingFace model and get response."""
        try:
            # Prepare the prompt with system instruction if provided
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {input}"
            else:
                full_prompt = input

            # Generate response using langchain
            response = self.client.invoke(full_prompt)
            return LLMResponse(content=response)

        except Exception as e:
            raise LLMGenerationError(f"Error generating response from HuggingFace: {str(e)}")

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Async version - for now just call the sync version."""
        return self.invoke(input, message_history, system_instruction)


def get_llm_client(settings: ProjectSettings) -> LLMInterface:
    provider = settings.llm.llm_provider.lower().strip()

    if provider == LLMProviders.OPENAI.value:
        return OpenAILLMWrapper(
            api_key=settings.llm.llm_api_key,
            model_name=settings.llm.llm_model,
            model_params={
                "temperature": settings.llm.llm_temperature,
                "max_tokens": settings.llm.llm_max_tokens,
            },
        )
    elif provider == LLMProviders.GEMINI.value:
        return GeminiLLMWrapper(
            api_key=settings.llm.llm_api_key,
            model_name=settings.llm.llm_model,
            model_params={
                "temperature": settings.llm.llm_temperature,
                "max_tokens": settings.llm.llm_max_tokens,
            },
        )
    elif provider == LLMProviders.OLLAMA.value:
        return OllamaLLMWrapper(
            model_name=settings.llm.llm_model,
            base_url=settings.llm.llm_endpoint or "http://localhost:11434",
            model_params={
                "temperature": settings.llm.llm_temperature,
                "max_tokens": settings.llm.llm_max_tokens,
            },
        )
    elif provider == LLMProviders.HUGGINGFACE.value:
        return HuggingFaceLLMWrapper(
            model_name=settings.llm.llm_model,
            model_params={
                "temperature": settings.llm.llm_temperature,
                "max_tokens": settings.llm.llm_max_tokens,
            },
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {settings.llm.llm_provider!r}. "
            "Expected 'openai', 'gemini', 'ollama', 'huggingface'."
        )
