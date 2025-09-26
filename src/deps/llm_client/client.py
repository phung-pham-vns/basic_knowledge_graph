from enum import Enum

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 1


class ModelSize(Enum):
    small = "small"
    medium = "medium"


class LLMConfig:
    """
    Configuration class for the Language Learning Model (LLM).

    This class encapsulates the necessary parameters to interact with an LLM API,
    such as OpenAI's GPT models. It stores the API key, model name, and base URL
    for making requests to the LLM service.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        small_model: str | None = None,
    ):
        """
        Initialize the LLMConfig with the provided parameters.

        Args:
            api_key (str): The authentication key for accessing the LLM API.This is required for making authorized requests.
            model (str, optional): The specific LLM model to use for generating responses. Defaults to "gpt-4.1-mini".
            base_url (str, optional): The base URL of the LLM API service. Defaults to "https://api.openai.com", which is OpenAI's standard API endpoint. This can be changed if using a different provider or a custom endpoint.
            small_model (str, optional): The specific LLM model to use for generating responses of simpler prompts. Defaults to "gpt-4.1-nano".
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.small_model = small_model
        self.temperature = temperature
        self.max_tokens = max_tokens
