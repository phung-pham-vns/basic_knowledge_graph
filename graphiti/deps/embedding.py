from math import e
from openai import OpenAI


class OpenAIEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        api_keys: str | list[str],
        model_id: str = "gemini-embedding-001",
        dimensions: int = 3072,
    ):
        if not api_keys or len(api_keys) == 0:
            raise ValueError("API Keys are required.")

        if isinstance(api_keys, str):
            api_keys = [api_keys]

        self.clients = [OpenAI(base_url=base_url, api_key=api_key) for api_key in api_keys]
        self.current_client_index = 0
        self.model_id = model_id
        self.dimensions = dimensions

    @property
    def client(self) -> OpenAI:
        client = self.clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client

    def embedding(self, input: str) -> list[float]:
        response = self.client.embeddings.create(
            input=input,
            model=self.model_id,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding


if __name__ == "__main__":
    client = OpenAIEmbeddingClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_keys=["<your-api-key-1>"],
        model_id="gemini-embedding-001",
    )

    embedding = client.embedding(input="Hello, how are you?")
    print("dimensions", len(embedding))
    print("embedding", embedding)
