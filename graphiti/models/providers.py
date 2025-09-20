from enum import Enum


class LLMProviders(Enum):
    OpenAI = "openai"
    Gemini = "gemini"


class EmbeddingProviders(Enum):
    OpenAI = "openai"
    Gemini = "gemini"


class GraphDBProviders(Enum):
    Neo4j = "neo4j"
    FalkorDB = "falkordb"
