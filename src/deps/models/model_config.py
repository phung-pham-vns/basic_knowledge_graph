"""
Model configuration for Agentic Graph RAG system
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class LLMType(Enum):
    """Supported LLM types."""

    GEMINI_FLASH = "gemini-2.0-flash"
    GEMINI_PRO = "gemini-1.5-pro"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"


@dataclass
class ModelConfig:
    """Configuration for LLM models."""

    # Main agent model
    agent_model: LLMType = LLMType.GEMINI_FLASH
    agent_temperature: float = 0.0

    # Reflection model (for quality assessment)
    reflection_model: LLMType = LLMType.GEMINI_FLASH
    reflection_temperature: float = 0.0

    # Generation model (for final answer)
    generation_model: LLMType = LLMType.GEMINI_FLASH
    generation_temperature: float = 0.1

    # Query refinement model
    refinement_model: LLMType = LLMType.GEMINI_FLASH
    refinement_temperature: float = 0.3

    # API settings
    max_tokens: Optional[int] = None
    timeout: int = 60
    max_retries: int = 3

    @classmethod
    def get_default_config(cls) -> "ModelConfig":
        """Get default model configuration."""
        return cls()

    @classmethod
    def get_high_quality_config(cls) -> "ModelConfig":
        """Get high-quality model configuration (slower but better results)."""
        return cls(
            agent_model=LLMType.GEMINI_PRO,
            reflection_model=LLMType.GEMINI_PRO,
            generation_model=LLMType.GEMINI_PRO,
            generation_temperature=0.05,
        )

    @classmethod
    def get_fast_config(cls) -> "ModelConfig":
        """Get fast model configuration (faster but potentially lower quality)."""
        return cls(
            agent_model=LLMType.GEMINI_FLASH,
            reflection_model=LLMType.GEMINI_FLASH,
            generation_model=LLMType.GEMINI_FLASH,
            generation_temperature=0.2,
            timeout=30,
        )
