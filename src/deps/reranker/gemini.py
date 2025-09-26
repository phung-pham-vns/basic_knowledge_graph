import re
import logging
from openai import AsyncOpenAI

from ..helpers import semaphore_gather

from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.helpers import semaphore_gather
from graphiti_core.cross_encoder import CrossEncoderClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-06-17"


class GeminiRerankerClient(CrossEncoderClient):
    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize the GeminiRerankerClient with the provided configuration and client.

        The Gemini Developer API does not yet support logprobs. Unlike the OpenAI reranker,
        this reranker uses the Gemini API to perform direct relevance scoring of passages.
        Each passage is scored individually on a 0-100 scale.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            client (genai.Client | None): An optional async client instance to use. If not provided, a new genai.Client is created.
        """
        if config is None:
            config = LLMConfig()

        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages based on their relevance to the query using direct scoring.

        Each passage is scored individually on a 0-100 scale, then normalized to [0,1].
        """
        system_instruction = "You are an expert at rating passage relevance. Respond with only a number from 0-100."
        if len(passages) <= 1:
            return [(passage, 1.0) for passage in passages]

        # Generate scoring prompts for each passage
        scoring_prompts = []
        for passage in passages:
            prompt = f"""Rate how well this passage answers or relates to the query. Use a scale from 0 to 100.

Query: {query}

Passage: {passage}

Provide only a number between 0 and 100 (no explanation, just the number):"""

            scoring_prompts.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_instruction},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            )

        try:
            # Execute all scoring requests concurrently - O(n) API calls
            responses = await semaphore_gather(
                *[
                    self.client.chat.completions.create(
                        model=self.config.model or DEFAULT_MODEL,
                        messages=prompt_messages,  # type: ignore
                        temperature=0.0,
                        max_tokens=3,
                    )
                    for prompt_messages in scoring_prompts
                ]
            )

            # Extract scores and create results
            results = []
            for passage, response in zip(passages, responses, strict=True):
                try:
                    if hasattr(response, "choices") and response.choices[0].message.content:
                        # Extract numeric score from response
                        score_text = response.choices[0].message.content.strip()
                        # Handle cases where model might return non-numeric text
                        score_match = re.search(r"\b(\d{1,3})\b", score_text)
                        if score_match:
                            score = float(score_match.group(1))
                            # Normalize to [0, 1] range and clamp to valid range
                            normalized_score = max(0.0, min(1.0, score / 100.0))
                            results.append((passage, normalized_score))
                        else:
                            logger.warning(f"Could not extract numeric score from response: {score_text}")
                            results.append((passage, 0.0))
                    else:
                        logger.warning("Empty response from Gemini for passage scoring")
                        results.append((passage, 0.0))
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing score from Gemini response: {e}")
                    results.append((passage, 0.0))

            # Sort by score in descending order (highest relevance first)
            results.sort(reverse=True, key=lambda x: x[1])
            return results

        except Exception as e:
            # Check if it's a rate limit error based on Gemini API error codes
            error_message = str(e).lower()
            if (
                "rate limit" in error_message
                or "quota" in error_message
                or "resource_exhausted" in error_message
                or "429" in str(e)
            ):
                raise RateLimitError from e

            logger.error(f"Error in generating LLM response: {e}")
            raise
