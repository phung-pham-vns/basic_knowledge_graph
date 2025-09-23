from abc import ABC, abstractmethod


class CrossEncoderClient(ABC):
    """
    CrossEncoderClient is an abstract base class that defines the interface
    for cross-encoder models used for ranking passages based on their relevance to a query.
    It allows for different implementations of cross-encoder models to be used interchangeably.
    """

    @abstractmethod
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank the given passages based on their relevance to the query.

        Args:
            query (str): The query string.
            passages (list[str]): A list of passages to rank.

        Returns:
            list[tuple[str, float]]: A list of tuples containing the passage and its score, sorted in descending order of relevance.
        """
        pass
