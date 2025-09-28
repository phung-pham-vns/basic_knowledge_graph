import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from graphiti_core.graphiti import Graphiti
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    NodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    EpisodeReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    CommunityReranker,
)
from src.core.utils import format_context
from src.core.graphiti_client import GraphitiClient


@dataclass
class SearchResults:
    retrieved_results: List[Dict[str, Any]]
    formatted_context: str


@dataclass
class SearchComponentConfig:
    edges: bool = True
    nodes: bool = True
    episodes: bool = True
    communities: bool = True


@dataclass
class RerankerType:
    mmr: str = "mmr"
    rrf: str = "rrf"
    cross_encoder: str = "cross_encoder"


def create_search_config(
    limit: int = 10,
    component_config: SearchComponentConfig = SearchComponentConfig(),
    edge_reranker: Optional[RerankerType] = None,
    node_reranker: Optional[RerankerType] = None,
    episode_reranker: Optional[RerankerType] = None,
    community_reranker: Optional[RerankerType] = None,
    edge_methods: Optional[List[EdgeSearchMethod]] = None,
    node_methods: Optional[List[NodeSearchMethod]] = None,
    episode_methods: Optional[List[EpisodeSearchMethod]] = None,
    community_methods: Optional[List[CommunitySearchMethod]] = None,
) -> SearchConfig:
    """Create a flexible search configuration based on selected components and methods."""
    if edge_reranker == RerankerType.mmr:
        edge_reranker = EdgeReranker.mmr
    elif edge_reranker == RerankerType.rrf:
        edge_reranker = EdgeReranker.rrf
    elif edge_reranker == RerankerType.cross_encoder:
        edge_reranker = EdgeReranker.cross_encoder

    if node_reranker == RerankerType.mmr:
        node_reranker = NodeReranker.mmr
    elif node_reranker == RerankerType.rrf:
        node_reranker = NodeReranker.rrf
    elif node_reranker == RerankerType.cross_encoder:
        node_reranker = NodeReranker.cross_encoder

    if episode_reranker == RerankerType.rrf:
        episode_reranker = EpisodeReranker.rrf
    elif episode_reranker == RerankerType.cross_encoder:
        episode_reranker = EpisodeReranker.cross_encoder

    if community_reranker == RerankerType.mmr:
        community_reranker = CommunityReranker.mmr
    elif community_reranker == RerankerType.rrf:
        community_reranker = CommunityReranker.rrf
    elif community_reranker == RerankerType.cross_encoder:
        community_reranker = CommunityReranker.cross_encoder

    edge_config = (
        EdgeSearchConfig(
            search_methods=edge_methods or [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=edge_reranker,
        )
        if component_config.edges
        else None
    )

    node_config = (
        NodeSearchConfig(
            search_methods=node_methods or [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=node_reranker,
        )
        if component_config.nodes
        else None
    )

    episode_config = (
        EpisodeSearchConfig(
            search_methods=episode_methods or [EpisodeSearchMethod.bm25],
            reranker=episode_reranker,
        )
        if component_config.episodes
        else None
    )

    community_config = (
        CommunitySearchConfig(
            search_methods=community_methods or [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=community_reranker,
        )
        if component_config.communities
        else None
    )

    return SearchConfig(
        edge_config=edge_config,
        node_config=node_config,
        episode_config=episode_config,
        community_config=community_config,
        limit=limit,
    )


async def search_knowledge_graph(
    graphiti: Graphiti,
    question: str,
    limit: int = 10,
    component_config: SearchComponentConfig = SearchComponentConfig(),
    edge_reranker: Optional[RerankerType] = None,
    node_reranker: Optional[RerankerType] = None,
    episode_reranker: Optional[RerankerType] = None,
    community_reranker: Optional[RerankerType] = None,
    edge_methods: Optional[List[EdgeSearchMethod]] = None,
    node_methods: Optional[List[NodeSearchMethod]] = None,
    episode_methods: Optional[List[EpisodeSearchMethod]] = None,
    community_methods: Optional[List[CommunitySearchMethod]] = None,
) -> SearchResults:
    """
    Search the knowledge graph with flexible component and method selection.

    Args:
        graphiti: Graphiti instance for searching
        question: The query string to search for
        limit: Maximum number of results to return
        component_config: Configuration specifying which components to search
        edge_methods: List of edge search methods to use
        node_methods: List of node search methods to use
        episode_methods: List of episode search methods to use
        community_methods: List of community search methods to use

    Returns:
        SearchResults containing retrieved results and formatted context
    """
    search_config = create_search_config(
        limit=limit,
        component_config=component_config,
        edge_reranker=edge_reranker,
        node_reranker=node_reranker,
        episode_reranker=episode_reranker,
        community_reranker=community_reranker,
        edge_methods=edge_methods,
        node_methods=node_methods,
        episode_methods=episode_methods,
        community_methods=community_methods,
    )

    retrieved_results = await graphiti.search_(query=question, config=search_config)
    formatted_context = format_context(retrieved_results)

    return SearchResults(
        retrieved_results=retrieved_results,
        formatted_context=formatted_context,
    )


async def main():
    """Main function to demonstrate flexible knowledge graph search."""
    graphiti = await GraphitiClient().create_client(
        clear_existing_graphdb_data=False,
        max_coroutines=1,
    )

    question = "My young durian leaves are curling and look scorched at the edges—could that be leafhopper damage and what should I do first?"
    results_partial = await search_knowledge_graph(
        graphiti=graphiti,
        question=question,
        limit=3,
        component_config=SearchComponentConfig(
            edges=True,
            nodes=True,
            episodes=False,
            communities=False,
        ),
        node_methods=[
            NodeSearchMethod.bm25,
            NodeSearchMethod.cosine_similarity,
            NodeSearchMethod.bfs,
        ],
        edge_methods=[
            EdgeSearchMethod.bm25,
            EdgeSearchMethod.cosine_similarity,
            EdgeSearchMethod.bfs,
        ],
        edge_reranker=RerankerType.cross_encoder,
        node_reranker=RerankerType.cross_encoder,
    )

    print(f"Partial search results: {results_partial.retrieved_results}")
    print(f"Partial search context: {results_partial.formatted_context}")

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
