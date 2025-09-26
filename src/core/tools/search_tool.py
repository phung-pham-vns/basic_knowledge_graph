from typing import List, Optional
from langchain_core.tools import tool
from graphiti_core.search.search_config import (
    EdgeSearchMethod,
    NodeSearchMethod,
    EpisodeSearchMethod,
    CommunitySearchMethod,
)

from src.core.config import SearchComponentConfig, RerankerType, create_search_config
from src.core.graphiti_client import GraphitiClient
from src.core.utils import format_context


@tool
async def search_knowledge_graph(
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
) -> str:
    """Search the knowledge graph and return formatted context."""

    graphiti = await GraphitiClient().create_client(
        clear_existing_graphdb_data=False,
        max_coroutines=1,
    )

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

    results = await graphiti.search_(query=question, config=search_config)
    return format_context(results)
