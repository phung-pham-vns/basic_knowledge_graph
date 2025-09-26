"""
Configuration classes and utilities for Agentic Graph RAG
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

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
    """Create search configuration with proper reranker mapping."""

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
            reranker=edge_reranker or EdgeReranker.rrf,
        )
        if component_config.edges
        else None
    )

    node_config = (
        NodeSearchConfig(
            search_methods=node_methods or [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=node_reranker or NodeReranker.rrf,
        )
        if component_config.nodes
        else None
    )

    episode_config = (
        EpisodeSearchConfig(
            search_methods=episode_methods or [EpisodeSearchMethod.bm25],
            reranker=episode_reranker or EpisodeReranker.rrf,
        )
        if component_config.episodes
        else None
    )

    community_config = (
        CommunitySearchConfig(
            search_methods=community_methods or [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=community_reranker or CommunityReranker.rrf,
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
