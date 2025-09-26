from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import SearchResults


def format_edge_facts(edges: list[EntityEdge]) -> str:
    if not edges:
        return ""
    return "\n".join(f"{i + 1}. {edge.fact}" for i, edge in enumerate(edges))


def format_node_summaries(nodes: list[EntityNode]) -> str:
    if not nodes:
        return ""
    return "\n".join(f"{i + 1}. {node.summary}" for i, node in enumerate(nodes))


def format_community_summaries(communities) -> str:
    if not communities:
        return ""
    return "\n".join(f"{i + 1}. {community.summary}" for i, community in enumerate(communities))


def format_context(result: SearchResults) -> str:
    context_parts = []

    if hasattr(result, "nodes") and result.nodes:
        node_context = format_node_summaries(result.nodes)
        if node_context:
            context_parts.append(f"Node Information:\n{node_context}")

    if hasattr(result, "edges") and result.edges:
        edge_context = format_edge_facts(result.edges)
        if edge_context:
            context_parts.append(f"Relationship Information:\n{edge_context}")

    if hasattr(result, "communities") and result.communities:
        community_context = format_community_summaries(result.communities)
        if community_context:
            context_parts.append(f"Community Information:\n{community_context}")

    return "\n\n".join(context_parts) if context_parts else "No relevant context found."
