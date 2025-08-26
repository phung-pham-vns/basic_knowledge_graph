def graph_schema_prompt(
    node_types: dict,
    relation_types: dict,
    allowed_relationships: list,
) -> str:
    prompt = """
You are an expert at extracting structured knowledge from agricultural disease data and converting it into a knowledge graph.

## KNOWLEDGE GRAPH SCHEMA

### NODE TYPES:
"""
    for node in node_types:
        prompt += f"\n**{node['label']}**: {node['description']}"
        if "properties" in node:
            prompt += "\n  Properties:"
            for prop in node["properties"]:
                required = " (REQUIRED)" if prop.get("required", False) else ""
                prompt += f"\n    - {prop['name']}: {prop['type']}{required} - {prop['description']}"

    prompt += """

### RELATIONSHIP TYPES:
"""
    for rel in relation_types:
        prompt += f"\n**{rel['label']}**: {rel['description']}"

    prompt += """

### ALLOWED RELATIONSHIPS:
"""

    for source, rel, target in allowed_relationships:
        prompt += f"\n- {source} --[{rel}]--> {target}"

    prompt += """

## INSTRUCTIONS:
1. Analyze the provided agricultural disease data carefully
2. Extract entities that match the defined node types
3. Create relationships between entities following the allowed relationship patterns
4. Ensure all nodes have appropriate properties where specified
5. Be precise and accurate in entity extraction and relationship creation
6. If uncertain about a relationship, err on the side of caution and don't create it

## OUTPUT FORMAT:
Return a structured knowledge graph with:
- Nodes: Each node should have a type (label) and relevant properties
- Relationships: Each relationship should connect two nodes with the specified relationship type
"""

    return prompt


if __name__ == "__main__":
    from src.schema.disease_schema import node_types, relation_types, allowed_relationships

    print(graph_schema_prompt(node_types, relation_types, allowed_relationships))
