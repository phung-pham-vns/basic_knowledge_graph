from schema.disease_schema import node_types, allowed_relationships


def create_detailed_schema_prompt():
    schema_prompt = """
You are an expert at extracting structured knowledge from agricultural disease data and converting it into a knowledge graph.

## KNOWLEDGE GRAPH SCHEMA

### NODE TYPES:
"""

    # Add detailed node information
    for node in node_types:
        schema_prompt += f"\n**{node['label']}**: {node['description']}"
        if "properties" in node:
            schema_prompt += "\n  Properties:"
            for prop in node["properties"]:
                required = " (REQUIRED)" if prop.get("required", False) else ""
                schema_prompt += f"\n    - {prop['name']}: {prop['type']}{required} - {prop['description']}"

    schema_prompt += """

### RELATIONSHIP TYPES:
"""

    # Import relation_types to get descriptions
    from schema.disease_schema import relation_types

    for rel in relation_types:
        schema_prompt += f"\n**{rel['label']}**: {rel['description']}"

    schema_prompt += """

### ALLOWED RELATIONSHIPS:
"""

    for source, rel, target in allowed_relationships:
        schema_prompt += f"\n- {source} --[{rel}]--> {target}"

    schema_prompt += """

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

    return schema_prompt
