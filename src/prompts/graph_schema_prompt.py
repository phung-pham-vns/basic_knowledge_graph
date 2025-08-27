def graph_schema_prompt(
    node_types: dict,
    relation_types: dict,
    allowed_relationships: list,
) -> str:
    prompt = """
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
"""

    return prompt


if __name__ == "__main__":
    from src.schema.disease_schema import node_types, relation_types, allowed_relationships

    print(graph_schema_prompt(node_types, relation_types, allowed_relationships))
