from langchain.prompts import PromptTemplate


entities_and_relationships_extraction_template = """
You are an expert at extracting structured knowledge from agricultural disease data and converting it into a knowledge graph.

## KNOWLEDGE GRAPH SCHEMA
{graph_schema}

## DOCUMENT TO PROCESS:
{document}

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


entities_and_relationships_extraction_prompt = PromptTemplate(
    template=entities_and_relationships_extraction_template,
    input_variables=["graph_schema", "category", "document"],
)


if __name__ == "__main__":
    prompt = entities_and_relationships_extraction_prompt.invoke(
        {
            "graph_schema": "This is graph schema",
            "category": "disease",
            "document": "This is Document",
        }
    )
    print(prompt.text)
