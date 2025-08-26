DOCUMENT_ENTITIES_RELATIONS_EXTRACTION_PROMPT = """
<<GRAPH_SCHEMA>>

## SPECIFIC INSTRUCTIONS FOR THIS DOCUMENT:
- Extract entities that match the defined node types from the agricultural <<CATEGORY>> data below
- Create relationships following the allowed patterns shown above
- Pay special attention to <<CATEGORY>> names, symptoms, pathogens, and treatments
- Ensure all extracted entities have appropriate node types and properties

## DOCUMENT TO PROCESS:
<<DOCUMENT>>

## REMINDER:
Follow the schema exactly. Only create nodes and relationships that match the defined types and patterns.
"""
