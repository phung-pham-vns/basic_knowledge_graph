from langchain_core.prompts.prompt import PromptTemplate

cypher_generation_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_generation_prompt = PromptTemplate(input_variables=["schema", "question"], template=cypher_generation_template)

if __name__ == "__main__":
    prompt = cypher_generation_prompt.invoke(
        input={
            "schema": "This is graph schema",
            "question": "This is user question",
        }
    )

    print(prompt.text)
