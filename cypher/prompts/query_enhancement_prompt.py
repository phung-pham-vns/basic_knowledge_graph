from langchain.prompts import PromptTemplate

query_enhancement_template = """
## TASK:
Generate a Cypher statement to query the graph database.

## INSTRUCTIONS:
Use only relationship types and properties provided in schema.
Do not use other relationship types or properties that are not provided.

## GRAPH SCHEMA:
{schema}

## NOTE:
- Do not include explanations or apologies in your answers.
- Do not answer questions that ask anything other than creating Cypher statements.
- Do not include any text other than generated Cypher statements.

## USER QUESTION:
{question}"""

query_enhancement_prompt = PromptTemplate(
    template=query_enhancement_template,
    input_variables=["schema", "question"],
)


if __name__ == "__main__":
    prompt = query_enhancement_prompt.invoke(
        input={
            "schema": "This is graph schema",
            "question": "This is user question",
        }
    )

    print(prompt.text)
