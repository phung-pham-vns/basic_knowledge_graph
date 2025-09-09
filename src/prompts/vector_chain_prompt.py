from langchain_core.prompts.prompt import PromptTemplate

vector_prompt_template = """Human: You are a Financial expert with SEC filings who can answer questions only based on the context below.
* Answer the question STRICTLY based on the context provided in JSON below.
* Do not assume or retrieve any information outside of the context 
* Use three sentences maximum and keep the answer concise
* Think step by step before answering.
* Do not return helpful or extra text or apologies
* Just return summary to the user. DO NOT start with Here is a summary
* List the results in rich text format if there are more than one results
* If the context is empty, just respond None

<question>
{input}
</question>

Here is the context:
<context>
{context}
</context>

Assistant:"""

vector_prompt = PromptTemplate(input_variables=["input", "context"], template=vector_prompt_template)

if __name__ == "__main__":
    prompt = vector_prompt.invoke(
        input={
            "input": "This is user input",
            "context": "This is context",
        }
    )

    print(prompt.text)
