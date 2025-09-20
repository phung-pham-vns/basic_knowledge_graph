import json
import time
import asyncio
from enum import Enum

from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    COMBINED_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_MMR,
)

from graphiti.core.combine_context import format_context
from graphiti.core.graphiti_client import GraphitiClient
from graphiti.prompts.generation import generation_prompt_template


class SearchType(Enum):
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER = "combined_hybrid_search_cross_encoder"
    EDGE_HYBRID_SEARCH_CROSS_ENCODER = "edge_hybrid_search_cross_encoder"
    COMBINED_HYBRID_SEARCH_RRF = "combined_hybrid_search_rrf"
    COMBINED_HYBRID_SEARCH_MMR = "combined_hybrid_search_mmr"


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


async def generate_answer(
    llm_client: LLMClient,
    question: str,
    context: str,
    max_tokens: int = 5000,
):
    prompt = generation_prompt_template.format(question=question, context=context)
    messages = [
        Message(role="user", content=prompt),
    ]
    return await llm_client.generate_response(
        messages=messages,
        max_tokens=max_tokens,
    )


async def rag(
    graphiti_client: Graphiti,
    llm_client: LLMClient,
    question: str,
    search_type: SearchType,
    limit: int = 10,
):
    if search_type == SearchType.COMBINED_HYBRID_SEARCH_CROSS_ENCODER:
        search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    elif search_type == SearchType.EDGE_HYBRID_SEARCH_CROSS_ENCODER:
        search_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    elif search_type == SearchType.COMBINED_HYBRID_SEARCH_RRF:
        search_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
    elif search_type == SearchType.COMBINED_HYBRID_SEARCH_MMR:
        search_config = COMBINED_HYBRID_SEARCH_MMR.model_copy(deep=True)

    search_config.limit = limit

    t1 = time.time()
    retrieved = await graphiti_client._search(
        query=question,
        config=search_config,
    )
    retrieval_time = time.time() - t1

    # TODO: get all relevant documents / episodes

    t1 = time.time()
    context = format_context(retrieved)
    argumentation_time = time.time() - t1

    t1 = time.time()
    generated = await generate_answer(llm_client, question, context)
    t2 = time.time()
    generation_time = t2 - t1

    return {
        "retrieval_time": retrieval_time,
        "argumentation_time": argumentation_time,
        "generation_time": generation_time,
        # "retrieved_results": retrieved,
        "combined_context": context,
        "generated_answer": generated["content"],
    }


async def main(
    input_json_path,
    output_json_path,
    search_type: SearchType,
    limit: int = 10,
):
    graphiti_client = await GraphitiClient(clear_existing_graphdb_data=False).get_graphiti_client()

    llm_client = graphiti_client.llm_client

    # Initialize the graph database with graphiti's indices if needed
    try:
        await graphiti_client.build_indices_and_constraints()
        print("Graphiti indices built successfully.")
    except Exception as e:
        print(f"Note: {str(e)}/nContinuing with existing indices...")

    try:
        with open(file=input_json_path, mode="r", encoding="utf-8") as file:
            data = json.load(file)

        for i, sample in enumerate(data):
            print("--" * 30)
            print(f"Sample {i + 1} / {len(data)}")
            question = sample["question"]
            print(f"Question: {question}\n")

            # if "successed" in sample and sample["successed"] == True:
            #     print(f"Answer: {sample['generation']}")
            #     print("\n")
            #     continue

            try:
                responses = await rag(
                    graphiti_client,
                    llm_client,
                    question,
                    search_type,
                    limit,
                )
                sample["generation"] = responses["generated_answer"]
                sample["argumentation"] = responses["combined_context"]
                # sample["retrieval"] = responses["retrieved_results"]
                sample["time_to_retrieval"] = responses["retrieval_time"]
                sample["time_to_generation"] = responses["generation_time"]
                sample["successed"] = True
                print(f"Groundtruth: {sample['answer']}\n")
                print(f"Prediction: {sample['generation']}")
            except Exception as e:
                sample["predict"] = f"Error: {e}"
                sample["successed"] = False
                print(f"Error: {e}")

            save_json(data, output_json_path)

    finally:
        await graphiti_client.close()
        print("\nConnection closed")


if __name__ == "__main__":
    asyncio.run(
        main(
            input_json_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/data/QA_17_pest_disease_COMBINED_HYBRID_SEARCH_RRF_limit_10.json",
            output_json_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/data/QA_17_pest_disease_COMBINED_HYBRID_SEARCH_RRF_limit_10.json",
            search_type=SearchType.COMBINED_HYBRID_SEARCH_RRF,
            limit=10,
        )
    )
