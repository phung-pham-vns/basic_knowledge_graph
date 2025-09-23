import json
import asyncio

from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerAccuracy


def save_json(data: dict, path: str):
    with open(file=path, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


async def main(
    file_path: str,
    score_threshold: float = 0.5,
    provider: str = "gemini",
    save_path: str = None,
    concurrency_limit: int = 25,
):
    # 1. Load and prepare data
    with open(file=file_path, mode="r", encoding="utf-8") as file:
        samples = json.load(file)

    # 2. Initialize Ragas components
    if provider == "openai":
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4.1",
                temperature=0.0,
                seed=42,
            )
        )
    elif provider == "gemini":
        evaluator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.0,
                model_kwargs={"seed": 42},
                # google_api_key=settings.llm_api_key,
                google_api_key="AIzaSyCJEHXFRh_tbiAkiIV55ieydAHLpJjJaCA",
            )
        )

    scorer = AnswerAccuracy(llm=evaluator_llm)

    # 3. Create Ragas samples
    evaluated_samples = [
        SingleTurnSample(
            user_input=sample["question"],
            reference=sample["answer"],
            response=sample.get("generation", "No response."),
        )
        for sample in samples
    ]

    # 4. Asynchronously evaluate samples with a semaphore for concurrency control
    semaphore = asyncio.Semaphore(value=concurrency_limit)

    async def evaluate_sample(sample):
        """Asynchronously evaluate a single sample with a semaphore."""
        async with semaphore:
            return await scorer.single_turn_ascore(sample)

    print("Starting evaluation...")
    tasks = [evaluate_sample(sample) for sample in evaluated_samples]
    scores = await tqdm_asyncio.gather(*tasks)
    print("Evaluation complete.")

    # 5. Add scores to DataFrame and save to CSV
    num_samples = 0
    num_passed = 0
    num_failed = 0
    sum_score = 0
    time_to_retrieval = 0
    time_to_generation = 0
    for sample, score in zip(samples, scores):
        sum_score += score
        sample["score"] = score
        time_to_retrieval += sample["time_to_retrieval"]
        time_to_generation += sample["time_to_generation"]
        if score >= score_threshold:
            num_passed += 1
            sample["conclusion"] = "pass"
        else:
            num_failed += 1
            sample["conclusion"] = "fail"
        num_samples += 1

    print(f"Number of samples: {num_samples}")
    print(f"Number of passed samples: {num_passed} ({num_passed / num_samples * 100:.2f}%)")
    print(f"Number of failed samples: {num_failed} ({num_failed / num_samples * 100:.2f}%)")
    print(f"Average time to retrieval: {time_to_retrieval / num_samples}")
    print(f"Average time to generation: {time_to_generation / num_samples}")
    print(f"Average score: {sum_score / num_samples}")

    # 6. Save file
    if save_path is None:
        save_json(samples, file_path)
        print(f"Scores saved to '{file_path}'.")
    else:
        save_json(samples, save_path)
        print(f"Scores saved to '{save_path}'.")


if __name__ == "__main__":
    asyncio.run(
        main(
            file_path="/Users/mac/Documents/PHUNGPX/knowledge_graph_searching/data/QA_17_pest_disease_GEMINI_PRO_COMBINED_HYBRID_SEARCH_CROSS_ENCODER_limit_10.json",
        )
    )
