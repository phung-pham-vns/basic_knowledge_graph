import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

from graphiti_core.nodes import EpisodeType

from src.core.graphiti_client import GraphitiClient
from src.logger import setup_logging


logger = setup_logging()


def load_documents(file_paths: list[Path]) -> list[dict[str, any]]:
    documents: list[dict[str, any]] = []
    for file_path in file_paths:
        file_name = file_path.name
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as e:
            logger.exception("Failed to read/parse JSON file: %s", file_path)
            continue

        count = len(data) if isinstance(data, list) else 0
        logger.info("Loading %s - %d items", file_path, count)

        document_id = data[0]["document_id"] if len(data) else None

        document = {
            "id": document_id,
            "name": file_name,
            "chunks": [],
            "total_chunk": 0,
            "text_chunk": 0,
            "image_chunk": 0,
            "table_chunk": 0,
        }

        if not isinstance(data, list):
            logger.warning("Expected a list in %s, got %s; skipping.", file_path, type(data).__name__)
            continue

        for item in data:
            # Text
            if item.get("text") is not None:
                content = (item["text"].get("text_translated") or "") or (item["text"].get("content") or "")
                if content:
                    document["chunks"].append({"id": item["id"], "text": content})
                    document["text_chunk"] += 1
                    document["total_chunk"] += 1

            # Image caption
            if item.get("image") is not None:
                caption = item["image"].get("image_caption")
                if caption:
                    document["chunks"].append({"id": item["id"], "text": caption})
                    document["image_chunk"] += 1
                    document["total_chunk"] += 1

            # Table
            if item.get("table") is not None:
                table_txt = item["table"].get("content")
                if table_txt:
                    document["chunks"].append({"id": item["id"], "text": table_txt})
                    document["table_chunk"] += 1
                    document["total_chunk"] += 1

        documents.append(document)

    return documents


async def main(
    documents: list[dict[str, any]],
    clear_existing_graphdb_data: bool = False,
    max_coroutines: int = 1,
    add_communities: bool = False,
):
    try:
        graphiti_client = await GraphitiClient().create_client(
            clear_existing_graphdb_data=clear_existing_graphdb_data,
            max_coroutines=max_coroutines,
        )

        if add_communities:
            await graphiti_client.build_communities(group_ids=None)

        successful_chunks = 0
        failed_chunks = 0

        total_docs = len(documents)
        logger.info("Beginning processing of %d document(s).", total_docs)

        for i, document in enumerate(documents, start=1):
            file_name = document["name"]
            document_id = document["id"]
            chunks = document["chunks"]

            logger.info("Processing document %d/%d - %s (%d chunks)", i, total_docs, file_name, len(chunks))

            for j, chunk in enumerate(chunks, start=1):
                try:
                    await graphiti_client.add_episode(
                        name=document_id,
                        episode_body=chunk["text"],
                        source_description=chunk["id"],
                        reference_time=datetime.now(),
                        source=EpisodeType.text,
                        group_id=document_id,
                        update_communities=True if add_communities else False,
                        # entity_types=ENTITY_TYPES,
                        # edge_types=EDGE_TYPES,
                        # edge_type_map=EDGE_TYPE_MAP,
                    )
                    logger.info("Chunk %d/%d for %s processed successfully.", j, len(chunks), file_name)
                    successful_chunks += 1
                except Exception as e:
                    logger.exception(
                        "Error processing chunk %d/%d for %s. Chunk ID: %s", j, len(chunks), file_name, chunk["id"]
                    )
                    failed_chunks += 1
                    continue

        total = successful_chunks + failed_chunks
        rate = (successful_chunks / total * 100.0) if total else 0.0

        logger.info("=== Processing Summary ===")
        logger.info("Successful chunks: %d", successful_chunks)
        logger.info("Failed chunks: %d", failed_chunks)
        logger.info("Total chunks: %d", total)
        logger.info("Chunk success rate: %.2f%%", rate)

    finally:
        try:
            await graphiti_client.driver.close()
            logger.info("Connection closed.")
        except Exception:
            logger.exception("Error while closing the Graphiti driver.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--clear-existing-graphdb-data", type=bool, default=False)
    parser.add_argument("--max-coroutines", type=int, default=1)
    args = parser.parse_args()

    # Adjust this path to your data directory as needed
    data_dir = Path(args.data_dir)
    json_paths = list(data_dir.glob("*.json"))

    if not json_paths:
        logger.warning("No JSON files found in %s", data_dir)

    documents = load_documents(json_paths)

    asyncio.run(main(documents, args.clear_existing_graphdb_data, args.max_coroutines))
