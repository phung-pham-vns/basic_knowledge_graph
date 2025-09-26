from cloud_hosted_embedding import CloudHostedEmbedder, CloudHostedEmbedderConfig


async def run_embedding() -> None:
    try:
        embedder = CloudHostedEmbedder(
            config=CloudHostedEmbedderConfig(
                embedding_base_url="https://api-ec2.aidata-dev.vnsilicon.cloud/v1",
                api_key="sk-0XVnNCM8f_3-G7EhPTrH4A",
                embedding_dim=1024,
                embedding_model="intfloat/multilingual-e5-large-instruct",
                embedding_max_tokens=512,
            ),
            batch_size=1,
        )

        # Single embedding
        text = "Hello, how are you?"
        embedding = await embedder.create(text)
        print(f"Embedding for '{text}':")
        print(f"Dimensions: {len(embedding)}")
        print(f"First 8 values: {embedding[:8]}")

        # Batch embedding
        texts = ["alpha", "beta", "gamma"]
        embeddings = await embedder.create_batch(texts)
        print(f"\nBatch embedding for {len(texts)} texts:")
        print(f"Number of embeddings: {len(embeddings)}")
        for text, emb in zip(texts, embeddings):
            print(f"Embedding for '{text}': dimensions={len(emb)}")

    except Exception as e:
        print(f"Error during embedding process: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_embedding())
