#!/usr/bin/env python3
"""
Test script to verify that all LLM and embedding providers work correctly.
This script tests the provider system without running the full pipeline.
"""

import sys
from pathlib import Path
import os

# Add the src directory to Python path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from deps.llms import get_llm_client
from deps.embeddings import get_embedding_client
from settings import ProjectSettings


def test_provider_combination(
    llm_provider: str,
    embedding_provider: str,
    llm_model: str = None,
    embedding_model: str = None,
    api_key: str = "test_key",
    endpoint: str = None,
):
    """Test a specific provider combination."""
    print(f"\n{'='*60}")
    print(f"Testing: {llm_provider.upper()} LLM + {embedding_provider.upper()} Embeddings")
    print(f"{'='*60}")

    try:
        # Create test settings
        test_settings = ProjectSettings()
        test_settings.llm.llm_provider = llm_provider
        test_settings.llm.embedding_provider = embedding_provider
        test_settings.llm.llm_api_key = api_key

        if llm_model:
            test_settings.llm.llm_model = llm_model
        if embedding_model:
            test_settings.llm.embedding_model = embedding_model
        if endpoint:
            test_settings.llm.llm_endpoint = endpoint

        # Test LLM client creation
        print(f"✓ Creating {llm_provider} LLM client...")
        llm_client = get_llm_client(test_settings)
        print(f"  - LLM Client: {type(llm_client).__name__}")

        # Test Embedding client creation
        print(f"✓ Creating {embedding_provider} embedding client...")
        embedding_client = get_embedding_client(test_settings)
        print(f"  - Embedding Client: {type(embedding_client).__name__}")

        print(f"✅ SUCCESS: {llm_provider}/{embedding_provider} combination works!")

    except Exception as e:
        print(f"❌ ERROR: {llm_provider}/{embedding_provider} combination failed:")
        print(f"   {str(e)}")


def main():
    """Test all provider combinations."""
    print("Testing Multi-Provider Knowledge Graph System")
    print("=" * 60)

    # Test combinations
    test_combinations = [
        # (llm_provider, embedding_provider, llm_model, embedding_model, endpoint)
        ("openai", "openai", "gpt-4o-mini", "text-embedding-3-small", None),
        ("gemini", "gemini", "gemini-1.5-flash", "models/text-embedding-004", None),
        ("ollama", "ollama", "llama3.1", "nomic-embed-text", "http://localhost:11434"),
        ("huggingface", "huggingface", "microsoft/DialoGPT-medium", "sentence-transformers/all-MiniLM-L6-v2", None),
        # Mixed combinations
        ("openai", "gemini", "gpt-4o-mini", "models/text-embedding-004", None),
        ("gemini", "openai", "gemini-1.5-flash", "text-embedding-3-small", None),
        ("ollama", "openai", "llama3.1", "text-embedding-3-small", "http://localhost:11434"),
    ]

    success_count = 0
    total_count = len(test_combinations)

    for llm_prov, emb_prov, llm_model, emb_model, endpoint in test_combinations:
        try:
            test_provider_combination(llm_prov, emb_prov, llm_model, emb_model, "test_key", endpoint)
            success_count += 1
        except Exception as e:
            print(f"❌ Unexpected error testing {llm_prov}/{emb_prov}: {e}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{total_count} provider combinations work correctly")
    print(f"{'='*60}")

    if success_count == total_count:
        print("🎉 All provider combinations are working!")
    else:
        print("⚠️  Some provider combinations have issues. Check the errors above.")


if __name__ == "__main__":
    main()

