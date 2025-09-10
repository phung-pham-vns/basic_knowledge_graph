# Multi-Provider Knowledge Graph System

This knowledge graph system now supports multiple LLM and embedding providers. You can mix and match different providers for LLMs and embeddings based on your needs.

## Supported Providers

### LLM Providers
1. **OpenAI** - GPT models (gpt-4o-mini, gpt-4, etc.)
2. **Gemini** - Google's Gemini models (gemini-1.5-flash, etc.)
3. **Ollama** - Local models (llama3.1, codellama, etc.)
4. **HuggingFace** - Self-hosted or HuggingFace Hub models

### Embedding Providers
1. **OpenAI** - text-embedding-3-small, text-embedding-3-large
2. **Gemini** - models/text-embedding-004
3. **Ollama** - nomic-embed-text, etc.
4. **HuggingFace** - sentence-transformers models, etc.

## Configuration

### Environment Variables

Set these environment variables in your `.env` file or export them:

```bash
# LLM Configuration
LLM_PROVIDER=openai              # openai, gemini, ollama, huggingface
LLM_MODEL=gpt-4o-mini           # Model name for the chosen provider
LLM_API_KEY=your_api_key_here   # API key (for OpenAI/Gemini)
LLM_ENDPOINT=http://localhost:11434  # Endpoint URL (for Ollama/self-hosted)
LLM_TEMPERATURE=0.0             # Temperature setting
LLM_MAX_TOKENS=16384           # Maximum tokens

# Embedding Configuration
EMBEDDING_PROVIDER=openai        # openai, gemini, ollama, huggingface
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model name
EMBEDDING_DIMENSIONS=1536        # Embedding dimensions (for OpenAI)

# Neo4j Configuration
GRAPH_DB_URL=neo4j://localhost:7687
GRAPH_DB_USER=neo4j
GRAPH_DB_PASSWORD=your_password_here
```

### Provider-Specific Configuration Examples

#### OpenAI
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-your-openai-key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

#### Gemini
```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
LLM_API_KEY=your-google-api-key
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/text-embedding-004
```

#### Ollama (Local)
```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
LLM_ENDPOINT=http://localhost:11434
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```

#### HuggingFace
```bash
LLM_PROVIDER=huggingface
LLM_MODEL=microsoft/DialoGPT-medium
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Mixed Configurations

You can mix different providers for LLMs and embeddings:

```bash
# Use OpenAI for LLM and Gemini for embeddings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-your-openai-key
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/text-embedding-004
```

## Usage

### Running the Knowledge Graph Builder

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with your configured providers
python scripts/simple_kg_builder_from_pdf.py
```

### Testing Providers

Test all provider combinations:

```bash
source .venv/bin/activate
python scripts/test_providers.py
```

Debug specific providers:

```bash
source .venv/bin/activate
python scripts/debug_providers.py
```

## Installation Requirements

### Core Requirements
```bash
uv add langchain-openai langchain-google-genai langchain-community langchain-ollama langchain-huggingface
```

### Provider-Specific Requirements

#### For HuggingFace
```bash
uv add transformers torch  # or tensorflow
```

#### For Ollama
Make sure Ollama is running locally:
```bash
ollama serve
ollama pull llama3.1  # or your preferred model
ollama pull nomic-embed-text  # for embeddings
```

## Model Recommendations

### For Production
- **LLM**: OpenAI gpt-4o-mini (cost-effective, good performance)
- **Embeddings**: OpenAI text-embedding-3-small (high quality, reasonable cost)

### For Local/Privacy
- **LLM**: Ollama with llama3.1 or codellama
- **Embeddings**: Ollama with nomic-embed-text

### For Experimentation
- **LLM**: Gemini gemini-1.5-flash (fast, good for testing)
- **Embeddings**: Gemini models/text-embedding-004

### For Custom/Fine-tuned Models
- **LLM**: HuggingFace with your custom model
- **Embeddings**: HuggingFace with sentence-transformers models

## Troubleshooting

### Common Issues

1. **Module not found errors**: Make sure virtual environment is activated
2. **API key errors**: Check that API keys are set correctly
3. **Model not found**: Verify model names are correct for each provider
4. **Ollama connection errors**: Ensure Ollama is running on the specified endpoint
5. **HuggingFace download errors**: Check internet connection and model availability

### Performance Tips

1. **OpenAI**: Use gpt-4o-mini for better cost/performance ratio
2. **Gemini**: Use gemini-1.5-flash for faster responses
3. **Ollama**: Use smaller models (7B parameters) for faster inference
4. **HuggingFace**: Consider using quantized models for better performance

## Architecture

The system uses a wrapper pattern to adapt different provider APIs to the neo4j_graphrag interface:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Settings      │    │   Provider       │    │   neo4j_graphrag│
│   Configuration │───▶│   Wrappers       │───▶│   Pipeline      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ LLM Providers:   │
                    │ • OpenAIWrapper  │
                    │ • GeminiWrapper  │
                    │ • OllamaWrapper  │
                    │ • HuggingFace    │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │Embedding Providers│
                    │ • OpenAIWrapper  │
                    │ • GeminiWrapper  │
                    │ • OllamaWrapper  │
                    │ • HuggingFace    │
                    └──────────────────┘
```

