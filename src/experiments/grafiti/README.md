# Graphiti Knowledge Graph Retrieval and Answer Generation

This directory contains the refined implementation for retrieving information from a Graphiti knowledge graph and generating comprehensive answers using LLM-based generation.

## Files

- `retrieve_kg.py` - Main script for knowledge graph retrieval and answer generation
- `construct_kg.py` - Script for constructing the knowledge graph (existing)
- `disease_schema.py` - Schema definitions for the disease domain (existing)

## Features

### Enhanced `retrieve_kg.py`

The refined script now includes:

1. **Answer Generation**: Automatically generates comprehensive answers based on search results
2. **Flexible Configuration**: Support for environment variables and command-line arguments
3. **Multiple LLM Providers**: Support for both OpenAI and Gemini
4. **Robust Error Handling**: Comprehensive logging and error recovery
5. **Progress Tracking**: Saves progress after each question and can skip already processed ones
6. **Performance Metrics**: Tracks search time, generation time, and total response time

## Usage

### Basic Usage

```bash
# Use default configuration
python retrieve_kg.py

# Specify input/output files
python retrieve_kg.py --input questions.json --output answers.json

# Use different search limit
python retrieve_kg.py --limit 5

# Process all questions (don't skip already processed)
python retrieve_kg.py --no-skip-processed

# Use Gemini provider
python retrieve_kg.py --llm-provider gemini --model gemini-2.0-flash

# Enable verbose logging
python retrieve_kg.py --verbose
```

### Environment Variables

Create a `.env` file in the project root with:

```env
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# Graph Database Configuration
GRAPHDB_PROVIDER=neo4j
GRAPHDB_URI=bolt://localhost:7687
GRAPHDB_USER=neo4j
GRAPHDB_PASSWORD=your_password

# Search Configuration
SEARCH_LIMIT=10
SKIP_PROCESSED=true

# File Paths
INPUT_FILE=/path/to/questions.json
OUTPUT_FILE=/path/to/answers.json
```

## Input Format

The input JSON file should contain an array of question objects:

```json
[
  {
    "question": "My young durian leaves are curling and look scorched at the edges—could that be leafhopper damage and what should I do first?",
    "answer": "Expected answer (optional)",
    "citation": ["Source citations (optional)"],
    "type": "multi-hop",
    "batch": "a"
  }
]
```

## Output Format

The script adds the following fields to each question object:

```json
{
  "question": "Original question",
  "predict": "Generated comprehensive answer",
  "successed": true,
  "response_time": 15.23,
  "search_time": 2.45,
  "generation_time": 12.78
}
```

## Answer Generation

The script uses a sophisticated prompt template that:

1. **Contextualizes** the search results from the knowledge graph
2. **Structures** the response with clear sections (Identification, Symptoms, Treatment)
3. **Provides** specific, actionable recommendations
4. **Includes** chemical names, application rates, and timing when available
5. **Handles** cases where information is incomplete

## Error Handling

The script includes comprehensive error handling:

- **Connection errors** to the graph database
- **API errors** from LLM providers  
- **File I/O errors** for input/output files
- **JSON parsing errors**
- **Individual question processing errors** (continues with remaining questions)

## Performance

- **Parallel processing**: Optimized for batch processing of multiple questions
- **Progress saving**: Saves after each question to prevent data loss
- **Skip processed**: Avoids reprocessing already completed questions
- **Configurable limits**: Adjustable search result limits for performance tuning

## Logging

The script provides detailed logging at multiple levels:

- **INFO**: General progress and status updates
- **DEBUG**: Detailed processing information (use `--verbose`)
- **ERROR**: Error conditions and recovery attempts
- **WARNING**: Non-fatal issues

## Example Output

```
2025-01-XX XX:XX:XX - INFO - Starting Graphiti Knowledge Graph Retrieval and Answer Generation
2025-01-XX XX:XX:XX - INFO - LLM Provider: openai
2025-01-XX XX:XX:XX - INFO - Processing 17 questions
============================================================
Sample 1 / 17
Question: My young durian leaves are curling and look scorched at the edges—could that be leafhopper damage and what should I do first?
Search Time: 2.45 seconds
Generation Time: 12.78 seconds
Total Time: 15.23 seconds
Answer Preview: The symptoms you describe—young durian leaves curling and having scorched edges—are consistent with damage caused by leafhoppers...
============================================================

PROCESSING COMPLETE
============================================================
Successfully processed: 17/17
Failed: 0/17
Average response time: 14.56 seconds
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure correct API keys are set in environment variables
2. **Connection Errors**: Verify Neo4j database is running and accessible
3. **Memory Issues**: Reduce search limit or process in smaller batches
4. **Rate Limiting**: The script includes automatic retry logic for API rate limits

### Debug Mode

Use `--verbose` flag to enable detailed debug logging:

```bash
python retrieve_kg.py --verbose
```

This will show detailed information about search results, API calls, and processing steps.
