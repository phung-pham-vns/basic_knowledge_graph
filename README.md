# Knowledge Graph
Knowledge Graph Builder

A knowledge graph (KG) construction pipeline requires a few components:
- Document Loader: extract text from files (PDFs, ...)
- Text Splitter: split text into smaller pieces of text (chunks), manageable by the LLM context window (token limit).
- Chunking embedder (optional): compute the chunk embeddings
- Schema builder: provide a schema to ground the LLM extracted nodes and relationship types and obtain an easily navigable KG. Schema can be provided manually or extracted automatically using LLMs.
- Lexical graph builder
- Entity and relationship extractor
- Graph pruner: Clean the graph based on schema, if provided.
- Knowledge graph writer: save the identified entities and relations.
- Entity resolver: merge similar entities into a single node.
